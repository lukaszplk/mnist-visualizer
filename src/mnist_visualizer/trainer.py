"""
mnist_visualizer.trainer
~~~~~~~~~~~~~~~~~~~~~~~~
Training loop that runs in a background thread and emits TrainStats
via a callback after every batch and epoch-end.

Training does NOT start automatically — call .start() explicitly.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import MLP, TrainStats


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def _get_data(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=_TRANSFORM)
    val_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False, num_workers=0)
    return train_loader, val_loader


def _val_accuracy(model: MLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    model.train()
    return correct / total * 100


class Trainer:
    """Runs training in a daemon thread; calls *on_step* with live stats.

    Training starts only when .start() is called explicitly.
    """

    def __init__(
        self,
        on_step: Callable[[TrainStats], None],
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._on_step   = on_step
        self._epochs    = epochs
        self._batch_size = batch_size
        self._lr        = lr
        self._data_dir  = data_dir or Path.home() / ".cache" / "mnist_visualizer"
        self._stop      = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.paused     = False
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._model: Optional[MLP] = None
        self._model_lock = threading.Lock()

    # ── Model access (thread-safe for inference) ──────────────────────────────

    @property
    def model(self) -> Optional[MLP]:
        return self._model

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause_event.set()

    def toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self._pause_event.set()
        else:
            self.paused = True
            self._pause_event.clear()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Training loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = MLP().to(device)
        with self._model_lock:
            self._model = model

        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        train_loader, val_loader = _get_data(self._data_dir, self._batch_size)
        total_batches = len(train_loader)
        stats = TrainStats(total_batches=total_batches)

        for epoch in range(1, self._epochs + 1):
            if self._stop.is_set():
                break

            epoch_correct = epoch_total = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                self._pause_event.wait()
                if self._stop.is_set():
                    break

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss   = criterion(logits, y)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                batch_correct  = (preds == y).sum().item()
                epoch_correct += batch_correct
                epoch_total   += y.size(0)

                layer_stats, act_arrays, weight_arrays = model.collect_stats()

                stats.epoch          = epoch
                stats.batch          = batch_idx + 1
                stats.loss           = loss.item()
                stats.batch_accuracy = batch_correct / y.size(0) * 100
                stats.epoch_accuracy = epoch_correct / epoch_total * 100
                stats.layer_stats    = layer_stats
                stats.activations    = act_arrays
                stats.weights        = weight_arrays

                self._on_step(stats)

            if not self._stop.is_set():
                stats.val_accuracy = _val_accuracy(model, val_loader, device)
                self._on_step(stats)
