"""
mnist_visualizer.model
~~~~~~~~~~~~~~~~~~~~~~
Three-layer MLP for MNIST digit classification.

Architecture: 784 → 128 → 64 → 10
Activations:  ReLU → ReLU → (raw logits, CrossEntropyLoss applies softmax)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerStats:
    """Per-layer statistics captured after each forward pass."""
    activations_mean: float = 0.0
    activations_std: float = 0.0
    activations_min: float = 0.0
    activations_max: float = 0.0
    dead_neurons_pct: float = 0.0   # % of neurons with mean activation ≤ 0
    weights_mean: float = 0.0
    weights_std: float = 0.0
    grad_mean: float = 0.0          # populated after backward()
    grad_std: float = 0.0


@dataclass
class TrainStats:
    """Snapshot of training state for one step."""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    loss: float = 0.0
    batch_accuracy: float = 0.0
    epoch_accuracy: float = 0.0     # updated at epoch end
    val_accuracy: float = 0.0       # updated at epoch end
    layer_stats: list[LayerStats] = field(default_factory=list)
    # raw activations per layer for graph colouring (list of 1-D numpy arrays)
    activations: list = field(default_factory=list)
    # weight matrices per layer for edge colouring (list of 2-D numpy arrays)
    weights: list = field(default_factory=list)


class MLP(nn.Module):
    """784 → 128 → 64 → 10 fully-connected network."""

    LAYER_SIZES = [784, 128, 64, 10]

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

        # hooks store intermediate activations during forward pass
        self._activations: list[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def make_hook(idx: int):
            def hook(module, input, output):
                self._activations.append(output.detach())
            return hook

        self.fc1.register_forward_hook(make_hook(0))
        self.fc2.register_forward_hook(make_hook(1))
        self.fc3.register_forward_hook(make_hook(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activations.clear()
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def collect_stats(self) -> tuple[list[LayerStats], list, list]:
        """Build LayerStats + raw activation/weight arrays from last forward pass."""
        layers = [self.fc1, self.fc2, self.fc3]
        stats: list[LayerStats] = []
        act_arrays = []
        weight_arrays = []

        for i, (layer, act) in enumerate(zip(layers, self._activations)):
            a = act.cpu()
            # mean activation per neuron across the batch
            neuron_means = a.mean(dim=0)
            s = LayerStats(
                activations_mean=float(neuron_means.mean()),
                activations_std=float(neuron_means.std()),
                activations_min=float(neuron_means.min()),
                activations_max=float(neuron_means.max()),
                dead_neurons_pct=float((neuron_means <= 0).float().mean() * 100),
                weights_mean=float(layer.weight.data.mean()),
                weights_std=float(layer.weight.data.std()),
            )
            # gradient stats (only available after backward)
            if layer.weight.grad is not None:
                s.grad_mean = float(layer.weight.grad.abs().mean())
                s.grad_std = float(layer.weight.grad.std())

            stats.append(s)
            act_arrays.append(neuron_means.numpy())
            weight_arrays.append(layer.weight.data.cpu().numpy())

        return stats, act_arrays, weight_arrays
