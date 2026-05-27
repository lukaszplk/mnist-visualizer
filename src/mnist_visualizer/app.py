"""
mnist_visualizer.app
~~~~~~~~~~~~~~~~~~~~
Dear PyGui application: live network graph on the left, stats dashboard
on the right.

Layout
------
┌───────────────────────────────────────────────────────────────────┐
│  MNIST Visualizer                                          [Pause] │
├──────────────────────────────┬────────────────────────────────────┤
│  Network Graph               │  Stats Dashboard                   │
│  (nodes = activations,       │  ┌ Loss curve                      │
│   edges = weight magnitude)  │  ├ Accuracy curve (batch / epoch)  │
│                              │  ├ Per-layer table                  │
│                              │  └ Epoch / batch counter           │
└──────────────────────────────┴────────────────────────────────────┘
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np

from .model import TrainStats
from .trainer import Trainer

# ── Layout constants ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 1300, 760
GRAPH_W = 520
STATS_W = WIN_W - GRAPH_W - 20
PLOT_H  = 180
NODE_R  = 7          # node radius in pixels
MAX_EDGES_PER_LAYER = 80   # cap drawn edges for performance

# layer display sizes (cap large layers for drawing)
_DRAW_SIZES = [16, 16, 16, 10]   # nodes actually drawn per layer (visual)
_LAYER_NAMES = ["Input\n(784)", "Hidden 1\n(128)", "Hidden 2\n(64)", "Output\n(10)"]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _activation_color(value: float) -> tuple[int, int, int, int]:
    """Map normalised activation [0,1] → blue→white→red RGBA."""
    v = _clamp(value, 0.0, 1.0)
    if v < 0.5:
        t = v * 2
        r, g, b = int(30 + t * 200), int(100 + t * 120), int(220 - t * 20)
    else:
        t = (v - 0.5) * 2
        r, g, b = int(230 + t * 25), int(220 - t * 180), int(200 - t * 170)
    return (r, g, b, 220)


def _weight_color(value: float) -> tuple[int, int, int, int]:
    """Map weight sign/magnitude → green (positive) / red (negative)."""
    alpha = int(_clamp(abs(value) * 3, 0.05, 0.6) * 255)
    if value >= 0:
        return (50, 200, 80, alpha)
    return (200, 60, 60, alpha)


@dataclass
class _NodePos:
    x: float
    y: float


def _compute_node_positions(
    layer_sizes: list[int],
    canvas_w: float,
    canvas_y0: float,
    canvas_h: float,
) -> list[list[_NodePos]]:
    """Compute (x, y) for every drawn node."""
    n_layers = len(layer_sizes)
    x_step = canvas_w / (n_layers + 1)
    positions = []
    for li, n in enumerate(layer_sizes):
        x = x_step * (li + 1)
        y_step = canvas_h / (n + 1)
        layer_pos = [_NodePos(x, canvas_y0 + y_step * (ni + 1)) for ni in range(n)]
        positions.append(layer_pos)
    return positions


class App:
    def __init__(self) -> None:
        self._stats: Optional[TrainStats] = None
        self._lock = threading.Lock()
        self._trainer: Optional[Trainer] = None

        # history for plots
        self._loss_history: list[float] = []
        self._batch_acc_history: list[float] = []
        self._epoch_acc_history: list[float] = []
        self._val_acc_history: list[float] = []
        self._step = 0

    # ── Build GUI ─────────────────────────────────────────────────────────────

    def _build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="MNIST Neural Network Visualizer",
                            width=WIN_W, height=WIN_H, resizable=False)
        dpg.setup_dearpygui()

        with dpg.window(label="MNIST Visualizer", tag="main_win",
                        width=WIN_W, height=WIN_H, no_resize=True,
                        no_move=True, no_title_bar=True):

            # ── Top bar ───────────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("MNIST Neural Network Visualizer", color=(200, 220, 255))
                dpg.add_spacer(width=20)
                dpg.add_button(label="Pause", tag="btn_pause",
                               callback=self._on_pause)
                dpg.add_button(label="Stop", tag="btn_stop",
                               callback=self._on_stop)
                dpg.add_spacer(width=30)
                dpg.add_text("", tag="txt_status", color=(180, 255, 180))

            dpg.add_separator()

            # ── Two-column layout ─────────────────────────────────────────────
            with dpg.group(horizontal=True):

                # Left: network graph canvas
                with dpg.child_window(width=GRAPH_W, height=WIN_H - 60,
                                      tag="graph_win", border=True):
                    dpg.add_text("Network  (node colour = activation  |  edge = weight)",
                                 color=(160, 160, 200))
                    with dpg.drawlist(width=GRAPH_W - 10,
                                      height=WIN_H - 100, tag="graph_canvas"):
                        pass   # drawn dynamically

                # Right: stats
                with dpg.child_window(width=STATS_W, height=WIN_H - 60,
                                      tag="stats_win", border=True):
                    dpg.add_text("Training Statistics", color=(160, 160, 200))

                    # counter line
                    dpg.add_text("Epoch: —   Batch: —   Loss: —   Acc: —",
                                 tag="txt_counters", color=(220, 220, 100))
                    dpg.add_text("Val accuracy: —",
                                 tag="txt_val", color=(100, 220, 180))

                    dpg.add_separator()

                    # Loss plot
                    dpg.add_text("Loss", color=(200, 160, 100))
                    with dpg.plot(height=PLOT_H, width=-1, tag="plot_loss",
                                  no_title=True):
                        dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="loss_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="loss", tag="loss_y")
                        dpg.add_line_series([], [], label="loss",
                                            parent="loss_y", tag="series_loss")

                    # Accuracy plot
                    dpg.add_text("Accuracy (%)", color=(100, 200, 160))
                    with dpg.plot(height=PLOT_H, width=-1, tag="plot_acc",
                                  no_title=True):
                        dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="acc_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="%", tag="acc_y")
                        dpg.add_line_series([], [], label="batch acc",
                                            parent="acc_y", tag="series_batch_acc")
                        dpg.add_line_series([], [], label="epoch acc",
                                            parent="acc_y", tag="series_epoch_acc")
                        dpg.add_line_series([], [], label="val acc",
                                            parent="acc_y", tag="series_val_acc")
                        dpg.add_plot_legend()

                    dpg.add_separator()

                    # Per-layer table
                    dpg.add_text("Per-layer stats", color=(160, 200, 220))
                    with dpg.table(tag="layer_table", header_row=True,
                                   borders_innerH=True, borders_outerH=True,
                                   borders_innerV=True, borders_outerV=True,
                                   row_background=True):
                        dpg.add_table_column(label="Layer")
                        dpg.add_table_column(label="Act μ")
                        dpg.add_table_column(label="Act σ")
                        dpg.add_table_column(label="Dead %")
                        dpg.add_table_column(label="W μ")
                        dpg.add_table_column(label="∇W μ")

                        for i, name in enumerate(["fc1 (→128)", "fc2 (→64)", "fc3 (→10)"]):
                            with dpg.table_row(tag=f"row_{i}"):
                                dpg.add_text(name, tag=f"cell_{i}_name")
                                dpg.add_text("—", tag=f"cell_{i}_act_mean")
                                dpg.add_text("—", tag=f"cell_{i}_act_std")
                                dpg.add_text("—", tag=f"cell_{i}_dead")
                                dpg.add_text("—", tag=f"cell_{i}_w_mean")
                                dpg.add_text("—", tag=f"cell_{i}_grad")

        dpg.set_primary_window("main_win", True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_pause(self) -> None:
        if self._trainer:
            self._trainer.toggle_pause()
            label = "Resume" if self._trainer.paused else "Pause"
            dpg.set_item_label("btn_pause", label)

    def _on_stop(self) -> None:
        if self._trainer:
            self._trainer.stop()

    # ── Stats receiver (called from trainer thread) ────────────────────────────

    def _on_step(self, stats: TrainStats) -> None:
        with self._lock:
            self._stats = stats

    # ── Per-frame render update ────────────────────────────────────────────────

    def _update(self) -> None:
        with self._lock:
            stats = self._stats
            self._stats = None

        if stats is None:
            return

        self._step += 1
        self._loss_history.append(stats.loss)
        self._batch_acc_history.append(stats.batch_accuracy)
        self._epoch_acc_history.append(stats.epoch_accuracy)
        self._val_acc_history.append(stats.val_accuracy)

        xs = list(range(len(self._loss_history)))

        # counters
        dpg.set_value("txt_counters",
            f"Epoch: {stats.epoch}/{self._trainer._epochs}   "
            f"Batch: {stats.batch}/{stats.total_batches}   "
            f"Loss: {stats.loss:.4f}   "
            f"Batch acc: {stats.batch_accuracy:.1f}%")
        dpg.set_value("txt_val", f"Val accuracy: {stats.val_accuracy:.2f}%")

        # plots
        dpg.set_value("series_loss",       [xs, self._loss_history])
        dpg.set_value("series_batch_acc",  [xs, self._batch_acc_history])
        dpg.set_value("series_epoch_acc",  [xs, self._epoch_acc_history])
        dpg.set_value("series_val_acc",    [xs, self._val_acc_history])
        dpg.fit_axis_data("loss_x"); dpg.fit_axis_data("loss_y")
        dpg.fit_axis_data("acc_x");  dpg.fit_axis_data("acc_y")

        # per-layer table
        for i, ls in enumerate(stats.layer_stats):
            dead_color = (255, 100, 100) if ls.dead_neurons_pct > 20 else (200, 220, 200)
            dpg.set_value(f"cell_{i}_act_mean", f"{ls.activations_mean:.4f}")
            dpg.set_value(f"cell_{i}_act_std",  f"{ls.activations_std:.4f}")
            dpg.configure_item(f"cell_{i}_dead", color=dead_color)
            dpg.set_value(f"cell_{i}_dead",     f"{ls.dead_neurons_pct:.1f}%")
            dpg.set_value(f"cell_{i}_w_mean",   f"{ls.weights_mean:.4f}")
            dpg.set_value(f"cell_{i}_grad",
                f"{ls.grad_mean:.5f}" if ls.grad_mean else "—")

        # network graph
        self._draw_network(stats)

    def _draw_network(self, stats: TrainStats) -> None:
        dpg.delete_item("graph_canvas", children_only=True)

        canvas_w = GRAPH_W - 10
        canvas_h = WIN_H - 110
        y0 = 5

        positions = _compute_node_positions(
            _DRAW_SIZES, canvas_w, y0, canvas_h - y0
        )

        # ── Draw edges (sampled for performance) ──────────────────────────────
        if stats.weights:
            for li in range(len(positions) - 1):
                src_pos = positions[li]
                dst_pos = positions[li + 1]
                W = stats.weights[li] if li < len(stats.weights) else None
                if W is None:
                    continue

                n_src = len(src_pos)
                n_dst = len(dst_pos)

                # map W rows=dst, cols=src  →  subsample indices
                step_src = max(1, W.shape[1] // n_src)
                step_dst = max(1, W.shape[0] // n_dst)

                for di, dp_ in enumerate(dst_pos):
                    w_row = W[di * step_dst, :]
                    for si, sp in enumerate(src_pos):
                        w_val = float(w_row[si * step_src])
                        col = _weight_color(w_val)
                        thickness = _clamp(abs(w_val) * 2, 0.3, 2.5)
                        dpg.draw_line(
                            (sp.x, sp.y), (dp_.x, dp_.y),
                            color=col, thickness=thickness,
                            parent="graph_canvas",
                        )

        # ── Draw nodes ────────────────────────────────────────────────────────
        layer_acts: list[Optional[np.ndarray]] = []
        # input layer has no activation stats — use uniform grey
        layer_acts.append(None)
        for a in stats.activations:
            layer_acts.append(a)

        for li, (layer_pos, acts) in enumerate(zip(positions, layer_acts)):
            n_full = _DRAW_SIZES[li]
            for ni, pos in enumerate(layer_pos):
                if acts is not None and len(acts) > 0:
                    step = max(1, len(acts) // n_full)
                    raw = float(acts[min(ni * step, len(acts) - 1)])
                    # normalise to [0,1] using tanh
                    norm = (math.tanh(raw) + 1) / 2
                    col = _activation_color(norm)
                else:
                    col = (120, 130, 160, 200)

                dpg.draw_circle(
                    center=(pos.x, pos.y), radius=NODE_R,
                    color=(255, 255, 255, 80),
                    fill=col,
                    parent="graph_canvas",
                )

        # ── Layer labels ─────────────────────────────────────────────────────
        n_layers = len(positions)
        x_step = canvas_w / (n_layers + 1)
        for li, label in enumerate(_LAYER_NAMES):
            x = x_step * (li + 1)
            dpg.draw_text(
                (x - 25, canvas_h - 20), label.replace("\n", " "),
                color=(180, 190, 220), size=12,
                parent="graph_canvas",
            )

    # ── Main ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._build()
        dpg.show_viewport()

        self._trainer = Trainer(on_step=self._on_step)
        self._trainer.start()

        dpg.set_value("txt_status", "Training…")

        while dpg.is_dearpygui_running():
            self._update()
            dpg.render_dearpygui_frame()

        if self._trainer:
            self._trainer.stop()
        dpg.destroy_context()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
