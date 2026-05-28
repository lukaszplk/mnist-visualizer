"""
mnist_visualizer.app
~~~~~~~~~~~~~~~~~~~~
Dear PyGui application with three panels:

  Left   — live network graph (node colour = activation, edge = weight)
  Centre — stats dashboard (loss / accuracy plots, per-layer table)
  Right  — draw-a-digit pad with inference and decision-path highlighting

Controls in the top bar: Start · Pause · Stop · Epochs · Batch · LR
"""

from __future__ import annotations

import colorsys
import math
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np

from .model import TrainStats
from .trainer import Trainer

# ── Layout (initial defaults — recalculated on resize) ────────────────────────
WIN_W, WIN_H   = 1580, 800
_GRAPH_RATIO   = 0.29     # fraction of viewport width
_STATS_RATIO   = 0.37
# draw panel gets the remainder
CONTENT_H      = WIN_H - 90   # updated dynamically
PLOT_H         = 165           # initial plot height (recalculated on resize)

NODE_R         = 7
DRAW_PX        = 9          # screen pixels per MNIST pixel

# node-activity chart
_NODE_SHOWN    = [16, 16, 10]     # fc1, fc2, fc3
_NODE_STEPS    = [8,   4,  1]     # sampling stride
_HIST_LEN      = 300              # batches in rolling window
_ROLLING_N     = 20               # window for rolling avg / std

# dataset browser
_IMG_SZ        = 252                   # 9 × 28 — fills ~half the stats panel width

# activations tab cell sizes
_ACT_INPUT_PX  = 7     # px per MNIST pixel in input grid  (28×28 → 196×196)
_ACT_H1_PX     = 20    # px per neuron in hidden-1 grid    (16×8  → 320×160)
_ACT_H2_PX     = 26    # px per neuron in hidden-2 grid    (8×8   → 208×208)
_ACT_BAR_W     = 340   # width of output probability bars
_ACT_BAR_H     = 28    # height per class bar
_METRICS        = ["Activation value", "Running average", "Rolling std"]
_WEIGHT_METRICS = ["Weight mean", "Weight std"]
DRAW_GRID      = 28
DRAW_CANVAS_SZ = DRAW_PX * DRAW_GRID   # 252

_DRAW_SIZES  = [16, 16, 16, 10]
_LAYER_NAMES = ["Input (784)", "Hidden 1 (128)", "Hidden 2 (64)", "Output (10)"]
_DIGIT_LABELS = [str(i) for i in range(10)]


# ── Colour helpers ────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _activation_color(value: float, highlight: bool = False) -> tuple:
    v = _clamp(value, 0.0, 1.0)
    if highlight:
        return (255, 220, 50, 255)
    if v < 0.5:
        t = v * 2
        return (int(30 + t*200), int(100 + t*120), int(220 - t*20), 220)
    t = (v - 0.5) * 2
    return (int(230 + t*25), int(220 - t*180), int(200 - t*170), 220)


def _node_color(node_idx: int, n_nodes: int, alpha: int = 220) -> tuple:
    """Evenly spaced hue wheel colour for a node line."""
    h = node_idx / max(n_nodes, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


def _weight_color(value: float, highlight: bool = False) -> tuple:
    if highlight:
        return (255, 220, 50, 200)
    alpha = int(_clamp(abs(value) * 3, 0.05, 0.55) * 255)
    return (50, 200, 80, alpha) if value >= 0 else (200, 60, 60, alpha)


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
    n_layers = len(layer_sizes)
    x_step   = canvas_w / (n_layers + 1)
    positions = []
    for li, n in enumerate(layer_sizes):
        x      = x_step * (li + 1)
        y_step = canvas_h / (n + 1)
        positions.append([_NodePos(x, canvas_y0 + y_step * (ni + 1)) for ni in range(n)])
    return positions


# ── App ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self) -> None:
        self._stats: Optional[TrainStats] = None
        self._lock  = threading.Lock()
        self._trainer: Optional[Trainer] = None

        self._loss_history:      list[float] = []
        self._batch_acc_history: list[float] = []
        self._epoch_acc_history: list[float] = []
        self._val_acc_history:   list[float] = []

        # drawing pad state
        self._draw_grid  = np.zeros((DRAW_GRID, DRAW_GRID), dtype=np.float32)
        self._draw_dirty = False
        self._infer_result: Optional[tuple] = None   # (pred, probs, acts, weights)

        # dynamic layout (updated on resize)
        self._vp_w      = WIN_W
        self._vp_h      = WIN_H
        self._graph_w   = int(WIN_W * _GRAPH_RATIO)
        self._stats_w   = int(WIN_W * _STATS_RATIO)
        self._draw_w    = WIN_W - self._graph_w - self._stats_w - 30
        self._content_h = WIN_H - 90

        # per-node histories: [layer][node] → list of floats
        self._node_act_history: list[list[list[float]]] = [
            [[] for _ in range(_NODE_SHOWN[li])] for li in range(3)
        ]
        self._node_avg_history: list[list[list[float]]] = [
            [[] for _ in range(_NODE_SHOWN[li])] for li in range(3)
        ]
        self._node_std_history: list[list[list[float]]] = [
            [[] for _ in range(_NODE_SHOWN[li])] for li in range(3)
        ]
        self._node_weight_mean_history: list[list[list[float]]] = [
            [[] for _ in range(_NODE_SHOWN[li])] for li in range(3)
        ]
        self._node_weight_std_history: list[list[list[float]]] = [
            [[] for _ in range(_NODE_SHOWN[li])] for li in range(3)
        ]
        self._node_metric:   str = _METRICS[0]
        self._weight_metric: str = _WEIGHT_METRICS[0]

        # per-layer node filters: list of visible draw-indices (None = show all)
        self._node_filter:   list[Optional[set]] = [None, None, None]
        self._weight_filter: list[Optional[set]] = [None, None, None]

        # dataset browser
        self._dataset          = None   # loaded lazily
        self._ds_offset        = 0
        self._ds_preds: list[int] = []
        self._ds_needs_refresh = False  # set by bg thread, consumed by main thread
        self._ds_error: str = ""

        # activations tab
        self._act_input_img: Optional[np.ndarray] = None  # last (28,28) inference input
        self._act_needs_refresh = False

        # metrics / confusion matrix
        self._cm: Optional[np.ndarray]      = None   # (10,10) int
        self._class_metrics: Optional[dict] = None   # per-class stats
        self._metrics_needs_refresh         = False
        self._metrics_computing             = False
        self._last_val_epoch                = -1

        # graph redraw throttle
        self._last_graph_step   = -1
        self._graph_redraw_rate = 5    # redraw every N training steps
        self._inference_drawn   = False

        # highlighted nodes/edges from last inference
        self._highlight_nodes:  list[set] = [set() for _ in range(4)]
        self._highlight_edges:  list[set] = [set() for _ in range(3)]
        self._inference_active  = False

    # ── Per-series colour theme ───────────────────────────────────────────────

    @staticmethod
    def _make_line_theme(color: tuple) -> int:
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                    category=dpg.mvThemeCat_Plots)
        return t

    # ── Resize ────────────────────────────────────────────────────────────────

    def _resize(self, vp_w: int, vp_h: int) -> None:
        self._vp_w      = vp_w
        self._vp_h      = vp_h
        self._graph_w   = int(vp_w * _GRAPH_RATIO)
        self._stats_w   = int(vp_w * _STATS_RATIO)
        self._draw_w    = vp_w - self._graph_w - self._stats_w - 30
        self._content_h = vp_h - 90

        ph = max(120, int(self._content_h * 0.21))   # proportional plot height

        dpg.set_item_width( "graph_win",    self._graph_w)
        dpg.set_item_height("graph_win",    self._content_h)
        dpg.set_item_width( "graph_canvas", self._graph_w - 12)
        dpg.set_item_height("graph_canvas", self._content_h - 36)

        dpg.set_item_width( "stats_win",    self._stats_w)
        dpg.set_item_height("stats_win",    self._content_h)

        dpg.set_item_width( "draw_win",     self._draw_w)
        dpg.set_item_height("draw_win",     self._content_h)

        dpg.set_item_height("plot_loss",    ph)
        dpg.set_item_height("plot_acc",     ph)
        for li in range(3):
            dpg.set_item_height(f"plot_nodes_{li}", max(100, ph - 20))
            dpg.set_item_height(f"wplot_{li}",      max(100, ph - 20))

        dpg.set_item_width("plot_conf", self._draw_w - 24)

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (10,  10,  15,  255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        (16,  16,  24,  255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (30,  30,  45,  255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (45,  45,  65,  255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        (5,   5,   10,  255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (10,  10,  20,  255))
                dpg.add_theme_color(dpg.mvThemeCol_Button,         (40,  60,  100, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (60,  90,  150, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (80,  120, 200, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header,         (40,  55,  90,  255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (55,  75,  120, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,           (210, 215, 230, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border,         (50,  55,  80,  255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,    (10,  10,  15,  255))
                dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, (50, 55, 80, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBg,     (18,  18,  28,  255))
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt,  (24,  24,  36,  255))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 5)
        dpg.bind_theme(global_theme)

    # ── Build GUI ─────────────────────────────────────────────────────────────

    def _build(self) -> None:
        dpg.create_context()
        self._apply_theme()
        dpg.create_viewport(
            title="MNIST Neural Network Visualizer",
            width=WIN_W, height=WIN_H,
            min_width=900, min_height=500,
            clear_color=(5, 5, 10, 255),
        )
        dpg.setup_dearpygui()

        # (no texture registry needed — dataset image drawn via drawlist)

        with dpg.window(tag="main_win",
                        no_resize=True, no_move=True, no_title_bar=True,
                        no_scrollbar=True):

            # ── Top control bar ───────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("MNIST Visualizer", color=(140, 170, 255))
                dpg.add_spacer(width=16)
                dpg.add_button(label="Start", tag="btn_start",
                               callback=self._on_start,
                               width=90)
                dpg.add_button(label="Pause", tag="btn_pause",
                               callback=self._on_pause,
                               width=90, enabled=False)
                dpg.add_button(label="Stop", tag="btn_stop",
                               callback=self._on_stop,
                               width=90, enabled=False)
                dpg.add_spacer(width=20)
                dpg.add_text("Epochs:", color=(160, 160, 200))
                dpg.add_input_int(tag="inp_epochs", default_value=20,
                                  min_value=1, max_value=200,
                                  width=70, step=0)
                dpg.add_spacer(width=8)
                dpg.add_text("Batch:", color=(160, 160, 200))
                dpg.add_input_int(tag="inp_batch", default_value=64,
                                  min_value=8, max_value=512,
                                  width=70, step=0)
                dpg.add_spacer(width=8)
                dpg.add_text("LR:", color=(160, 160, 200))
                dpg.add_input_float(tag="inp_lr", default_value=0.001,
                                    min_value=1e-5, max_value=0.1,
                                    format="%.4f", width=80, step=0)
                dpg.add_spacer(width=20)
                dpg.add_text("", tag="txt_status", color=(100, 220, 140))

            dpg.add_separator()

            # ── Three-panel row ───────────────────────────────────────────────
            with dpg.group(horizontal=True):

                # ── Left: network graph ───────────────────────────────────────
                with dpg.child_window(width=self._graph_w, height=self._content_h,
                                      tag="graph_win", border=True, no_scrollbar=True):
                    dpg.add_text("Network  (node = activation  |  edge = weight)",
                                 color=(120, 130, 190))
                    dpg.add_separator()
                    with dpg.drawlist(width=self._graph_w - 12,
                                      height=self._content_h - 36,
                                      tag="graph_canvas"):
                        pass

                # ── Centre: stats ─────────────────────────────────────────────
                with dpg.child_window(width=self._stats_w, height=self._content_h,
                                      tag="stats_win", border=True):

                    with dpg.tab_bar():

                        # ── Tab 1: Overview ───────────────────────────────────
                        with dpg.tab(label="Overview"):
                            dpg.add_text("Epoch: —   Batch: —   Loss: —   Acc: —",
                                         tag="txt_counters", color=(220, 200, 80))
                            dpg.add_text("Val accuracy: —",
                                         tag="txt_val", color=(80, 200, 160))
                            dpg.add_separator()

                            dpg.add_text("Loss", color=(200, 140, 80))
                            with dpg.plot(height=PLOT_H, width=-1, tag="plot_loss",
                                          no_title=True):
                                dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="loss_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="loss", tag="loss_y")
                                dpg.add_line_series([], [], label="loss",
                                                    parent="loss_y", tag="series_loss")

                            dpg.add_text("Accuracy (%)", color=(80, 180, 140))
                            with dpg.plot(height=PLOT_H, width=-1, tag="plot_acc",
                                          no_title=True):
                                dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="acc_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="%",    tag="acc_y")
                                dpg.add_line_series([], [], label="batch",
                                                    parent="acc_y", tag="series_batch_acc")
                                dpg.add_line_series([], [], label="epoch",
                                                    parent="acc_y", tag="series_epoch_acc")
                                dpg.add_line_series([], [], label="val",
                                                    parent="acc_y", tag="series_val_acc")
                                dpg.add_plot_legend()

                            dpg.add_separator()
                            dpg.add_text("Per-layer stats", color=(120, 170, 210))
                            with dpg.table(tag="layer_table", header_row=True,
                                           borders_innerH=True, borders_outerH=True,
                                           borders_innerV=True, borders_outerV=True,
                                           row_background=True):
                                for col in ["Layer", "Act μ", "Act σ", "Dead %", "W μ", "∇W μ"]:
                                    dpg.add_table_column(label=col)
                                for i, name in enumerate(["fc1 (128)", "fc2 (64)", "fc3 (10)"]):
                                    with dpg.table_row(tag=f"row_{i}"):
                                        dpg.add_text(name, tag=f"c{i}_name")
                                        dpg.add_text("—",  tag=f"c{i}_am")
                                        dpg.add_text("—",  tag=f"c{i}_as")
                                        dpg.add_text("—",  tag=f"c{i}_dead")
                                        dpg.add_text("—",  tag=f"c{i}_wm")
                                        dpg.add_text("—",  tag=f"c{i}_gm")

                        # ── Tab 2: Node Activity ──────────────────────────────
                        with dpg.tab(label="Node Activity"):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Metric:", color=(160, 160, 200))
                                dpg.add_combo(
                                    _METRICS,
                                    default_value=_METRICS[0],
                                    tag="combo_metric",
                                    width=180,
                                    callback=self._on_metric_change,
                                )
                                dpg.add_text(
                                    "  (sampled: 16 / 16 / 10 nodes)",
                                    color=(100, 110, 150))
                            dpg.add_separator()

                            _layer_labels = [
                                ("fc1  (128 nodes, 16 shown)", 175),
                                ("fc2  (64 nodes, 16 shown)",  155),
                                ("fc3 / Output  (10 nodes)",   155),
                            ]
                            for li, (lbl, ph) in enumerate(_layer_labels):
                                with dpg.group(horizontal=True):
                                    dpg.add_text(lbl, color=(160, 170, 220))
                                    dpg.add_spacer(width=8)
                                    dpg.add_text("show:", color=(120, 120, 160))
                                    dpg.add_input_text(
                                        tag=f"inp_node_filter_{li}",
                                        hint=f"0-{_NODE_SHOWN[li]-1}, empty=all",
                                        width=130, on_enter=True,
                                        user_data=li,
                                        callback=lambda s, a, u:
                                            self._apply_filter(u, a, "node"),
                                    )
                                    dpg.add_button(
                                        label="all", width=32,
                                        user_data=li,
                                        callback=lambda s, a, u:
                                            self._clear_filter(u, "node"),
                                    )
                                with dpg.plot(height=ph, width=-1,
                                              tag=f"plot_nodes_{li}",
                                              no_title=True):
                                    dpg.add_plot_axis(dpg.mvXAxis,
                                                      label="batch",
                                                      tag=f"node_x_{li}")
                                    dpg.add_plot_axis(dpg.mvYAxis,
                                                      label="activation",
                                                      tag=f"node_y_{li}")
                                    n = _NODE_SHOWN[li]
                                    step = _NODE_STEPS[li]
                                    for ni in range(n):
                                        node_idx = ni * step
                                        col = _node_color(ni, n)
                                        lbl_s = (f"out:{node_idx}"
                                                 if li == 2
                                                 else f"n{node_idx}")
                                        dpg.add_line_series(
                                            [], [],
                                            label=lbl_s,
                                            parent=f"node_y_{li}",
                                            tag=f"node_series_{li}_{ni}",
                                        )
                                        dpg.bind_item_theme(
                                            f"node_series_{li}_{ni}",
                                            self._make_line_theme(col),
                                        )
                                    dpg.add_plot_legend()

                        # ── Tab 3: Weights per node ───────────────────────────
                        with dpg.tab(label="Weights"):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Metric:", color=(160, 160, 200))
                                dpg.add_combo(
                                    _WEIGHT_METRICS,
                                    default_value=_WEIGHT_METRICS[0],
                                    tag="combo_weight_metric",
                                    width=160,
                                    callback=self._on_weight_metric_change,
                                )
                                dpg.add_text(
                                    "  W[node, :] — incoming weights per node",
                                    color=(100, 110, 150))
                            dpg.add_separator()

                            _wlayer_labels = [
                                ("fc1  (128 nodes, 16 shown)", 175),
                                ("fc2  (64 nodes, 16 shown)",  155),
                                ("fc3 / Output  (10 nodes)",   155),
                            ]
                            for li, (lbl, ph) in enumerate(_wlayer_labels):
                                with dpg.group(horizontal=True):
                                    dpg.add_text(lbl, color=(160, 170, 220))
                                    dpg.add_spacer(width=8)
                                    dpg.add_text("show:", color=(120, 120, 160))
                                    dpg.add_input_text(
                                        tag=f"inp_weight_filter_{li}",
                                        hint=f"0-{_NODE_SHOWN[li]-1}, empty=all",
                                        width=130, on_enter=True,
                                        user_data=li,
                                        callback=lambda s, a, u:
                                            self._apply_filter(u, a, "weight"),
                                    )
                                    dpg.add_button(
                                        label="all", width=32,
                                        user_data=li,
                                        callback=lambda s, a, u:
                                            self._clear_filter(u, "weight"),
                                    )
                                with dpg.plot(height=ph, width=-1,
                                              tag=f"wplot_{li}",
                                              no_title=True):
                                    dpg.add_plot_axis(dpg.mvXAxis,
                                                      label="batch",
                                                      tag=f"wnode_x_{li}")
                                    dpg.add_plot_axis(dpg.mvYAxis,
                                                      label="weight",
                                                      tag=f"wnode_y_{li}")
                                    n    = _NODE_SHOWN[li]
                                    step = _NODE_STEPS[li]
                                    for ni in range(n):
                                        node_idx = ni * step
                                        col = _node_color(ni, n)
                                        lbl_s = (f"out:{node_idx}"
                                                 if li == 2
                                                 else f"n{node_idx}")
                                        dpg.add_line_series(
                                            [], [],
                                            label=lbl_s,
                                            parent=f"wnode_y_{li}",
                                            tag=f"wseries_{li}_{ni}",
                                        )
                                        dpg.bind_item_theme(
                                            f"wseries_{li}_{ni}",
                                            self._make_line_theme(col),
                                        )
                                    dpg.add_plot_legend()

                        # ── Tab 4: Dataset ───────────────────────────────────
                        with dpg.tab(label="Dataset"):
                            # nav row
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="< Prev", width=60,
                                               callback=self._on_ds_prev)
                                dpg.add_button(label="Next >", width=60,
                                               callback=self._on_ds_next)
                                dpg.add_spacer(width=8)
                                dpg.add_button(label="Predict",
                                               width=80,
                                               tag="btn_ds_predict",
                                               callback=self._on_ds_predict)
                                dpg.add_spacer(width=12)
                                dpg.add_text("Loading...", tag="txt_ds_page",
                                             color=(160, 160, 200))
                            dpg.add_separator()

                            # digit drawn as pixel rectangles (no texture needed)
                            with dpg.drawlist(width=_IMG_SZ, height=_IMG_SZ,
                                              tag="ds_drawlist"):
                                pass
                            dpg.add_spacer(height=6)
                            dpg.add_text("—", tag="lbl_ds_0",
                                         color=(200, 200, 160))
                            dpg.add_text("", tag="lbl_ds_pred",
                                         color=(80, 220, 80))

                        # ── Tab 5: Metrics ────────────────────────────────────
                        with dpg.tab(label="Metrics"):
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Recompute",
                                               tag="btn_recompute",
                                               callback=self._on_recompute)
                                dpg.add_spacer(width=10)
                                dpg.add_text("", tag="txt_metrics_status",
                                             color=(160, 200, 160))
                            dpg.add_text("", tag="txt_overall_metrics",
                                         color=(220, 200, 80))
                            dpg.add_separator()

                            dpg.add_text("Confusion matrix  (row=actual, col=predicted)",
                                         color=(140, 150, 200))
                            with dpg.plot(height=260, width=-1,
                                          tag="plot_cm", no_title=True,
                                          equal_aspects=True):
                                cm_ax = dpg.add_plot_axis(
                                    dpg.mvXAxis, label="Predicted",
                                    tag="cm_x", no_gridlines=True)
                                dpg.set_axis_limits("cm_x", -0.5, 9.5)
                                dpg.set_axis_ticks("cm_x",
                                    tuple((str(i), float(i)) for i in range(10)))
                                cm_ay = dpg.add_plot_axis(
                                    dpg.mvYAxis, label="Actual",
                                    tag="cm_y", no_gridlines=True)
                                dpg.set_axis_limits("cm_y", -0.5, 9.5)
                                dpg.set_axis_ticks("cm_y",
                                    tuple((str(i), float(i)) for i in range(10)))
                                dpg.add_heat_series(
                                    [0.0] * 100,
                                    rows=10, cols=10,
                                    scale_min=0, scale_max=1,
                                    bounds_min=(-0.5, -0.5),
                                    bounds_max=(9.5, 9.5),
                                    parent="cm_y", tag="series_cm",
                                    format=""
                                )
                                dpg.bind_colormap("plot_cm",
                                                  dpg.mvPlotColormap_Hot)

                            dpg.add_separator()
                            dpg.add_text("Per-class metrics", color=(140, 150, 200))
                            with dpg.table(tag="metrics_table", header_row=True,
                                           borders_innerH=True, borders_outerH=True,
                                           borders_innerV=True, borders_outerV=True,
                                           row_background=True):
                                for col in ["Class", "Precision", "Recall",
                                            "F1", "Support"]:
                                    dpg.add_table_column(label=col)
                                for i in range(10):
                                    with dpg.table_row(tag=f"mrow_{i}"):
                                        dpg.add_text(str(i),  tag=f"mc_{i}_cls",
                                                     color=(200, 200, 120))
                                        dpg.add_text("—", tag=f"mc_{i}_prec")
                                        dpg.add_text("—", tag=f"mc_{i}_rec")
                                        dpg.add_text("—", tag=f"mc_{i}_f1")
                                        dpg.add_text("—", tag=f"mc_{i}_sup")
                                with dpg.table_row(tag="mrow_macro"):
                                    dpg.add_text("macro", tag="mc_macro_cls",
                                                 color=(180, 180, 255))
                                    dpg.add_text("—", tag="mc_macro_prec")
                                    dpg.add_text("—", tag="mc_macro_rec")
                                    dpg.add_text("—", tag="mc_macro_f1")
                                    dpg.add_text("—", tag="mc_macro_sup")

                        # ── Tab 6: Activations ───────────────────────────────
                        with dpg.tab(label="Activations"):
                            dpg.add_text("", tag="txt_act_status",
                                         color=(160, 160, 200))
                            dpg.add_separator()
                            with dpg.child_window(tag="act_scroll",
                                                  width=-1, height=-1,
                                                  border=False):
                                # Input layer ─ 28×28 grid
                                dpg.add_text("Input  (784 pixels)",
                                             color=(140, 170, 220))
                                with dpg.drawlist(
                                        width=28 * _ACT_INPUT_PX,
                                        height=28 * _ACT_INPUT_PX,
                                        tag="act_draw_input"):
                                    pass
                                dpg.add_spacer(height=8)

                                # Hidden 1 ─ 128 neurons as 16×8 grid
                                dpg.add_text("Hidden 1  (128 neurons)",
                                             color=(140, 170, 220))
                                with dpg.drawlist(
                                        width=16 * _ACT_H1_PX,
                                        height=8  * _ACT_H1_PX,
                                        tag="act_draw_h1"):
                                    pass
                                dpg.add_spacer(height=8)

                                # Hidden 2 ─ 64 neurons as 8×8 grid
                                dpg.add_text("Hidden 2  (64 neurons)",
                                             color=(140, 170, 220))
                                with dpg.drawlist(
                                        width=8 * _ACT_H2_PX,
                                        height=8 * _ACT_H2_PX,
                                        tag="act_draw_h2"):
                                    pass
                                dpg.add_spacer(height=8)

                                # Output ─ 10 probability bars
                                dpg.add_text("Output  (10 classes)",
                                             color=(140, 170, 220))
                                with dpg.drawlist(
                                        width=_ACT_BAR_W,
                                        height=10 * _ACT_BAR_H,
                                        tag="act_draw_out"):
                                    pass

                # ── Right: draw & recognise ───────────────────────────────────
                with dpg.child_window(width=self._draw_w, height=self._content_h,
                                      tag="draw_win", border=True):
                    dpg.add_text("Draw a digit (0–9)", color=(120, 130, 190))
                    dpg.add_separator()

                    # drawing canvas
                    with dpg.drawlist(width=DRAW_CANVAS_SZ, height=DRAW_CANVAS_SZ,
                                      tag="draw_canvas"):
                        pass

                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Clear",     width=100,
                                       callback=self._on_clear_draw)
                        dpg.add_button(label="Recognise", width=110,
                                       callback=self._on_recognise,
                                       tag="btn_recognise")

                    dpg.add_spacer(height=8)
                    dpg.add_text("", tag="txt_pred",  color=(255, 220, 80))
                    dpg.add_spacer(height=4)

                    # confidence bar chart
                    with dpg.plot(height=160, width=self._draw_w - 24,
                                  tag="plot_conf", no_title=True,
                                  no_mouse_pos=True):
                        dpg.add_plot_axis(dpg.mvXAxis, tag="conf_x", no_gridlines=True)
                        dpg.set_axis_ticks("conf_x",
                            tuple((str(i), float(i)) for i in range(10)))
                        dpg.add_plot_axis(dpg.mvYAxis, tag="conf_y",
                                          label="%", no_gridlines=True)
                        dpg.set_axis_limits("conf_y", 0, 100)
                        dpg.add_bar_series(list(range(10)), [0]*10,
                                           weight=0.6,
                                           parent="conf_y", tag="series_conf")

        dpg.set_primary_window("main_win", True)

    # ── Top-bar callbacks ─────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._trainer and self._trainer.is_running:
            return
        epochs     = dpg.get_value("inp_epochs")
        batch_size = dpg.get_value("inp_batch")
        lr         = float(dpg.get_value("inp_lr"))

        self._trainer = Trainer(
            on_step=self._on_step,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        self._trainer.start()

        dpg.configure_item("btn_start",  enabled=False)
        dpg.configure_item("btn_pause",  enabled=True)
        dpg.configure_item("btn_stop",   enabled=True)
        dpg.configure_item("inp_epochs", enabled=False)
        dpg.configure_item("inp_batch",  enabled=False)
        dpg.configure_item("inp_lr",     enabled=False)
        dpg.set_value("txt_status", "Training...")

    def _on_pause(self) -> None:
        if not self._trainer:
            return
        self._trainer.toggle_pause()
        if self._trainer.paused:
            dpg.set_item_label("btn_pause", "Resume")
            dpg.set_value("txt_status", "Paused")
        else:
            dpg.set_item_label("btn_pause", "Pause")
            dpg.set_value("txt_status", "Training...")

    def _on_stop(self) -> None:
        if self._trainer:
            self._trainer.stop()
        dpg.configure_item("btn_start",  enabled=True)
        dpg.configure_item("btn_pause",  enabled=False)
        dpg.configure_item("btn_stop",   enabled=False)
        dpg.configure_item("inp_epochs", enabled=True)
        dpg.configure_item("inp_batch",  enabled=True)
        dpg.configure_item("inp_lr",     enabled=True)
        dpg.set_item_label("btn_pause",  "Pause")
        dpg.set_value("txt_status", "Stopped")

    # ── Dataset callbacks ─────────────────────────────────────────────────────

    def _ensure_dataset(self) -> bool:
        """Load the MNIST test set.  Safe to call from any thread."""
        if self._dataset is not None:
            return True
        try:
            from torchvision import datasets, transforms
            from pathlib import Path
            t = transforms.ToTensor()
            self._dataset = datasets.MNIST(
                Path.home() / ".cache" / "mnist_visualizer",
                train=False, download=True, transform=t,
            )
            self._ds_needs_refresh = True
            return True
        except Exception as exc:
            # surface the error on the Dataset tab's status text
            self._ds_error = str(exc)
            self._ds_needs_refresh = True
            return False

    # ── Activations tab ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_act_grid(tag: str, values: "np.ndarray",
                       cols: int, cell: int) -> None:
        """Fill a drawlist with a colored grid of neuron activations."""
        import numpy as _np
        dpg.delete_item(tag, children_only=True)
        rows = int(_np.ceil(len(values) / cols))
        v_min, v_max = float(values.min()), float(values.max())
        span = (v_max - v_min) or 1.0
        for idx, raw in enumerate(values):
            v = float((raw - v_min) / span)   # normalise to [0,1]
            r = int(v * 80)
            g = int(v * 160)
            b = int(50 + v * 200)
            col_i = idx % cols
            row_i = idx // cols
            x0, y0 = col_i * cell, row_i * cell
            dpg.draw_rectangle(
                (x0, y0), (x0 + cell - 1, y0 + cell - 1),
                color=(r, g, b, 255), fill=(r, g, b, 255),
                parent=tag, thickness=0,
            )
        # draw thin grid lines for readability
        w = cols * cell
        h = rows * cell
        for c in range(cols + 1):
            x = c * cell
            dpg.draw_line((x, 0), (x, h),
                          color=(30, 30, 40, 180), parent=tag, thickness=1)
        for r in range(rows + 1):
            y = r * cell
            dpg.draw_line((0, y), (w, y),
                          color=(30, 30, 40, 180), parent=tag, thickness=1)

    @staticmethod
    def _draw_act_input(img: "np.ndarray") -> None:
        """Render the 28×28 input image into act_draw_input."""
        import numpy as _np
        dpg.delete_item("act_draw_input", children_only=True)
        px = _ACT_INPUT_PX
        img_g = _np.clip(img ** 0.5, 0.0, 1.0)
        # dark background
        dpg.draw_rectangle(
            (0, 0), (28 * px, 28 * px),
            color=(12, 12, 20, 255), fill=(12, 12, 20, 255),
            parent="act_draw_input", thickness=0,
        )
        for row in range(28):
            for col in range(28):
                v = float(img_g[row, col])
                if v < 0.03:
                    continue
                vi = int(v * 255)
                color = (vi, vi, min(vi + 40, 255), 255)
                x0, y0 = col * px, row * px
                dpg.draw_rectangle(
                    (x0, y0), (x0 + px, y0 + px),
                    color=color, fill=color,
                    parent="act_draw_input", thickness=0,
                )

    @staticmethod
    def _draw_act_bars(probs: "np.ndarray") -> None:
        """Render output class probability bars into act_draw_out."""
        dpg.delete_item("act_draw_out", children_only=True)
        best = int(probs.argmax())
        for i, p in enumerate(probs):
            y0 = i * _ACT_BAR_H
            bar_w = int(p * (_ACT_BAR_W - 60))
            # bar fill
            if i == best:
                color = (80, 220, 120, 255)
            else:
                color = (60, 100, 180, 255)
            dpg.draw_rectangle(
                (50, y0 + 2), (50 + max(bar_w, 2), y0 + _ACT_BAR_H - 2),
                color=color, fill=color,
                parent="act_draw_out", thickness=0,
            )
            # class label
            dpg.draw_text(
                (2, y0 + 4), f"{i}:",
                color=(200, 200, 160, 255), size=14,
                parent="act_draw_out",
            )
            # percentage
            dpg.draw_text(
                (55 + max(bar_w, 2), y0 + 4), f"{p*100:.1f}%",
                color=(160, 160, 200, 255), size=13,
                parent="act_draw_out",
            )

    def _update_activations_display(
        self,
        acts: list,
        probs: "np.ndarray",
        input_img: Optional["np.ndarray"] = None,
        status: str = "",
    ) -> None:
        """Refresh the Activations tab. Must run on the main thread."""
        import numpy as _np
        if status:
            dpg.set_value("txt_act_status", status)

        if input_img is not None:
            self._draw_act_input(input_img)
        elif self._act_input_img is not None:
            self._draw_act_input(self._act_input_img)

        # hidden layers
        if len(acts) >= 1:
            a1 = _np.array(acts[0], dtype=_np.float32)
            self._draw_act_grid("act_draw_h1", a1, cols=16, cell=_ACT_H1_PX)
        if len(acts) >= 2:
            a2 = _np.array(acts[1], dtype=_np.float32)
            self._draw_act_grid("act_draw_h2", a2, cols=8,  cell=_ACT_H2_PX)

        # output
        if probs is not None and len(probs) == 10:
            self._draw_act_bars(_np.array(probs, dtype=_np.float32))

    def _draw_mnist_image(self, img: "np.ndarray") -> None:
        """Render a (28,28) float32 image into the ds_drawlist via pixel rectangles."""
        dpg.delete_item("ds_drawlist", children_only=True)
        px = _IMG_SZ / 28          # 252/28 = 9.0 screen pixels per MNIST pixel

        # dark background so digit has contrast
        dpg.draw_rectangle(
            (0, 0), (_IMG_SZ, _IMG_SZ),
            color=(15, 15, 25, 255), fill=(15, 15, 25, 255),
            parent="ds_drawlist", thickness=0,
        )

        import numpy as _np
        # stronger gamma + contrast stretch for a crisp, bright digit
        img_g = _np.clip(img ** 0.4, 0.0, 1.0)
        img_g = _np.clip((img_g - 0.1) / 0.9, 0.0, 1.0)  # stretch away from bg
        for row in range(28):
            for col in range(28):
                v = float(img_g[row, col])
                if v < 0.04:       # skip near-black pixels for speed
                    continue
                vi = int(v * 255)
                color = (vi, vi, min(vi + 40, 255), 255)  # warm-white tinted blue
                x0, y0 = col * px, row * px
                dpg.draw_rectangle(
                    (x0, y0), (x0 + px, y0 + px),
                    color=color, fill=color,
                    parent="ds_drawlist", thickness=0,
                )

    def _update_dataset_display(self) -> None:
        """Must be called from the main (render) thread."""
        if self._dataset is None:
            if self._ds_error:
                dpg.set_value("txt_ds_page",
                              f"Error loading dataset: {self._ds_error}")
            return

        n = len(self._dataset)
        idx = self._ds_offset
        dpg.set_value("txt_ds_page", f"{idx + 1} / {n}")

        img_tensor, label = self._dataset[idx]
        img = img_tensor.squeeze().numpy()   # (28,28) float32 [0,1]

        self._draw_mnist_image(img)
        dpg.set_value("lbl_ds_0", f"Label: {label}")

        if self._ds_preds:
            p = self._ds_preds[0]
            if p == label:
                dpg.configure_item("lbl_ds_pred", color=(80, 220, 80))
                dpg.set_value("lbl_ds_pred", f"Predicted: {p}  [correct]")
            else:
                dpg.configure_item("lbl_ds_pred", color=(220, 80, 80))
                dpg.set_value("lbl_ds_pred", f"Predicted: {p}  [wrong]  (true: {label})")
        else:
            dpg.set_value("lbl_ds_pred", "")

    def _on_ds_prev(self) -> None:
        self._ds_preds = []
        if not self._ensure_dataset():
            return
        self._ds_offset = max(0, self._ds_offset - 1)
        self._ds_needs_refresh = True

    def _on_ds_next(self) -> None:
        self._ds_preds = []
        if not self._ensure_dataset():
            return
        n = len(self._dataset)
        self._ds_offset = min(self._ds_offset + 1, n - 1)
        self._ds_needs_refresh = True

    def _on_ds_predict(self) -> None:
        if not self._ensure_dataset():
            return
        if self._trainer is None or self._trainer.model is None:
            dpg.set_value("txt_ds_page", "Train the network first!")
            return
        img_tensor, _ = self._dataset[self._ds_offset]
        img = img_tensor.squeeze().numpy()
        pred, *_ = self._trainer.model.predict(img)
        self._ds_preds = [pred]
        self._ds_needs_refresh = True

    # ── Node / weight filter ─────────────────────────────────────────────────

    def _apply_filter(self, li: int, text: str, kind: str) -> None:
        """Parse comma-separated indices and show/hide series accordingly."""
        text = text.strip()
        if not text:
            self._clear_filter(li, kind)
            return
        try:
            indices = {int(x.strip()) for x in text.split(",") if x.strip()}
        except ValueError:
            return  # ignore bad input

        n = _NODE_SHOWN[li]
        if kind == "node":
            self._node_filter[li] = indices
            for ni in range(n):
                tag = f"node_series_{li}_{ni}"
                dpg.show_item(tag) if ni in indices else dpg.hide_item(tag)
        else:
            self._weight_filter[li] = indices
            for ni in range(n):
                tag = f"wseries_{li}_{ni}"
                dpg.show_item(tag) if ni in indices else dpg.hide_item(tag)

    def _clear_filter(self, li: int, kind: str) -> None:
        n = _NODE_SHOWN[li]
        if kind == "node":
            self._node_filter[li] = None
            dpg.set_value(f"inp_node_filter_{li}", "")
            for ni in range(n):
                dpg.show_item(f"node_series_{li}_{ni}")
        else:
            self._weight_filter[li] = None
            dpg.set_value(f"inp_weight_filter_{li}", "")
            for ni in range(n):
                dpg.show_item(f"wseries_{li}_{ni}")

    # ── Metrics / confusion matrix ────────────────────────────────────────────

    def _on_recompute(self) -> None:
        if self._trainer is None or self._trainer.model is None:
            dpg.set_value("txt_metrics_status", "Train first!")
            return
        if self._metrics_computing:
            return
        dpg.set_value("txt_metrics_status", "Computing...")
        threading.Thread(target=self._compute_metrics, daemon=True).start()

    def _compute_metrics(self) -> None:
        self._metrics_computing = True
        try:
            if not self._ensure_dataset() or self._dataset is None:
                return
            import numpy as _np
            model = self._trainer.model
            n     = len(self._dataset)
            y_true, y_pred = [], []

            for i in range(n):
                img_tensor, label = self._dataset[i]
                img  = img_tensor.squeeze().numpy()
                pred, *_ = model.predict(img)
                y_true.append(int(label))
                y_pred.append(pred)

            y_true = _np.array(y_true)
            y_pred = _np.array(y_pred)

            # confusion matrix (rows=actual, cols=predicted)
            cm = _np.zeros((10, 10), dtype=_np.int32)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1

            # per-class precision / recall / F1
            metrics = {}
            for cls in range(10):
                tp = int(cm[cls, cls])
                fp = int(cm[:, cls].sum()) - tp
                fn = int(cm[cls, :].sum()) - tp
                tn = int(cm.sum()) - tp - fp - fn
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = (2 * prec * rec / (prec + rec)
                        if (prec + rec) > 0 else 0.0)
                metrics[cls] = dict(precision=prec, recall=rec,
                                    f1=f1, support=int(cm[cls].sum()))

            self._cm           = cm
            self._class_metrics = metrics
            self._metrics_needs_refresh = True
        finally:
            self._metrics_computing = False

    def _update_metrics_display(self) -> None:
        if self._cm is None or self._class_metrics is None:
            return
        import numpy as _np

        # heatmap — row-normalise; flip so digit 0 is at bottom
        cm = self._cm.astype(_np.float32)
        cm_norm = cm / _np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = _np.nan_to_num(cm_norm, nan=0.0)
        # flip rows (DPG heat_series origin = bottom-left)
        cm_flat = [float(v) for v in _np.ascontiguousarray(cm_norm[::-1]).ravel()]
        dpg.set_value("series_cm", [cm_flat])
        dpg.configure_item("series_cm", scale_max=1.0)

        # per-class table
        precisions, recalls, f1s = [], [], []
        for cls in range(10):
            m = self._class_metrics[cls]
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])

            # colour F1: red < 0.9, yellow < 0.97, green >= 0.97
            f1v = m["f1"]
            fc  = ((80, 220, 80) if f1v >= 0.97
                   else (220, 200, 80) if f1v >= 0.90
                   else (220, 80, 80))
            dpg.set_value(f"mc_{cls}_prec", f"{m['precision']:.3f}")
            dpg.set_value(f"mc_{cls}_rec",  f"{m['recall']:.3f}")
            dpg.configure_item(f"mc_{cls}_f1", color=fc)
            dpg.set_value(f"mc_{cls}_f1",   f"{m['f1']:.3f}")
            dpg.set_value(f"mc_{cls}_sup",  str(m["support"]))

        # macro averages
        mp = float(_np.mean(precisions))
        mr = float(_np.mean(recalls))
        mf = float(_np.mean(f1s))
        total = sum(m["support"] for m in self._class_metrics.values())
        acc   = float(self._cm.trace()) / total * 100

        dpg.set_value("mc_macro_prec", f"{mp:.3f}")
        dpg.set_value("mc_macro_rec",  f"{mr:.3f}")
        dpg.set_value("mc_macro_f1",   f"{mf:.3f}")
        dpg.set_value("mc_macro_sup",  str(total))
        dpg.set_value("txt_overall_metrics",
            f"Accuracy: {acc:.2f}%   Macro F1: {mf:.4f}")
        dpg.set_value("txt_metrics_status", "")

    # ── Node metric callback ──────────────────────────────────────────────────

    def _on_metric_change(self, sender, app_data) -> None:
        self._node_metric = app_data
        self._refresh_node_series()

    def _refresh_node_series(self) -> None:
        step_idx = len(self._node_act_history[0][0])
        if step_idx == 0:
            return
        hist_map = {
            _METRICS[0]: self._node_act_history,
            _METRICS[1]: self._node_avg_history,
            _METRICS[2]: self._node_std_history,
        }
        src = hist_map.get(self._node_metric, self._node_act_history)
        for li in range(3):
            hist_len = len(src[li][0])
            xs = list(range(step_idx - hist_len, step_idx))
            for ni in range(_NODE_SHOWN[li]):
                dpg.set_value(f"node_series_{li}_{ni}", [xs, src[li][ni]])
            dpg.fit_axis_data(f"node_x_{li}")
            dpg.fit_axis_data(f"node_y_{li}")

    def _on_weight_metric_change(self, sender, app_data) -> None:
        self._weight_metric = app_data
        self._refresh_weight_series()

    def _refresh_weight_series(self) -> None:
        step_idx = len(self._node_weight_mean_history[0][0])
        if step_idx == 0:
            return
        w_hist_map = {
            _WEIGHT_METRICS[0]: self._node_weight_mean_history,
            _WEIGHT_METRICS[1]: self._node_weight_std_history,
        }
        src = w_hist_map.get(self._weight_metric, self._node_weight_mean_history)
        for li in range(3):
            hist_len = len(src[li][0])
            xs = list(range(step_idx - hist_len, step_idx))
            for ni in range(_NODE_SHOWN[li]):
                dpg.set_value(f"wseries_{li}_{ni}", [xs, src[li][ni]])
            dpg.fit_axis_data(f"wnode_x_{li}")
            dpg.fit_axis_data(f"wnode_y_{li}")

    # ── Draw-pad callbacks ────────────────────────────────────────────────────

    def _on_clear_draw(self) -> None:
        self._draw_grid[:] = 0
        self._draw_dirty   = True
        self._infer_result = None
        self._highlight_nodes = [set() for _ in range(4)]
        self._highlight_edges = [set() for _ in range(3)]
        self._inference_active = False
        dpg.set_value("txt_pred", "")
        dpg.set_value("series_conf", [list(range(10)), [0.0]*10])

    def _on_recognise(self) -> None:
        if self._trainer is None or self._trainer.model is None:
            dpg.set_value("txt_pred", "Train the network first!")
            return
        if self._draw_grid.max() < 0.05:
            dpg.set_value("txt_pred", "Draw a digit first!")
            return

        pred, probs, acts, weights = self._trainer.model.predict(self._draw_grid)
        self._infer_result     = (pred, probs, acts, weights)
        self._inference_active = True
        self._inference_drawn  = False

        # store input for Activations tab
        self._act_input_img = self._draw_grid.copy()
        self._update_activations_display(
            acts, probs,
            input_img=self._draw_grid,
            status=f"Inference: predicted {pred}  ({float(probs[pred])*100:.1f}%)",
        )

        # compute decision-path highlights
        self._compute_highlights(pred, acts, weights)

        conf_pct = [float(p * 100) for p in probs]
        dpg.set_value("txt_pred",
            f"Prediction:  {pred}   ({conf_pct[pred]:.1f}% confidence)")
        dpg.set_value("series_conf", [list(range(10)), conf_pct])
        dpg.fit_axis_data("conf_x")

    def _compute_highlights(self, pred: int, acts: list, weights: list) -> None:
        """Find the top-N most influential nodes/edges for the predicted class."""
        n_highlight = 4   # top nodes to highlight per layer

        self._highlight_nodes = [set() for _ in range(4)]
        self._highlight_edges = [set() for _ in range(3)]

        # output layer — highlight the predicted class node
        n_out     = _DRAW_SIZES[3]
        step_out  = max(1, 10 // n_out)
        out_idx   = min(pred // step_out, n_out - 1)
        self._highlight_nodes[3].add(out_idx)

        # backtrack through layers: for each highlighted dst node, find top src nodes
        layer_acts_draw = [None] + [a for a in acts]  # index 0 = input (None)

        for li in range(len(weights) - 1, -1, -1):
            dst_set = self._highlight_nodes[li + 1]
            n_src   = _DRAW_SIZES[li]
            n_dst   = _DRAW_SIZES[li + 1]
            W       = weights[li]
            step_src = max(1, W.shape[1] // n_src)
            step_dst = max(1, W.shape[0] // n_dst)

            src_scores = np.zeros(n_src)
            for di in dst_set:
                w_row = W[di * step_dst, :]
                for si in range(n_src):
                    w_val = abs(float(w_row[si * step_src]))
                    # weight importance × source activation (if available)
                    act_val = 1.0
                    if layer_acts_draw[li] is not None:
                        a = layer_acts_draw[li]
                        step_a = max(1, len(a) // n_src)
                        act_val = max(0.0, float(a[min(si * step_a, len(a) - 1)]))
                    src_scores[si] += w_val * act_val
                    self._highlight_edges[li].add((si, di))

            # keep only top-n_highlight source nodes
            top_src = set(np.argsort(src_scores)[-n_highlight:].tolist())
            self._highlight_nodes[li].update(top_src)
            # prune edges to only those connecting highlighted src → highlighted dst
            self._highlight_edges[li] = {
                (si, di) for si, di in self._highlight_edges[li]
                if si in top_src and di in dst_set
            }

    # ── Training step receiver ────────────────────────────────────────────────

    def _on_step(self, stats: TrainStats) -> None:
        with self._lock:
            self._stats = stats

    # ── Per-frame render ──────────────────────────────────────────────────────

    def _handle_drawing(self) -> None:
        """Check if mouse is dragging over the draw canvas and paint pixels."""
        if not dpg.is_item_hovered("draw_canvas"):
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            return

        mx, my = dpg.get_mouse_pos(local=False)
        cx, cy = dpg.get_item_rect_min("draw_canvas")
        px = int((mx - cx) / DRAW_PX)
        py = int((my - cy) / DRAW_PX)

        # paint with a soft 3×3 brush
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = px + dx, py + dy
                if 0 <= nx < DRAW_GRID and 0 <= ny < DRAW_GRID:
                    strength = 1.0 if (dx == 0 and dy == 0) else 0.5
                    self._draw_grid[ny, nx] = min(1.0,
                        self._draw_grid[ny, nx] + strength)
        self._draw_dirty = True
        # clear stale inference when user draws again
        if self._inference_active:
            self._inference_active = False
            self._inference_drawn  = False
            self._highlight_nodes  = [set() for _ in range(4)]
            self._highlight_edges  = [set() for _ in range(3)]

    def _render_draw_canvas(self) -> None:
        dpg.delete_item("draw_canvas", children_only=True)
        # background
        dpg.draw_rectangle((0, 0), (DRAW_CANVAS_SZ, DRAW_CANVAS_SZ),
                            fill=(20, 20, 30, 255), color=(40, 40, 60, 255),
                            parent="draw_canvas")
        # pixels
        for row in range(DRAW_GRID):
            for col in range(DRAW_GRID):
                v = self._draw_grid[row, col]
                if v < 0.02:
                    continue
                intensity = int(v * 255)
                x0, y0 = col * DRAW_PX, row * DRAW_PX
                dpg.draw_rectangle(
                    (x0, y0), (x0 + DRAW_PX, y0 + DRAW_PX),
                    fill=(intensity, intensity, min(255, intensity + 40), 255),
                    color=(0, 0, 0, 0),
                    parent="draw_canvas",
                )

    def _update_stats(self, stats: TrainStats) -> None:
        self._loss_history.append(stats.loss)
        self._batch_acc_history.append(stats.batch_accuracy)
        self._epoch_acc_history.append(stats.epoch_accuracy)
        self._val_acc_history.append(stats.val_accuracy)
        xs = list(range(len(self._loss_history)))

        dpg.set_value("txt_counters",
            f"Epoch: {stats.epoch}/{self._trainer._epochs}   "
            f"Batch: {stats.batch}/{stats.total_batches}   "
            f"Loss: {stats.loss:.4f}   "
            f"Batch acc: {stats.batch_accuracy:.1f}%")
        dpg.set_value("txt_val", f"Val accuracy: {stats.val_accuracy:.2f}%")

        dpg.set_value("series_loss",      [xs, self._loss_history])
        dpg.set_value("series_batch_acc", [xs, self._batch_acc_history])
        dpg.set_value("series_epoch_acc", [xs, self._epoch_acc_history])
        dpg.set_value("series_val_acc",   [xs, self._val_acc_history])
        dpg.fit_axis_data("loss_x"); dpg.fit_axis_data("loss_y")
        dpg.fit_axis_data("acc_x");  dpg.fit_axis_data("acc_y")

        # ── node activity history ─────────────────────────────────────────────
        import numpy as _np
        step_idx = len(self._loss_history)
        for li, acts in enumerate(stats.activations):
            n      = _NODE_SHOWN[li]
            stride = _NODE_STEPS[li]
            for ni in range(n):
                node_real_idx = min(ni * stride, len(acts) - 1)
                val = float(acts[node_real_idx])

                # raw value
                hist = self._node_act_history[li][ni]
                hist.append(val)
                if len(hist) > _HIST_LEN:
                    hist.pop(0)

                # rolling average
                window = hist[-_ROLLING_N:]
                self._node_avg_history[li][ni].append(float(_np.mean(window)))
                if len(self._node_avg_history[li][ni]) > _HIST_LEN:
                    self._node_avg_history[li][ni].pop(0)

                # rolling std
                self._node_std_history[li][ni].append(
                    float(_np.std(window)) if len(window) > 1 else 0.0)
                if len(self._node_std_history[li][ni]) > _HIST_LEN:
                    self._node_std_history[li][ni].pop(0)

        # weight histories (one value per node = mean/std of incoming weights)
        for li, W in enumerate(stats.weights):
            n      = _NODE_SHOWN[li]
            stride = _NODE_STEPS[li]
            for ni in range(n):
                row = W[min(ni * stride, W.shape[0] - 1), :]
                wm = float(_np.mean(row))
                ws = float(_np.std(row))
                wm_hist = self._node_weight_mean_history[li][ni]
                ws_hist = self._node_weight_std_history[li][ni]
                wm_hist.append(wm)
                ws_hist.append(ws)
                if len(wm_hist) > _HIST_LEN:
                    wm_hist.pop(0)
                if len(ws_hist) > _HIST_LEN:
                    ws_hist.pop(0)

        # push activation metric to Node Activity tab
        act_hist_map = {
            _METRICS[0]: self._node_act_history,
            _METRICS[1]: self._node_avg_history,
            _METRICS[2]: self._node_std_history,
        }
        src = act_hist_map.get(self._node_metric, self._node_act_history)
        for li in range(len(stats.activations)):
            hist_len = len(src[li][0])
            xs_node  = list(range(step_idx - hist_len, step_idx))
            for ni in range(_NODE_SHOWN[li]):
                dpg.set_value(f"node_series_{li}_{ni}", [xs_node, src[li][ni]])
            dpg.fit_axis_data(f"node_x_{li}")
            dpg.fit_axis_data(f"node_y_{li}")

        # push weight metric to Weights tab
        w_hist_map = {
            _WEIGHT_METRICS[0]: self._node_weight_mean_history,
            _WEIGHT_METRICS[1]: self._node_weight_std_history,
        }
        wsrc = w_hist_map.get(self._weight_metric, self._node_weight_mean_history)
        for li in range(len(stats.weights)):
            hist_len = len(wsrc[li][0])
            xs_w     = list(range(step_idx - hist_len, step_idx))
            for ni in range(_NODE_SHOWN[li]):
                dpg.set_value(f"wseries_{li}_{ni}", [xs_w, wsrc[li][ni]])
            dpg.fit_axis_data(f"wnode_x_{li}")
            dpg.fit_axis_data(f"wnode_y_{li}")

        for i, ls in enumerate(stats.layer_stats):
            dead_col = (255, 90, 90) if ls.dead_neurons_pct > 20 else (170, 210, 170)
            dpg.set_value(f"c{i}_am",   f"{ls.activations_mean:.4f}")
            dpg.set_value(f"c{i}_as",   f"{ls.activations_std:.4f}")
            dpg.configure_item(f"c{i}_dead", color=dead_col)
            dpg.set_value(f"c{i}_dead", f"{ls.dead_neurons_pct:.1f}%")
            dpg.set_value(f"c{i}_wm",   f"{ls.weights_mean:.4f}")
            dpg.set_value(f"c{i}_gm",
                f"{ls.grad_mean:.5f}" if ls.grad_mean else "—")

        step_now = len(self._loss_history)
        if step_now - self._last_graph_step >= self._graph_redraw_rate:
            self._last_graph_step = step_now
            self._draw_network(stats.activations, stats.weights)
            # update activations tab every N steps during training
            import numpy as _np
            dummy_probs = _np.zeros(10, dtype=_np.float32)
            if stats.activations and len(stats.activations) >= 3:
                dummy_probs = _np.array(stats.activations[2], dtype=_np.float32)
                sm = _np.exp(dummy_probs - dummy_probs.max())
                dummy_probs = sm / sm.sum()
            self._update_activations_display(
                stats.activations, dummy_probs,
                status=f"Training  epoch {stats.epoch}  batch {stats.batch}",
            )

        # auto-recompute metrics at each epoch end
        if (stats.epoch != self._last_val_epoch
                and stats.batch == stats.total_batches
                and not self._metrics_computing
                and self._trainer.model is not None):
            self._last_val_epoch = stats.epoch
            dpg.set_value("txt_metrics_status", "Computing...")
            threading.Thread(target=self._compute_metrics, daemon=True).start()

        # check if training finished
        if not self._trainer.is_running:
            self._on_stop()
            dpg.set_value("txt_status", "Done!")

    def _draw_network(
        self,
        activations: list,
        weights: list,
        highlight_nodes: Optional[list] = None,
        highlight_edges: Optional[list] = None,
    ) -> None:
        dpg.delete_item("graph_canvas", children_only=True)

        canvas_w = self._graph_w - 12
        canvas_h = self._content_h - 36
        y0 = 5

        positions = _compute_node_positions(_DRAW_SIZES, canvas_w, y0, canvas_h - y0 - 25)

        hl_nodes = highlight_nodes or self._highlight_nodes
        hl_edges = highlight_edges or self._highlight_edges

        # edges
        if weights:
            for li in range(len(positions) - 1):
                src_pos = positions[li]
                dst_pos = positions[li + 1]
                W = weights[li] if li < len(weights) else None
                if W is None:
                    continue
                n_src = len(src_pos)
                n_dst = len(dst_pos)
                step_src = max(1, W.shape[1] // n_src)
                step_dst = max(1, W.shape[0] // n_dst)

                for di, dp_ in enumerate(dst_pos):
                    w_row = W[di * step_dst, :]
                    for si, sp in enumerate(src_pos):
                        is_hl = (si, di) in hl_edges[li]
                        w_val = float(w_row[si * step_src])
                        col   = _weight_color(w_val, highlight=is_hl)
                        thick = 3.0 if is_hl else _clamp(abs(w_val) * 2, 0.3, 2.0)
                        dpg.draw_line(
                            (sp.x, sp.y), (dp_.x, dp_.y),
                            color=col, thickness=thick,
                            parent="graph_canvas",
                        )

        # nodes
        layer_acts = [None] + list(activations)
        for li, (layer_pos, acts) in enumerate(zip(positions, layer_acts)):
            n_full = _DRAW_SIZES[li]
            for ni, pos in enumerate(layer_pos):
                is_hl = ni in hl_nodes[li]
                if acts is not None and len(acts) > 0:
                    step = max(1, len(acts) // n_full)
                    raw  = float(acts[min(ni * step, len(acts) - 1)])
                    norm = (math.tanh(raw) + 1) / 2
                    col  = _activation_color(norm, highlight=is_hl)
                else:
                    col = (255, 220, 50, 255) if is_hl else (80, 90, 120, 200)

                outline = (255, 255, 100, 255) if is_hl else (200, 210, 255, 80)
                radius  = NODE_R + 2 if is_hl else NODE_R
                dpg.draw_circle(
                    center=(pos.x, pos.y), radius=radius,
                    color=outline, fill=col,
                    parent="graph_canvas",
                )

        # layer labels
        n_layers = len(positions)
        x_step   = (canvas_w) / (n_layers + 1)
        for li, label in enumerate(_LAYER_NAMES):
            x = x_step * (li + 1)
            dpg.draw_text(
                (x - 30, canvas_h - 18), label,
                color=(140, 150, 200), size=11,
                parent="graph_canvas",
            )

    # ── Main loop ─────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_dark_titlebar(title: str) -> None:
        """Enable dark mode on the OS title bar (Windows 10 build 18985+ / 11)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, title)
            if not hwnd:
                return
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            pass  # not critical — older Windows versions may not support it

    def run(self) -> None:
        self._build()
        dpg.show_viewport()
        self._apply_dark_titlebar("MNIST Neural Network Visualizer")

        # draw empty canvas + empty network on startup
        self._render_draw_canvas()
        self._draw_network([], [])

        # load dataset in background so Dataset tab is ready immediately
        threading.Thread(target=self._ensure_dataset, daemon=True).start()

        while dpg.is_dearpygui_running():
            # metrics refresh (must happen on main thread)
            if self._metrics_needs_refresh:
                self._metrics_needs_refresh = False
                self._update_metrics_display()

            # dataset texture refresh (must happen on main thread)
            if self._ds_needs_refresh:
                self._ds_needs_refresh = False
                self._update_dataset_display()

            # handle viewport resize
            vp_w = dpg.get_viewport_width()
            vp_h = dpg.get_viewport_height()
            if vp_w != self._vp_w or vp_h != self._vp_h:
                self._resize(vp_w, vp_h)

            # handle mouse drawing
            self._handle_drawing()
            if self._draw_dirty:
                self._render_draw_canvas()
                self._draw_dirty = False

            # consume latest training stats
            with self._lock:
                stats = self._stats
                self._stats = None

            if stats is not None:
                self._update_stats(stats)
            elif self._inference_active and not self._inference_drawn:
                if self._infer_result:
                    _, _, acts, weights = self._infer_result
                    self._draw_network(acts, weights)
                    self._inference_drawn = True

            dpg.render_dearpygui_frame()

        if self._trainer:
            self._trainer.stop()
        dpg.destroy_context()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
