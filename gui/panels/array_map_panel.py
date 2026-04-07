"""
array_map_panel.py — Interactive electrode array spatial map with clickable channels.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel


class ArrayMapPanel(QWidget):
    """Spatial map of electrode positions. Click → select channel."""

    channel_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ei_positions: Optional[np.ndarray] = None
        self._channel_colors: dict = {}
        self._selected_ch: Optional[int] = None
        self._scatter: Optional[pg.ScatterPlotItem] = None
        self._color_mode: str = "flat"  # flat | fr | qc
        self._fr_values: dict = {}
        self._qc_colors: dict = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Coloring:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["Flat", "Firing Rate", "QC miss-rate"])
        self._color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        toolbar.addWidget(self._color_combo)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Plot
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setAspectLocked(True)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        layout.addWidget(self._plot)

        self._scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen("#5A5C65", width=0.5), brush=pg.mkBrush("#888888")
        )
        self._scatter.sigClicked.connect(self._on_click)
        self._plot.addItem(self._scatter)

    def set_array(self, ei_positions: np.ndarray):
        """Load electrode positions and render all channels."""
        self._ei_positions = np.asarray(ei_positions, dtype=np.float64)
        n = self._ei_positions.shape[0]
        self._channel_colors = {}
        self._qc_colors.clear()
        self._selected_ch = None

        xs = self._ei_positions[:, 0]
        ys = self._ei_positions[:, 1]
        self._scatter.setData(
            x=xs,
            y=ys,
            size=8,
            pen=pg.mkPen("#5A5C65", width=0.5),
            brush=pg.mkBrush("#888888"),
            data=np.arange(n),
        )
        self._plot.autoRange()

    def set_channel_colors(self, color_map: dict[int, tuple]):
        """Update all dot colors. color_map: {ch: (r,g,b)} 0-255."""
        self._channel_colors = color_map
        self._apply_colors()

    def set_selected_channel(self, ch: int):
        """Highlight the selected channel with a larger ring."""
        self._selected_ch = ch
        self._apply_colors()

    def set_qc_result_color(self, ch: int, miss_rate: Optional[float]):
        """
        Color a single channel after QC: miss_rate 0→green ... 1→red, None→gray.
        """
        if miss_rate is None:
            self._qc_colors[ch] = (136, 136, 136)
        else:
            r = int(255 * miss_rate)
            g = int(255 * (1.0 - miss_rate))
            self._qc_colors[ch] = (r, g, 40)
        self._apply_colors()

    def set_firing_rates(self, fr_values: dict[int, float]):
        """Set firing rates for FR coloring mode."""
        self._fr_values = fr_values

    def clear(self):
        """Reset the array display."""
        self._ei_positions = None
        self._channel_colors.clear()
        self._qc_colors.clear()
        self._fr_values.clear()
        self._selected_ch = None
        self._scatter.clear()

    # ── internal ───────────────────────────────────────────────────

    def _on_color_mode_changed(self, mode: str):
        if "Flat" in mode:
            self._color_mode = "flat"
        elif "Firing" in mode:
            self._color_mode = "fr"
        elif "QC" in mode:
            self._color_mode = "qc"
        self._apply_colors()

    def _apply_colors(self):
        if self._ei_positions is None or self._scatter is None:
            return

        n = self._ei_positions.shape[0]
        brushes = []
        pens = []
        sizes = []

        for i in range(n):
            r, g, b = 136, 136, 136  # default gray

            if self._color_mode == "qc" and i in self._qc_colors:
                r, g, b = self._qc_colors[i]
            elif self._color_mode == "fr" and i in self._fr_values:
                fr = self._fr_values[i]
                # blue (0) → yellow (high)
                t = min(1.0, fr / 50.0)
                r, g, b = int(30 + 225 * t), int(60 + 195 * t), int(200 * (1 - t))
            elif i in self._channel_colors:
                r, g, b = self._channel_colors[i]

            pen_color = "#2E6DD4" if i == self._selected_ch else "#5A5C65"
            size = 12 if i == self._selected_ch else 8

            brushes.append(pg.mkBrush(r, g, b))
            pens.append(pg.mkPen(pen_color, width=2 if i == self._selected_ch else 0.5))
            sizes.append(size)

        self._scatter.setBrush(brushes)
        self._scatter.setPen(pens)
        self._scatter.setSize(sizes)

    def _on_click(self, scatter, points):
        """Handle pyqtgraph scatter click."""
        if not points:
            return
        pt = list(points)[0]
        idx = pt.data()
        if idx is not None:
            ch = int(idx)
            self.channel_selected.emit(ch)
