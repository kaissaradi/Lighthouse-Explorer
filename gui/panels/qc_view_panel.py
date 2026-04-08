"""
qc_view_panel.py — Right side: 4 pyqtgraph plots (2×2) + summary stats bar.
"""
from __future__ import annotations
from typing import Optional
import pyqtgraph as pg
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from qtpy.QtCore import Qt
from core.result_types import QCResult

# Colors
COLORS = {
    "LH": "#4CAF50",
    "soup": "#FF9800",
    "uncertain_boundary": "#9E9E9E",
    "uncertain_lowBL": "#757575",
    "cluster0": "#2196F3",
    "cluster1": "#FF9800",
}


class QCViewPanel(QWidget):
    """QC visualization panel with 4 plots and summary bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_result: Optional[QCResult] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Summary bar ──────────────────────────────────────────
        self._summary_bar = QHBoxLayout()
        self._summary_labels: list[QLabel] = []
        for _ in range(8):
            lbl = QLabel("—")
            lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #F0F0F2;")
            self._summary_labels.append(lbl)
            self._summary_bar.addWidget(lbl)
        self._summary_bar.addStretch()
        layout.addLayout(self._summary_bar)

        # ── 2×2 Plot Grid ────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(4)

        self._plot_hist = pg.PlotWidget()
        self._plot_pca = pg.PlotWidget()
        self._plot_bltr = pg.PlotWidget()
        self._plot_waveforms = pg.PlotWidget()

        self._plot_hist.showGrid(x=True, y=True, alpha=0.15)
        self._plot_pca.showGrid(x=True, y=True, alpha=0.15)
        self._plot_bltr.showGrid(x=True, y=True, alpha=0.15)
        self._plot_waveforms.showGrid(x=True, y=True, alpha=0.15)

        grid.addWidget(self._plot_hist, 0, 0)
        grid.addWidget(self._plot_pca, 0, 1)
        grid.addWidget(self._plot_bltr, 1, 0)
        grid.addWidget(self._plot_waveforms, 1, 1)

        # Set stretch — give equal weight
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        layout.addLayout(grid)

        # ── Placeholder label ─────────────────────────────────────
        self._placeholder = QLabel(
            "Select a channel and run QC to see results here."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #5A5C65; font-size: 14px;"
        )
        layout.addWidget(self._placeholder)

    def show_result(self, result: QCResult):
        """Main entry point. Update all 4 plots and summary bar."""
        self._current_result = result
        self._placeholder.hide()
        self._update_summary_bar(result)
        self._update_amp_histogram(result)
        self._update_pca_scatter(result)
        self._update_bltr_scatter(result)
        self._update_waveforms(result)

    def show_loading(self, channel: int):
        """Show 'Running QC on CH X...' placeholder."""
        self._placeholder.setText(f"Running QC on CH {channel}…")
        self._placeholder.show()
        self._clear_plots()

    def show_error(self, msg: str):
        """Show error message."""
        self._placeholder.setText(f"Error: {msg}")
        self._placeholder.setStyleSheet("color: #F08080; font-size: 14px;")
        self._placeholder.show()
        self._clear_plots()

    def clear(self):
        """Reset all plots to empty state."""
        self._current_result = None
        self._placeholder.setText(
            "Select a channel and run QC to see results here."
        )
        self._placeholder.setStyleSheet("color: #5A5C65; font-size: 14px;")
        self._placeholder.show()
        self._clear_plots()
        for lbl in self._summary_labels:
            lbl.setText("—")

    # ── plot updaters ────────────────────────────────────────────

    def _clear_plots(self):
        for p in [
            self._plot_hist, self._plot_pca,
            self._plot_bltr, self._plot_waveforms,
        ]:
            p.clear()

    def _update_summary_bar(self, result: QCResult):
        fields = [
            f"CH: {result.channel}",
            f"Total: {result.n_total}",
            f"LH: {result.n_lh} ({result.n_lh/max(1,result.n_total)*100:.0f}%)",
            f"Soup: {result.n_soup} ({result.n_soup/max(1,result.n_total)*100:.0f}%)",
            f"Uncertain: {result.n_uncertain}",
            f"Sorter: {result.n_sorter_spikes}",
            f"Miss: {result.miss_rate:.1%}" if result.miss_rate is not None else "Miss: N/A",
            f"Valley {'OK' if result.valley.accepted else 'N/A'}",
        ]
        for i, lbl in enumerate(self._summary_labels):
            lbl.setText(fields[i] if i < len(fields) else "—")

    def _update_amp_histogram(self, result: QCResult):
        p = self._plot_hist
        p.clear()

        vals = result.valley.all_vals
        if vals.size == 0:
            p.setTitle("No crossings")
            return

        counts, edges = result.valley.amp_hist_counts, result.valley.amp_hist_edges

        # pyqtgraph with stepMode=True requires len(x) == len(y) + 1
        # Pass the full edges array (length N+1), not the centers
        p.plot(
            edges,
            counts,
            stepMode=True,
            fillLevel=0,
            brush="#2E6DD4",
            pen=pg.mkPen("#4A8BEF"),
        )

        # Valley threshold line
        if result.valley.valley_low is not None:
            vline = pg.InfiniteLine(
                pos=result.valley.valley_low,
                angle=90,
                pen=pg.mkPen("#FF9800", width=1.5, style=Qt.DashLine),
            )
            p.addItem(vline)

        p.setTitle(f"Amplitude Histogram (CH {result.channel})")
        p.setLabel("bottom", "ADC amplitude")
        p.setLabel("left", "Count")

    def _update_pca_scatter(self, result: QCResult):
        p = self._plot_pca
        p.clear()

        coords = result.pca_km.pca_coords
        labels = result.pca_km.km_labels
        if coords.shape[0] == 0:
            p.setTitle("No spikes")
            return

        evr = result.pca_km.explained_variance_ratio

        for cluster_id, color in [(0, COLORS["cluster0"]), (1, COLORS["cluster1"])]:
            mask = labels == cluster_id
            if mask.any():
                x = coords[mask, 0]
                y = coords[mask, 1]
                p.plot(
                    x, y,
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=None,
                )

        p.setTitle(f"PCA: PC1 vs PC2 (CH {result.channel})")
        p.setLabel(
            "bottom",
            f"PC1 ({evr[0]*100:.0f}%)" if evr.size > 0 else "PC1",
        )
        p.setLabel(
            "left",
            f"PC2 ({evr[1]*100:.0f}%)" if evr.size > 1 else "PC2",
        )

    def _update_bltr_scatter(self, result: QCResult):
        p = self._plot_bltr
        p.clear()

        bl = result.bltr.bl_bulk
        tr = result.bltr.tr_bulk
        labels = result.bltr.labels
        if labels.size == 0:
            p.setTitle("No spikes")
            return

        for label_name, color in [
            ("LH", COLORS["LH"]),
            ("soup", COLORS["soup"]),
            ("uncertain_boundary", COLORS["uncertain_boundary"]),
            ("uncertain_lowBL", COLORS["uncertain_lowBL"]),
        ]:
            mask = labels == label_name
            if mask.any():
                p.plot(
                    bl[mask], tr[mask],
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=None,
                )

        # Reference lines
        min_bl = float(result.bltr.counts.get("min_bl_bulk", 0.70))
        p.addItem(
            pg.InfiniteLine(pos=min_bl, angle=90, pen=pg.mkPen("#5A5C65", style=Qt.DashLine))
        )
        p.addItem(
            pg.InfiniteLine(pos=min_bl, angle=0, pen=pg.mkPen("#5A5C65", style=Qt.DashLine))
        )
        # Identity line
        mx = max(bl.max() if bl.size else 1, tr.max() if tr.size else 1)
        p.plot([0, mx], [0, mx], pen=pg.mkPen("#3D3F48", style=Qt.DashLine, width=1))

        p.setTitle(f"BL/TR Scatter (CH {result.channel})")
        p.setLabel("bottom", "BL_bulk")
        p.setLabel("left", "TR_bulk")

    def _update_waveforms(self, result: QCResult):
        p = self._plot_waveforms
        p.clear()

        means = result.pca_km.cluster_mean_waveforms
        if len(means) < 2:
            p.setTitle("No waveforms")
            return

        L = means[0].shape[0]
        x = np.arange(L, dtype=np.float64)

        for k, (color, label) in enumerate([
            (COLORS["cluster0"], "Cluster 0"),
            (COLORS["cluster1"], "Cluster 1"),
        ]):
            p.plot(
                x, means[k],
                pen=pg.mkPen(color, width=1.5),
                name=label,
            )

        p.setTitle(f"Mean Waveforms on CH {result.channel}")
        p.setLabel("bottom", "Samples")
        p.setLabel("left", "ADC amplitude")
        p.addLegend(offset=(10, 10))
