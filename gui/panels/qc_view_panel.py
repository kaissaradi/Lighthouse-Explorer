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
        self._plot_waveforms = pg.PlotWidget()

        self._plot_hist.showGrid(x=True, y=True, alpha=0.15)
        self._plot_pca.showGrid(x=True, y=True, alpha=0.15)
        self._plot_waveforms.showGrid(x=True, y=True, alpha=0.15)

        # Bottom-left: stacked KS fragmentation bar + FR overlay
        self._fr_layout = pg.GraphicsLayoutWidget()
        self._plot_fr   = self._fr_layout.addPlot(row=0, col=0)   # KS fragmentation
        self._plot_fr_time = self._fr_layout.addPlot(row=1, col=0) # FR over time
        self._fr_layout.ci.layout.setRowStretchFactor(0, 3)
        self._fr_layout.ci.layout.setRowStretchFactor(1, 2)
        self._plot_fr.showGrid(x=True, y=True, alpha=0.15)
        self._plot_fr_time.showGrid(x=True, y=True, alpha=0.15)

        grid.addWidget(self._plot_hist, 0, 0)
        grid.addWidget(self._plot_pca, 0, 1)
        grid.addWidget(self._fr_layout, 1, 0)
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
        
        # Intercept rejected channels
        if result.reject_reason:
            self._placeholder.setText(f"Channel {result.channel} Rejected: {result.reject_reason}")
            self._placeholder.setStyleSheet("color: #F08080; font-size: 16px; font-weight: bold;")
            self._placeholder.show()
            self._update_summary_bar(result)
            self._update_amp_histogram(result)
            self._update_pca_scatter(result)
            self._update_fr_plot(result)
            self._update_fr_time_plot(result)
            self._update_waveforms(result)
            self._update_summary_bar(result)
            return
            
        # Standard display for valid channels
        self._placeholder.hide()
        self._update_summary_bar(result)
        self._update_amp_histogram(result)
        self._update_pca_scatter(result)
        self._update_fr_plot(result)
        self._update_fr_time_plot(result)
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
            self._plot_fr, self._plot_fr_time,
            self._plot_waveforms,
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
        
        # --- NEW CODE: Clamp the visual zoom ---
        # Find the absolute minimum edge, but don't let the view zoom out further than -800
        view_min = max(np.min(edges), -800)
        p.setXRange(view_min, 0, padding=0)

        p.setYRange(0, 2000, padding=0)
        
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

    def _update_fr_plot(self, result):
        """Venn Diagram replacing KS fragmentation bar chart."""
        p = self._plot_fr
        p.clear()

        fs = getattr(result, "fs", 20_000)
        coincidence_samp = int(0.001 * fs)  # ±1 ms window

        # 1. Gather all LH spikes
        lh_times = np.array([], dtype=np.int64)
        if hasattr(result, "valley"):
            parts = [t for t in [result.valley.left_times, result.valley.rightk_times] if t.size]
            if parts:
                lh_times = np.sort(np.concatenate(parts))

        # 2. Gather all KS spikes for this channel
        sorter_unit_map = getattr(result, "sorter_unit_map", {})
        if sorter_unit_map:
            all_t = np.concatenate(list(sorter_unit_map.values()))
            ks_times = np.sort(all_t)
        else:
            ks_times = np.array([], dtype=np.int64)

        if not sorter_unit_map:
            p.setTitle(f"LH vs KS Venn Diagram — CH {result.channel} (no sorter)")
            return
        if lh_times.size == 0:
            p.setTitle(f"LH vs KS Venn Diagram — CH {result.channel} (no LH spikes)")
            return

        # 3. Fast two-pointer matching
        matched = 0
        i, j = 0, 0
        while i < len(lh_times) and j < len(ks_times):
            diff = lh_times[i] - ks_times[j]
            if abs(diff) <= coincidence_samp:
                matched += 1
                i += 1
                j += 1
            elif diff < 0:
                i += 1  # LH spike earlier
            else:
                j += 1  # KS spike earlier

        lh_only = len(lh_times) - matched
        ks_only = len(ks_times) - matched

        # 4. Draw the Venn Diagram 
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1.0
        
        # Circle boundaries
        x_lh = -0.5 + r * np.cos(theta)
        y_lh = r * np.sin(theta)
        
        x_ks = 0.5 + r * np.cos(theta)
        y_ks = r * np.sin(theta)
        
        # Add outlines
        p.addItem(pg.PlotCurveItem(x_lh, y_lh, pen=pg.mkPen("#4CAF50", width=2))) # Green LH
        p.addItem(pg.PlotCurveItem(x_ks, y_ks, pen=pg.mkPen("#2196F3", width=2))) # Blue KS
        
        # Add Text Labels
        text_lh = pg.TextItem(f"LH Only\n{lh_only}", color="#4CAF50", anchor=(0.5, 0.5))
        text_lh.setPos(-0.9, 0)
        p.addItem(text_lh)
        
        text_both = pg.TextItem(f"Matched\n{matched}", color="#FFFFFF", anchor=(0.5, 0.5))
        text_both.setPos(0, 0)
        p.addItem(text_both)
        
        text_ks = pg.TextItem(f"KS Only\n{ks_only}", color="#2196F3", anchor=(0.5, 0.5))
        text_ks.setPos(0.9, 0)
        p.addItem(text_ks)

        # 5. Clean up plot aesthetics
        p.hideAxis('bottom')
        p.hideAxis('left')
        p.setAspectLocked(True) # Keep circles round
        p.setXRange(-2, 2, padding=0)
        p.setYRange(-1.5, 1.5, padding=0)
        
        win_ms = coincidence_samp * 1000 / fs
        p.setTitle(f"LH vs KS Venn Diagram — CH {result.channel} (±{win_ms:.0f} ms)")

    def _update_fr_time_plot(self, result):
        """FR over time: LH (green) vs KS sorter (blue) — bottom sub-panel."""
        p = self._plot_fr_time  # PlotItem
        p.clear()
        p.addLegend(offset=(10, 10), labelTextSize="8pt")

        fs = getattr(result, "fs", 20_000)
        bin_s = 1.0
        lh_times = np.array([], dtype=np.int64)
        if hasattr(result, "valley"):
            parts = [t for t in [result.valley.left_times, result.valley.rightk_times] if t.size]
            if parts:
                lh_times = np.sort(np.concatenate(parts))

        sorter_times = getattr(result, "sorter_times", None)

        all_times = lh_times.copy()
        if sorter_times is not None and sorter_times.size:
            all_times = np.concatenate([all_times, sorter_times])
        if all_times.size == 0:
            p.setTitle("FR over time (no spikes)")
            return

        n_bins = max(1, int(all_times.max() / fs / bin_s) + 1)
        bins = np.arange(n_bins + 1, dtype=np.float64) * bin_s

        if lh_times.size:
            lh_counts, _ = np.histogram(lh_times / fs, bins=bins)
            p.plot(bins[:-1], lh_counts.astype(np.float64),
                   pen=pg.mkPen("#4CAF50", width=1.2), name="LH")
        if sorter_times is not None and sorter_times.size:
            s_counts, _ = np.histogram(sorter_times / fs, bins=bins)
            p.plot(bins[:-1], s_counts.astype(np.float64),
                   pen=pg.mkPen("#2196F3", width=1.2), name="KS")

        has_ks = sorter_times is not None and sorter_times.size > 0
        p.setTitle(f"FR over time — CH {result.channel}" + ("" if has_ks else " (no KS)"))
        p.setLabel("bottom", "Time (s)")
        p.setLabel("left", "Spikes/s")

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