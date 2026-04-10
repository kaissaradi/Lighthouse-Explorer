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
        self._plot_fr = pg.PlotWidget()
        self._plot_waveforms = pg.PlotWidget()

        self._plot_hist.showGrid(x=True, y=True, alpha=0.15)
        self._plot_pca.showGrid(x=True, y=True, alpha=0.15)
        self._plot_fr.showGrid(x=True, y=True, alpha=0.15)
        self._plot_waveforms.showGrid(x=True, y=True, alpha=0.15)

        grid.addWidget(self._plot_hist, 0, 0)
        grid.addWidget(self._plot_pca, 0, 1)
        grid.addWidget(self._plot_fr, 1, 0)
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
            self._update_waveforms(result)
            self._update_summary_bar(result) 
            return
            
        # Standard display for valid channels
        self._placeholder.hide()
        self._update_summary_bar(result)
        self._update_amp_histogram(result)
        self._update_pca_scatter(result)
        self._update_fr_plot(result)
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
            self._plot_fr, self._plot_waveforms,
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
        """KS fragmentation plot: for each LH spike, which KS unit claimed it?

        Algorithm
        ---------
        1. Gather all LH spike times for this channel (left + rightk valley times).
        2. Pool all KS spikes across units into a single sorted array, keeping a
           parallel unit-ID array.
        3. For each LH spike, binary-search for the nearest KS spike; if it falls
           within the coincidence window (default ±1 ms) record its unit ID.
        4. Plot a bar chart: x = KS unit ID (sorted by spike count desc), y = matched
           spike count.  Append a "Missed" bar in red for unmatched LH spikes.
        """
        p = self._plot_fr
        p.clear()

        fs = getattr(result, 'fs', 20_000)
        coincidence_samp = int(0.001 * fs)  # ±1 ms in samples

        # ── collect LH spike times ────────────────────────────────────────────
        lh_times = np.array([], dtype=np.int64)
        if hasattr(result, 'valley'):
            parts = []
            if result.valley.left_times.size:
                parts.append(result.valley.left_times)
            if result.valley.rightk_times.size:
                parts.append(result.valley.rightk_times)
            if parts:
                lh_times = np.sort(np.concatenate(parts))

        sorter_unit_map: dict = getattr(result, 'sorter_unit_map', {})

        # ── no sorter data: show informative placeholder ──────────────────────
        if not sorter_unit_map:
            p.setTitle(f"KS Fragmentation — CH {result.channel} (no sorter loaded)")
            p.setLabel("bottom", "KS unit / status")
            p.setLabel("left", "LH spikes matched")
            return

        if lh_times.size == 0:
            p.setTitle(f"KS Fragmentation — CH {result.channel} (no LH spikes)")
            return

        # ── pool KS spikes with unit labels ──────────────────────────────────
        all_ks_times_list, all_ks_units_list = [], []
        for uid, t in sorter_unit_map.items():
            all_ks_times_list.append(t)
            all_ks_units_list.append(np.full(len(t), uid, dtype=np.int64))

        ks_times = np.concatenate(all_ks_times_list)  # pooled, unsorted
        ks_units = np.concatenate(all_ks_units_list)

        order = np.argsort(ks_times)
        ks_times = ks_times[order]
        ks_units = ks_units[order]

        # ── match each LH spike to nearest KS spike within window ────────────
        match_counts: dict[int, int] = {}
        n_missed = 0

        insert_idx = np.searchsorted(ks_times, lh_times)  # vectorised anchor

        for i, (lh_t, idx) in enumerate(zip(lh_times, insert_idx)):
            best_uid = None
            best_dist = coincidence_samp + 1  # start outside window

            # check left neighbour
            if idx > 0:
                d = abs(int(lh_t) - int(ks_times[idx - 1]))
                if d < best_dist:
                    best_dist = d
                    best_uid = int(ks_units[idx - 1])

            # check right neighbour
            if idx < len(ks_times):
                d = abs(int(lh_t) - int(ks_times[idx]))
                if d < best_dist:
                    best_dist = d
                    best_uid = int(ks_units[idx])

            if best_uid is not None and best_dist <= coincidence_samp:
                match_counts[best_uid] = match_counts.get(best_uid, 0) + 1
            else:
                n_missed += 1

        # ── build bar chart data ──────────────────────────────────────────────
        # Sort matched units by count descending so the dominant unit is leftmost
        sorted_units = sorted(match_counts.items(), key=lambda kv: -kv[1])

        bar_labels: list[str] = [f"u{uid}" for uid, _ in sorted_units]
        bar_counts: list[int] = [cnt for _, cnt in sorted_units]

        if n_missed > 0:
            bar_labels.append("Missed")
            bar_counts.append(n_missed)

        n_bars = len(bar_labels)
        x = np.arange(n_bars, dtype=np.float64)

        # Draw bars one by one so we can color "Missed" red
        for i, (label, count) in enumerate(zip(bar_labels, bar_counts)):
            color = "#F44336" if label == "Missed" else "#4CAF50"
            bar = pg.BarGraphItem(
                x=[x[i]], height=[count], width=0.7,
                brush=pg.mkBrush(color),
                pen=pg.mkPen("#1a1a1a", width=0.5),
            )
            p.addItem(bar)

        # ── x-axis tick labels ────────────────────────────────────────────────
        ax = p.getAxis("bottom")
        ax.setTicks([[(xi, lbl) for xi, lbl in zip(x, bar_labels)]])

        # ── title and axis labels ─────────────────────────────────────────────
        win_ms = coincidence_samp * 1000 / fs
        n_units_matched = len(match_counts)
        p.setTitle(
            f"KS Fragmentation — CH {result.channel}  "
            f"({n_units_matched} unit{'s' if n_units_matched != 1 else ''} matched, "
            f"±{win_ms:.0f} ms window)"
        )
        p.setLabel("bottom", "KS unit ID")
        p.setLabel("left", "LH spikes matched")

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