"""
qc_summary_dialog.py — Recording-level QC summary across all channels.

Shows a 2×2 pyqtgraph panel:
  [0,0] Miss rate distribution (histogram across LH channels)
  [0,1] KS fragmentation index (bar: % channels with 1 / 2 / 3+ / missed-only units)
  [1,0] LH spike count vs KS spike count scatter (one dot per channel)
  [1,1] Array map coloured by miss rate (or fragmentation if no sorter)

Open via:  QCSummaryDialog(qc_results, sorter_spike_units, fs, parent).exec_()
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from qtpy.QtCore import Qt


def _fragmentation_index(sorter_unit_map: dict, lh_times: np.ndarray, fs: float) -> dict:
    """
    Given {unit_id: times} for one channel and its LH spike times,
    return a dict with keys: n_units_matched, n_missed, pct_missed, dominant_frac.
    Returns None if no sorter data.
    """
    if not sorter_unit_map or lh_times.size == 0:
        return None

    coincidence_samp = int(0.001 * fs)
    all_t = np.concatenate(list(sorter_unit_map.values()))
    all_u = np.concatenate([
        np.full(len(t), uid, dtype=np.int64)
        for uid, t in sorter_unit_map.items()
    ])
    order = np.argsort(all_t)
    ks_times = all_t[order]
    ks_units = all_u[order]

    match_counts: dict[int, int] = {}
    n_missed = 0
    ins = np.searchsorted(ks_times, lh_times)
    for lh_t, idx in zip(lh_times, ins):
        best_uid, best_d = None, coincidence_samp + 1
        if idx > 0:
            d = abs(int(lh_t) - int(ks_times[idx - 1]))
            if d < best_d:
                best_d, best_uid = d, int(ks_units[idx - 1])
        if idx < len(ks_times):
            d = abs(int(lh_t) - int(ks_times[idx]))
            if d < best_d:
                best_d, best_uid = d, int(ks_units[idx])
        if best_uid is not None and best_d <= coincidence_samp:
            match_counts[best_uid] = match_counts.get(best_uid, 0) + 1
        else:
            n_missed += 1

    n_total = lh_times.size
    dominant_frac = (max(match_counts.values()) / n_total) if match_counts else 0.0
    return dict(
        n_units_matched=len(match_counts),
        n_missed=n_missed,
        pct_missed=n_missed / max(1, n_total),
        dominant_frac=dominant_frac,
    )


class QCSummaryDialog(QDialog):
    """Pop-up recording-level QC summary panel."""

    def __init__(self, qc_results: dict, sorter_spike_units: dict,
                 fs: float = 20_000.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording QC Summary")
        self.resize(1100, 700)
        self._qc_results = qc_results
        self._sorter_spike_units = sorter_spike_units
        self._fs = fs
        self._build_ui()
        self._populate()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(6, 6, 6, 6)

        self._title = QLabel("Recording-level QC Summary")
        self._title.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0;")
        vlay.addWidget(self._title)

        self._glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self._glw)

        # 2×2 plot grid
        self._p_miss   = self._glw.addPlot(row=0, col=0, title="Miss Rate Distribution")
        self._p_frag   = self._glw.addPlot(row=0, col=1, title="KS Fragmentation Index")
        self._p_scatter= self._glw.addPlot(row=1, col=0, title="LH vs KS Spike Count")
        self._p_map    = self._glw.addPlot(row=1, col=1, title="Array Map (colour = miss rate)")

        for p in [self._p_miss, self._p_frag, self._p_scatter, self._p_map]:
            p.showGrid(x=True, y=True, alpha=0.15)

        # Close button
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        hlay = QHBoxLayout()
        hlay.addStretch()
        hlay.addWidget(btn)
        vlay.addLayout(hlay)

    # ── Data ─────────────────────────────────────────────────────────────────

    def _populate(self):
        results = self._qc_results
        units   = self._sorter_spike_units
        fs      = self._fs

        # Collect per-channel stats — only LH channels
        lh_channels = [ch for ch, r in results.items() if r.n_lh > 0]
        if not lh_channels:
            self._title.setText("No LH channels found — run batch QC first.")
            return

        miss_rates       = []   # float, NaN if no sorter
        frag_n_units     = []   # int: 0=missed-only, 1=clean, 2=split, 3+=bad
        lh_counts        = []
        ks_counts        = []
        channels_ordered = sorted(lh_channels)

        for ch in channels_ordered:
            r = results[ch]
            lh_times = np.sort(np.concatenate([
                t for t in [r.valley.left_times, r.valley.rightk_times] if t.size
            ])) if (r.valley.left_times.size + r.valley.rightk_times.size) else np.array([], dtype=np.int64)

            lh_counts.append(lh_times.size)
            ks_counts.append(r.n_sorter_spikes if r.n_sorter_spikes >= 0 else 0)

            unit_map = units.get(ch, {})
            fi = _fragmentation_index(unit_map, lh_times, fs)

            if fi is None:
                miss_rates.append(np.nan)
                frag_n_units.append(-1)
            else:
                miss_rates.append(fi["pct_missed"])
                frag_n_units.append(fi["n_units_matched"])

        miss_arr = np.array(miss_rates)
        frag_arr = np.array(frag_n_units)
        lh_arr   = np.array(lh_counts,  dtype=np.float64)
        ks_arr   = np.array(ks_counts,  dtype=np.float64)
        has_sorter = not np.all(np.isnan(miss_arr))

        self._title.setText(
            f"Recording QC Summary — {len(lh_channels)} LH channels"
            + (f"  |  median miss rate: {np.nanmedian(miss_arr)*100:.1f}%" if has_sorter else "  |  (no sorter loaded)")
        )

        # ── [0,0] Miss rate histogram ─────────────────────────────────────────
        p = self._p_miss
        if has_sorter:
            valid = miss_arr[~np.isnan(miss_arr)] * 100  # pct
            counts, edges = np.histogram(valid, bins=np.linspace(0, 100, 21))
            p.addItem(pg.BarGraphItem(
                x=edges[:-1], height=counts, width=(edges[1]-edges[0])*0.9,
                brush=pg.mkBrush("#2196F3"), pen=pg.mkPen("#1a1a1a"),
            ))
            p.setLabel("bottom", "Miss rate (%)")
            p.setLabel("left", "# channels")
            # Median line
            med = float(np.nanmedian(valid))
            p.addItem(pg.InfiniteLine(
                pos=med, angle=90,
                pen=pg.mkPen("#FF9800", width=1.5, style=Qt.DashLine),
                label=f"med={med:.1f}%",
                labelOpts={"color": "#FF9800", "position": 0.85},
            ))
        else:
            lbl = pg.TextItem("No sorter loaded", color="#5A5C65", anchor=(0.5, 0.5))
            p.addItem(lbl)
            lbl.setPos(0.5, 0.5)

        # ── [0,1] Fragmentation index bar chart ───────────────────────────────
        p = self._p_frag
        if has_sorter:
            bins_frag = {"Missed\nonly": 0, "1 unit\n(clean)": 1,
                         "2 units\n(split)": 2, "3+ units\n(bad)": 3}
            frag_counts = {}
            for label, lo in [("Missed\nonly", 0), ("1 unit\n(clean)", 1),
                               ("2 units\n(split)", 2), ("3+ units\n(bad)", 3)]:
                if lo == 0:
                    frag_counts[label] = int(np.sum(frag_arr == 0))
                elif lo == 3:
                    frag_counts[label] = int(np.sum(frag_arr >= 3))
                else:
                    frag_counts[label] = int(np.sum(frag_arr == lo))

            colors = ["#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
            labels = list(frag_counts.keys())
            vals   = list(frag_counts.values())
            xs = np.arange(len(labels), dtype=np.float64)
            for i, (lbl, val, col) in enumerate(zip(labels, vals, colors)):
                p.addItem(pg.BarGraphItem(
                    x=[xs[i]], height=[val], width=0.7,
                    brush=pg.mkBrush(col), pen=pg.mkPen("#1a1a1a"),
                ))
            ax = p.getAxis("bottom")
            ax.setTicks([[(xi, lbl) for xi, lbl in zip(xs, labels)]])
            p.setLabel("left", "# LH channels")
        else:
            lbl = pg.TextItem("No sorter loaded", color="#5A5C65", anchor=(0.5, 0.5))
            p.addItem(lbl)
            lbl.setPos(0.5, 0.5)

        # ── [1,0] LH vs KS scatter ───────────────────────────────────────────
        p = self._p_scatter
        if has_sorter and ks_arr.max() > 0:
            # Colour by miss rate: green=low, red=high
            norm_miss = np.nan_to_num(miss_arr, nan=0.0)
            colors_scatter = [
                pg.mkBrush(
                    int(255 * m), int(255 * (1 - m)), 60, 200
                ) for m in norm_miss
            ]
            scatter = pg.ScatterPlotItem(
                x=lh_arr, y=ks_arr,
                size=7, pen=pg.mkPen(None), brush=colors_scatter,
            )
            p.addItem(scatter)
            # Identity line
            mx = max(lh_arr.max(), ks_arr.max()) * 1.05
            p.plot([0, mx], [0, mx], pen=pg.mkPen("#555", width=1, style=Qt.DashLine))
            p.setLabel("bottom", "LH spike count")
            p.setLabel("left", "KS spike count")
        else:
            p.plot(lh_arr, ks_arr, pen=None, symbol="o",
                   symbolSize=5, symbolBrush=pg.mkBrush("#4CAF50"))
            p.setLabel("bottom", "LH spike count")
            p.setLabel("left", "KS spike count (0 = no sorter)")

        # ── [1,1] Array map coloured by miss rate ────────────────────────────
        p = self._p_map
        p.setAspectLocked(True)

        # Use channel index as a proxy for position if no geometry available
        # Arrange channels in a sqrt(C) × sqrt(C) grid
        all_chs = sorted(results.keys())
        C = len(all_chs)
        ncols = max(1, int(np.ceil(np.sqrt(C))))

        xs_map, ys_map, brushes = [], [], []
        for i, ch in enumerate(all_chs):
            xs_map.append(i % ncols)
            ys_map.append(i // ncols)
            r = results[ch]
            if r.n_lh == 0:
                # Not an LH channel — grey
                brushes.append(pg.mkBrush(80, 80, 80, 180))
            elif not has_sorter:
                # LH but no sorter — green shade by spike count
                brushes.append(pg.mkBrush("#4CAF50"))
            else:
                miss = miss_arr[channels_ordered.index(ch)] if ch in channels_ordered else np.nan
                if np.isnan(miss):
                    brushes.append(pg.mkBrush("#4CAF50"))
                else:
                    r_val = int(255 * miss)
                    g_val = int(255 * (1 - miss))
                    brushes.append(pg.mkBrush(r_val, g_val, 60, 220))

        sc = pg.ScatterPlotItem(
            x=xs_map, y=ys_map,
            size=9, pen=pg.mkPen(None), brush=brushes,
        )
        p.addItem(sc)
        p.setLabel("bottom", "col")
        p.setLabel("left", "row")
        p.setTitle("Array map  (grey=no LH, green=low miss, red=high miss)")