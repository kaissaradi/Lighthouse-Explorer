from __future__ import annotations
"""
theme.py — Centralised dark stylesheet for Lighthouse QC.

Extracted from gui/app.py so the application entry point stays purely functional.
"""

DARK_STYLESHEET = """
/* ── Base ───────────────────────────── */
QWidget {
    background-color: #111214;
    color: #F0F0F2;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #111214;
}

/* ── Splitter handles ────────────────── */
QSplitter::handle {
    background: #2E3038;
}
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }
QSplitter::handle:hover { background: #4A8BEF; }

/* ── Buttons ─────────────────────────── */
QPushButton {
    background-color: transparent;
    border: 0.5px solid #3D3F48;
    color: #9B9DA6;
    padding: 5px 12px;
    border-radius: 5px;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #1E2025;
    border-color: #5A5C65;
    color: #F0F0F2;
}
QPushButton:pressed {
    background-color: #282A30;
}
QPushButton:disabled {
    color: #3A3C44;
    border-color: #2E3038;
}

/* ── Labels ──────────────────────────── */
QLabel {
    color: #9B9DA6;
    font-size: 12px;
}

/* ── Inputs ──────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #18191C;
    border: 0.5px solid #3D3F48;
    border-radius: 4px;
    padding: 3px 6px;
    color: #F0F0F2;
    font-size: 12px;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #5A5C65;
}

/* ── ComboBox ────────────────────────── */
QComboBox {
    background-color: #18191C;
    border: 0.5px solid #3D3F48;
    border-radius: 4px;
    padding: 3px 8px;
    color: #F0F0F2;
    min-height: 22px;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background-color: #282A30;
    color: #F0F0F2;
    selection-background-color: rgba(46, 109, 212, 0.25);
}

/* ── GroupBox ────────────────────────── */
QGroupBox {
    border: 0.5px solid #2E3038;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: bold;
    color: #9B9DA6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

/* ── Scrollbars ──────────────────────── */
QScrollBar:vertical {
    background: #18191C;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #3D3F48;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #5A5C65; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Progress bar ────────────────────── */
QProgressBar {
    background-color: #18191C;
    border: 0.5px solid #3D3F48;
    border-radius: 4px;
    text-align: center;
    color: #9B9DA6;
    font-size: 11px;
    height: 8px;
}
QProgressBar::chunk {
    background-color: #2E6DD4;
    border-radius: 3px;
}

/* ── Status bar ──────────────────────── */
QStatusBar {
    color: #5A5C65;
    font-size: 11px;
    border-top: 0.5px solid #2E3038;
    background: #111214;
    padding: 2px 8px;
}
"""


# ── Semantic colour palette ─────────────────────────────────────────────────
# Import from this dict in all GUI panels to keep colours consistent.
COLORS = {
    # QC labels
    "lh": "#4CAF50",
    "soup": "#FF9800",
    "uncertain_boundary": "#9E9E9E",
    "uncertain_lowBL": "#757575",
    # PCA / waveform clusters
    "cluster0": "#2196F3",
    "cluster1": "#FF9800",
    # Sorter / overlay
    "sorter": "#2196F3",
    # Fragmentation bar chart
    "frag_missed": "#F44336",
    "frag_clean": "#4CAF50",
    "frag_split": "#FF9800",
    "frag_bad": "#9C27B0",
    # Error / status
    "error": "#F44336",
    # Histogram
    "hist_fill": "#2E6DD4",
    "hist_pen": "#4A8BEF",
    # Muted / placeholder
    "muted": "#5A5C65",
    "muted_text": "#888888",
}

from .panels import LoadPanel, ArrayMapPanel, QCViewPanel, QCSummaryDialog
"""
main_window.py — Top-level QMainWindow. Orchestrates panels and QC workflow.
"""
from typing import Optional
import numpy as np
import os
from qtpy.QtWidgets import (
    QMainWindow, QSplitter, QProgressBar, QPushButton, QFileDialog, QMessageBox,
)
from qtpy.QtCore import Qt, QThread
from .qc_worker import TaskManager, LoaderWorker, SorterLoaderWorker
from core.lh_qc_pipeline import DEFAULT_PARAMS
from core.lh_qc_pipeline import QCResult
from core.ks_export import export_phy_folder


class MainWindow(QMainWindow):
    """Main window for Lighthouse QC application."""

    def __init__(self, default_dat=None, default_n_channels=None):
        super().__init__()
        self.setWindowTitle("Lighthouse QC")
        self.setGeometry(100, 100, 1600, 900)

        # ── Recording state ──────────────────────────────────────────────────
        self.raw_data = None                    # np.memmap or ndarray [T, C]
        self.qc_results: dict = {}              # {ch: QCResult}
        self.current_channel: Optional[int] = None
        self.lh_params: dict = dict(DEFAULT_PARAMS)
        self.default_dat = default_dat
        self.default_n_channels = default_n_channels

        # ── KiloSort state ───────────────────────────────────────────────────
        # sorter_spike_times: {ch: np.ndarray of sample indices}
        #   — all KS spikes on that electrode channel, pooled across units.
        #   — used by TaskManager for per-channel miss-rate counting.
        self.sorter_spike_times: dict = {}

        # sorter_unit_map: {unit_id: np.ndarray of sample indices}
        #   — one entry per KS cluster, spike times in samples.
        self.sorter_unit_map: dict = {}

        # sorter_dom_channel: {unit_id: int}
        #   — dominant electrode channel for each KS cluster, derived from
        #     templates.npy via argmax of peak-to-peak amplitude across channels.
        #   — indices are 0-based electrode indices (same as raw_data columns
        #     after Litke TTL strip). Must match channel_map.npy from KS.
        self.sorter_dom_channel: dict = {}

        # sorter_units_by_channel: {ch: [unit_id, ...]} reverse index for fast attach
        self.sorter_units_by_channel: dict = {}

        # sorter_unit_labels: {unit_id: 'good'|'mua'|'unsorted'|...} from cluster_group.tsv
        self.sorter_unit_labels: dict = {}

        # Channel-map diagnostics from last KS parse
        self.sorter_channel_meta: dict = {}

        # ── Workers / threads ────────────────────────────────────────────────
        self._loader_thread: Optional[QThread] = None
        self._loader_worker: Optional[LoaderWorker] = None
        self._sorter_thread: Optional[QThread] = None
        self._sorter_worker: Optional[SorterLoaderWorker] = None

        # Unified task manager for single-channel AND batch QC (always 1 worker).
        self._task_manager = TaskManager(parent=self)
        self._connect_task_manager()

        self._build_ui()
        self._connect_signals()

        if self.default_dat or self.default_n_channels:
            self._load_panel.set_defaults(self.default_dat, self.default_n_channels)

    # ── TaskManager wiring ───────────────────────────────────────────────────

    def _connect_task_manager(self):
        """Wire TaskManager signals to MainWindow slots — once, at init."""
        tm = self._task_manager

        # Single-channel QC
        tm.single_result.connect(self.on_single_qc_finished)
        tm.single_error.connect(self.on_single_qc_error)
        tm.single_progress.connect(self._status_message)

        # Batch QC
        tm.batch_progress.connect(self._on_batch_progress)
        tm.batch_channel_done.connect(self._on_batch_channel_done)
        tm.batch_finished.connect(self._on_batch_finished)
        tm.batch_error.connect(self._on_batch_error)
        tm.batch_aborted.connect(self._on_batch_aborted)

    def _status_message(self, msg: str):
        """Slot adapter: forward a plain string to the status bar."""
        self._status_bar.showMessage(msg)

    # ── UI setup ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        """
        Layout:
        ┌─ LoadPanel (250px) ─┬─ Channel List (250px) ─┬─ QCViewPanel (fill) ─┐
        └─────────────────────┴────────────────────────┴──────────────────────┘
        Status bar at bottom.
        """
        self._load_panel = LoadPanel(self)
        self._channel_list = ArrayMapPanel(self)
        self._qc_view = QCViewPanel(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._load_panel)
        splitter.addWidget(self._channel_list)
        splitter.addWidget(self._qc_view)
        splitter.setSizes([250, 250, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        self.setCentralWidget(splitter)

        # Status bar
        self._status_bar = self.statusBar()
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(150)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)
        self._status_bar.showMessage("Ready. Load a recording to begin.")

        # Summary / export buttons (enabled once we have QC results)
        self._summary_btn = QPushButton("Recording Summary")
        self._summary_btn.setEnabled(False)
        self._summary_btn.setToolTip("Show recording-level QC summary across all channels")
        self._summary_btn.clicked.connect(self._show_summary)
        self._status_bar.addPermanentWidget(self._summary_btn)

        self._export_btn = QPushButton("Export Phy/KS")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip(
            "Write spike_times / spike_clusters / templates (Phy-compatible) "
            "from LH final_times"
        )
        self._export_btn.clicked.connect(self._export_phy)
        self._status_bar.addPermanentWidget(self._export_btn)

        self._batch_btn = QPushButton("Run Batch QC")
        self._batch_btn.setEnabled(False)
        self._batch_btn.setToolTip(
            "Run LH QC on all channels (single-threaded — safe for Numba). "
            "Does not auto-start after load unless enabled in Load panel."
        )
        self._batch_btn.clicked.connect(self._on_batch_btn_clicked)
        self._status_bar.addPermanentWidget(self._batch_btn)

    def _connect_signals(self):
        self._load_panel.load_requested.connect(self.on_load_requested)
        self._load_panel.sorter_load_requested.connect(self.on_sorter_load_requested)
        self._channel_list.channel_selected.connect(self.on_channel_selected)

    # ── Recording loading ─────────────────────────────────────────────────────

    def on_load_requested(self, params: dict):
        """Start background loading — supports flat .dat/.bin and Litke bin folder."""
        # Clear recording / QC state. KEEP sorter maps so KS can be loaded any time
        # (before or after recording) without wiping comparison data.
        self.raw_data = None
        self.qc_results.clear()
        self.current_channel = None

        self._qc_view.clear()
        self._channel_list.clear()
        self._summary_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)

        self.lh_params.update(params)
        # Absolute sample offset of memmap window (for KS time alignment)
        fs = float(params.get("fs", 20_000) or 20_000)
        start_min = float(params.get("start_min", 0.0) or 0.0)
        self.lh_params["start_sample"] = int(start_min * 60.0 * fs)

        dat_path = params.get("dat_path")
        if not dat_path:
            self._status_bar.showMessage("No .dat file or folder specified.")
            return

        # Warn on full-file huge loads (these can leave <20 GB free → OOM on QC)
        dur = params.get("duration_min")
        n_ch = int(params.get("n_channels", 0) or 0)
        fs = float(params.get("fs", 20_000) or 20_000)
        if dur is None:
            self._status_bar.showMessage(
                "Loading FULL file into RAM (baseline-sub needs writable data). "
                "If the process is Killed mid-batch, that is the Linux OOM killer — "
                "use a shorter duration (min) or free RAM. Prefer short duration for testing."
            )
        elif dur and n_ch:
            # Rough int16 size estimate for the window
            est_gb = (float(dur) * 60.0 * fs * n_ch * 2.0) / (1024.0 ** 3)
            if est_gb > 40:
                self._status_bar.showMessage(
                    f"Loading ~{est_gb:.0f} GB window — leave headroom; OOM = process Killed."
                )

        self._stop_loader(wait_ms=15_000)
        self._load_panel.set_loading_state(True)

        # Both flat .dat/.bin files AND Litke folders go through LoaderWorker.
        # Never short-circuit to lazy LitkeMultiFileArray — that skips baseline
        # subtraction and freezes the UI on slow per-chunk reads.
        self._loader_thread = QThread()
        self._loader_worker = LoaderWorker(
            dat_path=dat_path,
            n_channels=params["n_channels"],
            dtype=params.get("dtype", "int16"),
            start_min=params.get("start_min", 0.0),
            duration_min=params.get("duration_min", None),
            fs=params.get("fs", 20_000),
        )
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.progress.connect(self._status_bar.showMessage)
        self._loader_worker.finished.connect(self._on_loader_finished)
        self._loader_worker.error.connect(self._on_loader_error)
        self._loader_worker.aborted.connect(self._on_loader_aborted)

        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.error.connect(self._loader_thread.quit)
        self._loader_worker.aborted.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._on_loader_thread_finished)

        self._loader_thread.start()

    def _on_loader_thread_finished(self):
        """Clear thread/worker refs after QThread exits cleanly."""
        thr = self._loader_thread
        worker = self._loader_worker
        self._loader_thread = None
        self._loader_worker = None
        if worker is not None:
            worker.deleteLater()
        if thr is not None:
            thr.deleteLater()

    def _on_loader_finished(self, raw_data):
        """Loader completed — store data; batch QC only if user opted in."""
        n_ch = raw_data.shape[1]
        self.raw_data = raw_data
        self._channel_list.set_array(np.arange(n_ch).reshape(-1, 1))
        self._load_panel.set_loading_state(False)
        self._batch_btn.setEnabled(True)

        n_samp = int(raw_data.shape[0])
        mins = n_samp / max(1.0, float(self.lh_params.get("fs", 20_000))) / 60.0
        auto_batch = bool(self.lh_params.get("auto_batch_qc", True))

        # If KS was already loaded, re-check channel alignment vs this recording
        ks_note = ""
        if self.sorter_unit_map or self.sorter_spike_times:
            align = self._validate_sorter_channel_alignment()
            ks_note = f" | KS: {len(self.sorter_unit_map)} units"
            if align:
                ks_note += f" ({align})"

        if auto_batch:
            self._status_bar.showMessage(
                f"Loaded {n_samp:,} samples × {n_ch} ch ({mins:.1f} min)"
                f"{ks_note}. Starting batch QC (1 worker, sequential)…"
            )
            self._start_batch_qc()
        else:
            self._status_bar.showMessage(
                f"Loaded {n_samp:,} samples × {n_ch} ch ({mins:.1f} min)"
                f"{ks_note}. Click a channel for QC, or 'Run Batch QC'."
            )

    def _on_loader_error(self, msg: str):
        self._status_bar.showMessage(msg)
        self._qc_view.show_error(msg)
        self._load_panel.set_loading_state(False)

    def _on_loader_aborted(self):
        self._status_bar.showMessage("Loading cancelled.")
        self._load_panel.set_loading_state(False)

    def _stop_loader(self, wait_ms: int = 30_000) -> bool:
        """
        Request abort and wait for the loader thread to finish.

        Returns True if the thread is no longer running.
        Never calls terminate() (unsafe with Numba).
        """
        worker = self._loader_worker
        thr = self._loader_thread
        if worker is not None:
            worker.abort()
        if thr is None:
            return True
        if thr.isRunning():
            thr.quit()
            finished = thr.wait(wait_ms)
            if not finished:
                # Still blocked inside Numba — wait a bit longer once more.
                finished = thr.wait(5_000)
            if not finished:
                self._status_bar.showMessage(
                    "Loader still running in background — wait before closing again."
                )
                return False
        return True

    # ── KiloSort loading (background thread) ──────────────────────────────────

    def on_sorter_load_requested(self, path: str):
        """Start background KS parse — never blocks the UI thread."""
        path = (path or "").strip()
        if not path:
            self._status_bar.showMessage("No KS folder specified.")
            return
        if not os.path.isdir(path):
            self._status_bar.showMessage(f"KS path is not a directory: {path}")
            return

        self._stop_sorter(wait_ms=10_000)
        self._load_panel.set_sorter_loading_state(True)
        self._status_bar.showMessage("Loading KiloSort output (background)…")

        self._sorter_thread = QThread()
        self._sorter_worker = SorterLoaderWorker(path)
        self._sorter_worker.moveToThread(self._sorter_thread)
        self._sorter_thread.started.connect(self._sorter_worker.run)
        self._sorter_worker.progress.connect(self._status_bar.showMessage)
        self._sorter_worker.finished.connect(self._on_sorter_finished)
        self._sorter_worker.error.connect(self._on_sorter_error)
        self._sorter_worker.aborted.connect(self._on_sorter_aborted)

        self._sorter_worker.finished.connect(self._sorter_thread.quit)
        self._sorter_worker.error.connect(self._sorter_thread.quit)
        self._sorter_worker.aborted.connect(self._sorter_thread.quit)
        self._sorter_thread.finished.connect(self._on_sorter_thread_finished)

        self._sorter_thread.start()

    def _on_sorter_thread_finished(self):
        thr = self._sorter_thread
        worker = self._sorter_worker
        self._sorter_thread = None
        self._sorter_worker = None
        if worker is not None:
            worker.deleteLater()
        if thr is not None:
            thr.deleteLater()

    def _on_sorter_finished(self, result: dict):
        self.sorter_unit_map = result.get("unit_map", {})
        self.sorter_dom_channel = result.get("dom_channel", {})
        self.sorter_spike_times = result.get("spike_times_by_channel", {})
        self.sorter_units_by_channel = result.get("units_by_channel", {})
        self.sorter_unit_labels = result.get("unit_labels", {})
        self.sorter_channel_meta = {
            "channel_map_min": result.get("channel_map_min"),
            "channel_map_max": result.get("channel_map_max"),
            "channel_map_n": result.get("channel_map_n"),
            "channel_map_is_identity": result.get("channel_map_is_identity"),
            "template_n_channels": result.get("template_n_channels"),
        }
        self._load_panel.set_sorter_loading_state(False)

        # Drop / flag channels that don't exist on the loaded recording
        align_note = self._validate_sorter_channel_alignment()

        n_units = int(result.get("n_units", len(self.sorter_unit_map)))
        n_ch = int(result.get("n_channels_covered", len(self.sorter_spike_times)))
        n_noise = int(result.get("n_noise_excluded", 0))
        n_unmapped = int(result.get("n_unmapped_units", 0))
        cmap_n = result.get("channel_map_n")
        cmap_id = result.get("channel_map_is_identity")
        msg = f"KS loaded: {n_units} units across {n_ch} channels"
        if cmap_n is not None:
            msg += f" | map 0-based ch 0…{int(result.get('channel_map_max', 0))}"
            if cmap_id:
                msg += " (identity)"
        if n_noise:
            msg += f" ({n_noise} noise excluded)"
        if n_unmapped:
            msg += f" [{n_unmapped} units w/o peak channel]"
        if align_note:
            msg += f" | {align_note}"
        self._status_bar.showMessage(msg)

        # Re-attach to EVERY finished QC result so Venn/miss update immediately
        self._reattach_sorter_to_all_results()

    def _on_sorter_error(self, msg: str):
        self._load_panel.set_sorter_loading_state(False)
        self._status_bar.showMessage(msg.split("\n", 1)[0])

    def _on_sorter_aborted(self):
        self._load_panel.set_sorter_loading_state(False)
        self._status_bar.showMessage("KS load cancelled.")

    def _stop_sorter(self, wait_ms: int = 10_000) -> bool:
        worker = self._sorter_worker
        thr = self._sorter_thread
        if worker is not None:
            worker.abort()
        if thr is None:
            return True
        if thr.isRunning():
            thr.quit()
            return bool(thr.wait(wait_ms))
        return True

    def _clip_times_to_loaded_window(self, times: np.ndarray) -> np.ndarray:
        """Map absolute KS sample indices into the loaded memmap window."""
        times = np.asarray(times, dtype=np.int64)
        if times.size == 0:
            return times
        start = int(self.lh_params.get("start_sample", 0) or 0)
        rel = times - start
        if self.raw_data is None:
            return rel[rel >= 0]
        t_max = int(self.raw_data.shape[0])
        return rel[(rel >= 0) & (rel < t_max)]

    def _recording_n_channels(self) -> int | None:
        if self.raw_data is not None:
            return int(self.raw_data.shape[1])
        n = self.lh_params.get("n_channels")
        try:
            return int(n) if n is not None else None
        except (TypeError, ValueError):
            return None

    def _validate_sorter_channel_alignment(self) -> str:
        """
        Confirm KS peak-channel indices line up with recording columns.

        Convention (lab Litke + KS):
          • Litke loader strips TTL ch 0 → raw_data cols are electrodes 0..C-1
          • KS channel_map.npy is typically identity arange(C) on the same
            pre-stripped dat (n_channels_dat = C)
          • Both use 0-based indices; UI labels are 1-based (CH 1 = index 0)

        Returns a short status note (empty if nothing notable).
        """
        n_rec = self._recording_n_channels()
        if n_rec is None:
            return "load recording to verify channel alignment"

        meta = self.sorter_channel_meta or {}
        cmap_max = meta.get("channel_map_max")
        cmap_min = meta.get("channel_map_min")
        keys = list(self.sorter_spike_times.keys())
        if not keys and cmap_max is None:
            return "KS has no channel-mapped spikes"

        max_ks = int(cmap_max) if cmap_max is not None else (max(keys) if keys else -1)
        min_ks = int(cmap_min) if cmap_min is not None else (min(keys) if keys else 0)

        # Drop OOR channels so we never attach spikes to missing electrodes
        oor = [ch for ch in keys if ch < 0 or ch >= n_rec]
        if oor:
            for ch in oor:
                self.sorter_spike_times.pop(ch, None)
                self.sorter_units_by_channel.pop(ch, None)
            # Also drop unit maps whose dom ch is OOR
            bad_uids = [
                uid for uid, dch in self.sorter_dom_channel.items()
                if dch < 0 or dch >= n_rec
            ]
            for uid in bad_uids:
                self.sorter_unit_map.pop(uid, None)
                self.sorter_dom_channel.pop(uid, None)
                self.sorter_unit_labels.pop(uid, None)
            return (
                f"⚠️ {len(oor)} KS channels outside recording 0…{n_rec - 1} "
                f"(KS range {min_ks}…{max_ks}) — dropped"
            )

        if max_ks >= n_rec or min_ks < 0:
            return (
                f"⚠️ KS channel_map {min_ks}…{max_ks} vs recording 0…{n_rec - 1}"
            )

        if meta.get("channel_map_is_identity") and max_ks == n_rec - 1:
            return f"aligned ✓ (0…{n_rec - 1})"
        if max_ks < n_rec:
            return f"aligned ✓ (KS covers 0…{max_ks} of {n_rec} ch)"
        return "aligned ✓"

    def _reattach_sorter_to_all_results(self) -> None:
        """Push current KS maps onto every cached QCResult and refresh UI."""
        if not self.qc_results:
            return
        for ch, r in self.qc_results.items():
            self._attach_sorter_data(r)
            r.n_sorter_spikes = self._n_sorter_spikes_for_channel(ch)
            self._channel_list.update_channel_result(ch, r)
        if self.current_channel is not None and self.current_channel in self.qc_results:
            self._qc_view.show_result(self.qc_results[self.current_channel])

    # ── Batch QC lifecycle ────────────────────────────────────────────────────

    def _on_batch_btn_clicked(self):
        if self.raw_data is None:
            self._status_bar.showMessage("Load a recording first.")
            return
        if self._task_manager.is_batch_running:
            self._status_bar.showMessage("Batch QC already running…")
            return
        self._start_batch_qc()

    def _start_batch_qc(self):
        """Start running QC on all channels (always 1 pool thread — Numba-safe)."""
        if self.raw_data is None:
            return

        self._task_manager.abort_batch()
        self._channel_list.hide_progress()
        self._batch_btn.setEnabled(False)
        self._status_bar.showMessage(
            f"Batch QC starting on {self.raw_data.shape[1]} channels (1 worker)…"
        )

        self._task_manager.start_batch(
            raw_data=self.raw_data,
            params=self.lh_params,
            sorter_spike_times=self.sorter_spike_times,
            fs=self.lh_params.get("fs"),
        )

    def _on_batch_progress(self, msg: str, current: int, total: int):
        self._status_bar.showMessage(msg)
        self._channel_list.set_progress(current + 1, total, msg)

    def _attach_sorter_data(self, result: QCResult):
        """
        Attach KS data to a QCResult after pipeline completion.

        Channel index convention (must match on both sides):
          result.channel is 0-based raw_data column index (post Litke TTL strip).
          KS peak channels come from channel_map[argmax_ptp(templates)] and are
          also 0-based electrode indices into that same coordinate system.

        Populates:
          result.fs               — sampling rate from lh_params
          result.sorter_times     — pooled spike times for all KS units on this ch
                                    (clipped to loaded window; relative samples)
          result.sorter_unit_map  — {unit_id: times} for units whose dominant
                                    channel == result.channel
          result.sorter_unit_labels — {unit_id: good|mua|...} for those units
        """
        ch = result.channel
        result.fs = self.lh_params.get("fs", 20_000)

        # Pooled times — keyed by 0-based electrode index (same as raw_data cols)
        pooled = self.sorter_spike_times.get(ch, None)
        if pooled is not None:
            clipped = self._clip_times_to_loaded_window(pooled)
            result.sorter_times = clipped if clipped.size else np.array([], dtype=np.int64)
        else:
            result.sorter_times = np.array([], dtype=np.int64)

        # Per-unit map via reverse index (O(units on ch), not O(all units))
        unit_map: dict = {}
        unit_labels: dict = {}
        uids = self.sorter_units_by_channel.get(ch)
        if uids is None:
            # Fallback if older parse without reverse index
            uids = [
                uid for uid, dch in self.sorter_dom_channel.items() if dch == ch
            ]
        for uid in uids:
            times = self.sorter_unit_map.get(uid)
            if times is None:
                continue
            clipped = self._clip_times_to_loaded_window(times)
            if clipped.size:
                unit_map[uid] = clipped
                unit_labels[uid] = self.sorter_unit_labels.get(uid, "unsorted")
        result.sorter_unit_map = unit_map
        result.sorter_unit_labels = unit_labels

        # Keep n_sorter_spikes in sync (used by miss-rate badge)
        result.n_sorter_spikes = (
            int(result.sorter_times.size)
            if result.sorter_times is not None and result.sorter_times.size
            else (0 if self.sorter_spike_times else -1)
        )

    def _on_batch_channel_done(self, result: QCResult):
        ch = result.channel
        self._attach_sorter_data(result)
        self.qc_results[ch] = result
        self._channel_list.update_channel_result(ch, result)
        if self.current_channel == ch:
            self._qc_view.show_result(result)

    def _on_batch_finished(self, results: dict):
        self._channel_list.hide_progress()
        self._batch_btn.setEnabled(self.raw_data is not None)

        lh_count = sum(1 for r in self.qc_results.values() if r.n_lh > 0)
        total = results.get("total", len(self.qc_results))
        self._status_bar.showMessage(
            f"Batch QC complete: {lh_count}/{total} channels with LH spikes found."
        )
        self._summary_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._channel_list._view_combo.setCurrentText("LH Found")

        for ch, result in sorted(self.qc_results.items()):
            if result.n_lh > 0:
                self.current_channel = ch
                self._channel_list.set_selected_channel(ch)
                self._qc_view.show_result(result)
                break

    def _on_batch_error(self, msg: str):
        self._channel_list.hide_progress()
        self._batch_btn.setEnabled(self.raw_data is not None)
        self._status_bar.showMessage(f"Batch QC failed: {msg}")
        self._qc_view.show_error(msg)

    def _on_batch_aborted(self):
        self._channel_list.hide_progress()
        self._batch_btn.setEnabled(self.raw_data is not None)
        self._status_bar.showMessage(
            f"Batch QC cancelled. {len(self.qc_results)} channels completed."
        )

    # ── Single QC lifecycle ───────────────────────────────────────────────────

    def on_channel_selected(self, ch: int):
        """User clicked a channel in the list."""
        if self.raw_data is None:
            self._status_bar.showMessage("Load a recording first.")
            return

        self.current_channel = ch
        self._channel_list.set_selected_channel(ch)

        dch = ch + 1  # UI is 1-based
        if ch in self.qc_results:
            self._qc_view.show_result(self.qc_results[ch])
            self._status_bar.showMessage(f"CH {dch}: cached result displayed.")
            return

        if self._task_manager.is_batch_running:
            self._qc_view.show_loading(ch)
            self._status_bar.showMessage(f"CH {dch}: queued for batch QC…")
            return

        self._qc_view.show_loading(ch)
        self._start_single_qc(ch)

    def _start_single_qc(self, ch: int):
        """Dispatch a single-channel QC through the unified TaskManager."""
        n_sorter = self._n_sorter_spikes_for_channel(ch)
        self._task_manager.start_single(
            raw_data=self.raw_data,
            channel=ch,
            n_sorter_spikes=n_sorter,
            params=self.lh_params,
            fs=self.lh_params.get("fs"),
        )

    def on_single_qc_finished(self, result: QCResult):
        ch = result.channel
        self._attach_sorter_data(result)
        self.qc_results[ch] = result
        self._qc_view.show_result(result)
        self._channel_list.update_channel_result(ch, result)
        self._summary_btn.setEnabled(True)
        self._export_btn.setEnabled(True)

        dch = ch + 1
        label = (
            f"CH {dch}: {result.n_lh} LH, {result.n_soup} soup, "
            f"{result.n_uncertain} uncertain"
        )
        if result.miss_rate is not None:
            label += f", miss={result.miss_rate:.1%}"
        if result.n_sorter_spikes >= 0:
            label += f", KS={result.n_sorter_spikes}"
        self._status_bar.showMessage(label)

    def on_single_qc_error(self, msg: str):
        self._status_bar.showMessage(f"QC failed: {msg}")
        self._qc_view.show_error(msg)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _n_sorter_spikes_for_channel(self, ch: int) -> int:
        """Return count of pooled KS spikes on electrode ch in the loaded window.

        Returns -1 if no sorter has been loaded at all; 0 if sorter loaded but
        this channel has no spikes in-window.
        """
        if not self.sorter_spike_times and not self.sorter_unit_map:
            return -1
        times = self.sorter_spike_times.get(ch, None)
        if times is None or np.asarray(times).size == 0:
            return 0
        clipped = self._clip_times_to_loaded_window(times)
        return int(clipped.size)

    def _show_summary(self):
        """Open the recording-level QC summary dialog."""
        if not self.qc_results:
            self._status_bar.showMessage("No QC results yet.")
            return
        dlg = QCSummaryDialog(
            qc_results=self.qc_results,
            sorter_unit_map=self.sorter_unit_map,
            fs=self.lh_params.get("fs", 20_000),
            parent=self,
        )
        dlg.exec_()

    def _export_phy(self):
        """Write a Phy/Kilosort-compatible folder from current QC results."""
        if not self.qc_results:
            self._status_bar.showMessage("No QC results to export.")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select (or create) folder for Phy/KS export",
        )
        if not out_dir:
            return

        n_channels = (
            int(self.raw_data.shape[1])
            if self.raw_data is not None
            else int(self.lh_params.get("n_channels", 0) or 0)
        )
        if n_channels < 1:
            QMessageBox.warning(
                self,
                "Export failed",
                "Unknown channel count — load a recording first.",
            )
            return

        dat_path = self.lh_params.get("dat_path")
        fs = float(self.lh_params.get("fs", 20_000))
        dtype = str(self.lh_params.get("dtype", "int16"))

        try:
            info = export_phy_folder(
                out_dir,
                self.qc_results,
                n_channels=n_channels,
                fs=fs,
                dat_path=dat_path,
                dtype=dtype,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self._status_bar.showMessage(f"Export failed: {exc}")
            return

        msg = (
            f"Exported {info['n_units']} units / {info['n_spikes']} spikes → {info['out_dir']}"
        )
        self._status_bar.showMessage(msg)
        QMessageBox.information(self, "Export complete", msg)

    def closeEvent(self, event):
        """Graceful shutdown: abort workers and wait so QThreads don't get destroyed live."""
        self._task_manager.abort_batch()
        loader_ok = self._stop_loader(wait_ms=20_000)
        sorter_ok = self._stop_sorter(wait_ms=10_000)
        if not loader_ok or not sorter_ok:
            # Refuse to crash — ask user to wait (baselines mid-Numba can't interrupt).
            reply = QMessageBox.question(
                self,
                "Background work still running",
                "A loader/sorter thread is still finishing (Numba cannot interrupt mid-call).\n\n"
                "Wait a few more seconds and try closing again?\n"
                "Force quit may abort the process.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                event.ignore()
                return
        event.accept()

"""
app.py — QApplication setup and entry point.

The dark stylesheet lives in gui/theme.py.
"""
import sys
from qtpy.QtWidgets import QApplication


def create_app(argv) -> tuple[QApplication, MainWindow]:
    """Build QApplication, apply stylesheet, return (app, window)."""
    app = QApplication(argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    return app, window


def run(argv=None, default_dat=None, default_n_channels=None):
    """Full entry: create_app → show → exec."""
    if argv is None:
        argv = sys.argv
    app, window = create_app(argv)

    # Apply CLI defaults
    if default_dat:
        window.default_dat = default_dat
    if default_n_channels:
        window.default_n_channels = default_n_channels

    window.show()
    sys.exit(app.exec_())