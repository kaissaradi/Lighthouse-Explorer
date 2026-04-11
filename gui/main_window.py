"""
main_window.py — Top-level QMainWindow. Orchestrates panels and QC workflow.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import os
from qtpy.QtWidgets import QMainWindow, QSplitter, QProgressBar, QPushButton
from qtpy.QtCore import Qt, QThread
from .panels.load_panel import LoadPanel
from .panels.array_map_panel import ArrayMapPanel
from .panels.qc_view_panel import QCViewPanel
from .workers.qc_worker import QCWorker
from .workers.batch_qc_worker import BatchQCWorker
from .workers.loader_worker import LoaderWorker
from core.lh_qc_pipeline import DEFAULT_PARAMS
from core.result_types import QCResult
from .panels.qc_summary_dialog import QCSummaryDialog


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
        #   — used by BatchQCWorker for per-channel miss-rate counting.
        self.sorter_spike_times: dict = {}

        # sorter_unit_map: {unit_id: np.ndarray of sample indices}
        #   — one entry per KS cluster, spike times in samples.
        self.sorter_unit_map: dict = {}

        # sorter_dom_channel: {unit_id: int}
        #   — dominant electrode channel for each KS cluster, derived from
        #     templates.npy via argmax of peak-to-peak amplitude across channels.
        self.sorter_dom_channel: dict = {}

        # ── Workers / threads ────────────────────────────────────────────────
        self._loader_thread: Optional[QThread] = None
        self._loader_worker: Optional[LoaderWorker] = None
        self._single_thread: Optional[QThread] = None
        self._single_worker: Optional[QCWorker] = None
        self._batch_thread: Optional[QThread] = None
        self._batch_worker: Optional[BatchQCWorker] = None
        self._running_batch = False

        self._build_ui()
        self._connect_signals()

        if self.default_dat or self.default_n_channels:
            self._load_panel.set_defaults(self.default_dat, self.default_n_channels)

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

        # Summary button (enabled after batch QC)
        self._summary_btn = QPushButton("Recording Summary")
        self._summary_btn.setEnabled(False)
        self._summary_btn.setToolTip("Show recording-level QC summary across all channels")
        self._summary_btn.clicked.connect(self._show_summary)
        self._status_bar.addPermanentWidget(self._summary_btn)

    def _connect_signals(self):
        self._load_panel.load_requested.connect(self.on_load_requested)
        self._load_panel.sorter_load_requested.connect(self.on_sorter_load_requested)
        self._channel_list.channel_selected.connect(self.on_channel_selected)

    # ── Recording loading ─────────────────────────────────────────────────────

    def on_load_requested(self, params: dict):
        """Start background loading — supports flat .dat/.bin and Litke bin folder."""
        self.lh_params.update(params)

        dat_path = params.get("dat_path")
        if not dat_path:
            self._status_bar.showMessage("No .dat file or folder specified.")
            return

        self._abort_loader()
        self._load_panel.set_loading_state(True)

        # Both flat .dat/.bin files AND Litke folders go through LoaderWorker.
        # LoaderWorker.run() detects the format via os.path.isdir() and handles both:
        #   - flat file    -> np.memmap (copy-on-write) + in-place baseline subtraction
        #   - Litke folder -> materialised (T, C) ndarray + in-place baseline subtraction
        # Never short-circuit to the lazy LitkeMultiFileArray here — doing so skips
        # baseline subtraction entirely and blocks the main thread on slow per-chunk
        # reads, which freezes the UI and stalls batch QC on channel 0.
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
        self._loader_worker.finished.connect(self._loader_worker.deleteLater)
        self._loader_worker.error.connect(self._loader_thread.quit)
        self._loader_worker.aborted.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)

        self._loader_thread.start()

    def _on_loader_finished(self, raw_data):
        """Loader completed — store data and auto-start batch QC."""
        n_ch = raw_data.shape[1]
        self.raw_data = raw_data
        self._channel_list.set_array(np.arange(n_ch).reshape(-1, 1))
        self._status_bar.showMessage(
            f"Loaded {self.raw_data.shape[0]} samples × {n_ch} channels. Starting batch QC…"
        )
        self._load_panel.set_loading_state(False)
        self._start_batch_qc()

    def _on_loader_error(self, msg: str):
        self._status_bar.showMessage(msg)
        self._qc_view.show_error(msg)
        self._load_panel.set_loading_state(False)

    def _on_loader_aborted(self):
        self._status_bar.showMessage("Loading cancelled.")
        self._load_panel.set_loading_state(False)

    def _abort_loader(self):
        if self._loader_worker:
            self._loader_worker.abort()
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()

    # ── KiloSort loading ──────────────────────────────────────────────────────

    def on_sorter_load_requested(self, path: str):
        """
        Load a KiloSort output folder and build three lookup structures:

          self.sorter_unit_map     {unit_id: spike_times_array}
          self.sorter_dom_channel  {unit_id: dominant_electrode_channel}
          self.sorter_spike_times  {electrode_ch: pooled_spike_times_array}

        Required files (all standard KS2/KS4 outputs):
          spike_times.npy    — [nSpikes] uint64, sample indices
          spike_clusters.npy — [nSpikes] uint32, cluster ID per spike
          templates.npy      — [nTemplates, nTimepoints, nChannels] float32
          channel_map.npy    — [nChannels] int32, maps template ch idx → electrode ch

        Optional:
          cluster_group.tsv  — if present, noise clusters are excluded automatically.
        """
        try:
            self._status_bar.showMessage("Loading KiloSort output…")

            # ── 1. Load required npy files ───────────────────────────────────
            spike_times   = np.load(os.path.join(path, "spike_times.npy")).flatten().astype(np.int64)
            spike_clusters = np.load(os.path.join(path, "spike_clusters.npy")).flatten().astype(np.int32)
            templates     = np.load(os.path.join(path, "templates.npy"))   # [nTemplates, T, C]
            channel_map   = np.load(os.path.join(path, "channel_map.npy")).flatten().astype(np.int32)

            # ── 2. Optional: filter out noise clusters ───────────────────────
            noise_ids: set = set()
            group_path = os.path.join(path, "cluster_group.tsv")
            if os.path.exists(group_path):
                import csv
                with open(group_path, newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        if row.get("group", "").strip().lower() == "noise":
                            noise_ids.add(int(row["cluster_id"]))

            # ── 3. Dominant channel per unit ─────────────────────────────────
            # templates shape: [nTemplates, nTimepoints, nChannels]
            # ptp across time axis → [nTemplates, nChannels]
            # argmax across channel axis → [nTemplates] — index into channel_map
            template_ptp = templates.ptp(axis=1)           # [nTemplates, nChannels]
            template_dom_idx = template_ptp.argmax(axis=1)  # [nTemplates] — channel_map index
            # Map from template index → electrode channel number
            template_dom_ch = channel_map[template_dom_idx] # [nTemplates]

            # ── 4. Build unit_map and dom_channel ────────────────────────────
            self.sorter_unit_map = {}
            self.sorter_dom_channel = {}

            unique_units = np.unique(spike_clusters)
            for uid in unique_units:
                if uid in noise_ids:
                    continue
                mask = spike_clusters == uid
                unit_times = spike_times[mask]
                self.sorter_unit_map[int(uid)] = unit_times
                # cluster id == template id in KS output (may differ after manual
                # merges in Phy, but cluster id is still a valid template index
                # as long as it's in range)
                if uid < len(template_dom_ch):
                    self.sorter_dom_channel[int(uid)] = int(template_dom_ch[uid])
                else:
                    # Fallback: compute dominant channel directly from spikes
                    self.sorter_dom_channel[int(uid)] = -1

            # ── 5. Build per-electrode sorter_spike_times ────────────────────
            # This is the dict BatchQCWorker uses for miss-rate counting:
            # {electrode_ch: all spike times of all units whose dom ch == electrode_ch}
            ch_to_times: dict[int, list] = {}
            for uid, times in self.sorter_unit_map.items():
                dom_ch = self.sorter_dom_channel.get(uid, -1)
                if dom_ch < 0:
                    continue
                if dom_ch not in ch_to_times:
                    ch_to_times[dom_ch] = []
                ch_to_times[dom_ch].append(times)

            self.sorter_spike_times = {
                ch: np.sort(np.concatenate(time_lists))
                for ch, time_lists in ch_to_times.items()
            }

            n_units = len(self.sorter_unit_map)
            n_ch_covered = len(self.sorter_spike_times)
            self._status_bar.showMessage(
                f"KS loaded: {n_units} units across {n_ch_covered} channels"
                + (f" ({len(noise_ids)} noise excluded)" if noise_ids else "")
            )

        except FileNotFoundError as e:
            self._status_bar.showMessage(f"KS load failed — missing file: {e}")
        except Exception as e:
            self._status_bar.showMessage(f"KS load failed: {e}")

    # ── Batch QC lifecycle ────────────────────────────────────────────────────

    def _start_batch_qc(self):
        """Start running QC on all channels."""
        if self.raw_data is None:
            return

        self._abort_batch_worker()
        self._running_batch = True
        self._channel_list.hide_progress()

        self._batch_thread = QThread()
        self._batch_worker = BatchQCWorker(
            raw_data=self.raw_data,
            params=self.lh_params,
            sorter_spike_times=self.sorter_spike_times,
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.channel_done.connect(self._on_batch_channel_done)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.aborted.connect(self._on_batch_aborted)

        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.error.connect(self._batch_thread.quit)
        self._batch_worker.aborted.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)

        self._batch_thread.start()

    def _on_batch_progress(self, msg: str, current: int, total: int):
        self._status_bar.showMessage(msg)
        self._channel_list.set_progress(current + 1, total, msg)

    def _attach_sorter_data(self, result: QCResult):
        """
        Attach KS data to a QCResult after pipeline completion.

        Populates:
          result.fs               — sampling rate from lh_params
          result.sorter_times     — pooled spike times for all KS units on this ch
                                    (used for FR-over-time overlay in the view panel)
          result.sorter_unit_map  — {unit_id: times} for units whose dominant
                                    channel == result.channel
                                    (used for Venn / raster in the view panel)
        """
        ch = result.channel
        result.fs = self.lh_params.get("fs", 20_000)

        # Pooled times — already keyed by electrode channel
        result.sorter_times = self.sorter_spike_times.get(ch, None)

        # Per-unit map — only units whose dominant channel is this electrode
        result.sorter_unit_map = {
            uid: times
            for uid, times in self.sorter_unit_map.items()
            if self.sorter_dom_channel.get(uid, -1) == ch
        }

    def _on_batch_channel_done(self, result: QCResult):
        ch = result.channel
        self._attach_sorter_data(result)
        self.qc_results[ch] = result
        self._channel_list.update_channel_result(ch, result)
        if self.current_channel == ch:
            self._qc_view.show_result(result)

    def _on_batch_finished(self, results: dict):
        self._running_batch = False
        self._channel_list.hide_progress()

        lh_count = sum(1 for r in self.qc_results.values() if r.n_lh > 0)
        total = results.get("total", len(self.qc_results))
        self._status_bar.showMessage(
            f"Batch QC complete: {lh_count}/{total} channels with LH spikes found."
        )
        self._summary_btn.setEnabled(True)
        self._channel_list._view_combo.setCurrentText("LH Found")

        for ch, result in sorted(self.qc_results.items()):
            if result.n_lh > 0:
                self.current_channel = ch
                self._channel_list.set_selected_channel(ch)
                self._qc_view.show_result(result)
                break

    def _on_batch_error(self, msg: str):
        self._running_batch = False
        self._channel_list.hide_progress()
        self._status_bar.showMessage(f"Batch QC failed: {msg}")
        self._qc_view.show_error(msg)

    def _on_batch_aborted(self):
        self._running_batch = False
        self._channel_list.hide_progress()
        self._status_bar.showMessage(
            f"Batch QC cancelled. {len(self.qc_results)} channels completed."
        )

    # ── Single QC lifecycle ───────────────────────────────────────────────────

    def _abort_batch_worker(self):
        if self._batch_worker:
            self._batch_worker.abort()
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.quit()

    def on_channel_selected(self, ch: int):
        """User clicked a channel in the list."""
        if self.raw_data is None:
            self._status_bar.showMessage("Load a recording first.")
            return

        self.current_channel = ch
        self._channel_list.set_selected_channel(ch)

        if ch in self.qc_results:
            self._qc_view.show_result(self.qc_results[ch])
            self._status_bar.showMessage(f"CH {ch}: cached result displayed.")
            return

        if self._running_batch:
            self._qc_view.show_loading(ch)
            self._status_bar.showMessage(f"CH {ch}: queued for batch QC…")
            return

        self._abort_single_worker()
        self._qc_view.show_loading(ch)
        self._start_single_qc_worker(ch)

    def _start_single_qc_worker(self, ch: int):
        n_sorter = self._n_sorter_spikes_for_channel(ch)

        self._single_thread = QThread()
        self._single_worker = QCWorker(
            raw_data=self.raw_data,
            channel=ch,
            n_sorter_spikes=n_sorter,
            params=self.lh_params,
        )
        self._single_worker.moveToThread(self._single_thread)
        self._single_thread.started.connect(self._single_worker.run)
        self._single_worker.progress.connect(self._status_bar.showMessage)
        self._single_worker.finished.connect(self.on_single_qc_finished)
        self._single_worker.error.connect(self.on_single_qc_error)
        self._single_worker.aborted.connect(self.on_single_qc_aborted)

        self._single_worker.finished.connect(self._single_thread.quit)
        self._single_worker.finished.connect(self._single_worker.deleteLater)
        self._single_worker.error.connect(self._single_thread.quit)
        self._single_worker.aborted.connect(self._single_thread.quit)
        self._single_thread.finished.connect(self._single_thread.deleteLater)

        self._single_thread.start()

    def _abort_single_worker(self):
        if self._single_worker is not None:
            self._single_worker.abort()
        if self._single_thread is not None and self._single_thread.isRunning():
            self._single_thread.quit()

    def on_single_qc_finished(self, result: QCResult):
        ch = result.channel
        self._attach_sorter_data(result)
        self.qc_results[ch] = result
        self._qc_view.show_result(result)
        self._channel_list.update_channel_result(ch, result)

        label = f"CH {ch}: {result.n_lh} LH, {result.n_soup} soup, {result.n_uncertain} uncertain"
        if result.miss_rate is not None:
            label += f", miss={result.miss_rate:.1%}"
        self._status_bar.showMessage(label)

    def on_single_qc_error(self, msg: str):
        self._status_bar.showMessage(f"QC failed: {msg}")
        self._qc_view.show_error(msg)

    def on_single_qc_aborted(self):
        self._status_bar.showMessage(f"CH {self.current_channel}: QC cancelled.")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _n_sorter_spikes_for_channel(self, ch: int) -> int:
        """Return count of pooled KS spikes on electrode ch, or -1 if unknown."""
        if not self.sorter_spike_times:
            return -1
        times = self.sorter_spike_times.get(ch, None)
        if times is None:
            return -1
        return int(len(times))

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

    def closeEvent(self, event):
        self._abort_loader()
        if self._batch_worker:
            self._batch_worker.abort()
        if self._single_worker:
            self._single_worker.abort()
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.quit()
        if self._single_thread and self._single_thread.isRunning():
            self._single_thread.quit()
        event.accept()