"""
main_window.py — Top-level QMainWindow. Orchestrates panels and QC workflow.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from qtpy.QtWidgets import QMainWindow, QSplitter, QStatusBar, QProgressBar, QPushButton, QMessageBox
from qtpy.QtCore import Qt, QThread
from .panels.load_panel import LoadPanel
from .panels.array_map_panel import ArrayMapPanel
from .panels.qc_view_panel import QCViewPanel
from .workers.qc_worker import QCWorker
from .workers.batch_qc_worker import BatchQCWorker
from core import loader
from core.lh_qc_pipeline import DEFAULT_PARAMS
from core.result_types import QCResult


class MainWindow(QMainWindow):
    """Main window for Lighthouse QC application."""

    def __init__(self, default_dat=None, default_n_channels=None):
        super().__init__()
        self.setWindowTitle("Lighthouse QC")
        self.setGeometry(100, 100, 1600, 900)

        # State
        self.raw_data = None  # np.memmap [T, C]
        self.sorter_spike_times: dict = {}
        self.qc_results: dict = {}
        self.current_channel: Optional[int] = None
        self.lh_params: dict = dict(DEFAULT_PARAMS)
        self.default_dat = default_dat
        self.default_n_channels = default_n_channels

        # Workers / threads
        self._single_thread: Optional[QThread] = None
        self._single_worker: Optional[QCWorker] = None
        self._batch_thread: Optional[QThread] = None
        self._batch_worker: Optional[BatchQCWorker] = None
        self._running_batch = False

        self._build_ui()
        self._connect_signals()

        # Apply CLI defaults
        if self.default_dat or self.default_n_channels:
            self._load_panel.set_defaults(self.default_dat, self.default_n_channels)

    # ── UI setup ─────────────────────────────────────────────────

    def _build_ui(self):
        """
        Layout:
        ┌─ LoadPanel (250px) ─┬─ Channel List (250px) ─┬─ QCViewPanel (fill) ─┐
        └─────────────────────┴────────────────────────┴──────────────────────┘
        Status bar at bottom
        """
        self._load_panel = LoadPanel(self)
        self._channel_list = ArrayMapPanel(self)
        self._qc_view = QCViewPanel(self)

        # Run All button
        self._run_all_btn = QPushButton("▶ Run All Channels")
        self._run_all_btn.setStyleSheet(
            "QPushButton { background-color: #2E6DD4; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #4A8BEF; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self._run_all_btn.clicked.connect(self._on_run_all_clicked)
        self._load_panel.layout().insertWidget(self._load_panel.layout().count() - 1, self._run_all_btn)

        # Cancel button (hidden initially)
        self._cancel_btn = QPushButton("⏹ Cancel")
        self._cancel_btn.setStyleSheet(
            "QPushButton { background-color: #C62828; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #E53935; }"
        )
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._load_panel.layout().insertWidget(self._load_panel.layout().count() - 1, self._cancel_btn)

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

    def _connect_signals(self):
        self._load_panel.load_requested.connect(self.on_load_requested)
        self._load_panel.sorter_load_requested.connect(self.on_sorter_load_requested)
        self._channel_list.channel_selected.connect(self.on_channel_selected)

    # ── Data loading ─────────────────────────────────────────────

    def on_load_requested(self, params: dict):
        """Load raw data, update panels, auto-start batch QC."""
        try:
            self._load_panel.set_loading_state(True)
            self._status_bar.showMessage("Loading recording…")

            dat_path = params["dat_path"]
            if not dat_path:
                self._status_bar.showMessage("No .dat file specified.")
                return

            n_ch = params["n_channels"]
            self.raw_data = loader.load_raw_readonly(
                dat_path,
                n_channels=n_ch,
                dtype=params.get("dtype", "int16"),
                start_min=params.get("start_min", 0.0),
                duration_min=params.get("duration_min", None),
                fs=params.get("fs", 20_000),
            )

            # Build channel list
            self._channel_list.set_array(np.arange(n_ch).reshape(-1, 1))

            # Update params for QC runs
            self.lh_params.update({
                "min_valid_count": params.get("min_valid_count", 300),
                "min_bl_bulk": params.get("min_bl_bulk", 0.70),
                "max_snippets": params.get("max_snippets", 5000),
            })

            self._status_bar.showMessage(
                f"Loaded {self.raw_data.shape[0]} samples × {n_ch} channels. Starting batch QC…"
            )

            # Auto-start batch QC on all channels
            self._start_batch_qc()

        except Exception as e:
            self._status_bar.showMessage(f"Load failed: {e}")
            self._qc_view.show_error(f"Load failed: {e}")
        finally:
            self._load_panel.set_loading_state(False)

    def on_sorter_load_requested(self, path: str):
        """Load sorter spike times for miss-rate QC."""
        try:
            self._status_bar.showMessage("Loading sorter output…")
            n_ch = self.raw_data.shape[1] if self.raw_data is not None else 512
            self.sorter_spike_times = loader.load_sorter_spike_times(path, n_ch)

            if self.sorter_spike_times:
                duration_s = self.raw_data.shape[0] / self.lh_params.get("fs", 20_000)
                loader.compute_channel_firing_rates(
                    self.sorter_spike_times, duration_s
                )
                self._status_bar.showMessage(
                    f"Loaded sorter: {len(self.sorter_spike_times)} units."
                )
            else:
                self._status_bar.showMessage("No sorter spike times found.")
        except Exception as e:
            self._status_bar.showMessage(f"Sorter load failed: {e}")

    # ── Batch QC lifecycle ───────────────────────────────────────

    def _start_batch_qc(self):
        """Start running QC on all channels."""
        if self.raw_data is None:
            return

        # Abort any existing batch first
        self._abort_batch_worker()

        self._running_batch = True
        self._run_all_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
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

        # Asynchronous cleanup via Qt signals
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.error.connect(self._batch_thread.quit)
        self._batch_worker.aborted.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)

        self._batch_thread.start()

    def _on_batch_progress(self, msg: str, current: int, total: int):
        """Update progress during batch QC."""
        self._status_bar.showMessage(msg)
        self._channel_list.set_progress(current + 1, total, msg)

    def _on_batch_channel_done(self, result: QCResult):
        """A single channel QC completed during batch run."""
        ch = result.channel
        self.qc_results[ch] = result
        self._channel_list.update_channel_result(ch, result)

        # If this is the currently selected channel, show it
        if self.current_channel == ch:
            self._qc_view.show_result(result)

    def _on_batch_finished(self, results: dict):
        """All channels completed."""
        self._running_batch = False
        self._run_all_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._channel_list.hide_progress()

        # Count LH channels from our own dict
        lh_count = sum(1 for r in self.qc_results.values() if r.n_lh > 0)
        total = results.get("total", len(self.qc_results))
        self._status_bar.showMessage(
            f"Batch QC complete: {lh_count}/{total} channels with LH spikes found."
        )

        # Switch to "LH Found" view by default
        self._channel_list._view_combo.setCurrentText("LH Found")

        # Show first LH channel
        for ch, result in sorted(self.qc_results.items()):
            if result.n_lh > 0:
                self.current_channel = ch
                self._channel_list.set_selected_channel(ch)
                self._qc_view.show_result(result)
                break

    def _on_batch_error(self, msg: str):
        """Batch QC failed."""
        self._running_batch = False
        self._run_all_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._channel_list.hide_progress()
        self._status_bar.showMessage(f"Batch QC failed: {msg}")
        self._qc_view.show_error(msg)

    def _on_batch_aborted(self):
        """Batch QC was cancelled."""
        self._running_batch = False
        self._run_all_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._channel_list.hide_progress()
        completed = len(self.qc_results)
        self._status_bar.showMessage(
            f"Batch QC cancelled. {completed} channels completed."
        )

    # ── Single QC run lifecycle ──────────────────────────────────

    def _on_run_all_clicked(self):
        """Manually trigger batch QC on all channels."""
        if self.raw_data is None:
            self._status_bar.showMessage("Load a recording first.")
            return

        # Confirm if some channels already have results
        existing = len(self.qc_results)
        total = self.raw_data.shape[1]
        if existing > 0:
            reply = QMessageBox.question(
                self,
                "Re-run All?",
                f"{existing}/{total} channels already have results. Re-run all?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self.qc_results.clear()
        self._start_batch_qc()

    def _on_cancel_clicked(self):
        """Cancel running batch QC."""
        self._abort_batch_worker()

    def _abort_batch_worker(self):
        """Gracefully ask the batch worker to stop."""
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

        # Check cache
        if ch in self.qc_results:
            self._qc_view.show_result(self.qc_results[ch])
            self._status_bar.showMessage(f"CH {ch}: cached result displayed.")
            return

        # If batch is running, show loading placeholder
        if self._running_batch:
            self._qc_view.show_loading(ch)
            self._status_bar.showMessage(f"CH {ch}: queued for batch QC…")
            return

        # Run single channel QC
        self._abort_single_worker()
        self._qc_view.show_loading(ch)
        self._start_single_qc_worker(ch)

    def _start_single_qc_worker(self, ch: int):
        """Spin up QThread + QCWorker for a single channel."""
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

        # Asynchronous cleanup via Qt signals
        self._single_worker.finished.connect(self._single_thread.quit)
        self._single_worker.finished.connect(self._single_worker.deleteLater)
        self._single_worker.error.connect(self._single_thread.quit)
        self._single_worker.aborted.connect(self._single_thread.quit)
        self._single_thread.finished.connect(self._single_thread.deleteLater)

        self._single_thread.start()

    def _abort_single_worker(self):
        """Gracefully ask the single worker to stop."""
        if self._single_worker is not None:
            self._single_worker.abort()
        if self._single_thread is not None and self._single_thread.isRunning():
            self._single_thread.quit()

    def on_single_qc_finished(self, result: QCResult):
        """Slot for single QCWorker.finished."""
        ch = result.channel
        self.qc_results[ch] = result
        self._qc_view.show_result(result)
        self._channel_list.update_channel_result(ch, result)

        label = f"CH {ch}: {result.n_lh} LH, {result.n_soup} soup, {result.n_uncertain} uncertain"
        if result.miss_rate is not None:
            label += f", miss={result.miss_rate:.1%}"
        self._status_bar.showMessage(label)

    def on_single_qc_error(self, msg: str):
        """Show error for single QC run."""
        self._status_bar.showMessage(f"QC failed: {msg}")
        self._qc_view.show_error(msg)

    def on_single_qc_aborted(self):
        """Single QC run was cancelled."""
        self._status_bar.showMessage(f"CH {self.current_channel}: QC cancelled.")

    # ── Utilities ────────────────────────────────────────────────

    def _n_sorter_spikes_for_channel(self, ch: int) -> int:
        """Return count of sorter spikes on channel ch, or -1 if unknown."""
        if not self.sorter_spike_times:
            return -1
        times = self.sorter_spike_times.get(ch, None)
        if times is None:
            return -1
        if isinstance(times, list):
            return len(times)
        return int(len(times))

    def closeEvent(self, event):
        """Abort workers and request async cleanup, then accept close."""
        if self._batch_worker:
            self._batch_worker.abort()
        if self._single_worker:
            self._single_worker.abort()
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.quit()
        if self._single_thread and self._single_thread.isRunning():
            self._single_thread.quit()
        event.accept()
