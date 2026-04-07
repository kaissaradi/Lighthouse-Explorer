"""
main_window.py — Top-level QMainWindow. Orchestrates panels and QC workflow.
"""
from __future__ import annotations
from typing import Optional
from qtpy.QtWidgets import QMainWindow, QSplitter, QStatusBar, QProgressBar
from qtpy.QtCore import Qt, QThread
from .panels.load_panel import LoadPanel
from .panels.array_map_panel import ArrayMapPanel
from .panels.qc_view_panel import QCViewPanel
from .workers.qc_worker import QCWorker
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
        self.ei_positions = None  # [C, 2]
        self.sorter_spike_times: dict = {}
        self.qc_results: dict = {}
        self.current_channel: Optional[int] = None
        self.lh_params: dict = dict(DEFAULT_PARAMS)
        self.default_dat = default_dat
        self.default_n_channels = default_n_channels

        # Worker / thread
        self._qc_thread: Optional[QThread] = None
        self._qc_worker: Optional[QCWorker] = None

        self._build_ui()
        self._connect_signals()

        # Apply CLI defaults
        if self.default_dat or self.default_n_channels:
            self._load_panel.set_defaults(self.default_dat, self.default_n_channels)

    # ── UI setup ─────────────────────────────────────────────────

    def _build_ui(self):
        """
        Layout:
        ┌─ LoadPanel (250px) ─┬─ ArrayMapPanel (350px) ─┬─ QCViewPanel (fill) ─┐
        └─────────────────────┴───────────────────────────┴──────────────────────┘
        Status bar at bottom
        """
        self._load_panel = LoadPanel(self)
        self._array_map = ArrayMapPanel(self)
        self._qc_view = QCViewPanel(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._load_panel)
        splitter.addWidget(self._array_map)
        splitter.addWidget(self._qc_view)
        splitter.setSizes([250, 350, 1000])
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
        self._array_map.channel_selected.connect(self.on_channel_selected)

    # ── Data loading ─────────────────────────────────────────────

    def on_load_requested(self, params: dict):
        """Load raw data, channel map, update panels."""
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

            self.ei_positions = loader.load_channel_map(
                params.get("map_path"), n_ch
            )
            self._array_map.set_array(self.ei_positions)

            # Update params for QC runs
            self.lh_params.update({
                "min_valid_count": params.get("min_valid_count", 300),
                "min_bl_bulk": params.get("min_bl_bulk", 0.70),
                "max_snippets": params.get("max_snippets", 5000),
            })

            self._status_bar.showMessage(
                f"Loaded {self.raw_data.shape[0]} samples × {n_ch} channels."
            )
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
                fr = loader.compute_channel_firing_rates(
                    self.sorter_spike_times, duration_s
                )
                self._array_map.set_firing_rates(fr)
                self._status_bar.showMessage(
                    f"Loaded sorter: {len(self.sorter_spike_times)} units across {len(fr)} channels."
                )
            else:
                self._status_bar.showMessage("No sorter spike times found.")
        except Exception as e:
            self._status_bar.showMessage(f"Sorter load failed: {e}")

    # ── QC run lifecycle ─────────────────────────────────────────

    def on_channel_selected(self, ch: int):
        """User clicked a channel on the array map."""
        if self.raw_data is None:
            self._status_bar.showMessage("Load a recording first.")
            return

        self.current_channel = ch
        self._array_map.set_selected_channel(ch)

        # Check cache
        if ch in self.qc_results:
            self._qc_view.show_result(self.qc_results[ch])
            self._status_bar.showMessage(f"CH {ch}: cached result displayed.")
            return

        # Abort any running worker
        self._abort_current_worker()

        # Start new QC run
        self._qc_view.show_loading(ch)
        self._start_qc_worker(ch)

    def _start_qc_worker(self, ch: int):
        """Spin up QThread + QCWorker for channel ch."""
        n_sorter = self._n_sorter_spikes_for_channel(ch)

        self._qc_thread = QThread()
        self._qc_worker = QCWorker(
            raw_data=self.raw_data,
            channel=ch,
            n_sorter_spikes=n_sorter,
            params=self.lh_params,
        )
        self._qc_worker.moveToThread(self._qc_thread)
        self._qc_thread.started.connect(self._qc_worker.run)
        self._qc_worker.progress.connect(self._status_bar.showMessage)
        self._qc_worker.finished.connect(self.on_qc_finished)
        self._qc_worker.error.connect(self.on_qc_error)
        self._qc_thread.finished.connect(self._qc_thread.deleteLater)

        self._qc_thread.start()

    def _abort_current_worker(self):
        """Abort worker and clean up thread."""
        if self._qc_worker is not None:
            self._qc_worker.abort()
        if self._qc_thread is not None:
            self._qc_thread.quit()
            self._qc_thread.wait(3000)
        self._qc_thread = None
        self._qc_worker = None

    def on_qc_finished(self, result: QCResult):
        """Slot for QCWorker.finished."""
        ch = result.channel
        self.qc_results[ch] = result
        self._qc_view.show_result(result)
        mr = result.miss_rate
        self._array_map.set_qc_result_color(ch, mr)

        label = f"CH {ch}: {result.n_lh} LH, {result.n_soup} soup, {result.n_uncertain} uncertain"
        if mr is not None:
            label += f", miss={mr:.1%}"
        self._status_bar.showMessage(label)

        # Cleanup thread
        self._qc_thread = None
        self._qc_worker = None

    def on_qc_error(self, msg: str):
        """Show error in status bar and QC view."""
        self._status_bar.showMessage(msg)
        self._qc_view.show_error(msg)
        self._qc_thread = None
        self._qc_worker = None

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
        """Abort worker, accept close."""
        self._abort_current_worker()
        event.accept()
