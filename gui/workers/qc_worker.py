"""
qc_worker.py — Background worker for QC pipeline execution.
"""
from __future__ import annotations
from qtpy.QtCore import QObject, Signal
from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS


class QCWorker(QObject):
    """Runs QC pipeline on a background QThread."""

    progress = Signal(str)  # status message for status bar
    finished = Signal(object)  # emits QCResult on success
    error = Signal(str)  # emits error message string
    aborted = Signal()

    def __init__(
        self,
        raw_data,
        channel: int,
        n_sorter_spikes: int,
        params: dict,
        fs: float = 20000.0,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.channel = channel
        self.n_sorter_spikes = n_sorter_spikes
        self.params = params if params else dict(DEFAULT_PARAMS)
        self.fs = fs
        self._abort = False

    def abort(self):
        """Signal the worker to stop. Checked between pipeline steps."""
        self._abort = True

    def run(self):
        """Run the complete QC pipeline on background thread."""
        try:
            self.progress.emit(f"Running QC on CH {self.channel}...")
            if self._abort:
                self.aborted.emit()
                return

            result = run_qc_pipeline(
                raw_data=self.raw_data,
                ch=self.channel,
                n_sorter_spikes=self.n_sorter_spikes,
                params=self.params,
                fs=self.fs,
            )

            if self._abort:
                self.aborted.emit()
                return

            self.progress.emit(f"QC done for CH {self.channel}.")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"QC failed on CH {self.channel}: {e}")