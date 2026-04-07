"""
qc_worker.py — Background worker for QC pipeline execution.
"""
from __future__ import annotations
from qtpy.QtCore import QObject, Signal
from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS
from core.result_types import QCResult


class QCWorker(QObject):
    """Runs QC pipeline on a background QThread."""

    progress = Signal(str)  # status message for status bar
    finished = Signal(object)  # emits QCResult on success
    error = Signal(str)  # emits error message string

    def __init__(
        self,
        raw_data,
        channel: int,
        n_sorter_spikes: int,
        params: dict,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.channel = channel
        self.n_sorter_spikes = n_sorter_spikes
        self.params = params if params else dict(DEFAULT_PARAMS)
        self._abort = False

    def abort(self):
        """Signal the worker to stop. Checked between pipeline steps."""
        self._abort = True

    def run(self):
        """Run pipeline steps 1-4 sequentially on background thread."""
        try:
            self.progress.emit(f"Step 1/4: Valley detection on CH {self.channel}...")
            if self._abort:
                return

            result = run_qc_pipeline(
                raw_data=self.raw_data,
                ch=self.channel,
                n_sorter_spikes=self.n_sorter_spikes,
                params=self.params,
            )

            self.progress.emit(f"Step 4/4 complete — QC done for CH {self.channel}.")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"QC failed on CH {self.channel}: {e}")
