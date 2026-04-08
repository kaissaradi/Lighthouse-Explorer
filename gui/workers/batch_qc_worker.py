"""
batch_qc_worker.py — Runs QC pipeline on all channels in the background.
"""
from __future__ import annotations
from qtpy.QtCore import QObject, Signal
from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS
from core.result_types import QCResult


class BatchQCWorker(QObject):
    """Runs QC pipeline on all channels on a background QThread."""

    progress = Signal(str, int, int)  # (message, current_channel, total_channels)
    channel_done = Signal(object)  # emits QCResult for each completed channel
    finished = Signal(dict)  # emits {ch: QCResult} dict when all done
    error = Signal(str)  # emits error message string
    aborted = Signal()

    def __init__(
        self,
        raw_data,
        params: dict,
        sorter_spike_times: dict = None,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.params = params if params else dict(DEFAULT_PARAMS)
        self.sorter_spike_times = sorter_spike_times or {}
        self._abort = False

    def abort(self):
        """Signal the worker to stop."""
        self._abort = True

    def run(self):
        """Run pipeline on all channels sequentially."""
        try:
            _, n_channels = self.raw_data.shape

            for ch in range(n_channels):
                if self._abort:
                    self.aborted.emit()
                    return

                self.progress.emit(
                    f"Running QC on CH {ch} ({ch+1}/{n_channels})...",
                    ch,
                    n_channels,
                )

                n_sorter = len(self.sorter_spike_times.get(ch, []))

                result = run_qc_pipeline(
                    raw_data=self.raw_data,
                    ch=ch,
                    n_sorter_spikes=n_sorter,
                    params=self.params,
                )

                # Emit immediately — don't accumulate in memory
                self.channel_done.emit(result)

            # Signal completion with count only
            self.finished.emit({"total": n_channels})

        except Exception as e:
            self.error.emit(f"Batch QC failed: {e}")
