"""
qc_worker.py — Background worker for QC pipeline execution.
"""
from __future__ import annotations
from qtpy.QtCore import QObject, Signal
from core.lh_qc_pipeline import (
    DEFAULT_PARAMS,
    run_valley_detection,
    run_snippet_extraction,
    run_pca_kmeans,
    run_bltr_labeling,
)
from core.result_types import QCResult


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
            # Step 1
            self.progress.emit(f"Step 1/4: Valley detection on CH {self.channel}...")
            if self._abort:
                self.aborted.emit()
                return
            valley = run_valley_detection(self.raw_data, self.channel, self.params)

            # Step 2
            self.progress.emit(f"Step 2/4: Snippet extraction on CH {self.channel}...")
            if self._abort:
                self.aborted.emit()
                return
            snippets = run_snippet_extraction(self.raw_data, valley.all_times, self.params)

            # Step 3
            self.progress.emit(f"Step 3/4: PCA + KMeans on CH {self.channel}...")
            if self._abort:
                self.aborted.emit()
                return
            pca_km = run_pca_kmeans(snippets, self.channel, self.params)

            # Step 4
            self.progress.emit(f"Step 4/4: BL/TR labeling on CH {self.channel}...")
            if self._abort:
                self.aborted.emit()
                return
            bltr = run_bltr_labeling(snippets, valley, self.params)

            result = QCResult(
                channel=self.channel,
                n_sorter_spikes=self.n_sorter_spikes,
                valley=valley,
                snippets=snippets,
                pca_km=pca_km,
                bltr=bltr,
            )

            self.progress.emit(f"Step 4/4 complete — QC done for CH {self.channel}.")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"QC failed on CH {self.channel}: {e}")
