"""
batch_qc_worker.py — Runs QC pipeline on all channels concurrently using a thread pool.
"""
from __future__ import annotations
import os
import traceback
from qtpy.QtCore import QObject, Signal, QRunnable, QThreadPool

# PREVENT OPENMP DEADLOCKS: Force NumPy/SciPy/Scikit-Learn to use 1 thread per task.
# This must happen before the pipeline runs, otherwise 8 concurrent KMeans calls
# will spawn 64+ threads and crash the C++ backend.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS
from core.result_types import QCResult


class QCChannelTaskSignals(QObject):
    """Signals for a single QC channel task running in a thread pool."""
    result_ready = Signal(object)  # emits QCResult
    error = Signal(dict)           # emits dict with ch and error message


class QCChannelTask(QRunnable):
    """A worker task that runs QC on a single channel."""
    
    def __init__(self, raw_data, ch: int, n_sorter: int, params: dict, fs: float):
        super().__init__()
        self.raw_data = raw_data
        self.ch = ch
        self.n_sorter = n_sorter
        self.params = params
        self.fs = fs
        self.signals = QCChannelTaskSignals()

    def run(self):
        try:
            result = run_qc_pipeline(
                raw_data=self.raw_data,
                ch=self.ch,
                n_sorter_spikes=self.n_sorter,
                params=self.params,
                fs=self.fs,
            )
            self.signals.result_ready.emit(result)
        except Exception as e:
            # Capture the full traceback
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit({"ch": self.ch, "msg": err_msg})


class BatchQCWorker(QObject):
    """Dispatches QC tasks to a QThreadPool and aggregates results sequentially for the UI."""

    progress = Signal(str, int, int)  # (message, current_channel, total_channels)
    channel_done = Signal(object)     # emits QCResult for each completed channel
    finished = Signal(dict)           # emits {ch: QCResult} dict when all done
    error = Signal(str)               # emits error message string (Fatal errors only)
    aborted = Signal()

    def __init__(
        self,
        raw_data,
        params: dict,
        sorter_spike_times: dict = None,
        fs: float = 20000.0,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.params = params if params else dict(DEFAULT_PARAMS)
        self.sorter_spike_times = sorter_spike_times or {}
        self.fs = fs
        
        self._abort = False
        self._completed_count = 0
        self._total_channels = 0
        self._tasks = []  # <--- CRITICAL: Prevents Python Garbage Collector from killing tasks
        
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)  

    def abort(self):
        """Signal the worker to stop and clear pending tasks."""
        self._abort = True
        self._pool.clear()
        self.aborted.emit()

    def run(self):
        """Queue all channels into the thread pool."""
        try:
            _, n_channels = self.raw_data.shape
            self._total_channels = n_channels
            self._completed_count = 0
            self._tasks.clear()

            for ch in range(n_channels):
                if self._abort:
                    return

                n_sorter = len(self.sorter_spike_times.get(ch, [])) if self.sorter_spike_times else -1

                task = QCChannelTask(
                    raw_data=self.raw_data,
                    ch=ch,
                    n_sorter=n_sorter,
                    params=self.params,
                    fs=self.fs
                )
                
                task.signals.result_ready.connect(self._on_task_result)
                task.signals.error.connect(self._on_task_error)
                
                # Keep a Python reference so the Garbage Collector doesn't eat it!
                self._tasks.append(task)
                
                self._pool.start(task)

        except Exception as e:
            self.error.emit(f"Batch QC initialization failed: {e}")

    def _on_task_result(self, result: QCResult):
        """Handle successful completion of a single channel."""
        if self._abort: return
            
        self._completed_count += 1
        self.channel_done.emit(result)
        
        self.progress.emit(
            f"Running QC... ({self._completed_count}/{self._total_channels})",
            self._completed_count,
            self._total_channels,
        )
        self._check_finished()

    def _on_task_error(self, err_info: dict):
        """Handle failure of a single channel without crashing the UI."""
        if self._abort: return
        
        ch = err_info["ch"]
        msg = err_info["msg"]
        print(f"Skipping CH {ch} due to error:\n{msg}")
        
        self._completed_count += 1
        self.progress.emit(
            f"Running QC... ({self._completed_count}/{self._total_channels}) [CH {ch} failed]",
            self._completed_count,
            self._total_channels,
        )
        self._check_finished()
        
    def _check_finished(self):
        """Emit finished signal if all tasks are complete."""
        if self._completed_count == self._total_channels:
            # Clean up the task list references to free memory
            self._tasks.clear()
            self.finished.emit({"total": self._total_channels})