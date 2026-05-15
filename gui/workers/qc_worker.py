"""
qc_worker.py — Unified worker module for all background tasks.

Contains:
  • Native threading helpers (configure_native_thread_environment, native_thread_limits)
  • QCChannelTaskSignals / QCChannelTask — the atomic QRunnable unit
  • BatchQCWorker — legacy dispatcher (used directly by some tests)
  • TaskManager — single-channel + batch dispatch via shared QThreadPool
  • LoaderWorker — file I/O + baseline subtraction on a background QThread

Both single-channel QC and batch QC now route through the same QCChannelTask
(QRunnable) dispatched onto a shared QThreadPool.  This eliminates the old
QObject/QThread boilerplate while retaining the memory-safety guardrails:

  • self._tasks list keeps Python references alive until completion/abort.
  • .deleteLater() is called on signal objects via _cleanup_task().
  • native_thread_limits(1) is enforced inside each QCChannelTask.run().
"""
from __future__ import annotations

import os
import traceback
from contextlib import nullcontext
from qtpy.QtCore import QObject, Signal, QRunnable, QThreadPool

from core import loader
from lh_deps.axolotl_utils_ram import (
    compute_baselines_int16_deriv_robust,
    subtract_segment_baselines_int16,
)


# ── Native threading limits ─────────────────────────────────────────────────
# Inlined from the former core/native_threading.py to reduce file count.

NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_native_thread_environment(threads: int = 1) -> None:
    """
    Set process-level limits before NumPy/SciPy/scikit-learn are imported.

    These variables only reliably affect native libraries before they are loaded,
    so channel workers also use ``native_thread_limits`` at runtime.
    """
    value = str(max(1, int(threads)))
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = value


def native_thread_limits(threads: int = 1):
    """Return a context manager that limits already-loaded BLAS/OpenMP pools."""
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()

    return threadpool_limits(limits=max(1, int(threads)))


# PREVENT OPENMP DEADLOCKS: Force NumPy/SciPy/Scikit-Learn to use 1 thread per task.
# These defaults help when this module is imported early. Each task also uses
# threadpoolctl at runtime because NumPy/scikit-learn may already be loaded.
configure_native_thread_environment()

from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS
from core.result_types import QCResult


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_BATCH_MAX_WORKERS = 4
DEFAULT_NATIVE_THREADS_PER_TASK = 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _positive_int(value, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _resolve_worker_count(params: dict, explicit_count=None) -> int:
    if explicit_count is not None:
        return _positive_int(explicit_count, DEFAULT_BATCH_MAX_WORKERS)

    if params and params.get("batch_max_workers") is not None:
        return _positive_int(params.get("batch_max_workers"), DEFAULT_BATCH_MAX_WORKERS)

    return _positive_int(
        os.environ.get("LIGHTHOUSE_QC_BATCH_THREADS"),
        DEFAULT_BATCH_MAX_WORKERS,
    )


# ── QCChannelTask primitives ────────────────────────────────────────────────

class QCChannelTaskSignals(QObject):
    """Signals for a single QC channel task running in a thread pool."""
    result_ready = Signal(object)  # emits QCResult
    error = Signal(dict)           # emits dict with ch and error message


class QCChannelTask(QRunnable):
    """A worker task that runs QC on a single channel."""
    
    def __init__(
        self,
        raw_data,
        ch: int,
        n_sorter: int,
        params: dict,
        fs: float,
        native_threads: int = DEFAULT_NATIVE_THREADS_PER_TASK,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.ch = ch
        self.n_sorter = n_sorter
        self.params = params
        self.fs = fs
        self.native_threads = _positive_int(native_threads, DEFAULT_NATIVE_THREADS_PER_TASK)
        self.signals = QCChannelTaskSignals()

    def run(self):
        try:
            with native_thread_limits(self.native_threads):
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


# ── BatchQCWorker (legacy dispatcher) ───────────────────────────────────────

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
        max_workers: int | None = None,
        native_threads_per_task: int = DEFAULT_NATIVE_THREADS_PER_TASK,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.params = params if params else dict(DEFAULT_PARAMS)
        self.sorter_spike_times = sorter_spike_times or {}
        self.fs = fs
        self.max_workers = _resolve_worker_count(self.params, max_workers)
        self.native_threads_per_task = _positive_int(
            native_threads_per_task,
            DEFAULT_NATIVE_THREADS_PER_TASK,
        )
        
        self._abort = False
        self._completed_count = 0
        self._total_channels = 0
        self._tasks = []  # <--- CRITICAL: Prevents Python Garbage Collector from killing tasks
        
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(self.max_workers)

    def abort(self):
        """Signal the worker to stop and clear pending tasks."""
        self._abort = True
        self._pool.clear()
        self._tasks.clear()  # <--- Ensure tasks are cleared on abort too
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
                    fs=self.fs,
                    native_threads=self.native_threads_per_task,
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


# ── TaskManager (unified dispatch) ──────────────────────────────────────────

class TaskManager(QObject):
    """
    Unified dispatch for single-channel and batch QC.

    Signals
    -------
    single_result : QCResult
        Emitted when a single-channel QC task completes successfully.
    single_error : str
        Emitted when a single-channel QC task fails.
    single_progress : str
        Status message for the status bar during single-channel QC.
    batch_progress : (str, int, int)
        (message, completed_count, total_channels) during batch QC.
    batch_channel_done : QCResult
        Emitted for each channel that finishes during a batch run.
    batch_finished : dict
        Emitted when the entire batch is complete: {"total": N}.
    batch_error : str
        Emitted on fatal batch-level errors.
    batch_aborted : (no args)
        Emitted when a batch run is cancelled.
    """

    # ── Single-channel signals ──────────────────────────────────────────────
    single_result = Signal(object)   # QCResult
    single_error = Signal(str)
    single_progress = Signal(str)

    # ── Batch signals (same semantics as the old BatchQCWorker) ─────────────
    batch_progress = Signal(str, int, int)
    batch_channel_done = Signal(object)   # QCResult
    batch_finished = Signal(dict)
    batch_error = Signal(str)
    batch_aborted = Signal()

    def __init__(
        self,
        max_workers: int | None = None,
        native_threads_per_task: int = DEFAULT_NATIVE_THREADS_PER_TASK,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.max_workers = _resolve_worker_count({}, max_workers)
        self.native_threads_per_task = _positive_int(
            native_threads_per_task,
            DEFAULT_NATIVE_THREADS_PER_TASK,
        )

        # ── Internal state ──────────────────────────────────────────────────
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(self.max_workers)

        # CRITICAL: Prevents Python GC from killing in-flight QRunnable tasks.
        self._tasks: list[QCChannelTask] = []

        self._batch_abort = False
        self._batch_completed = 0
        self._batch_total = 0
        self._running_batch = False

    # ── Single-channel QC ───────────────────────────────────────────────────

    def start_single(
        self,
        raw_data,
        channel: int,
        n_sorter_spikes: int,
        params: dict,
        fs: float = 20000.0,
    ) -> None:
        """
        Submit a single-channel QC task to the thread pool.

        Results are emitted via ``single_result`` / ``single_error``.
        """
        self.single_progress.emit(f"Running QC on CH {channel}...")

        task = QCChannelTask(
            raw_data=raw_data,
            ch=channel,
            n_sorter=n_sorter_spikes,
            params=params if params else dict(DEFAULT_PARAMS),
            fs=fs,
            native_threads=self.native_threads_per_task,
        )

        task.signals.result_ready.connect(self._on_single_result)
        task.signals.error.connect(self._on_single_error)

        # Prevent GC, then dispatch.
        self._tasks.append(task)
        self._pool.start(task)

    def abort_single(self) -> None:
        """Cancel any pending single-channel tasks (best-effort)."""
        # QThreadPool.clear() removes queued-but-not-started tasks.
        # In-flight tasks will complete — acceptable for single-channel.
        pass  # No separate abort flag needed; single tasks are fire-and-forget.

    def _on_single_result(self, result: QCResult) -> None:
        self.single_progress.emit(f"QC done for CH {result.channel}.")
        self.single_result.emit(result)
        self._cleanup_task_by_channel(result.channel)

    def _on_single_error(self, err_info: dict) -> None:
        ch = err_info["ch"]
        msg = err_info["msg"]
        self.single_error.emit(f"QC failed on CH {ch}: {msg}")
        self._cleanup_task_by_channel(ch)

    # ── Batch QC ────────────────────────────────────────────────────────────

    @property
    def is_batch_running(self) -> bool:
        return self._running_batch

    def start_batch(
        self,
        raw_data,
        params: dict,
        sorter_spike_times: dict | None = None,
        fs: float = 20000.0,
    ) -> None:
        """
        Submit QC tasks for every channel in ``raw_data`` to the thread pool.

        Progress is reported via ``batch_progress`` and ``batch_channel_done``.
        On completion, ``batch_finished`` is emitted with ``{"total": N}``.
        """
        try:
            self.abort_batch()  # clean up any previous batch

            self._batch_abort = False
            self._running_batch = True
            sorter_spike_times = sorter_spike_times or {}
            params = params if params else dict(DEFAULT_PARAMS)

            _, n_channels = raw_data.shape
            self._batch_total = n_channels
            self._batch_completed = 0
            self._tasks.clear()

            for ch in range(n_channels):
                if self._batch_abort:
                    return

                n_sorter = (
                    len(sorter_spike_times.get(ch, []))
                    if sorter_spike_times
                    else -1
                )

                task = QCChannelTask(
                    raw_data=raw_data,
                    ch=ch,
                    n_sorter=n_sorter,
                    params=params,
                    fs=fs,
                    native_threads=self.native_threads_per_task,
                )

                task.signals.result_ready.connect(self._on_batch_task_result)
                task.signals.error.connect(self._on_batch_task_error)

                # CRITICAL: prevent GC from killing active tasks.
                self._tasks.append(task)
                self._pool.start(task)

        except Exception as e:
            self._running_batch = False
            self.batch_error.emit(f"Batch QC initialization failed: {e}")

    def abort_batch(self) -> None:
        """Cancel the current batch run and clean up references."""
        if not self._running_batch:
            return
        self._batch_abort = True
        self._pool.clear()
        self._tasks.clear()
        self._running_batch = False
        self.batch_aborted.emit()

    def _on_batch_task_result(self, result: QCResult) -> None:
        if self._batch_abort:
            return
        self._batch_completed += 1
        self.batch_channel_done.emit(result)
        self.batch_progress.emit(
            f"Running QC... ({self._batch_completed}/{self._batch_total})",
            self._batch_completed,
            self._batch_total,
        )
        self._check_batch_finished()

    def _on_batch_task_error(self, err_info: dict) -> None:
        if self._batch_abort:
            return
        ch = err_info["ch"]
        msg = err_info["msg"]
        print(f"Skipping CH {ch} due to error:\n{msg}")
        self._batch_completed += 1
        self.batch_progress.emit(
            f"Running QC... ({self._batch_completed}/{self._batch_total}) [CH {ch} failed]",
            self._batch_completed,
            self._batch_total,
        )
        self._check_batch_finished()

    def _check_batch_finished(self) -> None:
        if self._batch_completed == self._batch_total:
            self._tasks.clear()
            self._running_batch = False
            self.batch_finished.emit({"total": self._batch_total})

    # ── Memory cleanup ──────────────────────────────────────────────────────

    def _cleanup_task_by_channel(self, ch: int) -> None:
        """
        Remove completed task references and call deleteLater() on their
        signal objects to prevent C++ side leaks.
        """
        remaining = []
        for task in self._tasks:
            if task.ch == ch:
                task.signals.deleteLater()
            else:
                remaining.append(task)
        self._tasks = remaining


# ── LoaderWorker (file I/O + baseline subtraction) ──────────────────────────

class LoaderWorker(QObject):
    """
    Loads raw data and subtracts baselines on a background QThread.

    Pass either:
      - A path to a flat binary file (.dat / .bin)  → memory-mapped (COW)
      - A path to a Litke .bin *folder*             → materialised ndarray

    The object emitted by ``finished`` is always a writable (T, C) int16
    array-like with a ``.shape`` attribute, compatible with the rest of the
    pipeline (batch QC worker, snippet extraction, etc.).
    """

    progress = Signal(str)    # status message
    finished = Signal(object) # emits writable (T, C) array
    error = Signal(str)       # emits error message
    aborted = Signal()

    def __init__(
        self,
        dat_path: str,
        n_channels: int,
        dtype: str = "int16",
        start_min: float = 0.0,
        duration_min: float | None = None,
        fs: int = 20_000,
    ):
        super().__init__()
        self.dat_path = dat_path
        self.n_channels = n_channels
        self.dtype = dtype
        self.start_min = start_min
        self.duration_min = duration_min
        self.fs = fs
        self._abort = False

    def abort(self):
        """Signal the worker to stop."""
        self._abort = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_litke_folder(self) -> bool:
        """Return True when dat_path is a directory (Litke .bin folder)."""
        return os.path.isdir(self.dat_path)

    def _load_flat(self):
        """Memory-map a flat binary file (copy-on-write)."""
        self.progress.emit("Mapping recording…")
        raw_data = loader.load_raw_readonly(
            self.dat_path,
            n_channels=self.n_channels,
            dtype=self.dtype,
            start_min=self.start_min,
            duration_min=self.duration_min,
            fs=self.fs,
            writable=True,  # copy-on-write — safe for in-place baseline sub
        )
        return raw_data

    def _load_litke(self):
        """
        Read a Litke .bin folder into a contiguous writable ndarray.

        Reports chunk progress so the UI stays responsive during the read.
        Aborts early if self._abort is set between chunks.
        """
        self.progress.emit("Opening Litke folder…")

        # Peek at total length with the lazy virtual array (zero-copy).
        virtual = loader.load_litke_folder(
            self.dat_path, self.n_channels, self.dtype
        )
        total_samples = virtual.shape[0]

        # Resolve the window to report a meaningful denominator.
        start_sample = int(self.start_min * 60.0 * self.fs)
        start_sample = max(0, min(start_sample, total_samples))
        if self.duration_min is not None and self.duration_min > 0:
            n_samples = int(self.duration_min * 60.0 * self.fs)
        else:
            n_samples = total_samples - start_sample
        n_samples = max(1, min(n_samples, total_samples - start_sample))

        duration_s = n_samples / self.fs
        self.progress.emit(
            f"Reading {n_samples:,} samples ({duration_s:.1f} s) from Litke folder…"
        )

        def _progress_cb(n_loaded: int, n_total: int) -> None:
            # Called from within load_litke_as_writable_array after each chunk.
            # Qt signals are thread-safe so we can emit freely here.
            if self._abort:
                raise InterruptedError("aborted")
            pct = int(100 * n_loaded / max(n_total, 1))
            self.progress.emit(f"Reading Litke data… {pct}%")

        raw_data = loader.load_litke_as_writable_array(
            self.dat_path,
            n_channels=self.n_channels,
            dtype=self.dtype,
            start_min=self.start_min,
            duration_min=self.duration_min,
            fs=self.fs,
            chunk_samples=100_000,
            progress_cb=_progress_cb,
        )
        return raw_data

    # ------------------------------------------------------------------
    # Main entry point (called by QThread)
    # ------------------------------------------------------------------

    def run(self):
        try:
            # ── Step 1: Load raw data ──────────────────────────────────
            if self._is_litke_folder():
                try:
                    raw_data = self._load_litke()
                except InterruptedError:
                    self.aborted.emit()
                    return
            else:
                raw_data = self._load_flat()

            if self._abort:
                self.aborted.emit()
                return

            # ── Step 2: Compute baselines ──────────────────────────────
            self.progress.emit("Computing baselines…")
            baselines = compute_baselines_int16_deriv_robust(raw_data, stride=10)

            if self._abort:
                self.aborted.emit()
                return

            # ── Step 3: Subtract baselines ─────────────────────────────
            # Works in-place on both the COW memmap and the Litke ndarray.
            self.progress.emit("Subtracting baselines…")
            subtract_segment_baselines_int16(raw_data, baselines)

            if self._abort:
                self.aborted.emit()
                return

            self.progress.emit("Ready.")
            self.finished.emit(raw_data)

        except Exception as e:
            self.error.emit(f"Load failed: {e}")
