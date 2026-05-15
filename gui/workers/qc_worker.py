"""
qc_worker.py — Unified TaskManager for single-channel and batch QC execution.

Both single-channel QC and batch QC now route through the same QCChannelTask
(QRunnable) dispatched onto a shared QThreadPool.  This eliminates the old
QObject/QThread boilerplate while retaining the memory-safety guardrails:

  • self._tasks list keeps Python references alive until completion/abort.
  • .deleteLater() is called on signal objects via _cleanup_task().
  • native_thread_limits(1) is enforced inside each QCChannelTask.run().
"""
from __future__ import annotations

import os
from qtpy.QtCore import QObject, Signal, QThreadPool
from core.native_threading import (
    configure_native_thread_environment,
    native_thread_limits,
)

# PREVENT OPENMP DEADLOCKS: Force NumPy/SciPy/Scikit-Learn to use 1 thread per task.
configure_native_thread_environment()

from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS
from core.result_types import QCResult
from .batch_qc_worker import (
    QCChannelTask,
    QCChannelTaskSignals,
    DEFAULT_BATCH_MAX_WORKERS,
    DEFAULT_NATIVE_THREADS_PER_TASK,
    _positive_int,
    _resolve_worker_count,
)


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
