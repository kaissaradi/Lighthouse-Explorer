from __future__ import annotations

import os

import numpy as np
import pytest

import psutil
import gc
from qtpy.QtCore import QCoreApplication, QRunnable, QThread, QThreadPool
from core import configure_native_thread_environment, DEFAULT_NUMBA_THREADING_LAYER, NUMBA_THREADING_LAYER_ENV
from gui.qc_worker import TaskManager, QCChannelTask, DEFAULT_BATCH_MAX_WORKERS, BatchQCWorker
from core.lh_qc_pipeline import extract_snippets_fast_ram

from gui import qc_worker as batch_module
from tests.factories import make_qc_result


class FakeThreadPool:
    def __init__(self, *, auto_run: bool = False):
        self.auto_run = auto_run
        self.max_thread_count = None
        self.started = []
        self.clear_called = False

    def setMaxThreadCount(self, count: int):
        self.max_thread_count = count

    def start(self, task):
        self.started.append(task)
        if self.auto_run:
            task.run()

    def clear(self):
        self.clear_called = True


@pytest.fixture
def raw_data():
    return np.zeros((2000, 4), dtype=np.int16)


def make_worker(monkeypatch, raw_data, *, pool=None, params=None, **kwargs):
    pool = pool or FakeThreadPool()
    monkeypatch.setattr(batch_module, "QThreadPool", lambda: pool)
    worker = batch_module.BatchQCWorker(raw_data=raw_data, params=params or {}, **kwargs)
    return worker, pool


def test_native_thread_limiter_environment_is_set():
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert os.environ["NUMBA_NUM_THREADS"] == "1"
    assert os.environ["NUMBA_THREADING_LAYER"] == "workqueue"


def test_batch_worker_uses_configured_qt_thread_count(monkeypatch, raw_data):
    worker, pool = make_worker(
        monkeypatch,
        raw_data,
        params={"batch_max_workers": 3},
    )

    assert pool.max_thread_count == 1
    assert worker.max_workers == 1
    assert worker._pool is pool


def test_batch_worker_explicit_thread_count_overrides_params(monkeypatch, raw_data):
    worker, pool = make_worker(
        monkeypatch,
        raw_data,
        params={"batch_max_workers": 3},
        max_workers=6,
    )

    assert pool.max_thread_count == 1
    assert worker.max_workers == 1


def test_run_queues_every_channel_and_keeps_task_references(monkeypatch, raw_data):
    worker, pool = make_worker(
        monkeypatch,
        raw_data,
        sorter_spike_times={0: [10, 20], 2: [30]},
        fs=30_000.0,
    )

    worker.run()

    assert [task.ch for task in pool.started] == [0, 1, 2, 3]
    assert worker._tasks == pool.started
    assert [task.n_sorter for task in pool.started] == [2, 0, 1, 0]
    assert all(task.raw_data is raw_data for task in pool.started)
    assert all(task.fs == 30_000.0 for task in pool.started)
    assert all(task.native_threads == 1 for task in pool.started)


def test_each_channel_task_enters_native_thread_limit(monkeypatch, raw_data):
    pool = FakeThreadPool(auto_run=True)
    active_limits = []
    entered_limits = []

    class FakeNativeLimit:
        def __init__(self, threads):
            self.threads = threads

        def __enter__(self):
            active_limits.append(self.threads)
            entered_limits.append(self.threads)

        def __exit__(self, exc_type, exc, tb):
            active_limits.pop()

    def fake_pipeline(raw_data, ch, n_sorter_spikes, params, fs):
        assert active_limits == [1]
        return make_qc_result(channel=ch)

    monkeypatch.setattr(batch_module, "native_thread_limits", FakeNativeLimit)
    monkeypatch.setattr(batch_module, "run_qc_pipeline", fake_pipeline)
    worker, _ = make_worker(monkeypatch, raw_data, pool=pool, max_workers=4)

    worker.run()

    assert entered_limits == [1, 1, 1, 1]
    assert active_limits == []


def test_thread_count_can_be_set_above_one_without_changing_native_limit(monkeypatch, raw_data):
    worker, pool = make_worker(
        monkeypatch,
        raw_data,
        params={"batch_max_workers": 8},
    )

    worker.run()

    assert pool.max_thread_count == 1
    assert len(pool.started) == 4
    assert all(task.native_threads == 1 for task in pool.started)


def test_successful_tasks_emit_channel_progress_and_finish(monkeypatch, raw_data):
    pool = FakeThreadPool(auto_run=True)

    def fake_pipeline(raw_data, ch, n_sorter_spikes, params, fs):
        return make_qc_result(channel=ch, n_sorter_spikes=n_sorter_spikes)

    monkeypatch.setattr(batch_module, "run_qc_pipeline", fake_pipeline)
    worker, pool = make_worker(monkeypatch, raw_data, pool=pool)

    seen_channels = []
    seen_progress = []
    seen_finished = []
    worker.channel_done.connect(lambda result: seen_channels.append(result.channel))
    worker.progress.connect(lambda msg, current, total: seen_progress.append((msg, current, total)))
    worker.finished.connect(lambda payload: seen_finished.append(payload))

    worker.run()

    assert seen_channels == [0, 1, 2, 3]
    assert seen_progress[-1] == ("Running QC... (4/4)", 4, 4)
    assert seen_finished == [{"total": 4}]
    assert worker._tasks == []


def test_channel_errors_are_counted_without_stopping_batch(monkeypatch, raw_data):
    pool = FakeThreadPool(auto_run=True)

    def fake_pipeline(raw_data, ch, n_sorter_spikes, params, fs):
        if ch == 1:
            raise RuntimeError("synthetic bad channel")
        return make_qc_result(channel=ch)

    monkeypatch.setattr(batch_module, "run_qc_pipeline", fake_pipeline)
    worker, pool = make_worker(monkeypatch, raw_data, pool=pool)

    seen_channels = []
    seen_progress = []
    seen_finished = []
    worker.channel_done.connect(lambda result: seen_channels.append(result.channel))
    worker.progress.connect(lambda msg, current, total: seen_progress.append((msg, current, total)))
    worker.finished.connect(lambda payload: seen_finished.append(payload))

    worker.run()

    assert seen_channels == [0, 2, 3]
    assert seen_progress[1][0].endswith("[CH 1 failed]")
    assert seen_progress[-1] == ("Running QC... (4/4)", 4, 4)
    assert seen_finished == [{"total": 4}]


def test_abort_clears_pending_pool_tasks_and_python_references(monkeypatch, raw_data):
    worker, pool = make_worker(monkeypatch, raw_data)
    aborted = []
    worker.aborted.connect(lambda: aborted.append(True))

    worker.run()
    assert len(worker._tasks) == 4

    worker.abort()

    assert pool.clear_called is True
    assert worker._tasks == []


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # In MB

@pytest.mark.parametrize("iterations", [5])
def test_batch_qc_worker_memory_leak(qtbot, iterations):
    # Mock raw data: 1000 samples, 10 channels
    raw_data = np.random.randint(-1000, 1000, (1000, 10), dtype=np.int16)
    params = {}
    
    initial_mem = get_memory_usage()
    
    for i in range(iterations):
        worker = BatchQCWorker(raw_data=raw_data, params=params)
        
        with qtbot.waitSignal(worker.finished, timeout=10000):
            worker.run()
        
        # Explicitly delete and collect
        del worker
        gc.collect()
        QCoreApplication.processEvents()
    
    final_mem = get_memory_usage()
    
    # Allow some growth, but if it's substantial, it's a leak.
    assert final_mem - initial_mem < 5.0, f"Memory leak detected: {final_mem - initial_mem:.2f} MB increase"


def test_configure_native_thread_environment_sets_workqueue_layer():
    configure_native_thread_environment()
    assert os.environ[NUMBA_THREADING_LAYER_ENV] == DEFAULT_NUMBA_THREADING_LAYER
    assert os.environ["NUMBA_NUM_THREADS"] == "1"

def test_resolve_fs_from_params():
    from gui.qc_worker import _resolve_fs

    assert _resolve_fs({"fs": 30_000}, None) == 30_000.0
    assert _resolve_fs({"fs": 30_000}, 20_000.0) == 20_000.0
    assert _resolve_fs({}, None) == 20_000.0

def test_batch_pool_default_is_four():
    manager = TaskManager()
    assert DEFAULT_BATCH_MAX_WORKERS == 1
    assert manager.max_workers == 1
    assert manager._pool.maxThreadCount() == 1

def test_task_manager_honors_batch_max_workers_from_params(raw_data):
    manager = TaskManager()
    params = {"batch_max_workers": 1}
    manager.start_batch(raw_data=raw_data, params=params)
    assert manager.max_workers == 1
    assert manager._pool.maxThreadCount() == 1
    manager.abort_batch()

def test_concurrent_qthreadpool_numba_completes(qtbot, raw_data):
    """workqueue + four pool threads without the lock hangs or corrupts memory."""
    from core.lh_qc_pipeline import extract_snippets_fast_ram

    errors = []
    done = []

    class SnippetTask(QRunnable):
        def __init__(self, ch: int):
            super().__init__()
            self.ch = ch

        def run(self):
            try:
                times = np.array(
                    [200 + self.ch * 50, 800 + self.ch * 50],
                    dtype=np.int64,
                )
                extract_snippets_fast_ram(
                    raw_data,
                    times,
                    (-10, 10),
                    np.array([0], dtype=np.int32),
                )
                done.append(self.ch)
            except Exception as exc:
                errors.append((self.ch, exc))

    pool = QThreadPool.globalInstance()
    pool.setMaxThreadCount(4)
    tasks = [SnippetTask(ch) for ch in range(8)]
    for task in tasks:
        pool.start(task)

    qtbot.waitUntil(lambda: len(done) + len(errors) == 8, timeout=30_000)

    assert not errors, errors
    assert len(done) == 8

def test_qc_channel_task_runs_off_main_thread(monkeypatch, qtbot, raw_data):
    main_thread = QThread.currentThread()
    observed_thread = {}

    def fake_pipeline(raw_data, ch, n_sorter_spikes, params, fs):
        observed_thread["thread"] = QThread.currentThread()
        from tests.factories import make_qc_result

        return make_qc_result(channel=ch)

    monkeypatch.setattr(
        "gui.qc_worker.run_qc_pipeline",
        fake_pipeline,
    )

    task = QCChannelTask(
        raw_data=raw_data,
        ch=0,
        n_sorter=0,
        params={},
        fs=20_000.0,
    )
    finished = []

    def on_result(_result):
        finished.append(True)

    task.signals.result_ready.connect(on_result)

    with qtbot.waitSignal(task.signals.result_ready, timeout=10_000):
        QThreadPool.globalInstance().start(task)

    assert finished
    assert observed_thread["thread"] is not None
    assert observed_thread["thread"] is not main_thread
