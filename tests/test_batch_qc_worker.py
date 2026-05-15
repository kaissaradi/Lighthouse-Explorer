from __future__ import annotations

import os

import numpy as np
import pytest

from gui.workers import qc_worker as batch_module
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
    return np.zeros((100, 4), dtype=np.int16)


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


def test_batch_worker_uses_configured_qt_thread_count(monkeypatch, raw_data):
    worker, pool = make_worker(
        monkeypatch,
        raw_data,
        params={"batch_max_workers": 3},
    )

    assert pool.max_thread_count == 3
    assert worker.max_workers == 3
    assert worker._pool is pool


def test_batch_worker_explicit_thread_count_overrides_params(monkeypatch, raw_data):
    worker, pool = make_worker(
        monkeypatch,
        raw_data,
        params={"batch_max_workers": 3},
        max_workers=6,
    )

    assert pool.max_thread_count == 6
    assert worker.max_workers == 6


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

    assert pool.max_thread_count == 8
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
