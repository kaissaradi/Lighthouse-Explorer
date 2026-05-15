from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import loader
from core.lh_qc_pipeline import DEFAULT_PARAMS
from gui.workers.qc_worker import BatchQCWorker
from lh_deps.axolotl_utils_ram import (
    compute_baselines_int16_deriv_robust,
    subtract_segment_baselines_int16,
)


DATA000_PATH = Path("/home/localadmin/Documents/Development/data/raw/20260501A/data000")
DATA000_CHANNELS_WITH_TTL = 520
DATA000_ELECTRODE_CHANNELS = DATA000_CHANNELS_WITH_TTL - 1


@pytest.fixture(scope="session")
def data000_active_channel_slice():
    if not DATA000_PATH.is_dir():
        pytest.skip(f"Real-data fixture is not available: {DATA000_PATH}")

    raw = loader.load_litke_as_writable_array(
        str(DATA000_PATH),
        n_channels=DATA000_ELECTRODE_CHANNELS,
        duration_min=0.05,
        chunk_samples=100_000,
    )
    baselines = compute_baselines_int16_deriv_robust(raw, stride=10)
    subtract_segment_baselines_int16(raw, baselines)

    peak_to_peak = np.ptp(raw.astype(np.int32), axis=0)
    active_channels = np.argsort(peak_to_peak)[-8:][::-1]
    return np.ascontiguousarray(raw[:, active_channels]), active_channels


def test_batch_qc_runs_data000_with_configured_parallel_qt_pool(
    qtbot,
    data000_active_channel_slice,
):
    raw_subset, active_channels = data000_active_channel_slice
    params = dict(DEFAULT_PARAMS)
    params.update(
        batch_max_workers=4,
        min_valid_count=20,
        max_valley_count=200,
    )

    worker = BatchQCWorker(raw_data=raw_subset, params=params, max_workers=4)

    assert worker.max_workers == 4
    assert worker._pool.maxThreadCount() == 4

    completed_results = []
    progress_updates = []
    worker.channel_done.connect(lambda result: completed_results.append(result))
    worker.progress.connect(
        lambda msg, current, total: progress_updates.append((msg, current, total))
    )

    with qtbot.waitSignal(worker.finished, timeout=60_000) as blocker:
        worker.run()

    assert blocker.args == [{"total": raw_subset.shape[1]}]
    assert len(completed_results) == raw_subset.shape[1]
    assert any(
        (result.reject_reason or "").startswith("kmeans_reject")
        for result in completed_results
    )
    assert progress_updates[-1] == (
        f"Running QC... ({raw_subset.shape[1]}/{raw_subset.shape[1]})",
        raw_subset.shape[1],
        raw_subset.shape[1],
    )

    worker._pool.waitForDone(5_000)
    assert len(active_channels) == raw_subset.shape[1]
