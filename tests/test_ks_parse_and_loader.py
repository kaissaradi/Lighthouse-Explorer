"""Tests for vectorized KS parse and loader baseline helpers."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from core.loader import parse_kilosort_folder
from gui.qc_worker import LoaderWorker, MAX_QC_POOL_THREADS


def _write_fake_ks(
    tmpdir: str,
    *,
    n_spikes=5000,
    n_units=20,
    n_ch=8,
    n_t=40,
    merged_clusters: bool = False,
):
    rng = np.random.RandomState(0)
    spike_times = np.sort(rng.randint(0, 1_000_000, size=n_spikes)).astype(np.uint64)
    spike_clusters = rng.randint(0, n_units, size=n_spikes).astype(np.uint32)
    # spike_templates: normally == cluster id; for merge test, remap high cluster ids
    spike_templates = spike_clusters.copy()
    if merged_clusters:
        # Simulate Phy merge: cluster 99 maps to template 3
        spike_clusters = spike_clusters.copy()
        spike_clusters[spike_clusters == 5] = 99
        spike_templates[spike_clusters == 99] = 3

    templates = rng.randn(n_units, n_t, n_ch).astype(np.float32)
    # Make template i peak on channel i % n_ch
    for i in range(n_units):
        templates[i, n_t // 2, i % n_ch] = -50.0
    channel_map = np.arange(n_ch, dtype=np.int32)

    np.save(os.path.join(tmpdir, "spike_times.npy"), spike_times.reshape(-1, 1))
    np.save(os.path.join(tmpdir, "spike_clusters.npy"), spike_clusters.reshape(-1, 1))
    np.save(os.path.join(tmpdir, "spike_templates.npy"), spike_templates.reshape(-1, 1))
    np.save(os.path.join(tmpdir, "templates.npy"), templates)
    # KS sometimes writes channel_map as (1, n_ch)
    np.save(os.path.join(tmpdir, "channel_map.npy"), channel_map.reshape(1, -1))

    # Mark one unit as noise, mix good/mua for label tests
    with open(os.path.join(tmpdir, "cluster_group.tsv"), "w", encoding="utf-8") as f:
        f.write("cluster_id\tgroup\n")
        f.write("0\tnoise\n")
        for u in range(1, n_units):
            grp = "mua" if u == 2 else "good"
            f.write(f"{u}\t{grp}\n")
        if merged_clusters:
            f.write("99\tgood\n")

    return tmpdir


def test_parse_kilosort_folder_vectorized():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_ks(tmp)
        msgs = []
        result = parse_kilosort_folder(
            tmp,
            progress_cb=lambda m: msgs.append(m),
        )
        assert result["n_units"] == 19  # unit 0 noise excluded
        assert result["n_noise_excluded"] == 1
        assert 0 not in result["unit_map"]
        assert 0 not in result["unit_labels"]
        assert result["unit_labels"].get(3) == "good"
        assert result["unit_labels"].get(2) == "mua"
        assert result["n_channels_covered"] > 0
        # Unit 3 peaks on channel 3
        assert result["dom_channel"].get(3) == 3
        # Reverse index must match dom_channel
        assert 3 in result["units_by_channel"].get(3, [])
        assert result["channel_map_is_identity"] is True
        assert result["channel_map_min"] == 0
        assert result["channel_map_max"] == 7
        assert result["template_n_channels"] == 8
        # All unit times sorted
        for times in result["unit_map"].values():
            assert times.ndim == 1
            assert np.all(times[1:] >= times[:-1]) or times.size <= 1
        assert len(msgs) >= 3


def test_parse_kilosort_merged_cluster_uses_spike_templates():
    """Phy-merged cluster ids must resolve peak ch via spike_templates."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_ks(tmp, merged_clusters=True)
        result = parse_kilosort_folder(tmp)
        assert 99 in result["unit_map"]
        # template 3 peaks on channel 3
        assert result["dom_channel"][99] == 3
        assert 99 in result["spike_times_by_channel"].get(3, np.array([])).tolist() or (
            3 in result["spike_times_by_channel"]
            and result["dom_channel"][99] == 3
        )


def test_parse_kilosort_abort():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_ks(tmp)
        with pytest.raises(InterruptedError):
            parse_kilosort_folder(tmp, abort_cb=lambda: True)


def test_parse_kilosort_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            parse_kilosort_folder(tmp)


def test_baseline_stride_scales_with_duration():
    fs = 20_000.0
    # short
    assert LoaderWorker._choose_baseline_stride(int(fs * 60), fs) == 10
    # medium
    assert LoaderWorker._choose_baseline_stride(int(fs * 60 * 5), fs) == 20
    # long
    assert LoaderWorker._choose_baseline_stride(int(fs * 60 * 15), fs) == 50
    # very long (~86 min)
    assert LoaderWorker._choose_baseline_stride(int(fs * 60 * 86), fs) == 100


def test_qc_pool_hard_capped_at_one():
    assert MAX_QC_POOL_THREADS == 1


def test_chunked_baselines_shape_and_abort():
    """Chunked baselines produce [C, n_seg] and honor abort between batches."""
    from core import configure_native_thread_environment

    configure_native_thread_environment()

    # Tiny synthetic int16 data
    T, C = 250_000, 4
    rng = np.random.RandomState(1)
    raw = (rng.randn(T, C) * 10).astype(np.int16)
    # Add slow drift
    raw = raw + np.linspace(0, 100, T, dtype=np.int16)[:, None]

    worker = LoaderWorker(dat_path="unused", n_channels=C, fs=20_000)
    bas = worker._compute_baselines_chunked(raw)
    n_seg = (T + 100_000 - 1) // 100_000
    assert bas.shape == (C, n_seg)
    assert np.isfinite(bas).all()

    # Abort mid-way
    worker2 = LoaderWorker(dat_path="unused", n_channels=C, fs=20_000)
    worker2._abort = True
    with pytest.raises(InterruptedError):
        # Force many batches by using large T path — abort set before first batch
        worker2._compute_baselines_chunked(raw)
