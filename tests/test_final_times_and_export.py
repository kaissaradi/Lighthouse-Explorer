"""Tests for n_total/final_times, BL/TR filtering, and Phy export."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from core.lh_qc_pipeline import (
    BLTRResult,
    build_final_times_from_bltr,
    compute_mean_ei,
    slim_qc_result,
    QCResult,
)
from core.ks_export import export_phy_folder, collect_exportable_units
from tests.factories import make_qc_result


def _bltr(
    *,
    ok=True,
    counts=None,
    labels=None,
    times=None,
    bl_keep=None,
    bl_unc=None,
    bl_rej=None,
    tr_keep=None,
    tr_unc=None,
    tr_rej=None,
) -> BLTRResult:
    labels = np.asarray(labels if labels is not None else [], dtype=object)
    times = np.asarray(times if times is not None else [], dtype=np.int64)
    return BLTRResult(
        labels=labels,
        bl_bulk=np.zeros(labels.size, dtype=np.float32),
        tr_bulk=np.zeros(labels.size, dtype=np.float32),
        counts=counts
        or {"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
        times=times,
        ok=ok,
        bl_keep_times=np.asarray(bl_keep if bl_keep is not None else [], dtype=np.int64),
        bl_uncertain_times=np.asarray(bl_unc if bl_unc is not None else [], dtype=np.int64),
        bl_reject_times=np.asarray(bl_rej if bl_rej is not None else [], dtype=np.int64),
        tr_keep_times=np.asarray(tr_keep if tr_keep is not None else [], dtype=np.int64),
        tr_uncertain_times=np.asarray(tr_unc if tr_unc is not None else [], dtype=np.int64),
        tr_reject_times=np.asarray(tr_rej if tr_rej is not None else [], dtype=np.int64),
    )


# ── 1. n_total + final_times ────────────────────────────────────────────────


def test_n_total_from_counts_when_labels_empty():
    """Success path used to leave labels empty — n_total must not be 0."""
    r = make_qc_result(channel=3, n_lh=10, n_soup=4)
    # Simulate old success path: empty labels, nonzero counts + final_times
    r.bltr.labels = np.array([], dtype=object)
    r.bltr.times = np.array([], dtype=np.int64)
    r.bltr.counts = {
        "LH": 10,
        "soup": 4,
        "uncertain_boundary": 1,
        "uncertain_lowBL": 1,
    }
    # final_times drives n_lh when present
    r.final_times = np.arange(10, dtype=np.int64)
    r.reject_reason = None

    assert r.n_lh == 10
    assert r.n_soup == 4
    assert r.n_uncertain == 2
    assert r.n_total == 10 + 4 + 2
    assert r.n_total > 0


def test_n_lh_prefers_final_times():
    r = make_qc_result(channel=0, n_lh=3, n_soup=2)
    r.final_times = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    r.bltr.counts["LH"] = 3  # probe-level LH count differs
    assert r.n_lh == 5


def test_n_lh_uses_counts_when_rejected():
    r = make_qc_result(channel=0, n_lh=3, n_soup=0)
    r.final_times = np.array([], dtype=np.int64)
    r.reject_reason = "kmeans_reject: TWO-UNITS-like (reject)"
    r.bltr.counts["LH"] = 0
    assert r.n_lh == 0
    assert r.n_total == 0


def test_miss_rate_coincidence_with_sorter_times():
    r = make_qc_result(channel=0, n_lh=5, n_soup=0)
    r.final_times = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    r.sorter_times = np.array([1000, 2005, 9000], dtype=np.int64)  # 2 matches ±1ms@20k
    r.n_sorter_spikes = 3
    r.fs = 20_000.0
    # coincidence = 20 samples; 2005-2000=5 → match; 9000 unmatched KS
    # matched: 1000, 2000 → lh_only = 3 → miss_rate = 3/5
    assert abs(r.miss_rate - 0.6) < 1e-9


# ── 2. Notebook BL/TR final_times filtering ─────────────────────────────────


def test_build_final_times_filters_bl_reject_and_uncertain():
    left = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    rightk = np.array([100, 200, 300], dtype=np.int64)
    bltr = _bltr(
        ok=True,
        bl_rej=[20],
        bl_unc=[40],
        tr_keep=[100, 300],
        tr_rej=[200],
        counts={"LH": 3, "soup": 1, "uncertain_boundary": 1, "uncertain_lowBL": 0},
    )
    final = build_final_times_from_bltr(left, rightk, bltr)
    # left without 20,40 → 10,30,50; right only keep → 100,300
    assert np.array_equal(final, np.array([10, 30, 50, 100, 300], dtype=np.int64))


def test_build_final_times_fallback_when_bltr_not_ok():
    left = np.array([1, 2, 3], dtype=np.int64)
    rightk = np.array([8, 9], dtype=np.int64)
    bltr = _bltr(ok=False)
    final = build_final_times_from_bltr(left, rightk, bltr)
    assert np.array_equal(final, np.array([1, 2, 3, 8, 9], dtype=np.int64))


def test_build_final_times_empty_inputs():
    bltr = _bltr(ok=False)
    final = build_final_times_from_bltr(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        bltr,
    )
    assert final.size == 0


# ── 3. Phy / KS export ──────────────────────────────────────────────────────


def test_export_phy_folder_writes_required_files():
    r0 = make_qc_result(channel=0, n_lh=5, n_soup=1)
    r0.final_times = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    r0.final_ei = np.zeros((4, 61), dtype=np.float32)
    r0.final_ei[0, 20] = -80.0

    r1 = make_qc_result(channel=2, n_lh=3, n_soup=0)
    r1.final_times = np.array([150, 250, 350], dtype=np.int64)
    r1.final_ei = np.zeros((4, 61), dtype=np.float32)
    r1.final_ei[2, 20] = -60.0

    rejected = make_qc_result(channel=1, n_lh=0, n_soup=0)
    rejected.reject_reason = "valley_not_accepted"
    rejected.final_times = np.array([], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        info = export_phy_folder(
            tmp,
            {0: r0, 1: rejected, 2: r1},
            n_channels=4,
            fs=20_000.0,
            dat_path="/fake/data.dat",
        )
        assert info["n_units"] == 2
        assert info["n_spikes"] == 8
        assert info["unit_channels"] == [0, 2]

        required = [
            "spike_times.npy",
            "spike_clusters.npy",
            "spike_templates.npy",
            "templates.npy",
            "amplitudes.npy",
            "channel_map.npy",
            "channel_positions.npy",
            "similar_templates.npy",
            "params.py",
            "cluster_group.tsv",
            "cluster_info.tsv",
            "lh_unit_detect_channels.npy",
        ]
        for name in required:
            assert os.path.isfile(os.path.join(tmp, name)), name

        st = np.load(os.path.join(tmp, "spike_times.npy")).reshape(-1)
        sc = np.load(os.path.join(tmp, "spike_clusters.npy")).reshape(-1)
        assert st.dtype == np.uint64
        assert st.size == 8
        assert np.all(st[1:] >= st[:-1])  # time-sorted
        assert set(sc.tolist()) == {0, 1}

        tmpl = np.load(os.path.join(tmp, "templates.npy"))
        assert tmpl.shape == (2, 61, 4)

        with open(os.path.join(tmp, "params.py"), encoding="utf-8") as f:
            params_txt = f.read()
        assert "sample_rate = 20000.0" in params_txt
        assert "n_channels_dat = 4" in params_txt


def test_export_phy_folder_raises_without_units():
    r = make_qc_result(channel=0, n_lh=0)
    r.final_times = np.array([], dtype=np.int64)
    r.reject_reason = "too_few_final_spikes"
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="No exportable units"):
            export_phy_folder(tmp, {0: r}, n_channels=2)


def test_collect_exportable_units_skips_rejects():
    good = make_qc_result(channel=5, n_lh=2)
    good.final_times = np.array([1, 2], dtype=np.int64)
    bad = make_qc_result(channel=6, n_lh=0)
    bad.reject_reason = "x"
    bad.final_times = np.array([], dtype=np.int64)
    units = collect_exportable_units({5: good, 6: bad})
    assert len(units) == 1
    assert units[0][0] == 0
    assert units[0][1].channel == 5


def test_existing_result_types_still_pass():
    """Regression: factories / property contract used by GUI tests."""
    r = make_qc_result(channel=1, n_lh=3, n_soup=2)
    assert r.n_lh == 3
    assert r.n_soup == 2
    assert r.n_total == 5
    assert r.final_times.size == 3


# ── Memory-safe EI + slim ───────────────────────────────────────────────────


def test_compute_mean_ei_chunked_matches_full():
    """Chunked mean EI should match a single-batch mean within float tolerance."""
    rng = np.random.RandomState(0)
    T, C = 5000, 8
    raw = (rng.randn(T, C) * 20).astype(np.int16)
    # Plant a negative deflection at fixed times on ch 2
    times = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600], dtype=np.int64)
    for t in times:
        raw[t - 5 : t + 5, 2] = -200

    ei_small = compute_mean_ei(raw, times, window=(-20, 40), max_spikes=100, batch_size=3)
    ei_one = compute_mean_ei(raw, times, window=(-20, 40), max_spikes=100, batch_size=100)
    assert ei_small is not None and ei_one is not None
    assert ei_small.shape == (C, 61)
    assert np.allclose(ei_small, ei_one, rtol=1e-5, atol=1e-3)
    # Detect channel should have deeper trough than neighbors
    assert ei_small[2].min() < ei_small[0].min()


def test_slim_qc_result_drops_heavy_arrays():
    r = make_qc_result(channel=1, n_lh=5, n_soup=1)
    r.valley.all_times = np.arange(100_000, dtype=np.int64)
    r.valley.all_vals = np.full(100_000, -50.0, dtype=np.float32)
    r.valley.valley_times = np.arange(1000, dtype=np.int64)
    r.pca_km.pca_coords = np.random.randn(5000, 3).astype(np.float32)
    r.pca_km.km_labels = np.zeros(5000, dtype=np.int64)
    r.km_info = {"proceed": True, "verdict": "ok", "precheck": {"huge": np.zeros(10000)}}
    r.bltr.labels = np.array(["LH"] * 2000, dtype=object)

    slim_qc_result(r, max_pca_points=1500)
    assert r.valley.all_times.size == 0
    assert r.valley.all_vals.size == 0
    assert r.valley.valley_times.size == 0
    assert r.pca_km.pca_coords.shape[0] == 1500
    assert r.pca_km.km_labels.shape[0] == 1500
    assert "precheck" not in r.km_info
    assert r.km_info.get("verdict") == "ok"
    assert r.bltr.labels.size == 0
    # final_times preserved for export
    assert r.final_times.size == 5
