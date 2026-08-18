"""Tests for core.loader match_spikes and core.spike_match compare modes."""
from __future__ import annotations

import numpy as np

from core.loader import match_spikes
from core.spike_match import (
    DEFAULT_COMPARE_MODE,
    compare_lh_ks,
)


class TestMatchSpikes:
    """Unit tests for the two-pointer spike matching algorithm."""

    def test_empty_a_returns_zero_matched(self):
        n_matched, a_only, b_only, idx = match_spikes(
            np.array([], dtype=np.int64),
            np.array([10, 20, 30], dtype=np.int64),
            coincidence_samples=20,
        )
        assert n_matched == 0
        assert a_only == 0
        assert b_only == 3
        assert idx.size == 0

    def test_empty_b_returns_zero_matched(self):
        n_matched, a_only, b_only, idx = match_spikes(
            np.array([10, 20, 30], dtype=np.int64),
            np.array([], dtype=np.int64),
            coincidence_samples=20,
        )
        assert n_matched == 0
        assert a_only == 3
        assert b_only == 0
        assert idx.size == 0

    def test_both_empty(self):
        n_matched, a_only, b_only, idx = match_spikes(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            coincidence_samples=20,
        )
        assert n_matched == 0
        assert a_only == 0
        assert b_only == 0

    def test_exact_matches(self):
        """Identical spike trains should all match."""
        times = np.array([100, 200, 300, 400], dtype=np.int64)
        n_matched, a_only, b_only, idx = match_spikes(times, times.copy(), coincidence_samples=0)
        assert n_matched == 4
        assert a_only == 0
        assert b_only == 0
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])

    def test_within_window(self):
        """Spikes within the coincidence window should match."""
        a = np.array([100, 200, 300], dtype=np.int64)
        b = np.array([105, 195, 320], dtype=np.int64)
        n_matched, a_only, b_only, idx = match_spikes(a, b, coincidence_samples=20)
        assert n_matched == 3
        assert a_only == 0
        assert b_only == 0

    def test_outside_window(self):
        """Spikes beyond the coincidence window should not match."""
        a = np.array([100, 200, 300], dtype=np.int64)
        b = np.array([150, 250, 350], dtype=np.int64)
        n_matched, a_only, b_only, idx = match_spikes(a, b, coincidence_samples=10)
        assert n_matched == 0
        assert a_only == 3
        assert b_only == 3

    def test_partial_match(self):
        """Some spikes match and some don't."""
        a = np.array([100, 200, 300], dtype=np.int64)
        b = np.array([101, 500], dtype=np.int64)
        n_matched, a_only, b_only, idx = match_spikes(a, b, coincidence_samples=5)
        assert n_matched == 1
        assert a_only == 2
        assert b_only == 1
        np.testing.assert_array_equal(idx, [0])  # b[0]=101 matched

    def test_boundary_coincidence(self):
        """Spikes exactly at the window boundary should match (inclusive)."""
        a = np.array([100], dtype=np.int64)
        b = np.array([120], dtype=np.int64)
        n_matched, _, _, _ = match_spikes(a, b, coincidence_samples=20)
        assert n_matched == 1

    def test_one_just_outside_boundary(self):
        """Spikes one sample beyond the window should not match."""
        a = np.array([100], dtype=np.int64)
        b = np.array([121], dtype=np.int64)
        n_matched, _, _, _ = match_spikes(a, b, coincidence_samples=20)
        assert n_matched == 0

    def test_matched_indices_b_are_correct(self):
        """The returned indices should reference the correct positions in times_b."""
        a = np.array([100, 300, 500], dtype=np.int64)
        b = np.array([50, 102, 200, 299, 600], dtype=np.int64)
        n_matched, a_only, b_only, idx = match_spikes(a, b, coincidence_samples=5)
        assert n_matched == 2  # a[0]↔b[1], a[1]↔b[3]
        np.testing.assert_array_equal(idx, [1, 3])
        assert a_only == 1  # a[2]=500 unmatched
        assert b_only == 3  # b[0]=50, b[2]=200, b[4]=600 unmatched

    def test_counts_sum_correctly(self):
        """a_only + n_matched == len(a) and b_only + n_matched == len(b)."""
        rng = np.random.default_rng(42)
        a = np.sort(rng.integers(0, 100_000, size=200))
        b = np.sort(rng.integers(0, 100_000, size=300))
        n_matched, a_only, b_only, idx = match_spikes(a, b, coincidence_samples=20)
        assert a_only + n_matched == len(a)
        assert b_only + n_matched == len(b)
        assert idx.size == n_matched


class TestCompareLhKs:
    """LH ground-truth vs KS unit comparison modes."""

    def _lh(self, n=100, start=0, step=100):
        return np.arange(start, start + n * step, step, dtype=np.int64)

    def test_default_mode_is_per_unit(self):
        assert DEFAULT_COMPARE_MODE == "per_unit"

    def test_per_unit_confident_primary(self):
        # One good unit owns almost all LH spikes
        lh = self._lh(100)
        unit_map = {
            12: lh.copy(),  # perfect match
            99: np.array([50_000, 50_100], dtype=np.int64),  # unrelated
        }
        labels = {12: "good", 99: "mua"}
        r = compare_lh_ks(lh, unit_map, unit_labels=labels, mode="per_unit", fs=20_000)
        assert r.mode == "per_unit"
        assert r.n_matched == 100
        assert r.n_lh_only == 0
        assert r.primary_unit_id == 12
        assert r.confident is True
        prim = r.primary
        assert prim is not None
        assert prim.n_matched == 100
        assert prim.label == "good"
        assert prim.recall == 1.0

    def test_per_unit_split_not_confident(self):
        # Two units each take half of LH — no confident primary
        lh = self._lh(100)
        unit_map = {
            1: lh[:50].copy(),
            2: lh[50:].copy(),
        }
        labels = {1: "good", 2: "good"}
        r = compare_lh_ks(lh, unit_map, unit_labels=labels, mode="per_unit", fs=20_000)
        assert r.n_matched == 100
        assert r.primary_unit_id in (1, 2)
        # Each has 50% recall → below 55% dominance of total matched → not confident
        # (dominance of each is 0.5 < 0.55)
        assert r.confident is False
        assert len(r.unit_stats) == 2

    def test_good_only_excludes_mua(self):
        lh = self._lh(50)
        unit_map = {
            1: lh.copy(),  # good — full match
            2: lh.copy() + 1,  # mua — almost same times but would match too
        }
        # Make mua times clearly matching as well but filter by label
        unit_map[2] = lh.copy()
        labels = {1: "good", 2: "mua"}
        r_good = compare_lh_ks(lh, unit_map, unit_labels=labels, mode="good_only", fs=20_000)
        r_all = compare_lh_ks(lh, unit_map, unit_labels=labels, mode="all_pool", fs=20_000)
        assert r_good.n_matched == 50
        # good_only should only include unit 1 in unit_stats
        assert all(s.unit_id == 1 for s in r_good.unit_stats)
        assert r_all.n_ks_total >= r_good.n_ks_total

    def test_all_pool_matches_pooled(self):
        lh = np.array([100, 200, 300, 400], dtype=np.int64)
        unit_map = {
            1: np.array([100, 200], dtype=np.int64),
            2: np.array([300], dtype=np.int64),
        }
        r = compare_lh_ks(lh, unit_map, mode="all_pool", fs=20_000)
        assert r.n_matched == 3
        assert r.n_lh_only == 1
        assert r.n_ks_only == 0

    def test_empty_lh_still_lists_units(self):
        unit_map = {
            5: np.array([10, 20, 30], dtype=np.int64),
            6: np.array([40], dtype=np.int64),
        }
        labels = {5: "mua", 6: "good"}
        r = compare_lh_ks(
            np.array([], dtype=np.int64),
            unit_map,
            unit_labels=labels,
            mode="per_unit",
        )
        assert r.n_lh == 0
        assert r.n_ks_total == 4
        assert len(r.unit_stats) == 2
        assert r.unit_stats[0].n_unit == 3  # sorted by size

    def test_pooled_times_fallback(self):
        lh = np.array([100, 200], dtype=np.int64)
        pooled = np.array([100, 200, 300], dtype=np.int64)
        r = compare_lh_ks(lh, {}, pooled_times=pooled, mode="all_pool", fs=20_000)
        assert r.n_matched == 2
        assert r.n_ks_only == 1
