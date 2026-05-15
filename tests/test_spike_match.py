"""Tests for core.loader — shared spike coincidence matching."""
from __future__ import annotations

import numpy as np
import pytest

from core.loader import match_spikes


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
