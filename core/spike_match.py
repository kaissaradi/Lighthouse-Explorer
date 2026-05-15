"""
spike_match.py — Shared spike coincidence matching utility.

Provides a single ``match_spikes`` function used by both ``qc_view_panel.py``
(Venn diagram) and ``qc_summary_dialog.py`` (fragmentation index) so the
coincidence-window logic only needs to be maintained in one place.
"""
from __future__ import annotations

import numpy as np


def match_spikes(
    times_a: np.ndarray,
    times_b: np.ndarray,
    coincidence_samples: int,
) -> tuple[int, int, int, np.ndarray]:
    """Match spikes between two sorted spike-time arrays within a ±window.

    Uses a fast two-pointer sweep.  Both ``times_a`` and ``times_b`` must be
    **sorted** ascending.

    Parameters
    ----------
    times_a : np.ndarray
        Sorted spike times for source A (e.g., Lighthouse).
    times_b : np.ndarray
        Sorted spike times for source B (e.g., KiloSort).
    coincidence_samples : int
        Maximum absolute sample-distance for two spikes to be considered a
        match (inclusive).  Typically ``int(0.001 * fs)`` for a ±1 ms window.

    Returns
    -------
    n_matched : int
        Number of spikes matched across the two arrays.
    a_only : int
        Number of spikes in *times_a* that had no match in *times_b*.
    b_only : int
        Number of spikes in *times_b* that had no match in *times_a*.
    matched_indices_b : np.ndarray[int64]
        Indices into *times_b* for each matched spike (length ``n_matched``).
        Useful when downstream code needs to know *which* B-spikes matched.
    """
    times_a = np.asarray(times_a, dtype=np.int64)
    times_b = np.asarray(times_b, dtype=np.int64)

    if times_a.size == 0 or times_b.size == 0:
        return 0, int(times_a.size), int(times_b.size), np.array([], dtype=np.int64)

    matched_b_indices: list[int] = []
    i, j = 0, 0
    while i < len(times_a) and j < len(times_b):
        diff = int(times_a[i]) - int(times_b[j])
        if abs(diff) <= coincidence_samples:
            matched_b_indices.append(j)
            i += 1
            j += 1
        elif diff < 0:
            i += 1  # A spike earlier — advance A
        else:
            j += 1  # B spike earlier — advance B

    n_matched = len(matched_b_indices)
    a_only = int(times_a.size) - n_matched
    b_only = int(times_b.size) - n_matched

    return n_matched, a_only, b_only, np.array(matched_b_indices, dtype=np.int64)
