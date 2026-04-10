"""
axolotl_utils_ram.py — High‑performance baseline and snippet extraction.

All heavy loops are Numba‑accelerated (parallel + cache).
Subsampling in baseline computation provides a massive speedup with
negligible accuracy loss for QC purposes.
"""
import numpy as np
import os
from numba import njit, prange

# ----------------------------------------------------------------------
# Baseline computation (subsampled, derivative‑masked, parallel)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def compute_baselines_int16_deriv_robust(
    raw_data,
    segment_len=100_000,
    diff_thresh=50,
    stride=1
):
    """
    Mean baseline per channel over non‑overlapping segments.

    The baseline is robust to spikes because samples whose absolute
    difference from the *previous considered sample* exceeds `diff_thresh`
    are excluded from the mean. Subsampling (`stride > 1`) speeds up the
    computation dramatically with minimal impact on the final baseline.

    Parameters
    ----------
    raw_data : np.ndarray [T, C] int16
        The raw recording.
    segment_len : int
        Number of samples per baseline segment.
    diff_thresh : int
        Spike rejection threshold (absolute ADC difference).
    stride : int
        Subsampling factor. 1 = exact, >1 = approximate but much faster.

    Returns
    -------
    baselines : np.ndarray [C, n_segments] float32
    """
    T, C = raw_data.shape
    n_seg = (T + segment_len - 1) // segment_len
    out = np.empty((C, n_seg), dtype=np.float32)

    for seg in prange(n_seg):
        start = seg * segment_len
        end = min(start + segment_len, T)

        # Process each channel independently inside the segment loop
        for c in range(C):
            s = 0.0
            cnt = 0

            # First sample is always included
            prev = raw_data[start, c]
            s += prev
            cnt += 1

            # Strided loop over the segment
            for i in range(start + stride, end, stride):
                cur = raw_data[i, c]
                if abs(cur - prev) < diff_thresh:
                    s += cur
                    cnt += 1
                prev = cur

            out[c, seg] = s / cnt if cnt > 0 else 0.0

    return out


# ----------------------------------------------------------------------
# Baseline subtraction (in‑place, parallel)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def subtract_segment_baselines_int16(
    raw_data,
    baselines_f32,
    segment_len=100_000
):
    """
    Subtract pre‑computed segment baselines from raw int16 data (in‑place).

    Parameters
    ----------
    raw_data : np.ndarray [T, C] int16
        The raw recording. Modified in‑place.
    baselines_f32 : np.ndarray [C, n_segments] float32
        Baselines computed by `compute_baselines_int16_deriv_robust`.
    segment_len : int
        Must match the value used during baseline computation.
    """
    T, C = raw_data.shape
    Cb, n_seg = baselines_f32.shape
    if Cb != C:
        raise ValueError("Channel count mismatch between raw_data and baselines")

    # Quantise baselines to int16 once (no per‑sample rounding)
    bl_i16 = np.empty((C, n_seg), dtype=np.int16)
    for seg in range(n_seg):
        for c in range(C):
            bl_i16[c, seg] = int(round(baselines_f32[c, seg]))

    # Subtract in a single parallel pass over time
    for t in prange(T):
        seg = t // segment_len
        if seg >= n_seg:
            seg = n_seg - 1
        for c in range(C):
            raw_data[t, c] -= bl_i16[c, seg]


# ----------------------------------------------------------------------
# Fast snippet extraction (parallel gather)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def extract_snippets_fast_ram(
    raw_data,           # [T, C] int16
    spike_times,        # [N] int64
    window,             # (pre, post)
    selected_channels   # [K] int32
):
    """
    Extract snippets from the raw data into a [K, L, N_valid] float32 array.

    Spikes that would extend beyond the recording boundaries are silently
    dropped. The returned array is pre‑allocated and filled in parallel.

    Parameters
    ----------
    raw_data : np.ndarray [T, C] int16
    spike_times : np.ndarray [N] int64
        Sample indices of spike peaks.
    window : tuple[int, int]
        (pre, post) samples relative to the spike time.
    selected_channels : np.ndarray [K] int32
        Channel indices to extract.

    Returns
    -------
    snippets : np.ndarray [K, L, N_valid] float32
    valid_times : np.ndarray [N_valid] int64
        The spike times that were actually used (within bounds).
    """
    pre, post = window
    L = post - pre + 1
    T, C_raw = raw_data.shape
    N = spike_times.size
    K = selected_channels.size

    # Pre‑filter valid spikes (Numba does not support boolean indexing in prange,
    # so we do this outside the jitted function – but we can keep it here because
    # the whole function is jitted and the filter is a small NumPy operation)
    valid_mask = (spike_times + pre >= 0) & (spike_times + post < T)
    valid_idx = np.where(valid_mask)[0]
    N_valid = valid_idx.size

    if N_valid == 0:
        return np.empty((K, L, 0), dtype=np.float32), spike_times[valid_mask]

    out = np.empty((K, L, N_valid), dtype=np.float32)

    # Parallel gather over valid spikes
    for idx in prange(N_valid):
        n = valid_idx[idx]
        t0 = spike_times[n] + pre
        for l in range(L):
            sample = t0 + l
            for k in range(K):
                ch = selected_channels[k]
                out[k, l, idx] = raw_data[sample, ch]

    return out, spike_times[valid_mask]