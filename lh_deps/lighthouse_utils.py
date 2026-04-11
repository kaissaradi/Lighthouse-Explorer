# -*- coding: utf-8 -*-
"""
lighthouse_utils.py
(adaptive deep-dive valley detection replaced with correct two-peak argmin approach)
"""

from __future__ import annotations
import numpy as np

# ------------------------
# 1) FIND VALLEY & EVENTS
# ------------------------
def find_valley_and_times(
    raw_data,                # (T_total, C), int16 or float
    ch,                      # int, channel index to analyze
    *,
    window=(-40, 80),        # (pre, post) samples
    start=0,                 # inclusive start sample for analysis
    stop=None,               # exclusive stop sample for analysis
    bin_width=10.0,          # ADC units per histogram bin
    valley_bins=5,           # (DEPRECATED - kept for signature compatibility)
    min_valid_count=300,     # require Left >= this many spikes

    ratio_base=3,            # base multiplier for required Left/Valley ratio
    ratio_step=100,          # each extra 100 valley spikes raises the ratio by +1
    ratio_floor=2,           # never require less than 2x
    ratio_cap=10,            # cap required ratio at 10x
    right_k=500,             # return >= this many "right-of-left" spikes
    min_trough=None,         # optional signed floor on minima (e.g. -2500)
):
    """
    Finds the valley between the spike bump (left) and noise floor bump (right)
    in the amplitude histogram of local minima.

    Strategy:
      1) Extract local minima from the raw trace.
      2) Build an amplitude histogram.
      3) Smooth and find local peaks/valleys.
      4) Identify the two dominant peaks (spike peak left, noise peak right).
      5) Valley = argmin of smoothed counts strictly between those two peaks,
         expanded by +/- valley_half_width bins.
      6) left events  = vals < valley_low   (spikes)
         valley events = valley_low <= vals < valley_high  (exclusion zone)
         right events  = vals >= valley_low  (baseline/noise support)
    """

    def _empty(s, e):
        return dict(
            accepted=False,
            valley_low=None, valley_high=None,
            left_times=np.array([], dtype=np.int64),
            left_vals=np.array([], dtype=np.float32),
            valley_times=np.array([], dtype=np.int64),
            valley_vals=np.array([], dtype=np.float32),
            left_count=0, valley_count=0, required_ratio=None,
            analysis_span=(s, e),
            rightk_times_by_amp=np.array([], dtype=np.int64),
            rightk_vals_by_amp=np.array([], dtype=np.float32),
            rightk_times_sorted=np.array([], dtype=np.int64),
            amp_hist_counts=np.array([], dtype=np.int32),
            amp_hist_edges=np.array([], dtype=np.float32),
            all_times=np.array([], dtype=np.int64),
            all_vals=np.array([], dtype=np.float32),
        )

    T, C = raw_data.shape
    if stop is None:
        stop = T
    if ch < 0 or ch >= C:
        raise ValueError(f"Channel {ch} out of range 0..{C-1}")

    pre, post = int(window[0]), int(window[1])

    # ---- 1) Extract Local Minima ----
    x = raw_data[start:stop, ch].astype(np.float32, copy=False)
    if x.size < 3:
        return _empty(start, stop)

    msk = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    idx_local = np.where(msk)[0] + 1
    if idx_local.size == 0:
        return _empty(start, stop)

    times_abs = start + idx_local.astype(np.int64)
    vals = x[idx_local].astype(np.float32)

    # drop events whose snippet would fall outside the analysis window
    ok = (times_abs + pre >= start) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals      = vals[ok]

    if min_trough is not None:
        keep = vals >= float(min_trough)
        times_abs = times_abs[keep]
        vals      = vals[keep]

    if times_abs.size == 0:
        return _empty(start, stop)

    all_times = times_abs.astype(np.int64, copy=False)
    all_vals  = vals.astype(np.float32, copy=False)

    # ---- 2) Build Histogram ----
    bw = float(bin_width)
    vmin = float(np.min(vals))
    low_edge  = bw * np.floor(vmin / bw)
    high_edge = 0.0

    n_bins = int(round((high_edge - low_edge) / bw))
    if n_bins < 3:
        low_edge -= bw * (3 - n_bins)
        n_bins = int(round((high_edge - low_edge) / bw))

    edges = np.linspace(low_edge, high_edge, n_bins + 1, dtype=np.float64)
    counts, edges = np.histogram(vals, bins=edges)

    if counts.size < 3:
        return _empty(start, stop)

    # ---- 3) Smooth & Find Local Peaks ----
    kernel = np.array([0.25, 0.5, 0.25])
    Sn = np.convolve(counts.astype(np.float64), kernel, mode='same')

    m = Sn.size
    loc_peak = np.zeros(m, dtype=bool)
    if m >= 3:
        loc_peak[1:-1] = (Sn[1:-1] > Sn[:-2]) & (Sn[1:-1] >= Sn[2:])
        loc_peak[0]  = Sn[0]  > Sn[1]
        loc_peak[-1] = Sn[-1] > Sn[-2]
    else:
        loc_peak[np.argmax(Sn)] = True

    peak_idxs = np.where(loc_peak)[0]
    if peak_idxs.size < 2:
        # Can't find two peaks → no separable valley
        return _empty(start, stop)

    # ---- 4) Identify Spike Peak (left) and Noise Peak (right) ----
    # Noise peak = rightmost dominant peak (highest bin index with large count).
    # We take the tallest peak as noise, then the tallest peak strictly left of it
    # as the spike peak.
    noise_peak_idx  = int(peak_idxs[np.argmax(Sn[peak_idxs])])
    left_candidates = peak_idxs[peak_idxs < noise_peak_idx]

    if left_candidates.size == 0:
        # All peaks are at or right of the tallest → no spike peak found
        return _empty(start, stop)

    spike_peak_idx = int(left_candidates[np.argmax(Sn[left_candidates])])

    # ---- 5) Valley = single bin at argmin between the two peaks ----
    between = np.arange(spike_peak_idx + 1, noise_peak_idx)
    if between.size == 0:
        return _empty(start, stop)

    trough_idx = int(between[np.argmin(Sn[between])])

    valley_low  = float(edges[trough_idx])
    valley_high = float(edges[trough_idx + 1])

    # ---- 6) Classify Events ----
    left_mask   = vals < valley_low
    valley_mask = (vals >= valley_low) & (vals < valley_high)

    left_times_  = times_abs[left_mask]
    left_vals_   = vals[left_mask]
    valley_times_= times_abs[valley_mask]
    valley_vals_ = vals[valley_mask]

    left_count   = int(left_mask.sum())
    valley_count = int(valley_mask.sum())

    req = min(ratio_cap, max(ratio_floor,
              ratio_base + (valley_count // int(max(1, ratio_step)))))
    accepted = (left_count >= max(min_valid_count, req * max(1, valley_count)))

    # ---- 7) Build Right-of-Valley Support (baseline / template selection) ----
    right_mask      = vals >= valley_low
    right_times_full= times_abs[right_mask]
    right_vals_full = vals[right_mask]
    r_order         = np.argsort(right_vals_full)          # most-negative first

    keep_n = min(int(right_k), right_times_full.size)
    rightk_times_by_amp = right_times_full[r_order][:keep_n].astype(np.int64)
    rightk_vals_by_amp  = right_vals_full[r_order][:keep_n].astype(np.float32)

    return dict(
        accepted=bool(accepted),
        valley_low=valley_low,
        valley_high=valley_high,
        left_times=left_times_[np.argsort(left_times_)].astype(np.int64),
        left_vals=left_vals_[np.argsort(left_times_)].astype(np.float32),
        valley_times=valley_times_[np.argsort(valley_times_)].astype(np.int64),
        valley_vals=valley_vals_[np.argsort(valley_times_)].astype(np.float32),
        left_count=left_count,
        valley_count=valley_count,
        required_ratio=float(req),
        analysis_span=(start, stop),
        rightk_times_by_amp=rightk_times_by_amp,
        rightk_vals_by_amp=rightk_vals_by_amp,
        rightk_times_sorted=rightk_times_by_amp[np.argsort(rightk_times_by_amp)],
        amp_hist_counts=counts.astype(np.int32),
        amp_hist_edges=edges.astype(np.float32),
        all_times=all_times,
        all_vals=all_vals,
    )