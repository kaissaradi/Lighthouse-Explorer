# -*- coding: utf-8 -*-
"""
lighthouse_utils.py - Cleaned version
(unused functions deleted)
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
    window=(-40, 80),        # (pre, post) samples; used to ensure event windows fit inside [start, stop)
    start=0,                 # inclusive start sample for analysis
    stop=None,               # exclusive stop sample for analysis; None → end of recording
    bin_width=10.0,          # ADC units per histogram bin
    valley_bins=5,           # number of consecutive bins forming the “valley” (e.g. 5 → 50 ADC)
    min_valid_count=300,     # require Left ≥ this many spikes
    ratio_base=3,            # base multiplier for required Left/Valley ratio (e.g. 3)
    ratio_step=100,          # each extra 100 valley spikes raises the ratio by +1
    ratio_floor=2,           # never require less than 2×
    ratio_cap=10,            # **cap** required ratio at 10×
    right_k=500,             # return ≥ this many “right-of-left” spikes (valley or valley+fill)
    min_trough=None          # optional signed floor on minima; exclude vals < min_trough (e.g. -1200)
):
    """
    Scan all negative-amplitude bins (fixed width), compute 5-bin sums, pick the **leftmost local peak**
    (deep cluster center), then the **lowest valley to its right**. Accept if:

        Left_count ≥ required_ratio(Valley_count) * Valley_count  AND  Left_count ≥ min_valid_count,

    where required_ratio(V) = min(ratio_cap, max(ratio_floor, ratio_base + floor(V/ratio_step))).

    Returns:
      {
        'accepted': bool,
        'valley_low': float, 'valley_high': float,
        'left_times':   int64[NL], 'left_vals':   float[NL],        # chronological
        'valley_times': int64[NV], 'valley_vals': float[NV],        # chronological (just the valley band)
        'left_count': int, 'valley_count': int, 'required_ratio': float,
        'analysis_span': (start, stop),

        # NEW: “right-of-left” selection for downstream windows:
        #  — if valley_count ≥ right_k: return the ENTIRE valley (≥ right_k)
        #  — else return VALLEY + the most-negative spikes to the right of the valley (>= valley_high)
        #     until total reaches right_k
        'rightk_times_by_amp':  int64[NR],  # amplitude-sorted (most negative first)
        'rightk_vals_by_amp':   float[NR],  # aligned values
        'rightk_times_sorted':  int64[NR],  # chronological view of the same selection
      }
    """

    def _empty_step1(_start, _stop):
        return dict(
            accepted=False,
            valley_low=None, valley_high=None,
            left_times=np.array([], dtype=np.int64),
            left_vals=np.array([], dtype=np.float32),
            valley_times=np.array([], dtype=np.int64),
            valley_vals=np.array([], dtype=np.float32),
            left_count=0, valley_count=0, required_ratio=None,
            analysis_span=(_start, _stop),
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
    if pre >= 0 or post <= 0:
        raise ValueError("`window` must straddle the event (use negative pre, positive post)")

    # ---- 1) Local minima (no refractory), filtered to fit the EI window in [start, stop)
    x = raw_data[start:stop, ch].astype(np.float32, copy=False)
    if x.size < 3:
        return _empty_step1(start, stop)

    msk = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    idx_local = np.where(msk)[0] + 1
    if idx_local.size == 0:
        return _empty_step1(start, stop)

    times_abs = start + idx_local.astype(np.int64)
    vals = x[idx_local].astype(np.float32)

    ok = (times_abs + pre >= start) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals      = vals[ok]

    # Optional artifact floor on signed minima:
    # keep only minima that are not "too deep", e.g. min_trough=-1200 excludes vals < -1200
    if min_trough is not None:
        keep_amp = vals >= float(min_trough)
        times_abs = times_abs[keep_amp]
        vals      = vals[keep_amp]

    if times_abs.size == 0:
        return _empty_step1(start, stop)
    
    # Keep all detected minima that survived window + amplitude filtering
    all_times = times_abs.astype(np.int64, copy=False)
    all_vals  = vals.astype(np.float32, copy=False)

    # ---- 2) Negative-side histogram up to 0
    vmin = float(np.min(vals))
    if vmin >= 0:
        return _empty_step1(start, stop)

    bw = float(bin_width)
    low_edge  = bw * np.floor(vmin / bw)
    high_edge = 0.0
    edges = np.linspace(low_edge, high_edge, int(round((high_edge - low_edge) / bw)) + 1, dtype=np.float64)
    if edges.size < valley_bins + 1:
        low_edge = low_edge - bw * ((valley_bins + 1) - edges.size + 1)
        edges = np.linspace(low_edge, high_edge, int(round((high_edge - low_edge) / bw)) + 1, dtype=np.float64)

    counts, edges = np.histogram(vals, bins=edges)
    nb = counts.size
    if nb < valley_bins + 1:
        return _empty_step1(start, stop)


    # ---- 3) Sliding 5-bin sums over fully negative windows
    kernel = np.ones(int(valley_bins), dtype=np.int64)
    S = np.convolve(counts, kernel, mode='valid')         # length = nb - valley_bins + 1
    win_low_edges  = edges[:-valley_bins]
    win_high_edges = edges[valley_bins:]
    neg_mask = (win_high_edges <= 0.0)
    if not np.any(neg_mask):
        return _empty_step1(start, stop)


    Sn    = S[neg_mask]
    lows  = win_low_edges[neg_mask]
    highs = win_high_edges[neg_mask]
    m     = Sn.size

    # ---- 4) Right-anchored noise peak, then valley to its LEFT (robust split)
    # ---- 4) Peaks & valleys of Sn (over negative-only windows)
    loc_peak = np.zeros(m, dtype=bool)
    if m >= 3:
        loc_peak[1:-1] = (Sn[1:-1] > Sn[:-2]) & (Sn[1:-1] >= Sn[2:])
        if Sn[0] > Sn[1]:
            loc_peak[0] = True
        if Sn[-1] > Sn[-2]:
            loc_peak[-1] = True
    else:
        loc_peak[np.argmax(Sn)] = True
    peak_idxs = np.where(loc_peak)[0]

    loc_valley = np.zeros(m, dtype=bool)
    if m >= 3:
        loc_valley[1:-1] = (Sn[1:-1] < Sn[:-2]) & (Sn[1:-1] <= Sn[2:])
        if Sn[0] < Sn[1]:
            loc_valley[0] = True
        if Sn[-1] < Sn[-2]:
            loc_valley[-1] = True
    else:
        loc_valley[np.argmin(Sn)] = True
    valley_idxs = np.where(loc_valley)[0]
    if valley_idxs.size == 0:
        return _empty_step1(start, stop)


    # ---- 5) Choose the LEFTMOST valley that still has a sizeable left bump
    # "Sizeable" = the mass of events strictly left of the valley_low is >= min_valid_count.
    # We compute this cheaply from the histogram prefix-sum (no per-event scan).
    prefix = np.concatenate(([0], np.cumsum(counts)))  # length nb+1, prefix[k]=sum(counts[:k])

    chosen = None
    for j in valley_idxs:  # iterate from leftmost to right
        vlow = float(lows[j])   # this equals an entry in `edges`
        # index in `counts` that corresponds to bins strictly below vlow
        k_idx = int(np.searchsorted(edges, vlow, side='left'))
        left_est = int(prefix[k_idx])          # estimated #events with vals < vlow
        # require a left peak to exist (nearest one to the left in Sn-space)
        has_left_peak = np.any(peak_idxs < j)
        if has_left_peak and left_est >= int(min_valid_count):
            chosen = j
            break

    if chosen is None:
        # Fallback: keep previous behavior (pick nearest valley left of the right-anchored noise peak)
        # Identify a right-anchored dominant peak and take the nearest valley to its left.
        last_frac = 0.35
        right_bound = max(0, int(np.floor((1.0 - last_frac) * m)))
        right_candidates = peak_idxs[peak_idxs >= right_bound]
        if right_candidates.size == 0:
            right_peak = int(peak_idxs[np.argmax(Sn[peak_idxs])])
        else:
            right_peak = int(right_candidates[np.argmax(Sn[right_candidates])])
        # nearest valley strictly to the left
        cand = valley_idxs[valley_idxs < right_peak]
        if cand.size == 0:
            return _empty_step1(start, stop)

        valley_rel = int(cand[-1])  # nearest to the right peak
    else:
        valley_rel = int(chosen)

    valley_low  = float(lows[valley_rel])
    valley_high = float(highs[valley_rel])



    # ---- 6) Event-level counts using actual minima
    left_ev_mask     = (vals < valley_low)
    valley_ev_mask   = (vals >= valley_low) & (vals < valley_high)
    right_of_val_mask= (vals >= valley_high)   # strictly to the right of the valley band

    left_times  = times_abs[left_ev_mask]
    left_vals_  = vals[left_ev_mask]
    valley_times_ = times_abs[valley_ev_mask]
    valley_vals_  = vals[valley_ev_mask]
    right_times_  = times_abs[right_of_val_mask]
    right_vals_   = vals[right_of_val_mask]

    left_count   = int(left_ev_mask.sum())
    valley_count = int(valley_ev_mask.sum())

    req = min(ratio_cap, max(ratio_floor, ratio_base + (valley_count // int(max(1, ratio_step)))))
    accepted = (left_count >= max(min_valid_count, req * max(1, valley_count)))

    # Sort outputs chronologically
    li = np.argsort(left_times);    vi = np.argsort(valley_times_)
    left_times  = left_times[li].astype(np.int64, copy=False)
    left_vals_  = left_vals_[li].astype(np.float32, copy=False)
    valley_times_ = valley_times_[vi].astype(np.int64, copy=False)
    valley_vals_  = valley_vals_[vi].astype(np.float32, copy=False)

    # ---- 7) Build the “top-right” selection for BL/TR support:
    # exactly right_k strongest spikes on the RIGHT side (vals >= valley_low)
    right_ev_mask_full = (vals >= valley_low)
    right_times_full = times_abs[right_ev_mask_full]
    right_vals_full  = vals[right_ev_mask_full]

    r_full_order = np.argsort(right_vals_full)   # ascending => most negative first
    right_times_amp = right_times_full[r_full_order]
    right_vals_amp  = right_vals_full[r_full_order]

    keep_n = min(int(right_k), right_times_amp.size)
    rightk_times_by_amp = right_times_amp[:keep_n].astype(np.int64, copy=False)
    rightk_vals_by_amp  = right_vals_amp[:keep_n].astype(np.float32, copy=False)

    rk_sort = np.argsort(rightk_times_by_amp)
    rightk_times_sorted = rightk_times_by_amp[rk_sort]

    return dict(
        accepted=bool(accepted),
        valley_low=valley_low, valley_high=valley_high,
        left_times=left_times, left_vals=left_vals_,
        valley_times=valley_times_, valley_vals=valley_vals_,
        left_count=int(left_count), valley_count=int(valley_count), required_ratio=float(req),
        analysis_span=(start, stop),
        rightk_times_by_amp=rightk_times_by_amp,
        rightk_vals_by_amp=rightk_vals_by_amp,
        rightk_times_sorted=rightk_times_sorted,
        amp_hist_counts=counts.astype(np.int32, copy=False),
        amp_hist_edges=edges.astype(np.float32, copy=False),
        all_times=all_times,
        all_vals=all_vals,
    )
