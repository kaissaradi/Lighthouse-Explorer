# -*- coding: utf-8 -*-
"""
lighthouse_utils.py — Robust spike detection and valley finding for Lighthouse QC.

Improvements over original:
- Peak‑to‑peak detection (catches biphasic spikes)
- Adaptive noise floor via MAD (gain‑agnostic)
- Optional GMM‑based valley detection (more robust histogram splitting)
"""

from __future__ import annotations
import numpy as np

# Try importing sklearn for GMM (optional)
try:
    from sklearn.mixture import GaussianMixture
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ----------------------------------------------------------------------
# Helper: adaptive noise floor from quiet period
# ----------------------------------------------------------------------
def _estimate_noise_mad(signal, quiet_start=0, quiet_duration=10.0, fs=20000):
    """
    Estimate noise level using Median Absolute Deviation (MAD) on a quiet segment.

    Parameters
    ----------
    signal : 1D array
    quiet_start : float
        Start time (seconds) of quiet period.
    quiet_duration : float
        Duration (seconds) of quiet period.
    fs : float
        Sampling rate.

    Returns
    -------
    mad : float
        Median absolute deviation of the quiet segment.
    """
    start_samp = int(quiet_start * fs)
    end_samp = int((quiet_start + quiet_duration) * fs)
    start_samp = max(0, min(start_samp, len(signal)))
    end_samp = max(start_samp + 1, min(end_samp, len(signal)))
    quiet = signal[start_samp:end_samp]
    if quiet.size == 0:
        quiet = signal[:min(10*fs, len(signal))]
    return np.median(np.abs(quiet - np.median(quiet)))


# ----------------------------------------------------------------------
# Peak‑to‑peak detection (Numba‑compatible if needed, here pure NumPy)
# ----------------------------------------------------------------------
def _detect_spikes_ptp(signal, fs=20000, ptp_window_s=0.001, ptp_threshold_sigma=5.0,
                       refractory_s=0.002, min_trough=None):
    """
    Detect spikes using peak‑to‑peak amplitude in a sliding window.

    Parameters
    ----------
    signal : 1D float32 array
    fs : float
    ptp_window_s : float
        Window length for peak‑to‑peak calculation.
    ptp_threshold_sigma : float
        Threshold as multiple of noise sigma (MAD*1.4826).
    refractory_s : float
        Minimum distance between detected spikes.
    min_trough : float or None
        Optional absolute floor on trough value (e.g., -2500). If provided,
        spikes with trough above this are discarded.

    Returns
    -------
    spike_times : 1D int64 array
        Sample indices of detected spikes.
    spike_vals : 1D float32 array
        Trough values at each spike time (for amplitude histogram).
    """
    win = int(ptp_window_s * fs)
    if win < 2:
        win = 2
    n = len(signal)
    # Compute rolling peak‑to‑peak efficiently using sliding window min/max
    # Simple but memory-light: loop in Python (for N<10M, it's fine)
    ptp = np.zeros(n - win + 1, dtype=np.float32)
    for i in range(len(ptp)):
        ptp[i] = signal[i:i+win].max() - signal[i:i+win].min()

    # Noise level from MAD of the PTP trace (more robust than raw signal)
    noise_mad = _estimate_noise_mad(ptp, quiet_duration=10.0, fs=fs)
    noise_sigma = noise_mad * 1.4826  # convert MAD to std dev for normal distribution
    threshold = noise_sigma * ptp_threshold_sigma

    # Find threshold crossings
    crossings = np.where(ptp > threshold)[0]
    if crossings.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Refractory period
    refrac = int(refractory_s * fs)
    spikes = []
    last = -refrac
    for idx in crossings:
        if idx - last >= refrac:
            spikes.append(idx + win//2)  # center of window
            last = idx
    spike_times = np.array(spikes, dtype=np.int64)

    # Extract trough values at spike times
    spike_vals = signal[spike_times].astype(np.float32)

    # Apply optional absolute trough threshold
    if min_trough is not None:
        keep = spike_vals >= min_trough
        spike_times = spike_times[keep]
        spike_vals = spike_vals[keep]

    return spike_times, spike_vals


# ----------------------------------------------------------------------
# GMM valley detection (optional)
# ----------------------------------------------------------------------
def _find_valley_gmm(vals, bin_width=10.0):
    """
    Fit a 2‑component GMM to amplitude values and return the valley threshold.

    The valley is defined as the point where the two Gaussian PDFs intersect
    (i.e., equal likelihood). Returns the lower bound of the valley bin.
    """
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not installed; cannot use GMM valley.")
    vals = vals.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(vals)
    means = gmm.means_.flatten()
    covs = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Sort components by mean: left = noise/spike? Actually we want the one with
    # more negative mean as spike component (left), more positive as noise (right).
    idx = np.argsort(means)
    mean_left, mean_right = means[idx[0]], means[idx[1]]
    var_left, var_right = covs[idx[0]], covs[idx[1]]

    # Find intersection of two Gaussians (solve quadratic)
    # w1 * N(x|μ1,σ1²) = w2 * N(x|μ2,σ2²)
    # Taking log: -0.5*(x-μ1)²/σ1² - 0.5*log(2πσ1²) + log(w1) = ...
    # This leads to a quadratic equation: a*x² + b*x + c = 0
    a = (1/(2*var_right)) - (1/(2*var_left))
    b = (mean_left/var_left) - (mean_right/var_right)
    c = (mean_right**2/(2*var_right)) - (mean_left**2/(2*var_left)) + \
        np.log(weights[idx[1]]/weights[idx[0]]) - 0.5*np.log(var_right/var_left)

    # Solve quadratic
    disc = b**2 - 4*a*c
    if disc < 0 or a == 0:
        # Fallback: simple midpoint
        valley = (mean_left + mean_right) / 2
    else:
        sqrt_disc = np.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2*a)
        x2 = (-b - sqrt_disc) / (2*a)
        # Choose the root between the two means
        if mean_left < x1 < mean_right:
            valley = x1
        elif mean_left < x2 < mean_right:
            valley = x2
        else:
            valley = (mean_left + mean_right) / 2

    # Quantize to bin_width
    valley_low = np.floor(valley / bin_width) * bin_width
    return float(valley_low)


# ----------------------------------------------------------------------
# Main valley detection function (same signature, improved internals)
# ----------------------------------------------------------------------
def find_valley_and_times(
    raw_data,                # (T_total, C), int16 or float
    ch,                      # int, channel index to analyze
    *,
    window=(-40, 80),        # (pre, post) samples for snippet extraction (used later)
    start=0,                 # inclusive start sample for analysis
    stop=None,               # exclusive stop sample for analysis
    bin_width=10.0,          # ADC units per histogram bin
    valley_bins=5,           # (DEPRECATED - kept for compatibility)
    min_valid_count=300,     # require Left >= this many spikes

    ratio_base=3,            # base multiplier for required Left/Valley ratio
    ratio_step=100,          # each extra 100 valley spikes raises the ratio by +1
    ratio_floor=2,           # never require less than 2x
    ratio_cap=10,            # cap required ratio at 10x
    right_k=500,             # return >= this many "right-of-left" spikes
    min_trough=None,         # optional signed floor on minima (e.g. -2500)

    # --- New parameters for improved detection ---
    use_ptp_detection=True,          # use peak‑to‑peak instead of local minima
    ptp_window_s=0.001,              # window for PTP (seconds)
    ptp_threshold_sigma=5.0,         # threshold in units of noise sigma
    refractory_s=0.002,              # refractory period (seconds)
    use_gmm_valley=True,             # use GMM for valley (falls back if sklearn missing)
    quiet_start_s=0.0,               # start of quiet period for noise estimation
    quiet_duration_s=10.0,           # duration of quiet period
    fs=20000,                        # sampling rate (Hz)
):
    """
    Finds the valley between spike and noise distributions in amplitude space.

    Improvements over original:
    - Peak‑to‑peak detection catches biphasic spikes.
    - Adaptive noise floor (MAD) makes thresholds gain‑agnostic.
    - Optional GMM valley gives robust separation.

    Returns dictionary with same keys as original version.
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

    # ---- 1) Extract signal for this channel ----
    x = raw_data[start:stop, ch].astype(np.float32, copy=False)
    if x.size < 3:
        return _empty(start, stop)

    # ---- 2) Spike detection (peak‑to‑peak or legacy local minima) ----
    if use_ptp_detection:
        times_abs, vals = _detect_spikes_ptp(
            x, fs=fs, ptp_window_s=ptp_window_s,
            ptp_threshold_sigma=ptp_threshold_sigma,
            refractory_s=refractory_s, min_trough=min_trough
        )
        # Convert local indices to absolute
        times_abs = start + times_abs
    else:
        # Legacy local minima detection (kept for compatibility)
        msk = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
        idx_local = np.where(msk)[0] + 1
        if idx_local.size == 0:
            return _empty(start, stop)
        times_abs = start + idx_local.astype(np.int64)
        vals = x[idx_local].astype(np.float32)
        if min_trough is not None:
            keep = vals >= float(min_trough)
            times_abs = times_abs[keep]
            vals = vals[keep]

    # Filter events that would produce snippets outside analysis window
    ok = (times_abs + pre >= start) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals = vals[ok]

    if times_abs.size == 0:
        return _empty(start, stop)

    all_times = times_abs.astype(np.int64, copy=False)
    all_vals = vals.astype(np.float32, copy=False)

    # ---- 3) Build amplitude histogram ----
    bw = float(bin_width)
    vmin = float(np.min(vals))
    low_edge = bw * np.floor(vmin / bw)
    high_edge = 0.0
    n_bins = int(round((high_edge - low_edge) / bw))
    if n_bins < 3:
        low_edge -= bw * (3 - n_bins)
        n_bins = int(round((high_edge - low_edge) / bw))
    edges = np.linspace(low_edge, high_edge, n_bins + 1, dtype=np.float64)
    counts, edges = np.histogram(vals, bins=edges)

    if counts.size < 3:
        return _empty(start, stop)

    # ---- 4) Find valley threshold ----
    if use_gmm_valley and _SKLEARN_AVAILABLE:
        try:
            valley_low = _find_valley_gmm(vals, bin_width=bw)
            # Find bin index for valley_low to compute valley_high
            valley_idx = np.searchsorted(edges, valley_low) - 1
            valley_idx = max(0, min(valley_idx, len(edges)-2))
            valley_high = float(edges[valley_idx + 1])
        except Exception:
            # Fallback to legacy method
            use_gmm_valley = False

    if not (use_gmm_valley and _SKLEARN_AVAILABLE):
        # Legacy peak‑finding method
        kernel = np.array([0.25, 0.5, 0.25])
        Sn = np.convolve(counts.astype(np.float64), kernel, mode='same')
        m = Sn.size
        loc_peak = np.zeros(m, dtype=bool)
        if m >= 3:
            loc_peak[1:-1] = (Sn[1:-1] > Sn[:-2]) & (Sn[1:-1] >= Sn[2:])
            loc_peak[0] = Sn[0] > Sn[1]
            loc_peak[-1] = Sn[-1] > Sn[-2]
        else:
            loc_peak[np.argmax(Sn)] = True
        peak_idxs = np.where(loc_peak)[0]
        if peak_idxs.size < 2:
            return _empty(start, stop)

        noise_peak_idx = int(peak_idxs[np.argmax(Sn[peak_idxs])])
        left_candidates = peak_idxs[peak_idxs < noise_peak_idx]
        if left_candidates.size == 0:
            return _empty(start, stop)
        spike_peak_idx = int(left_candidates[np.argmax(Sn[left_candidates])])
        between = np.arange(spike_peak_idx + 1, noise_peak_idx)
        if between.size == 0:
            return _empty(start, stop)
        trough_idx = int(between[np.argmin(Sn[between])])
        valley_low = float(edges[trough_idx])
        valley_high = float(edges[trough_idx + 1])

    # ---- 5) Classify events ----
    left_mask = vals < valley_low
    valley_mask = (vals >= valley_low) & (vals < valley_high)

    left_times_ = times_abs[left_mask]
    left_vals_ = vals[left_mask]
    valley_times_ = times_abs[valley_mask]
    valley_vals_ = vals[valley_mask]

    left_count = int(left_mask.sum())
    valley_count = int(valley_mask.sum())

    # Required ratio logic (unchanged)
    req = min(ratio_cap, max(ratio_floor,
              ratio_base + (valley_count // int(max(1, ratio_step)))))
    accepted = (left_count >= max(min_valid_count, req * max(1, valley_count)))

    # ---- 6) Right‑of‑valley support (baseline spikes) ----
    right_mask = vals >= valley_low
    right_times_full = times_abs[right_mask]
    right_vals_full = vals[right_mask]
    r_order = np.argsort(right_vals_full)  # most negative first
    keep_n = min(int(right_k), right_times_full.size)
    rightk_times_by_amp = right_times_full[r_order][:keep_n].astype(np.int64)
    rightk_vals_by_amp = right_vals_full[r_order][:keep_n].astype(np.float32)

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