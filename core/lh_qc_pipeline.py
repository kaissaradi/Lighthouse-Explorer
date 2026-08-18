"""
lh_qc_pipeline.py — 4-step LH QC pipeline matching lh-uw.py logic.

All fixes applied:
- No subsampling of valley / rightk spikes.
- Adaptive window from left spikes.
- K-means precheck + verdict (rejects two-unit channels).
- Early reject: ISI 10-30 pairs, max valley count.
- Final spike count threshold (>=200).
- Full BL/TR support labeling (not simplified).
- Memory efficient: no [C, L, N] for all spikes.
"""
from __future__ import annotations

"""
Thread-safe entry points for vendored Numba kernels in ``lh_deps/``.

Numba's ``workqueue`` threading layer is fork-friendlier than TBB but is **not**
safe for concurrent calls from multiple Python threads (e.g. ``QThreadPool``).
All pool workers must use these locked wrappers — see ``docs/specs/numba_fork_safety.md``.
"""


import functools
import threading

_NUMBA_LOCK = threading.RLock()


def numba_parallel_lock() -> threading.RLock:
    """The lock serializing Numba ``parallel=True`` calls across worker threads."""
    return _NUMBA_LOCK


def _locked_function(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _NUMBA_LOCK:
            return fn(*args, **kwargs)

    return wrapper


try:
    from lh_deps.axolotl_utils_ram import (
        compute_baselines_int16_deriv_robust as _compute_baselines_int16_deriv_robust,
        subtract_segment_baselines_int16 as _subtract_segment_baselines_int16,
        extract_snippets_fast_ram as _extract_snippets_fast_ram,
    )
except ImportError:
    from axolotl_utils_ram import (  # type: ignore[no-redef]
        compute_baselines_int16_deriv_robust as _compute_baselines_int16_deriv_robust,
        subtract_segment_baselines_int16 as _subtract_segment_baselines_int16,
        extract_snippets_fast_ram as _extract_snippets_fast_ram,
    )

compute_baselines_int16_deriv_robust = _locked_function(
    _compute_baselines_int16_deriv_robust
)
subtract_segment_baselines_int16 = _locked_function(_subtract_segment_baselines_int16)
extract_snippets_fast_ram = _locked_function(_extract_snippets_fast_ram)



from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ValleyResult:
    """Output of find_valley_and_times for one channel."""
    accepted: bool
    valley_low: Optional[float]
    valley_high: Optional[float]
    left_times: np.ndarray      # int64 [NL]
    left_vals: np.ndarray       # float32 [NL]
    valley_times: np.ndarray    # int64 [NV]
    valley_vals: np.ndarray     # float32 [NV]
    all_times: np.ndarray       # int64 — ALL threshold crossings
    all_vals: np.ndarray        # float32
    amp_hist_counts: np.ndarray
    amp_hist_edges: np.ndarray
    left_count: int
    valley_count: int
    rightk_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class SnippetResult:
    """Output of snippet extraction step."""
    snippets: np.ndarray        # float32 [C, L, N] — all channels, all spikes
    times: np.ndarray           # int64 [N] — valid spike times (boundary-filtered)
    n_channels: int
    snippet_len: int


@dataclass
class PCAKMeansResult:
    """Output of PCA + KMeans step."""
    pca_coords: np.ndarray      # float32 [N, n_pcs] — for scatter plot
    km_labels: np.ndarray       # int [N] — cluster 0 or 1
    cluster_mean_waveforms: list  # [np.ndarray, np.ndarray] each shape [L] on detect_ch
    explained_variance_ratio: np.ndarray
    n_pcs_used: int


@dataclass
class BLTRResult:
    """Output of BL/TR support labeling step.

    Probe-level labels/times are stored for diagnostics. Keep/reject arrays
    follow the notebook convention and are used to build ``QCResult.final_times``.
    """
    labels: np.ndarray          # object dtype — probe labels (BL then TR)
    bl_bulk: np.ndarray         # float32 [N_probe]
    tr_bulk: np.ndarray         # float32 [N_probe]
    counts: dict                # {'LH': int, 'soup': int, 'uncertain_boundary': int, ...}
    times: np.ndarray           # int64 [N_probe] — same indexing as labels
    ok: bool = False            # True when probe BL/TR labeling succeeded
    bl_keep_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    bl_uncertain_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    bl_reject_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    tr_keep_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    tr_uncertain_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    tr_reject_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class QCResult:
    """Full result for one channel QC run."""
    channel: int
    n_sorter_spikes: int        # how many spikes the existing sorter (KS/LH) found on this ch
    valley: ValleyResult
    snippets: SnippetResult
    pca_km: PCAKMeansResult
    bltr: BLTRResult
    reject_reason: Optional[str] = None         # set on early pipeline rejection
    sorter_times: Optional[np.ndarray] = None   # spike times from sorter, attached by main_window
    fs: float = 20_000.0                        # sampling rate, set by main_window

    # Clean LH spike times after BL/TR filtering (empty if rejected).
    final_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # Mean EI over final_times [n_channels, n_samples]; used for Phy/KS templates.
    final_ei: Optional[np.ndarray] = None

    # ── KMeans verdict info ──────────────────────────────────────────────────
    # Populated by run_pca_kmeans_on_left_spikes. Keys:
    #   proceed      : bool   — whether pipeline continued past KMeans
    #   verdict      : str    — e.g. 'single_unit', 'two_unit_reject', 'low_n'
    #   cos_sim      : float  — cosine similarity between cluster mean waveforms
    #   n_spikes     : int    — number of spikes fed into KMeans
    #   cluster_sizes: list[int]
    km_info: dict = field(default_factory=dict)

    # ── Biophysics payload ───────────────────────────────────────────────────
    # Populated by compute_biophysics() at the end of run_qc_pipeline. Keys:
    #   firing_rate_hz      : float  — mean FR over recording duration
    #   isi_violations_pct  : float  — % ISI pairs < 1.5 ms (refractory violations)
    #   burst_fraction_pct  : float  — % ISI pairs < 5.0 ms (burst criterion)
    #   trough_to_peak_ms   : float  — waveform trough→peak time in ms
    #   peak_to_trough_ratio: float  — abs(peak) / abs(trough)
    #   median_amp_adc      : float  — median absolute trough amplitude (ADC units)
    biophysics: dict = field(default_factory=dict)

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def n_lh(self) -> int:
        """Number of clean LH spikes. Prefers filtered final_times when present."""
        if self.reject_reason is None and self.final_times.size > 0:
            return int(self.final_times.size)
        return int(self.bltr.counts.get("LH", 0))

    @property
    def n_soup(self) -> int:
        return int(self.bltr.counts.get("soup", 0))

    @property
    def n_uncertain(self) -> int:
        counts = self.bltr.counts
        return int(counts.get("uncertain_boundary", 0) + counts.get("uncertain_lowBL", 0))

    @property
    def n_total(self) -> int:
        """Total spikes accounted for in QC display (LH + soup + uncertain).

        Always derived from the category properties so UI percentages stay
        consistent. ``n_lh`` prefers filtered ``final_times`` when present.
        """
        return int(self.n_lh + self.n_soup + self.n_uncertain)

    @property
    def miss_rate(self) -> Optional[float]:
        """Fraction of LH spikes not found by sorter.

        When both ``final_times`` and ``sorter_times`` are available, uses
        coincidence matching (±1 ms). Otherwise falls back to a count ratio.
        """
        if self.n_sorter_spikes < 0:
            return None
        if self.n_lh == 0:
            return 0.0
        if (
            self.final_times.size > 0
            and self.sorter_times is not None
            and np.asarray(self.sorter_times).size > 0
        ):
            try:
                from core.loader import match_spikes
            except ImportError:
                from loader import match_spikes  # type: ignore
            fs = float(self.fs) if self.fs else 20_000.0
            coincidence = max(1, int(0.001 * fs))
            _n_matched, n_lh_only, _, _ = match_spikes(
                np.sort(np.asarray(self.final_times, dtype=np.int64)),
                np.sort(np.asarray(self.sorter_times, dtype=np.int64)),
                coincidence,
            )
            return float(n_lh_only / max(1, self.n_lh))
        missed = max(0, self.n_lh - self.n_sorter_spikes)
        return missed / self.n_lh

    @property
    def sorter_yield_ratio(self) -> Optional[float]:
        """n_sorter / n_lh. >1 means sorter found more than LH (possible false positives)."""
        if self.n_sorter_spikes < 0 or self.n_lh == 0:
            return None
        return self.n_sorter_spikes / self.n_lh


from typing import Optional, Tuple
import numpy as np

# ── Imports from notebook helpers ───────────────────────────────────────────
try:
    from lh_deps.lighthouse_utils import find_valley_and_times
except ImportError:
    from lighthouse_utils import find_valley_and_times  # type: ignore




# ============================================================================
# Helper functions from lh-uw.py (copied verbatim)
# ============================================================================

def _flatten_masked_snips(snips_12_l_n, mask_12_l):
    snips_12_l_n = np.asarray(snips_12_l_n, dtype=np.float32)
    mask_12_l = np.asarray(mask_12_l, dtype=bool)
    assert snips_12_l_n.ndim == 3
    assert mask_12_l.shape == snips_12_l_n.shape[:2]
    X = snips_12_l_n.transpose(2, 0, 1).reshape(snips_12_l_n.shape[2], -1)
    return X[:, mask_12_l.ravel()].astype(np.float32)


def _row_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


def _topk_mean_curve(sorted_desc):
    sorted_desc = np.asarray(sorted_desc, dtype=np.float32)
    return np.cumsum(sorted_desc) / np.arange(1, sorted_desc.size + 1, dtype=np.float32)


def _resolve_k_list(k_list, n_available):
    valid = [int(k) for k in k_list if int(k) >= 1 and int(k) <= int(n_available)]
    # Fallback: if all requested k's are too large, just use all available spikes
    if not valid and int(n_available) >= 1:
        return [int(n_available)]
    return valid


def _support_metrics_from_curves(bl_curve, tr_curve, k_peak, k_bulk):
    kmax = min(bl_curve.size, tr_curve.size)
    if kmax < 1:
        raise ValueError("Need at least one neighbor on each side.")
    k_peak_use = _resolve_k_list(k_peak, kmax)
    k_bulk_use = _resolve_k_list(k_bulk, kmax)
    if len(k_peak_use) == 0 or len(k_bulk_use) == 0:
        raise ValueError(f"k lists invalid for kmax={kmax}")

    bl_peak = float(np.mean([bl_curve[k - 1] for k in k_peak_use]))
    tr_peak = float(np.mean([tr_curve[k - 1] for k in k_peak_use]))
    bl_bulk = float(np.mean([bl_curve[k - 1] for k in k_bulk_use]))
    tr_bulk = float(np.mean([tr_curve[k - 1] for k in k_bulk_use]))
    d_peak = bl_peak - tr_peak
    d_bulk = bl_bulk - tr_bulk

    return dict(
        kmax=int(kmax),
        k_peak_used=k_peak_use,
        k_bulk_used=k_bulk_use,
        BL_peak=bl_peak,
        TR_peak=tr_peak,
        BL_bulk=bl_bulk,
        TR_bulk=tr_bulk,
        D_peak=d_peak,
        D_bulk=d_bulk,
    )


def _assign_support_label(metrics, min_bl_bulk, diag_eps):
    bl_bulk = float(metrics["BL_bulk"])
    d_bulk = float(metrics["D_bulk"])

    if abs(d_bulk) <= float(diag_eps):
        return "uncertain_boundary"

    if d_bulk > float(diag_eps):
        if bl_bulk >= float(min_bl_bulk):
            return "LH"
        return "uncertain_lowBL"

    return "soup"


def _compute_one_spike_metrics(v, X_bl_n, X_tr_n, side, idx, k_peak, k_bulk):
    cos_bl = X_bl_n @ v
    cos_tr = X_tr_n @ v

    if side == "BL":
        cos_bl = cos_bl.copy()
        if 0 <= int(idx) < cos_bl.size:
            cos_bl[int(idx)] = np.nan
    elif side == "TR":
        cos_tr = cos_tr.copy()
        if 0 <= int(idx) < cos_tr.size:
            cos_tr[int(idx)] = np.nan
    else:
        raise ValueError("side must be 'BL' or 'TR'")

    bl_valid = cos_bl[np.isfinite(cos_bl)]
    tr_valid = cos_tr[np.isfinite(cos_tr)]

    if bl_valid.size == 0 or tr_valid.size == 0:
        return dict(
            kmax=0,
            k_peak_used=[],
            k_bulk_used=[],
            BL_peak=np.nan,
            TR_peak=np.nan,
            BL_bulk=np.nan,
            TR_bulk=np.nan,
            D_peak=np.nan,
            D_bulk=np.nan,
            side=side,
            idx=int(idx),
            cos_to_BL_sorted=np.asarray([], dtype=np.float32),
            cos_to_TR_sorted=np.asarray([], dtype=np.float32),
            BL_curve=np.asarray([], dtype=np.float32),
            TR_curve=np.asarray([], dtype=np.float32),
            diff_curve=np.asarray([], dtype=np.float32),
        )

    bl_sorted = np.sort(bl_valid)[::-1]
    tr_sorted = np.sort(tr_valid)[::-1]
    bl_curve = _topk_mean_curve(bl_sorted)
    tr_curve = _topk_mean_curve(tr_sorted)

    metrics = _support_metrics_from_curves(bl_curve, tr_curve, k_peak, k_bulk)
    metrics.update(
        side=side,
        idx=int(idx),
        cos_to_BL_sorted=bl_sorted,
        cos_to_TR_sorted=tr_sorted,
        BL_curve=bl_curve,
        TR_curve=tr_curve,
        diff_curve=bl_curve[:metrics["kmax"]] - tr_curve[:metrics["kmax"]],
    )
    return metrics


def compute_bl_tr_support_decisions_from_groups(
    sn_bl,
    sn_tr,
    *,
    cos_mask_adc=30.0,
    k_peak=(5, 10, 20),
    k_bulk=(50, 100, 200),
    min_bl_bulk=0.70,
    diag_eps=0.05,
):
    """
    Full BL/TR support labeling as in lh-uw.py.
    sn_bl, sn_tr: arrays of shape [C, L, N_bl] and [C, L, N_tr]
    """
    sn_bl = np.asarray(sn_bl, dtype=np.float32)
    sn_tr = np.asarray(sn_tr, dtype=np.float32)

    if sn_bl.ndim != 3 or sn_tr.ndim != 3:
        raise ValueError("sn_bl and sn_tr must be [C, L, N]")
    if sn_bl.shape[0] != sn_tr.shape[0] or sn_bl.shape[1] != sn_tr.shape[1]:
        raise ValueError("sn_bl and sn_tr must match on [C, L]")
    if sn_bl.shape[2] < 2 or sn_tr.shape[2] < 2:
        raise ValueError("Need at least 2 BL and 2 TR spikes for support decisions.")

    med_bl = np.median(sn_bl, axis=2).astype(np.float32)
    med_tr = np.median(sn_tr, axis=2).astype(np.float32)

    mask = (np.abs(med_bl) >= float(cos_mask_adc)) | (np.abs(med_tr) >= float(cos_mask_adc))
    if int(mask.sum()) == 0:
        raise ValueError(f"Support mask is empty for cos_mask_adc={cos_mask_adc}.")

    X_bl = _flatten_masked_snips(sn_bl, mask)
    X_tr = _flatten_masked_snips(sn_tr, mask)
    X_bl_n = _row_normalize(X_bl)
    X_tr_n = _row_normalize(X_tr)

    bl_metrics = []
    bl_labels = []
    for idx in range(X_bl_n.shape[0]):
        m = _compute_one_spike_metrics(
            v=X_bl_n[idx],
            X_bl_n=X_bl_n,
            X_tr_n=X_tr_n,
            side="BL",
            idx=idx,
            k_peak=k_peak,
            k_bulk=k_bulk,
        )
        label = _assign_support_label(m, min_bl_bulk=min_bl_bulk, diag_eps=diag_eps)
        m["label"] = label
        bl_metrics.append(m)
        bl_labels.append(label)

    tr_metrics = []
    tr_labels = []
    for idx in range(X_tr_n.shape[0]):
        m = _compute_one_spike_metrics(
            v=X_tr_n[idx],
            X_bl_n=X_bl_n,
            X_tr_n=X_tr_n,
            side="TR",
            idx=idx,
            k_peak=k_peak,
            k_bulk=k_bulk,
        )
        label = _assign_support_label(m, min_bl_bulk=min_bl_bulk, diag_eps=diag_eps)
        m["label"] = label
        tr_metrics.append(m)
        tr_labels.append(label)

    def _count_labels(lbls):
        lbls = np.asarray(lbls, dtype=object)
        return dict(
            LH=int(np.sum(lbls == "LH")),
            soup=int(np.sum(lbls == "soup")),
            uncertain_boundary=int(np.sum(lbls == "uncertain_boundary")),
            uncertain_lowBL=int(np.sum(lbls == "uncertain_lowBL")),
            total=int(lbls.size),
        )

    return dict(
        params=dict(
            COS_MASK_ADC=float(cos_mask_adc),
            K_PEAK=list(k_peak),
            K_BULK=list(k_bulk),
            MIN_BL_BULK=float(min_bl_bulk),
            DIAG_EPS=float(diag_eps),
        ),
        med_bl=med_bl,
        med_tr=med_tr,
        mask=mask,
        bl_metrics=bl_metrics,
        tr_metrics=tr_metrics,
        bl_labels=np.asarray(bl_labels, dtype=object),
        tr_labels=np.asarray(tr_labels, dtype=object),
        bl_counts=_count_labels(bl_labels),
        tr_counts=_count_labels(tr_labels),
    )


def choose_adaptive_km_window(
    raw_data,
    left_times,
    *,
    probe_n=500,
    probe_win=(-40, 80),
    fallback_win=(-20, 40),
    time_amp_thr=30.0,
    ch_ptp_thr=30.0,
    pad_left=3,
    pad_right=3,
    min_pre=16,
    min_post=28,
    rng=None,
):
    """
    Determine optimal snippet window based on left spikes.
    Copied from lh-uw.py.
    """
    left_times = np.asarray(left_times, dtype=np.int64)
    fallback_win = (int(fallback_win[0]), int(fallback_win[1]))
    probe_win = (int(probe_win[0]), int(probe_win[1]))

    if left_times.size == 0:
        return fallback_win, dict(
            status="no_left_times",
            probe_n_req=0,
            probe_n_valid=0,
            n_ch_keep=0,
            left_rel=None,
            right_rel=None,
        )

    if rng is None:
        rng = np.random.RandomState(0)

    if left_times.size > int(probe_n):
        pick = left_times[rng.choice(left_times.size, int(probe_n), replace=False)]
    else:
        pick = left_times

    snips_probe, _ = extract_snippets_fast_ram(
        raw_data,
        pick,
        window=probe_win,
        selected_channels=np.arange(raw_data.shape[1], dtype=np.int32),
    )

    if snips_probe.shape[2] == 0:
        return fallback_win, dict(
            status="probe_empty",
            probe_n_req=int(pick.size),
            probe_n_valid=0,
            n_ch_keep=0,
            left_rel=None,
            right_rel=None,
        )

    ei_probe = snips_probe.mean(axis=2).astype(np.float32)
    p2p = ei_probe.max(axis=1) - ei_probe.min(axis=1)

    ch_keep = np.flatnonzero(p2p >= float(ch_ptp_thr))
    if ch_keep.size == 0:
        ch_keep = np.argsort(p2p)[-16:]
        ch_keep.sort()

    time_keep = np.any(np.abs(ei_probe[ch_keep, :]) >= float(time_amp_thr), axis=0)
    if not np.any(time_keep):
        return fallback_win, dict(
            status="no_time_support",
            probe_n_req=int(pick.size),
            probe_n_valid=int(snips_probe.shape[2]),
            n_ch_keep=int(ch_keep.size),
            left_rel=None,
            right_rel=None,
        )

    idx = np.flatnonzero(time_keep)
    left_rel = int(probe_win[0] + idx[0])
    right_rel = int(probe_win[0] + idx[-1])

    km_pre = int(max(probe_win[0], min(-int(min_pre), left_rel - int(pad_left))))
    km_post = int(min(probe_win[1], max(int(min_post), right_rel + int(pad_right))))

    if km_pre >= km_post:
        return fallback_win, dict(
            status="bad_window",
            probe_n_req=int(pick.size),
            probe_n_valid=int(snips_probe.shape[2]),
            n_ch_keep=int(ch_keep.size),
            left_rel=int(left_rel),
            right_rel=int(right_rel),
        )

    return (km_pre, km_post), dict(
        status="ok",
        probe_n_req=int(pick.size),
        probe_n_valid=int(snips_probe.shape[2]),
        n_ch_keep=int(ch_keep.size),
        left_rel=int(left_rel),
        right_rel=int(right_rel),
    )


# ============================================================================
# K-means precheck and verdict (copied from notebook)
# ============================================================================

def kmeans_pair_metrics(
    ei0,
    ei1,
    *,
    support_rel=0.10,
    support_abs=30.0,
    max_lag=1,
):
    """Compute cosine similarity on union of significant channels."""
    def support_from_ei(ei, support_rel, support_abs):
        p2p = np.ptp(ei, axis=1)
        thr = max(support_abs, support_rel * p2p.max())
        return p2p >= thr, p2p, thr

    S0, p2p0, thr0 = support_from_ei(ei0, support_rel, support_abs)
    S1, p2p1, thr1 = support_from_ei(ei1, support_rel, support_abs)

    U = S0 | S1
    if not np.any(U):
        U = (p2p0 > 0) | (p2p1 > 0)
    if not np.any(U):
        U = np.zeros(ei0.shape[0], dtype=bool)
        U[0] = True

    A = np.asarray(ei0[U, :], dtype=np.float32).ravel()
    nA = float(np.linalg.norm(A) + 1e-12)

    best = dict(lag=0, cos=-np.inf)
    for lag in range(-int(max_lag), int(max_lag) + 1):
        def shift_ei(ei, lag):
            out = np.zeros_like(ei)
            if lag == 0:
                out[:] = ei
            elif lag > 0:
                out[:, lag:] = ei[:, :-lag]
            else:
                s = -lag
                out[:, :ei.shape[1] - s] = ei[:, s:]
            return out
        Y = shift_ei(ei1, lag)
        B = np.asarray(Y[U, :], dtype=np.float32).ravel()
        nB = float(np.linalg.norm(B) + 1e-12)
        cos = float((A @ B) / (nA * nB))
        if cos > best["cos"]:
            best = dict(lag=int(lag), cos=float(cos))

    return dict(
        cos=float(best["cos"]),
        lag=int(best["lag"]),
        union_n=int(np.sum(U)),
        unique0=int(np.sum(S0 & ~S1)),
        unique1=int(np.sum(S1 & ~S0)),
        thr0=float(thr0),
        thr1=float(thr1),
    )


def kmeans_precheck_decision(
    vr,
    n0,
    n1,
    ei0,
    ei1,
    *,
    pc_var_thr=0.10,
    minor_frac_thr=0.10,
    cos_oneunit_thr=0.95,
    asym_unique_ch_min=3,
    support_rel=0.10,
    support_abs=30.0,
    cos_lag=1,
):
    vr = np.asarray(vr, dtype=np.float32).ravel()
    vr12 = vr[:2].copy()

    # 1) PCA variance rule: reject as TWO UNITS
    if np.any(vr12 > float(pc_var_thr)):
        return dict(
            decided=True,
            proceed=False,
            verdict="TWO-UNITS-like (PC variance)",
            reason="pc_var",
            detail=(
                "expl_var=[" + ", ".join(f"{100.0 * float(v):.2f}%" for v in vr12[:2]) + "] "
                f"> {100.0 * float(pc_var_thr):.1f}%"
            ),
            pair=None,
            minor_frac=None,
        )

    # 2) Tiny secondary cluster: accept as ONE UNIT
    n_big = int(max(n0, n1))
    n_small = int(min(n0, n1))
    minor_frac = float(n_small / max(n_big, 1))

    if minor_frac < float(minor_frac_thr):
        return dict(
            decided=True,
            proceed=True,
            verdict="ONE UNIT (tiny secondary cluster)",
            reason="cluster_size",
            detail=(
                f"n0={int(n0)} n1={int(n1)} "
                f"minor_frac={100.0 * float(minor_frac):.1f}% < {100.0 * float(minor_frac_thr):.1f}%"
            ),
            pair=None,
            minor_frac=minor_frac,
        )

    # 3) High EI cosine on union of significant channels: accept as ONE UNIT
    pair = kmeans_pair_metrics(
        ei0,
        ei1,
        support_rel=support_rel,
        support_abs=support_abs,
        max_lag=cos_lag,
    )

    if pair["cos"] > float(cos_oneunit_thr):
        return dict(
            decided=True,
            proceed=True,
            verdict="ONE UNIT (high EI cosine)",
            reason="ei_cos",
            detail=f"cos={float(pair['cos']):.2f} lag={pair['lag']} union_n={pair['union_n']} > {cos_oneunit_thr:.2f}",
            pair=pair,
            minor_frac=minor_frac,
        )

    # 4) Symmetric unique significant channels: reject as TWO UNITS
    if (pair["unique0"] >= int(asym_unique_ch_min)) and (pair["unique1"] >= int(asym_unique_ch_min)):
        return dict(
            decided=True,
            proceed=False,
            verdict="TWO-UNITS-like (asymmetric significant channels)",
            reason="asym_sig_channels",
            detail=f"unique0={pair['unique0']} unique1={pair['unique1']} >= {int(asym_unique_ch_min)}",
            pair=pair,
            minor_frac=minor_frac,
        )

    # Inconclusive -> caller should fall back to verdict_from_kmeans
    vr_str = "[" + ", ".join(f"{100.0 * float(v):.2f}%" for v in vr12[:2]) + "]"

    return dict(
        decided=False,
        proceed=True,
        verdict="INCONCLUSIVE",
        reason="inconclusive",
        detail=(
            f"expl_var={vr_str} "
            f"minor_frac={100.0 * float(minor_frac):.1f}% "
            f"cos={float(pair['cos']):.2f} lag={pair['lag']} "
            f"unique0={pair['unique0']} unique1={pair['unique1']}"
        ),
        pair=pair,
        minor_frac=minor_frac,
    )


def verdict_from_kmeans(
    ei0,
    ei1,
    *,
    max_lag=3,
    support_rel=0.10,
    support_abs=30.0,
    time_keep_rel=0.10,
    frac_in_thr=0.20,
    out_in_ratio_thr=2.0,
    resid_frac_min=0.08,
    shared_cos_thr=0.95,
    shared_alpha_thr=0.95,
):
    """Full verdict from containment metrics."""
    def shift_ei(ei, lag):
        out = np.zeros_like(ei)
        if lag == 0:
            out[:] = ei
        elif lag > 0:
            out[:, lag:] = ei[:, :-lag]
        else:
            s = -lag
            out[:, :ei.shape[1] - s] = ei[:, s:]
        return out

    def support_from_ei(ei, support_rel, support_abs):
        p2p = np.ptp(ei, axis=1)
        thr = max(support_abs, support_rel * p2p.max())
        return p2p >= thr, p2p, thr

    def best_lag_on_support(X, Y, S, max_lag, time_keep_rel):
        Xs = X[S, :]
        env = np.max(np.abs(Xs), axis=0) if Xs.size else np.max(np.abs(X), axis=0)
        tthr = time_keep_rel * (env.max() + 1e-12)
        Tmask = env >= tthr
        if not np.any(Tmask):
            Tmask = np.ones(X.shape[1], dtype=bool)

        best = dict(lag=0, cos=-np.inf, T=Tmask)
        A = X[S, :][:, Tmask].ravel()
        nA = np.linalg.norm(A) + 1e-12
        for lag in range(-int(max_lag), int(max_lag) + 1):
            Ys = shift_ei(Y, lag)
            B = Ys[S, :][:, Tmask].ravel()
            nB = np.linalg.norm(B) + 1e-12
            cos = float((A @ B) / (nA * nB))
            if cos > best["cos"]:
                best = dict(lag=int(lag), cos=cos, T=Tmask)
        return best

    def containment_metrics(X, Y, max_lag, support_rel, support_abs, time_keep_rel):
        S, _, thr = support_from_ei(X, support_rel, support_abs)
        best = best_lag_on_support(X, Y, S, max_lag, time_keep_rel)
        lag = best["lag"]
        Yal = shift_ei(Y, lag)
        Tmask = best["T"]

        A = X[S, :][:, Tmask].ravel()
        B = Yal[S, :][:, Tmask].ravel()
        alpha = float((A @ B) / ((A @ A) + 1e-12))

        R = Yal - alpha * X
        Yin = Yal[S, :]
        Rin = R[S, :]
        Rout = R[~S, :]

        Ein = float(np.linalg.norm(Rin))
        Eout = float(np.linalg.norm(Rout))

        frac_in = float(np.linalg.norm(Rin) / (np.linalg.norm(Yin) + 1e-12))
        frac_all = float(np.linalg.norm(R) / (np.linalg.norm(Yal) + 1e-12))
        out_in = float(Eout / (Ein + 1e-12))

        return dict(
            lag=lag,
            cos_on_support=float(best["cos"]),
            alpha=alpha,
            support_n=int(np.sum(S)),
            support_thr=float(thr),
            frac_in=frac_in,
            frac_all=frac_all,
            out_in=out_in,
        )

    m01 = containment_metrics(
        ei0, ei1,
        max_lag=max_lag,
        support_rel=support_rel,
        support_abs=support_abs,
        time_keep_rel=time_keep_rel,
    )
    m10 = containment_metrics(
        ei1, ei0,
        max_lag=max_lag,
        support_rel=support_rel,
        support_abs=support_abs,
        time_keep_rel=time_keep_rel,
    )

    shared01 = (m01["cos_on_support"] >= shared_cos_thr) and (m01["alpha"] >= shared_alpha_thr)
    shared10 = (m10["cos_on_support"] >= shared_cos_thr) and (m10["alpha"] >= shared_alpha_thr)
    shared_core = bool(shared01 or shared10)

    def is_contained(m):
        return (m["frac_in"] <= frac_in_thr) and (m["out_in"] >= out_in_ratio_thr)

    c01 = is_contained(m01)
    c10 = is_contained(m10)

    if shared_core and c01 and c10 and (m01["frac_all"] < resid_frac_min) and (m10["frac_all"] < resid_frac_min):
        verdict = "SAME UNIT split (amplitude/drift)"
        proceed = True
    elif shared_core and ((c01 and not c10) or (c10 and not c01)):
        verdict = "AB-SHARD-like (shared core)"
        proceed = True
    elif shared_core:
        verdict = "SHARED-CORE (overlap/AA/complex)"
        proceed = True
    elif (not c01) and (not c10):
        verdict = "TWO-UNITS-like (reject)"
        proceed = False
    else:
        verdict = "AMBIGUOUS (reject)"
        proceed = False

    return dict(
        verdict=verdict,
        proceed=bool(proceed),
        shared_core=shared_core,
        shared_dirs=dict(ei0_to_ei1=bool(shared01), ei1_to_ei0=bool(shared10)),
        m01=m01,
        m10=m10,
    )


# ============================================================================
# Main pipeline steps
# ============================================================================

# Default parameters (matching notebook, except fixed top 16 for PCA)
DEFAULT_PARAMS = dict(
    # Valley detection (fine-grained for low-amplitude units)
    window=(-20, 40),
    bin_width=10.0,            # was 10.0
    valley_bins=3,           # was 5
    min_valid_count=50,       # was 900
    ratio_base=3,
    ratio_step=500,
    ratio_floor=2,
    ratio_cap=10,
    right_k=2000,
    min_trough=-2500,

    # Adaptive window
    km_probe_n=500,
    km_probe_win=(-40, 80),
    km_probe_time_amp_thr=30.0,
    km_probe_ch_ptp_thr=30.0,
    km_win_pad_left=3,
    km_win_pad_right=3,
    km_win_min_pre=15,
    km_win_min_post=30,

    # Early reject thresholds
    max_valley_count=500,
    isi_10_30_max=10,

    # PCA / KMeans (on left spikes only)
    n_pcs=3,
    n_clusters=2,
    n_left_spikes_for_pca=5000,    # use all spikes (no subsampling)
    n_top_channels_for_pca=7,       # fixed top 16 (as you prefer)
    random_state=42,

    # K-means verdict thresholds
    pc_var_thr=0.10,
    minor_frac_thr=0.10,
    cos_oneunit_thr=0.95,
    asym_unique_ch_min=3,
    max_lag=3,
    support_rel=0.10,
    support_abs=30.0,
    time_keep_rel=0.10,
    frac_in_thr=0.20,
    out_in_ratio_thr=2.0,
    resid_frac_min=0.08,
    shared_cos_thr=0.95,
    shared_alpha_thr=0.95,

    # BL/TR support
    support_n_probe_per_side=2000,
    support_top_channels=12,
    support_cos_mask_adc=30.0,
    support_k_peak=(5, 10, 20),
    support_k_bulk=(50, 100, 200),
    support_min_bl_bulk=0.70,
    support_diag_eps=0.05,

    # Final acceptance
    min_final_spikes=200,

    # Mean EI for templates — keep small; chunked accumulation (see compute_mean_ei).
    # NEVER reuse n_left_spikes_for_pca here (5000 × C × L floats OOMs full-file runs).
    n_spikes_for_mean_ei=300,
    mean_ei_batch_size=64,
)


def run_valley_detection(
    raw_data: np.ndarray, ch: int, params: dict
) -> ValleyResult:
    """Step 1: Run find_valley_and_times on channel `ch` (all spikes, no subsampling)."""
    if raw_data is None:
        raise ValueError("raw_data is None")
    _, C = raw_data.shape
    if ch < 0 or ch >= C:
        raise ValueError(f"Channel {ch} out of range [0, {C})")

    raw = find_valley_and_times(
        raw_data,
        ch,
        window=tuple(params.get("window", (-40, 80))),
        bin_width=float(params.get("bin_width", 2.0)),
        valley_bins=int(params.get("valley_bins", 3)),
        min_valid_count=int(params.get("min_valid_count", 50)),
        ratio_base=int(params.get("ratio_base", 3)),
        ratio_step=int(params.get("ratio_step", 500)),
        ratio_floor=int(params.get("ratio_floor", 2)),
        ratio_cap=int(params.get("ratio_cap", 10)),
        right_k=int(params.get("right_k", 2000)),
        min_trough=params.get("min_trough", -2500),
    )

    return ValleyResult(
        accepted=bool(raw["accepted"]),
        valley_low=raw["valley_low"],
        valley_high=raw["valley_high"],
        left_times=raw["left_times"].astype(np.int64),
        left_vals=raw["left_vals"].astype(np.float32),
        valley_times=raw["valley_times"].astype(np.int64),
        valley_vals=raw["valley_vals"].astype(np.float32),
        all_times=raw["all_times"].astype(np.int64),
        all_vals=raw["all_vals"].astype(np.float32),
        amp_hist_counts=raw["amp_hist_counts"],
        amp_hist_edges=raw["amp_hist_edges"],
        left_count=int(raw["left_count"]),
        valley_count=int(raw["valley_count"]),
        rightk_times=raw.get("rightk_times_sorted", np.array([], dtype=np.int64)),
    )


def run_pca_kmeans_on_left_spikes(
    raw_data: np.ndarray,
    left_times: np.ndarray,
    detect_ch: int,
    params: dict,
) -> Tuple[PCAKMeansResult, dict]:
    """
    Step 2: PCA + KMeans on LEFT spikes only, using top channels (fixed number).
    Returns (PCAKMeansResult, km_info_dict) where km_info contains verdict, proceed, etc.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    if left_times.size == 0:
        empty_pca = PCAKMeansResult(
            pca_coords=np.empty((0, 3), dtype=np.float32),
            km_labels=np.array([], dtype=np.int64),
            cluster_mean_waveforms=[np.zeros(61), np.zeros(61)],
            explained_variance_ratio=np.array([0.33, 0.33, 0.33], dtype=np.float32),
            n_pcs_used=3,
        )
        km_info = dict(proceed=False, verdict="no_left_spikes", reason="no_left_spikes", detail="")
        return empty_pca, km_info

    # Adaptive window
    km_win, _ = choose_adaptive_km_window(
        raw_data, left_times,
        probe_n=params.get("km_probe_n", 500),
        probe_win=params.get("km_probe_win", (-30, 50)),
        fallback_win=params.get("window", (-20, 40)),
        time_amp_thr=params.get("km_probe_time_amp_thr", 30.0),
        ch_ptp_thr=params.get("km_probe_ch_ptp_thr", 30.0),
        pad_left=params.get("km_win_pad_left", 3),
        pad_right=params.get("km_win_pad_right", 3),
        min_pre=params.get("km_win_min_pre", 15),
        min_post=params.get("km_win_min_post", 30),
        rng=np.random.RandomState(params.get("random_state", 42)),
    )

    # Subsample left spikes for PCA if needed
    n_max = int(params.get("n_left_spikes_for_pca", 5000))
    if left_times.size > n_max:
        rng = np.random.RandomState(params.get("random_state", 42))
        idx = rng.choice(left_times.size, n_max, replace=False)
        left_use = left_times[idx]
    else:
        left_use = left_times

    # Determine top channels (fixed number, as you prefer)
    n_top = int(params.get("n_top_channels_for_pca", 16))
    # Use a small sample to compute RMS and pick top channels
    sample_times = left_use[:min(100, len(left_use))]
    sn_sample, _ = extract_snippets_fast_ram(
        raw_data, sample_times, window=km_win,
        selected_channels=np.arange(raw_data.shape[1], dtype=np.int32)
    )
    rms = np.sqrt(np.mean(sn_sample**2, axis=(1,2)))
    top_ch = np.argsort(rms)[-n_top:][::-1].astype(np.int32)

    # Extract snippets on top channels only
    sn_left, valid_times = extract_snippets_fast_ram(
        raw_data, left_use, window=km_win, selected_channels=top_ch
    )
    if sn_left.shape[2] == 0:
        empty_pca = PCAKMeansResult(
            pca_coords=np.empty((0, 3), dtype=np.float32),
            km_labels=np.array([], dtype=np.int64),
            cluster_mean_waveforms=[np.zeros(km_win[1]-km_win[0]+1),
                                    np.zeros(km_win[1]-km_win[0]+1)],
            explained_variance_ratio=np.array([0.33, 0.33, 0.33], dtype=np.float32),
            n_pcs_used=3,
        )
        km_info = dict(proceed=False, verdict="extraction_failed", reason="extraction_failed", detail="")
        return empty_pca, km_info

    C, L, N = sn_left.shape
    X = sn_left.transpose(2, 0, 1).reshape(N, C * L).astype(np.float32)

    n_pcs = int(params.get("n_pcs", 3))
    n_pcs_actual = min(n_pcs, N - 1, X.shape[1])
    if n_pcs_actual < 1:
        n_pcs_actual = 1

    pca = PCA(n_components=n_pcs_actual, svd_solver="full",
              random_state=params.get("random_state", 42))
    pca_coords = pca.fit_transform(X).astype(np.float32)

    km = KMeans(n_clusters=int(params.get("n_clusters", 2)),
                n_init=10, random_state=params.get("random_state", 42))
    labels = km.fit_predict(pca_coords).astype(np.int64)

    # Cluster mean waveforms on detection channel
    # Find index of detect_ch within top_ch
    if detect_ch in top_ch:
        local_idx = np.where(top_ch == detect_ch)[0][0]
        waveforms_det = sn_left[local_idx, :, :]  # [L, N]
    else:
        # fallback: use first channel
        waveforms_det = sn_left[0, :, :]

    cluster_means = []
    for k in range(2):
        mask = labels == k
        if mask.any():
            mean_wave = waveforms_det[:, mask].mean(axis=1)
        else:
            mean_wave = np.zeros(L, dtype=np.float32)
        cluster_means.append(mean_wave)

    # Now run K-means precheck + verdict
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    ei_c0 = sn_left[:, :, labels == 0].mean(axis=2).astype(np.float32)
    ei_c1 = sn_left[:, :, labels == 1].mean(axis=2).astype(np.float32)

    precheck = kmeans_precheck_decision(
        pca.explained_variance_ratio_,
        n0, n1,
        ei_c0, ei_c1,
        pc_var_thr=params.get("pc_var_thr", 0.10),
        minor_frac_thr=params.get("minor_frac_thr", 0.10),
        cos_oneunit_thr=params.get("cos_oneunit_thr", 0.95),
        asym_unique_ch_min=params.get("asym_unique_ch_min", 3),
        support_rel=params.get("support_rel", 0.10),
        support_abs=params.get("support_abs", 30.0),
        cos_lag=params.get("max_lag", 1),
    )

    if precheck["decided"]:
        verdict = precheck["verdict"]
        proceed = bool(precheck["proceed"])
        reason = precheck["reason"]
        detail = precheck["detail"]
        shared_core = None
    else:
        verdict_info = verdict_from_kmeans(
            ei_c0, ei_c1,
            max_lag=params.get("max_lag", 3),
            support_rel=params.get("support_rel", 0.10),
            support_abs=params.get("support_abs", 30.0),
            time_keep_rel=params.get("time_keep_rel", 0.10),
            frac_in_thr=params.get("frac_in_thr", 0.20),
            out_in_ratio_thr=params.get("out_in_ratio_thr", 2.0),
            resid_frac_min=params.get("resid_frac_min", 0.08),
            shared_cos_thr=params.get("shared_cos_thr", 0.95),
            shared_alpha_thr=params.get("shared_alpha_thr", 0.95),
        )
        verdict = verdict_info["verdict"]
        proceed = bool(verdict_info["proceed"])
        reason = "verdict_from_kmeans"
        detail = precheck["detail"]
        shared_core = verdict_info.get("shared_core", None)

    km_info = dict(
        proceed=proceed,
        verdict=verdict,
        reason=reason,
        detail=detail,
        n0=n0,
        n1=n1,
        vr=pca.explained_variance_ratio_,
        precheck=precheck,
        shared_core=shared_core,
    )

    pca_result = PCAKMeansResult(
        pca_coords=pca_coords,
        km_labels=labels,
        cluster_mean_waveforms=cluster_means,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_pcs_used=n_pcs_actual,
    )
    return pca_result, km_info


def _empty_bltr_counts() -> dict:
    return {"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0}


def _empty_bltr_result() -> BLTRResult:
    return BLTRResult(
        labels=np.array([], dtype=object),
        bl_bulk=np.array([], dtype=np.float32),
        tr_bulk=np.array([], dtype=np.float32),
        counts=_empty_bltr_counts(),
        times=np.array([], dtype=np.int64),
        ok=False,
    )


def build_bl_tr_probe_times(
    raw_data: np.ndarray,
    left_times: np.ndarray,
    tr_candidate_times: np.ndarray,
    *,
    main_ch: int,
    win: tuple,
    n_per_side: int = 2000,
) -> dict:
    """Pick BL/TR probe spikes by amplitude (notebook ``build_bl_tr_probe_times``).

    BL probes  = weakest (smallest abs amp) left spikes.
    TL probes  = strongest left spikes (returned for completeness).
    TR probes  = strongest right-of-valley candidates.
    """
    left_times = np.sort(np.unique(np.asarray(left_times, dtype=np.int64)))
    tr_candidate_times = np.sort(
        np.unique(np.asarray(tr_candidate_times, dtype=np.int64))
    )

    if left_times.size == 0:
        return dict(ok=False, reason="no_left_times")

    sn_left, left_valid = extract_snippets_fast_ram(
        raw_data,
        left_times,
        window=win,
        selected_channels=np.asarray([int(main_ch)], dtype=np.int32),
    )
    left_valid = np.asarray(left_valid, dtype=np.int64)
    if sn_left.shape[2] == 0:
        return dict(ok=False, reason="no_valid_left_snips")

    amp_left = np.max(np.abs(sn_left[0, :, :]), axis=0).astype(np.float32)
    order_left = np.argsort(amp_left)  # weakest first
    n_left_pick = int(min(int(n_per_side), order_left.size))
    bl_times = left_valid[order_left[:n_left_pick]].astype(np.int64)
    tl_times = left_valid[order_left[-n_left_pick:]].astype(np.int64)

    if tr_candidate_times.size == 0:
        return dict(
            ok=True,
            bl_times=bl_times,
            tl_times=tl_times,
            tr_times=np.asarray([], dtype=np.int64),
        )

    sn_right, right_valid = extract_snippets_fast_ram(
        raw_data,
        tr_candidate_times,
        window=win,
        selected_channels=np.asarray([int(main_ch)], dtype=np.int32),
    )
    right_valid = np.asarray(right_valid, dtype=np.int64)
    if sn_right.shape[2] == 0:
        tr_times = np.asarray([], dtype=np.int64)
    else:
        amp_right = np.max(np.abs(sn_right[0, :, :]), axis=0).astype(np.float32)
        order_right = np.argsort(amp_right)[::-1]  # strongest first
        n_right_pick = int(min(int(n_per_side), order_right.size))
        tr_times = right_valid[order_right[:n_right_pick]].astype(np.int64)

    return dict(ok=True, bl_times=bl_times, tl_times=tl_times, tr_times=tr_times)


def build_final_times_from_bltr(
    left_times: np.ndarray,
    rightk_times: np.ndarray,
    bltr: BLTRResult,
) -> np.ndarray:
    """Notebook final-spike rule: drop BL reject/uncertain probes; keep TR LH only.

    If BL/TR support did not succeed (``bltr.ok`` is False), fall back to the
    unfiltered union of left + rightk candidates (same as notebook when the
    support filter is off or fails).
    """
    left_times = np.sort(np.unique(np.asarray(left_times, dtype=np.int64)))
    rightk_times = np.sort(np.unique(np.asarray(rightk_times, dtype=np.int64)))

    if not bltr.ok:
        if left_times.size == 0 and rightk_times.size == 0:
            return np.array([], dtype=np.int64)
        parts = [t for t in (left_times, rightk_times) if t.size]
        return np.sort(np.unique(np.concatenate(parts))) if parts else np.array([], dtype=np.int64)

    bl_reject = np.asarray(bltr.bl_reject_times, dtype=np.int64)
    bl_uncertain = np.asarray(bltr.bl_uncertain_times, dtype=np.int64)
    if bl_reject.size or bl_uncertain.size:
        bl_drop = np.sort(np.unique(np.concatenate([bl_reject, bl_uncertain])))
    else:
        bl_drop = np.asarray([], dtype=np.int64)

    clean_left = np.setdiff1d(left_times, bl_drop, assume_unique=False)
    clean_right = np.sort(np.unique(np.asarray(bltr.tr_keep_times, dtype=np.int64)))

    parts = [t for t in (clean_left, clean_right) if t.size]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.sort(np.unique(np.concatenate(parts)))


def run_bltr_support(
    raw_data: np.ndarray,
    valley: ValleyResult,
    detect_ch: int,
    params: dict,
) -> BLTRResult:
    """
    Step 3: BL/TR support labeling matching the notebook.

    1. Adaptive snippet window from left spikes.
    2. Amplitude-ranked probe selection (weakest BL, strongest TR).
    3. Cosine-support labeling on probe snippets only.
    4. Return keep/reject/uncertain time arrays for final_times filtering.
    """
    left_times = np.asarray(valley.left_times, dtype=np.int64)
    rightk_times = np.asarray(
        getattr(valley, "rightk_times", np.array([], dtype=np.int64)),
        dtype=np.int64,
    )

    if left_times.size < 2 or rightk_times.size < 2:
        return _empty_bltr_result()

    # Adaptive window
    km_win, _ = choose_adaptive_km_window(
        raw_data, left_times,
        probe_n=params.get("km_probe_n", 500),
        probe_win=params.get("km_probe_win", (-40, 80)),
        fallback_win=params.get("window", (-20, 40)),
        time_amp_thr=params.get("km_probe_time_amp_thr", 30.0),
        ch_ptp_thr=params.get("km_probe_ch_ptp_thr", 30.0),
        pad_left=params.get("km_win_pad_left", 3),
        pad_right=params.get("km_win_pad_right", 3),
        min_pre=params.get("km_win_min_pre", 15),
        min_post=params.get("km_win_min_post", 30),
        rng=np.random.RandomState(params.get("random_state", 42)),
    )

    n_per_side = int(params.get("support_n_probe_per_side", 2000))
    probe = build_bl_tr_probe_times(
        raw_data,
        left_times,
        rightk_times,
        main_ch=int(detect_ch),
        win=km_win,
        n_per_side=n_per_side,
    )
    if not probe.get("ok", False):
        return _empty_bltr_result()

    bl_times = np.asarray(probe["bl_times"], dtype=np.int64)
    tr_times = np.asarray(probe["tr_times"], dtype=np.int64)
    if bl_times.size < 2 or tr_times.size < 2:
        return _empty_bltr_result()

    # Top channels for support labeling (RMS on a small left sample)
    n_top = int(params.get("support_top_channels", 12))
    sample_times = left_times[: min(100, len(left_times))]
    sn_sample, _ = extract_snippets_fast_ram(
        raw_data, sample_times, window=km_win,
        selected_channels=np.arange(raw_data.shape[1], dtype=np.int32),
    )
    rms = np.sqrt(np.mean(sn_sample ** 2, axis=(1, 2)))
    top_ch = np.argsort(rms)[-n_top:][::-1].astype(np.int32)
    if top_ch.size == 0:
        top_ch = np.arange(raw_data.shape[1], dtype=np.int32)[:n_top]

    sn_bl, bl_valid = extract_snippets_fast_ram(
        raw_data, bl_times, window=km_win, selected_channels=top_ch
    )
    sn_tr, tr_valid = extract_snippets_fast_ram(
        raw_data, tr_times, window=km_win, selected_channels=top_ch
    )
    bl_valid = np.asarray(bl_valid, dtype=np.int64)
    tr_valid = np.asarray(tr_valid, dtype=np.int64)

    if sn_bl.shape[2] < 2 or sn_tr.shape[2] < 2:
        return _empty_bltr_result()

    try:
        decision = compute_bl_tr_support_decisions_from_groups(
            sn_bl,
            sn_tr,
            cos_mask_adc=params.get("support_cos_mask_adc", 30.0),
            k_peak=params.get("support_k_peak", (5, 10, 20)),
            k_bulk=params.get("support_k_bulk", (50, 100, 200)),
            min_bl_bulk=params.get("support_min_bl_bulk", 0.70),
            diag_eps=params.get("support_diag_eps", 0.05),
        )
    except ValueError:
        return _empty_bltr_result()

    bl_labels = np.asarray(decision["bl_labels"], dtype=object)
    tr_labels = np.asarray(decision["tr_labels"], dtype=object)

    # Align labels to valid extraction times (same length as decision arrays)
    if bl_labels.size != bl_valid.size:
        bl_valid = bl_valid[: bl_labels.size]
    if tr_labels.size != tr_valid.size:
        tr_valid = tr_valid[: tr_labels.size]

    bl_keep_mask = bl_labels == "LH"
    bl_uncertain_mask = np.isin(
        bl_labels, np.asarray(["uncertain_boundary", "uncertain_lowBL"], dtype=object)
    )
    tr_keep_mask = tr_labels == "LH"
    tr_uncertain_mask = np.isin(
        tr_labels, np.asarray(["uncertain_boundary", "uncertain_lowBL"], dtype=object)
    )
    bl_reject_mask = ~(bl_keep_mask | bl_uncertain_mask)
    tr_reject_mask = ~(tr_keep_mask | tr_uncertain_mask)

    bl_bulk = np.array(
        [float(m.get("BL_bulk", np.nan)) for m in decision["bl_metrics"]],
        dtype=np.float32,
    )
    tr_bulk_bl = np.array(
        [float(m.get("TR_bulk", np.nan)) for m in decision["bl_metrics"]],
        dtype=np.float32,
    )
    bl_bulk_tr = np.array(
        [float(m.get("BL_bulk", np.nan)) for m in decision["tr_metrics"]],
        dtype=np.float32,
    )
    tr_bulk = np.array(
        [float(m.get("TR_bulk", np.nan)) for m in decision["tr_metrics"]],
        dtype=np.float32,
    )

    labels = np.concatenate([bl_labels, tr_labels])
    times = np.concatenate([bl_valid, tr_valid]).astype(np.int64)
    bulk_bl = np.concatenate([bl_bulk, bl_bulk_tr]).astype(np.float32)
    bulk_tr = np.concatenate([tr_bulk_bl, tr_bulk]).astype(np.float32)

    blc = decision["bl_counts"]
    trc = decision["tr_counts"]
    counts = {
        "LH": int(blc["LH"] + trc["LH"]),
        "soup": int(blc["soup"] + trc["soup"]),
        "uncertain_boundary": int(blc["uncertain_boundary"] + trc["uncertain_boundary"]),
        "uncertain_lowBL": int(blc["uncertain_lowBL"] + trc["uncertain_lowBL"]),
    }

    return BLTRResult(
        labels=labels,
        bl_bulk=bulk_bl,
        tr_bulk=bulk_tr,
        counts=counts,
        times=times,
        ok=True,
        bl_keep_times=np.asarray(bl_valid[bl_keep_mask], dtype=np.int64),
        bl_uncertain_times=np.asarray(bl_valid[bl_uncertain_mask], dtype=np.int64),
        bl_reject_times=np.asarray(bl_valid[bl_reject_mask], dtype=np.int64),
        tr_keep_times=np.asarray(tr_valid[tr_keep_mask], dtype=np.int64),
        tr_uncertain_times=np.asarray(tr_valid[tr_uncertain_mask], dtype=np.int64),
        tr_reject_times=np.asarray(tr_valid[tr_reject_mask], dtype=np.int64),
    )


def compute_mean_ei(
    raw_data: np.ndarray,
    spike_times: np.ndarray,
    window: tuple,
    max_spikes: int = 300,
    random_state: int = 42,
    batch_size: int = 64,
) -> Optional[np.ndarray]:
    """Mean EI [n_channels, n_samples] over a subset of spike times.

    **Memory:** never materializes a full ``[C, L, N]`` block for all spikes.
    With a 512-ch probe, ``N=5000`` would be ~0.6–0.8 GB per channel and was
    OOM-killing full-file runs (~100 GB raw already resident). We accumulate
    in small batches instead (peak ~ tens of MB).
    """
    spike_times = np.asarray(spike_times, dtype=np.int64)
    if spike_times.size == 0:
        return None
    max_spikes = max(1, int(max_spikes))
    batch_size = max(8, int(batch_size))
    if spike_times.size > max_spikes:
        rng = np.random.RandomState(random_state)
        spike_times = np.sort(rng.choice(spike_times, max_spikes, replace=False))

    chans = np.arange(raw_data.shape[1], dtype=np.int32)
    acc: Optional[np.ndarray] = None
    n_used = 0
    for i0 in range(0, spike_times.size, batch_size):
        batch = spike_times[i0 : i0 + batch_size]
        snips, _ = extract_snippets_fast_ram(
            raw_data, batch, window=window, selected_channels=chans
        )
        if snips.size == 0 or snips.shape[2] == 0:
            continue
        batch_sum = snips.sum(axis=2)  # [C, L]
        if acc is None:
            acc = batch_sum.astype(np.float64, copy=False)
        else:
            acc += batch_sum
        n_used += int(snips.shape[2])
        del snips, batch_sum

    if acc is None or n_used == 0:
        return None
    return (acc / float(n_used)).astype(np.float32)


def slim_qc_result(result: "QCResult", *, max_pca_points: int = 1500) -> "QCResult":
    """Drop heavy diagnostic arrays so batch QC does not accumulate OOM.

    Keeps everything needed for UI (hist, PCA scatter subsample, counts,
    final_times, final_ei) and Phy export. Called at end of every channel.
    """
    v = result.valley
    # Amp hist plot uses counts/edges; all local-minima arrays are huge on long recs.
    v.all_times = np.array([], dtype=np.int64)
    v.all_vals = np.array([], dtype=np.float32)
    v.valley_times = np.array([], dtype=np.int64)
    v.valley_vals = np.array([], dtype=np.float32)
    # rightk only needed during BL/TR (already done)
    v.rightk_times = np.array([], dtype=np.int64)

    # left_times/vals still used for export amplitudes + FR fallback — keep, but
    # cap extremely large left sets (rare) to protect RAM.
    if v.left_times is not None and v.left_times.size > 50_000:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(v.left_times.size, 50_000, replace=False))
        v.left_times = np.asarray(v.left_times, dtype=np.int64)[keep]
        if v.left_vals is not None and v.left_vals.size >= keep.size:
            v.left_vals = np.asarray(v.left_vals, dtype=np.float32)[keep]
        v.left_count = int(v.left_times.size)

    # PCA scatter: keep a display subsample
    pca = result.pca_km
    n_pts = int(pca.pca_coords.shape[0]) if pca.pca_coords is not None else 0
    if n_pts > max_pca_points:
        rng = np.random.RandomState(1)
        idx = np.sort(rng.choice(n_pts, max_pca_points, replace=False))
        pca.pca_coords = np.asarray(pca.pca_coords, dtype=np.float32)[idx]
        if pca.km_labels is not None and pca.km_labels.size == n_pts:
            pca.km_labels = np.asarray(pca.km_labels)[idx]

    # km_info often embeds large EIs / nested dicts from precheck
    if result.km_info:
        keep_keys = {
            "proceed", "verdict", "reason", "detail", "n0", "n1",
            "n_spikes", "cluster_sizes", "cos_sim",
        }
        slim_info = {k: result.km_info[k] for k in keep_keys if k in result.km_info}
        # store scalar explained-variance summary only
        vr = result.km_info.get("vr")
        if vr is not None:
            try:
                slim_info["vr"] = np.asarray(vr, dtype=np.float32).ravel()[:3]
            except Exception:
                pass
        result.km_info = slim_info

    # BL/TR probe arrays — counts + keep times already applied to final_times
    bl = result.bltr
    bl.labels = np.array([], dtype=object)
    bl.bl_bulk = np.array([], dtype=np.float32)
    bl.tr_bulk = np.array([], dtype=np.float32)
    bl.times = np.array([], dtype=np.int64)
    bl.bl_uncertain_times = np.array([], dtype=np.int64)
    bl.bl_reject_times = np.array([], dtype=np.int64)
    bl.tr_uncertain_times = np.array([], dtype=np.int64)
    bl.tr_reject_times = np.array([], dtype=np.int64)
    # keep bl_keep_times / tr_keep_times only if small; final_times is source of truth
    if bl.bl_keep_times is not None and bl.bl_keep_times.size > 5000:
        bl.bl_keep_times = bl.bl_keep_times[:5000]
    if bl.tr_keep_times is not None and bl.tr_keep_times.size > 5000:
        bl.tr_keep_times = bl.tr_keep_times[:5000]

    # Empty multi-channel snippet placeholder was [C, L, 0] — shrink metadata
    sn = result.snippets
    if sn is not None and getattr(sn, "snippets", None) is not None:
        L = int(sn.snippet_len) if sn.snippet_len else 61
        result.snippets = SnippetResult(
            snippets=np.empty((1, L, 0), dtype=np.float32),
            times=np.array([], dtype=np.int64),
            n_channels=1,
            snippet_len=L,
        )

    return result


def compute_biophysics(
    raw_data: np.ndarray,
    detect_ch: int,
    spike_times: np.ndarray,
    fs: float,
    duration_s: float,
    window: tuple[int, int] = (-20, 40),
) -> dict:
    """
    Calculate physiological ground-truth metrics for a set of spike times.
    """
    if spike_times.size < 2:
        return {
            "firing_rate_hz": spike_times.size / max(0.001, duration_s),
            "isi_violations_pct": 0.0,
            "burst_fraction_pct": 0.0,
            "trough_to_peak_ms": 0.0,
            "peak_to_trough_ratio": 0.0,
            "median_amp_adc": 0.0,
        }

    # 1. Temporal Metrics
    sorted_times = np.sort(spike_times)
    diffs_ms = np.diff(sorted_times) / fs * 1000.0
    isi_violations = np.sum(diffs_ms < 1.5)
    burst_fraction = np.sum(diffs_ms < 5.0)
    
    # Percentages based on number of intervals
    n_intervals = diffs_ms.size
    isi_violations_pct = float(isi_violations / n_intervals * 100.0)
    burst_fraction_pct = float(burst_fraction / n_intervals * 100.0)
    firing_rate_hz = float(spike_times.size / duration_s)

    # 2. Waveform Metrics (on a subset to be fast)
    # Extract mean waveform
    MAX_FOR_MEAN = 500
    if spike_times.size > MAX_FOR_MEAN:
        rng = np.random.RandomState(42)
        subset_times = rng.choice(spike_times, MAX_FOR_MEAN, replace=False)
    else:
        subset_times = spike_times
    
    # extract_snippets_fast_ram expects [ch1, ch2, ...]
    snips, valid = extract_snippets_fast_ram(
        raw_data, subset_times, window=window, selected_channels=np.array([detect_ch], dtype=np.int32)
    )
    
    if snips.size == 0:
        return {
            "firing_rate_hz": firing_rate_hz,
            "isi_violations_pct": isi_violations_pct,
            "burst_fraction_pct": burst_fraction_pct,
            "trough_to_peak_ms": 0.0,
            "peak_to_trough_ratio": 0.0,
            "median_amp_adc": 0.0,
        }
    
    # snips shape is [1, L, N]
    mean_wf = np.mean(snips[0], axis=1) # [L]
    
    # Metrics based on mean waveform
    # Assume trough is at window[0] offset (usually 20 samples in)
    # But let's be more robust: find the global minimum in the snippet
    trough_idx = np.argmin(mean_wf)
    trough_val = mean_wf[trough_idx]
    
    # Peak is the maximum AFTER the trough
    if trough_idx < mean_wf.size - 1:
        peak_idx = trough_idx + np.argmax(mean_wf[trough_idx:])
        peak_val = mean_wf[peak_idx]
        trough_to_peak_ms = float((peak_idx - trough_idx) / fs * 1000.0)
    else:
        peak_val = 0.0
        trough_to_peak_ms = 0.0
        
    peak_to_trough_ratio = float(abs(peak_val) / max(1.0, abs(trough_val)))
    
    # Median absolute amplitude of individual spikes at their own trough
    # (not just the mean waveform trough)
    # Actually, easier to just use the min per snippet
    median_amp_adc = float(np.median(np.abs(np.min(snips[0], axis=0))))

    return {
        "firing_rate_hz": firing_rate_hz,
        "isi_violations_pct": isi_violations_pct,
        "burst_fraction_pct": burst_fraction_pct,
        "trough_to_peak_ms": trough_to_peak_ms,
        "peak_to_trough_ratio": peak_to_trough_ratio,
        "median_amp_adc": median_amp_adc,
    }


def run_qc_pipeline(
    raw_data: np.ndarray,
    ch: int,
    n_sorter_spikes: int = -1,
    params: Optional[dict] = None,
    fs: float = 20_000.0,
) -> QCResult:
    """
    Top-level QC pipeline with all fixes.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)

    # Step 1: Valley detection (all spikes)
    valley = run_valley_detection(raw_data, ch, params)
    # Free full local-minima arrays ASAP (hist is enough for plots). On long
    # recordings these alone are tens–hundreds of MB per channel if retained.
    valley.all_times = np.array([], dtype=np.int64)
    valley.all_vals = np.array([], dtype=np.float32)

    # Early reject: valley not accepted
    if not valley.accepted:
        return slim_qc_result(
            _empty_qc_result(ch, n_sorter_spikes, valley, params, reason="valley_not_accepted")
        )

    # Early reject: valley count > max
    max_valley = int(params.get("max_valley_count", 500))
    if valley.valley_count > max_valley:
        return slim_qc_result(
            _empty_qc_result(
                ch, n_sorter_spikes, valley, params, reason=f"valley_count>{max_valley}"
            )
        )

    # Early reject: ISI 10-30 pairs
    isi_max = int(params.get("isi_10_30_max", 10))
    left_times = valley.left_times
    if left_times.size >= 2:
        diffs = np.diff(np.sort(left_times))
        isi_pairs = int(np.sum((diffs >= 10) & (diffs <= 30)))
        if isi_pairs > isi_max:
            return slim_qc_result(
                _empty_qc_result(
                    ch, n_sorter_spikes, valley, params, reason=f"isi_10_30>{isi_max}"
                )
            )

    # Step 2: PCA + KMeans on left spikes
    pca_km, km_info = run_pca_kmeans_on_left_spikes(raw_data, left_times, ch, params)

    if not km_info["proceed"]:
        return slim_qc_result(
            _empty_qc_result(
                ch,
                n_sorter_spikes,
                valley,
                params,
                reason=f"kmeans_reject: {km_info['verdict']}",
                km_info=km_info,
                pca_km=pca_km,
            )
        )

    # Step 3: BL/TR support (amplitude-ranked probes + keep/drop times)
    bltr = run_bltr_support(raw_data, valley, ch, params)

    # Step 3b: notebook final-spike filtering
    final_times = build_final_times_from_bltr(
        valley.left_times,
        getattr(valley, "rightk_times", np.array([], dtype=np.int64)),
        bltr,
    )
    min_spikes = int(params.get("min_final_spikes", 200))
    if final_times.size < min_spikes:
        return slim_qc_result(
            _empty_qc_result(
                ch,
                n_sorter_spikes,
                valley,
                params,
                reason=f"too_few_final_spikes ({final_times.size}<{min_spikes})",
                bltr=bltr,
                km_info=km_info,
                pca_km=pca_km,
            )
        )

    win = params.get("window", (-20, 40))
    snippet_len = int(win[1] - win[0] + 1)
    # Tiny placeholder — never allocate [C, L, 0] with C=512 retained × N channels
    dummy_snippets = SnippetResult(
        snippets=np.empty((1, snippet_len, 0), dtype=np.float32),
        times=np.array([], dtype=np.int64),
        n_channels=1,
        snippet_len=snippet_len,
    )

    # Mean EI for Phy templates — small N + chunked (see compute_mean_ei)
    final_ei = compute_mean_ei(
        raw_data,
        final_times,
        window=win,
        max_spikes=int(params.get("n_spikes_for_mean_ei", 300)),
        random_state=int(params.get("random_state", 42)),
        batch_size=int(params.get("mean_ei_batch_size", 64)),
    )

    # ── Step 4: Biophysics Extraction ─────────────────────────────────────
    duration_s = raw_data.shape[0] / fs
    biophysics = compute_biophysics(
        raw_data=raw_data,
        detect_ch=ch,
        spike_times=final_times,
        fs=fs,
        duration_s=duration_s,
        window=win,
    )

    result = QCResult(
        channel=ch,
        n_sorter_spikes=n_sorter_spikes,
        valley=valley,
        snippets=dummy_snippets,
        pca_km=pca_km,
        bltr=bltr,
        biophysics=biophysics,
        km_info=km_info,
        fs=fs,
        final_times=final_times,
        final_ei=final_ei,
    )
    return slim_qc_result(result)


def _empty_qc_result(
    ch,
    n_sorter_spikes,
    valley,
    params,
    reason,
    bltr=None,
    km_info=None,
    pca_km=None,
):
    """Return a QCResult indicating rejection with empty data."""
    snippet_len = params.get("window", (-20, 40))[1] - params.get("window", (-20, 40))[0] + 1
    dummy_snippets = SnippetResult(
        snippets=np.empty((1, snippet_len, 0), dtype=np.float32),
        times=np.array([], dtype=np.int64),
        n_channels=1,
        snippet_len=snippet_len,
    )
    if pca_km is None:
        pca_km = PCAKMeansResult(
            pca_coords=np.empty((0, 3), dtype=np.float32),
            km_labels=np.array([], dtype=np.int64),
            cluster_mean_waveforms=[np.zeros(snippet_len), np.zeros(snippet_len)],
            explained_variance_ratio=np.array([0.33, 0.33, 0.33], dtype=np.float32),
            n_pcs_used=3,
        )
    if bltr is None:
        bltr = _empty_bltr_result()

    return QCResult(
        channel=ch,
        n_sorter_spikes=n_sorter_spikes,
        valley=valley,
        snippets=dummy_snippets,
        pca_km=pca_km,
        bltr=bltr,
        reject_reason=reason,
        km_info=km_info or {},
        final_times=np.array([], dtype=np.int64),
        final_ei=None,
    )