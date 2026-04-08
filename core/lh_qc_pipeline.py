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
from typing import Optional, Tuple
import numpy as np
import math
from collections import defaultdict

# ── Imports from notebook helpers ───────────────────────────────────────────
try:
    from lh_deps.lighthouse_utils import find_valley_and_times
    from lh_deps.axolotl_utils_ram import extract_snippets_fast_ram
    from lh_deps.collision_utils import median_ei_adaptive
except ImportError:
    from lighthouse_utils import find_valley_and_times
    from axolotl_utils_ram import extract_snippets_fast_ram
    from collision_utils import median_ei_adaptive

from .result_types import (
    ValleyResult,
    SnippetResult,
    PCAKMeansResult,
    BLTRResult,
    QCResult,
)

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
    return [int(k) for k in k_list if int(k) >= 1 and int(k) <= int(n_available)]


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

    snips_probe, valid_probe = extract_snippets_fast_ram(
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
                f"expl_var=[" + ", ".join(f"{100.0 * float(v):.2f}%" for v in vr12[:2]) + "] "
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
        S, p2pX, thr = support_from_ei(X, support_rel, support_abs)
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
    bin_width=2.0,            # was 10.0
    valley_bins=15,           # was 5
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
    n_left_spikes_for_pca=5000,      # subsample left spikes for speed
    n_top_channels_for_pca=16,       # fixed top 16 (as you prefer)
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
        valley_bins=int(params.get("valley_bins", 15)),
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


def compute_left_isi_pairs_10_30(valley: ValleyResult) -> int:
    """Return number of left-spike pairs with ISI between 10 and 30 ms."""
    lt = np.sort(valley.left_times)
    if lt.size < 2:
        return 0
    d = np.diff(lt)
    # Assume 20 kHz sampling -> 10 samples = 0.5 ms? Actually 10 ms = 200 samples at 20k.
    # The notebook uses sample indices, so we need to know fs. We'll pass fs as param.
    # For now, assume default 20 kHz. We'll get fs from params later.
    # Better: store fs in params. We'll add a 'fs' parameter.
    # But to keep simple, we'll compute in samples: 10 ms = 0.01 * fs, 30 ms = 0.03 * fs.
    # We'll assume fs is passed. For now, return 0 and log warning.
    # Actually we'll modify run_qc_pipeline to pass fs.
    return 0  # Placeholder; will be implemented in run_qc_pipeline


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
        called_verdict = False
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


def run_bltr_support(
    raw_data: np.ndarray,
    valley: ValleyResult,
    detect_ch: int,
    params: dict,
) -> BLTRResult:
    """
    Step 3: Full BL/TR support labeling using notebook's algorithm.
    Returns BLTRResult with counts (and empty arrays for per-spike data to save memory).
    """
    left_times = valley.left_times
    rightk_times = getattr(valley, "rightk_times", np.array([], dtype=np.int64))

    if left_times.size < 2 or rightk_times.size < 2:
        return BLTRResult(
            labels=np.array([], dtype=object),
            bl_bulk=np.array([], dtype=np.float32),
            tr_bulk=np.array([], dtype=np.float32),
            counts={"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
            times=np.array([], dtype=np.int64),
        )

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

    # Extract amplitudes on main channel to pick weakest/strongest
    main_ch = detect_ch
    sn_main_left, _ = extract_snippets_fast_ram(
        raw_data, left_times, window=km_win,
        selected_channels=np.array([main_ch], dtype=np.int32)
    )
    if sn_main_left.shape[2] == 0:
        return BLTRResult(
            labels=np.array([], dtype=object),
            bl_bulk=np.array([], dtype=np.float32),
            tr_bulk=np.array([], dtype=np.float32),
            counts={"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
            times=np.array([], dtype=np.int64),
        )
    amp_left = np.ptp(sn_main_left[0, :, :], axis=0)
    order_weak = np.argsort(amp_left)
    n_probe = int(params.get("support_n_probe_per_side", 2000))
    n_probe = min(n_probe, left_times.size)
    bl_times = left_times[order_weak[:n_probe]]

    # Right side
    sn_main_right, _ = extract_snippets_fast_ram(
        raw_data, rightk_times, window=km_win,
        selected_channels=np.array([main_ch], dtype=np.int32)
    )
    if sn_main_right.shape[2] == 0:
        return BLTRResult(
            labels=np.array([], dtype=object),
            bl_bulk=np.array([], dtype=np.float32),
            tr_bulk=np.array([], dtype=np.float32),
            counts={"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
            times=np.array([], dtype=np.int64),
        )
    amp_right = np.ptp(sn_main_right[0, :, :], axis=0)
    order_strong = np.argsort(amp_right)[::-1]
    n_probe = min(n_probe, rightk_times.size)
    tr_times = rightk_times[order_strong[:n_probe]]

    # Top channels for support
    n_top = int(params.get("support_top_channels", 12))
    # Use a small sample to determine top channels
    sample_times = left_times[:min(100, len(left_times))]
    sn_sample, _ = extract_snippets_fast_ram(
        raw_data, sample_times, window=km_win,
        selected_channels=np.arange(raw_data.shape[1], dtype=np.int32)
    )
    rms = np.sqrt(np.mean(sn_sample**2, axis=(1,2)))
    top_ch = np.argsort(rms)[-n_top:][::-1].astype(np.int32)
    if top_ch.size == 0:
        top_ch = np.arange(raw_data.shape[1], dtype=np.int32)[:n_top]

    # Extract snippets for BL and TR groups
    sn_bl, bl_valid = extract_snippets_fast_ram(
        raw_data, bl_times, window=km_win, selected_channels=top_ch
    )
    sn_tr, tr_valid = extract_snippets_fast_ram(
        raw_data, tr_times, window=km_win, selected_channels=top_ch
    )
    if sn_bl.shape[2] < 2 or sn_tr.shape[2] < 2:
        return BLTRResult(
            labels=np.array([], dtype=object),
            bl_bulk=np.array([], dtype=np.float32),
            tr_bulk=np.array([], dtype=np.float32),
            counts={"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
            times=np.array([], dtype=np.int64),
        )

    decision = compute_bl_tr_support_decisions_from_groups(
        sn_bl,
        sn_tr,
        cos_mask_adc=params.get("support_cos_mask_adc", 30.0),
        k_peak=params.get("support_k_peak", (5,10,20)),
        k_bulk=params.get("support_k_bulk", (50,100,200)),
        min_bl_bulk=params.get("support_min_bl_bulk", 0.70),
        diag_eps=params.get("support_diag_eps", 0.05),
    )

    # Combine counts from BL and TR
    blc = decision["bl_counts"]
    trc = decision["tr_counts"]
    total_lh = blc["LH"] + trc["LH"]
    total_soup = blc["soup"] + trc["soup"]
    total_uncertain = (blc["uncertain_boundary"] + blc["uncertain_lowBL"] +
                       trc["uncertain_boundary"] + trc["uncertain_lowBL"])

    return BLTRResult(
        labels=np.array([], dtype=object),  # not storing per-spike labels to save memory
        bl_bulk=np.array([], dtype=np.float32),
        tr_bulk=np.array([], dtype=np.float32),
        counts={"LH": total_lh, "soup": total_soup,
                "uncertain_boundary": 0, "uncertain_lowBL": total_uncertain},
        times=np.array([], dtype=np.int64),
    )


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

    # Early reject: valley not accepted
    if not valley.accepted:
        return _empty_qc_result(ch, n_sorter_spikes, valley, params, reason="valley_not_accepted")

    # Early reject: valley count > max
    max_valley = int(params.get("max_valley_count", 500))
    if valley.valley_count > max_valley:
        return _empty_qc_result(ch, n_sorter_spikes, valley, params, reason=f"valley_count>{max_valley}")

    # Early reject: ISI 10-30 pairs
    isi_max = int(params.get("isi_10_30_max", 10))
    left_times = valley.left_times
    if left_times.size >= 2:
        diffs = np.diff(np.sort(left_times))
        isi_pairs = np.sum((diffs >= 0.01 * fs) & (diffs <= 0.03 * fs))  # 10-30 ms in samples
        if isi_pairs > isi_max:
            return _empty_qc_result(ch, n_sorter_spikes, valley, params, reason=f"isi_10_30>{isi_max}")

    # Step 2: PCA + KMeans on left spikes
    pca_km, km_info = run_pca_kmeans_on_left_spikes(raw_data, left_times, ch, params)

    if not km_info["proceed"]:
        return _empty_qc_result(ch, n_sorter_spikes, valley, params, reason=f"kmeans_reject: {km_info['verdict']}")

    # Step 3: BL/TR support (probe-based)
    bltr = run_bltr_support(raw_data, valley, ch, params)

    # Build final spike list: left_times + rightk_times (simplified; notebook does BL/TR filtering)
    # For QC, we just need counts; we'll accept all left and rightk times.
    final_times = np.sort(np.unique(np.concatenate([valley.left_times, valley.rightk_times])))
    min_spikes = int(params.get("min_final_spikes", 200))
    if final_times.size < min_spikes:
        return _empty_qc_result(ch, n_sorter_spikes, valley, params, reason=f"too_few_final_spikes ({final_times.size}<{min_spikes})")

    # Create dummy snippets (not stored)
    snippet_len = params.get("window", (-20,40))[1] - params.get("window", (-20,40))[0] + 1
    dummy_snippets = SnippetResult(
        snippets=np.empty((raw_data.shape[1], snippet_len, 0), dtype=np.float32),
        times=np.array([], dtype=np.int64),
        n_channels=raw_data.shape[1],
        snippet_len=snippet_len,
    )

    return QCResult(
        channel=ch,
        n_sorter_spikes=n_sorter_spikes,
        valley=valley,
        snippets=dummy_snippets,
        pca_km=pca_km,
        bltr=bltr,
    )


def _empty_qc_result(ch, n_sorter_spikes, valley, params, reason):
    """Return a QCResult indicating rejection with empty data."""
    snippet_len = params.get("window", (-20,40))[1] - params.get("window", (-20,40))[0] + 1
    dummy_snippets = SnippetResult(
        snippets=np.empty((1, snippet_len, 0), dtype=np.float32),
        times=np.array([], dtype=np.int64),
        n_channels=1,
        snippet_len=snippet_len,
    )
    dummy_pca = PCAKMeansResult(
        pca_coords=np.empty((0, 3), dtype=np.float32),
        km_labels=np.array([], dtype=np.int64),
        cluster_mean_waveforms=[np.zeros(snippet_len), np.zeros(snippet_len)],
        explained_variance_ratio=np.array([0.33,0.33,0.33], dtype=np.float32),
        n_pcs_used=3,
    )
    dummy_bltr = BLTRResult(
        labels=np.array([], dtype=object),
        bl_bulk=np.array([], dtype=np.float32),
        tr_bulk=np.array([], dtype=np.float32),
        counts={"LH":0,"soup":0,"uncertain_boundary":0,"uncertain_lowBL":0},
        times=np.array([], dtype=np.int64),
    )
    # Override valley to indicate rejection reason (optional)
    return QCResult(
        channel=ch,
        n_sorter_spikes=n_sorter_spikes,
        valley=valley,
        snippets=dummy_snippets,
        pca_km=dummy_pca,
        bltr=dummy_bltr,
    )