# %% [markdown]
# # LH loop with BL/TR uncertainty labeling

# %%
import axolotl_litke_loader as litke
print(litke._BIN2PY_AVAILABLE)

# %%
import axolotl_litke_loader as litke

folder_path = "/Volumes/Data/2018-08-07-9/data004/"

raw_data = litke.load_litke_as_writable_array(
    folder_path=folder_path,
    n_channels=512,
    dtype="int16",
    start_min=0.0,
    duration_min=0.1,
    fs=20000,
    chunk_samples=100000,
)

print(raw_data.shape)
print(raw_data.dtype)
print(raw_data[:5, :5])

# %% [markdown]
# # Imports

# %%
# standard
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import h5py
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import math
import gc
from collections import defaultdict
import pickle
import tempfile
from scipy.ndimage import gaussian_filter1d



# custom

import axolotl_litke_loader as litke

import axolotl_utils_ram
importlib.reload(axolotl_utils_ram)
from axolotl_utils_ram import extract_snippets_fast_ram


import lighthouse_utils
importlib.reload(lighthouse_utils)
from lighthouse_utils import find_valley_and_times

import plot_ei_waveforms
importlib.reload(plot_ei_waveforms)
import plot_ei_waveforms as pew

import collision_utils
importlib.reload(collision_utils)
from collision_utils import select_template_channels, median_ei_adaptive

import compute_sta_from_spikes
importlib.reload(compute_sta_from_spikes)

import joint_utils
importlib.reload(joint_utils)


# %% [markdown]
# # Load data

# %% [markdown]
# ### Option 1. Load Litke files.

# %%
folder_path = "/Volumes/Data/2018-08-07-9/data007/"   # Litke folder, not .dat file

n_channels = 512
dtype = "int16"
fs = 20_000

start_minutes   = 0
minutes_to_load = 3

raw_orig = litke.load_litke_as_writable_array(
    folder_path=folder_path,
    n_channels=n_channels,
    dtype=dtype,
    start_min=start_minutes,
    duration_min=minutes_to_load,
    fs=fs,
    chunk_samples=100000,
)

print(f"Loaded raw_orig {raw_orig.shape} = {raw_orig.shape[0]/fs/60:.1f} minutes")

# %% [markdown]
# ### Option 2. Load .dat file (KS)

# %%
# --- Path and recording setup ---
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/20250306C/data019_ks/data019.dat" #,length 20 min exactly
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/20250307C/data023_ks/data023.dat" # length 17480000/20000/60 - ~14.5min
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/202510236/data018.dat" # length - limit to 30 min
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/200804040/data000/data000.dat" # length - limit to 30 min (total orig is 45) MOUSE
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/200804050/data005/data005.dat" # length - 30 min MOUSE
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/202310301/data002/data002.dat" # length - 30 min RAT
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/202512300/data001_ks/data001.dat" # length - 2hrs; start 20 min, read 30 min; MOUSE from Greg, 519
# dat_path = "/Volumes/Lab/Users/alexth/axolotl/201808079/data007.dat" # length - limit to 30 min
dat_path = "/Volumes/Lab/Users/alexth/axolotl/201711290/data004.dat" # length - limit to 30 min

n_channels = 512 # IMPORTANT!!!!
dtype = np.int16

# Sampling rate (your usual)
fs = 20_000

# What chunk to load
start_minutes   = 0
minutes_to_load = 30

# --- Get total number of samples in file ---
file_size_bytes = os.path.getsize(dat_path)
total_samples_in_file = file_size_bytes // (np.dtype(dtype).itemsize * n_channels)

# --- Convert desired time window -> samples ---
start_sample = int(start_minutes * 60 * fs)
n_samples    = int(minutes_to_load * 60 * fs)

# Clamp to file bounds
if start_sample >= total_samples_in_file:
    raise ValueError(f"start_sample={start_sample} beyond file length {total_samples_in_file}")

n_samples = min(n_samples, total_samples_in_file - start_sample)

# --- Read only that chunk ---
offset_bytes = start_sample * n_channels * np.dtype(dtype).itemsize
count_vals   = n_samples * n_channels

with open(dat_path, "rb") as f:
    f.seek(offset_bytes, os.SEEK_SET)
    raw_orig = np.fromfile(f, dtype=dtype, count=count_vals)

raw_orig = raw_orig.reshape((n_samples, n_channels))  # [T, C]
print(f"Loaded raw_orig {raw_orig.shape} = {n_samples/fs/60:.1f} minutes")



# %% [markdown]
# ### Load EI positions, deal with baselines

# %%

# LOAD ONLY EI POSITIONS
h5_in_path = '/Volumes/Lab/Users/alexth/axolotl/201703151_kilosort_data001_spike_times.h5'  # from MATLAB export, to get EI positions - 60 micron
# h5_in_path = '/Volumes/Lab/Users/alexth/axolotl/201811126_kilosort_data000_spike_times.h5'  # from MATLAB export, to get EI positions - 30 micron

with h5py.File(h5_in_path, 'r') as f:
    # Load electrode positions
    ei_positions = f['/ei_positions'][:].T  # shape becomes [512 x 2]



# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/20250306C/data019_ks/data019_baseline_derivative_20k.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/20250307C/data023_ks/data023_baseline_derivative_20k.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/202510236/data018_baseline_derivative_20k.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/200804040/data000/data000_baseline_derivative_20k_30min.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/200804050/data005/data005_baseline_derivative_20k.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/202310301/data002/data002_baseline_derivative_20k.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/202512300/data001_ks/data001_baseline_derivative_20k_20_50_min.json"
# baseline_path = "/Volumes/Lab/Users/alexth/axolotl/201808079/data007_baseline_derivative_20k.json"
baseline_path = "/Volumes/Lab/Users/alexth/axolotl/201711290/data004_baseline_derivative_20k_30min.json"

segment_len = 20_000
if os.path.exists(baseline_path):
    print(f"Loading baselines")
    with open(baseline_path, 'r') as f:
        data = json.load(f)
    baselines = np.array(data['baselines'], dtype=np.float32)
else:
    print(f"Computing baselines")
    baselines = axolotl_utils_ram.compute_baselines_int16_deriv_robust(raw_orig, segment_len=segment_len, diff_thresh=10, trim_fraction=0.15) # shape (512, 360)

    with open(baseline_path, 'w') as f:
        json.dump({
            'baselines': baselines.tolist(),
        }, f)

print("subtracting baselines")

axolotl_utils_ram.subtract_segment_baselines_int16(raw_data=raw_orig,
                                     baselines_f32=baselines,
                                     segment_len=segment_len) 


# %% [markdown]
# # Copy original to modifiable copy - we will use that for subtraction
# DOUBLES THE AMOUNT OF DATA IN MEMORY!!! NOT NECESSARY AND CAN BE BYPASSED, see cell below

# %%

raw_orig.setflags(write=False)           # any accidental write will error
raw_mod = raw_orig.copy()                # your working residual
raw_mod.setflags(write=True)

print("raw_orig dtype/shape:", raw_orig.dtype, raw_orig.shape, "GB:", raw_orig.nbytes/1e9)
print("raw_mod  dtype/shape:", raw_mod.dtype,  raw_mod.shape,  "GB:", raw_mod.nbytes/1e9)
print("raw_orig is frozen, raw_mod is mutable.")
print("shares_memory:", np.shares_memory(raw_orig, raw_mod))


# %% [markdown]
# ### BYPASS - no memory doubling

# %%
raw_mod = raw_orig
del raw_orig   # optional

# %% [markdown]
# # Restore data FROM ORIGINAL - DO NOT RUN UNLESS WANT TO RESTORE

# %%
# To restore:
raw_mod[:] = raw_orig

# %% [markdown]
# # Main

# %% [markdown]
# ### Helpers

# %%
# ============================================================
# LH + BL/TR support helpers
# ============================================================

# Required names already imported:
# np, plt, time, math, defaultdict, PCA, KMeans
# extract_snippets_fast_ram
# find_valley_and_times
# pew  (import plot_ei_waveforms as pew)

def _stage_start(label, channel_tag=None):
    if VERBOSE_TIMING:
        prefix = f"[CH {channel_tag}] " if channel_tag is not None else ""
        print(f"{prefix}START {label}")
    return time.perf_counter()

def _stage_end(label, t0, timings=None, channel_tag=None):
    dt = time.perf_counter() - t0
    if timings is not None:
        timings[label] = dt
    if VERBOSE_TIMING:
        prefix = f"[CH {channel_tag}] " if channel_tag is not None else ""
        print(f"{prefix}DONE  {label} | {dt:.2f} s")
    return dt

def _timed_call(label, func, *args, channel_tag=None, timings=None, **kwargs):
    t0 = _stage_start(label, channel_tag=channel_tag)
    try:
        return func(*args, **kwargs)
    finally:
        _stage_end(label, t0, timings=timings, channel_tag=channel_tag)
        
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


def _resolve_k_list(k_list, n_available):
    return [int(k) for k in k_list if int(k) >= 1 and int(k) <= int(n_available)]


def _topk_mean_curve(sorted_desc):
    sorted_desc = np.asarray(sorted_desc, dtype=np.float32)
    return np.cumsum(sorted_desc) / np.arange(1, sorted_desc.size + 1, dtype=np.float32)


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
    d_bulk = float(metrics["D_bulk"])  # BL_bulk - TR_bulk

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


def plot_bl_tr_support_scatter(decision_out, title=None):
    bl_metrics = decision_out["bl_metrics"]
    tr_metrics = decision_out["tr_metrics"]

    bl_x = np.array([m["BL_bulk"] for m in bl_metrics], dtype=float)
    bl_y = np.array([m["TR_bulk"] for m in bl_metrics], dtype=float)
    tr_x = np.array([m["BL_bulk"] for m in tr_metrics], dtype=float)
    tr_y = np.array([m["TR_bulk"] for m in tr_metrics], dtype=float)

    p = decision_out["params"]
    min_bl_bulk = float(p["MIN_BL_BULK"])
    diag_eps = float(p["DIAG_EPS"])

    plt.figure(figsize=(7, 7))
    plt.scatter(bl_x, bl_y, s=20, alpha=0.70, label="BL spikes")
    plt.scatter(tr_x, tr_y, s=20, alpha=0.70, label="TR spikes")

    xx0 = min(np.min(bl_x), np.min(tr_x), np.min(bl_y), np.min(tr_y)) - 0.02
    xx1 = max(np.max(bl_x), np.max(tr_x), np.max(bl_y), np.max(tr_y)) + 0.02
    xx = np.linspace(xx0, xx1, 200)
    plt.plot(xx, xx, "--", linewidth=1, alpha=0.8, label="BL_bulk = TR_bulk")
    plt.axvline(min_bl_bulk, linestyle=":", linewidth=1, alpha=0.8)
    plt.axhline(min_bl_bulk, linestyle=":", linewidth=1, alpha=0.8)
    plt.fill_between(xx, xx - diag_eps, xx + diag_eps, alpha=0.08)

    plt.xlabel("BL bulk support")
    plt.ylabel("TR bulk support")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.title("BL vs TR bulk support" if title is None else title)
    plt.tight_layout()
    plt.show()


def _norm_cdf(x, mu, sig):
    z = (x - mu) / (sig * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def choose_adaptive_km_window(
    raw_data,
    left_times,
    *,
    probe_n=500,
    probe_win=(-50, 100),
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
    Use a small long-window probe set from LEFT spikes to choose an adaptive
    downstream snippet window. Step1 stays fixed; everything after this can use
    the returned km_win.
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

def compute_left_isi_pairs_10_30(step1):
    lt = np.asarray(step1.get("left_times", []), dtype=np.int64)
    lt = np.sort(lt)
    if lt.size < 2:
        return 0
    d = np.diff(lt)
    return int(np.sum((d >= 10) & (d <= 30)))


def plot_amp_hist(step1, ch, ycap=5000):
    edges = np.asarray(step1["amp_hist_edges"], dtype=np.float64)
    counts = np.asarray(step1["amp_hist_counts"], dtype=np.float64)

    plt.figure(figsize=(12, 3))
    plt.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
    plt.axvspan(step1["valley_low"], step1["valley_high"], color="r", alpha=0.25)
    plt.ylim(0, ycap)
    plt.title(
        f"CH {ch}: amplitude histogram | left={step1['left_count']} valley={step1['valley_count']}"
    )
    plt.xlabel("minima amplitude (ADC)")
    plt.ylabel("count")
    plt.grid(alpha=0.20)
    plt.tight_layout()
    plt.show()


def plot_final_ei(ei, ref_channel, title):
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    pew.plot_ei_waveforms(
        ei,
        positions=ei_positions,
        ref_channel=int(ref_channel),
        scale=70.0,
        ax=ax,
        colors="black",
        alpha=1.0,
        linewidth=0.8,
        box_height=1.0,
        box_width=50.0,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def shift_ei(ei, lag):
    ei = np.asarray(ei)
    out = np.zeros_like(ei)
    if lag == 0:
        out[:] = ei
    elif lag > 0:
        out[:, lag:] = ei[:, :-lag]
    else:
        s = -lag
        out[:, :ei.shape[1] - s] = ei[:, s:]
    return out


def support_from_ei(ei, support_rel=0.10, support_abs=30.0):
    p2p = np.ptp(ei, axis=1)
    thr = max(float(support_abs), float(support_rel) * float(p2p.max()))
    S = p2p >= thr
    return S, p2p, thr


def best_lag_on_support(X, Y, S, max_lag=3, time_keep_rel=0.10):
    Xs = X[S, :]
    env = np.max(np.abs(Xs), axis=0) if Xs.size else np.max(np.abs(X), axis=0)
    tthr = float(time_keep_rel) * float(env.max() + 1e-12)
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


def containment_metrics(X, Y, *, max_lag=3, support_rel=0.10, support_abs=30.0, time_keep_rel=0.10):
    S, p2pX, thr = support_from_ei(X, support_rel=support_rel, support_abs=support_abs)
    best = best_lag_on_support(X, Y, S, max_lag=max_lag, time_keep_rel=time_keep_rel)
    lag = best["lag"]
    Yal = shift_ei(Y, lag)
    Tmask = best["T"]

    A = X[S, :][:, Tmask].ravel()
    B = Yal[S, :][:, Tmask].ravel()
    alpha = float((A @ B) / ((A @ A) + 1e-12))

    R = Yal - alpha * X
    Yin  = Yal[S, :]
    Rin  = R[S, :]
    Rout = R[~S, :]

    Ein = float(np.linalg.norm(Rin))
    Eout = float(np.linalg.norm(Rout))

    frac_in  = float(np.linalg.norm(Rin) / (np.linalg.norm(Yin) + 1e-12))
    frac_all = float(np.linalg.norm(R) / (np.linalg.norm(Yal) + 1e-12))
    out_in   = float(Eout / (Ein + 1e-12))

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


def kmeans_pair_metrics(
    ei0,
    ei1,
    *,
    support_rel=0.10,
    support_abs=30.0,
    max_lag=1,
):
    S0, p2p0, thr0 = support_from_ei(ei0, support_rel=support_rel, support_abs=support_abs)
    S1, p2p1, thr1 = support_from_ei(ei1, support_rel=support_rel, support_abs=support_abs)

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




def stable_unique_int64(x):
    x = np.asarray(x, dtype=np.int64).ravel()
    if x.size == 0:
        return x
    x_sorted = np.sort(x)
    keep = np.ones(x_sorted.size, dtype=bool)
    keep[1:] = x_sorted[1:] != x_sorted[:-1]
    return x_sorted[keep]


def build_template_bank(step4, *, raw_mod, win, sr, n_bins=5, spikes_per=5000, ei_p2p_thr=50.0):
    main_ch = int(step4.get("main_ch", 0))
    final_ei = np.asarray(step4["final_ei"], dtype=np.float32)
    ei_p2p = np.ptp(final_ei, axis=1)

    chan_sel = np.flatnonzero(ei_p2p >= float(ei_p2p_thr)).astype(np.int32)
    if chan_sel.size == 0:
        raise RuntimeError(f"No channels pass EI p2p >= {ei_p2p_thr} (max={ei_p2p.max():.1f})")

    chan_sel = chan_sel[np.argsort(ei_p2p[chan_sel])[::-1]]
    if not np.any(chan_sel == main_ch):
        chan_sel = np.concatenate([np.asarray([main_ch], dtype=np.int32), chan_sel])
        chan_sel = np.unique(chan_sel)

    times_all = np.sort(np.asarray(step4.get("final_times", step4.get("final_times_used", [])), dtype=np.int64))
    if times_all.size == 0:
        raise RuntimeError("No spike times in step4.")

    sn_main, valid_times = extract_snippets_fast_ram(
        raw_mod,
        times_all,
        window=win,
        selected_channels=np.asarray([main_ch], dtype=np.int32),
    )
    valid_times = np.asarray(valid_times, dtype=np.int64)
    amps = np.ptp(sn_main[0, :, :], axis=0).astype(np.float32)

    n_bins = int(n_bins)
    spikes_per = int(spikes_per)
    if amps.size < n_bins * 10:
        raise RuntimeError(f"Too few valid snippets for template binning: Nvalid={amps.size}")

    order_desc = np.argsort(amps)[::-1]
    N = int(order_desc.size)
    bin_times = []
    bin_amp_ranges = []

    if N >= n_bins * spikes_per:
        starts = np.round(np.linspace(0, N - spikes_per, n_bins)).astype(int) if n_bins > 1 else np.array([0], dtype=int)
        for s in starts:
            idx = order_desc[s:s + spikes_per]
            tbin = valid_times[idx]
            abin = amps[idx]
            bin_times.append(tbin)
            bin_amp_ranges.append((float(np.min(abin)), float(np.max(abin))))
    else:
        groups = np.array_split(order_desc, n_bins)
        for g in groups:
            tbin = valid_times[g]
            abin = amps[g]
            bin_times.append(tbin)
            if abin.size:
                bin_amp_ranges.append((float(np.min(abin)), float(np.max(abin))))
            else:
                bin_amp_ranges.append((np.nan, np.nan))

    if any(t.size == 0 for t in bin_times):
        raise RuntimeError("Empty template bin; reduce n_bins or get more spikes.")

    templates = []
    for tbin in bin_times:
        sn, _ = extract_snippets_fast_ram(raw_mod, tbin, window=win, selected_channels=chan_sel)
        templates.append(np.median(sn, axis=2).astype(np.float32))
    templates = np.stack(templates, axis=0)

    return dict(
        main_ch=int(main_ch),
        win=tuple(win),
        sample_rate_hz=float(sr),
        channels=chan_sel,
        ei_p2p=ei_p2p[chan_sel],
        bin_amp_ranges=bin_amp_ranges,
        templates=templates,
    )


def build_median_ei_from_times(raw_mod, times, window, n_channels):
    times = np.asarray(times, dtype=np.int64)
    times = np.sort(np.unique(times))
    if times.size == 0:
        return None, 0

    sn, _ = extract_snippets_fast_ram(
        raw_mod,
        times,
        window=window,
        selected_channels=np.arange(int(n_channels), dtype=np.int32),
    )
    if sn.shape[2] == 0:
        return None, 0
    return np.median(sn, axis=2).astype(np.float32), int(sn.shape[2])



def subtract_lh_masked(
    raw_mod,
    spike_times,
    bank,
    *,
    time_keep_frac=0.08,
    weight_power=1.0,
    use_ch_weight_sub=True,
    min_sep_ms=1.0,
    cos_thr_sub=0.90,
    jmax_sub=3,
    batch_sub=256,
):
    templates = np.asarray(bank["templates"], dtype=np.float32)
    chs = np.asarray(bank["channels"], dtype=np.int32)
    win_ = tuple(bank["win"])
    sr = float(bank["sample_rate_hz"])
    B, Kk, L = templates.shape
    templates_i16 = np.rint(templates).astype(np.int16)

    ref = np.median(templates, axis=0)
    absref = np.abs(ref)

    time_env = absref.max(axis=0)
    t_thr = float(time_keep_frac) * time_env.max()
    keep_t = time_env >= t_thr

    if bool(use_ch_weight_sub):
        ch_env = np.ptp(ref, axis=1)
        ch_w = ch_env / (ch_env.max() + 1e-12)
    else:
        ch_w = np.ones(Kk, dtype=np.float32)

    W = (absref ** float(weight_power)) * ch_w[:, None]
    W[:, ~keep_t] = 0.0

    mask_idx = np.flatnonzero(W.ravel() > 0)
    w_mask = W.ravel()[mask_idx].astype(np.float32)

    tmpl_flat_full = templates.reshape(B, -1).astype(np.float32)
    tmplW = tmpl_flat_full[:, mask_idx] * w_mask[None, :]
    tmplW_norm = np.linalg.norm(tmplW, axis=1) + 1e-12

    main_ch = int(bank.get("main_ch", 0))
    if not np.any(chs == main_ch):
        raise RuntimeError(f"main_ch={main_ch} not found in bank channels.")

    main_ci = int(np.where(chs == main_ch)[0][0])
    center_idx = int(-win_[0])
    tmin_main = int(np.argmin(ref[main_ci, :]))
    dt_main = int(tmin_main - center_idx)

    win_ext = (win_[0] - int(jmax_sub), win_[1] + int(jmax_sub))
    spike_times = np.asarray(spike_times, dtype=np.int64)
    spike_times = np.sort(np.unique(spike_times))
    if spike_times.size == 0:
        return None

    used_t_all = []
    best_b_all = []
    best_j_all = []
    best_c_all = []

    tmp_sn, _ = extract_snippets_fast_ram(raw_mod, spike_times[:1], window=win_ext, selected_channels=chs)
    if tmp_sn.shape[2] == 0:
        return None
    Lext0 = tmp_sn.shape[1]
    anchor0 = (Lext0 - L) // 2

    for s0 in range(0, spike_times.size, int(batch_sub)):
        t_batch = spike_times[s0:s0 + int(batch_sub)]
        sn_ext, t_valid = extract_snippets_fast_ram(raw_mod, t_batch, window=win_ext, selected_channels=chs)
        t_valid = np.asarray(t_valid, dtype=np.int64)
        N = sn_ext.shape[2]
        if N == 0:
            continue

        Lext = sn_ext.shape[1]
        anchor = (Lext - L) // 2

        best_cos = np.full(N, -np.inf, dtype=np.float32)
        best_b = np.zeros(N, dtype=np.int16)
        best_j = np.zeros(N, dtype=np.int8)

        for j in range(-int(jmax_sub), int(jmax_sub) + 1):
            start = anchor + j
            seg = sn_ext[:, start:start + L, :].astype(np.float32)
            if seg.shape[1] != L:
                continue

            Xfull = seg.transpose(2, 0, 1).reshape(N, -1)
            Xw = Xfull[:, mask_idx] * w_mask[None, :]
            Xn = np.linalg.norm(Xw, axis=1) + 1e-12
            dots = Xw @ tmplW.T
            cos = dots / (Xn[:, None] * tmplW_norm[None, :])

            b = np.argmax(cos, axis=1).astype(np.int16)
            c = cos[np.arange(N), b].astype(np.float32)

            upd = c > best_cos
            best_cos[upd] = c[upd]
            best_b[upd] = b[upd]
            best_j[upd] = np.int8(j)

        used_t_all.append(t_valid)
        best_b_all.append(best_b)
        best_j_all.append(best_j)
        best_c_all.append(best_cos)

    if len(used_t_all) == 0:
        return None

    used_t_all = np.concatenate(used_t_all).astype(np.int64)
    best_b_all = np.concatenate(best_b_all).astype(np.int16)
    best_j_all = np.concatenate(best_j_all).astype(np.int8)
    best_c_all = np.concatenate(best_c_all).astype(np.float32)

    t_center_all = used_t_all + best_j_all.astype(np.int64)
    did_sub = best_c_all >= float(cos_thr_sub)

    min_sep_samp = int(round((float(min_sep_ms) / 1000.0) * sr))
    accept_idx = np.flatnonzero(did_sub)
    if accept_idx.size > 1 and min_sep_samp > 0:
        accept_sorted = accept_idx[np.argsort(t_center_all[accept_idx])]
        tacc = t_center_all[accept_sorted]
        i = 0
        while i < accept_sorted.size:
            j = i
            while (j + 1 < accept_sorted.size) and ((tacc[j + 1] - tacc[j]) < min_sep_samp):
                j += 1
            if j > i:
                cluster = accept_sorted[i:j + 1]
                keep = cluster[np.argmax(best_c_all[cluster])]
                drop = cluster[cluster != keep]
                did_sub[drop] = False
            i = j + 1

    keep_idx = np.flatnonzero(did_sub)
    keep_idx = keep_idx[np.argsort(t_center_all[keep_idx])]

    for idx in keep_idx:
        t = int(used_t_all[idx])
        b = int(best_b_all[idx])
        j = int(best_j_all[idx])
        t0 = t + win_ext[0] + (anchor0 + j)
        t1 = t0 + L
        tmpl = templates_i16[b]
        for ci, ch_local in enumerate(chs):
            raw_mod[t0:t1, int(ch_local)] -= tmpl[ci, :]

    spike_times_center = np.sort(t_center_all[did_sub])
    spike_times_main = np.sort((t_center_all + dt_main)[did_sub])

    return dict(
        main_ch=int(main_ch),
        dt_main=int(dt_main),
        spike_times_center=spike_times_center,
        spike_times_main=spike_times_main,
        n_sub=int(did_sub.sum()),
        best_cosine=best_c_all,
        best_jitter=best_j_all,
        did_subtract=did_sub,
    )


def build_bl_tr_probe_times(
    raw_mod,
    left_times,
    tr_candidate_times,
    *,
    main_ch,
    win,
    n_per_side=2000,
):
    left_times = np.sort(np.unique(np.asarray(left_times, dtype=np.int64)))
    tr_candidate_times = np.sort(np.unique(np.asarray(tr_candidate_times, dtype=np.int64)))

    if left_times.size == 0:
        return dict(ok=False, reason="no_left_times")

    sn_left, left_valid = extract_snippets_fast_ram(
        raw_mod,
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
        raw_mod,
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

    return dict(
        ok=True,
        bl_times=bl_times,
        tl_times=tl_times,
        tr_times=tr_times,
    )


def run_bl_tr_support_for_channel(
    raw_mod,
    ei_info,
    left_times,
    tr_candidate_times,
    *,
    ch,
    win,
    n_probe_per_side=2000,
    n_top_channels=12,
    cos_mask_adc=30.0,
    k_peak=(5, 10, 20),
    k_bulk=(50, 100, 200),
    min_bl_bulk=0.70,
    diag_eps=0.05,
):
    main_ch = int(ch)
    probe = build_bl_tr_probe_times(
        raw_mod,
        left_times,
        tr_candidate_times,
        main_ch=main_ch,
        win=win,
        n_per_side=n_probe_per_side,
    )
    if not probe.get("ok", False):
        return dict(ok=False, reason=probe.get("reason", "probe_build_failed"))

    bl_times = np.asarray(probe["bl_times"], dtype=np.int64)
    tl_times = np.asarray(probe["tl_times"], dtype=np.int64)
    tr_times = np.asarray(probe["tr_times"], dtype=np.int64)

    if bl_times.size < 2:
        return dict(ok=False, reason="too_few_bl_probe")
    if tr_times.size < 2:
        return dict(
            ok=False,
            reason="too_few_tr_probe",
            probe=dict(bl_times=bl_times, tl_times=tl_times, tr_times=tr_times),
        )

    final_ei = np.asarray(ei_info["final_ei"], dtype=np.float32)
    chan_p2p = np.ptp(final_ei, axis=1)
    topn = int(min(int(n_top_channels), chan_p2p.size))
    top_ch = np.argsort(chan_p2p)[-topn:][::-1].astype(np.int32)

    sn_bl, bl_valid = extract_snippets_fast_ram(raw_mod, bl_times, window=win, selected_channels=top_ch)
    sn_tr, tr_valid = extract_snippets_fast_ram(raw_mod, tr_times, window=win, selected_channels=top_ch)
    bl_valid = np.asarray(bl_valid, dtype=np.int64)
    tr_valid = np.asarray(tr_valid, dtype=np.int64)

    if sn_bl.shape[2] < 2 or sn_tr.shape[2] < 2:
        return dict(ok=False, reason="too_few_valid_snips_after_extract")

    decision_out = compute_bl_tr_support_decisions_from_groups(
        sn_bl,
        sn_tr,
        cos_mask_adc=cos_mask_adc,
        k_peak=k_peak,
        k_bulk=k_bulk,
        min_bl_bulk=min_bl_bulk,
        diag_eps=diag_eps,
    )

    bl_labels = np.asarray(decision_out["bl_labels"], dtype=object)
    tr_labels = np.asarray(decision_out["tr_labels"], dtype=object)

    bl_keep_mask = (bl_labels == "LH")
    bl_uncertain_mask = np.isin(bl_labels, np.asarray(["uncertain_boundary", "uncertain_lowBL"], dtype=object))
    tr_keep_mask = (tr_labels == "LH")
    tr_uncertain_mask = np.isin(tr_labels, np.asarray(["uncertain_boundary", "uncertain_lowBL"], dtype=object))

    bl_reject_mask = ~(bl_keep_mask | bl_uncertain_mask)
    tr_reject_mask = ~(tr_keep_mask | tr_uncertain_mask)

    return dict(
        ok=True,
        probe=dict(
            bl_times=bl_valid,
            tl_times=tl_times,
            tr_times=tr_valid,
            top_ch=top_ch,
            main_ch=int(main_ch),
        ),
        decision_out=decision_out,
        bl_keep_times=np.asarray(bl_valid[bl_keep_mask], dtype=np.int64),
        bl_uncertain_times=np.asarray(bl_valid[bl_uncertain_mask], dtype=np.int64),
        tr_keep_times=np.asarray(tr_valid[tr_keep_mask], dtype=np.int64),
        tr_uncertain_times=np.asarray(tr_valid[tr_uncertain_mask], dtype=np.int64),
        bl_reject_times=np.asarray(bl_valid[bl_reject_mask], dtype=np.int64),
        tr_reject_times=np.asarray(tr_valid[tr_reject_mask], dtype=np.int64),
    )


def plot_kmeans_pc_scatter(km_plot, title=None):
    if km_plot is None:
        return
    Xpc = np.asarray(km_plot["Xpc"], dtype=np.float32)
    lab = np.asarray(km_plot["labels"], dtype=np.int32)
    vr = np.asarray(km_plot["vr"], dtype=np.float32)

    plt.figure(figsize=(4.8, 4.8))
    for k in np.unique(lab):
        m = (lab == k)
        plt.scatter(Xpc[m, 0], Xpc[m, 1], s=12, alpha=0.75, label=f"cluster {int(k)}")
    plt.xlabel(f"PC1 ({vr[0] * 100.0:.1f}% var)" if vr.size >= 1 else "PC1")
    plt.ylabel(f"PC2 ({vr[1] * 100.0:.1f}% var)" if vr.size >= 2 else "PC2")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.title("k=2 PCA scatter" if title is None else title)
    plt.tight_layout()
    plt.show()



def _new_audit_record(ch):
    return dict(
        ch=int(ch),
        status="started",               # started / success / rejected / exception
        final_reason=None,              # e.g. valley_not_accepted, valley_count>max, SUCCESS
        error_type=None,
        error_msg=None,

        # step1 summary
        step1_accepted=None,
        left_count=None,
        valley_count=None,
        required_ratio=None,
        valley_low=None,
        valley_high=None,

        # adaptive window summary
        km_win_pre=None,
        km_win_post=None,
        km_probe_status=None,
        km_probe_left_rel=None,
        km_probe_right_rel=None,
        km_probe_n_req=None,
        km_probe_n_valid=None,
        km_probe_n_ch=None,

        # timings we explicitly care about
        t_step1_s=np.nan,
        t_km_window_s=np.nan,
        t_kmeans_extract_s=np.nan,
        t_channel_total_s=np.nan,

        # downstream summary
        isi_pairs_10_30=None,
        kmeans_verdict=None,
        kmeans_reason=None,
        main_ch=None,
        final_times_n=None,
        final_spikes_main_n=None,
        valley0_n=None,

        bl_probe_to_soup=None,
        bl_probe_to_unc=None,
        tr_probe_to_lh=None,

        used_support_filter=False,
        unit_index=None,
    )

def _set_reject(audit, reason):
    audit["status"] = "rejected"
    audit["final_reason"] = str(reason)

def _set_success(audit):
    audit["status"] = "success"
    audit["final_reason"] = "SUCCESS"

def _append_audit_record(audit):
    lh_channel_audit.append(audit)

def _build_valley_overflow_summary(audit_records, valley_limit, bin_step=200, hard_cap=1000):
    vals = [
        int(rec["valley_count"])
        for rec in audit_records
        if rec.get("final_reason") == "valley_count>max" and rec.get("valley_count") is not None
    ]
    out = []
    if len(vals) == 0:
        return out

    lo = int(valley_limit) + 1
    while lo <= int(hard_cap):
        hi = min(lo + int(bin_step) - 1, int(hard_cap))
        cnt = int(np.sum((np.asarray(vals) >= lo) & (np.asarray(vals) <= hi)))
        out.append((lo, hi, cnt))
        lo += int(bin_step)

    gt_cap = int(np.sum(np.asarray(vals) > int(hard_cap)))
    out.append((int(hard_cap) + 1, None, gt_cap))
    return out


# %% [markdown]
# ## PAGER: Quick histo view

# %%
# ============================================================
# Quick LH histogram browser: 50 channels per run
# Re-run the cell to advance to the next 50 channels.
# Reset with: LH_HIST_NEXT_START = 0
# ============================================================
assert "find_valley_and_times" in globals(), "Need find_valley_and_times imported."
assert ("raw_mod" in globals()) or ("raw_orig" in globals()), "Need raw_mod or raw_orig loaded."

# ----------------------------
# User knobs
# ----------------------------
LH_HIST_DURATION_MIN = 5.0
LH_HIST_NCOLS = 5
LH_HIST_NROWS = 10
LH_HIST_PER_PAGE = LH_HIST_NCOLS * LH_HIST_NROWS   # 50

LH_HIST_BIN_WIDTH = 10.0
LH_HIST_VALLEY_BINS = 5
LH_HIST_MIN_VALID_COUNT = 500
LH_HIST_RATIO_BASE = 3
LH_HIST_RATIO_STEP = 500
LH_HIST_RATIO_FLOOR = 2
LH_HIST_RATIO_CAP = 10

LH_HIST_WIN = (-20, 40)

# Fixed y-limit for all panels
LH_HIST_YMAX = 2000

# ----------------------------
# Choose source data
# ----------------------------
raw_for_hist = raw_mod if "raw_mod" in globals() else raw_orig
T_total, C = raw_for_hist.shape

# Sampling rate
fs_local = float(globals().get("fs", globals().get("sample_rate_hz", 20_000)))

# Limit scan duration
stop_sample = min(T_total, int(round(LH_HIST_DURATION_MIN * 60.0 * fs_local)))

# ----------------------------
# Paging state across reruns
# ----------------------------
# LH_HIST_NEXT_START = 0

if "LH_HIST_NEXT_START" not in globals():
    LH_HIST_NEXT_START = 0

ch_start = int(LH_HIST_NEXT_START)
ch_stop = min(C, ch_start + LH_HIST_PER_PAGE)
channels_this_page = list(range(ch_start, ch_stop))

if len(channels_this_page) == 0:
    print("Reached the end. Reset with: LH_HIST_NEXT_START = 0")
else:
    fig, axes = plt.subplots(
        LH_HIST_NROWS, LH_HIST_NCOLS,
        figsize=(25, 25),
        sharex=False, sharey=True
    )
    axes = np.asarray(axes).ravel()

    for ax_i, ch in enumerate(channels_this_page):
        ax = axes[ax_i]

        try:
            step1 = find_valley_and_times(
                raw_for_hist,
                ch,
                window=LH_HIST_WIN,
                start=0,
                stop=stop_sample,
                bin_width=LH_HIST_BIN_WIDTH,
                valley_bins=LH_HIST_VALLEY_BINS,
                min_valid_count=LH_HIST_MIN_VALID_COUNT,
                ratio_base=LH_HIST_RATIO_BASE,
                ratio_step=LH_HIST_RATIO_STEP,
                ratio_floor=LH_HIST_RATIO_FLOOR,
                ratio_cap=LH_HIST_RATIO_CAP,
                right_k=1,   # irrelevant here
            )

            counts = np.asarray(step1.get("amp_hist_counts", []), dtype=np.int32)
            edges  = np.asarray(step1.get("amp_hist_edges",  []), dtype=np.float32)

            if counts.size > 0 and edges.size == counts.size + 1:
                centers = 0.5 * (edges[:-1] + edges[1:])
                width = np.diff(edges)

                ax.bar(
                    centers,
                    counts,
                    width=width,
                    color="0.65",
                    edgecolor="none",
                    align="center"
                )

                valley_low = step1.get("valley_low", None)
                valley_high = step1.get("valley_high", None)
                accepted = bool(step1.get("accepted", False))
                left_count = int(step1.get("left_count", 0))
                valley_count = int(step1.get("valley_count", 0))

                if (valley_low is not None) and (valley_high is not None):
                    ax.axvspan(
                        float(valley_low),
                        float(valley_high),
                        color=("limegreen" if accepted else "orange"),
                        alpha=0.28
                    )
                    ax.axvline(float(valley_low), color="k", linewidth=0.6, alpha=0.7)
                    ax.axvline(float(valley_high), color="k", linewidth=0.6, alpha=0.7)

                title = f"ch {ch}"
                if valley_low is not None:
                    title += " | LH?" if accepted else " | no"
                ax.set_title(title, fontsize=9)

                txt = f"L={left_count}\nV={valley_count}"
                ax.text(
                    0.98, 0.95, txt,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=7,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        alpha=0.7,
                        linewidth=0.0
                    )
                )

                ax.grid(True, alpha=0.15, linewidth=0.4)
                ax.tick_params(labelsize=7)

            else:
                ax.text(
                    0.5, 0.5, f"ch {ch}\nno peaks",
                    ha="center", va="center", fontsize=9
                )
                ax.set_title(f"ch {ch}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

        except Exception as e:
            ax.text(
                0.5, 0.5, f"ch {ch}\nERR\n{type(e).__name__}",
                ha="center", va="center", fontsize=8
            )
            ax.set_title(f"ch {ch}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Fixed y-limit on every panel
        ax.set_ylim(0, LH_HIST_YMAX)

    # Hide unused panels on the last page
    for j in range(len(channels_this_page), LH_HIST_PER_PAGE):
        axes[j].axis("off")

    fig.suptitle(
        f"LH histogram browser | channels {ch_start}-{ch_stop - 1} | "
        f"first {LH_HIST_DURATION_MIN:.1f} min",
        fontsize=14,
        y=0.995
    )
    plt.tight_layout()
    plt.show()

    LH_HIST_NEXT_START = ch_stop

    print(f"Plotted channels {ch_start}–{ch_stop - 1} from first {LH_HIST_DURATION_MIN:.1f} min.")
    if LH_HIST_NEXT_START < C:
        print(f"Next run will start at channel {LH_HIST_NEXT_START}.")
    else:
        print("Reached the end. Reset with: LH_HIST_NEXT_START = 0")

# %% [markdown]
# # MAIN LOOP
# ### Run channels

# %%
# ============================================================
# LH loop with per-channel audit trail + end-of-run summaries
# ============================================================

assert "raw_mod" in globals(), "Need raw_mod (baseline-subtracted int16) loaded."
assert "ei_positions" in globals(), "Need ei_positions loaded."
assert "extract_snippets_fast_ram" in globals(), "Need extract_snippets_fast_ram imported."
assert "find_valley_and_times" in globals(), "Need find_valley_and_times imported."

T, C = raw_mod.shape
sr = float(globals().get("sample_rate_hz", 20_000))

# ----------------------------
# USER KNOBS
# ----------------------------
CHANNELS_TO_RUN = list(range(C))   # debug: [103], or all channels: list(range(C))
PLOT_DIAGNOSTICS = True            # per-success plots during the run
PLOT_END_SUMMARY = True            # timing/window summary plots after the run
RESET_RESULTS = False
VERBOSE_TIMING = True

win = (-20, 40)

# Step 1 valley
BIN_WIDTH = 10.0
VALLEY_BINS = 5
MIN_VALID_COUNT = 900
RATIO_BASE = 3
RATIO_STEP = 500
RATIO_FLOOR = 2
RATIO_CAP = 10
MAX_VALLEY_COUNT = 500

# Early reject checks
ISI_10_30_MAX = 10

# Adaptive EI / k-means window
KM_PROBE_N = 500
KM_PROBE_WIN = (-40, 80)
KM_PROBE_TIME_AMP_THR = 30.0
KM_PROBE_CH_P2P_THR = 30.0
KM_WIN_PAD_LEFT = 3
KM_WIN_PAD_RIGHT = 3
KM_WIN_MIN_PRE = 15
KM_WIN_MIN_POST = 30

# Left-only KMeans / verdict
DO_KMEANS = True
SUBSAMPLE_MAX = 5_000
RMS_THRESH = 5.0
N_PC = 2
RNG = np.random.RandomState(123)

KM_PC_VAR_THR = 0.10
KM_MINOR_FRAC_THR = 0.10
KM_EI_COS_THR = 0.95
KM_ASYM_UNIQUE_CH_MIN = 3
KM_COS_LAG = 1

MAX_LAG = 3
SUPPORT_REL = 0.10
SUPPORT_ABS = 30.0
TIME_KEEP_REL = 0.10
FRAC_IN_THR = 0.20
OUT_IN_RATIO_THR = 2.0
RESID_FRAC_MIN = 0.08
SHARED_COS_THR = 0.95
SHARED_ALPHA_THR = 0.95

# Subtraction bank / subtraction
N_BINS = 5
SPIKES_PER = 5000
EI_P2P_THR = 50.0

TIME_KEEP_FRAC = 0.08
WEIGHT_POWER = 1.0

JMAX_SUB = 3
BATCH_SUB = 256
USE_CH_WEIGHT_SUB = True
MIN_SEP_MS = 1.0
COS_THR_SUB = 0.90

MIN_FINAL_SPIKES_TO_CALL_SUCCESS = 200

# BL/TR support filter
USE_BL_TR_SUPPORT_FILTER = True
SUPPORT_N_PROBE_PER_SIDE = 2000
SUPPORT_TOP_CHANNELS = 12
SUPPORT_COS_MASK_ADC = 30.0
SUPPORT_K_PEAK = (5, 10, 20)
SUPPORT_K_BULK = (50, 100, 200)
SUPPORT_MIN_BL_BULK = 0.70
SUPPORT_DIAG_EPS = 0.05


# ----------------------------
# Reset / persistent outputs
# ----------------------------
if RESET_RESULTS or ("lh_units" not in globals()):
    lh_units = []
if RESET_RESULTS or ("lh_uncertain_ledger" not in globals()):
    lh_uncertain_ledger = []
if RESET_RESULTS or ("lh_channel_audit" not in globals()):
    lh_channel_audit = []

# fresh run-local summaries
skip_counts = defaultdict(int)


t_loop0 = time.time()
print(f"Starting LH loop over {len(CHANNELS_TO_RUN)} channels...")

for ch in CHANNELS_TO_RUN:
    channel_stage_times = {}
    ch_t0 = time.perf_counter()
    audit = _new_audit_record(ch)

    try:
        # ----------------------------
        # STEP 1: valley detection
        # ----------------------------
        step1 = _timed_call(
            "step1.find_valley_and_times",
            find_valley_and_times,
            raw_mod,
            ch,
            channel_tag=ch,
            timings=channel_stage_times,
            window=win,
            start=0,
            stop=None,
            bin_width=BIN_WIDTH,
            valley_bins=VALLEY_BINS,
            min_valid_count=MIN_VALID_COUNT,
            ratio_base=RATIO_BASE,
            ratio_step=RATIO_STEP,
            ratio_floor=RATIO_FLOOR,
            ratio_cap=RATIO_CAP,
            right_k=SUPPORT_N_PROBE_PER_SIDE,
            min_trough=-2500,
        )

        audit["step1_accepted"] = bool(step1.get("accepted", False))
        audit["left_count"] = int(step1.get("left_count", 0))
        audit["valley_count"] = int(step1.get("valley_count", 0))
        audit["required_ratio"] = (
            None if step1.get("required_ratio", None) is None else float(step1.get("required_ratio"))
        )
        audit["valley_low"] = step1.get("valley_low", None)
        audit["valley_high"] = step1.get("valley_high", None)
        audit["t_step1_s"] = channel_stage_times.get("step1.find_valley_and_times", np.nan)

        if int(step1.get("valley_count", 0)) > int(MAX_VALLEY_COUNT):
            skip_counts["valley_count>max"] += 1
            _set_reject(audit, "valley_count>max")
            print(
                f"[CH {ch}] REJECT valley_count>max | "
                f"valley_count={int(step1.get('valley_count', 0))}"
            )
            continue

        if not step1.get("accepted", False):
            skip_counts["valley_not_accepted"] += 1
            _set_reject(audit, "valley_not_accepted")
            print(
                f"[CH {ch}] REJECT valley_not_accepted | "
                f"left={int(step1.get('left_count', -1))} "
                f"valley={int(step1.get('valley_count', -1))}"
            )
            continue

        left_times = np.asarray(step1["left_times"], dtype=np.int64)

        km_win, km_win_info = _timed_call(
            "step2.choose_adaptive_km_window",
            choose_adaptive_km_window,
            raw_mod,
            left_times,
            channel_tag=ch,
            timings=channel_stage_times,
            probe_n=KM_PROBE_N,
            probe_win=KM_PROBE_WIN,
            fallback_win=win,
            time_amp_thr=KM_PROBE_TIME_AMP_THR,
            ch_ptp_thr=KM_PROBE_CH_P2P_THR,
            pad_left=KM_WIN_PAD_LEFT,
            pad_right=KM_WIN_PAD_RIGHT,
            min_pre=KM_WIN_MIN_PRE,
            min_post=KM_WIN_MIN_POST,
            rng=RNG,
        )

        audit["km_win_pre"] = int(km_win[0])
        audit["km_win_post"] = int(km_win[1])
        audit["km_probe_status"] = str(km_win_info.get("status", "unknown"))
        audit["km_probe_left_rel"] = km_win_info.get("left_rel", None)
        audit["km_probe_right_rel"] = km_win_info.get("right_rel", None)
        audit["km_probe_n_req"] = km_win_info.get("probe_n_req", None)
        audit["km_probe_n_valid"] = km_win_info.get("probe_n_valid", None)
        audit["km_probe_n_ch"] = km_win_info.get("n_ch_keep", None)
        audit["t_km_window_s"] = channel_stage_times.get("step2.choose_adaptive_km_window", np.nan)

        print(
            f"[CH {ch}] km_win={km_win} | "
            f"probe={km_win_info['status']} | "
            f"span={km_win_info['left_rel']}..{km_win_info['right_rel']} | "
            f"nprobe={km_win_info['probe_n_valid']}/{km_win_info['probe_n_req']} | "
            f"nch={km_win_info['n_ch_keep']}"
        )

        # ----------------------------
        # Early reject checks
        # ----------------------------
        isi_pairs_10_30 = _timed_call(
            "step2.compute_left_isi_pairs_10_30",
            compute_left_isi_pairs_10_30,
            step1,
            channel_tag=ch,
            timings=channel_stage_times,
        )
        audit["isi_pairs_10_30"] = int(isi_pairs_10_30)

        if isi_pairs_10_30 > ISI_10_30_MAX:
            skip_counts["abort_ISI10_30"] += 1
            _set_reject(audit, "abort_ISI10_30")
            print(f"[CH {ch}] REJECT abort_ISI10_30 | pairs={isi_pairs_10_30}")
            continue

        # ----------------------------
        # KMeans verdict on LEFT spikes
        # ----------------------------
        km_info = None
        km_plot = None
        base_ei = None

        if DO_KMEANS:
            left_times = np.asarray(step1["left_times"], dtype=np.int64)

            if left_times.size > int(SUBSAMPLE_MAX):
                pick = left_times[RNG.choice(left_times.size, int(SUBSAMPLE_MAX), replace=False)]
            else:
                pick = left_times

            snips_km, valid_times_km = _timed_call(
                "step3a.extract_snippets_fast_ram[kmeans]",
                extract_snippets_fast_ram,
                raw_mod,
                pick,
                channel_tag=ch,
                timings=channel_stage_times,
                window=km_win,
                selected_channels=np.arange(C, dtype=np.int32),
            )
            audit["t_kmeans_extract_s"] = channel_stage_times.get(
                "step3a.extract_snippets_fast_ram[kmeans]", np.nan
            )

            N = snips_km.shape[2]

            ei_mean = snips_km.mean(axis=2).astype(np.float32)
            base_ei = ei_mean
            rms = np.sqrt(np.mean(ei_mean ** 2, axis=1))
            sel = np.flatnonzero(rms > RMS_THRESH)
            if sel.size == 0:
                sel = np.argsort(rms)[-16:]
                sel.sort()

            X = snips_km[sel, :, :].transpose(2, 0, 1).reshape(
                N, sel.size * snips_km.shape[1]
            ).astype(np.float32)

            n_pc = int(min(N_PC, X.shape[0], X.shape[1]))
            pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
            Xpc = _timed_call(
                "step3b.PCA.fit_transform",
                pca.fit_transform,
                X,
                channel_tag=ch,
                timings=channel_stage_times,
            )
            vr = pca.explained_variance_ratio_

            km = KMeans(n_clusters=2, n_init=50, random_state=0)
            lab = _timed_call(
                "step3c.KMeans.fit_predict",
                km.fit_predict,
                Xpc,
                channel_tag=ch,
                timings=channel_stage_times,
            )

            n0 = int(np.sum(lab == 0))
            n1 = int(np.sum(lab == 1))

            ei_c0 = snips_km[:, :, lab == 0].mean(axis=2).astype(np.float32)
            ei_c1 = snips_km[:, :, lab == 1].mean(axis=2).astype(np.float32)

            precheck = _timed_call(
                "step3d.kmeans_precheck_decision",
                kmeans_precheck_decision,
                vr,
                n0,
                n1,
                ei_c0,
                ei_c1,
                channel_tag=ch,
                timings=channel_stage_times,
                pc_var_thr=KM_PC_VAR_THR,
                minor_frac_thr=KM_MINOR_FRAC_THR,
                cos_oneunit_thr=KM_EI_COS_THR,
                asym_unique_ch_min=KM_ASYM_UNIQUE_CH_MIN,
                support_rel=SUPPORT_REL,
                support_abs=SUPPORT_ABS,
                cos_lag=KM_COS_LAG,
            )

            called_verdict = False
            shared_core = None
            shared_dirs = None
            m01 = None
            m10 = None

            if precheck["decided"]:
                verdict = precheck["verdict"]
                proceed = bool(precheck["proceed"])
                reason = precheck["reason"]
                detail = precheck["detail"]
            else:
                called_verdict = True
                verdict_info = _timed_call(
                    "step3e.verdict_from_kmeans",
                    verdict_from_kmeans,
                    ei_c0,
                    ei_c1,
                    channel_tag=ch,
                    timings=channel_stage_times,
                    max_lag=MAX_LAG,
                    support_rel=SUPPORT_REL,
                    support_abs=SUPPORT_ABS,
                    time_keep_rel=TIME_KEEP_REL,
                    frac_in_thr=FRAC_IN_THR,
                    out_in_ratio_thr=OUT_IN_RATIO_THR,
                    resid_frac_min=RESID_FRAC_MIN,
                    shared_cos_thr=SHARED_COS_THR,
                    shared_alpha_thr=SHARED_ALPHA_THR,
                )
                verdict = verdict_info["verdict"]
                proceed = bool(verdict_info["proceed"])
                reason = "verdict_from_kmeans"
                detail = precheck["detail"]
                shared_core = bool(verdict_info["shared_core"])
                shared_dirs = verdict_info["shared_dirs"]
                m01 = verdict_info["m01"]
                m10 = verdict_info["m10"]

            km_info = dict(
                n0=n0,
                n1=n1,
                ei_mean=ei_mean,
                sel=np.asarray(sel, dtype=np.int32),
                ei_c0=ei_c0,
                ei_c1=ei_c1,
                vr=np.asarray(vr, dtype=np.float32),
                verdict=verdict,
                proceed=bool(proceed),
                reason=reason,
                detail=detail,
                called_verdict=bool(called_verdict),
                precheck=precheck,
                shared_core=shared_core,
                shared_dirs=shared_dirs,
                m01=m01,
                m10=m10,
            )
            km_plot = dict(Xpc=Xpc[:, :2], labels=lab, vr=vr)

            audit["kmeans_verdict"] = str(km_info["verdict"])
            audit["kmeans_reason"] = str(km_info["reason"])

            print(
                f"[CH {ch}] KMEANS | verdict={km_info['verdict']} | "
                f"reason={km_info['reason']} | detail={km_info['detail']} | "
                f"called_verdict={km_info['called_verdict']}"
            )

            if not km_info["proceed"]:
                skip_counts["kmeans_reject"] += 1
                _set_reject(audit, "kmeans_reject")
                print(f"[CH {ch}] REJECT kmeans_reject | verdict={km_info['verdict']}")

                if PLOT_DIAGNOSTICS and km_plot is not None:
                    plot_kmeans_pc_scatter(
                        km_plot,
                        title=f"CH {ch} | {km_info['verdict']} | {km_info['reason']}"
                    )

                continue

        if base_ei is None:
            left_times = np.asarray(step1["left_times"], dtype=np.int64)

            if left_times.size > int(SUBSAMPLE_MAX):
                pick = left_times[RNG.choice(left_times.size, int(SUBSAMPLE_MAX), replace=False)]
            else:
                pick = left_times

            snips_base, valid_times_base = _timed_call(
                "step3a.extract_snippets_fast_ram[kmeans]",
                extract_snippets_fast_ram,
                raw_mod,
                pick,
                channel_tag=ch,
                timings=channel_stage_times,
                window=km_win,
                selected_channels=np.arange(C, dtype=np.int32),
            )
            audit["t_kmeans_extract_s"] = channel_stage_times.get(
                "step3a.extract_snippets_fast_ram[kmeans]", np.nan
            )

            if snips_base.shape[2] == 0:
                skip_counts["base_ei_empty"] += 1
                _set_reject(audit, "base_ei_empty")
                print(f"[CH {ch}] REJECT base_ei_empty")
                continue

            base_ei = snips_base.mean(axis=2).astype(np.float32)

            print(
                f"[CH {ch}] KMEANS skipped | using mean EI from LEFT spikes "
                f"(N={snips_base.shape[2]}) | km_win={km_win}"
            )

        # ----------------------------
        # BL/TR candidates direct from step 1
        # ----------------------------
        tr_candidate_times = np.sort(
            np.unique(np.asarray(step1["rightk_times_sorted"], dtype=np.int64))
        )

        support_ei_info = dict(
            main_ch=int(np.argmin(np.asarray(base_ei, dtype=np.float32).min(axis=1))),
            final_ei=np.asarray(base_ei, dtype=np.float32),
        )

        # ----------------------------
        # BL/TR support filter
        # ----------------------------
        support_info = dict(ok=False, reason="not_run")
        clean_left_times = np.sort(np.unique(np.asarray(step1["left_times"], dtype=np.int64)))
        clean_right_times = np.sort(np.unique(np.asarray(tr_candidate_times, dtype=np.int64)))
        used_support_filter = False

        if USE_BL_TR_SUPPORT_FILTER:
            support_info = run_bl_tr_support_for_channel(
                raw_mod,
                support_ei_info,
                step1["left_times"],
                tr_candidate_times,
                ch=ch,
                win=km_win,
                n_probe_per_side=SUPPORT_N_PROBE_PER_SIDE,
                n_top_channels=SUPPORT_TOP_CHANNELS,
                cos_mask_adc=SUPPORT_COS_MASK_ADC,
                k_peak=SUPPORT_K_PEAK,
                k_bulk=SUPPORT_K_BULK,
                min_bl_bulk=SUPPORT_MIN_BL_BULK,
                diag_eps=SUPPORT_DIAG_EPS,
            )

            if support_info.get("ok", False):
                used_support_filter = True
                bl_reject = np.asarray(support_info["bl_reject_times"], dtype=np.int64)
                bl_uncertain = np.asarray(support_info["bl_uncertain_times"], dtype=np.int64)

                if bl_reject.size or bl_uncertain.size:
                    bl_drop = np.sort(np.unique(np.concatenate([bl_reject, bl_uncertain])))
                else:
                    bl_drop = np.asarray([], dtype=np.int64)

                clean_left_times = np.setdiff1d(clean_left_times, bl_drop, assume_unique=False)
                clean_right_times = np.sort(
                    np.unique(np.asarray(support_info["tr_keep_times"], dtype=np.int64))
                )

        # ----------------------------
        # Final spike list + final EI
        # ----------------------------
        final_times = np.sort(
            np.unique(
                np.concatenate([
                    np.asarray(clean_left_times, dtype=np.int64),
                    np.asarray(clean_right_times, dtype=np.int64),
                ])
            )
        )

        final_ei = np.asarray(base_ei, dtype=np.float32)
        main_ch_final = int(np.argmin(final_ei.min(axis=1)))
        audit["main_ch"] = int(main_ch_final)
        audit["final_times_n"] = int(final_times.size)
        audit["valley0_n"] = int(step1.get("valley_count", 0))
        audit["used_support_filter"] = bool(used_support_filter)

        if main_ch_final != int(ch):
            print(
                f"[CH {ch}] NOTE main/ref mismatch | "
                f"ref_ch={int(ch)} main_ch={main_ch_final}"
            )

        if final_times.size < int(MIN_FINAL_SPIKES_TO_CALL_SUCCESS):
            skip_counts["too_few_final_spikes"] += 1
            _set_reject(audit, "too_few_final_spikes")
            print(f"[CH {ch}] REJECT too_few_final_spikes | n={final_times.size}")
            continue

        support_summary = None
        if support_info.get("ok", False):
            do = support_info["decision_out"]
            support_summary = dict(
                params=do["params"],
                bl_counts=do["bl_counts"],
                tr_counts=do["tr_counts"],
                bl_labels=np.asarray(do["bl_labels"], dtype=object),
                tr_labels=np.asarray(do["tr_labels"], dtype=object),
                bl_probe_times=np.asarray(support_info["probe"]["bl_times"], dtype=np.int64),
                tr_probe_times=np.asarray(support_info["probe"]["tr_times"], dtype=np.int64),
                bl_uncertain_times=np.asarray(support_info["bl_uncertain_times"], dtype=np.int64),
                tr_uncertain_times=np.asarray(support_info["tr_uncertain_times"], dtype=np.int64),
                tr_keep_times=np.asarray(support_info["tr_keep_times"], dtype=np.int64),
                bl_points=np.asarray(
                    [[m["BL_bulk"], m["TR_bulk"]] for m in do["bl_metrics"]],
                    dtype=np.float32,
                ),
                tr_points=np.asarray(
                    [[m["BL_bulk"], m["TR_bulk"]] for m in do["tr_metrics"]],
                    dtype=np.float32,
                ),
            )

            lh_uncertain_ledger.append(
                dict(
                    detect_ch=int(ch),
                    main_ch=int(main_ch_final),
                    bl_uncertain_times=np.asarray(support_info["bl_uncertain_times"], dtype=np.int64),
                    tr_uncertain_times=np.asarray(support_info["tr_uncertain_times"], dtype=np.int64),
                    bl_probe_times=np.asarray(support_info["probe"]["bl_times"], dtype=np.int64),
                    tr_probe_times=np.asarray(support_info["probe"]["tr_times"], dtype=np.int64),
                )
            )

        # ----------------------------
        # Commit stage: only now build subtraction bank and subtract
        # ----------------------------
        final_proto = dict(
            main_ch=int(main_ch_final),
            final_ei=final_ei,
            final_times=final_times,
        )

        clean_bank = _timed_call(
            "step5.build_template_bank",
            build_template_bank,
            final_proto,
            channel_tag=ch,
            timings=channel_stage_times,
            raw_mod=raw_mod,
            win=km_win,
            sr=sr,
            n_bins=N_BINS,
            spikes_per=SPIKES_PER,
            ei_p2p_thr=EI_P2P_THR,
        )

        spike_times_to_subtract = np.asarray(final_times, dtype=np.int64)

        match = _timed_call(
            "step6.subtract_lh_masked",
            subtract_lh_masked,
            raw_mod,
            spike_times_to_subtract,
            clean_bank,
            channel_tag=ch,
            timings=channel_stage_times,
            time_keep_frac=TIME_KEEP_FRAC,
            weight_power=WEIGHT_POWER,
            use_ch_weight_sub=USE_CH_WEIGHT_SUB,
            min_sep_ms=MIN_SEP_MS,
            cos_thr_sub=COS_THR_SUB,
            jmax_sub=JMAX_SUB,
            batch_sub=BATCH_SUB,
        )

        if match is None or int(match.get("n_sub", 0)) == 0:
            skip_counts["subtraction_empty"] += 1
            _set_reject(audit, "subtraction_empty")
            print(f"[CH {ch}] REJECT subtraction_empty")
            continue

        final_spikes_main = np.asarray(match["spike_times_main"], dtype=np.int64)
        audit["final_spikes_main_n"] = int(final_spikes_main.size)

        unit = dict(
            detect_ch=int(ch),
            step1=step1,
            tr_candidate_times=np.asarray(tr_candidate_times, dtype=np.int64),
            final_times=np.asarray(final_times, dtype=np.int64),
            final_ei=np.asarray(final_ei, dtype=np.float32),
            bank=clean_bank,
            match=match,
            final_spikes_main=final_spikes_main,
            kmeans=km_info,
            support=support_summary,
        )

        unit["diag_isi_pairs_10_30"] = int(isi_pairs_10_30)
        unit["used_support_filter"] = bool(used_support_filter)

        lh_units.append(unit)
        audit["unit_index"] = int(len(lh_units))
        _set_success(audit)
        skip_counts["SUCCESS"] += 1

        if support_info.get("ok", False):
            do = support_info["decision_out"]
            blc = do["bl_counts"]
            trc = do["tr_counts"]

            bl_probe_soup_n = int(blc.get("soup", 0))
            bl_probe_unc_n = int(blc.get("uncertain_boundary", 0)) + int(blc.get("uncertain_lowBL", 0))
            tr_probe_lh_n = int(trc.get("LH", 0))

            audit["bl_probe_to_soup"] = bl_probe_soup_n
            audit["bl_probe_to_unc"] = bl_probe_unc_n
            audit["tr_probe_to_lh"] = tr_probe_lh_n

            extra_msg = (
                f" | valley0={audit['valley0_n']}"
                f" | BLprobe→soup={bl_probe_soup_n}"
                f" | BLprobe→unc={bl_probe_unc_n}"
                f" | TRprobe→LH={tr_probe_lh_n}"
            )
        else:
            extra_msg = f" | valley0={audit['valley0_n']} | BL/TRprobe=NA"

        print(
            f"✅ SUCCESS unit#{len(lh_units)} | detect_ch={ch} | "
            f"main_ch={match['main_ch']} | Nspikes={final_spikes_main.size}"
            f"{extra_msg}"
        )

        if PLOT_DIAGNOSTICS:
            _t_stage = _stage_start("step7.plot_diagnostics", channel_tag=ch)

            plot_amp_hist(step1, ch)
            if km_plot is not None:
                plot_kmeans_pc_scatter(
                    km_plot,
                    title=f"CH {ch} | {km_info['verdict']} | {km_info['reason']}"
                )
            if support_info.get("ok", False):
                plot_bl_tr_support_scatter(
                    support_info["decision_out"],
                    title=f"CH {ch} | BL/TR bulk support",
                )
            plot_final_ei(
                final_ei,
                clean_bank["main_ch"],
                title=f"Final EI | detect_ch={ch} main_ch={clean_bank['main_ch']} | N={final_spikes_main.size}",
            )

            _stage_end("step7.plot_diagnostics", _t_stage, timings=channel_stage_times, channel_tag=ch)

    except Exception as e:
        skip_counts["exception"] += 1
        audit["status"] = "exception"
        audit["final_reason"] = "exception"
        audit["error_type"] = type(e).__name__
        audit["error_msg"] = str(e)
        print(f"[CH {ch}] EXCEPTION: {type(e).__name__}: {e}")

    finally:
        audit["t_channel_total_s"] = time.perf_counter() - ch_t0
        if np.isnan(audit["t_km_window_s"]):
            audit["t_km_window_s"] = channel_stage_times.get("step2.choose_adaptive_km_window", np.nan)
        if np.isnan(audit["t_kmeans_extract_s"]):
            audit["t_kmeans_extract_s"] = channel_stage_times.get("step3a.extract_snippets_fast_ram[kmeans]", np.nan)
        if np.isnan(audit["t_step1_s"]):
            audit["t_step1_s"] = channel_stage_times.get("step1.find_valley_and_times", np.nan)

        _append_audit_record(audit)

        if VERBOSE_TIMING:
            print(f"[CH {ch}] ---- stage timings ----")
            for _label, _dt in channel_stage_times.items():
                print(f"[CH {ch}]   {_label:<36s} {_dt:7.2f} s")
            print(f"[CH {ch}] TOTAL channel | {audit['t_channel_total_s']:.2f} s")

        # Critical for large all-channel runs: do not accumulate figures
        try:
            plt.close("all")
        except Exception:
            pass


# ----------------------------
# End-of-run summaries
# ----------------------------
lh_channel_audit_by_ch = {int(rec["ch"]): rec for rec in lh_channel_audit}

success_records = [rec for rec in lh_channel_audit if rec.get("status") == "success"]
error_records = [rec for rec in lh_channel_audit if rec.get("status") == "exception"]
valley_not_accepted_records = [
    rec for rec in lh_channel_audit if rec.get("final_reason") == "valley_not_accepted"
]
valley_overflow_records = [
    rec for rec in lh_channel_audit if rec.get("final_reason") == "valley_count>max"
]

lh_round2_plan = dict(
    exclude_error_channels=np.array([rec["ch"] for rec in error_records], dtype=np.int32),
    exclude_valley_not_accepted_channels=np.array(
        [rec["ch"] for rec in valley_not_accepted_records], dtype=np.int32
    ),
    retry_valley_count_gt_max_channels=np.array(
        [rec["ch"] for rec in valley_overflow_records], dtype=np.int32
    ),
    successful_channels=np.array([rec["ch"] for rec in success_records], dtype=np.int32),
)

t_loop1 = time.time()
print("\n====================")
print(f"Done. Found {len(lh_units)} successful units.")
print(f"Elapsed: {t_loop1 - t_loop0:.1f} s")

print("\nRequested stats:")
print(f"  successful             : {len(success_records)}")
print(f"  errored                : {len(error_records)}")
print(f"  valley_not_accepted    : {len(valley_not_accepted_records)}")
print(f"  valley_count>max       : {len(valley_overflow_records)}")

overflow_bins = _build_valley_overflow_summary(
    lh_channel_audit,
    valley_limit=MAX_VALLEY_COUNT,
    bin_step=200,
    hard_cap=1000,
)
if len(overflow_bins) > 0:
    print(f"\nValley overflow breakdown (> {MAX_VALLEY_COUNT}):")
    for lo, hi, cnt in overflow_bins:
        if hi is None:
            print(f"  {lo:>4d}+ : {cnt}")
        else:
            print(f"  {lo:>4d}-{hi:<4d} : {cnt}")

print("\nAll rejection / outcome counts:")
reason_counts = defaultdict(int)
for rec in lh_channel_audit:
    key = rec.get("final_reason", "unknown")
    reason_counts[key] += 1
for k in sorted(reason_counts.keys()):
    print(f"  {k:>24s} : {reason_counts[k]}")

# ----------------------------
# End-of-run plots for successful channels
# ----------------------------
if PLOT_END_SUMMARY and len(success_records) > 0:
    success_records = sorted(success_records, key=lambda rec: rec["unit_index"])
    unit_order = np.arange(1, len(success_records) + 1, dtype=int)

    t_km_window = np.array([rec["t_km_window_s"] for rec in success_records], dtype=float)
    t_kmeans_extract = np.array([rec["t_kmeans_extract_s"] for rec in success_records], dtype=float)
    km_pre = np.array([rec["km_win_pre"] for rec in success_records], dtype=float)
    km_post = np.array([rec["km_win_post"] for rec in success_records], dtype=float)

    plt.figure(figsize=(12, 4))
    plt.plot(unit_order, t_km_window, marker="o", label="adaptive km_window")
    plt.plot(unit_order, t_kmeans_extract, marker="o", label="extract 5k for k-means")
    plt.xlabel("Successful unit order")
    plt.ylabel("Time (s)")
    plt.title("Adaptive-window timing on successful channels")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, max(4, 0.20 * len(success_records) + 2)))
    plt.scatter(km_pre, unit_order, label="pre", s=30)
    plt.scatter(km_post, unit_order, label="post", s=30)
    plt.xlabel("km_window limit (samples)")
    plt.ylabel("Successful unit order")
    plt.title("Adaptive km_window limits")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

# %% [markdown]
# # Save stuff 

# %%
import pickle
from pathlib import Path

# save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202510236/data018/lh_run_dump.pkl")
# save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202510236/lh_run_dump.pkl")
# save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202310301/data002/lh_run_dump.pkl")
# save_path = Path("/Volumes/Lab/Users/alexth/axolotl/202512300/data001/lh_run_dump.pkl")
# save_path = Path("/Volumes/Lab/Users/alexth/axolotl/201808079/data007_lh_run_dump.pkl")
save_path = Path("/Volumes/Lab/Users/alexth/axolotl/201711290/data004_lh_run_dump.pkl")


lh_dump = {
    "lh_units": lh_units,
    "lh_uncertain_ledger": lh_uncertain_ledger,
    "lh_channel_audit": lh_channel_audit,
    "lh_channel_audit_by_ch": lh_channel_audit_by_ch,
    "lh_round2_plan": lh_round2_plan,
    "skip_counts": dict(skip_counts),
    "channels_ran": CHANNELS_TO_RUN,
    "max_valley_count": MAX_VALLEY_COUNT,
}

with open(save_path, "wb") as f:
    pickle.dump(lh_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {save_path}")

# %% [markdown]
# ## Load saved stuff

# %%
import pickle
from pathlib import Path

save_path = Path("/Volumes/Lab/Users/alexth/axolotl/20250306C/data019_ks/lh_run_dump.pkl")

with open(save_path, "rb") as f:
    lh_dump = pickle.load(f)

lh_units = lh_dump["lh_units"]
lh_uncertain_ledger = lh_dump["lh_uncertain_ledger"]
lh_channel_audit = lh_dump["lh_channel_audit"]
lh_channel_audit_by_ch = lh_dump["lh_channel_audit_by_ch"]
lh_round2_plan = lh_dump["lh_round2_plan"]
skip_counts = lh_dump["skip_counts"]

print("Loaded dump.")

globals().update(lh_dump)

# %% [markdown]
# # ProjectionPursuit - new code

# %% [markdown]
# ## LOOP

# %% [markdown]
# ### helpers

# %%
# === LH/PP loop helpers (cleaned, deduplicated) ===

# NOTE:
# - If you want tiny-split AB-shard checking, make sure
#   `kmeans_precheck_decision` and `verdict_from_kmeans` are already loaded
#   into the notebook globals before using these helpers.


def make_default_lh_pp_checkpoint_path(dat_path, suffix="_lh_pp_loop.pkl"):
    base = os.path.splitext(os.path.basename(dat_path))[0]
    return os.path.join(os.path.dirname(dat_path), base + suffix)


def _sorted_unique_int64(x):
    x = np.asarray(x, dtype=np.int64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.int64)
    return np.unique(x)


def _ensure_int64_sorted(x):
    x = np.asarray(x, dtype=np.int64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.int64)
    return np.sort(x)


def _times_near_exclusions(times, exclude_times, radius):
    times = np.asarray(times, dtype=np.int64).ravel()
    if times.size == 0:
        return np.zeros(0, dtype=bool)

    if exclude_times is None:
        return np.zeros(times.size, dtype=bool)

    excl = np.asarray(exclude_times, dtype=np.int64).ravel()
    if excl.size == 0 or radius < 0:
        return np.zeros(times.size, dtype=bool)

    excl = np.unique(excl)
    left = np.searchsorted(excl, times - int(radius), side="left")
    right = np.searchsorted(excl, times + int(radius), side="right")
    return right > left


def _count_isi_10_30(times):
    times = np.asarray(times, dtype=np.int64).ravel()
    if times.size < 2:
        return 0
    d = np.diff(np.sort(times))
    return int(np.sum((d >= 10) & (d <= 30)))


def _sample_evenly_sorted_indices(n_total, n_keep):
    n_total = int(n_total)
    n_keep = int(min(n_keep, n_total))
    if n_keep <= 0:
        return np.array([], dtype=np.int64)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    idx = np.linspace(0, n_total - 1, n_keep)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, n_total - 1)
    idx = np.unique(idx)
    if idx.size == n_keep:
        return idx
    mask = np.ones(n_total, dtype=bool)
    mask[idx] = False
    extra = np.where(mask)[0]
    need = n_keep - idx.size
    if need > 0:
        idx = np.sort(np.concatenate([idx, extra[:need]]))
    return idx.astype(np.int64, copy=False)


def _roll_zero_2d(arr, shift):
    arr = np.asarray(arr)
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :arr.shape[1] - shift]
    else:
        s = -shift
        out[:, :arr.shape[1] - s] = arr[:, s:]
    return out


def _weighted_rms_score(x, tmpl, weights):
    resid = x - tmpl
    rms_pre_ch = np.sqrt(np.mean(x ** 2, axis=1))
    rms_post_ch = np.sqrt(np.mean(resid ** 2, axis=1))
    delta_rms_ch = rms_pre_ch - rms_post_ch
    score_pre = float(np.sum(weights * rms_pre_ch))
    score_post = float(np.sum(weights * rms_post_ch))
    return score_pre, score_post, rms_pre_ch, rms_post_ch, delta_rms_ch


def skyline_scan_with_exclusions(raw_mod, channel_state, params):
    """
    Recompute skyline ranking from scratch.
    For each channel, ignore local minima within +/- exclusion_radius of that channel's deferred times.
    """
    data_matrix = raw_mod
    if data_matrix.ndim != 2:
        raise ValueError("raw_mod must be [T, C]")

    n_samples_scan = int(params["skyline_n_samples"])
    skyline_top_k = int(params["skyline_top_k"])
    exclusion_radius = int(params["exclude_radius_samples"])

    T_total, n_channels = data_matrix.shape
    T_scan = min(n_samples_scan, T_total)

    mean_amp_topk = np.full(n_channels, np.nan, dtype=np.float32)
    n_minima_kept = np.zeros(n_channels, dtype=np.int32)
    n_minima_all = np.zeros(n_channels, dtype=np.int32)

    for ch in range(n_channels):
        x = data_matrix[:T_scan, ch].astype(np.float32, copy=False)
        idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
        n_minima_all[ch] = idx.size
        if idx.size == 0:
            continue

        excl = None
        if channel_state.get(ch) is not None:
            excl = channel_state[ch].get("exclude_times", None)

        near = _times_near_exclusions(idx.astype(np.int64), excl, exclusion_radius)
        idx_keep = idx[~near]
        n_minima_kept[ch] = idx_keep.size
        if idx_keep.size == 0:
            continue

        vals = x[idx_keep]
        k = min(skyline_top_k, vals.size)
        smallest_k = np.partition(vals, k - 1)[:k]
        mean_amp_topk[ch] = smallest_k.mean()

    score_for_sort = mean_amp_topk.copy()
    score_for_sort[np.isnan(score_for_sort)] = np.inf
    channel_order = np.argsort(score_for_sort)

    return {
        "channel_order": channel_order.astype(np.int32),
        "mean_amp_topk": mean_amp_topk,
        "n_minima_kept": n_minima_kept,
        "n_minima_all": n_minima_all,
        "n_samples_scan": int(T_scan),
        "skyline_top_k": int(skyline_top_k),
    }


def find_valley_and_times_excluding(
    raw_data,
    ch,
    window=(-20, 50),
    start=0,
    stop=None,
    bin_width=10.0,
    valley_bins=5,
    min_valid_count=300,
    ratio_base=3,
    ratio_step=100,
    ratio_floor=2,
    ratio_cap=10,
    exclude_times=None,
    exclude_radius=20,
):
    """
    Variant of find_valley_and_times that removes local minima near channel-local exclusions
    BEFORE building the histogram and valley decision.
    """
    T_total, n_channels = raw_data.shape
    if stop is None:
        stop = T_total
    if ch < 0 or ch >= n_channels:
        raise ValueError("Channel out of range")

    pre, post = int(window[0]), int(window[1])
    if pre >= 0 or post <= 0:
        raise ValueError("window must straddle the event")

    def _empty():
        return {
            "accepted": False,
            "valley_low": None,
            "valley_high": None,
            "left_times": np.array([], dtype=np.int64),
            "left_vals": np.array([], dtype=np.float32),
            "valley_times": np.array([], dtype=np.int64),
            "valley_vals": np.array([], dtype=np.float32),
            "left_count": 0,
            "valley_count": 0,
            "required_ratio": None,
            "analysis_span": (int(start), int(stop)),
            "amp_hist_counts": np.array([], dtype=np.int32),
            "amp_hist_edges": np.array([], dtype=np.float32),
            "all_times": np.array([], dtype=np.int64),
            "all_vals": np.array([], dtype=np.float32),
        }

    x = raw_data[start:stop, ch].astype(np.float32, copy=False)
    if x.size < 3:
        return _empty()

    idx_local = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    if idx_local.size == 0:
        return _empty()

    times_abs = start + idx_local.astype(np.int64)
    vals = x[idx_local].astype(np.float32, copy=False)

    ok = (times_abs + pre >= start) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals = vals[ok]
    if times_abs.size == 0:
        return _empty()

    if exclude_times is not None:
        near = _times_near_exclusions(times_abs, exclude_times, int(exclude_radius))
        times_abs = times_abs[~near]
        vals = vals[~near]
        if times_abs.size == 0:
            return _empty()

    all_times = np.asarray(times_abs, dtype=np.int64)
    all_vals = np.asarray(vals, dtype=np.float32)

    vmin = float(np.min(vals))
    if vmin >= 0:
        return _empty()

    bw = float(bin_width)
    low_edge = bw * np.floor(vmin / bw)
    high_edge = 0.0
    edges = np.linspace(low_edge, high_edge, int(round((high_edge - low_edge) / bw)) + 1, dtype=np.float64)
    if edges.size < valley_bins + 1:
        low_edge = low_edge - bw * ((valley_bins + 1) - edges.size + 1)
        edges = np.linspace(low_edge, high_edge, int(round((high_edge - low_edge) / bw)) + 1, dtype=np.float64)

    counts, edges = np.histogram(vals, bins=edges)
    nb = counts.size
    if nb < valley_bins + 1:
        return _empty()

    kernel = np.ones(int(valley_bins), dtype=np.int64)
    sums = np.convolve(counts, kernel, mode="valid")
    win_low_edges = edges[:-valley_bins]
    win_high_edges = edges[valley_bins:]
    neg_mask = win_high_edges <= 0.0
    if not np.any(neg_mask):
        return _empty()

    Sn = sums[neg_mask]
    lows = win_low_edges[neg_mask]
    highs = win_high_edges[neg_mask]
    m = Sn.size

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
    if peak_idxs.size == 0:
        return _empty()

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
        return _empty()

    prefix = np.concatenate(([0], np.cumsum(counts)))
    chosen = None
    for j in valley_idxs:
        vlow = float(lows[j])
        k_idx = int(np.searchsorted(edges, vlow, side="left"))
        left_est = int(prefix[k_idx])
        has_left_peak = np.any(peak_idxs < j)
        if has_left_peak and left_est >= int(min_valid_count):
            chosen = j
            break

    if chosen is None:
        last_frac = 0.35
        right_bound = max(0, int(np.floor((1.0 - last_frac) * m)))
        right_candidates = peak_idxs[peak_idxs >= right_bound]
        if right_candidates.size == 0:
            right_peak = int(peak_idxs[np.argmax(Sn[peak_idxs])])
        else:
            right_peak = int(right_candidates[np.argmax(Sn[right_candidates])])
        cand = valley_idxs[valley_idxs < right_peak]
        if cand.size == 0:
            return _empty()
        valley_rel = int(cand[-1])
    else:
        valley_rel = int(chosen)

    valley_low = float(lows[valley_rel])
    valley_high = float(highs[valley_rel])

    left_ev_mask = vals < valley_low
    valley_ev_mask = (vals >= valley_low) & (vals < valley_high)

    left_times = times_abs[left_ev_mask]
    left_vals = vals[left_ev_mask]
    valley_times = times_abs[valley_ev_mask]
    valley_vals = vals[valley_ev_mask]

    left_count = int(left_ev_mask.sum())
    valley_count = int(valley_ev_mask.sum())

    required_ratio = min(ratio_cap, max(ratio_floor, ratio_base + (valley_count // int(max(1, ratio_step)))))
    accepted = (left_count >= max(min_valid_count, required_ratio * max(1, valley_count)))

    li = np.argsort(left_times)
    vi = np.argsort(valley_times)

    return {
        "accepted": bool(accepted),
        "valley_low": valley_low,
        "valley_high": valley_high,
        "left_times": left_times[li].astype(np.int64, copy=False),
        "left_vals": left_vals[li].astype(np.float32, copy=False),
        "valley_times": valley_times[vi].astype(np.int64, copy=False),
        "valley_vals": valley_vals[vi].astype(np.float32, copy=False),
        "left_count": int(left_count),
        "valley_count": int(valley_count),
        "required_ratio": float(required_ratio),
        "analysis_span": (int(start), int(stop)),
        "amp_hist_counts": counts.astype(np.int32, copy=False),
        "amp_hist_edges": edges.astype(np.float32, copy=False),
        "all_times": all_times.astype(np.int64, copy=False),
        "all_vals": all_vals.astype(np.float32, copy=False),
    }


def find_pp_seed_valley_and_times_excluding(
    raw_data,
    ch,
    window=(-20, 50),
    start=0,
    stop=None,
    bin_width=10.0,
    valley_bins=5,
    min_valid_count=300,
    ratio_base=3,
    ratio_step=100,
    ratio_floor=2,
    ratio_cap=10,
    exclude_times=None,
    exclude_radius=20,
    min_left_count=300,
    max_valley_count=1000,
    max_valley_frac_of_left=1.0,
):
    """
    PP-loop seed version of the valley check.

    Reuse the existing valley *locator* unchanged, but apply a looser acceptance rule:
      - require a valid valley to be found
      - require enough spikes on the left
      - require the valley not to be huge in absolute count
      - require valley_count / left_count not to exceed a user threshold

    This keeps the LH-specific finder intact while letting PP use a softer seed criterion.
    """
    step1 = find_valley_and_times_excluding(
        raw_data=raw_data,
        ch=ch,
        window=window,
        start=start,
        stop=stop,
        bin_width=bin_width,
        valley_bins=valley_bins,
        min_valid_count=min_valid_count,
        ratio_base=ratio_base,
        ratio_step=ratio_step,
        ratio_floor=ratio_floor,
        ratio_cap=ratio_cap,
        exclude_times=exclude_times,
        exclude_radius=exclude_radius,
    )

    out = dict(step1)
    left_count = int(out.get("left_count", 0))
    valley_count = int(out.get("valley_count", 0))
    valley_found = out.get("valley_low", None) is not None

    if left_count > 0:
        valley_frac_of_left = float(valley_count) / float(left_count)
    else:
        valley_frac_of_left = np.inf

    if not valley_found:
        pp_seed_ok = False
        pp_seed_reason = "no_valley_found"
    elif left_count < int(min_left_count):
        pp_seed_ok = False
        pp_seed_reason = "left_count_too_small"
    elif valley_count > int(max_valley_count):
        pp_seed_ok = False
        pp_seed_reason = "valley_count_too_large"
    elif valley_frac_of_left > float(max_valley_frac_of_left):
        pp_seed_ok = False
        pp_seed_reason = "valley_fraction_too_high"
    else:
        pp_seed_ok = True
        pp_seed_reason = "ok"

    out.update({
        "lh_accepted": bool(step1.get("accepted", False)),
        "pp_seed_ok": bool(pp_seed_ok),
        "pp_seed_reason": str(pp_seed_reason),
        "valley_frac_of_left": float(valley_frac_of_left),
        "pp_min_left_count": int(min_left_count),
        "pp_max_valley_count": int(max_valley_count),
        "pp_max_valley_frac_of_left": float(max_valley_frac_of_left),
    })
    return out


def build_pp_pool_for_channel(raw_mod, detect_ch, channel_state, params):
    """
    Cell 2 logic for the PP loop, with same-channel exclusions applied before valley finding.

    Important change:
      - keep the existing LH valley *finder*
      - use a looser PP-specific seed acceptance rule
    """
    fs = int(params["fs"])
    duration_sec = float(params["pp_duration_sec"])
    stop = min(int(duration_sec * fs), raw_mod.shape[0])
    window_pp = tuple(params["window_pp"])
    exclusion_radius = int(params["exclude_radius_samples"])

    excl = np.array([], dtype=np.int64)
    if channel_state.get(int(detect_ch)) is not None:
        excl = channel_state[int(detect_ch)].get("exclude_times", np.array([], dtype=np.int64))

    step1 = find_pp_seed_valley_and_times_excluding(
        raw_data=raw_mod,
        ch=int(detect_ch),
        window=window_pp,
        start=0,
        stop=stop,
        bin_width=float(params["valley_bin_width"]),
        valley_bins=int(params["valley_bins"]),
        min_valid_count=int(params["valley_min_valid_count"]),
        ratio_base=int(params["valley_ratio_base"]),
        ratio_step=int(params["valley_ratio_step"]),
        ratio_floor=int(params["valley_ratio_floor"]),
        ratio_cap=int(params["valley_ratio_cap"]),
        exclude_times=excl,
        exclude_radius=exclusion_radius,
        min_left_count=int(params["pp_valley_min_left_count"]),
        max_valley_count=int(params["pp_valley_max_count"]),
        max_valley_frac_of_left=float(params["pp_valley_max_frac_of_left"]),
    )

    if not step1["pp_seed_ok"]:
        return {
            "status": "fail",
            "reason": "pp_seed_{}".format(step1["pp_seed_reason"]),
            "step1": step1,
        }

    all_times = np.asarray(step1["all_times"], dtype=np.int64)
    all_vals = np.asarray(step1["all_vals"], dtype=np.float32)
    if all_times.size == 0:
        return {
            "status": "fail",
            "reason": "empty_minima_after_exclusion",
            "step1": step1,
        }

    valley_low = float(step1["valley_low"])
    left_mask = all_vals < valley_low
    right_mask = all_vals >= valley_low

    left_times = all_times[left_mask]
    right_times_all = all_times[right_mask]
    right_vals_all = all_vals[right_mask]

    n_left = int(left_times.size)
    if n_left == 0:
        return {
            "status": "fail",
            "reason": "empty_left_pool",
            "step1": step1,
        }

    right_order = np.argsort(right_vals_all)
    right_keep = right_order[:min(n_left, right_times_all.size)]
    right_times = right_times_all[right_keep]
    if right_times.size == 0:
        return {
            "status": "fail",
            "reason": "empty_right_pool",
            "step1": step1,
        }

    full_times = np.concatenate([left_times, right_times]).astype(np.int64)
    full_side = np.concatenate([
        np.zeros(left_times.size, dtype=np.int32),
        np.ones(right_times.size, dtype=np.int32),
    ])

    return {
        "status": "ok",
        "detect_ch": int(detect_ch),
        "stop": int(stop),
        "window_pp": tuple(window_pp),
        "step1": step1,
        "left_times": left_times.astype(np.int64, copy=False),
        "right_times": right_times.astype(np.int64, copy=False),
        "full_times": full_times,
        "full_side": full_side,
    }


def _local_maxima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    if y[0] > y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] > y[i - 1]) and (y[i] >= y[i + 1]):
            out.append(i)
    if y[-1] > y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=np.int64)


def _local_minima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n < 2:
        return np.array([], dtype=np.int64)
    if y[0] < y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] < y[i - 1]) and (y[i] <= y[i + 1]):
            out.append(i)
    if y[-1] < y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=np.int64)


def _normalize_direction(w):
    w = np.asarray(w, dtype=np.float32).ravel()
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)


def _robust_zscore_1d_fit(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32).ravel()
    med = float(np.median(x)) if x.size else 0.0
    mad = float(np.median(np.abs(x - med))) if x.size else 0.0
    scale = float(1.4826 * max(mad, eps))
    z = (x - med) / scale
    return z.astype(np.float32), med, scale


def _robust_zscore_1d_apply(x, med, scale):
    x = np.asarray(x, dtype=np.float32).ravel()
    return ((x - float(med)) / float(scale)).astype(np.float32)


def _robust_zscore_cols_fit(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float32)
    med = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - med[None, :]), axis=0).astype(np.float32)
    scale = (1.4826 * np.maximum(mad, eps)).astype(np.float32)
    Z = (X - med[None, :]) / scale[None, :]
    return Z.astype(np.float32), med, scale


def _robust_zscore_cols_apply(X, med, scale):
    X = np.asarray(X, dtype=np.float32)
    med = np.asarray(med, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    return ((X - med[None, :]) / scale[None, :]).astype(np.float32)


def _detect_row_trough_feature(sn_sel, ref_row, center_idx):
    sn_sel = np.asarray(sn_sel, dtype=np.float32)
    lo = max(0, int(center_idx) - 1)
    hi = min(sn_sel.shape[1] - 1, int(center_idx) + 1)
    amp = -sn_sel[int(ref_row), lo:hi + 1, :].min(axis=0)
    return np.asarray(amp, dtype=np.float32)


def _build_biased_direction_bank(
    n_aug,
    n_pcs,
    rng,
    amp_pc1_weight=0.5,
    n_tail_dirs=16,
    tail_scale=0.08,
):
    dirs = []

    if n_pcs >= 1:
        w = np.zeros(n_aug, dtype=np.float32)
        w[1] = 1.0
        dirs.append(("pc1_only", _normalize_direction(w)))

        w = np.zeros(n_aug, dtype=np.float32)
        w[0] = 1.0
        w[1] = float(amp_pc1_weight)
        dirs.append(("amp_plus_pc1", _normalize_direction(w)))

        n_tail = max(0, n_aug - 2)
        for i in range(int(n_tail_dirs)):
            w = np.zeros(n_aug, dtype=np.float32)
            w[0] = 1.0
            w[1] = float(amp_pc1_weight)
            if n_tail > 0:
                w[2:] = rng.uniform(
                    -float(tail_scale),
                    float(tail_scale),
                    size=n_tail,
                ).astype(np.float32)
            dirs.append((f"amp_plus_pc1_tail_{i:02d}", _normalize_direction(w)))

    return dirs


def _random_unit_directions(n_dirs, dim, rng):
    W = rng.normal(size=(n_dirs, dim)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W


def _score_projection_1d(z, bins, sigma, min_child_n, min_group_frac, target_frac):
    z = np.asarray(z, dtype=np.float32).ravel()
    N = z.size
    hist, edges = np.histogram(z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = gaussian_filter1d(hist.astype(np.float32), sigma=sigma)

    peaks = _local_maxima(hs)
    valleys = _local_minima(hs)

    best = None
    for v in valleys:
        left_peaks = peaks[peaks < v]
        right_peaks = peaks[peaks > v]
        if left_peaks.size == 0 or right_peaks.size == 0:
            continue

        pl = left_peaks[-1]
        pr = right_peaks[0]

        valley_h = float(hs[v])
        peak_l_h = float(hs[pl])
        peak_r_h = float(hs[pr])
        min_peak_h = min(peak_l_h, peak_r_h)
        if min_peak_h <= 0:
            continue

        thr = float(centers[v])
        mask_left = z <= thr
        n_left = int(mask_left.sum())
        n_right = int(N - n_left)
        n_minor = min(n_left, n_right)
        frac_minor = n_minor / float(N)

        if n_left < int(min_child_n) or n_right < int(min_child_n):
            continue
        if frac_minor < float(min_group_frac):
            continue

        depth = 1.0 - (valley_h / (min_peak_h + 1e-12))

        zl = z[mask_left]
        zr = z[~mask_left]
        mu_l = float(zl.mean())
        mu_r = float(zr.mean())
        sd_l = float(zl.std(ddof=1)) if zl.size > 1 else 1e-12
        sd_r = float(zr.std(ddof=1)) if zr.size > 1 else 1e-12
        pooled_sd = np.sqrt(0.5 * (sd_l * sd_l + sd_r * sd_r) + 1e-12)
        sep = abs(mu_r - mu_l) / pooled_sd

        size_bonus = 1.0 - np.exp(-frac_minor / float(target_frac))
        score = depth * sep * size_bonus

        cand = {
            "score": float(score),
            "thr": thr,
            "depth": float(depth),
            "sep": float(sep),
            "size_bonus": float(size_bonus),
            "n_left": n_left,
            "n_right": n_right,
            "n_minor": n_minor,
            "frac_minor": float(frac_minor),
            "hist": hist,
            "hist_smooth": hs,
            "edges": edges,
            "centers": centers,
            "valley_idx": int(v),
            "mask_left": mask_left,
            "mu_l": mu_l,
            "mu_r": mu_r,
            "sd_l": sd_l,
            "sd_r": sd_r,
            "pooled_sd": float(pooled_sd),
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    return best


def run_recursive_pp_for_channel(raw_mod, ei_positions, pp_pool_state, params):
    """
    Recursive PP for the loop version.

    Key updates vs the older loop version:
      - augmented feature space with detect-channel amplitude
      - deterministic direction bank before random directions
      - per-node split bookkeeping so preflight can use the deepest *scoreable* split
      - tiny-child detailed check via kmeans_precheck_decision / verdict_from_kmeans
      - shard-like tiny branches are kept as deferred leaves (not merged back)
    """
    if pp_pool_state["status"] != "ok":
        raise ValueError("pp_pool_state must be successful")

    data_matrix = raw_mod
    anchor_ch = int(pp_pool_state["detect_ch"])
    window_pp = tuple(pp_pool_state["window_pp"])
    full_times = np.asarray(pp_pool_state["full_times"], dtype=np.int64)
    full_side = np.asarray(pp_pool_state["full_side"], dtype=np.int32)

    N0 = full_times.size
    if full_side.size != N0:
        raise RuntimeError("full_times/full_side mismatch")

    max_depth = int(params["pp_max_depth"])
    min_node_n = int(params["pp_min_node_n"])
    min_child_n = int(params["pp_min_child_n"])
    discover_max_per_side = int(params["pp_discover_max_per_side"])
    n_search_pcs = int(params["pp_n_search_pcs"])
    n_random_dirs = int(params["pp_n_random_dirs"])
    hist_bins = int(params["pp_hist_bins"])
    smooth_sigma = float(params["pp_smooth_sigma"])
    min_score = float(params["pp_min_score"])
    min_depth = float(params["pp_min_depth"])
    min_sep = float(params["pp_min_sep"])
    min_group_frac = float(params["pp_min_group_frac"])
    target_frac = float(params["pp_target_frac"])
    p2p_thr = float(params["pp_p2p_thr"])
    max_channels = int(params["pp_max_channels"])
    min_channels = int(params["pp_min_channels"])
    left_recurse_thresh = float(params["pp_left_recurse_thresh"])
    rng = np.random.default_rng(int(params["pp_rng_seed"]))

    amp_pc1_weight = float(params["pp_amp_pc1_weight"])
    n_biased_tail_dirs = int(params["pp_n_biased_tail_dirs"])
    biased_tail_scale = float(params["pp_biased_tail_scale"])

    scoreable_minor_n_cap = int(params["pp_scoreable_minor_n_cap"])
    scoreable_minor_frac = float(params["pp_scoreable_minor_frac"])

    km_pc_var_thr = float(params["pp_abcheck_pc_var_thr"])
    km_minor_frac_thr = float(params["pp_abcheck_minor_frac_thr"])
    km_ei_cos_thr = float(params["pp_abcheck_ei_cos_thr"])
    km_asym_unique_ch_min = int(params["pp_abcheck_asym_unique_ch_min"])
    km_support_rel = float(params["pp_abcheck_support_rel"])
    km_support_abs = float(params["pp_abcheck_support_abs"])
    km_cos_lag = int(params["pp_abcheck_cos_lag"])

    max_lag_verdict = int(params["pp_abcheck_max_lag"])
    time_keep_rel = float(params["pp_abcheck_time_keep_rel"])
    frac_in_thr = float(params["pp_abcheck_frac_in_thr"])
    out_in_ratio_thr = float(params["pp_abcheck_out_in_ratio_thr"])
    resid_frac_min = float(params["pp_abcheck_resid_frac_min"])
    shared_cos_thr = float(params["pp_abcheck_shared_cos_thr"])
    shared_alpha_thr = float(params["pp_abcheck_shared_alpha_thr"])

    def extract_snips_for_times(times, selected_channels):
        sn, valid_times = extract_snippets_fast_ram(
            raw_data=data_matrix,
            spike_times=np.asarray(times, dtype=np.int64),
            window=window_pp,
            selected_channels=np.asarray(selected_channels, dtype=np.int32),
        )
        sn = sn.astype(np.float32, copy=False)
        valid_times = np.asarray(valid_times, dtype=np.int64)
        if valid_times.size != np.asarray(times).size:
            raise RuntimeError("Unexpected edge-drop during snippet extraction in PP recursion")
        return sn

    def build_node_discovery_subset(idx_node):
        idx_node = np.asarray(idx_node, dtype=np.int64)
        side = full_side[idx_node]
        idx_left = idx_node[side == 0]
        idx_right = idx_node[side == 1]
        if idx_left.size > discover_max_per_side:
            keep = np.sort(rng.choice(idx_left.size, size=discover_max_per_side, replace=False))
            idx_left = idx_left[keep]
        if idx_right.size > discover_max_per_side:
            keep = np.sort(rng.choice(idx_right.size, size=discover_max_per_side, replace=False))
            idx_right = idx_right[keep]
        idx_disc = np.concatenate([idx_left, idx_right])
        return np.asarray(idx_disc, dtype=np.int64)

    def summarize_child(idx_child, times_child):
        idx_child = np.asarray(idx_child, dtype=np.int64)
        times_child = np.asarray(times_child, dtype=np.int64)
        n = idx_child.size
        n_orig_left = int(np.sum(full_side[idx_child] == 0))
        n_orig_right = int(np.sum(full_side[idx_child] == 1))
        frac_orig_left = n_orig_left / float(n) if n > 0 else np.nan
        frac_orig_right = n_orig_right / float(n) if n > 0 else np.nan

        if n > 0:
            sn_full = extract_snips_for_times(times_child, np.arange(data_matrix.shape[1], dtype=np.int32))
            if sn_full.shape[2] == 0:
                amp_anchor = np.nan
                ei_child = None
            else:
                ei_child = sn_full.mean(axis=2).astype(np.float32)
                amp_anchor = float(-ei_child[anchor_ch].min())
        else:
            amp_anchor = np.nan
            ei_child = None

        return {
            "n": int(n),
            "n_orig_left": n_orig_left,
            "n_orig_right": n_orig_right,
            "frac_orig_left": frac_orig_left,
            "frac_orig_right": frac_orig_right,
            "amp_anchor": amp_anchor,
            "majority_left": bool(frac_orig_left > left_recurse_thresh) if np.isfinite(frac_orig_left) else False,
            "ei": ei_child,
        }

    def make_leaf(idx_leaf, depth, path, reason):
        return {
            "type": "leaf",
            "path": path,
            "depth": depth,
            "n": int(np.asarray(idx_leaf, dtype=np.int64).size),
            "reason": str(reason),
            "idx": np.asarray(idx_leaf, dtype=np.int64),
        }

    def maybe_recurse_child(idx_child, depth, path, summary_child):
        if not summary_child["majority_left"]:
            frac = 100.0 * summary_child["frac_orig_left"]
            return make_leaf(
                idx_child,
                depth,
                path,
                "not_pursued_right_heavy (orig-left={:.1f}% <= {:.1f}%)".format(
                    frac,
                    100.0 * left_recurse_thresh,
                ),
            )
        return try_split(idx_child, depth, path)

    def _classify_small_split(ei0, ei1, n0, n1):
        info = {
            "called": False,
            "source": None,
            "verdict": None,
            "proceed": None,
            "detail": None,
            "shared_core": None,
            "raw": None,
        }

        if (ei0 is None) or (ei1 is None):
            info["verdict"] = "missing_child_ei"
            info["proceed"] = False
            return info

        if ("kmeans_precheck_decision" not in globals()) or ("verdict_from_kmeans" not in globals()):
            info["verdict"] = "kmeans_helpers_missing"
            info["proceed"] = False
            return info

        try:
            precheck = kmeans_precheck_decision(
                np.array([0.0, 0.0], dtype=np.float32),
                int(n0),
                int(n1),
                np.asarray(ei0, dtype=np.float32),
                np.asarray(ei1, dtype=np.float32),
                pc_var_thr=km_pc_var_thr,
                minor_frac_thr=km_minor_frac_thr,
                cos_oneunit_thr=km_ei_cos_thr,
                asym_unique_ch_min=km_asym_unique_ch_min,
                support_rel=km_support_rel,
                support_abs=km_support_abs,
                cos_lag=km_cos_lag,
            )

            if precheck["decided"]:
                info["called"] = True
                info["source"] = "kmeans_precheck_decision"
                info["verdict"] = str(precheck["verdict"])
                info["proceed"] = bool(precheck["proceed"])
                info["detail"] = str(precheck.get("detail", ""))
                info["raw"] = precheck
                return info

            verdict_info = verdict_from_kmeans(
                np.asarray(ei0, dtype=np.float32),
                np.asarray(ei1, dtype=np.float32),
                max_lag=max_lag_verdict,
                support_rel=km_support_rel,
                support_abs=km_support_abs,
                time_keep_rel=time_keep_rel,
                frac_in_thr=frac_in_thr,
                out_in_ratio_thr=out_in_ratio_thr,
                resid_frac_min=resid_frac_min,
                shared_cos_thr=shared_cos_thr,
                shared_alpha_thr=shared_alpha_thr,
            )
            info["called"] = True
            info["source"] = "verdict_from_kmeans"
            info["verdict"] = str(verdict_info["verdict"])
            info["proceed"] = bool(verdict_info["proceed"])
            info["detail"] = str(verdict_info.get("verdict", ""))
            info["shared_core"] = bool(verdict_info.get("shared_core", False))
            info["raw"] = verdict_info
            return info

        except Exception as exc:
            info["called"] = True
            info["source"] = "exception"
            info["verdict"] = "small_split_check_error"
            info["proceed"] = False
            info["detail"] = str(exc)
            return info

    def _choose_deferred_side(summary_left, summary_right):
        if summary_left["n"] < summary_right["n"]:
            return "L"
        if summary_right["n"] < summary_left["n"]:
            return "R"

        if summary_left["majority_left"] and (not summary_right["majority_left"]):
            return "R"
        if summary_right["majority_left"] and (not summary_left["majority_left"]):
            return "L"

        frac_left_L = float(summary_left["frac_orig_left"]) if np.isfinite(summary_left["frac_orig_left"]) else -np.inf
        frac_left_R = float(summary_right["frac_orig_left"]) if np.isfinite(summary_right["frac_orig_left"]) else -np.inf
        if frac_left_L < frac_left_R:
            return "L"
        if frac_left_R < frac_left_L:
            return "R"

        return "R"

    def try_split(idx_node, depth, path):
        idx_node = np.asarray(idx_node, dtype=np.int64)
        n_full = idx_node.size
        if n_full < min_node_n:
            return make_leaf(idx_node, depth, path, "too_small_for_split (<{})".format(min_node_n))

        idx_disc = build_node_discovery_subset(idx_node)
        n_disc = idx_disc.size
        if n_disc < min_node_n:
            return make_leaf(idx_node, depth, path, "discovery_subset_too_small (<{})".format(min_node_n))

        disc_times = full_times[idx_disc]
        sn_disc_full = extract_snips_for_times(disc_times, np.arange(data_matrix.shape[1], dtype=np.int32))
        if sn_disc_full.shape[2] == 0:
            return make_leaf(idx_node, depth, path, "no_valid_discovery_snippets")

        idx_disc_valid = idx_disc.copy()
        side_disc_valid = full_side[idx_disc_valid]
        ei_disc = sn_disc_full.mean(axis=2).astype(np.float32)

        selected_channels, _ = select_template_channels(
            ei_disc,
            p2p_thr=p2p_thr,
            max_n=max_channels,
            min_n=min_channels,
            force_include_main=True,
        )
        selected_channels = np.asarray(selected_channels, dtype=np.int64)
        if anchor_ch not in selected_channels:
            selected_channels = np.concatenate([selected_channels, np.array([anchor_ch], dtype=np.int64)])
        selected_channels = np.asarray(np.unique(selected_channels), dtype=np.int64)
        anchor_row = int(np.where(selected_channels == anchor_ch)[0][0])

        sn_disc_sel = sn_disc_full[selected_channels, :, :]
        X_disc = sn_disc_sel.transpose(2, 0, 1).reshape(sn_disc_sel.shape[2], -1).astype(np.float32)

        n_pcs = int(min(n_search_pcs, X_disc.shape[0], X_disc.shape[1]))
        if n_pcs < 2:
            return make_leaf(idx_node, depth, path, "not_enough_rank_for_pca")

        pca = PCA(n_components=n_pcs)
        pcs_disc = pca.fit_transform(X_disc).astype(np.float32)

        center_idx = int(-window_pp[0])
        amp_disc = _detect_row_trough_feature(sn_disc_sel, anchor_row, center_idx)

        pcs_disc_z, pc_med, pc_scale = _robust_zscore_cols_fit(pcs_disc)
        amp_disc_z, amp_med, amp_scale = _robust_zscore_1d_fit(amp_disc)

        feat_disc = np.concatenate([amp_disc_z[:, None], pcs_disc_z], axis=1).astype(np.float32)
        feature_names = ["amp_det"] + [f"PC{i+1}" for i in range(n_pcs)]
        n_aug = int(feat_disc.shape[1])

        candidate_dirs = _build_biased_direction_bank(
            n_aug=n_aug,
            n_pcs=n_pcs,
            rng=rng,
            amp_pc1_weight=amp_pc1_weight,
            n_tail_dirs=n_biased_tail_dirs,
            tail_scale=biased_tail_scale,
        )

        W_rand = _random_unit_directions(n_random_dirs, n_aug, rng)
        for i_dir in range(n_random_dirs):
            candidate_dirs.append((f"random_{i_dir:04d}", W_rand[i_dir]))

        best_disc = None
        best_w = None
        best_dir_name = None
        best_proj_disc = None

        for dir_name, w in candidate_dirs:
            z_disc = feat_disc @ w
            res = _score_projection_1d(
                z_disc,
                bins=hist_bins,
                sigma=smooth_sigma,
                min_child_n=min_child_n,
                min_group_frac=min_group_frac,
                target_frac=target_frac,
            )
            if res is None:
                continue
            if (best_disc is None) or (res["score"] > best_disc["score"]):
                best_disc = res
                best_w = w.copy()
                best_dir_name = str(dir_name)
                best_proj_disc = z_disc.copy()

        if best_disc is None:
            return make_leaf(idx_node, depth, path, "no_valid_valley_split_on_discovery_subset")

        pos_map_full = {int(i): pos for pos, i in enumerate(idx_node.tolist())}

        z_full = np.empty(n_full, dtype=np.float32)
        pcs_full = np.empty((n_full, n_pcs), dtype=np.float32)
        feat_full = np.empty((n_full, n_aug), dtype=np.float32)

        z_disc_best = feat_disc @ best_w
        for gidx, pcs_row, feat_row, zval in zip(idx_disc_valid.tolist(), pcs_disc, feat_disc, z_disc_best):
            pos = pos_map_full[int(gidx)]
            pcs_full[pos] = pcs_row
            feat_full[pos] = feat_row
            z_full[pos] = zval

        disc_set = set(idx_disc_valid.tolist())
        idx_extra = np.array([i for i in idx_node.tolist() if i not in disc_set], dtype=np.int64)
        extra_times = full_times[idx_extra]

        if idx_extra.size > 0:
            sn_extra_sel = extract_snips_for_times(extra_times, selected_channels)
            if sn_extra_sel.shape[2] > 0:
                X_extra = sn_extra_sel.transpose(2, 0, 1).reshape(sn_extra_sel.shape[2], -1).astype(np.float32)
                pcs_extra = pca.transform(X_extra).astype(np.float32)
                pcs_extra_z = _robust_zscore_cols_apply(pcs_extra, pc_med, pc_scale)

                amp_extra = _detect_row_trough_feature(sn_extra_sel, anchor_row, center_idx)
                amp_extra_z = _robust_zscore_1d_apply(amp_extra, amp_med, amp_scale)

                feat_extra = np.concatenate([amp_extra_z[:, None], pcs_extra_z], axis=1).astype(np.float32)
                z_extra = feat_extra @ best_w

                for gidx, pcs_row, feat_row, zval in zip(idx_extra.tolist(), pcs_extra, feat_extra, z_extra):
                    pos = pos_map_full[int(gidx)]
                    pcs_full[pos] = pcs_row
                    feat_full[pos] = feat_row
                    z_full[pos] = zval

        best_full = _score_projection_1d(
            z_full,
            bins=hist_bins,
            sigma=smooth_sigma,
            min_child_n=min_child_n,
            min_group_frac=min_group_frac,
            target_frac=target_frac,
        )
        if best_full is None:
            return make_leaf(idx_node, depth, path, "no_valid_valley_split_on_full_node")

        if best_full["score"] < min_score:
            return make_leaf(idx_node, depth, path, "score_too_low ({:.3f} < {:.3f})".format(best_full["score"], min_score))
        if best_full["depth"] < min_depth:
            return make_leaf(idx_node, depth, path, "depth_too_low ({:.3f} < {:.3f})".format(best_full["depth"], min_depth))
        if best_full["sep"] < min_sep:
            return make_leaf(idx_node, depth, path, "sep_too_low ({:.3f} < {:.3f})".format(best_full["sep"], min_sep))

        thr_full = best_full["thr"]
        proj_left_mask_full = z_full <= thr_full

        disc_pos_in_full = np.array([pos_map_full[int(i)] for i in idx_disc_valid.tolist()], dtype=np.int64)
        z_disc_under_full = z_full[disc_pos_in_full]
        disc_mask_proj_left = z_disc_under_full <= thr_full

        if np.any(disc_mask_proj_left):
            frac_left_proj_left = np.mean(side_disc_valid[disc_mask_proj_left] == 0)
        else:
            frac_left_proj_left = -np.inf

        if np.any(~disc_mask_proj_left):
            frac_left_proj_right = np.mean(side_disc_valid[~disc_mask_proj_left] == 0)
        else:
            frac_left_proj_right = -np.inf

        if frac_left_proj_left >= frac_left_proj_right:
            mask_leftlike = proj_left_mask_full
            leftlike_is_proj_left = True
        else:
            mask_leftlike = ~proj_left_mask_full
            leftlike_is_proj_left = False

        idx_left = idx_node[mask_leftlike]
        idx_right = idx_node[~mask_leftlike]
        times_left = full_times[idx_left]
        times_right = full_times[idx_right]
        summary_left = summarize_child(idx_left, times_left)
        summary_right = summarize_child(idx_right, times_right)

        n_minor = int(min(summary_left["n"], summary_right["n"]))
        sizeable_minor_thr = int(max(1, min(scoreable_minor_n_cap, np.floor(scoreable_minor_frac * n_full))))
        scoreable_for_preflight = bool(n_minor >= sizeable_minor_thr)

        small_split_check = {
            "called": False,
            "source": None,
            "verdict": None,
            "proceed": None,
            "detail": None,
            "shared_core": None,
            "raw": None,
        }
        shard_like_small_split = False

        if not scoreable_for_preflight:
            small_split_check = _classify_small_split(
                summary_left["ei"],
                summary_right["ei"],
                summary_left["n"],
                summary_right["n"],
            )
            shard_like_small_split = bool(small_split_check.get("proceed", False))

        node = {
            "type": "split",
            "path": path,
            "depth": depth,
            "n_full": int(n_full),
            "n_disc": int(n_disc),
            "idx": idx_node,
            "idx_disc": idx_disc_valid,
            "selected_channels": selected_channels,
            "pca": pca,
            "best_w": best_w,
            "pcs_full": pcs_full,
            "feat_full": feat_full,
            "feature_names": feature_names,
            "best_dir_name": best_dir_name,
            "best_disc": best_disc,
            "best_full": best_full,
            "best_proj_full": z_full,
            "best_proj_disc": best_proj_disc,
            "mask_leftlike": mask_leftlike,
            "leftlike_is_proj_left": bool(leftlike_is_proj_left),
            "idx_left": idx_left,
            "idx_right": idx_right,
            "summary_left": summary_left,
            "summary_right": summary_right,
            "scoreable_for_preflight": bool(scoreable_for_preflight),
            "sizeable_minor_threshold": int(sizeable_minor_thr),
            "small_split_check": small_split_check,
            "small_split_shard_like": bool(shard_like_small_split),
        }

        if depth >= max_depth:
            node["left_child"] = make_leaf(idx_left, depth + 1, path + ".L", "max_depth_reached ({})".format(max_depth))
            node["right_child"] = make_leaf(idx_right, depth + 1, path + ".R", "max_depth_reached ({})".format(max_depth))
            return node

        if shard_like_small_split and (not scoreable_for_preflight):
            deferred_side = _choose_deferred_side(summary_left, summary_right)
            verdict = str(small_split_check.get("verdict", "AB-like"))
            source = str(small_split_check.get("source", "unknown"))

            if deferred_side == "L":
                node["left_child"] = make_leaf(
                    idx_left,
                    depth + 1,
                    path + ".L",
                    f"deferred_small_split [{verdict}] via {source}",
                )
                node["right_child"] = maybe_recurse_child(idx_right, depth + 1, path + ".R", summary_right)
            else:
                node["left_child"] = maybe_recurse_child(idx_left, depth + 1, path + ".L", summary_left)
                node["right_child"] = make_leaf(
                    idx_right,
                    depth + 1,
                    path + ".R",
                    f"deferred_small_split [{verdict}] via {source}",
                )
            return node

        node["left_child"] = maybe_recurse_child(idx_left, depth + 1, path + ".L", summary_left)
        node["right_child"] = maybe_recurse_child(idx_right, depth + 1, path + ".R", summary_right)
        return node

    def collect_leaves(node, out=None):
        if out is None:
            out = []
        if node["type"] == "leaf":
            out.append(node)
        else:
            collect_leaves(node["left_child"], out)
            collect_leaves(node["right_child"], out)
        return out

    root_idx = np.arange(N0, dtype=np.int64)
    pp_tree = try_split(root_idx, depth=0, path="root")
    pp_leaves = collect_leaves(pp_tree)

    return {
        "status": "ok",
        "anchor_ch": int(anchor_ch),
        "window_pp": tuple(window_pp),
        "left_recurse_thresh": float(left_recurse_thresh),
        "tree": pp_tree,
        "leaves": pp_leaves,
        "full_times": full_times,
        "full_side": full_side,
    }


def finalize_labels_and_short_bank(raw_mod, pp_tree_state, pool_state, params):
    """
    Final labeling semantics:

    - main_path_score comes from the LAST scoreable split on the chosen main path
    - trusted_left = only the chosen main leaf, after removing boundary-uncertain spikes
      from scoreable splits up to and including that final score node
    - uncertain = boundary-uncertain spikes from ALL scoreable path nodes up to and
      including the final score node (both sides of those splits)
    - extra_left = ONLY sibling subtrees produced AFTER the final score node, along the
      ultimately chosen main path (AB-shard / same-unit-small-split style leftovers)
    - everything else = trusted_not_left

    This means:
      * large sibling branches on the opposite side of the final valid score are NOT extra_left
      * exclusion/deferral will later follow exactly these corrected labels
    """
    data_matrix = raw_mod
    anchor_ch = int(pp_tree_state["anchor_ch"])
    window_pp = tuple(pp_tree_state["window_pp"])
    pp_tree = pp_tree_state["tree"]
    pp_leaves = pp_tree_state["leaves"]
    full_times = np.asarray(pp_tree_state["full_times"], dtype=np.int64)
    full_side = np.asarray(pp_tree_state["full_side"], dtype=np.int32)

    margin_k = float(params["pp_margin_k"])
    left_leaf_thresh = float(params["pp_left_leaf_thresh"])
    template_reducer = str(params["short_bank_reducer"])
    template_n_bins = int(params["short_bank_n_bins"])
    template_min_bin_size = int(params["short_bank_min_bin_size"])
    close_dt = int(params["trusted_close_dt"])
    lag_radius = int(params["assign_lag_radius"])

    def extract_full_snips(times):
        sn, valid_times = extract_snippets_fast_ram(
            raw_data=data_matrix,
            spike_times=np.asarray(times, dtype=np.int64),
            window=window_pp,
            selected_channels=np.arange(data_matrix.shape[1], dtype=np.int32),
        )
        sn = sn.astype(np.float32, copy=False)
        valid_times = np.asarray(valid_times, dtype=np.int64)
        if valid_times.size != np.asarray(times).size:
            raise RuntimeError("Unexpected edge-drop while building full snippets")
        return sn

    def summarize_leaf(idx_leaf):
        idx_leaf = np.asarray(idx_leaf, dtype=np.int64)
        n = idx_leaf.size
        n_left = int(np.sum(full_side[idx_leaf] == 0))
        n_right = int(np.sum(full_side[idx_leaf] == 1))
        frac_left = n_left / float(n) if n > 0 else np.nan
        if n > 0:
            sn = extract_full_snips(full_times[idx_leaf])
            ei = sn.mean(axis=2).astype(np.float32)
            amp_anchor = float(-ei[anchor_ch].min())
        else:
            ei = None
            amp_anchor = np.nan
        return {
            "n": int(n),
            "n_left": n_left,
            "n_right": n_right,
            "frac_left": frac_left,
            "amp_anchor": amp_anchor,
            "ei": ei,
        }

    def _is_deferred_small_split_leaf(leaf):
        return str(leaf.get("reason", "")).startswith("deferred_small_split")

    def collect_majority_left_leaves(leaves, thresh):
        out = []
        for leaf in leaves:
            # keep old safeguard: tiny deferred leaves should never become the main leaf
            if _is_deferred_small_split_leaf(leaf):
                continue
            idx = np.asarray(leaf["idx"], dtype=np.int64)
            if idx.size == 0:
                continue
            frac_left = np.mean(full_side[idx] == 0)
            if frac_left > thresh:
                out.append(leaf)
        return out

    def choose_main_left_leaf(leaves, thresh):
        cand = collect_majority_left_leaves(leaves, thresh=thresh)
        if len(cand) == 0:
            raise RuntimeError("No majority-left leaves found.")
        scored = []
        for leaf in cand:
            s = summarize_leaf(leaf["idx"])
            scored.append((leaf, s))
        scored.sort(key=lambda t: (t[1]["n"], t[1]["amp_anchor"], t[1]["frac_left"]), reverse=True)
        return scored[0][0], scored

    def get_nodes_on_path(tree, leaf_path):
        parts = leaf_path.split(".")
        if parts[0] != "root":
            raise ValueError("Unexpected leaf_path")
        nodes = []
        node = tree
        for branch in parts[1:]:
            if node["type"] != "split":
                raise RuntimeError("Path hits a leaf too early")
            nodes.append((node, branch))
            node = node["left_child"] if branch == "L" else node["right_child"]
        return nodes

    def classify_main_leaf_core(main_leaf, path_nodes_local, score_node_pos_local, margin_k_local):
        """
        Keep only spikes in the chosen main leaf that are confidently on the chosen side
        of ALL scoreable splits up to and including the final score node.

        Important: post-score tiny splits do NOT participate here.
        """
        main_idx = np.asarray(main_leaf["idx"], dtype=np.int64)
        if main_idx.size == 0:
            return np.array([], dtype=np.int64)

        trusted_mask = np.ones(main_idx.size, dtype=bool)

        if score_node_pos_local < 0:
            return main_idx.copy()

        for node, branch in path_nodes_local[:score_node_pos_local + 1]:
            node_idx = np.asarray(node["idx"], dtype=np.int64)
            z_full = np.asarray(node["best_proj_full"], dtype=np.float32)
            thr = float(node["best_full"]["thr"])
            pooled_sd = float(node["best_full"]["pooled_sd"])
            margin = float(margin_k_local) * pooled_sd

            idx_to_local = {int(gidx): pos for pos, gidx in enumerate(node_idx.tolist())}
            local_pos = np.array([idx_to_local[int(gidx)] for gidx in main_idx], dtype=np.int64)
            z_main = z_full[local_pos]

            leftlike_is_proj_left = bool(node["leftlike_is_proj_left"])
            if branch == "L":
                if leftlike_is_proj_left:
                    cond = z_main <= (thr - margin)
                else:
                    cond = z_main >= (thr + margin)
            elif branch == "R":
                if leftlike_is_proj_left:
                    cond = z_main >= (thr + margin)
                else:
                    cond = z_main <= (thr - margin)
            else:
                raise RuntimeError("Unexpected branch in path")

            trusted_mask &= cond

        return main_idx[trusted_mask]

    def build_lh_style_template_bank(times):
        times = np.asarray(times, dtype=np.int64)
        if times.size == 0:
            raise RuntimeError("No trusted-left spikes to build template bank")

        sn_full = extract_full_snips(times)
        C, L, N = sn_full.shape

        if template_reducer == "median":
            provisional_ei = median_ei_adaptive(sn_full).astype(np.float32)
        elif template_reducer == "mean":
            provisional_ei = sn_full.mean(axis=2).astype(np.float32)
        else:
            raise ValueError("short_bank_reducer must be 'median' or 'mean'")

        main_ch = int(np.argmin(provisional_ei.min(axis=1)))
        t0 = int(np.argmin(provisional_ei[main_ch]))
        lo = max(0, t0 - 1)
        hi = min(L - 1, t0 + 1)

        spike_amp = sn_full[main_ch, lo:hi + 1, :].min(axis=0).astype(np.float32)
        order = np.argsort(spike_amp)
        times_sorted = times[order]
        amp_sorted = spike_amp[order]
        sn_sorted = sn_full[:, :, order]

        n_bins_eff = min(int(template_n_bins), max(1, N // int(template_min_bin_size)))
        groups = np.array_split(np.arange(N), n_bins_eff)

        templates = []
        for bi, g in enumerate(groups):
            if g.size == 0:
                continue
            sn_bin = sn_sorted[:, :, g]
            if template_reducer == "median":
                tmpl = median_ei_adaptive(sn_bin).astype(np.float32)
            else:
                tmpl = sn_bin.mean(axis=2).astype(np.float32)
            templates.append({
                "bin_index": int(bi),
                "n_spikes": int(g.size),
                "times": times_sorted[g].astype(np.int64),
                "amp_min": float(np.min(amp_sorted[g])),
                "amp_max": float(np.max(amp_sorted[g])),
                "template": tmpl,
            })

        return {
            "window": tuple(window_pp),
            "reducer": template_reducer,
            "main_ch": int(main_ch),
            "t0_main": int(t0),
            "provisional_ei": provisional_ei,
            "times_all": times_sorted,
            "amp_all": amp_sorted,
            "templates": templates,
        }

    def score_single_time_against_bank(t, template_bank, lag_radius_local, channels=None):
        t = int(t)
        templates = template_bank["templates"]
        if len(templates) == 0:
            return np.inf, None, None

        if channels is None:
            ei_ref = np.asarray(template_bank["provisional_ei"], dtype=np.float32)
            p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
            channels = np.argsort(p2p)[-20:].astype(np.int32)
        else:
            channels = np.asarray(channels, dtype=np.int32)

        best_score = np.inf
        best_bin = None
        best_lag = None

        for lag in range(-int(lag_radius_local), int(lag_radius_local) + 1):
            tt = t + lag
            sn, valid_times = extract_snippets_fast_ram(
                raw_data=data_matrix,
                spike_times=np.array([tt], dtype=np.int64),
                window=window_pp,
                selected_channels=channels,
            )
            sn = sn.astype(np.float32, copy=False)
            if sn.shape[2] != 1:
                continue
            x = sn[:, :, 0]
            for rec in templates:
                tmpl = np.asarray(rec["template"], dtype=np.float32)[channels, :]
                resid = x - tmpl
                score = float(np.sqrt(np.mean(resid ** 2)))
                if score < best_score:
                    best_score = score
                    best_bin = int(rec["bin_index"])
                    best_lag = int(lag)

        return best_score, best_bin, best_lag

    def collapse_close_times_by_template_fit(times, template_bank):
        times = np.asarray(times, dtype=np.int64)
        if times.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), []

        t_sorted = np.sort(times)
        groups = []
        start = 0
        for i in range(t_sorted.size - 1):
            if (t_sorted[i + 1] - t_sorted[i]) >= int(close_dt):
                groups.append(t_sorted[start:i + 1])
                start = i + 1
        groups.append(t_sorted[start:])

        kept = []
        dropped = []
        records = []

        ei_ref = np.asarray(template_bank["provisional_ei"], dtype=np.float32)
        p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
        score_channels = np.argsort(p2p)[-20:].astype(np.int32)

        for g in groups:
            if g.size == 1:
                kept.append(int(g[0]))
                records.append({
                    "group_times": g.astype(np.int64),
                    "winner_time": int(g[0]),
                    "winner_score": np.nan,
                    "winner_bin": None,
                    "winner_lag": None,
                    "n_group": 1,
                })
                continue

            cand_scores = []
            for t in g:
                sc, bi, lg = score_single_time_against_bank(
                    t,
                    template_bank,
                    lag_radius_local=lag_radius,
                    channels=score_channels,
                )
                cand_scores.append((float(sc), int(t), bi, lg))

            cand_scores.sort(key=lambda x: x[0])
            winner_score, winner_time, winner_bin, winner_lag = cand_scores[0]
            kept.append(int(winner_time))
            for _, t, _, _ in cand_scores[1:]:
                dropped.append(int(t))
            records.append({
                "group_times": g.astype(np.int64),
                "winner_time": int(winner_time),
                "winner_score": float(winner_score),
                "winner_bin": winner_bin,
                "winner_lag": winner_lag,
                "n_group": int(g.size),
                "all_scores": cand_scores,
            })

        kept = np.array(sorted(set(kept)), dtype=np.int64)
        dropped = np.array(sorted(set(dropped)), dtype=np.int64)
        return kept, dropped, records

    def collect_boundary_uncertain_idx(path_nodes_local, upto_pos_inclusive, margin_k_local):
        """
        Union of near-threshold spikes from ALL scoreable path nodes up to and including
        the final score node. This pools uncertainty from both sides of each scored split.
        """
        pieces = []
        used_paths = []

        if upto_pos_inclusive < 0:
            return np.array([], dtype=np.int64), []

        for node, _branch in path_nodes_local[:upto_pos_inclusive + 1]:
            node_idx = np.asarray(node["idx"], dtype=np.int64)
            z_full = np.asarray(node["best_proj_full"], dtype=np.float32)
            thr = float(node["best_full"]["thr"])
            pooled_sd = float(node["best_full"]["pooled_sd"])
            if (not np.isfinite(thr)) or (not np.isfinite(pooled_sd)):
                continue
            margin = float(margin_k_local) * float(pooled_sd)
            near = np.abs(z_full - thr) < margin
            if np.any(near):
                pieces.append(node_idx[near])
                used_paths.append(str(node["path"]))

        if len(pieces) == 0:
            return np.array([], dtype=np.int64), []
        return _sorted_unique_int64(np.concatenate(pieces)), used_paths

    def collect_postscore_extra_idx(path_nodes_local, score_node_pos_local):
        """
        Extra-left = sibling subtrees from splits AFTER the final score node,
        along the chosen main path only.
        """
        pieces = []
        used_paths = []

        if score_node_pos_local < 0:
            return np.array([], dtype=np.int64), []

        for node, branch in path_nodes_local[score_node_pos_local + 1:]:
            sibling = node["right_child"] if branch == "L" else node["left_child"]
            sib_idx = np.asarray(sibling.get("idx", []), dtype=np.int64)
            if sib_idx.size == 0:
                continue
            pieces.append(sib_idx)
            used_paths.append(str(sibling.get("path", None)))

        if len(pieces) == 0:
            return np.array([], dtype=np.int64), []
        return _sorted_unique_int64(np.concatenate(pieces)), used_paths

    if pp_tree["type"] != "split":
        raise RuntimeError("PP root did not split; no unit to finalize.")

    main_leaf, scored_majority_left = choose_main_left_leaf(pp_leaves, thresh=left_leaf_thresh)
    main_path = main_leaf["path"]
    majority_left_leaves = [leaf for leaf, _ in scored_majority_left]

    path_nodes = get_nodes_on_path(pp_tree, main_path)
    if len(path_nodes) > 0:
        last_node = path_nodes[-1][0]
        last_path_score = float(last_node["best_full"]["score"])
        last_path_sep = float(last_node["best_full"]["sep"])
        last_path_depth = float(last_node["best_full"]["depth"])
        last_path_split_path = str(last_node["path"])

        scoreable_pos = [i for i, (node, _br) in enumerate(path_nodes) if bool(node.get("scoreable_for_preflight", False))]
        if len(scoreable_pos) > 0:
            score_node_pos = int(scoreable_pos[-1])
            score_node = path_nodes[score_node_pos][0]
            main_path_score = float(score_node["best_full"]["score"])
            main_path_sep = float(score_node["best_full"]["sep"])
            main_path_depth = float(score_node["best_full"]["depth"])
            main_path_split_path = str(score_node["path"])
        else:
            # compatibility fallback
            score_node_pos = len(path_nodes) - 1
            score_node = last_node
            main_path_score = float(last_path_score)
            main_path_sep = float(last_path_sep)
            main_path_depth = float(last_path_depth)
            main_path_split_path = str(last_path_split_path)
    else:
        score_node_pos = -1
        score_node = None
        main_path_score = np.nan
        main_path_sep = np.nan
        main_path_depth = np.nan
        main_path_split_path = None
        last_path_score = np.nan
        last_path_sep = np.nan
        last_path_depth = np.nan
        last_path_split_path = None

    trusted_left_idx_raw = classify_main_leaf_core(
        main_leaf,
        path_nodes_local=path_nodes,
        score_node_pos_local=score_node_pos,
        margin_k_local=margin_k,
    )

    uncertain_idx_raw, uncertain_source_paths = collect_boundary_uncertain_idx(
        path_nodes,
        upto_pos_inclusive=score_node_pos,
        margin_k_local=margin_k,
    )

    extra_left_idx_raw, extra_left_source_paths = collect_postscore_extra_idx(
        path_nodes,
        score_node_pos_local=score_node_pos,
    )

    mask_extra_left = np.zeros(full_times.size, dtype=bool)
    if extra_left_idx_raw.size > 0:
        mask_extra_left[extra_left_idx_raw] = True

    uncertain_idx = uncertain_idx_raw[~mask_extra_left[uncertain_idx_raw]]
    mask_uncertain = np.zeros(full_times.size, dtype=bool)
    if uncertain_idx.size > 0:
        mask_uncertain[uncertain_idx] = True

    trusted_left_idx = trusted_left_idx_raw[
        (~mask_extra_left[trusted_left_idx_raw]) & (~mask_uncertain[trusted_left_idx_raw])
    ]
    mask_trusted_left = np.zeros(full_times.size, dtype=bool)
    if trusted_left_idx.size > 0:
        mask_trusted_left[trusted_left_idx] = True

    mask_trusted_not_left = ~(mask_trusted_left | mask_extra_left | mask_uncertain)

    overlap_count = (
        np.sum(mask_trusted_left & mask_extra_left)
        + np.sum(mask_trusted_left & mask_uncertain)
        + np.sum(mask_extra_left & mask_uncertain)
    )
    if overlap_count != 0:
        raise RuntimeError("Label overlap detected")

    trusted_left_times = full_times[mask_trusted_left]
    extra_left_times = full_times[mask_extra_left]
    uncertain_times = full_times[mask_uncertain]
    trusted_not_left_times = full_times[mask_trusted_not_left]

    if trusted_left_times.size == 0:
        raise RuntimeError("No trusted_left spikes after PP finalization.")

    lh_template_bank_provisional = build_lh_style_template_bank(trusted_left_times)
    trusted_left_times_clean, trusted_left_times_dropped_close, close_group_records = collapse_close_times_by_template_fit(
        trusted_left_times,
        lh_template_bank_provisional,
    )

    if trusted_left_times_clean.size == 0:
        raise RuntimeError("No trusted_left spikes remain after close-neighbor cleanup.")

    lh_template_bank = build_lh_style_template_bank(trusted_left_times_clean)
    isi_10_30_trusted_left = _count_isi_10_30(trusted_left_times_clean)

    labels = np.empty(full_times.size, dtype=object)
    labels[mask_trusted_left] = "trusted_left"
    labels[mask_extra_left] = "extra_left"
    labels[mask_uncertain] = "uncertain"
    labels[mask_trusted_not_left] = "trusted_not_left"

    final_left_state = {
        "anchor_ch": int(anchor_ch),
        "window_pp": tuple(window_pp),
        "step1": pool_state["step1"],
        "main_leaf_path": main_path,
        "margin_k": float(margin_k),
        "trusted_left_idx": trusted_left_idx.astype(np.int64, copy=False),
        "extra_left_idx": extra_left_idx_raw.astype(np.int64, copy=False),
        "uncertain_idx": uncertain_idx.astype(np.int64, copy=False),
        "deferred_small_split_idx": np.array([], dtype=np.int64),  # retired bucket
        "trusted_not_left_idx": np.where(mask_trusted_not_left)[0].astype(np.int64, copy=False),
        "trusted_left_times_raw": trusted_left_times.astype(np.int64, copy=False),
        "trusted_left_times": trusted_left_times_clean.astype(np.int64, copy=False),
        "trusted_left_times_dropped_close": trusted_left_times_dropped_close.astype(np.int64, copy=False),
        "close_group_records": close_group_records,
        "isi_10_30_trusted_left": int(isi_10_30_trusted_left),
        "extra_left_times": extra_left_times.astype(np.int64, copy=False),
        "uncertain_times": uncertain_times.astype(np.int64, copy=False),
        "deferred_small_split_times": np.array([], dtype=np.int64),  # retired bucket
        "trusted_not_left_times": trusted_not_left_times.astype(np.int64, copy=False),
        "labels": labels,
        "majority_left_leaves": [leaf["path"] for leaf in majority_left_leaves],
        "extra_left_leaf_paths": list(extra_left_source_paths),   # subtree roots after final score
        "deferred_small_split_leaf_paths": [],                    # retired bucket
        "uncertain_source_paths": list(uncertain_source_paths),
        "main_path_score": float(main_path_score),
        "main_path_sep": float(main_path_sep),
        "main_path_depth": float(main_path_depth),
        "main_path_split_path": main_path_split_path,
        "last_path_score": float(last_path_score),
        "last_path_sep": float(last_path_sep),
        "last_path_depth": float(last_path_depth),
        "last_path_split_path": last_path_split_path,
    }

    return final_left_state, lh_template_bank


def evaluate_pp_preflight(unit_record_preview, pool_state, params):
    """
    Stage-1 preflight after a dry-run assignment, before mutating raw_mod.

    Metrics used for first-pass channel commitment:
      - valley_count
      - valley_frac_of_left
      - PP score
      - deferral fraction
    """
    step1 = dict(pool_state["step1"])
    summary = dict(unit_record_preview["summary"])

    n_accepted = int(summary["n_accepted"])
    n_deferred = int(summary["n_deferred_unique"])
    denom = max(1, n_accepted + n_deferred)
    deferral_frac = float(n_deferred) / float(denom)

    pp_score = float(unit_record_preview["pp_main_score"])
    min_pp_score = float(params["preflight_min_pp_score"])
    max_deferral_frac = float(params["preflight_max_deferral_frac"])

    checks = {
        "pp_score_ok": bool(pp_score >= min_pp_score),
        "deferral_ok": bool(deferral_frac <= max_deferral_frac),
    }

    if not checks["pp_score_ok"]:
        reason = "preflight_pp_score_low"
    elif not checks["deferral_ok"]:
        reason = "preflight_deferral_frac_high"
    else:
        reason = "ok"

    return {
        "ok": bool(all(checks.values())),
        "reason": str(reason),
        "pp_score": float(pp_score),
        "pp_score_min": float(min_pp_score),
        "deferral_frac": float(deferral_frac),
        "deferral_frac_max": float(max_deferral_frac),
        "left_count": int(step1["left_count"]),
        "valley_count": int(step1["valley_count"]),
        "valley_frac_of_left": float(step1["valley_frac_of_left"]),
        "lh_accepted": bool(step1.get("lh_accepted", False)),
        "checks": checks,
    }


def assign_and_subtract_unit(raw_mod, final_left_state, lh_template_bank, detect_ch, params, dry_run=False):
    """
    Cell 6 logic, then canonicalize to main-channel minima, post-lag dedup, build long bank, subtract.

    If dry_run=True:
      - do the full assignment / bookkeeping preview
      - do NOT mutate raw_mod
    """
    data_matrix = raw_mod
    window_pp = tuple(final_left_state["window_pp"])
    trusted_left_times = np.asarray(final_left_state["trusted_left_times"], dtype=np.int64)
    if trusted_left_times.size == 0:
        return {"status": "fail", "reason": "no_trusted_left_times"}

    # Average detection-channel spike amplitude from trusted-left spikes.
    # Use the short PP window and take the local trough around the window center.
    detect_ch = int(detect_ch)
    center_idx_short = int(-window_pp[0])

    sn_detect, valid_detect_times = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=trusted_left_times,
        window=window_pp,
        selected_channels=np.array([detect_ch], dtype=np.int32),
    )
    sn_detect = sn_detect.astype(np.float32, copy=False)

    if sn_detect.shape[2] > 0:
        lo = max(0, center_idx_short - 1)
        hi = min(sn_detect.shape[1] - 1, center_idx_short + 1)
        trusted_left_detect_amp_mean = float(-np.mean(np.min(sn_detect[0, lo:hi + 1, :], axis=0)))
    else:
        trusted_left_detect_amp_mean = np.nan

    lag_radius = int(params["assign_lag_radius"])
    n_score_channels = int(params["assign_n_score_channels"])
    min_improvement_frac = float(params["assign_min_improvement_frac"])
    require_main_improvement = bool(params["assign_require_main_improvement"])
    postlag_close_dt = int(params["postlag_close_dt"])
    long_window = tuple(params["long_subtraction_window"])
    long_reducer = str(params["long_bank_reducer"])
    long_max_per_bin = int(params["long_bank_max_spikes_per_bin"])

    ei_ref = np.asarray(lh_template_bank["provisional_ei"], dtype=np.float32)
    main_ch = int(lh_template_bank["main_ch"])
    t0_main = int(lh_template_bank["t0_main"])

    p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
    score_channels = np.argsort(p2p)[-n_score_channels:][::-1].astype(np.int32)
    if main_ch not in score_channels:
        score_channels[-1] = main_ch
    score_channels = np.array(sorted(set(score_channels.tolist()), key=lambda c: p2p[c], reverse=True), dtype=np.int32)

    weights = p2p[score_channels].astype(np.float32)
    weights = weights / (weights.sum() + 1e-12)

    sn_score, valid_times = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=trusted_left_times,
        window=window_pp,
        selected_channels=score_channels,
    )
    sn_score = sn_score.astype(np.float32, copy=False)
    valid_times = np.asarray(valid_times, dtype=np.int64)

    if valid_times.size != trusted_left_times.size:
        return {"status": "fail", "reason": "edge_drop_in_assignment"}

    Csel, L, N = sn_score.shape
    template_records = lh_template_bank["templates"]
    if len(template_records) == 0:
        return {"status": "fail", "reason": "empty_short_template_bank"}

    lags = np.arange(-lag_radius, lag_radius + 1, dtype=int)
    tmpl_score_bank = {}
    for rec in template_records:
        bi = int(rec["bin_index"])
        tmpl_full = np.asarray(rec["template"], dtype=np.float32)
        tmpl_score = tmpl_full[score_channels, :]
        for lag in lags:
            tmpl_score_bank[(bi, int(lag))] = _roll_zero_2d(tmpl_score, int(lag))

    best_bin = np.full(N, -1, dtype=np.int32)
    best_lag = np.full(N, 0, dtype=np.int32)
    pre_score = np.full(N, np.nan, dtype=np.float32)
    post_score = np.full(N, np.nan, dtype=np.float32)
    accepted = np.zeros(N, dtype=bool)
    main_rms_pre = np.full(N, np.nan, dtype=np.float32)
    main_rms_post = np.full(N, np.nan, dtype=np.float32)
    delta_rms_ch = np.full((score_channels.size, N), np.nan, dtype=np.float32)
    aligned_main_amp = np.full(N, np.nan, dtype=np.float32)

    main_row_arr = np.where(score_channels == main_ch)[0]
    main_row = int(main_row_arr[0]) if main_row_arr.size else 0

    for i in range(N):
        x = sn_score[:, :, i]
        best_post = np.inf
        best_pre = np.nan
        best_bin_i = -1
        best_lag_i = 0
        best_delta_ch = None
        best_rms_pre_ch = None
        best_rms_post_ch = None

        for rec in template_records:
            bi = int(rec["bin_index"])
            for lag in lags:
                tmpl = tmpl_score_bank[(bi, int(lag))]
                sc_pre, sc_post, rms_pre_ch, rms_post_ch, d_ch = _weighted_rms_score(x, tmpl, weights)
                if sc_post < best_post:
                    best_post = sc_post
                    best_pre = sc_pre
                    best_bin_i = bi
                    best_lag_i = int(lag)
                    best_delta_ch = d_ch
                    best_rms_pre_ch = rms_pre_ch
                    best_rms_post_ch = rms_post_ch

        best_bin[i] = best_bin_i
        best_lag[i] = best_lag_i
        pre_score[i] = best_pre
        post_score[i] = best_post
        delta_rms_ch[:, i] = best_delta_ch.astype(np.float32)
        main_rms_pre[i] = float(best_rms_pre_ch[main_row])
        main_rms_post[i] = float(best_rms_post_ch[main_row])

        improves_global = best_post < best_pre * (1.0 - float(min_improvement_frac))
        improves_main = main_rms_post[i] < main_rms_pre[i]
        if require_main_improvement:
            accepted[i] = bool(improves_global and improves_main)
        else:
            accepted[i] = bool(improves_global)

        trough_idx = int(np.clip(t0_main + best_lag_i, 0, L - 1))
        lo = max(0, trough_idx - 1)
        hi = min(L - 1, trough_idx + 1)
        aligned_main_amp[i] = float(np.min(x[main_row, lo:hi + 1]))

    improvement = pre_score - post_score

    center_idx = int(-window_pp[0])
    canonical_main_times = valid_times + best_lag.astype(np.int64) + int(t0_main - center_idx)

    long_pre, long_post = int(long_window[0]), int(long_window[1])
    T_total = data_matrix.shape[0]
    long_valid = (canonical_main_times + long_pre >= 0) & (canonical_main_times + long_post < T_total)

    accepted_initial_mask = accepted & long_valid
    edge_invalid_long_detect_times = valid_times[accepted & (~long_valid)].astype(np.int64, copy=False)
    rejected_fit_detect_times = valid_times[~accepted].astype(np.int64, copy=False)

    if not np.any(accepted_initial_mask):
        return {"status": "fail", "reason": "no_accepted_fits"}

    acc_idx = np.where(accepted_initial_mask)[0]
    acc_main_times = canonical_main_times[acc_idx]
    acc_detect_times = valid_times[acc_idx]
    acc_best_bin = best_bin[acc_idx]
    acc_best_lag = best_lag[acc_idx]
    acc_pre_score = pre_score[acc_idx]
    acc_post_score = post_score[acc_idx]
    acc_improvement = improvement[acc_idx]
    acc_main_amp = aligned_main_amp[acc_idx]

    if acc_idx.size == 0:
        return {"status": "fail", "reason": "no_accepted_fits"}

    if acc_idx.size == 1:
        keep_local = np.array([0], dtype=np.int64)
        drop_local = np.array([], dtype=np.int64)
    else:
        order = np.argsort(acc_main_times)
        t_sorted = acc_main_times[order]
        groups = []
        start = 0
        for i in range(t_sorted.size - 1):
            if (t_sorted[i + 1] - t_sorted[i]) >= int(postlag_close_dt):
                groups.append(order[start:i + 1])
                start = i + 1
        groups.append(order[start:])

        keep_local_list = []
        drop_local_list = []
        for g in groups:
            if g.size == 1:
                keep_local_list.append(int(g[0]))
                continue
            cand = []
            for j in g:
                cand.append((float(acc_post_score[j]), float(acc_main_amp[j]), int(j)))
            cand.sort(key=lambda z: (z[0], z[1]))
            keep_local_list.append(cand[0][2])
            for _, _, j in cand[1:]:
                drop_local_list.append(int(j))

        keep_local = np.array(sorted(set(keep_local_list)), dtype=np.int64)
        drop_local = np.array(sorted(set(drop_local_list)), dtype=np.int64)

    postlag_dropped_detect_times = acc_detect_times[drop_local].astype(np.int64, copy=False)
    postlag_dropped_main_times = acc_main_times[drop_local].astype(np.int64, copy=False)

    acc_main_times = acc_main_times[keep_local]
    acc_detect_times = acc_detect_times[keep_local]
    acc_best_bin = acc_best_bin[keep_local]
    acc_best_lag = acc_best_lag[keep_local]
    acc_pre_score = acc_pre_score[keep_local]
    acc_post_score = acc_post_score[keep_local]
    acc_improvement = acc_improvement[keep_local]
    acc_main_amp = acc_main_amp[keep_local]

    if acc_main_times.size == 0:
        return {"status": "fail", "reason": "no_accepted_fits_after_postlag_cleanup"}

    order_final = np.argsort(acc_main_times)
    acc_main_times = acc_main_times[order_final]
    acc_detect_times = acc_detect_times[order_final]
    acc_best_bin = acc_best_bin[order_final]
    acc_best_lag = acc_best_lag[order_final]
    acc_pre_score = acc_pre_score[order_final]
    acc_post_score = acc_post_score[order_final]
    acc_improvement = acc_improvement[order_final]
    acc_main_amp = acc_main_amp[order_final]

    unique_bins = np.unique(acc_best_bin)
    long_bank_templates = {}
    long_bank_records = []

    all_ch = np.arange(data_matrix.shape[1], dtype=np.int32)
    for bi in unique_bins:
        idx_bin = np.where(acc_best_bin == bi)[0]
        if idx_bin.size == 0:
            continue

        sort_idx = idx_bin[np.argsort(acc_main_amp[idx_bin])]
        pick_rel = _sample_evenly_sorted_indices(sort_idx.size, min(long_max_per_bin, sort_idx.size))
        chosen_idx = sort_idx[pick_rel]
        chosen_main_times = acc_main_times[chosen_idx].astype(np.int64, copy=False)

        sn_long, valid_long = extract_snippets_fast_ram(
            raw_data=data_matrix,
            spike_times=chosen_main_times,
            window=long_window,
            selected_channels=all_ch,
        )
        sn_long = sn_long.astype(np.float32, copy=False)
        valid_long = np.asarray(valid_long, dtype=np.int64)
        if valid_long.size == 0:
            continue

        if long_reducer == "median":
            tmpl_long = median_ei_adaptive(sn_long).astype(np.float32)
        elif long_reducer == "mean":
            tmpl_long = sn_long.mean(axis=2).astype(np.float32)
        else:
            raise ValueError("long_bank_reducer must be 'median' or 'mean'")

        long_bank_templates[int(bi)] = tmpl_long
        long_bank_records.append({
            "bin_index": int(bi),
            "source_main_times": valid_long.astype(np.int64, copy=False),
            "n_source_spikes": int(valid_long.size),
            "n_assigned_total": int(idx_bin.size),
            "amp_min": float(np.min(acc_main_amp[idx_bin])),
            "amp_max": float(np.max(acc_main_amp[idx_bin])),
        })

    if len(long_bank_templates) == 0:
        return {"status": "fail", "reason": "empty_long_template_bank"}

    long_bank_channel_minima = np.min(
        np.stack([np.min(tmpl, axis=1).astype(np.float32) for tmpl in long_bank_templates.values()], axis=0),
        axis=0,
    ).astype(np.float32)

    subtract_idx = []
    for i in range(acc_main_times.size):
        if int(acc_best_bin[i]) in long_bank_templates:
            subtract_idx.append(i)
    if len(subtract_idx) == 0:
        return {"status": "fail", "reason": "no_subtractable_spikes"}

    subtract_idx = np.asarray(subtract_idx, dtype=np.int64)
    acc_main_times = acc_main_times[subtract_idx]
    acc_detect_times = acc_detect_times[subtract_idx]
    acc_best_bin = acc_best_bin[subtract_idx]
    acc_best_lag = acc_best_lag[subtract_idx]
    acc_pre_score = acc_pre_score[subtract_idx]
    acc_post_score = acc_post_score[subtract_idx]
    acc_improvement = acc_improvement[subtract_idx]
    acc_main_amp = acc_main_amp[subtract_idx]


    # Deferral/exclusion semantics:
    #   - uncertain = boundary-uncertain spikes from scored splits up to the final score node
    #   - extra_left = post-score sibling subtrees inside the chosen unit branch
    #   - rejected_fit = spikes that reached template scoring but did not pass the fit gate
    #
    # close-drop / postlag-close-drop / edge-long remain reported only, not deferred.
    deferred_by_reason = {
        "uncertain": np.asarray(final_left_state["uncertain_times"], dtype=np.int64),
        "extra_left": np.asarray(final_left_state["extra_left_times"], dtype=np.int64),
        "rejected_fit": rejected_fit_detect_times.astype(np.int64, copy=False),
    }

    deferred_all = np.array([], dtype=np.int64)
    for arr in deferred_by_reason.values():
        if arr.size:
            deferred_all = np.concatenate([deferred_all, np.asarray(arr, dtype=np.int64)])
    deferred_all = _sorted_unique_int64(deferred_all)
    defer_frac = float(deferred_all.size) / float(max(1, acc_main_times.size + deferred_all.size))

    unit_record = {
        "detect_ch": int(detect_ch),
        "main_ch": int(main_ch),
        "initial_left_count": int(final_left_state["step1"]["left_count"]),
        "initial_valley_count": int(final_left_state["step1"]["valley_count"]),
        "n_extra_left": int(np.asarray(final_left_state["extra_left_times"], dtype=np.int64).size),
        "trusted_left_detect_amp_mean": float(trusted_left_detect_amp_mean),
        "short_window": tuple(window_pp),
        "long_window": tuple(long_window),
        "t0_main_short": int(t0_main),
        "score_channels": score_channels.astype(np.int32, copy=False),
        "pp_main_leaf_path": str(final_left_state["main_leaf_path"]),
        "pp_main_split_path": final_left_state["main_path_split_path"],
        "pp_main_score": float(final_left_state["main_path_score"]),
        "pp_main_sep": float(final_left_state["main_path_sep"]),
        "pp_main_depth": float(final_left_state["main_path_depth"]),
        "pp_last_split_path": final_left_state["last_path_split_path"],
        "pp_last_score": float(final_left_state["last_path_score"]),
        "pp_last_sep": float(final_left_state["last_path_sep"]),
        "pp_last_depth": float(final_left_state["last_path_depth"]),
        "accepted_detect_times": acc_detect_times.astype(np.int64, copy=False),
        "accepted_main_times": acc_main_times.astype(np.int64, copy=False),
        "accepted_best_bin": acc_best_bin.astype(np.int32, copy=False),
        "accepted_best_lag": acc_best_lag.astype(np.int32, copy=False),
        "accepted_main_amp": acc_main_amp.astype(np.float32, copy=False),
        "accepted_pre_score": acc_pre_score.astype(np.float32, copy=False),
        "accepted_post_score": acc_post_score.astype(np.float32, copy=False),
        "accepted_improvement": acc_improvement.astype(np.float32, copy=False),
        "isi_10_30_main": int(_count_isi_10_30(acc_main_times)),
        "deferred_by_reason": deferred_by_reason,
        "deferred_all_detect_times": deferred_all,
        "rejected_fit_detect_times": rejected_fit_detect_times.astype(np.int64, copy=False),
        "postlag_dropped_detect_times": postlag_dropped_detect_times.astype(np.int64, copy=False),
        "postlag_dropped_main_times": postlag_dropped_main_times.astype(np.int64, copy=False),
        "edge_invalid_long_detect_times": edge_invalid_long_detect_times.astype(np.int64, copy=False),
        "long_bank_channel_minima": long_bank_channel_minima.astype(np.float32, copy=False),
        "long_bank": {
            "window": tuple(long_window),
            "reducer": str(long_reducer),
            "center_index": int(-long_window[0]),
            "bins": long_bank_records,
        },
        "summary": {
            "n_trusted_left_clean": int(trusted_left_times.size),
            "n_accepted": int(acc_main_times.size),
            "n_deferred_unique": int(deferred_all.size),
            "n_rejected_fit": int(rejected_fit_detect_times.size),
            "n_postlag_drop": int(postlag_dropped_detect_times.size),
            "defer_frac": float(defer_frac),
        },
    }

    if dry_run:
        return {
            "status": "ok",
            "unit_record": unit_record,
        }

    for i in range(acc_main_times.size):
        bi = int(acc_best_bin[i])
        tmpl_full = long_bank_templates[bi]
        start = int(acc_main_times[i]) + long_pre
        end = int(acc_main_times[i]) + long_post + 1
        if start < 0 or end > T_total:
            continue
        data_matrix[start:end, :] -= tmpl_full.T.astype(data_matrix.dtype, copy=False)

    return {
        "status": "ok",
        "unit_record": unit_record,
    }


def _ensure_channel_state_entry(channel_state, detect_ch):
    st = channel_state.setdefault(int(detect_ch), {})
    if "exclude_times" not in st:
        st["exclude_times"] = np.array([], dtype=np.int64)
    if "records" not in st:
        st["records"] = []
    st.setdefault("skip_streak", 0)
    st.setdefault("last_skip_reason", None)
    st.setdefault("last_attempt_outcome", None)
    st.setdefault("last_attempt_success_count", -1)
    st.setdefault("retry_valley_low", None)
    st.setdefault("retry_valley_high", None)
    st.setdefault("needs_retry", False)
    st.setdefault("retry_trigger_unit_id", None)
    st.setdefault("retry_trigger_detect_ch", None)
    st.setdefault("retry_trigger_minimum", None)
    return st


def channel_is_retry_eligible(channel_state, detect_ch, success_count, params):
    st = channel_state.get(int(detect_ch), None)
    if st is None:
        return True

    st = _ensure_channel_state_entry(channel_state, detect_ch)
    if st.get("last_attempt_outcome", None) != "fail":
        return True

    if bool(st.get("needs_retry", False)):
        return True

    gap = int(success_count) - int(st.get("last_attempt_success_count", -10**9))
    return gap >= int(params["retry_force_success_gap"])


def note_channel_skip(channel_state, detect_ch, success_count, params, reason=None, step1=None):
    st = _ensure_channel_state_entry(channel_state, detect_ch)

    st["skip_streak"] = int(st.get("skip_streak", 0)) + 1
    st["last_skip_reason"] = None if reason is None else str(reason)
    st["last_attempt_outcome"] = "fail"
    st["last_attempt_success_count"] = int(success_count)
    st["needs_retry"] = False
    st["retry_trigger_unit_id"] = None
    st["retry_trigger_detect_ch"] = None
    st["retry_trigger_minimum"] = None

    if step1 is not None:
        valley_low = step1.get("valley_low", None)
        valley_high = step1.get("valley_high", None)
        st["retry_valley_low"] = None if valley_low is None else float(valley_low)
        st["retry_valley_high"] = None if valley_high is None else float(valley_high)

    return st


def reset_channel_skip_cooldown(channel_state, detect_ch):
    # kept as a compatibility helper; semantics now mean "mark channel successful / awake"
    st = _ensure_channel_state_entry(channel_state, detect_ch)
    st["skip_streak"] = 0
    st["last_skip_reason"] = None
    st["last_attempt_outcome"] = "success"
    st["needs_retry"] = False
    st["retry_trigger_unit_id"] = None
    st["retry_trigger_detect_ch"] = None
    st["retry_trigger_minimum"] = None
    return st


def update_channel_state_after_success(channel_state, unit_record):
    detect_ch = int(unit_record["detect_ch"])
    unit_id = int(unit_record["unit_id"])

    st = _ensure_channel_state_entry(channel_state, detect_ch)
    st["skip_streak"] = 0
    st["last_skip_reason"] = None
    st["last_attempt_outcome"] = "success"
    st["last_attempt_success_count"] = int(unit_id)
    st["needs_retry"] = False
    st["retry_trigger_unit_id"] = None
    st["retry_trigger_detect_ch"] = None
    st["retry_trigger_minimum"] = None

    recs = channel_state[detect_ch]["records"]
    for reason, times in unit_record["deferred_by_reason"].items():
        arr = _ensure_int64_sorted(times)
        if arr.size == 0:
            continue
        recs.append({
            "unit_id": int(unit_record["unit_id"]),
            "reason": str(reason),
            "times": arr,
        })

    exclude_now = channel_state[detect_ch]["exclude_times"]
    exclude_add = np.asarray(unit_record["deferred_all_detect_times"], dtype=np.int64)
    if exclude_add.size:
        channel_state[detect_ch]["exclude_times"] = _sorted_unique_int64(
            np.concatenate([exclude_now, exclude_add])
        )

    unit_channel_minima = np.asarray(unit_record.get("long_bank_channel_minima", []), dtype=np.float32)
    if unit_channel_minima.size == 0:
        return

    for ch in range(unit_channel_minima.size):
        st_ch = _ensure_channel_state_entry(channel_state, ch)
        if st_ch.get("last_attempt_outcome", None) != "fail":
            continue

        valley_high = st_ch.get("retry_valley_high", None)
        if valley_high is None:
            continue

        if float(unit_channel_minima[ch]) <= float(valley_high):
            st_ch["needs_retry"] = True
            st_ch["retry_trigger_unit_id"] = int(unit_id)
            st_ch["retry_trigger_detect_ch"] = int(detect_ch)
            st_ch["retry_trigger_minimum"] = float(unit_channel_minima[ch])


def save_lh_pp_loop_checkpoint(checkpoint_path, dat_path, units_found, channel_state, params, next_unit_id, skyline_state=None):
    payload = {
        "version": 1,
        "dat_path": str(dat_path),
        "units": units_found,
        "channel_state": channel_state,
        "params": params,
        "next_unit_id": int(next_unit_id),
        "skyline_state": skyline_state,
    }
    directory = os.path.dirname(checkpoint_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".lh_pp_loop_", suffix=".tmp", dir=directory if directory else None)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, checkpoint_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_lh_pp_loop_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def replay_lh_pp_checkpoint(raw_mod, checkpoint, verbose=True):
    """
    Rebuild the residual by replaying saved units in order.
    Assumes raw_mod is freshly loaded, baseline-subtracted working data.
    """
    units = checkpoint.get("units", [])
    if len(units) == 0:
        if verbose:
            print("Checkpoint contains no units.")
        return

    all_ch = np.arange(raw_mod.shape[1], dtype=np.int32)
    for unit in units:
        long_window = tuple(unit["long_window"])
        long_pre, long_post = int(long_window[0]), int(long_window[1])

        long_templates = {}
        for rec in unit["long_bank"]["bins"]:
            src_times = np.asarray(rec["source_main_times"], dtype=np.int64)
            if src_times.size == 0:
                continue
            sn_long, valid_long = extract_snippets_fast_ram(
                raw_data=raw_mod,
                spike_times=src_times,
                window=long_window,
                selected_channels=all_ch,
            )
            sn_long = sn_long.astype(np.float32, copy=False)
            if sn_long.shape[2] == 0:
                continue
            reducer = str(unit["long_bank"]["reducer"])
            if reducer == "median":
                tmpl = median_ei_adaptive(sn_long).astype(np.float32)
            elif reducer == "mean":
                tmpl = sn_long.mean(axis=2).astype(np.float32)
            else:
                raise ValueError("Unknown long bank reducer in checkpoint")
            long_templates[int(rec["bin_index"])] = tmpl

        acc_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
        acc_best_bin = np.asarray(unit["accepted_best_bin"], dtype=np.int32)

        for t_main, bi in zip(acc_main_times.tolist(), acc_best_bin.tolist()):
            if int(bi) not in long_templates:
                continue
            tmpl = long_templates[int(bi)]
            start = int(t_main) + long_pre
            end = int(t_main) + long_post + 1
            if start < 0 or end > raw_mod.shape[0]:
                continue
            raw_mod[start:end, :] -= tmpl.T.astype(raw_mod.dtype, copy=False)

        if verbose:
            print(
                "Replayed unit {uid:03d} ch={ch} main={main} n={n}".format(
                    uid=int(unit["unit_id"]),
                    ch=int(unit["detect_ch"]),
                    main=int(unit["main_ch"]),
                    n=int(acc_main_times.size),
                )
            )


def attempt_one_channel_pp_unit(raw_mod, ei_positions, detect_ch, channel_state, params):
    """
    Run the full one-channel PP pipeline with a dry-run preflight before subtraction.

    No channel_state updates happen here.
    raw_mod is mutated only after the candidate passes preflight and the ISI gate.
    """
    pool = build_pp_pool_for_channel(raw_mod, detect_ch, channel_state, params)
    if pool["status"] != "ok":
        return {
            "status": "fail",
            "reason": pool["reason"],
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
        }

    pp_tree_state = run_recursive_pp_for_channel(raw_mod, ei_positions, pool, params)
    if pp_tree_state["tree"]["type"] != "split":
        return {
            "status": "fail",
            "reason": pp_tree_state["tree"]["reason"],
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
        }

    try:
        final_left_state, lh_template_bank = finalize_labels_and_short_bank(raw_mod, pp_tree_state, pool, params)
    except Exception as exc:
        return {
            "status": "fail",
            "reason": "finalize_error: {}".format(str(exc)),
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
        }

    preview_res = assign_and_subtract_unit(
        raw_mod,
        final_left_state,
        lh_template_bank,
        detect_ch,
        params,
        dry_run=True,
    )
    if preview_res["status"] != "ok":
        return {
            "status": "fail",
            "reason": preview_res["reason"],
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
        }

    preview_unit = preview_res["unit_record"]
    max_isi_10_30_main = int(params["max_isi_10_30_main"])
    isi_10_30_main = int(preview_unit.get("isi_10_30_main", 0))
    if isi_10_30_main > max_isi_10_30_main:
        return {
            "status": "fail",
            "reason": "isi_violation_skip(isi10_30={}>{})".format(
                int(isi_10_30_main),
                int(max_isi_10_30_main),
            ),
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
            "unit_record": preview_unit,
        }

    preflight = evaluate_pp_preflight(preview_unit, pool, params)
    if not preflight["ok"]:
        return {
            "status": "fail",
            "reason": preflight["reason"],
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
            "preflight": preflight,
            "unit_record": preview_unit,
        }

    assign_res = assign_and_subtract_unit(
        raw_mod,
        final_left_state,
        lh_template_bank,
        detect_ch,
        params,
        dry_run=False,
    )
    if assign_res["status"] != "ok":
        return {
            "status": "fail",
            "reason": assign_res["reason"],
            "detect_ch": int(detect_ch),
            "step1": pool.get("step1"),
            "preflight": preflight,
        }

    unit_record = assign_res["unit_record"]
    unit_record["preflight"] = preflight
    return {
        "status": "ok",
        "detect_ch": int(detect_ch),
        "unit_record": unit_record,
        "preflight": preflight,
    }


def format_channel_attempt_line(result):
    def _fmt_int(x):
        if x is None:
            return None
        try:
            if np.isfinite(x):
                return str(int(x))
        except Exception:
            pass
        return None

    def _fmt_amp(x):
        if x is None:
            return None
        try:
            if np.isfinite(x):
                return str(int(np.rint(x)))
        except Exception:
            pass
        return None

    def _fmt_float(x, nd=2):
        if x is None:
            return None
        try:
            if np.isfinite(x):
                return f"{float(x):.{nd}f}"
        except Exception:
            pass
        return None

    if result["status"] != "ok":
        ch = int(result["detect_ch"])
        reason = str(result["reason"])
        parts = [f"ch={ch:3d} SKIP {reason}"]

        # Stage 1 / valley info
        step1 = result.get("step1", None)
        if step1 is not None:
            left_txt = _fmt_int(step1.get("left_count", None))
            valley_txt = _fmt_int(step1.get("valley_count", None))
            vfrac_txt = _fmt_float(step1.get("valley_frac_of_left", None), nd=2)

            if left_txt is not None:
                parts.append(f"left={left_txt}")
            if valley_txt is not None:
                parts.append(f"valley={valley_txt}")
            if vfrac_txt is not None:
                parts.append(f"vfrac={vfrac_txt}")

        # Preflight info, if we got that far
        pf = result.get("preflight", None)
        if pf is not None:
            pp_txt = _fmt_float(pf.get("pp_score", None), nd=2)
            dfrac_txt = _fmt_float(pf.get("deferral_frac", None), nd=2)

            if pp_txt is not None:
                parts.append(f"pp={pp_txt}")
            if dfrac_txt is not None:
                parts.append(f"dfrac={dfrac_txt}")

        # Finalization preview info, if available on a failed result
        unit = result.get("unit_record", None)
        if unit is not None:
            xleft_txt = _fmt_int(unit.get("n_extra_left", None))
            amp_txt = _fmt_amp(unit.get("trusted_left_detect_amp_mean", None))

            if xleft_txt is not None:
                parts.append(f"xleft={xleft_txt}")
            if amp_txt is not None:
                parts.append(f"amp={amp_txt}")
                
        skip_streak_txt = _fmt_int(result.get("skip_streak", None))
        cooldown_txt = _fmt_int(result.get("cooldown_until_success_count", None))
        success_count_txt = _fmt_int(result.get("success_count", None))

        if skip_streak_txt is not None:
            parts.append(f"skips={skip_streak_txt}")

        if cooldown_txt is not None and success_count_txt is not None:
            try:
                if int(success_count_txt) < int(cooldown_txt):
                    parts.append(f"cooldown_until={cooldown_txt}")
            except Exception:
                pass

        return " ".join(parts)

    unit = result["unit_record"]

    amp_txt = _fmt_amp(unit.get("trusted_left_detect_amp_mean", np.nan))
    if amp_txt is None:
        amp_txt = "nan"

    msg = (
        "U{uid:03d} ch={ch:3d} main={main:3d} OK "
        "n={n} defer={defer} pp={pp:.2f} "
        "left={left} valley={valley} xleft={xleft} amp={amp}"
    ).format(
        uid=int(unit["unit_id"]),
        ch=int(unit["detect_ch"]),
        main=int(unit["main_ch"]),
        n=int(unit["summary"]["n_accepted"]),
        defer=int(unit["summary"]["n_deferred_unique"]),
        pp=float(unit["pp_main_score"]),
        left=int(unit["initial_left_count"]),
        valley=int(unit["initial_valley_count"]),
        xleft=int(unit["n_extra_left"]),
        amp=amp_txt,
    )

    isi = int(unit["isi_10_30_main"])
    if isi != 0:
        msg += " isi10_30={}".format(isi)
    return msg


# %% [markdown]
# ### Parameters

# %%
LH_PP_LOOP_PARAMS = {
    # -----------------------------
    # Windows / timebase
    # -----------------------------
    "fs": int(fs),
    "window_pp": (-20, 50),                 # short window used for PP, short bank, and short assignment
    "long_subtraction_window": (-40, 80),   # long window used ONLY for subtraction templates
    "pp_duration_sec": 5 * 60,              # how much data to scan for the local PP pool on each channel

    # -----------------------------
    # Skyline / exclusions
    # -----------------------------
    "skyline_n_samples": 1_000_000,         # skyline uses first N samples of the current residual
    "skyline_top_k": 1000,                  # mean of K deepest local minima per channel
    "exclude_radius_samples": 20,           # same-channel exclusion radius around deferred spikes
    "loop_max_units": None,                 # set an int to stop after a fixed number of successful units

    # -----------------------------
    # Valley location (reuse strict LH locator)
    # -----------------------------
    "valley_bin_width": 10.0,
    "valley_bins": 5,
    "valley_min_valid_count": 300,
    "valley_ratio_base": 3,
    "valley_ratio_step": 100,
    "valley_ratio_floor": 2,
    "valley_ratio_cap": 10,

    # -----------------------------
    # PP seed valley acceptance (looser than LH)
    # -----------------------------
    "pp_valley_min_left_count": 400,
    "pp_valley_max_count": 3000,
    "pp_valley_max_frac_of_left": 0.50,

    # -----------------------------
    # Recursive PP
    # -----------------------------
    "pp_max_depth": 3,
    "pp_min_node_n": 300,
    "pp_min_child_n": 120,
    "pp_discover_max_per_side": 5000,

    # Projection-pursuit search
    "pp_n_search_pcs": 6,
    "pp_n_random_dirs": 2500,
    "pp_hist_bins": 120,
    "pp_smooth_sigma": 2.0,
    "pp_rng_seed": 123,

    # Deterministic augmented-direction bank
    "pp_amp_pc1_weight": 0.50,
    "pp_n_biased_tail_dirs": 16,
    "pp_biased_tail_scale": 0.08,

    # Split acceptance thresholds (full-node classification)
    "pp_min_score": 0.90,
    "pp_min_depth": 0.20,
    "pp_min_sep": 1.15,
    "pp_min_group_frac": 0.03,
    "pp_target_frac": 0.08,

    # Feature construction
    "pp_p2p_thr": 30.0,
    "pp_max_channels": 80,
    "pp_min_channels": 10,

    # Recursion policy
    "pp_left_recurse_thresh": 0.50,

    # Which split is allowed to count for preflight?
    # Only splits with a sizeable minor branch are scoreable for preflight.
    "pp_scoreable_minor_n_cap": 1000,
    "pp_scoreable_minor_frac": 0.20,

    # Detailed tiny-split check (same helper family as the single-channel flow)
    "pp_abcheck_pc_var_thr": 0.10,
    "pp_abcheck_minor_frac_thr": 0.10,
    "pp_abcheck_ei_cos_thr": 0.95,
    "pp_abcheck_asym_unique_ch_min": 3,
    "pp_abcheck_support_rel": 0.10,
    "pp_abcheck_support_abs": 30.0,
    "pp_abcheck_cos_lag": 1,
    "pp_abcheck_max_lag": 3,
    "pp_abcheck_time_keep_rel": 0.10,
    "pp_abcheck_frac_in_thr": 0.20,
    "pp_abcheck_out_in_ratio_thr": 2.0,
    "pp_abcheck_resid_frac_min": 0.08,
    "pp_abcheck_shared_cos_thr": 0.95,
    "pp_abcheck_shared_alpha_thr": 0.95,

    # -----------------------------
    # Finalize labels / short bank
    # -----------------------------
    "pp_margin_k": 0.40,
    "pp_left_leaf_thresh": 0.50,
    "short_bank_reducer": "median",         # "median" or "mean"
    "short_bank_n_bins": 6,
    "short_bank_min_bin_size": 100,
    "trusted_close_dt": 10,                 # detect-time close-neighbor cleanup inside trusted_left

    # -----------------------------
    # Short assignment / fit gate
    # -----------------------------
    "assign_lag_radius": 3,
    "assign_n_score_channels": 20,
    "assign_min_improvement_frac": 0.00,
    "assign_require_main_improvement": True,
    "postlag_close_dt": 10,                 # dedup AFTER lag-correction on the canonical main-channel clock

    # -----------------------------
    # Preflight after dry-run assignment
    # -----------------------------
    "preflight_min_pp_score": 2.0,
    "preflight_max_deferral_frac": 0.40,

    # -----------------------------
    # Hard skip before subtraction
    # -----------------------------
    "max_isi_10_30_main": 60,

    # -----------------------------
    # Long subtraction bank
    # -----------------------------
    "long_bank_reducer": "median",          # "median" or "mean"
    "long_bank_max_spikes_per_bin": 1000,   # cap template-building spikes per accepted bin; sampled evenly in amp rank

    # -----------------------------
    # Retry logic for skipped channels
    # Re-run only if:
    #   (a) a newly found unit plausibly touched that channel's left/valley regime, or
    #   (b) this many successful units have been found since the last failed attempt.
    # -----------------------------
    "retry_force_success_gap": 10,

    # -----------------------------
    # Checkpointing
    # -----------------------------
    "checkpoint_path": make_default_lh_pp_checkpoint_path(dat_path),
}

print("Checkpoint file:")
print(LH_PP_LOOP_PARAMS["checkpoint_path"])

# %% [markdown]
# ### LH projection-pursuit loop

# %%
# %%
# Fresh run:
# units_found = []
# channel_state = {}
# next_unit_id = 1

# -----------------------------
# Resume recipe
# -----------------------------
# ckpt = load_lh_pp_loop_checkpoint(LH_PP_LOOP_PARAMS["checkpoint_path"])
# units_found = ckpt["units"]
# channel_state = ckpt["channel_state"]
# next_unit_id = int(ckpt["next_unit_id"])
# replay_lh_pp_checkpoint(raw_mod, ckpt, verbose=True)

# LH_PP_LOOP_PARAMS["max_isi_10_30_main"] = 60

n_marked = 0
for ch, st in channel_state.items():
    if st.get("last_attempt_outcome", None) == "fail":
        st["needs_retry"] = True
        n_marked += 1

print("marked for retry:", n_marked)


skyline_state = skyline_scan_with_exclusions(raw_mod, channel_state, LH_PP_LOOP_PARAMS)
failed_this_pass = set()

while True:
    max_units = LH_PP_LOOP_PARAMS["loop_max_units"]
    if (max_units is not None) and (len(units_found) >= int(max_units)):
        print("Reached loop_max_units = {}".format(int(max_units)))
        break

    channel_order = np.asarray(skyline_state["channel_order"], dtype=np.int64)
    skyline_scores = np.asarray(skyline_state["mean_amp_topk"], dtype=np.float32)

    success_count = int(next_unit_id) - 1

    candidate_channels = []
    n_retry_sleeping = 0

    for ch in channel_order.tolist():
        ch = int(ch)

        if not np.isfinite(skyline_scores[ch]):
            continue
        if ch in failed_this_pass:
            continue
        if not channel_is_retry_eligible(channel_state, ch, success_count, LH_PP_LOOP_PARAMS):
            n_retry_sleeping += 1
            continue

        candidate_channels.append(ch)

    if len(candidate_channels) == 0:
        if n_retry_sleeping > 0:
            print(
                "No remaining eligible channels in the current skyline pass. "
                f"{n_retry_sleeping} channel(s) are waiting for either an affected-unit trigger "
                f"or {int(LH_PP_LOOP_PARAMS['retry_force_success_gap'])} successful units since last attempt."
            )
        else:
            print("No remaining channels in the current skyline pass. Stopping.")
        break

    detect_ch = int(candidate_channels[0])
    print(f"Current channel: {int(detect_ch)}")

    result = attempt_one_channel_pp_unit(
        raw_mod=raw_mod,
        ei_positions=ei_positions,
        detect_ch=detect_ch,
        channel_state=channel_state,
        params=LH_PP_LOOP_PARAMS,
    )

    if result["status"] == "ok":
        unit_record = result["unit_record"]
        unit_record["unit_id"] = int(next_unit_id)
        units_found.append(unit_record)

        update_channel_state_after_success(channel_state, unit_record)

        ok_result = {
            "status": "ok",
            "detect_ch": detect_ch,
            "unit_record": unit_record,
        }
        print(format_channel_attempt_line(ok_result))

        next_unit_id += 1
        skyline_state = skyline_scan_with_exclusions(raw_mod, channel_state, LH_PP_LOOP_PARAMS)
        failed_this_pass = set()

        save_lh_pp_loop_checkpoint(
            checkpoint_path=LH_PP_LOOP_PARAMS["checkpoint_path"],
            dat_path=dat_path,
            units_found=units_found,
            channel_state=channel_state,
            params=LH_PP_LOOP_PARAMS,
            next_unit_id=next_unit_id,
            skyline_state=skyline_state,
        )

    else:
        success_count = int(next_unit_id) - 1
        st = note_channel_skip(
            channel_state=channel_state,
            detect_ch=detect_ch,
            success_count=success_count,
            params=LH_PP_LOOP_PARAMS,
            reason=result.get("reason", None),
            step1=result.get("step1", None),
        )

        result["skip_streak"] = int(st.get("skip_streak", 0))
        result["success_count"] = int(success_count)

        print(format_channel_attempt_line(result))
        failed_this_pass.add(int(detect_ch))

# %%
print("units_found in globals():", "units_found" in globals())

if "units_found" in globals():
    print("type(units_found):", type(units_found))
    print("n_units:", len(units_found))

# %%
print(unit_ei_summary.loc[unit_ei_summary["saved_main_ch"] == 93, "unit_id"].tolist())

# %%
UNIT_ID_TO_TEST = 272

# %%
import numpy as np
import matplotlib.pyplot as plt

from axolotl_utils_ram import extract_snippets_fast_ram
from plot_ei_waveforms import plot_ei_waveforms

# =========================
# config
# =========================
# UNIT_ID_TO_TEST = 261
MAX_SPIKES = 1000
WINDOW = (-40, 80)
RNG_SEED = 0

# =========================
# helper: get unit record
# =========================
def get_unit_record(units_found, unit_id):
    if isinstance(units_found, dict):
        if unit_id in units_found:
            return units_found[unit_id]
        for k in [f"U{int(unit_id):03d}", str(unit_id), f"unit_{int(unit_id)}"]:
            if k in units_found:
                return units_found[k]
        for rec in units_found.values():
            if isinstance(rec, dict) and rec.get("unit_id", None) == unit_id:
                return rec
        raise KeyError(f"Could not find unit {unit_id} in units_found dict.")

    if isinstance(units_found, (list, tuple)):
        for rec in units_found:
            if isinstance(rec, dict) and rec.get("unit_id", None) == unit_id:
                return rec
        if isinstance(unit_id, int) and 1 <= unit_id <= len(units_found):
            return units_found[unit_id - 1]
        raise KeyError(f"Could not find unit {unit_id} in units_found list.")

    raise TypeError("units_found must be dict, list, or tuple.")

# =========================
# helpers for this unit format
# =========================
def get_spike_times_from_unit(rec):
    # prefer main times for EI extraction / same-main comparisons
    for k in ["accepted_main_times", "accepted_detect_times"]:
        if k in rec:
            x = np.asarray(rec[k]).astype(np.int64).ravel()
            if x.size > 0:
                return x, k
    raise KeyError(f"Could not find accepted spike times. Keys are: {list(rec.keys())}")

def get_detect_channel(rec):
    return int(rec["detect_ch"]) if "detect_ch" in rec else None

def get_saved_main_channel(rec):
    return int(rec["main_ch"]) if "main_ch" in rec else None

def get_defer_count(rec):
    if "deferred_all_detect_times" in rec:
        return len(np.asarray(rec["deferred_all_detect_times"]).ravel())
    return None

def count_units_same_main(units_found, target_main):
    if isinstance(units_found, dict):
        iterator = units_found.values()
    else:
        iterator = units_found

    n = 0
    for rec in iterator:
        if isinstance(rec, dict) and rec.get("main_ch", None) == target_main:
            n += 1
    return n

def infer_main_channel_from_ei(ei):
    p2p = ei.max(axis=1) - ei.min(axis=1)
    return int(np.argmax(p2p))

# =========================
# pick raw array
# =========================
if "raw_orig" in globals():
    raw_for_ei = raw_orig
    raw_name = "raw_orig"
elif "raw_mod" in globals():
    raw_for_ei = raw_mod
    raw_name = "raw_mod"
else:
    raise RuntimeError("Neither raw_orig nor raw_mod is present.")

# =========================
# get record and spikes
# =========================
rec = get_unit_record(units_found, UNIT_ID_TO_TEST)
spike_times, spike_field = get_spike_times_from_unit(rec)

detect_ch = get_detect_channel(rec)
saved_main_ch = get_saved_main_channel(rec)
n_defer = get_defer_count(rec)

# subsample up to MAX_SPIKES
rng = np.random.default_rng(RNG_SEED)
if spike_times.size > MAX_SPIKES:
    pick = np.sort(rng.choice(spike_times.size, size=MAX_SPIKES, replace=False))
    spike_times_use = np.sort(spike_times[pick])
else:
    spike_times_use = np.sort(spike_times)

# extract all channels
selected_channels = np.arange(raw_for_ei.shape[1], dtype=np.int32)
snips, valid_times = extract_snippets_fast_ram(
    raw_data=raw_for_ei,
    spike_times=spike_times_use,
    window=WINDOW,
    selected_channels=selected_channels,
)

if snips.shape[2] == 0:
    raise RuntimeError("No valid snippets extracted.")

# mean EI
ei = snips.mean(axis=2).astype(np.float32)
main_ch_from_ei = infer_main_channel_from_ei(ei)

# for same-main count, use saved main_ch from units_found
main_for_count = saved_main_ch if saved_main_ch is not None else main_ch_from_ei
n_same_main = count_units_same_main(units_found, main_for_count)

# title
n_total_accepted = len(spike_times)

title_parts = [f"U{UNIT_ID_TO_TEST:03d}", f"main={main_ch_from_ei}"]
if detect_ch is not None and detect_ch != main_ch_from_ei:
    title_parts.append(f"detect={detect_ch}")
title_parts.append(f"N={n_total_accepted}")
if n_defer is not None:
    title_parts.append(f"defer={n_defer}")
title_parts.append(f"same-main={n_same_main}")

title = " | ".join(title_parts)

# plot
fig, ax = plt.subplots(figsize=(20, 12))
plot_ei_waveforms(
    ei,
    ei_positions,
    ref_channel=main_ch_from_ei,
    scale=70.0,
    box_height=1.0,
    box_width=50.0,
    ax=ax,
)
ax.set_title(title)
plt.tight_layout()
plt.show()

print("unit_id:", rec.get("unit_id"))
print("spike field used:", spike_field)
print("raw source:", raw_name)
print("total accepted spikes:", len(spike_times))
print("spikes used for EI:", len(valid_times))
print("saved main_ch:", saved_main_ch)
print("detect_ch:", detect_ch)
print("main_ch from EI:", main_ch_from_ei)
print("deferred count:", n_defer)
print("units with same saved main_ch:", n_same_main)

# %%
import numpy as np
import pandas as pd

from axolotl_utils_ram import extract_snippets_fast_ram

# =========================
# config
# =========================
EI_WINDOW = (-40, 80)
EI_MAX_SPIKES = 1000
EI_RNG_SEED = 0

# =========================
# helpers
# =========================
def iter_unit_records(units_found):
    """
    Yield unit records as dicts, regardless of whether units_found is a list or dict.
    """
    if isinstance(units_found, dict):
        for rec in units_found.values():
            yield rec
    elif isinstance(units_found, (list, tuple)):
        for rec in units_found:
            yield rec
    else:
        raise TypeError("units_found must be dict, list, or tuple.")

def get_spike_times_from_unit(rec):
    """
    Prefer accepted_main_times for EI extraction.
    """
    for k in ["accepted_main_times", "accepted_detect_times"]:
        if k in rec:
            x = np.asarray(rec[k]).astype(np.int64).ravel()
            if x.size > 0:
                return x, k
    raise KeyError(f"Could not find accepted spike times. Keys are: {list(rec.keys())}")

def get_detect_channel(rec):
    return int(rec["detect_ch"]) if "detect_ch" in rec else None

def get_saved_main_channel(rec):
    return int(rec["main_ch"]) if "main_ch" in rec else None

def get_defer_count(rec):
    if "deferred_all_detect_times" in rec:
        return len(np.asarray(rec["deferred_all_detect_times"]).ravel())
    return 0

def infer_main_channel_from_ei(ei):
    p2p = ei.max(axis=1) - ei.min(axis=1)
    return int(np.argmax(p2p))

# =========================
# choose raw array
# =========================
if "raw_orig" not in globals():
    raise RuntimeError("raw_orig is not available.")

raw_for_ei = raw_orig
selected_channels = np.arange(raw_for_ei.shape[1], dtype=np.int32)

# =========================
# build EI cache
# =========================
rng = np.random.default_rng(EI_RNG_SEED)

unit_ei_cache = {}
summary_rows = []

n_total = 0
n_ok = 0
n_fail = 0

for rec in iter_unit_records(units_found):
    n_total += 1

    try:
        unit_id = int(rec["unit_id"])
        detect_ch = get_detect_channel(rec)
        saved_main_ch = get_saved_main_channel(rec)
        defer_count = get_defer_count(rec)

        spike_times_all, spike_field = get_spike_times_from_unit(rec)
        n_total_accepted = len(spike_times_all)

        if n_total_accepted > EI_MAX_SPIKES:
            pick = np.sort(rng.choice(n_total_accepted, size=EI_MAX_SPIKES, replace=False))
            spike_times_use = np.sort(spike_times_all[pick])
        else:
            spike_times_use = np.sort(spike_times_all)

        snips, valid_times = extract_snippets_fast_ram(
            raw_data=raw_for_ei,
            spike_times=spike_times_use,
            window=EI_WINDOW,
            selected_channels=selected_channels,
        )

        if snips.shape[2] == 0:
            raise RuntimeError("No valid snippets extracted.")

        ei = snips.mean(axis=2).astype(np.float32)
        main_ch_from_ei = infer_main_channel_from_ei(ei)

        unit_ei_cache[unit_id] = {
            "unit_id": unit_id,
            "ei": ei,
            "detect_ch": detect_ch,
            "saved_main_ch": saved_main_ch,
            "main_ch_from_ei": main_ch_from_ei,
            "n_total_accepted": n_total_accepted,
            "n_deferred": defer_count,
            "spike_field": spike_field,
            "sampled_spike_times": valid_times.astype(np.int64),
            "all_spike_times": spike_times_all.astype(np.int64),
        }

        summary_rows.append({
            "unit_id": unit_id,
            "detect_ch": detect_ch,
            "saved_main_ch": saved_main_ch,
            "main_ch_from_ei": main_ch_from_ei,
            "n_total_accepted": n_total_accepted,
            "n_used_for_ei": int(len(valid_times)),
            "n_deferred": defer_count,
            "spike_field": spike_field,
            "status": "ok",
        })
        n_ok += 1

    except Exception as e:
        unit_id = rec.get("unit_id", None)
        summary_rows.append({
            "unit_id": unit_id,
            "detect_ch": rec.get("detect_ch", None),
            "saved_main_ch": rec.get("main_ch", None),
            "main_ch_from_ei": None,
            "n_total_accepted": None,
            "n_used_for_ei": None,
            "n_deferred": None,
            "spike_field": None,
            "status": f"FAIL: {type(e).__name__}: {e}",
        })
        n_fail += 1

# =========================
# summary table
# =========================
unit_ei_summary = pd.DataFrame(summary_rows).sort_values("unit_id").reset_index(drop=True)

print(f"Built EI cache for {n_ok} / {n_total} units. Failed: {n_fail}")
print(f"unit_ei_cache keys = unit_id, total cached = {len(unit_ei_cache)}")

display(unit_ei_summary.head(10))

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from itertools import combinations

from plot_ei_waveforms import plot_ei_waveforms

# =========================================================
# Config
# =========================================================
SIG_MODE = "relative"   # "relative" or "absolute"
SIG_REL = 0.10          # significant if p2p >= SIG_REL * max_p2p of that EI
SIG_ABS = 30.0          # used only if SIG_MODE == "absolute"
MAX_ALIGN_LAG = 3       # search small integer lag for best cosine
USE_ABS_WAVEFORM = False  # False = raw waveform cosine, True = abs(waveform) cosine


# =========================================================
# Helpers
# =========================================================
def roll_zero_2d(arr, shift):
    """
    Zero-padded shift along time axis for [C, T].
    Positive shift moves waveform to the right.
    """
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :-shift]
    else:
        s = -shift
        out[:, :-s] = arr[:, s:]
    return out

def get_sig_channels(ei, mode="relative", rel=0.10, abs_thr=30.0):
    p2p = ei.max(axis=1) - ei.min(axis=1)
    if mode == "relative":
        thr = rel * np.max(p2p)
    elif mode == "absolute":
        thr = abs_thr
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")
    return np.flatnonzero(p2p >= thr)

def cosine_on_union_sig(ei1, ei2, mode="relative", rel=0.10, abs_thr=30.0, max_lag=3, use_abs=False):
    """
    Compute cosine on the union of significant channels, with small lag search.
    Returns:
        best_cos, best_lag, union_channels
    """
    sig1 = get_sig_channels(ei1, mode=mode, rel=rel, abs_thr=abs_thr)
    sig2 = get_sig_channels(ei2, mode=mode, rel=rel, abs_thr=abs_thr)
    union = np.union1d(sig1, sig2)

    if union.size == 0:
        return np.nan, 0, union

    A = ei1[union].astype(np.float32)
    B = ei2[union].astype(np.float32)

    if use_abs:
        A = np.abs(A)
        B = np.abs(B)

    best_cos = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        B_shift = roll_zero_2d(B, lag)
        a = A.ravel()
        b = B_shift.ravel()

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue

        c = float(np.dot(a, b) / (na * nb))
        if c > best_cos:
            best_cos = c
            best_lag = lag

    return best_cos, best_lag, union

def build_same_main_pairs(unit_ei_summary):
    """
    Use recorded main channel: saved_main_ch
    Returns a list of dicts with fields:
      saved_main_ch, unit_id_1, unit_id_2
    """
    pairs = []
    df = unit_ei_summary.copy()

    df = df[df["status"] == "ok"].copy()
    df = df[df["saved_main_ch"].notna()].copy()

    for main_ch, sub in df.groupby("saved_main_ch"):
        unit_ids = sorted(sub["unit_id"].astype(int).tolist())
        if len(unit_ids) < 2:
            continue
        for u1, u2 in combinations(unit_ids, 2):
            pairs.append({
                "saved_main_ch": int(main_ch),
                "unit_id_1": int(u1),
                "unit_id_2": int(u2),
            })

    return pairs


# =========================================================
# Build pair list
# =========================================================
same_main_pairs = build_same_main_pairs(unit_ei_summary)

print(f"Found {len(same_main_pairs)} pairwise comparisons among units sharing the same recorded main channel.")

if len(same_main_pairs) == 0:
    raise RuntimeError("No same-recorded-main-channel pairs found.")


# =========================================================
# Browser state
# =========================================================
state = {"idx": 0}

out = widgets.Output(layout={"border": "1px solid #ccc"})
btn_prev = widgets.Button(description="Prev", button_style="")
btn_next = widgets.Button(description="Next", button_style="")
pair_dropdown = widgets.Dropdown(
    options=[(f"{i:03d}: main {p['saved_main_ch']} | U{p['unit_id_1']:03d} vs U{p['unit_id_2']:03d}", i)
             for i, p in enumerate(same_main_pairs)],
    value=0,
    description="Pair:",
    layout=widgets.Layout(width="700px"),
)
info_html = widgets.HTML()


def render_pair(idx):
    idx = int(np.clip(idx, 0, len(same_main_pairs) - 1))
    state["idx"] = idx
    pair_dropdown.value = idx

    pair = same_main_pairs[idx]
    u1 = pair["unit_id_1"]
    u2 = pair["unit_id_2"]
    main_ch = pair["saved_main_ch"]

    rec1 = unit_ei_cache[u1]
    rec2 = unit_ei_cache[u2]

    ei1 = rec1["ei"]
    ei2 = rec2["ei"]

    cos_val, best_lag, union_ch = cosine_on_union_sig(
        ei1, ei2,
        mode=SIG_MODE,
        rel=SIG_REL,
        abs_thr=SIG_ABS,
        max_lag=MAX_ALIGN_LAG,
        use_abs=USE_ABS_WAVEFORM,
    )

    ei2_shift = roll_zero_2d(ei2, best_lag)

    title = (
        f"Recorded main={main_ch} | "
        f"U{u1:03d} (detect={rec1['detect_ch']}, N={rec1['n_total_accepted']}, defer={rec1['n_deferred']})  vs  "
        f"U{u2:03d} (detect={rec2['detect_ch']}, N={rec2['n_total_accepted']}, defer={rec2['n_deferred']})\n"
        f"cos_union_sig={cos_val:.4f} | best_lag={best_lag:+d} | union_sig_ch={len(union_ch)} | pair {idx+1}/{len(same_main_pairs)}"
    )

    info_html.value = (
        f"<b>Pair {idx+1}/{len(same_main_pairs)}</b> &nbsp;&nbsp; "
        f"recorded main = <b>{main_ch}</b> &nbsp;&nbsp; "
        f"U{u1:03d} vs U{u2:03d} &nbsp;&nbsp; "
        f"cos = <b>{cos_val:.4f}</b> &nbsp;&nbsp; "
        f"lag = <b>{best_lag:+d}</b> &nbsp;&nbsp; "
        f"union sig chans = <b>{len(union_ch)}</b>"
    )

    with out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(16, 10))
        plot_ei_waveforms(
            [ei1, ei2_shift],
            ei_positions,
            ref_channel=main_ch,
            scale=70.0,
            box_height=1.0,
            box_width=50.0,
            ax=ax,
            colors=["black", "red"],
            alpha=[0.9, 0.9],
            linewidth=[0.6, 0.8],
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


def on_prev(_):
    render_pair(state["idx"] - 1)

def on_next(_):
    render_pair(state["idx"] + 1)

def on_dropdown_change(change):
    if change["name"] == "value" and change["new"] is not None:
        render_pair(change["new"])

btn_prev.on_click(on_prev)
btn_next.on_click(on_next)
pair_dropdown.observe(on_dropdown_change)

controls = widgets.HBox([btn_prev, btn_next])
ui = widgets.VBox([controls, pair_dropdown, info_html, out])

display(ui)
render_pair(0)

# %%
import numpy as np

from axolotl_utils_ram import extract_snippets_fast_ram
from joint_utils import classify_two_cells_vs_ab_shard

# =========================
# config
# =========================
U_A = 47
U_B = 219
WINDOW = (-40, 80)
MAX_SPIKES_PER_UNIT = 1000
RNG_SEED = 0

if "raw_orig" not in globals():
    raise RuntimeError("raw_orig is required and not found.")

if "unit_ei_cache" not in globals():
    raise RuntimeError("unit_ei_cache not found. Build the EI cache first.")

if U_A not in unit_ei_cache or U_B not in unit_ei_cache:
    raise KeyError("One of the requested units is missing from unit_ei_cache.")

rng = np.random.default_rng(RNG_SEED)

recA = unit_ei_cache[U_A]
recB = unit_ei_cache[U_B]

# cached EIs
EI_A = recA["ei"]
EI_B = recB["ei"]

# all accepted spike times
tA_all = np.asarray(recA["all_spike_times"], dtype=np.int64)
tB_all = np.asarray(recB["all_spike_times"], dtype=np.int64)

# subsample up to 1k per unit
if tA_all.size > MAX_SPIKES_PER_UNIT:
    tA = np.sort(tA_all[rng.choice(tA_all.size, size=MAX_SPIKES_PER_UNIT, replace=False)])
else:
    tA = np.sort(tA_all)

if tB_all.size > MAX_SPIKES_PER_UNIT:
    tB = np.sort(tB_all[rng.choice(tB_all.size, size=MAX_SPIKES_PER_UNIT, replace=False)])
else:
    tB = np.sort(tB_all)

# combine and remember which spikes came from which unit
times_all = np.concatenate([tA, tB])
labels = np.concatenate([
    np.zeros(len(tA), dtype=np.int32),
    np.ones(len(tB), dtype=np.int32),
])

# sort by time so extraction is neat
order = np.argsort(times_all)
times_all = times_all[order]
labels = labels[order]

# extract snippets on all channels
selected_channels = np.arange(raw_orig.shape[1], dtype=np.int32)
snips, valid_times = extract_snippets_fast_ram(
    raw_data=raw_orig,
    spike_times=times_all,
    window=WINDOW,
    selected_channels=selected_channels,
)

# valid_times may drop edge-near spikes, so remap labels to valid_times
time_to_label_queue = {}
for t, lab in zip(times_all.tolist(), labels.tolist()):
    time_to_label_queue.setdefault(int(t), []).append(int(lab))

valid_labels = []
for t in valid_times.tolist():
    valid_labels.append(time_to_label_queue[int(t)].pop(0))
valid_labels = np.asarray(valid_labels, dtype=np.int32)

idxA = np.flatnonzero(valid_labels == 0)
idxB = np.flatnonzero(valid_labels == 1)

print(f"U{U_A:03d}: total={len(tA_all)}, used={len(tA)}, valid={len(idxA)}, "
      f"main={recA['saved_main_ch']}, detect={recA['detect_ch']}, defer={recA['n_deferred']}")
print(f"U{U_B:03d}: total={len(tB_all)}, used={len(tB)}, valid={len(idxB)}, "
      f"main={recB['saved_main_ch']}, detect={recB['detect_ch']}, defer={recB['n_deferred']}")

metrics = classify_two_cells_vs_ab_shard(
    EI_A, EI_B, snips, idxA, idxB,
    p2p_thr=30.0,
    max_channels=80,
    min_channels=10,
    lag_radius=0,
    weight_by_p2p=True,
    weight_beta=0.7,
    rms_thr_support=10.0,
    asym_strong_z=2.0,
    asym_pure_z=1.0,
)

print("\nVERDICT:", metrics["label"])
print("DETAILS:", metrics["details"])
print("A0_z:", metrics["A0_z"])
print("A1_z:", metrics["A1_z"])
print("Containment:", metrics["containment"])

# %%
import numpy as np
from itertools import combinations

from axolotl_utils_ram import extract_snippets_fast_ram
from joint_utils import classify_two_cells_vs_ab_shard

# =========================================================
# Config
# =========================================================
SIG_MODE = "relative"   # "relative" or "absolute"
SIG_REL = 0.10
SIG_ABS = 30.0

SUS_OVERLAP_FRAC = 0.70
SUS_COS_THR = 0.80

WINDOW = (-40, 80)
MAX_SPIKES_PER_UNIT = 1000
RNG_SEED = 0

# AB-check params
AB_P2P_THR = 30.0
AB_MAX_CHANNELS = 80
AB_MIN_CHANNELS = 10
AB_LAG_RADIUS = 0
AB_WEIGHT_BY_P2P = True
AB_WEIGHT_BETA = 0.7
AB_RMS_THR_SUPPORT = 10.0
AB_ASYM_STRONG_Z = 2.0
AB_ASYM_PURE_Z = 1.0

# =========================================================
# Sanity checks
# =========================================================
if "unit_ei_cache" not in globals():
    raise RuntimeError("unit_ei_cache not found.")
if "raw_orig" not in globals():
    raise RuntimeError("raw_orig not found.")

rng = np.random.default_rng(RNG_SEED)

# =========================================================
# Helpers
# =========================================================
def roll_zero_2d(arr, shift):
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :-shift]
    else:
        s = -shift
        out[:, :-s] = arr[:, s:]
    return out

def get_sig_channels(ei, mode="relative", rel=0.10, abs_thr=30.0):
    p2p = ei.max(axis=1) - ei.min(axis=1)
    if mode == "relative":
        thr = rel * np.max(p2p)
    elif mode == "absolute":
        thr = abs_thr
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")
    return np.flatnonzero(p2p >= thr)

def cosine_on_union_sig(ei1, ei2, mode="relative", rel=0.10, abs_thr=30.0, max_lag=3):
    sig1 = get_sig_channels(ei1, mode=mode, rel=rel, abs_thr=abs_thr)
    sig2 = get_sig_channels(ei2, mode=mode, rel=rel, abs_thr=abs_thr)
    union = np.union1d(sig1, sig2)
    if union.size == 0:
        return np.nan, 0, sig1, sig2

    A = ei1[union].astype(np.float32)
    B = ei2[union].astype(np.float32)

    best_cos = -np.inf
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        B_shift = roll_zero_2d(B, lag)
        a = A.ravel()
        b = B_shift.ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        c = float(np.dot(a, b) / (na * nb))
        if c > best_cos:
            best_cos = c
            best_lag = lag

    return best_cos, best_lag, sig1, sig2

def sig_overlap_frac(sig1, sig2):
    sig1 = np.asarray(sig1, dtype=int)
    sig2 = np.asarray(sig2, dtype=int)
    if len(sig1) == 0 or len(sig2) == 0:
        return 0.0
    inter = np.intersect1d(sig1, sig2).size
    return inter / min(len(sig1), len(sig2))

def subsample_times(times, max_n, rng):
    times = np.asarray(times, dtype=np.int64)
    if len(times) > max_n:
        pick = np.sort(rng.choice(len(times), size=max_n, replace=False))
        return np.sort(times[pick])
    return np.sort(times)

def build_joint_snips_for_pair(recA, recB, window, max_spikes_per_unit, rng):
    tA_all = np.asarray(recA["all_spike_times"], dtype=np.int64)
    tB_all = np.asarray(recB["all_spike_times"], dtype=np.int64)

    tA = subsample_times(tA_all, max_spikes_per_unit, rng)
    tB = subsample_times(tB_all, max_spikes_per_unit, rng)

    times_all = np.concatenate([tA, tB])
    labels = np.concatenate([
        np.zeros(len(tA), dtype=np.int32),
        np.ones(len(tB), dtype=np.int32),
    ])

    order = np.argsort(times_all)
    times_all = times_all[order]
    labels = labels[order]

    selected_channels = np.arange(raw_orig.shape[1], dtype=np.int32)
    snips, valid_times = extract_snippets_fast_ram(
        raw_data=raw_orig,
        spike_times=times_all,
        window=window,
        selected_channels=selected_channels,
    )

    # remap labels after edge-drop
    time_to_label_queue = {}
    for t, lab in zip(times_all.tolist(), labels.tolist()):
        time_to_label_queue.setdefault(int(t), []).append(int(lab))

    valid_labels = []
    for t in valid_times.tolist():
        valid_labels.append(time_to_label_queue[int(t)].pop(0))
    valid_labels = np.asarray(valid_labels, dtype=np.int32)

    idxA = np.flatnonzero(valid_labels == 0)
    idxB = np.flatnonzero(valid_labels == 1)

    return snips, idxA, idxB

# =========================================================
# Stage 1: find suspect pairs
# =========================================================
unit_ids = sorted(unit_ei_cache.keys())
sus_pairs = []

for u1, u2 in combinations(unit_ids, 2):
    ei1 = unit_ei_cache[u1]["ei"]
    ei2 = unit_ei_cache[u2]["ei"]

    cos_val, best_lag, sig1, sig2 = cosine_on_union_sig(
        ei1, ei2,
        mode=SIG_MODE,
        rel=SIG_REL,
        abs_thr=SIG_ABS,
        max_lag=3,
    )
    overlap = sig_overlap_frac(sig1, sig2)

    if (overlap >= SUS_OVERLAP_FRAC) or (cos_val > SUS_COS_THR):
        sus_pairs.append({
            "u1": u1,
            "u2": u2,
            "cos": cos_val,
            "lag": best_lag,
            "overlap": overlap,
        })

print(f"Suspect pairs to AB-check: {len(sus_pairs)}")

# =========================================================
# Stage 2: AB-check suspect pairs
# =========================================================
ab_pairs = []
ab_shard_units = set()

for p in sus_pairs:
    u1 = p["u1"]
    u2 = p["u2"]

    rec1 = unit_ei_cache[u1]
    rec2 = unit_ei_cache[u2]

    try:
        snips, idx1, idx2 = build_joint_snips_for_pair(
            rec1, rec2,
            window=WINDOW,
            max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
            rng=rng,
        )

        if len(idx1) == 0 or len(idx2) == 0:
            continue

        metrics = classify_two_cells_vs_ab_shard(
            rec1["ei"], rec2["ei"], snips, idx1, idx2,
            p2p_thr=AB_P2P_THR,
            max_channels=AB_MAX_CHANNELS,
            min_channels=AB_MIN_CHANNELS,
            lag_radius=AB_LAG_RADIUS,
            weight_by_p2p=AB_WEIGHT_BY_P2P,
            weight_beta=AB_WEIGHT_BETA,
            rms_thr_support=AB_RMS_THR_SUPPORT,
            asym_strong_z=AB_ASYM_STRONG_Z,
            asym_pure_z=AB_ASYM_PURE_Z,
        )

        if metrics["label"] == "AB sharding":
            ab_pairs.append((u1, u2, metrics, p))

            pure_like = metrics["details"].get("pure_like", None)
            collision_like = metrics["details"].get("collision_like", None)

            # Map EI index -> unit id
            if pure_like == 0:
                pure_unit = u1
            elif pure_like == 1:
                pure_unit = u2
            else:
                pure_unit = None

            if collision_like == 0:
                shard_unit = u1
            elif collision_like == 1:
                shard_unit = u2
            else:
                shard_unit = None

            if shard_unit is not None:
                ab_shard_units.add(shard_unit)

    except Exception:
        # skip cursed pairs quietly
        continue

# =========================================================
# Stage 3: print only AB-shard-looking units
# =========================================================
print("\nUnits that appear to be AB shards:")
if len(ab_shard_units) == 0:
    print("None")
else:
    for u in sorted(ab_shard_units):
        print(f"U{u:03d}")

# %%
print(f"Total AB pairs: {len(ab_pairs)}\n")

if len(ab_pairs) == 0:
    print("None found.")
else:
    for i, item in enumerate(ab_pairs, start=1):
        u1, u2, metrics, p = item

        pure_like = metrics["details"].get("pure_like", None)
        collision_like = metrics["details"].get("collision_like", None)

        if pure_like == 0:
            pure_unit = f"U{u1:03d}"
        elif pure_like == 1:
            pure_unit = f"U{u2:03d}"
        else:
            pure_unit = "None"

        if collision_like == 0:
            shard_unit = f"U{u1:03d}"
        elif collision_like == 1:
            shard_unit = f"U{u2:03d}"
        else:
            shard_unit = "None"

        print(
            f"{i:3d}. "
            f"U{u1:03d} vs U{u2:03d} | "
            f"cos={p['cos']:.3f} | "
            f"overlap={p['overlap']:.3f} | "
            f"pure={pure_unit} | "
            f"shard={shard_unit}"
        )

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from plot_ei_waveforms import plot_ei_waveforms

# =========================================================
# Config
# =========================================================
SIG_MODE = "relative"   # "relative" or "absolute"
SIG_REL = 0.10
SIG_ABS = 30.0
MAX_ALIGN_LAG = 3

if "ab_pairs" not in globals():
    raise RuntimeError("ab_pairs not found. Run the AB-suspect pair check first.")

if "unit_ei_cache" not in globals():
    raise RuntimeError("unit_ei_cache not found.")

if len(ab_pairs) == 0:
    raise RuntimeError("ab_pairs is empty.")

# =========================================================
# Helpers
# =========================================================
def roll_zero_2d(arr, shift):
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :-shift]
    else:
        s = -shift
        out[:, :-s] = arr[:, s:]
    return out

def get_sig_channels(ei, mode="relative", rel=0.10, abs_thr=30.0):
    p2p = ei.max(axis=1) - ei.min(axis=1)
    if mode == "relative":
        thr = rel * np.max(p2p)
    elif mode == "absolute":
        thr = abs_thr
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")
    return np.flatnonzero(p2p >= thr)

def best_cosine_alignment(ei1, ei2, mode="relative", rel=0.10, abs_thr=30.0, max_lag=3):
    sig1 = get_sig_channels(ei1, mode=mode, rel=rel, abs_thr=abs_thr)
    sig2 = get_sig_channels(ei2, mode=mode, rel=rel, abs_thr=abs_thr)
    union = np.union1d(sig1, sig2)

    if union.size == 0:
        return np.nan, 0, union

    A = ei1[union].astype(np.float32)
    B = ei2[union].astype(np.float32)

    best_cos = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        B_shift = roll_zero_2d(B, lag)
        a = A.ravel()
        b = B_shift.ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        c = float(np.dot(a, b) / (na * nb))
        if c > best_cos:
            best_cos = c
            best_lag = lag

    return best_cos, best_lag, union

def fmt_unit(u):
    return f"U{int(u):03d}"

# =========================================================
# UI
# =========================================================
state = {"idx": 0}
out = widgets.Output(layout={"border": "1px solid #ccc"})
btn_prev = widgets.Button(description="Prev")
btn_next = widgets.Button(description="Next")
pair_dropdown = widgets.Dropdown(
    options=[(f"{i+1:02d}: {fmt_unit(item[0])} vs {fmt_unit(item[1])}", i)
             for i, item in enumerate(ab_pairs)],
    value=0,
    description="Pair:",
    layout=widgets.Layout(width="420px"),
)
info_html = widgets.HTML()

def render_pair(idx):
    idx = int(np.clip(idx, 0, len(ab_pairs) - 1))
    state["idx"] = idx
    pair_dropdown.value = idx

    u1, u2, metrics, p = ab_pairs[idx]
    rec1 = unit_ei_cache[u1]
    rec2 = unit_ei_cache[u2]

    ei1 = rec1["ei"]
    ei2 = rec2["ei"]

    cos_align, lag_align, union_ch = best_cosine_alignment(
        ei1, ei2,
        mode=SIG_MODE,
        rel=SIG_REL,
        abs_thr=SIG_ABS,
        max_lag=MAX_ALIGN_LAG,
    )
    ei2_shift = roll_zero_2d(ei2, lag_align)

    pure_like = metrics["details"].get("pure_like", None)
    collision_like = metrics["details"].get("collision_like", None)

    if pure_like == 0:
        pure_unit = fmt_unit(u1)
    elif pure_like == 1:
        pure_unit = fmt_unit(u2)
    else:
        pure_unit = "None"

    if collision_like == 0:
        shard_unit = fmt_unit(u1)
    elif collision_like == 1:
        shard_unit = fmt_unit(u2)
    else:
        shard_unit = "None"

    info_html.value = (
        f"<b>Pair {idx+1}/{len(ab_pairs)}</b> &nbsp;&nbsp; "
        f"{fmt_unit(u1)} vs {fmt_unit(u2)} &nbsp;&nbsp; "
        f"AB verdict: <b>{metrics['label']}</b> &nbsp;&nbsp; "
        f"pure=<b>{pure_unit}</b> &nbsp;&nbsp; "
        f"shard=<b>{shard_unit}</b>"
    )

    title = (
        f"{fmt_unit(u1)} (main={rec1['saved_main_ch']}, detect={rec1['detect_ch']}, "
        f"N={rec1['n_total_accepted']}, defer={rec1['n_deferred']})   vs   "
        f"{fmt_unit(u2)} (main={rec2['saved_main_ch']}, detect={rec2['detect_ch']}, "
        f"N={rec2['n_total_accepted']}, defer={rec2['n_deferred']})\n"
        f"AB={metrics['label']} | pure={pure_unit} | shard={shard_unit} | "
        f"cos(screen)={p['cos']:.3f} | overlap={p['overlap']:.3f} | "
        f"cos(plot)={cos_align:.3f} | lag={lag_align:+d} | union_sig={len(union_ch)}"
    )

    with out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_ei_waveforms(
            [ei1, ei2_shift],
            ei_positions,
            ref_channel=rec1["saved_main_ch"] if rec1["saved_main_ch"] is not None else rec1["main_ch_from_ei"],
            scale=70.0,
            box_height=1.0,
            box_width=50.0,
            ax=ax,
            colors=["black", "red"],
            alpha=[0.9, 0.9],
            linewidth=[0.6, 0.8],
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

def on_prev(_):
    render_pair(state["idx"] - 1)

def on_next(_):
    render_pair(state["idx"] + 1)

def on_dropdown_change(change):
    if change["name"] == "value" and change["new"] is not None:
        render_pair(change["new"])

btn_prev.on_click(on_prev)
btn_next.on_click(on_next)
pair_dropdown.observe(on_dropdown_change)

controls = widgets.HBox([btn_prev, btn_next])
ui = widgets.VBox([controls, pair_dropdown, info_html, out])

display(ui)
render_pair(0)

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from plot_ei_waveforms import plot_ei_waveforms

# =========================================================
# Config
# =========================================================
ISI_BIN_MS = 0.5
ISI_MAX_MS = 20.0
FS = 20000  # Hz

if "unit_ei_cache" not in globals():
    raise RuntimeError("unit_ei_cache not found.")

unit_ids_sorted = sorted(unit_ei_cache.keys())
if len(unit_ids_sorted) == 0:
    raise RuntimeError("unit_ei_cache is empty.")

# =========================================================
# Helpers
# =========================================================
def compute_isi_ms(spike_times_samples, fs=20000):
    t = np.asarray(spike_times_samples, dtype=np.int64)
    if t.size < 2:
        return np.array([], dtype=np.float32)
    t = np.sort(t)
    isi_ms = np.diff(t) * 1000.0 / fs
    return isi_ms.astype(np.float32)

def render_unit(unit_id, out_widget):
    rec = unit_ei_cache[unit_id]
    ei = rec["ei"]
    spike_times = np.asarray(rec["all_spike_times"], dtype=np.int64)

    detect_ch = rec.get("detect_ch", None)
    main_ch = rec.get("saved_main_ch", None)
    if main_ch is None:
        main_ch = rec.get("main_ch_from_ei", None)

    n_acc = rec.get("n_total_accepted", len(spike_times))
    n_def = rec.get("n_deferred", None)

    isi_ms = compute_isi_ms(spike_times, fs=FS)
    bins = np.arange(0, ISI_MAX_MS + ISI_BIN_MS, ISI_BIN_MS)

    with out_widget:
        clear_output(wait=True)

        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.0])

        # EI panel
        ax0 = fig.add_subplot(gs[0, 0])
        plot_ei_waveforms(
            ei,
            ei_positions,
            ref_channel=main_ch,
            scale=70.0,
            box_height=1.0,
            box_width=50.0,
            ax=ax0,
        )

        title = f"U{unit_id:03d} | main={main_ch}"
        if detect_ch is not None and detect_ch != main_ch:
            title += f" | detect={detect_ch}"
        title += f" | N={n_acc}"
        if n_def is not None:
            title += f" | defer={n_def}"
        ax0.set_title(title)

        # ISI panel
        ax1 = fig.add_subplot(gs[0, 1])
        if isi_ms.size > 0:
            ax1.hist(isi_ms, bins=bins)
        ax1.set_xlim(0, ISI_MAX_MS)
        ax1.set_xlabel("ISI (ms)")
        ax1.set_ylabel("Count")
        ax1.set_title("ISI (0–20 ms)")
        ax1.set_xticks(np.arange(0, ISI_MAX_MS + 1e-9, 5.0))

        plt.tight_layout()
        plt.show()

# =========================================================
# UI
# =========================================================
state = {"idx": 0}
out = widgets.Output(layout={"border": "1px solid #ccc"})
btn_prev = widgets.Button(description="Prev")
btn_next = widgets.Button(description="Next")
unit_dropdown = widgets.Dropdown(
    options=[(f"U{u:03d}", i) for i, u in enumerate(unit_ids_sorted)],
    value=0,
    description="Unit:",
    layout=widgets.Layout(width="220px"),
)
info_html = widgets.HTML()

def update():
    idx = state["idx"]
    idx = int(np.clip(idx, 0, len(unit_ids_sorted) - 1))
    state["idx"] = idx
    unit_dropdown.value = idx

    unit_id = unit_ids_sorted[idx]
    rec = unit_ei_cache[unit_id]

    info_html.value = (
        f"<b>{idx+1}/{len(unit_ids_sorted)}</b> &nbsp;&nbsp; "
        f"U{unit_id:03d} &nbsp;&nbsp; "
        f"main=<b>{rec.get('saved_main_ch', rec.get('main_ch_from_ei', None))}</b> &nbsp;&nbsp; "
        f"detect=<b>{rec.get('detect_ch', None)}</b> &nbsp;&nbsp; "
        f"N=<b>{rec.get('n_total_accepted', None)}</b> &nbsp;&nbsp; "
        f"defer=<b>{rec.get('n_deferred', None)}</b>"
    )

    render_unit(unit_id, out)

def on_prev(_):
    state["idx"] -= 1
    update()

def on_next(_):
    state["idx"] += 1
    update()

def on_dropdown_change(change):
    if change["name"] == "value" and change["new"] is not None:
        state["idx"] = int(change["new"])
        update()

btn_prev.on_click(on_prev)
btn_next.on_click(on_next)
unit_dropdown.observe(on_dropdown_change)

ui = widgets.VBox([
    widgets.HBox([btn_prev, btn_next]),
    unit_dropdown,
    info_html,
    out,
])

display(ui)
update()

# %%
import numpy as np
import os

fs = 20_000
minutes_to_save = 5
n_samples = int(minutes_to_save * 60 * fs)

out_path = "/Volumes/Lab/Users/alexth/axolotl/201711290/data004_raw_mod_first5min.dat"

chunk = raw_mod[:n_samples]

print("shape:", chunk.shape, "dtype:", chunk.dtype)

# make sure dtype is what you want on disk
chunk_to_save = np.asarray(chunk, dtype=np.int16)

chunk_to_save.tofile(out_path)

print(f"Saved: {out_path}")
print(f"Bytes: {os.path.getsize(out_path)}")

# %%
n_channels = 512
x = np.fromfile(out_path, dtype=np.int16).reshape(-1, n_channels)

# %%
save_lh_pp_loop_checkpoint(
    checkpoint_path="/Volumes/Lab/Users/alexth/axolotl/201711290/data004_lh_pp_loop.pkl",
    dat_path=dat_path,
    units_found=units_found,
    channel_state=channel_state,
    params={},
    next_unit_id=next_unit_id,
    skyline_state=skyline_state,
)

# %%
with open("/Volumes/Lab/Users/alexth/axolotl/201711290/data004_lh_pp_loop.pkl", "rb") as f:
    payload = pickle.load(f)

print(payload.keys())
print("n_units:", len(payload["units"]))
print("n_channels_in_state:", len(payload["channel_state"]))
print("next_unit_id:", payload["next_unit_id"])
print("params:", payload["params"])

# %% [markdown]
# ### extra checks for 2 units (same channel)

# %%
# === Rebuild and plot EIs for units from the PP loop, using raw_orig ===

import numpy as np
import matplotlib.pyplot as plt

from axolotl_utils_ram import extract_snippets_fast_ram
from collision_utils import median_ei_adaptive
import plot_ei_waveforms as pew

if "units_found" not in globals() or len(units_found) == 0:
    raise RuntimeError("units_found not found or empty. Run the loop first.")

# -----------------------------
# Choose which units to inspect
# -----------------------------
# Option 1: by 1-based unit IDs from the printed loop output
unit_ids_to_plot = [151, 156]   # example: U009 and U013 on ch 395

# Option 2 (alternative): by detect channel
# detect_ch_to_plot = 395
# unit_ids_to_plot = [u["unit_id"] for u in units_found if int(u["detect_ch"]) == detect_ch_to_plot]

# -----------------------------
# EI extraction settings
# -----------------------------
ei_window = (-40, 80)   # same long EI window
reducer = "median"      # "median" or "mean"

# -----------------------------
# Build EIs
# -----------------------------
eis = []
colors = []
labels = []

for uid in unit_ids_to_plot:
    matches = [u for u in units_found if int(u["unit_id"]) == int(uid)]
    if len(matches) == 0:
        print(f"Unit {uid} not found in units_found")
        continue

    u = matches[0]

    spike_times = np.asarray(u["accepted_main_times"], dtype=np.int64)
    main_ch = int(u["main_ch"])
    detect_ch = int(u["detect_ch"])

    snips, valid_times = extract_snippets_fast_ram(
        raw_data=raw_orig,
        spike_times=spike_times,
        window=ei_window,
        selected_channels=np.arange(raw_orig.shape[1], dtype=np.int32),
    )

    if snips.shape[2] == 0:
        print(f"Unit {uid}: no valid snippets after edge filtering")
        continue

    if reducer == "median":
        ei = median_ei_adaptive(snips).astype(np.float32)
    elif reducer == "mean":
        ei = snips.mean(axis=2).astype(np.float32)
    else:
        raise ValueError("reducer must be 'median' or 'mean'")

    eis.append(ei)
    labels.append(f"U{uid:03d} | detect_ch={detect_ch} | main_ch={main_ch} | n={valid_times.size}")

# -----------------------------
# Plot overlaid EIs
# -----------------------------
if len(eis) == 0:
    raise RuntimeError("No EIs were built.")

# simple color list; extend if needed
base_colors = ["black", "red", "blue", "green", "purple", "orange"]
colors = base_colors[:len(eis)]

fig, ax = plt.subplots(figsize=(20, 12))
pew.plot_ei_waveforms(
    eis,
    ei_positions,
    ref_channel=int(matches[0]["main_ch"]) if len(unit_ids_to_plot) == 1 else None,
    scale=70.0,
    box_height=1.0,
    box_width=50.0,
    ax=ax,
    colors=colors,
    alpha=[0.9] * len(eis),
    linewidth=[0.8] * len(eis),
)

ax.set_title("Overlay of rebuilt EIs from raw_orig")
plt.tight_layout()
plt.show()

for lab in labels:
    print(lab)

# %%
# === Count near-coincident spikes between 2 accepted units ===

import numpy as np

# choose units by printed ID
uid_a = 151
uid_b = 156
max_dt = 30   # samples

# fetch records
uA = next(u for u in units_found if int(u["unit_id"]) == uid_a)
uB = next(u for u in units_found if int(u["unit_id"]) == uid_b)

tA = np.asarray(uA["accepted_main_times"], dtype=np.int64)
tB = np.asarray(uB["accepted_main_times"], dtype=np.int64)

# both should already be sorted, but let's be safe
tA = np.sort(tA)
tB = np.sort(tB)

# two-pointer sweep: collect all cross pairs with |dt| <= max_dt
i = 0
j = 0
pairs = []

while i < len(tA) and j < len(tB):
    dt = int(tB[j] - tA[i])

    if dt < -max_dt:
        j += 1
        continue
    if dt > max_dt:
        i += 1
        continue

    # we're in overlap range; collect all local matches for this A spike
    jj = j
    while jj < len(tB) and (tB[jj] - tA[i]) <= max_dt:
        d = int(tB[jj] - tA[i])
        if abs(d) <= max_dt:
            pairs.append((i, jj, int(tA[i]), int(tB[jj]), d))
        jj += 1

    i += 1

n_pairs = len(pairs)

if n_pairs == 0:
    print(f"U{uid_a:03d} vs U{uid_b:03d}: no cross-unit pairs within {max_dt} samples.")
else:
    idxA = np.array([p[0] for p in pairs], dtype=np.int64)
    idxB = np.array([p[1] for p in pairs], dtype=np.int64)

    uniqueA = np.unique(idxA)
    uniqueB = np.unique(idxB)

    print(f"U{uid_a:03d} vs U{uid_b:03d}")
    print(f"cross-unit pairs within {max_dt} samples : {n_pairs}")
    print(f"unique spikes from U{uid_a:03d} involved  : {uniqueA.size} / {tA.size}")
    print(f"unique spikes from U{uid_b:03d} involved  : {uniqueB.size} / {tB.size}")

    print("\nFirst 20 example pairs:")
    for k, (_, _, ta, tb, d) in enumerate(pairs[:20], start=1):
        print(f"{k:2d}. tA={ta:8d}   tB={tb:8d}   dt={d:4d}")

# %%
ch = 3

if "channel_state" not in globals():
    print("channel_state not found in globals()")
elif ch not in channel_state:
    print(f"channel_state has no entry for channel {ch}")
else:
    ex = channel_state[ch].get("exclude_times", None)
    if ex is None:
        print(f'channel_state[{ch}]["exclude_times"] is None')
    else:
        ex = np.asarray(ex)
        print(f"len(channel_state[{ch}]['exclude_times']) = {len(ex)}")
        print("first 20 excluded times:")
        print(ex[:20])

# %%
# %% DEBUG ONE-CHANNEL PP LOOP (safe by default; no mutation unless commit=True)

import copy
import numpy as np
import matplotlib.pyplot as plt


def _dbg_fmt(x, nd=3):
    try:
        if x is None:
            return "None"
        if isinstance(x, (list, tuple, np.ndarray)):
            return str(x)
        if np.isfinite(x):
            if float(x).is_integer():
                return str(int(x))
            return f"{float(x):.{nd}f}"
    except Exception:
        pass
    return str(x)


def _dbg_print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _dbg_print_kv(title, items):
    print(f"\n[{title}]")
    for k, v in items:
        print(f"  {k:<28s} {v}")


def _dbg_plot_step1_hist(step1, detect_ch):
    counts = np.asarray(step1.get("amp_hist_counts", []))
    edges = np.asarray(step1.get("amp_hist_edges", []))
    if counts.size == 0 or edges.size < 2:
        print("[plot] no step1 histogram available")
        return

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    plt.figure(figsize=(9, 4))
    plt.bar(centers, counts, width=widths, align="center", alpha=0.75)

    valley_low = step1.get("valley_low", None)
    valley_high = step1.get("valley_high", None)
    if valley_low is not None:
        plt.axvline(valley_low, linestyle="--", linewidth=1.5, label=f"valley_low={valley_low:.1f}")
    if valley_high is not None:
        plt.axvline(valley_high, linestyle="--", linewidth=1.5, label=f"valley_high={valley_high:.1f}")
    if valley_low is not None and valley_high is not None:
        plt.axvspan(valley_low, valley_high, alpha=0.12)

    plt.title(
        f"ch {int(detect_ch)} minima histogram | "
        f"left={int(step1.get('left_count', 0))} "
        f"valley={int(step1.get('valley_count', 0))} "
        f"vfrac={float(step1.get('valley_frac_of_left', np.nan)):.3f}"
    )
    plt.xlabel("minimum amplitude")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _dbg_tree_leaf_table(leaves, full_side):
    print("\n[PP leaves]")
    if leaves is None or len(leaves) == 0:
        print("  (none)")
        return

    for leaf in leaves:
        idx = np.asarray(leaf.get("idx", []), dtype=np.int64)
        n = idx.size
        n_left = int(np.sum(full_side[idx] == 0)) if n else 0
        n_right = int(np.sum(full_side[idx] == 1)) if n else 0
        frac_left = (n_left / float(n)) if n else np.nan
        print(
            "  path={path:<16s} n={n:<6d} left={nl:<6d} right={nr:<6d} frac_left={fl:>6s} reason={reason}".format(
                path=str(leaf.get("path", "")),
                n=int(n),
                nl=int(n_left),
                nr=int(n_right),
                fl=_dbg_fmt(frac_left, nd=3),
                reason=str(leaf.get("reason", None)),
            )
        )


def _dbg_walk_tree(node, full_side, indent=""):
    if node is None:
        print(indent + "(none)")
        return

    node_type = str(node.get("type", "unknown"))
    path = str(node.get("path", "?"))
    idx = np.asarray(node.get("idx", []), dtype=np.int64)
    n = idx.size
    n_left = int(np.sum(full_side[idx] == 0)) if n else 0
    n_right = int(np.sum(full_side[idx] == 1)) if n else 0
    frac_left = (n_left / float(n)) if n else np.nan

    if node_type == "split":
        best_full = node.get("best_full", {})
        score = best_full.get("score", None)
        sep = best_full.get("sep", None)
        depth = best_full.get("depth", None)
        thr = best_full.get("thr", None)
        pooled_sd = best_full.get("pooled_sd", None)
        leftlike = node.get("leftlike_is_proj_left", None)
        scoreable = node.get("scoreable_for_preflight", None)

        print(
            indent +
            "SPLIT path={path:<16s} n={n:<6d} left={nl:<6d} right={nr:<6d} "
            "frac_left={fl:>6s} score={sc:>8s} sep={sep:>8s} depth={dep:>8s} "
            "thr={thr:>8s} psd={psd:>8s} leftlike={ll} scoreable={scoreable}".format(
                path=path,
                n=int(n),
                nl=int(n_left),
                nr=int(n_right),
                fl=_dbg_fmt(frac_left, nd=3),
                sc=_dbg_fmt(score, nd=3),
                sep=_dbg_fmt(sep, nd=3),
                dep=_dbg_fmt(depth, nd=3),
                thr=_dbg_fmt(thr, nd=3),
                psd=_dbg_fmt(pooled_sd, nd=3),
                ll=str(leftlike),
                scoreable=str(scoreable),
            )
        )
        _dbg_walk_tree(node.get("left_child", None), full_side, indent + "    ")
        _dbg_walk_tree(node.get("right_child", None), full_side, indent + "    ")

    else:
        print(
            indent +
            "LEAF  path={path:<16s} n={n:<6d} left={nl:<6d} right={nr:<6d} "
            "frac_left={fl:>6s} reason={reason}".format(
                path=path,
                n=int(n),
                nl=int(n_left),
                nr=int(n_right),
                fl=_dbg_fmt(frac_left, nd=3),
                reason=str(node.get("reason", None)),
            )
        )


def _dbg_plot_provisional_ei(ei, ei_positions, ref_channel, title):
    plotter = globals().get("pew", None)
    if plotter is None or ei_positions is None:
        return
    try:
        fig, ax = plt.subplots(figsize=(20, 12))
        plotter.plot_ei_waveforms(
            ei,
            ei_positions,
            ref_channel=int(ref_channel),
            scale=70.0,
            box_height=1.0,
            box_width=50.0,
            ax=ax,
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"[plot] EI plot skipped: {exc}")


def debug_one_channel_pp_loop(
    detect_ch,
    *,
    raw_mod_=None,
    ei_positions_=None,
    channel_state_=None,
    params_=None,
    units_found_=None,
    commit=False,
    unit_id=None,
    save_checkpoint=False,
    show_plots=True,
    show_tree=True,
    show_leaf_table=True,
):
    """
    One-channel mirror of the PP loop with verbose reporting.

    Default behavior:
      - does NOT mutate raw_mod
      - does NOT update channel_state / units_found
      - runs the same stages as the loop and prints intermediate details

    Set commit=True if you want it to actually subtract / append / update state.
    """

    # Pull notebook globals by default
    if raw_mod_ is None:
        raw_mod_ = raw_mod
    if ei_positions_ is None:
        ei_positions_ = ei_positions
    if channel_state_ is None:
        channel_state_ = channel_state
    if params_ is None:
        params_ = LH_PP_LOOP_PARAMS
    if units_found_ is None and "units_found" in globals():
        units_found_ = units_found

    detect_ch = int(detect_ch)
    params_local = copy.deepcopy(params_)

    out = {
        "detect_ch": detect_ch,
        "skyline_state": None,
        "pool": None,
        "pp_tree_state": None,
        "final_left_state": None,
        "lh_template_bank": None,
        "preview_res": None,
        "preflight": None,
        "commit_res": None,
    }

    _dbg_print_header(f"PP DEBUG FOR CHANNEL {detect_ch}")

    # ------------------------------------------------------------------
    # Stage 0: skyline / channel state context
    # ------------------------------------------------------------------
    skyline_state = skyline_scan_with_exclusions(raw_mod_, channel_state_, params_local)
    out["skyline_state"] = skyline_state

    order = np.asarray(skyline_state["channel_order"], dtype=np.int64)
    rank_hits = np.where(order == detect_ch)[0]
    skyline_rank = int(rank_hits[0]) if rank_hits.size else None

    skyline_scores = np.asarray(skyline_state["mean_amp_topk"], dtype=np.float32)
    n_kept = np.asarray(skyline_state["n_minima_kept"], dtype=np.int32)
    n_all = np.asarray(skyline_state["n_minima_all"], dtype=np.int32)

    st = channel_state_.get(detect_ch, {})
    excl = np.asarray(st.get("exclude_times", []), dtype=np.int64)

    _dbg_print_kv("channel context", [
        ("detect_ch", detect_ch),
        ("skyline_rank", skyline_rank),
        ("skyline_mean_topk", _dbg_fmt(skyline_scores[detect_ch], nd=3)),
        ("skyline_n_minima_kept", int(n_kept[detect_ch])),
        ("skyline_n_minima_all", int(n_all[detect_ch])),
        ("exclude_times_len", int(excl.size)),
        ("exclude_times_first20", excl[:20].tolist()),
        ("skip_streak", st.get("skip_streak", None)),
        ("last_skip_reason", st.get("last_skip_reason", None)),
        ("last_attempt_outcome", st.get("last_attempt_outcome", None)),
        ("last_attempt_success_count", st.get("last_attempt_success_count", None)),
        ("needs_retry", st.get("needs_retry", None)),
        ("retry_valley_low", st.get("retry_valley_low", None)),
        ("retry_valley_high", st.get("retry_valley_high", None)),
        ("retry_trigger_unit_id", st.get("retry_trigger_unit_id", None)),
        ("retry_trigger_detect_ch", st.get("retry_trigger_detect_ch", None)),
        ("retry_trigger_minimum", st.get("retry_trigger_minimum", None)),
    ])

    # ------------------------------------------------------------------
    # Stage 1: PP pool / valley seed
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 1 — build_pp_pool_for_channel")
    pool = build_pp_pool_for_channel(raw_mod_, detect_ch, channel_state_, params_local)
    out["pool"] = pool

    step1 = pool.get("step1", None)
    if step1 is not None:
        _dbg_print_kv("step1 / valley", [
            ("pool_status", pool.get("status", None)),
            ("pool_reason", pool.get("reason", None)),
            ("lh_accepted", step1.get("lh_accepted", None)),
            ("pp_seed_ok", step1.get("pp_seed_ok", None)),
            ("pp_seed_reason", step1.get("pp_seed_reason", None)),
            ("valley_low", step1.get("valley_low", None)),
            ("valley_high", step1.get("valley_high", None)),
            ("left_count", step1.get("left_count", None)),
            ("valley_count", step1.get("valley_count", None)),
            ("valley_frac_of_left", _dbg_fmt(step1.get("valley_frac_of_left", None), nd=4)),
            ("required_ratio", step1.get("required_ratio", None)),
            ("pp_min_left_count", step1.get("pp_min_left_count", None)),
            ("pp_max_valley_count", step1.get("pp_max_valley_count", None)),
            ("pp_max_valley_frac_of_left", step1.get("pp_max_valley_frac_of_left", None)),
            ("n_all_times", len(np.asarray(step1.get("all_times", []), dtype=np.int64))),
            ("n_left_times", len(np.asarray(step1.get("left_times", []), dtype=np.int64))),
            ("n_valley_times", len(np.asarray(step1.get("valley_times", []), dtype=np.int64))),
        ])

        if show_plots:
            _dbg_plot_step1_hist(step1, detect_ch)

    if pool["status"] != "ok":
        print("\nSTOP: stage 1 failed.")
        return out

    _dbg_print_kv("pool contents", [
        ("stop", pool.get("stop", None)),
        ("window_pp", pool.get("window_pp", None)),
        ("left_times_n", len(np.asarray(pool.get("left_times", []), dtype=np.int64))),
        ("right_times_n", len(np.asarray(pool.get("right_times", []), dtype=np.int64))),
        ("full_times_n", len(np.asarray(pool.get("full_times", []), dtype=np.int64))),
    ])

    # ------------------------------------------------------------------
    # Stage 2: recursive PP
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 2 — run_recursive_pp_for_channel")
    pp_tree_state = run_recursive_pp_for_channel(raw_mod_, ei_positions_, pool, params_local)
    out["pp_tree_state"] = pp_tree_state

    tree = pp_tree_state.get("tree", None)
    leaves = pp_tree_state.get("leaves", [])
    full_side = np.asarray(pp_tree_state.get("full_side", []), dtype=np.int32)

    _dbg_print_kv("PP tree summary", [
        ("tree_type", None if tree is None else tree.get("type", None)),
        ("tree_reason", None if tree is None else tree.get("reason", None)),
        ("n_leaves", len(leaves)),
        ("anchor_ch", pp_tree_state.get("anchor_ch", None)),
        ("window_pp", pp_tree_state.get("window_pp", None)),
        ("full_times_n", len(np.asarray(pp_tree_state.get("full_times", []), dtype=np.int64))),
    ])

    if show_leaf_table:
        _dbg_tree_leaf_table(leaves, full_side)

    if show_tree and tree is not None:
        print("\n[PP tree walk]")
        _dbg_walk_tree(tree, full_side)

    if tree is None or tree.get("type", None) != "split":
        print("\nSTOP: stage 2 did not produce a split.")
        return out

    # ------------------------------------------------------------------
    # Stage 3: finalize labels + short bank
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 3 — finalize_labels_and_short_bank")
    try:
        final_left_state, lh_template_bank = finalize_labels_and_short_bank(raw_mod_, pp_tree_state, pool, params_local)
    except Exception as exc:
        print(f"STOP: finalize failed with exception: {exc}")
        return out

    out["final_left_state"] = final_left_state
    out["lh_template_bank"] = lh_template_bank

    _dbg_print_kv("final left state", [
        ("main_leaf_path", final_left_state.get("main_leaf_path", None)),
        ("main_path_split_path", final_left_state.get("main_path_split_path", None)),
        ("main_path_score", _dbg_fmt(final_left_state.get("main_path_score", None), nd=3)),
        ("main_path_sep", _dbg_fmt(final_left_state.get("main_path_sep", None), nd=3)),
        ("main_path_depth", _dbg_fmt(final_left_state.get("main_path_depth", None), nd=3)),
        ("last_path_split_path", final_left_state.get("last_path_split_path", None)),
        ("last_path_score", _dbg_fmt(final_left_state.get("last_path_score", None), nd=3)),
        ("last_path_sep", _dbg_fmt(final_left_state.get("last_path_sep", None), nd=3)),
        ("last_path_depth", _dbg_fmt(final_left_state.get("last_path_depth", None), nd=3)),
        ("trusted_left_raw_n", len(np.asarray(final_left_state.get("trusted_left_times_raw", []), dtype=np.int64))),
        ("trusted_left_clean_n", len(np.asarray(final_left_state.get("trusted_left_times", []), dtype=np.int64))),
        ("trusted_left_dropped_close_n", len(np.asarray(final_left_state.get("trusted_left_times_dropped_close", []), dtype=np.int64))),
        ("extra_left_n", len(np.asarray(final_left_state.get("extra_left_times", []), dtype=np.int64))),
        ("uncertain_n", len(np.asarray(final_left_state.get("uncertain_times", []), dtype=np.int64))),
        ("deferred_small_split_n", len(np.asarray(final_left_state.get("deferred_small_split_times", []), dtype=np.int64))),
        ("trusted_not_left_n", len(np.asarray(final_left_state.get("trusted_not_left_times", []), dtype=np.int64))),
        ("isi_10_30_trusted_left", final_left_state.get("isi_10_30_trusted_left", None)),
        ("majority_left_leaves", final_left_state.get("majority_left_leaves", None)),
        ("extra_left_leaf_paths", final_left_state.get("extra_left_leaf_paths", None)),
        ("deferred_small_split_leaf_paths", final_left_state.get("deferred_small_split_leaf_paths", None)),
    ])

    _dbg_print_kv("short bank", [
        ("bank_window", lh_template_bank.get("window", None)),
        ("bank_reducer", lh_template_bank.get("reducer", None)),
        ("main_ch", lh_template_bank.get("main_ch", None)),
        ("t0_main", lh_template_bank.get("t0_main", None)),
        ("n_templates", len(lh_template_bank.get("templates", []))),
        ("times_all_n", len(np.asarray(lh_template_bank.get("times_all", []), dtype=np.int64))),
    ])

    for rec in lh_template_bank.get("templates", []):
        print(
            "  bin={bi:<3d} n_spikes={ns:<5d} amp_min={amin:>8s} amp_max={amax:>8s}".format(
                bi=int(rec.get("bin_index", -1)),
                ns=int(rec.get("n_spikes", 0)),
                amin=_dbg_fmt(rec.get("amp_min", None), nd=3),
                amax=_dbg_fmt(rec.get("amp_max", None), nd=3),
            )
        )

    if show_plots:
        _dbg_plot_provisional_ei(
            lh_template_bank.get("provisional_ei", None),
            ei_positions_,
            detect_ch,
            title=f"ch {detect_ch} provisional EI from trusted_left",
        )

    # ------------------------------------------------------------------
    # Stage 4: dry-run assignment preview
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 4 — assign_and_subtract_unit(dry_run=True)")
    preview_res = assign_and_subtract_unit(
        raw_mod_,
        final_left_state,
        lh_template_bank,
        detect_ch,
        params_local,
        dry_run=True,
    )
    out["preview_res"] = preview_res

    _dbg_print_kv("preview status", [
        ("preview_status", preview_res.get("status", None)),
        ("preview_reason", preview_res.get("reason", None)),
    ])

    if preview_res["status"] != "ok":
        print("\nSTOP: dry-run assignment failed.")
        return out

    preview_unit = preview_res["unit_record"]
    preview_summary = dict(preview_unit.get("summary", {}))

    _dbg_print_kv("preview unit", [
        ("detect_ch", preview_unit.get("detect_ch", None)),
        ("main_ch", preview_unit.get("main_ch", None)),
        ("initial_left_count", preview_unit.get("initial_left_count", None)),
        ("initial_valley_count", preview_unit.get("initial_valley_count", None)),
        ("n_extra_left", preview_unit.get("n_extra_left", None)),
        ("trusted_left_detect_amp_mean", _dbg_fmt(preview_unit.get("trusted_left_detect_amp_mean", None), nd=3)),
        ("pp_main_leaf_path", preview_unit.get("pp_main_leaf_path", None)),
        ("pp_main_split_path", preview_unit.get("pp_main_split_path", None)),
        ("pp_main_score", _dbg_fmt(preview_unit.get("pp_main_score", None), nd=3)),
        ("pp_main_sep", _dbg_fmt(preview_unit.get("pp_main_sep", None), nd=3)),
        ("pp_main_depth", _dbg_fmt(preview_unit.get("pp_main_depth", None), nd=3)),
        ("pp_last_split_path", preview_unit.get("pp_last_split_path", None)),
        ("pp_last_score", _dbg_fmt(preview_unit.get("pp_last_score", None), nd=3)),
        ("pp_last_sep", _dbg_fmt(preview_unit.get("pp_last_sep", None), nd=3)),
        ("pp_last_depth", _dbg_fmt(preview_unit.get("pp_last_depth", None), nd=3)),
        ("isi_10_30_main", preview_unit.get("isi_10_30_main", None)),
        ("accepted_detect_times_n", len(np.asarray(preview_unit.get("accepted_detect_times", []), dtype=np.int64))),
        ("accepted_main_times_n", len(np.asarray(preview_unit.get("accepted_main_times", []), dtype=np.int64))),
        ("deferred_all_detect_times_n", len(np.asarray(preview_unit.get("deferred_all_detect_times", []), dtype=np.int64))),
        ("long_bank_bins_n", len(preview_unit.get("long_bank", {}).get("bins", []))),
        ("summary_n_trusted_left_clean", preview_summary.get("n_trusted_left_clean", None)),
        ("summary_n_accepted", preview_summary.get("n_accepted", None)),
        ("summary_n_deferred_unique", preview_summary.get("n_deferred_unique", None)),
        ("summary_n_rejected_fit", preview_summary.get("n_rejected_fit", None)),
        ("summary_n_postlag_drop", preview_summary.get("n_postlag_drop", None)),
        ("summary_defer_frac", _dbg_fmt(preview_summary.get("defer_frac", None), nd=4)),
    ])

    tmp_unit_id = int(unit_id) if unit_id is not None else 0
    tmp_result = {"status": "ok", "unit_record": dict(preview_unit, unit_id=tmp_unit_id)}
    print("\n[preview loop line]")
    print(format_channel_attempt_line(tmp_result))

    # ------------------------------------------------------------------
    # Stage 5: ISI gate
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 5 — ISI gate")
    isi_10_30_main = int(preview_unit.get("isi_10_30_main", 0))
    max_isi_10_30_main = int(params_local["max_isi_10_30_main"])
    isi_ok = isi_10_30_main <= max_isi_10_30_main

    _dbg_print_kv("ISI", [
        ("isi_10_30_main", isi_10_30_main),
        ("max_isi_10_30_main", max_isi_10_30_main),
        ("isi_ok", isi_ok),
    ])

    if not isi_ok:
        print("\nSTOP: fails ISI gate.")
        return out

    # ------------------------------------------------------------------
    # Stage 6: preflight
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 6 — evaluate_pp_preflight")
    preflight = evaluate_pp_preflight(preview_unit, pool, params_local)
    out["preflight"] = preflight

    _dbg_print_kv("preflight", [
        ("ok", preflight.get("ok", None)),
        ("reason", preflight.get("reason", None)),
        ("pp_score", _dbg_fmt(preflight.get("pp_score", None), nd=3)),
        ("pp_score_min", _dbg_fmt(preflight.get("pp_score_min", None), nd=3)),
        ("deferral_frac", _dbg_fmt(preflight.get("deferral_frac", None), nd=4)),
        ("deferral_frac_max", _dbg_fmt(preflight.get("deferral_frac_max", None), nd=4)),
        ("left_count", preflight.get("left_count", None)),
        ("valley_count", preflight.get("valley_count", None)),
        ("valley_frac_of_left", _dbg_fmt(preflight.get("valley_frac_of_left", None), nd=4)),
        ("lh_accepted", preflight.get("lh_accepted", None)),
        ("checks", preflight.get("checks", None)),
    ])

    if not preflight.get("ok", False):
        print("\nSTOP: fails preflight.")
        return out

    # ------------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------------
    if not commit:
        _dbg_print_header("DRY RUN COMPLETE")
        print("No mutation performed. raw_mod, channel_state, and units_found were left unchanged.")
        print("If this looks good, rerun with commit=True.")
        return out

    # ------------------------------------------------------------------
    # Stage 7: real assignment / subtraction / state update
    # ------------------------------------------------------------------
    _dbg_print_header("STAGE 7 — COMMIT")
    commit_res = assign_and_subtract_unit(
        raw_mod_,
        final_left_state,
        lh_template_bank,
        detect_ch,
        params_local,
        dry_run=False,
    )
    out["commit_res"] = commit_res

    _dbg_print_kv("commit status", [
        ("commit_status", commit_res.get("status", None)),
        ("commit_reason", commit_res.get("reason", None)),
    ])

    if commit_res["status"] != "ok":
        print("\nSTOP: real assignment failed during commit.")
        return out

    unit_record = commit_res["unit_record"]
    unit_record["preflight"] = preflight

    if unit_id is None:
        if "next_unit_id" in globals():
            unit_id_local = int(next_unit_id)
        elif units_found_ is not None:
            unit_id_local = int(len(units_found_) + 1)
        else:
            unit_id_local = 1
    else:
        unit_id_local = int(unit_id)

    unit_record["unit_id"] = unit_id_local

    if units_found_ is not None:
        units_found_.append(unit_record)

    update_channel_state_after_success(channel_state_, unit_record)

    if "next_unit_id" in globals():
        globals()["next_unit_id"] = int(unit_id_local + 1)

    skyline_state_after = skyline_scan_with_exclusions(raw_mod_, channel_state_, params_local)
    out["skyline_state_after"] = skyline_state_after

    print("\n[committed loop line]")
    print(format_channel_attempt_line({"status": "ok", "unit_record": unit_record}))

    if save_checkpoint:
        if "dat_path" not in globals():
            print("\n[checkpoint] dat_path not found; skipping checkpoint save.")
        else:
            ckpt_path = params_local["checkpoint_path"]
            save_lh_pp_loop_checkpoint(
                checkpoint_path=ckpt_path,
                dat_path=dat_path,
                units_found=units_found_ if units_found_ is not None else [],
                channel_state=channel_state_,
                params=params_local,
                next_unit_id=int(unit_id_local + 1),
                skyline_state=skyline_state_after,
            )
            print(f"\n[checkpoint] saved -> {ckpt_path}")

    _dbg_print_header("COMMIT COMPLETE")
    return out

# %%
dbg = debug_one_channel_pp_loop(508, commit=False, show_plots=False)

print("main_path_split_path:", dbg["final_left_state"]["main_path_split_path"])
print("extra_left_n:", len(dbg["final_left_state"]["extra_left_times"]))
print("extra_left_paths:", dbg["final_left_state"]["extra_left_leaf_paths"])
print("uncertain_n:", len(dbg["final_left_state"]["uncertain_times"]))
print("uncertain_paths:", dbg["final_left_state"].get("uncertain_source_paths", None))

preview = dbg["preview_res"]["unit_record"]
print("deferred_by_reason keys:", preview["deferred_by_reason"].keys())
print("n_deferred_unique:", preview["summary"]["n_deferred_unique"])

# %%
dbg = debug_one_channel_pp_loop(160, commit=False, show_plots=True) # 24 and 98 are bad isi; 479

# %%
# ==== Rejected-fit spikes vs bad-ISI spikes, plus rejected-fit mean EI ====

import numpy as np
import matplotlib.pyplot as plt

def inspect_rejected_fit_vs_bad_isi(
    dbg,
    raw_data_for_ei=None,
    ei_positions_for_plot=None,
    isi_min=10,
    isi_max=30,
    window=None,
):
    """
    Uses dbg from debug_one_channel_pp_loop(...).

    1) Finds bad-ISI spikes from accepted_main_times / accepted_detect_times
    2) Checks intersection with rejected_fit_detect_times
       (and also postlag_drop / edge_long for completeness)
    3) Builds mean EI from rejected_fit spikes
    4) Overlays accepted mean EI vs rejected-fit mean EI
    """

    if dbg is None:
        raise ValueError("dbg is None")

    if raw_data_for_ei is None:
        if "raw_orig" in globals():
            raw_data_for_ei = raw_orig
        elif "raw_mod" in globals():
            raw_data_for_ei = raw_mod
        else:
            raise RuntimeError("No raw array found. Pass raw_data_for_ei explicitly.")

    if ei_positions_for_plot is None:
        if "ei_positions" in globals():
            ei_positions_for_plot = ei_positions
        else:
            raise RuntimeError("No ei_positions found. Pass ei_positions_for_plot explicitly.")

    # Prefer committed result if present; otherwise preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg.")

    detect_ch = int(unit["detect_ch"])
    main_ch = int(unit["main_ch"])

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    accepted_detect_times = np.asarray(unit["accepted_detect_times"], dtype=np.int64)

    rejected_fit_detect_times = np.asarray(unit.get("rejected_fit_detect_times", []), dtype=np.int64)
    postlag_dropped_detect_times = np.asarray(unit.get("postlag_dropped_detect_times", []), dtype=np.int64)
    edge_invalid_long_detect_times = np.asarray(unit.get("edge_invalid_long_detect_times", []), dtype=np.int64)

    if window is None:
        if "long_window" in unit:
            window = tuple(unit["long_window"])
        else:
            window = (-40, 80)
    pre, post = int(window[0]), int(window[1])

    # --- find bad-ISI spikes from accepted set ---
    if accepted_main_times.size < 2:
        raise RuntimeError("Not enough accepted spikes to form ISI pairs.")

    order = np.argsort(accepted_main_times)
    t_main_sorted = accepted_main_times[order]
    t_detect_sorted = accepted_detect_times[order]
    d = np.diff(t_main_sorted)

    bad_pair_idx = np.where((d >= int(isi_min)) & (d <= int(isi_max)))[0]
    bad_first_detect = t_detect_sorted[bad_pair_idx].astype(np.int64)
    bad_second_detect = t_detect_sorted[bad_pair_idx + 1].astype(np.int64)
    bad_all_detect = np.unique(np.concatenate([bad_first_detect, bad_second_detect])).astype(np.int64)

    # --- intersections ---
    inter_rejected = np.intersect1d(bad_all_detect, rejected_fit_detect_times)
    inter_postlag = np.intersect1d(bad_all_detect, postlag_dropped_detect_times)
    inter_edge = np.intersect1d(bad_all_detect, edge_invalid_long_detect_times)

    print("=" * 100)
    print("BAD-ISI vs REJECTED-FIT")
    print("=" * 100)
    print(f"detect_ch                         : {detect_ch}")
    print(f"main_ch                           : {main_ch}")
    print(f"accepted spikes                   : {accepted_detect_times.size}")
    print(f"bad ISI pairs                     : {bad_pair_idx.size}")
    print(f"unique bad-ISI spikes             : {bad_all_detect.size}")
    print(f"rejected_fit spikes               : {rejected_fit_detect_times.size}")
    print(f"postlag_dropped spikes            : {postlag_dropped_detect_times.size}")
    print(f"edge_invalid_long spikes          : {edge_invalid_long_detect_times.size}")
    print()
    print(f"intersection bad-ISI ∩ rejected_fit : {inter_rejected.size}")
    print(f"intersection bad-ISI ∩ postlag_drop : {inter_postlag.size}")
    print(f"intersection bad-ISI ∩ edge_long    : {inter_edge.size}")

    if inter_rejected.size:
        print("first 20 bad-ISI ∩ rejected_fit detect times:")
        print(inter_rejected[:20])

    if inter_postlag.size:
        print("first 20 bad-ISI ∩ postlag_drop detect times:")
        print(inter_postlag[:20])

    if inter_edge.size:
        print("first 20 bad-ISI ∩ edge_long detect times:")
        print(inter_edge[:20])

    # --- helper to build mean EI ---
    def build_mean_ei(spike_times):
        spike_times = np.asarray(spike_times, dtype=np.int64)
        if spike_times.size == 0:
            return None, np.array([], dtype=np.int64)

        all_ch = np.arange(raw_data_for_ei.shape[1], dtype=np.int32)
        sn, valid = extract_snippets_fast_ram(
            raw_data=raw_data_for_ei,
            spike_times=spike_times,
            window=(pre, post),
            selected_channels=all_ch,
        )
        sn = sn.astype(np.float32, copy=False)
        valid = np.asarray(valid, dtype=np.int64)

        if sn.shape[2] == 0:
            return None, valid

        ei = sn.mean(axis=2).astype(np.float32)
        return ei, valid

    ei_acc, valid_acc = build_mean_ei(accepted_detect_times)
    ei_rej, valid_rej = build_mean_ei(rejected_fit_detect_times)

    if ei_rej is None:
        print("\nNo valid rejected-fit snippets for EI.")
        return {
            "bad_all_detect": bad_all_detect,
            "rejected_fit_detect_times": rejected_fit_detect_times,
            "intersection_rejected": inter_rejected,
            "intersection_postlag": inter_postlag,
            "intersection_edge": inter_edge,
            "ei_accepted": ei_acc,
            "ei_rejected_fit": None,
        }

    # --- overlay accepted vs rejected-fit mean EI ---
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_ei_waveforms.plot_ei_waveforms(
        [ei_acc, ei_rej],
        ei_positions_for_plot,
        ref_channel=main_ch,
        scale=70.0,
        box_height=1.0,
        box_width=50.0,
        ax=ax,
        colors=["black", "red"],
    )
    ax.set_title(
        f"Accepted mean EI (black, n={valid_acc.size}) vs rejected-fit mean EI (red, n={valid_rej.size}) "
        f"| ch={detect_ch}, main={main_ch}"
    )
    plt.tight_layout()
    plt.show()

    return {
        "bad_all_detect": bad_all_detect,
        "bad_first_detect": bad_first_detect,
        "bad_second_detect": bad_second_detect,
        "rejected_fit_detect_times": rejected_fit_detect_times,
        "postlag_dropped_detect_times": postlag_dropped_detect_times,
        "edge_invalid_long_detect_times": edge_invalid_long_detect_times,
        "intersection_rejected": inter_rejected,
        "intersection_postlag": inter_postlag,
        "intersection_edge": inter_edge,
        "ei_accepted": ei_acc,
        "ei_rejected_fit": ei_rej,
        "valid_acc": valid_acc,
        "valid_rej": valid_rej,
    }


# ---- usage ----
rej = inspect_rejected_fit_vs_bad_isi(dbg, raw_data_for_ei=raw_mod)

# %%
# ==== Bad-ISI spikes vs residual-fit score of accepted spikes ====

import numpy as np
import matplotlib.pyplot as plt

def inspect_bad_isi_residual_scores(
    dbg,
    isi_min=10,
    isi_max=30,
    show_plots=True,
):
    """
    Compare bad-ISI spikes to the rest of the accepted spikes using the
    short-bank fit scores already stored in dbg.

    Uses:
      accepted_post_score   : weighted residual RMS after best fit (lower is better)
      accepted_pre_score    : pre-fit weighted RMS
      accepted_improvement  : pre - post (higher is better)
    """

    if dbg is None:
        raise ValueError("dbg is None")

    # Prefer committed result if present; otherwise preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg.")

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    accepted_detect_times = np.asarray(unit["accepted_detect_times"], dtype=np.int64)
    accepted_pre_score = np.asarray(unit["accepted_pre_score"], dtype=np.float32)
    accepted_post_score = np.asarray(unit["accepted_post_score"], dtype=np.float32)
    accepted_improvement = np.asarray(unit["accepted_improvement"], dtype=np.float32)
    accepted_best_bin = np.asarray(unit["accepted_best_bin"], dtype=np.int32)
    accepted_best_lag = np.asarray(unit["accepted_best_lag"], dtype=np.int32)

    if not (
        accepted_main_times.size
        == accepted_detect_times.size
        == accepted_pre_score.size
        == accepted_post_score.size
        == accepted_improvement.size
        == accepted_best_bin.size
        == accepted_best_lag.size
    ):
        raise RuntimeError("Accepted arrays have mismatched lengths.")

    if accepted_main_times.size < 2:
        raise RuntimeError("Not enough accepted spikes to form ISI pairs.")

    # --- identify bad adjacent pairs using canonical main-channel times ---
    order = np.argsort(accepted_main_times)
    t_main_sorted = accepted_main_times[order]
    d = np.diff(t_main_sorted)

    bad_pair_idx = np.where((d >= int(isi_min)) & (d <= int(isi_max)))[0]
    if bad_pair_idx.size == 0:
        print(f"No adjacent accepted spike pairs with ISI in [{isi_min}, {isi_max}] samples.")
        return None

    first_rows = order[bad_pair_idx]
    second_rows = order[bad_pair_idx + 1]
    bad_all_rows = np.unique(np.concatenate([first_rows, second_rows]).astype(np.int64))
    other_rows = np.setdiff1d(np.arange(accepted_main_times.size, dtype=np.int64), bad_all_rows)

    post_first = accepted_post_score[first_rows]
    post_second = accepted_post_score[second_rows]
    post_bad = accepted_post_score[bad_all_rows]
    post_other = accepted_post_score[other_rows]

    pre_first = accepted_pre_score[first_rows]
    pre_second = accepted_pre_score[second_rows]
    pre_bad = accepted_pre_score[bad_all_rows]
    pre_other = accepted_pre_score[other_rows]

    imp_first = accepted_improvement[first_rows]
    imp_second = accepted_improvement[second_rows]
    imp_bad = accepted_improvement[bad_all_rows]
    imp_other = accepted_improvement[other_rows]

    ratio_all = accepted_post_score / (accepted_pre_score + 1e-12)
    ratio_first = ratio_all[first_rows]
    ratio_second = ratio_all[second_rows]
    ratio_bad = ratio_all[bad_all_rows]
    ratio_other = ratio_all[other_rows]

    def _summ(name, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            print(f"{name:<22s} n=0")
            return
        print(
            f"{name:<22s} n={x.size:4d}  "
            f"mean={x.mean():9.4f}  median={np.median(x):9.4f}  "
            f"std={x.std(ddof=1) if x.size > 1 else np.nan:9.4f}"
        )

    def _percentile_of_subset(subset, population):
        subset = np.asarray(subset, dtype=np.float64)
        population = np.asarray(population, dtype=np.float64)
        if subset.size == 0 or population.size == 0:
            return np.nan
        return np.mean([np.mean(population <= v) for v in subset])

    print("=" * 100)
    print("BAD-ISI SPIKES VS SHORT-BANK RESIDUAL SCORE")
    print("=" * 100)
    print(f"detect_ch              : {int(unit['detect_ch'])}")
    print(f"main_ch                : {int(unit['main_ch'])}")
    print(f"accepted spikes        : {accepted_main_times.size}")
    print(f"bad ISI pairs          : {bad_pair_idx.size}")
    print(f"unique bad spikes      : {bad_all_rows.size}")
    print()

    print("[accepted_post_score]  lower = better residual")
    _summ("first spikes", post_first)
    _summ("second spikes", post_second)
    _summ("all bad spikes", post_bad)
    _summ("other accepted", post_other)
    print(f"mean percentile of bad spikes within all accepted post_score: {_percentile_of_subset(post_bad, accepted_post_score):.4f}")
    print()

    print("[accepted_pre_score]   lower = cleaner before fit")
    _summ("first spikes", pre_first)
    _summ("second spikes", pre_second)
    _summ("all bad spikes", pre_bad)
    _summ("other accepted", pre_other)
    print()

    print("[accepted_improvement] higher = template helped more")
    _summ("first spikes", imp_first)
    _summ("second spikes", imp_second)
    _summ("all bad spikes", imp_bad)
    _summ("other accepted", imp_other)
    print()

    print("[post/pre ratio]       lower = bigger fractional improvement")
    _summ("first spikes", ratio_first)
    _summ("second spikes", ratio_second)
    _summ("all bad spikes", ratio_bad)
    _summ("other accepted", ratio_other)
    print()

    print("First 20 bad pairs:")
    for k, i in enumerate(bad_pair_idx[:20], start=1):
        r1 = order[i]
        r2 = order[i + 1]
        dt = int(t_main_sorted[i + 1] - t_main_sorted[i])
        print(
            f"{k:2d}: rows=({r1:4d}, {r2:4d}) dt={dt:2d}  "
            f"post=({accepted_post_score[r1]:.4f}, {accepted_post_score[r2]:.4f})  "
            f"impr=({accepted_improvement[r1]:.4f}, {accepted_improvement[r2]:.4f})  "
            f"bin=({int(accepted_best_bin[r1])}, {int(accepted_best_bin[r2])})  "
            f"lag=({int(accepted_best_lag[r1])}, {int(accepted_best_lag[r2])})"
        )

    if show_plots:
        # Histogram of residual scores
        plt.figure(figsize=(8, 4))
        bins = 40
        if post_other.size:
            plt.hist(post_other, bins=bins, alpha=0.5, label="other accepted")
        if post_bad.size:
            plt.hist(post_bad, bins=bins, alpha=0.5, label="bad-ISI spikes")
        plt.xlabel("accepted_post_score (weighted residual RMS)")
        plt.ylabel("count")
        plt.title("Bad-ISI spikes vs other accepted spikes")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

        # Improvement scatter
        plt.figure(figsize=(8, 4))
        x = np.arange(accepted_main_times.size)
        plt.scatter(x[other_rows], accepted_improvement[other_rows], s=10, alpha=0.35, label="other accepted")
        plt.scatter(x[bad_all_rows], accepted_improvement[bad_all_rows], s=30, alpha=0.9, label="bad-ISI spikes")
        plt.xlabel("accepted spike index")
        plt.ylabel("accepted_improvement = pre - post")
        plt.title("How much did the short-bank fit help?")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    return {
        "bad_pair_idx": bad_pair_idx,
        "first_rows": first_rows,
        "second_rows": second_rows,
        "bad_all_rows": bad_all_rows,
        "other_rows": other_rows,
        "post_first": post_first,
        "post_second": post_second,
        "post_bad": post_bad,
        "post_other": post_other,
        "pre_first": pre_first,
        "pre_second": pre_second,
        "pre_bad": pre_bad,
        "pre_other": pre_other,
        "imp_first": imp_first,
        "imp_second": imp_second,
        "imp_bad": imp_bad,
        "imp_other": imp_other,
        "ratio_first": ratio_first,
        "ratio_second": ratio_second,
        "ratio_bad": ratio_bad,
        "ratio_other": ratio_other,
    }


# Example:
fitdiag = inspect_bad_isi_residual_scores(dbg, isi_min=10, isi_max=30, show_plots=True)

# %%
# ==== Mean EI overlay for low-improvement vs high-improvement accepted spikes ====

import numpy as np
import matplotlib.pyplot as plt

def plot_ei_by_accepted_improvement(
    dbg,
    raw_data_for_ei=None,
    ei_positions_for_plot=None,
    low_thr=30.0,
    high_thr=40.0,
    window=None,
):
    """
    Build mean EIs from accepted spikes split by accepted_improvement:
      - low group:  accepted_improvement < low_thr
      - high group: accepted_improvement > high_thr

    Uses accepted_main_times (canonical main-channel times) and mean EI.
    """

    if dbg is None:
        raise ValueError("dbg is None")

    if raw_data_for_ei is None:
        if "raw_orig" in globals():
            raw_data_for_ei = raw_orig
        elif "raw_mod" in globals():
            raw_data_for_ei = raw_mod
        else:
            raise RuntimeError("No raw array found. Pass raw_data_for_ei explicitly.")

    if ei_positions_for_plot is None:
        if "ei_positions" in globals():
            ei_positions_for_plot = ei_positions
        else:
            raise RuntimeError("No ei_positions found. Pass ei_positions_for_plot explicitly.")

    # Prefer committed result if present; otherwise preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg.")

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    accepted_improvement = np.asarray(unit["accepted_improvement"], dtype=np.float32)
    main_ch = int(unit["main_ch"])
    detect_ch = int(unit["detect_ch"])

    if accepted_main_times.size != accepted_improvement.size:
        raise RuntimeError("accepted_main_times / accepted_improvement mismatch")

    if window is None:
        if "long_window" in unit:
            window = tuple(unit["long_window"])
        else:
            window = (-40, 80)
    pre, post = int(window[0]), int(window[1])

    low_mask = accepted_improvement < float(low_thr)
    high_mask = accepted_improvement > float(high_thr)

    low_times = accepted_main_times[low_mask]
    high_times = accepted_main_times[high_mask]

    print("=" * 100)
    print("EI BY ACCEPTED_IMPROVEMENT")
    print("=" * 100)
    print(f"detect_ch                    : {detect_ch}")
    print(f"main_ch                      : {main_ch}")
    print(f"n accepted total             : {accepted_main_times.size}")
    print(f"low threshold                : improvement < {low_thr}")
    print(f"high threshold               : improvement > {high_thr}")
    print(f"n low-improvement spikes     : {low_times.size}")
    print(f"n high-improvement spikes    : {high_times.size}")

    if low_times.size == 0:
        raise RuntimeError(f"No spikes with accepted_improvement < {low_thr}")
    if high_times.size == 0:
        raise RuntimeError(f"No spikes with accepted_improvement > {high_thr}")

    all_ch = np.arange(raw_data_for_ei.shape[1], dtype=np.int32)

    sn_low, valid_low = extract_snippets_fast_ram(
        raw_data=raw_data_for_ei,
        spike_times=low_times,
        window=(pre, post),
        selected_channels=all_ch,
    )
    sn_high, valid_high = extract_snippets_fast_ram(
        raw_data=raw_data_for_ei,
        spike_times=high_times,
        window=(pre, post),
        selected_channels=all_ch,
    )

    sn_low = sn_low.astype(np.float32, copy=False)
    sn_high = sn_high.astype(np.float32, copy=False)
    valid_low = np.asarray(valid_low, dtype=np.int64)
    valid_high = np.asarray(valid_high, dtype=np.int64)

    if sn_low.shape[2] == 0:
        raise RuntimeError("No valid snippets for low-improvement group")
    if sn_high.shape[2] == 0:
        raise RuntimeError("No valid snippets for high-improvement group")

    ei_low = sn_low.mean(axis=2).astype(np.float32)
    ei_high = sn_high.mean(axis=2).astype(np.float32)

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_ei_waveforms.plot_ei_waveforms(
        [ei_high, ei_low],
        ei_positions_for_plot,
        ref_channel=main_ch,
        scale=70.0,
        box_height=1.0,
        box_width=50.0,
        ax=ax,
        colors=["black", "red"],
    )
    ax.set_title(
        f"Accepted spike EI overlay | high improvement > {high_thr} (black, n={valid_high.size}) "
        f"vs low improvement < {low_thr} (red, n={valid_low.size})"
    )
    plt.tight_layout()
    plt.show()

    return {
        "low_times": low_times,
        "high_times": high_times,
        "valid_low": valid_low,
        "valid_high": valid_high,
        "ei_low": ei_low,
        "ei_high": ei_high,
    }


# Example:
ei_imp = plot_ei_by_accepted_improvement(dbg, raw_data_for_ei=raw_orig, low_thr=10.0, high_thr=20.0)

# %%
# ==== Plot bad-ISI pairs from debug_one_channel_pp_loop result ====

import numpy as np
import matplotlib.pyplot as plt

def plot_bad_isi_pairs_from_debug(
    dbg,
    raw_data_for_plot=None,
    n_pairs=10,
    isi_min=10,
    isi_max=30,
    n_channels_to_show=5,
    window=None,
    use_raw="raw_mod",   # just for title
):
    """
    Plot adjacent accepted spike pairs whose ISI is in [isi_min, isi_max] samples.

    One bad pair per row.
    Columns = main channel + next-largest channels by p2p (default total 5).
    Overlays the two snippets in each subplot.

    Parameters
    ----------
    dbg : dict
        Output of debug_one_channel_pp_loop(...)
    raw_data_for_plot : ndarray [T, C]
        Usually raw_mod. If None, tries global raw_mod, then raw_orig.
    n_pairs : int
        Max number of bad ISI pairs to plot
    isi_min, isi_max : int
        ISI window in samples to flag as violation-like pair
    n_channels_to_show : int
        Number of channels to plot per pair (default 5)
    window : tuple or None
        Snippet window to extract. If None, uses unit_record["long_window"] if present,
        else falls back to (-40, 80).
    use_raw : str
        Label only, for title.
    """

    if raw_data_for_plot is None:
        if "raw_mod" in globals():
            raw_data_for_plot = raw_mod
        elif "raw_orig" in globals():
            raw_data_for_plot = raw_orig
        else:
            raise RuntimeError("No raw array found. Pass raw_data_for_plot explicitly.")

    if dbg is None:
        raise ValueError("dbg is None")

    # Prefer committed result if present; otherwise preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg (preview_res/commit_res).")

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    main_ch = int(unit["main_ch"])

    if accepted_main_times.size < 2:
        print("Not enough accepted spikes to form pairs.")
        return

    # Find adjacent bad pairs in sorted main times
    order = np.argsort(accepted_main_times)
    t_sorted = accepted_main_times[order]
    d = np.diff(t_sorted)

    bad_idx = np.where((d >= int(isi_min)) & (d <= int(isi_max)))[0]
    if bad_idx.size == 0:
        print(f"No adjacent accepted spike pairs with ISI in [{isi_min}, {isi_max}] samples.")
        return

    bad_idx = bad_idx[:int(n_pairs)]
    pair_times = [(int(t_sorted[i]), int(t_sorted[i + 1]), int(d[i])) for i in bad_idx]

    # Choose plotting channels from provisional EI
    if dbg.get("lh_template_bank", None) is not None:
        ei_ref = np.asarray(dbg["lh_template_bank"]["provisional_ei"], dtype=np.float32)
    else:
        raise RuntimeError("dbg['lh_template_bank'] missing; cannot choose largest channels.")

    p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
    order_ch = np.argsort(p2p)[::-1]

    chs = [main_ch]
    for ch in order_ch:
        ch = int(ch)
        if ch not in chs:
            chs.append(ch)
        if len(chs) >= int(n_channels_to_show):
            break
    chs = np.asarray(chs, dtype=np.int32)

    # Window
    if window is None:
        if "long_window" in unit:
            window = tuple(unit["long_window"])
        else:
            window = (-40, 80)
    pre, post = int(window[0]), int(window[1])

    # Extract snippets for both spikes in each pair
    n_rows = len(pair_times)
    n_cols = len(chs)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    t_axis = np.arange(pre, post + 1)

    for r, (t0, t1, dt) in enumerate(pair_times):
        spike_times_pair = np.array([t0, t1], dtype=np.int64)
        sn, valid = extract_snippets_fast_ram(
            raw_data=raw_data_for_plot,
            spike_times=spike_times_pair,
            window=(pre, post),
            selected_channels=chs,
        )
        sn = sn.astype(np.float32, copy=False)

        if sn.shape[2] != 2:
            for c in range(n_cols):
                axes[r, c].text(0.5, 0.5, "edge drop", ha="center", va="center", transform=axes[r, c].transAxes)
                axes[r, c].axis("off")
            continue

        for c, ch in enumerate(chs):
            ax = axes[r, c]
            y0 = sn[c, :, 0]
            y1 = sn[c, :, 1]

            ax.plot(t_axis, y0, linewidth=1.2, alpha=0.9, label=f"{t0}")
            ax.plot(t_axis, y1, linewidth=1.2, alpha=0.9, label=f"{t1}")
            ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.5)

            if r == 0:
                title = f"ch {int(ch)}"
                if int(ch) == main_ch:
                    title += " (main)"
                ax.set_title(title, fontsize=10)

            if c == 0:
                ax.set_ylabel(f"pair {r+1}\nΔ={dt}", fontsize=9)

            ax.grid(True, alpha=0.2)

    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Bad ISI pairs on ch {int(unit['detect_ch'])} | main={main_ch} | "
        f"{use_raw} | ISI in [{isi_min}, {isi_max}] samples",
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    print("Plotted pairs:")
    for i, (t0, t1, dt) in enumerate(pair_times, start=1):
        print(f"  {i:2d}: {t0}  {t1}   dt={dt}")


# ---- example usage ----
# dbg24 = debug_one_channel_pp_loop(24, commit=False, show_plots=False)
plot_bad_isi_pairs_from_debug(dbg, raw_data_for_plot=raw_mod, n_pairs=10)

# %%
# ==== Build mean EIs from first-vs-second spikes in bad ISI pairs and overlay them ====

import numpy as np
import matplotlib.pyplot as plt

def plot_bad_isi_pair_ei_overlay(
    dbg,
    raw_data_for_plot=None,
    isi_min=10,
    isi_max=30,
    n_pairs=10,
    window=None,
    ei_positions_for_plot=None,
):
    """
    From debug_one_channel_pp_loop result `dbg`:
      - find adjacent accepted spike pairs with ISI in [isi_min, isi_max]
      - take the first spike from each pair -> mean EI
      - take the second spike from each pair -> mean EI
      - overlay the two mean EIs

    Uses accepted_main_times (canonical main-channel times).
    EI is computed with MEAN, not median.
    """

    if raw_data_for_plot is None:
        if "raw_mod" in globals():
            raw_data_for_plot = raw_mod
        elif "raw_orig" in globals():
            raw_data_for_plot = raw_orig
        else:
            raise RuntimeError("No raw array found. Pass raw_data_for_plot explicitly.")

    if ei_positions_for_plot is None:
        if "ei_positions" in globals():
            ei_positions_for_plot = ei_positions
        else:
            raise RuntimeError("No ei_positions found. Pass ei_positions_for_plot explicitly.")

    if dbg is None:
        raise ValueError("dbg is None")

    # Prefer committed result if present; otherwise preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg.")

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    detect_ch = int(unit["detect_ch"])
    main_ch = int(unit["main_ch"])

    if accepted_main_times.size < 2:
        print("Not enough accepted spikes to form pairs.")
        return None

    if window is None:
        if "long_window" in unit:
            window = tuple(unit["long_window"])
        else:
            window = (-40, 80)
    pre, post = int(window[0]), int(window[1])

    # Find adjacent bad ISI pairs
    order = np.argsort(accepted_main_times)
    t_sorted = accepted_main_times[order]
    d = np.diff(t_sorted)

    bad_idx = np.where((d >= int(isi_min)) & (d <= int(isi_max)))[0]
    if bad_idx.size == 0:
        print(f"No adjacent accepted spike pairs with ISI in [{isi_min}, {isi_max}] samples.")
        return None

    bad_idx = bad_idx[:int(n_pairs)]
    first_times = t_sorted[bad_idx].astype(np.int64)
    second_times = t_sorted[bad_idx + 1].astype(np.int64)
    pair_dts = d[bad_idx].astype(np.int64)

    all_ch = np.arange(raw_data_for_plot.shape[1], dtype=np.int32)

    # Extract snippets for first spikes
    sn1, valid1 = extract_snippets_fast_ram(
        raw_data=raw_data_for_plot,
        spike_times=first_times,
        window=(pre, post),
        selected_channels=all_ch,
    )
    sn1 = sn1.astype(np.float32, copy=False)

    # Extract snippets for second spikes
    sn2, valid2 = extract_snippets_fast_ram(
        raw_data=raw_data_for_plot,
        spike_times=second_times,
        window=(pre, post),
        selected_channels=all_ch,
    )
    sn2 = sn2.astype(np.float32, copy=False)

    if sn1.shape[2] == 0 or sn2.shape[2] == 0:
        print("No valid snippets extracted.")
        return None

    # Keep only times that survived edge checks in both groups
    set1 = {int(t) for t in valid1.tolist()}
    set2 = {int(t) for t in valid2.tolist()}

    keep_pairs = []
    for t1, t2, dt in zip(first_times.tolist(), second_times.tolist(), pair_dts.tolist()):
        if (int(t1) in set1) and (int(t2) in set2):
            keep_pairs.append((int(t1), int(t2), int(dt)))

    if len(keep_pairs) == 0:
        print("All bad pairs were dropped by edge checks.")
        return None

    first_keep = np.array([p[0] for p in keep_pairs], dtype=np.int64)
    second_keep = np.array([p[1] for p in keep_pairs], dtype=np.int64)

    sn1, valid1 = extract_snippets_fast_ram(
        raw_data=raw_data_for_plot,
        spike_times=first_keep,
        window=(pre, post),
        selected_channels=all_ch,
    )
    sn2, valid2 = extract_snippets_fast_ram(
        raw_data=raw_data_for_plot,
        spike_times=second_keep,
        window=(pre, post),
        selected_channels=all_ch,
    )
    sn1 = sn1.astype(np.float32, copy=False)
    sn2 = sn2.astype(np.float32, copy=False)

    ei1 = sn1.mean(axis=2).astype(np.float32)
    ei2 = sn2.mean(axis=2).astype(np.float32)

    ref_ch = main_ch

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_ei_waveforms.plot_ei_waveforms(
        [ei1, ei2],
        ei_positions_for_plot,
        ref_channel=ref_ch,
        scale=70.0,
        box_height=1.0,
        box_width=50.0,
        ax=ax,
        colors=["black", "red"],
    )
    ax.set_title(
        f"Bad-ISI pair EI overlay | detect_ch={detect_ch} main_ch={main_ch} | "
        f"first spikes (black, n={sn1.shape[2]}) vs second spikes (red, n={sn2.shape[2]})"
    )
    plt.tight_layout()
    plt.show()

    print("Pairs used:")
    for i, (t1, t2, dt) in enumerate(keep_pairs, start=1):
        print(f"  {i:2d}: {t1}  {t2}   dt={dt}")

    return {
        "first_times": first_keep,
        "second_times": second_keep,
        "ei_first": ei1,
        "ei_second": ei2,
        "pairs_used": keep_pairs,
    }


# ---- example ----
out_bad = plot_bad_isi_pair_ei_overlay(dbg, raw_data_for_plot=raw_mod, n_pairs=63)

# %%
# ==== Distance from last valid separator for bad-ISI spikes vs rest of accepted unit ====

import numpy as np
import matplotlib.pyplot as plt

def analyze_bad_isi_distance_to_last_valid_separator(
    dbg,
    isi_min=10,
    isi_max=30,
    show_plots=True,
):
    """
    Uses dbg from debug_one_channel_pp_loop(...).

    For the last valid separation (main_path_split_path):
      - computes signed distance from separator for each accepted spike
      - identifies bad-ISI spikes from accepted_main_times
      - compares first-in-pair, second-in-pair, all bad-pair spikes, and all other accepted spikes

    Signed distance is defined so that larger positive values mean "deeper inside
    the accepted side of the split". The uncertainty threshold at this split is:

        margin = margin_k * pooled_sd

    and spikes with signed_distance < margin are "uncertain" at this split.
    """

    if dbg is None:
        raise ValueError("dbg is None")

    # -------- pull objects --------
    pp_tree_state = dbg.get("pp_tree_state", None)
    final_left_state = dbg.get("final_left_state", None)

    if pp_tree_state is None or final_left_state is None:
        raise RuntimeError("dbg is missing pp_tree_state or final_left_state")

    # Prefer committed record if present, else preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg")

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    accepted_detect_times = np.asarray(unit["accepted_detect_times"], dtype=np.int64)

    if accepted_main_times.size < 2:
        raise RuntimeError("Not enough accepted spikes")

    if accepted_main_times.size != accepted_detect_times.size:
        raise RuntimeError("accepted_main_times / accepted_detect_times mismatch")

    full_times = np.asarray(pp_tree_state["full_times"], dtype=np.int64)
    margin_k = float(final_left_state["margin_k"])
    main_path_split_path = str(final_left_state["main_path_split_path"])
    main_leaf_path = str(final_left_state["main_leaf_path"])

    # -------- helper: get node by path --------
    def get_node_by_path(tree, path):
        if path == "root":
            return tree
        parts = path.split(".")
        if parts[0] != "root":
            raise ValueError(f"Unexpected path: {path}")
        node = tree
        for branch in parts[1:]:
            if node["type"] != "split":
                raise RuntimeError(f"Path {path} hits a leaf too early")
            node = node["left_child"] if branch == "L" else node["right_child"]
        return node

    tree = pp_tree_state["tree"]
    score_node = get_node_by_path(tree, main_path_split_path)

    node_idx = np.asarray(score_node["idx"], dtype=np.int64)
    z_full = np.asarray(score_node["best_proj_full"], dtype=np.float32)
    thr = float(score_node["best_full"]["thr"])
    pooled_sd = float(score_node["best_full"]["pooled_sd"])
    margin = float(margin_k) * float(pooled_sd)
    leftlike_is_proj_left = bool(score_node["leftlike_is_proj_left"])

    # Which branch of this score node contains the chosen main leaf?
    if main_leaf_path == main_path_split_path:
        raise RuntimeError("main_leaf_path equals main_path_split_path; expected leaf below split")
    suffix = main_leaf_path[len(main_path_split_path):]
    if not suffix.startswith("."):
        raise RuntimeError("main_leaf_path is not below main_path_split_path")
    chosen_branch = suffix.split(".")[1]  # 'L' or 'R'

    # signed distance: positive = deeper inside accepted side of this split
    accepted_side_is_low = (
        (chosen_branch == "L" and leftlike_is_proj_left) or
        (chosen_branch == "R" and (not leftlike_is_proj_left))
    )

    def signed_distance_from_thr(z):
        z = np.asarray(z, dtype=np.float32)
        return (thr - z) if accepted_side_is_low else (z - thr)

    # map accepted_detect_times -> positions in score_node's z_full
    pos_map = {int(gidx): pos for pos, gidx in enumerate(node_idx.tolist())}
    full_time_map = {int(t): i for i, t in enumerate(full_times.tolist())}

    accepted_full_idx = []
    missing_times = []
    for t in accepted_detect_times.tolist():
        if int(t) not in full_time_map:
            missing_times.append(int(t))
            continue
        fi = full_time_map[int(t)]
        if fi not in pos_map:
            missing_times.append(int(t))
            continue
        accepted_full_idx.append(fi)

    if len(missing_times) > 0:
        raise RuntimeError(
            f"{len(missing_times)} accepted_detect_times were not found in last valid split node. "
            f"First few: {missing_times[:10]}"
        )

    accepted_full_idx = np.asarray(accepted_full_idx, dtype=np.int64)
    accepted_node_pos = np.array([pos_map[int(i)] for i in accepted_full_idx], dtype=np.int64)
    accepted_z = z_full[accepted_node_pos]
    accepted_signed_dist = signed_distance_from_thr(accepted_z)

    # -------- identify bad ISI pairs from accepted_main_times --------
    order = np.argsort(accepted_main_times)
    t_main_sorted = accepted_main_times[order]
    t_detect_sorted = accepted_detect_times[order]

    d = np.diff(t_main_sorted)
    bad_pair_idx = np.where((d >= int(isi_min)) & (d <= int(isi_max)))[0]

    if bad_pair_idx.size == 0:
        print(f"No bad ISI pairs in [{isi_min}, {isi_max}] samples.")
        return None

    first_detect = t_detect_sorted[bad_pair_idx]
    second_detect = t_detect_sorted[bad_pair_idx + 1]

    bad_detect_all = np.unique(np.concatenate([first_detect, second_detect]).astype(np.int64))

    # map detect times to accepted row
    acc_time_to_row = {int(t): i for i, t in enumerate(accepted_detect_times.tolist())}

    first_rows = np.array([acc_time_to_row[int(t)] for t in first_detect], dtype=np.int64)
    second_rows = np.array([acc_time_to_row[int(t)] for t in second_detect], dtype=np.int64)
    bad_all_rows = np.array([acc_time_to_row[int(t)] for t in bad_detect_all], dtype=np.int64)

    all_rows = np.arange(accepted_detect_times.size, dtype=np.int64)
    other_rows = np.setdiff1d(all_rows, bad_all_rows, assume_unique=False)

    dist_first = accepted_signed_dist[first_rows]
    dist_second = accepted_signed_dist[second_rows]
    dist_bad_all = accepted_signed_dist[bad_all_rows]
    dist_other = accepted_signed_dist[other_rows]

    # -------- report --------
    print("=" * 100)
    print("LAST VALID SEPARATOR ANALYSIS")
    print("=" * 100)
    print(f"main_path_split_path : {main_path_split_path}")
    print(f"main_leaf_path       : {main_leaf_path}")
    print(f"chosen_branch        : {chosen_branch}")
    print(f"leftlike_is_proj_left: {leftlike_is_proj_left}")
    print(f"accepted_side_is_low : {accepted_side_is_low}")
    print(f"thr                  : {thr:.6f}")
    print(f"pooled_sd            : {pooled_sd:.6f}")
    print(f"margin_k             : {margin_k:.6f}")
    print(f"UNCERTAIN THRESHOLD  : signed_distance < {margin:.6f}")
    print()

    def summarize(name, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            print(f"{name:<22s} n=0")
            return
        print(
            f"{name:<22s} n={arr.size:4d}  "
            f"mean={arr.mean():9.4f}  median={np.median(arr):9.4f}  "
            f"std={arr.std(ddof=1) if arr.size > 1 else np.nan:9.4f}  "
            f"frac_uncertain={(arr < margin).mean():7.4f}"
        )

    summarize("first spikes in pairs", dist_first)
    summarize("second spikes in pairs", dist_second)
    summarize("all bad-pair spikes", dist_bad_all)
    summarize("other accepted spikes", dist_other)

    print()
    print(f"n bad pairs          : {bad_pair_idx.size}")
    print(f"n unique bad spikes  : {bad_all_rows.size}")
    print(f"n other accepted     : {other_rows.size}")

    # show a few pair details
    print("\nFirst 15 bad pairs:")
    for k, i in enumerate(bad_pair_idx[:15], start=1):
        r1 = acc_time_to_row[int(t_detect_sorted[i])]
        r2 = acc_time_to_row[int(t_detect_sorted[i + 1])]
        print(
            f"{k:2d}: main_times=({int(t_main_sorted[i])}, {int(t_main_sorted[i+1])}) "
            f"dt={int(d[i]):2d}  "
            f"detect_times=({int(t_detect_sorted[i])}, {int(t_detect_sorted[i+1])})  "
            f"signed_dist=({accepted_signed_dist[r1]:.4f}, {accepted_signed_dist[r2]:.4f})"
        )

    if show_plots:
        # Histogram
        plt.figure(figsize=(8, 4))
        bins = 40
        if dist_other.size:
            plt.hist(dist_other, bins=bins, alpha=0.5, label="other accepted")
        if dist_bad_all.size:
            plt.hist(dist_bad_all, bins=bins, alpha=0.5, label="bad-pair spikes")
        plt.axvline(margin, linestyle="--", linewidth=1.5, label=f"uncertain threshold = {margin:.3f}")
        plt.xlabel("signed distance from last valid separator")
        plt.ylabel("count")
        plt.title("Bad-pair spikes vs rest of unit")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

        # Scatter by accepted spike order
        plt.figure(figsize=(10, 4))
        plt.scatter(other_rows, dist_other, s=10, alpha=0.5, label="other accepted")
        plt.scatter(bad_all_rows, dist_bad_all, s=18, alpha=0.9, label="bad-pair spikes")
        plt.axhline(margin, linestyle="--", linewidth=1.5, label=f"uncertain threshold = {margin:.3f}")
        plt.xlabel("accepted spike index")
        plt.ylabel("signed distance from last valid separator")
        plt.title("Bad-pair spikes in separator-distance space")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    return {
        "main_path_split_path": main_path_split_path,
        "main_leaf_path": main_leaf_path,
        "thr": thr,
        "pooled_sd": pooled_sd,
        "margin_k": margin_k,
        "uncertain_threshold": margin,
        "accepted_signed_dist": accepted_signed_dist,
        "accepted_detect_times": accepted_detect_times,
        "accepted_main_times": accepted_main_times,
        "bad_pair_idx_sorted": bad_pair_idx,
        "first_rows": first_rows,
        "second_rows": second_rows,
        "bad_all_rows": bad_all_rows,
        "other_rows": other_rows,
        "dist_first": dist_first,
        "dist_second": dist_second,
        "dist_bad_all": dist_bad_all,
        "dist_other": dist_other,
    }


# ---- usage ----
sep24 = analyze_bad_isi_distance_to_last_valid_separator(dbg, isi_min=10, isi_max=30, show_plots=True)

# %%
# ==== BL/TR-style bulk support on the last valid split ====

import numpy as np
import matplotlib.pyplot as plt

def plot_bltr_support_last_valid_split(
    dbg,
    raw_data_for_plot=None,
    ei_positions_for_plot=None,
    isi_min=10,
    isi_max=30,
    p2p_thr=30.0,
    max_channels=40,
    min_channels=10,
    rel_time_mask=0.10,
    baseline_n=5,
    chunk_size=256,
):
    """
    BL/TR-style support analysis on the LAST VALID split.

    "Left" = branch of the last valid split that contains the prospective unit.
    "Right" = sibling branch.

    For each spike in both groups:
      - left_support  = mean cosine to all other spikes in LEFT
      - right_support = mean cosine to all other spikes in RIGHT

    Then plots:
      1) EI overlay (left vs right)
      2) support scatter: x=left_support, y=right_support

    Also highlights bad-ISI spikes from the accepted unit if present in dbg.
    """

    if raw_data_for_plot is None:
        if "raw_mod" in globals():
            raw_data_for_plot = raw_mod
        elif "raw_orig" in globals():
            raw_data_for_plot = raw_orig
        else:
            raise RuntimeError("No raw array found. Pass raw_data_for_plot explicitly.")

    if ei_positions_for_plot is None:
        if "ei_positions" in globals():
            ei_positions_for_plot = ei_positions
        else:
            raise RuntimeError("No ei_positions found. Pass ei_positions_for_plot explicitly.")

    if dbg is None:
        raise ValueError("dbg is None")

    pp_tree_state = dbg.get("pp_tree_state", None)
    final_left_state = dbg.get("final_left_state", None)
    if pp_tree_state is None or final_left_state is None:
        raise RuntimeError("dbg is missing pp_tree_state or final_left_state")

    # Prefer committed unit if present; otherwise preview
    unit = None
    if dbg.get("commit_res", None) is not None and dbg["commit_res"].get("status", None) == "ok":
        unit = dbg["commit_res"]["unit_record"]
    elif dbg.get("preview_res", None) is not None and dbg["preview_res"].get("status", None) == "ok":
        unit = dbg["preview_res"]["unit_record"]
    else:
        raise RuntimeError("No usable unit_record found in dbg")

    accepted_main_times = np.asarray(unit["accepted_main_times"], dtype=np.int64)
    accepted_detect_times = np.asarray(unit["accepted_detect_times"], dtype=np.int64)

    full_times = np.asarray(pp_tree_state["full_times"], dtype=np.int64)
    tree = pp_tree_state["tree"]
    main_path_split_path = str(final_left_state["main_path_split_path"])
    main_leaf_path = str(final_left_state["main_leaf_path"])
    window_pp = tuple(final_left_state["window_pp"])
    main_ch = int(unit["main_ch"])
    detect_ch = int(unit["detect_ch"])

    # ---------------- helpers ----------------
    def get_node_by_path(tree_, path):
        if path == "root":
            return tree_
        parts = path.split(".")
        if parts[0] != "root":
            raise ValueError(f"Unexpected path: {path}")
        node = tree_
        for branch in parts[1:]:
            if node["type"] != "split":
                raise RuntimeError(f"Path {path} hits a leaf too early")
            node = node["left_child"] if branch == "L" else node["right_child"]
        return node

    def choose_sig_channels_from_ei(ei, p2p_thr=30.0, max_channels=40, min_channels=10):
        p2p = ei.max(axis=1) - ei.min(axis=1)
        sel = np.where(p2p >= float(p2p_thr))[0]
        if sel.size > int(max_channels):
            sel = np.argsort(p2p)[-int(max_channels):]
        elif sel.size < int(min_channels):
            sel = np.argsort(p2p)[-int(min_channels):]
        return np.sort(sel.astype(np.int32)), p2p.astype(np.float32)

    def build_feature_matrix(snips, ei_left_sel, ei_right_sel, rel_time_mask=0.10, baseline_n=5):
        """
        snips: [Csel, T, N]
        returns X: [N, D] normalized features, plus mask
        """
        sn = np.asarray(snips, dtype=np.float32).copy()

        # baseline correction per channel/spike
        b = sn[:, :int(baseline_n), :].mean(axis=1, keepdims=True)
        sn -= b

        # informative time mask from either EI
        maxabs_left = np.max(np.abs(ei_left_sel), axis=1, keepdims=True) + 1e-12
        maxabs_right = np.max(np.abs(ei_right_sel), axis=1, keepdims=True) + 1e-12
        mask = (
            (np.abs(ei_left_sel) >= float(rel_time_mask) * maxabs_left) |
            (np.abs(ei_right_sel) >= float(rel_time_mask) * maxabs_right)
        )

        mask_flat = mask.reshape(-1)
        X = sn.transpose(2, 0, 1).reshape(sn.shape[2], -1)[:, mask_flat]

        # row normalize
        X = X.astype(np.float32, copy=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X /= norms
        return X, mask

    def mean_support_same_group(X, chunk_size=256):
        """
        mean cosine to all OTHER spikes in same group
        """
        N = X.shape[0]
        if N == 1:
            return np.full(1, np.nan, dtype=np.float32)

        out = np.empty(N, dtype=np.float32)
        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)
            sims = X[start:end] @ X.T                      # [chunk, N]
            row_sums = sims.sum(axis=1)
            # subtract self cosine = 1
            diag_idx = np.arange(start, end) - start
            row_sums -= 1.0
            out[start:end] = row_sums / float(N - 1)
        return out

    def mean_support_cross_group(X_query, X_ref, chunk_size=256):
        """
        mean cosine to all spikes in other group
        """
        M = X_query.shape[0]
        N = X_ref.shape[0]
        if N == 0:
            return np.full(M, np.nan, dtype=np.float32)

        out = np.empty(M, dtype=np.float32)
        for start in range(0, M, int(chunk_size)):
            end = min(start + int(chunk_size), M)
            sims = X_query[start:end] @ X_ref.T           # [chunk, N]
            out[start:end] = sims.mean(axis=1)
        return out

    # ---------------- find last valid split and define left/right ----------------
    score_node = get_node_by_path(tree, main_path_split_path)

    if main_leaf_path == main_path_split_path:
        raise RuntimeError("main_leaf_path equals main_path_split_path; expected leaf below split")

    suffix = main_leaf_path[len(main_path_split_path):]
    if not suffix.startswith("."):
        raise RuntimeError("main_leaf_path is not below main_path_split_path")
    chosen_branch = suffix.split(".")[1]  # "L" or "R"

    if chosen_branch == "L":
        idx_left = np.asarray(score_node["idx_left"], dtype=np.int64)
        idx_right = np.asarray(score_node["idx_right"], dtype=np.int64)
        ei_left = np.asarray(score_node["summary_left"]["ei"], dtype=np.float32)
        ei_right = np.asarray(score_node["summary_right"]["ei"], dtype=np.float32)
    elif chosen_branch == "R":
        idx_left = np.asarray(score_node["idx_right"], dtype=np.int64)
        idx_right = np.asarray(score_node["idx_left"], dtype=np.int64)
        ei_left = np.asarray(score_node["summary_right"]["ei"], dtype=np.float32)
        ei_right = np.asarray(score_node["summary_left"]["ei"], dtype=np.float32)
    else:
        raise RuntimeError(f"Unexpected chosen_branch {chosen_branch}")

    left_times = full_times[idx_left].astype(np.int64, copy=False)
    right_times = full_times[idx_right].astype(np.int64, copy=False)

    print("=" * 100)
    print("LAST VALID SPLIT BL/TR SUPPORT")
    print("=" * 100)
    print(f"main_path_split_path : {main_path_split_path}")
    print(f"main_leaf_path       : {main_leaf_path}")
    print(f"chosen_branch        : {chosen_branch}  (this is LEFT / prospective unit)")
    print(f"left_n               : {left_times.size}")
    print(f"right_n              : {right_times.size}")

    # ---------------- significant channels ----------------
    ch_left, p2p_left = choose_sig_channels_from_ei(
        ei_left, p2p_thr=p2p_thr, max_channels=max_channels, min_channels=min_channels
    )
    ch_right, p2p_right = choose_sig_channels_from_ei(
        ei_right, p2p_thr=p2p_thr, max_channels=max_channels, min_channels=min_channels
    )
    ch_union = np.union1d(ch_left, ch_right).astype(np.int32)

    if main_ch not in ch_union:
        ch_union = np.sort(np.unique(np.concatenate([ch_union, np.array([main_ch], dtype=np.int32)])))

    print(f"sig channels left    : {len(ch_left)}")
    print(f"sig channels right   : {len(ch_right)}")
    print(f"union channels       : {len(ch_union)}")

    ei_left_sel = ei_left[ch_union, :]
    ei_right_sel = ei_right[ch_union, :]

    # EI overlay
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_ei_waveforms.plot_ei_waveforms(
        [ei_left, ei_right],
        ei_positions_for_plot,
        ref_channel=main_ch,
        scale=70.0,
        box_height=1.0,
        box_width=50.0,
        ax=ax,
        colors=["black", "red"],
    )
    ax.set_title(
        f"Last valid split EIs | left=prospective unit (black, n={left_times.size}) "
        f"vs right (red, n={right_times.size})"
    )
    plt.tight_layout()
    plt.show()

    # ---------------- extract snippets on union channels ----------------
    sn_left, valid_left = extract_snippets_fast_ram(
        raw_data=raw_data_for_plot,
        spike_times=left_times,
        window=window_pp,
        selected_channels=ch_union,
    )
    sn_right, valid_right = extract_snippets_fast_ram(
        raw_data=raw_data_for_plot,
        spike_times=right_times,
        window=window_pp,
        selected_channels=ch_union,
    )
    sn_left = sn_left.astype(np.float32, copy=False)
    sn_right = sn_right.astype(np.float32, copy=False)

    if valid_left.size != left_times.size or valid_right.size != right_times.size:
        raise RuntimeError("Unexpected edge drops while extracting left/right snippets")

    # ---------------- build normalized feature matrices ----------------
    X_left, mask_time = build_feature_matrix(
        sn_left, ei_left_sel, ei_right_sel,
        rel_time_mask=rel_time_mask, baseline_n=baseline_n
    )
    X_right, _ = build_feature_matrix(
        sn_right, ei_left_sel, ei_right_sel,
        rel_time_mask=rel_time_mask, baseline_n=baseline_n
    )

    n_features = X_left.shape[1]
    print(f"masked feature count : {n_features}")

    # ---------------- compute BL/TR bulk-style support ----------------
    left_support_left = mean_support_same_group(X_left, chunk_size=chunk_size)
    right_support_left = mean_support_cross_group(X_left, X_right, chunk_size=chunk_size)

    left_support_right = mean_support_cross_group(X_right, X_left, chunk_size=chunk_size)
    right_support_right = mean_support_same_group(X_right, chunk_size=chunk_size)

    # ---------------- identify bad-ISI accepted spikes to highlight ----------------
    bad_first_detect = np.array([], dtype=np.int64)
    bad_second_detect = np.array([], dtype=np.int64)

    if accepted_main_times.size >= 2:
        order_acc = np.argsort(accepted_main_times)
        t_main_sorted = accepted_main_times[order_acc]
        t_detect_sorted = accepted_detect_times[order_acc]
        d_acc = np.diff(t_main_sorted)

        bad_pair_idx = np.where((d_acc >= int(isi_min)) & (d_acc <= int(isi_max)))[0]
        if bad_pair_idx.size > 0:
            bad_first_detect = t_detect_sorted[bad_pair_idx].astype(np.int64)
            bad_second_detect = t_detect_sorted[bad_pair_idx + 1].astype(np.int64)

    left_time_to_row = {int(t): i for i, t in enumerate(left_times.tolist())}

    bad_first_rows = np.array(
        [left_time_to_row[int(t)] for t in bad_first_detect.tolist() if int(t) in left_time_to_row],
        dtype=np.int64
    )
    bad_second_rows = np.array(
        [left_time_to_row[int(t)] for t in bad_second_detect.tolist() if int(t) in left_time_to_row],
        dtype=np.int64
    )

    # ---------------- scatter ----------------
    plt.figure(figsize=(7, 6))
    plt.scatter(
        left_support_left, right_support_left,
        s=8, alpha=0.35, label=f"left group (n={left_times.size})"
    )
    plt.scatter(
        left_support_right, right_support_right,
        s=8, alpha=0.35, label=f"right group (n={right_times.size})"
    )

    if bad_first_rows.size > 0:
        plt.scatter(
            left_support_left[bad_first_rows],
            right_support_left[bad_first_rows],
            s=90,
            marker="o",
            facecolors="none",
            edgecolors="magenta",
            linewidths=1.8,
            zorder=6,
            label=f"bad-ISI first spikes (n={bad_first_rows.size})"
        )

    if bad_second_rows.size > 0:
        plt.scatter(
            left_support_left[bad_second_rows],
            right_support_left[bad_second_rows],
            s=70,
            marker="x",
            color="black",
            linewidths=1.8,
            zorder=7,
            label=f"bad-ISI second spikes (n={bad_second_rows.size})"
        )

    lims = plt.axis()
    lo = min(lims[0], lims[2])
    hi = max(lims[1], lims[3])
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.6)

    plt.xlabel("mean cosine support to LEFT group")
    plt.ylabel("mean cosine support to RIGHT group")
    plt.title(
        f"BL/TR-style bulk support on last valid split | ch={detect_ch}, main={main_ch}"
    )
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    print(f"bad-ISI first spikes found in left group   : {bad_first_rows.size}")
    print(f"bad-ISI second spikes found in left group  : {bad_second_rows.size}")

    return {
        "main_path_split_path": main_path_split_path,
        "main_leaf_path": main_leaf_path,
        "chosen_branch": chosen_branch,
        "left_times": left_times,
        "right_times": right_times,
        "ei_left": ei_left,
        "ei_right": ei_right,
        "sig_channels_left": ch_left,
        "sig_channels_right": ch_right,
        "sig_channels_union": ch_union,
        "feature_mask": mask_time,
        "left_support_left": left_support_left,
        "right_support_left": right_support_left,
        "left_support_right": left_support_right,
        "right_support_right": right_support_right,
        "bad_first_rows": bad_first_rows,
        "bad_second_rows": bad_second_rows,
        "bad_first_detect": bad_first_detect,
        "bad_second_detect": bad_second_detect,
    }


# ---- usage ----
bltr = plot_bltr_support_last_valid_split(dbg, raw_data_for_plot=raw_mod)

# %% [markdown]
# ## Single channel

# %% [markdown]
# ### identify amplitude of top 1k spikes on channels

# %%
# === Cell 1: skyline scan across all channels ===
# mean amplitude of 1000 deepest local minima in first 1M samples
# optional: exclude deferred spikes on each channel

import numpy as np
import matplotlib.pyplot as plt

data_matrix = raw_mod
n_samples_scan = 1_000_000
top_k = 1000

# -----------------------------
# Optional same-channel exclusion toggle
# If enabled, ignore minima within +/- exclude_radius_samples
# of channel_state[ch]["exclude_times"] on each channel.
# -----------------------------
exclude_deferred_in_skyline = True
exclude_radius_samples = 20

if data_matrix.ndim != 2:
    raise ValueError(f"Expected [T, C] array, got shape {data_matrix.shape}")

if exclude_deferred_in_skyline and "channel_state" not in globals():
    raise RuntimeError(
        "exclude_deferred_in_skyline=True but channel_state is not in globals()."
    )

T_total, n_channels = data_matrix.shape
T_scan = min(n_samples_scan, T_total)

mean_amp_topk = np.full(n_channels, np.nan, dtype=np.float32)
n_minima = np.zeros(n_channels, dtype=np.int32)
n_minima_kept = np.zeros(n_channels, dtype=np.int32)

for ch in range(n_channels):
    x = data_matrix[:T_scan, ch].astype(np.float32, copy=False)
    idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    n_minima[ch] = idx.size
    if idx.size == 0:
        continue

    if exclude_deferred_in_skyline:
        excl = None
        if channel_state.get(ch) is not None:
            excl = channel_state[ch].get("exclude_times", None)

        near = _times_near_exclusions(
            idx.astype(np.int64),
            excl,
            int(exclude_radius_samples),
        )
        idx = idx[~near]

    n_minima_kept[ch] = idx.size
    if idx.size == 0:
        continue

    vals = x[idx]
    k = min(top_k, vals.size)
    smallest_k = np.partition(vals, k - 1)[:k]
    mean_amp_topk[ch] = smallest_k.mean()

order = np.argsort(mean_amp_topk)

print(f"Scanned first {T_scan:,} samples across {n_channels} channels")
print(f"Exclude deferred in skyline: {exclude_deferred_in_skyline}")
if exclude_deferred_in_skyline:
    print(f"Exclude radius: +/- {exclude_radius_samples} samples")

print(f"\nTop 150 channels by strongest mean negative amplitude:")
for rank, ch in enumerate(order[:150], start=1):
    kept_txt = f"{n_minima_kept[ch]:7d}" if exclude_deferred_in_skyline else f"{n_minima[ch]:7d}"
    print(
        f"{rank:2d}. ch {ch:3d}  "
        f"mean_top{min(top_k, n_minima_kept[ch] if exclude_deferred_in_skyline else n_minima[ch])} = {mean_amp_topk[ch]:7.2f}   "
        f"n_minima = {n_minima[ch]:7d}"
        + (f"   kept = {kept_txt}" if exclude_deferred_in_skyline else "")
    )

plt.figure(figsize=(14, 4))
plt.plot(np.arange(n_channels), mean_amp_topk, linewidth=1.2)
plt.xlabel("Channel")
plt.ylabel(f"Mean amplitude of {top_k} deepest local minima (ADC)")
title = f"First {T_scan:,} samples | raw_mod | per-channel mean of {top_k} deepest local minima"
if exclude_deferred_in_skyline:
    title += " | deferred excluded"
plt.title(title)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## check if spikes on channel from skyline are available

# %%
# === Cell 2: LH-anchored full pool setup ===
# Builds lh_pp_state from scratch.
# Keep skyline scan cell unchanged.

import numpy as np
import matplotlib.pyplot as plt

from lighthouse_utils import find_valley_and_times

# -----------------------------
# Choose channel to test
# mode:
#   "rank"   -> use skyline ranking
#   "manual" -> use explicit channel ID
# -----------------------------
channel_select_mode = "manual"   # "rank" or "manual"

channel_rank = 4               # used if mode == "rank"
manual_channel_id = 241         # used if mode == "manual"

channel_order = np.argsort(mean_amp_topk)   # ascending: most negative first

if channel_select_mode == "rank":
    if not (0 <= channel_rank < len(channel_order)):
        raise ValueError(
            f"channel_rank={channel_rank} out of range for {len(channel_order)} channels"
        )
    top_ch = int(channel_order[channel_rank])
    channel_label = f"skyline-ranked channel #{channel_rank}"
elif channel_select_mode == "manual":
    if not (0 <= manual_channel_id < data_matrix.shape[1]):
        raise ValueError(
            f"manual_channel_id={manual_channel_id} out of range for "
            f"{data_matrix.shape[1]} channels"
        )
    top_ch = int(manual_channel_id)

    # optional: recover its current skyline rank for printout
    rank_lookup = np.empty_like(channel_order)
    rank_lookup[channel_order] = np.arange(len(channel_order))
    manual_rank = int(rank_lookup[top_ch])

    channel_label = f"manual channel {top_ch} (current skyline rank {manual_rank})"
else:
    raise ValueError(
        f"Unknown channel_select_mode={channel_select_mode!r}. Use 'rank' or 'manual'."
    )

# -----------------------------
# Config
# -----------------------------
data_matrix = raw_mod
fs = 20_000
duration_sec = 5 * 60
stop = min(duration_sec * fs, data_matrix.shape[0])

# Shorter window for everything downstream
window_pp = (-20, 50)

max_valley_count = 1000

# Valley finder params
bin_width = 10.0
valley_bins = 5
min_valid_count = 300
ratio_base = 3
ratio_step = 100
ratio_floor = 2
ratio_cap = 10


# -----------------------------
# Fallback pool params
# If left_count is too small, force a 10k amplitude-ranked pool:
#   most-negative 5k -> faux "left"
#   next 5k         -> faux "right"
# -----------------------------
fallback_enable = True
fallback_left_min_count = 500
fallback_total_keep = 10_000

print(f"Testing {channel_label}")
print(f"mean_topk amplitude = {mean_amp_topk[top_ch]:.2f}")
print(f"Scanning first {stop:,} samples ({stop/fs/60:.1f} min)")
print(f"Projection/EI window = {window_pp}")

# -----------------------------
# Optional same-channel exclusion toggle
# If enabled, skip minima near deferred/excluded spikes on this channel
# before valley finding AND before fallback pool construction.
# -----------------------------
exclude_deferred_current_channel = True
exclude_radius_samples = 20

# -----------------------------
# Channel-local exclusions (optional)
# -----------------------------
exclude_times_current = np.array([], dtype=np.int64)
if exclude_deferred_current_channel:
    if "channel_state" not in globals():
        raise RuntimeError(
            "exclude_deferred_current_channel=True but channel_state is not in globals()."
        )
    if channel_state.get(int(top_ch)) is not None:
        exclude_times_current = np.asarray(
            channel_state[int(top_ch)].get("exclude_times", []),
            dtype=np.int64,
        )

print(f"Exclude deferred on current channel: {exclude_deferred_current_channel}")
print(f"exclude_times_current n = {exclude_times_current.size}")

# -----------------------------
# Find LH valley
# -----------------------------
step1 = find_valley_and_times_excluding(
    data_matrix,
    top_ch,
    window=window_pp,
    start=0,
    stop=stop,
    bin_width=bin_width,
    valley_bins=valley_bins,
    min_valid_count=min_valid_count,
    ratio_base=ratio_base,
    ratio_step=ratio_step,
    ratio_floor=ratio_floor,
    ratio_cap=ratio_cap,
    exclude_times=exclude_times_current if exclude_deferred_current_channel else None,
    exclude_radius=exclude_radius_samples,
)

print(f"valley accepted   : {step1['accepted']}")
print(f"left_count        : {step1['left_count']}")
print(f"valley_count      : {step1['valley_count']}")
print(f"valley_low/high   : {step1['valley_low']} / {step1['valley_high']}")

# Pager-like histogram
if step1["amp_hist_counts"].size > 0:
    centers = 0.5 * (step1["amp_hist_edges"][:-1] + step1["amp_hist_edges"][1:])
    widths = np.diff(step1["amp_hist_edges"])

    plt.figure(figsize=(7, 4))
    plt.bar(centers, step1["amp_hist_counts"], width=widths, align="center", alpha=0.7)
    if step1["valley_low"] is not None:
        plt.axvline(step1["valley_low"], linestyle="--", linewidth=1.2, label=f"valley_low = {step1['valley_low']:.1f}")
    if step1["valley_high"] is not None:
        plt.axvline(step1["valley_high"], linestyle="--", linewidth=1.2, label=f"valley_high = {step1['valley_high']:.1f}")
    plt.xlabel(f"Amplitude at local minima on channel {top_ch} (ADC)")
    plt.ylabel("Count")
    plt.title(f"Channel {top_ch} | pager-like amplitude histogram")
    plt.legend()
    plt.tight_layout()
    plt.show()

all_times = np.asarray(step1.get("all_times", []), dtype=np.int64)
all_vals = np.asarray(step1.get("all_vals", []), dtype=np.float32)

# Separate the strict LH verdict from whether we can still build a PP pool
valley_status = "accepted" if step1["accepted"] else "not_accepted"
valley_reason = None
if not step1["accepted"]:
    valley_reason = (
        f"left_count={step1['left_count']} too small relative to "
        f"valley_count={step1['valley_count']}"
    )
    print(f"\nValley not accepted on channel {top_ch}: {valley_reason}")

# defaults
pool_status = "ok"
pool_reason = None
pool_builder = "valley_match"
fallback_used = False
fallback_info = None

left_times = np.array([], dtype=np.int64)
left_vals = np.array([], dtype=np.float32)
right_times = np.array([], dtype=np.int64)
right_vals = np.array([], dtype=np.float32)
full_times = np.array([], dtype=np.int64)
full_side = np.array([], dtype=np.int32)

def _build_fallback_amp_split(all_times, all_vals, total_keep):
    """
    Take the `total_keep` most negative minima and split evenly by amplitude.
    First half (most negative) becomes faux-left, second half faux-right.
    """
    all_times = np.asarray(all_times, dtype=np.int64)
    all_vals = np.asarray(all_vals, dtype=np.float32)

    if all_times.size < 2:
        return None

    keep_n = min(int(total_keep), int(all_times.size))
    half_n = keep_n // 2
    keep_n = 2 * half_n  # force equal halves
    if half_n == 0:
        return None

    amp_order = np.argsort(all_vals)[:keep_n]  # ascending => most negative first
    keep_times_amp = all_times[amp_order]
    keep_vals_amp = all_vals[amp_order]

    left_times_amp = keep_times_amp[:half_n]
    left_vals_amp = keep_vals_amp[:half_n]

    right_times_amp = keep_times_amp[half_n:]
    right_vals_amp = keep_vals_amp[half_n:]

    li = np.argsort(left_times_amp)
    ri = np.argsort(right_times_amp)

    return {
        "keep_n": int(keep_n),
        "half_n": int(half_n),
        "left_times": left_times_amp[li].astype(np.int64, copy=False),
        "left_vals": left_vals_amp[li].astype(np.float32, copy=False),
        "right_times": right_times_amp[ri].astype(np.int64, copy=False),
        "right_vals": right_vals_amp[ri].astype(np.float32, copy=False),
        "amp_cut_left_max": float(np.max(left_vals_amp)) if left_vals_amp.size else np.nan,
        "amp_cut_right_min": float(np.min(right_vals_amp)) if right_vals_amp.size else np.nan,
    }

# -----------------------------
# Build PP pool
# -----------------------------
if all_times.size == 0:
    pool_status = "empty_all_times"
    pool_reason = "all_times is empty."
    print(f"\nCannot build PP pool on channel {top_ch}: {pool_reason}")

else:
    use_fallback = False

    if fallback_enable and int(step1["left_count"]) < int(fallback_left_min_count):
        use_fallback = True
        pool_builder = "fallback_topk_amp_split"
        pool_reason = (
            f"left_count={int(step1['left_count'])} < "
            f"fallback_left_min_count={int(fallback_left_min_count)}"
        )

    if use_fallback:
        fb = _build_fallback_amp_split(
            all_times=all_times,
            all_vals=all_vals,
            total_keep=fallback_total_keep,
        )

        if fb is None:
            pool_status = "fallback_failed"
            print(f"\nFallback failed on channel {top_ch}: not enough minima to split.")
        else:
            fallback_used = True
            fallback_info = fb

            left_times = fb["left_times"]
            left_vals = fb["left_vals"]
            right_times = fb["right_times"]
            right_vals = fb["right_vals"]

            full_times = np.concatenate([left_times, right_times]).astype(np.int64)
            full_side = np.concatenate([
                np.zeros(left_times.size, dtype=np.int32),
                np.ones(right_times.size, dtype=np.int32),
            ])

            pool_status = "ok"

            print("\nUsing fallback amplitude-ranked split:")
            print(f"  reason        : {pool_reason}")
            print(f"  kept minima   : {fb['keep_n']}")
            print(f"  faux-left     : {left_times.size}")
            print(f"  faux-right    : {right_times.size}")
            print(f"  left max amp  : {fb['amp_cut_left_max']:.1f}")
            print(f"  right min amp : {fb['amp_cut_right_min']:.1f}")

    else:
        if step1["valley_low"] is None:
            pool_status = "missing_valley_low"
            pool_reason = "valley_low is None."
            print(f"\nCannot build PP pool on channel {top_ch}: {pool_reason}")

        else:
            valley_low = float(step1["valley_low"])

            # -----------------------------
            # Build full matched LH pool:
            # all left spikes + equally many strongest right spikes
            # -----------------------------
            left_mask = all_vals < valley_low
            left_times = all_times[left_mask]
            left_vals = all_vals[left_mask]

            right_mask = all_vals >= valley_low
            right_times_all = all_times[right_mask]
            right_vals_all = all_vals[right_mask]

            right_order = np.argsort(right_vals_all)   # ascending => strongest negative first
            n_left = left_times.size
            right_keep = right_order[:min(n_left, right_times_all.size)]

            right_times = right_times_all[right_keep]
            right_vals = right_vals_all[right_keep]

            print("\nMatched set sizes:")
            print(f"  left  : {left_times.size}")
            print(f"  right : {right_times.size}")

            if step1["valley_count"] > max_valley_count:
                print(
                    f"  note  : valley_count={step1['valley_count']} > max_valley_count={max_valley_count}"
                )

            if left_times.size == 0 or right_times.size == 0:
                pool_status = "empty_matched_side"
                pool_reason = "One side is empty after matching left/right counts."
                print(f"\nCannot build usable PP pool on channel {top_ch}: {pool_reason}")
            else:
                full_times = np.concatenate([left_times, right_times]).astype(np.int64)
                full_side = np.concatenate([
                    np.zeros(left_times.size, dtype=np.int32),   # 0 = original-left
                    np.ones(right_times.size, dtype=np.int32),   # 1 = original-right
                ])
                print(f"Full pool size for recursive PP: {full_times.size}")

# -----------------------------
# Save state for recursion / inspection
# -----------------------------
lh_pp_state = {
    "top_ch": top_ch,
    "channel_rank": channel_rank,
    "stop": stop,
    "window_pp": window_pp,
    "step1": step1,

    # strict LH decision from find_valley_and_times
    "valley_status": valley_status,
    "valley_reason": valley_reason,

    # actual usability for recursive PP
    "pool_status": pool_status,
    "pool_reason": pool_reason,
    "pool_builder": pool_builder,
    "fallback_used": bool(fallback_used),
    "fallback_info": fallback_info,

    "left_times": left_times,
    "left_vals": left_vals,
    "right_times": right_times,
    "right_vals": right_vals,
    "full_times": full_times,
    "full_side": full_side,   # 0 = original-left, 1 = original-right
}

print(
    f"\nlh_pp_state rebuilt. valley_status = {valley_status} | "
    f"pool_status = {pool_status} | pool_builder = {pool_builder}"
)

# %%
# === Cell 3: recursive PP with subset discovery + full-node classification ===
# Discovery uses a capped subset.
# Classification uses the full node pool, extracting only selected channels.
# Recursion only continues into majority-left-origin children.
# Stores enough info for final trusted-left / extra-left / uncertain labeling.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import plot_ei_waveforms as pew

from axolotl_utils_ram import extract_snippets_fast_ram
from collision_utils import select_template_channels

# -----------------------------
# Basic checks
# -----------------------------
if "lh_pp_state" not in globals():
    raise RuntimeError("lh_pp_state not found. Run Cell 2 first.")

if lh_pp_state.get("pool_status", "ok") != "ok":
    raise RuntimeError(
        f"Cell 2 did not produce a usable PP pool: "
        f"pool_status={lh_pp_state.get('pool_status')} | "
        f"reason={lh_pp_state.get('pool_reason')}"
    )

full_times = np.asarray(lh_pp_state["full_times"], dtype=np.int64)
full_side = np.asarray(lh_pp_state["full_side"], dtype=np.int32)

if full_times.size == 0:
    raise RuntimeError("Cell 2 produced an empty PP pool.")

data_matrix = raw_mod
ei_pos = ei_positions

anchor_ch = int(lh_pp_state["top_ch"])
window_pp = tuple(lh_pp_state["window_pp"])
full_times = np.asarray(lh_pp_state["full_times"], dtype=np.int64)
full_side = np.asarray(lh_pp_state["full_side"], dtype=np.int32)

N0 = full_times.size
if full_side.size != N0:
    raise RuntimeError("Mismatch in lh_pp_state full_times/full_side.")

print(f"Recursive PP starting from top channel {anchor_ch}")
print(f"Initial full pool size: {N0}")
print(f"Window used: {window_pp}")

# -----------------------------
# Tuning knobs
# -----------------------------
max_depth = 5
min_node_n = 200
min_child_n = 120

# Discovery subset cap per ancestry side inside each node
discover_max_per_side = 5000

# Projection-pursuit search
n_search_pcs = 6
n_random_dirs = 2500
hist_bins = 120
smooth_sigma = 2.0
rng_seed = 123

# Augmented-direction bank
amp_pc1_weight = 0.50
n_biased_tail_dirs = 16
biased_tail_scale = 0.08

# Split acceptance thresholds (applied to FULL-node classification result)
min_score = 0.90
min_depth = 0.22
min_sep   = 1.15
min_group_frac = 0.03
target_frac = 0.08

# Feature construction
p2p_thr = 30.0
max_channels = 80
min_channels = 10

# Recursion policy
left_recurse_thresh = 0.50   # recurse only if orig-left fraction > 0.50
plot_every_split = True

rng = np.random.default_rng(rng_seed)

# -----------------------------
# Helpers
# -----------------------------
def _local_maxima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    if y[0] > y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] > y[i - 1]) and (y[i] >= y[i + 1]):
            out.append(i)
    if y[-1] > y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=int)

def _local_minima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n < 2:
        return np.array([], dtype=int)
    if y[0] < y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] < y[i - 1]) and (y[i] <= y[i + 1]):
            out.append(i)
    if y[-1] < y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=int)

def normalize_direction(w):
    w = np.asarray(w, dtype=np.float32).ravel()
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)

def robust_zscore_1d_fit(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32).ravel()
    med = float(np.median(x)) if x.size else 0.0
    mad = float(np.median(np.abs(x - med))) if x.size else 0.0
    scale = float(1.4826 * max(mad, eps))
    z = (x - med) / scale
    return z.astype(np.float32), med, scale

def robust_zscore_1d_apply(x, med, scale):
    x = np.asarray(x, dtype=np.float32).ravel()
    return ((x - float(med)) / float(scale)).astype(np.float32)

def robust_zscore_cols_fit(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float32)
    med = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - med[None, :]), axis=0).astype(np.float32)
    scale = (1.4826 * np.maximum(mad, eps)).astype(np.float32)
    Z = (X - med[None, :]) / scale[None, :]
    return Z.astype(np.float32), med, scale

def robust_zscore_cols_apply(X, med, scale):
    X = np.asarray(X, dtype=np.float32)
    med = np.asarray(med, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    return ((X - med[None, :]) / scale[None, :]).astype(np.float32)

def detect_row_trough_feature(sn_sel, ref_row, center_idx):
    sn_sel = np.asarray(sn_sel, dtype=np.float32)
    lo = max(0, int(center_idx) - 1)
    hi = min(sn_sel.shape[1] - 1, int(center_idx) + 1)
    amp = -sn_sel[int(ref_row), lo:hi + 1, :].min(axis=0)
    return np.asarray(amp, dtype=np.float32)

def build_biased_direction_bank(
    n_aug,
    n_pcs,
    rng,
    amp_pc1_weight=0.5,
    n_tail_dirs=16,
    tail_scale=0.08,
):
    """
    Candidate directions in augmented feature space:
      feature 0 = detect-channel amplitude
      feature 1 = PC1
      feature 2.. = PC2, PC3, ...

    IMPORTANT: no amplitude-only direction here.
    """
    dirs = []

    if n_pcs >= 1:
        # PC1 only
        w = np.zeros(n_aug, dtype=np.float32)
        w[1] = 1.0
        dirs.append(("pc1_only", normalize_direction(w)))

        # amplitude + 0.5 * PC1
        w = np.zeros(n_aug, dtype=np.float32)
        w[0] = 1.0
        w[1] = float(amp_pc1_weight)
        dirs.append(("amp_plus_pc1", normalize_direction(w)))

        # amplitude + 0.5 * PC1 + tiny random tail on the rest
        n_tail = max(0, n_aug - 2)
        for i in range(int(n_tail_dirs)):
            w = np.zeros(n_aug, dtype=np.float32)
            w[0] = 1.0
            w[1] = float(amp_pc1_weight)
            if n_tail > 0:
                w[2:] = rng.uniform(
                    -float(tail_scale),
                    float(tail_scale),
                    size=n_tail
                ).astype(np.float32)
            dirs.append((f"amp_plus_pc1_tail_{i:02d}", normalize_direction(w)))

    return dirs

def random_unit_directions(n_dirs, dim, rng):
    W = rng.normal(size=(n_dirs, dim)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W

def score_projection_1d(z, bins=120, sigma=2.0,
                        min_child_n=120, min_group_frac=0.03, target_frac=0.08):
    """
    On a FIXED 1D projection z, find the best valley threshold.
    Returns the best split for that 1D array.
    """
    z = np.asarray(z, dtype=np.float32).ravel()
    N = z.size

    hist, edges = np.histogram(z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = gaussian_filter1d(hist.astype(np.float32), sigma=sigma)

    peaks = _local_maxima(hs)
    valleys = _local_minima(hs)

    best = None

    for v in valleys:
        left_peaks = peaks[peaks < v]
        right_peaks = peaks[peaks > v]
        if left_peaks.size == 0 or right_peaks.size == 0:
            continue

        pl = left_peaks[-1]
        pr = right_peaks[0]

        valley_h = float(hs[v])
        peak_l_h = float(hs[pl])
        peak_r_h = float(hs[pr])
        min_peak_h = min(peak_l_h, peak_r_h)

        if min_peak_h <= 0:
            continue

        thr = float(centers[v])
        mask_left = z <= thr
        n_left = int(mask_left.sum())
        n_right = int(N - n_left)
        n_minor = min(n_left, n_right)
        frac_minor = n_minor / float(N)

        if n_left < int(min_child_n) or n_right < int(min_child_n):
            continue
        if frac_minor < float(min_group_frac):
            continue

        depth = 1.0 - (valley_h / (min_peak_h + 1e-12))

        zl = z[mask_left]
        zr = z[~mask_left]
        mu_l = float(zl.mean())
        mu_r = float(zr.mean())
        sd_l = float(zl.std(ddof=1)) if zl.size > 1 else 1e-12
        sd_r = float(zr.std(ddof=1)) if zr.size > 1 else 1e-12
        pooled_sd = np.sqrt(0.5 * (sd_l * sd_l + sd_r * sd_r) + 1e-12)
        sep = abs(mu_r - mu_l) / pooled_sd

        size_bonus = 1.0 - np.exp(-frac_minor / float(target_frac))
        score = depth * sep * size_bonus

        cand = {
            "score": float(score),
            "thr": thr,
            "depth": float(depth),
            "sep": float(sep),
            "size_bonus": float(size_bonus),
            "n_left": n_left,
            "n_right": n_right,
            "n_minor": n_minor,
            "frac_minor": float(frac_minor),
            "hist": hist,
            "hist_smooth": hs,
            "edges": edges,
            "centers": centers,
            "valley_idx": int(v),
            "mask_left": mask_left,
            "mu_l": mu_l,
            "mu_r": mu_r,
            "sd_l": sd_l,
            "sd_r": sd_r,
            "pooled_sd": float(pooled_sd),
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    return best

def extract_snips_for_times(times, selected_channels):
    sn, valid_times = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=np.asarray(times, dtype=np.int64),
        window=window_pp,
        selected_channels=np.asarray(selected_channels, dtype=np.int32),
    )
    sn = sn.astype(np.float32, copy=False)
    valid_times = np.asarray(valid_times, dtype=np.int64)

    # In this workflow, all times should already be window-valid because Cell 2
    # used the same window to build the pool. If not, fail loudly.
    if valid_times.size != np.asarray(times).size:
        raise RuntimeError(
            f"Unexpected edge-drop during snippet extraction: "
            f"requested {np.asarray(times).size}, got {valid_times.size}. "
            f"Cell 2 and Cell 3 windows may be inconsistent."
        )
    return sn

def build_node_discovery_subset(idx_node):
    """
    Choose capped subset for discovery inside this node:
    up to discover_max_per_side original-left spikes and up to discover_max_per_side original-right spikes.
    """
    idx_node = np.asarray(idx_node, dtype=np.int64)
    side = full_side[idx_node]

    idx_left = idx_node[side == 0]
    idx_right = idx_node[side == 1]

    if idx_left.size > discover_max_per_side:
        keep = np.sort(rng.choice(idx_left.size, size=discover_max_per_side, replace=False))
        idx_left = idx_left[keep]
    if idx_right.size > discover_max_per_side:
        keep = np.sort(rng.choice(idx_right.size, size=discover_max_per_side, replace=False))
        idx_right = idx_right[keep]

    idx_disc = np.concatenate([idx_left, idx_right])
    return np.asarray(idx_disc, dtype=np.int64)

def summarize_child(idx_child, times_child):
    idx_child = np.asarray(idx_child, dtype=np.int64)
    times_child = np.asarray(times_child, dtype=np.int64)

    n = idx_child.size
    n_orig_left = int(np.sum(full_side[idx_child] == 0))
    n_orig_right = int(np.sum(full_side[idx_child] == 1))
    frac_orig_left = n_orig_left / float(n) if n > 0 else np.nan
    frac_orig_right = n_orig_right / float(n) if n > 0 else np.nan

    if n > 0:
        sn_full = extract_snips_for_times(times_child, np.arange(data_matrix.shape[1], dtype=np.int32))
        if sn_full.shape[2] == 0:
            amp_anchor = np.nan
            ei_child = None
        else:
            ei_child = sn_full.mean(axis=2).astype(np.float32)
            amp_anchor = float(-ei_child[anchor_ch].min())
    else:
        amp_anchor = np.nan
        ei_child = None

    return {
        "n": int(n),
        "n_orig_left": n_orig_left,
        "n_orig_right": n_orig_right,
        "frac_orig_left": frac_orig_left,
        "frac_orig_right": frac_orig_right,
        "amp_anchor": amp_anchor,
        "majority_left": bool(frac_orig_left > left_recurse_thresh) if np.isfinite(frac_orig_left) else False,
        "ei": ei_child,
    }

def print_node_summary(node):
    sl = node["summary_left"]
    sr = node["summary_right"]

    print(f"\n[{node['path']}] accepted split")
    print(f"  N(full)              : {node['n_full']}")
    print(f"  N(discovery subset)  : {node['n_disc']}")
    print(f"  full score/depth/sep : {node['best_full']['score']:.3f} / {node['best_full']['depth']:.3f} / {node['best_full']['sep']:.3f}")
    print(f"  best direction       : {node.get('best_dir_name', 'unknown')}")
    print(f"  child sizes          : L={sl['n']}, R={sr['n']}")
    print(f"  left child (.L)      : orig-left={sl['n_orig_left']} ({100*sl['frac_orig_left']:.1f}%), "
          f"orig-right={sl['n_orig_right']} ({100*sl['frac_orig_right']:.1f}%), "
          f"anchor amp={sl['amp_anchor']:.2f}, recurse={sl['majority_left']}")
    print(f"  right child (.R)     : orig-left={sr['n_orig_left']} ({100*sr['frac_orig_left']:.1f}%), "
          f"orig-right={sr['n_orig_right']} ({100*sr['frac_orig_right']:.1f}%), "
          f"anchor amp={sr['amp_anchor']:.2f}, recurse={sr['majority_left']}")

def plot_split_diagnostics(node):
    best = node["best_full"]
    hist = best["hist"]
    hs = best["hist_smooth"]
    edges = best["edges"]
    centers = best["centers"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].bar(centers, hist, width=np.diff(edges), alpha=0.35, align="center")
    axes[0].plot(centers, hs, linewidth=2)
    axes[0].axvline(best["thr"], linestyle="--", linewidth=1.5)
    txt = (
        f"path={node['path']}\n"
        f"Nfull={node['n_full']}\n"
        f"Ndisc={node['n_disc']}\n"
        f"best={node.get('best_dir_name', '?')}\n"
        f"score={best['score']:.2f}\n"
        f"depth={best['depth']:.2f}\n"
        f"sep={best['sep']:.2f}\n"
        f"split={best['n_left']}/{best['n_right']}"
    )
    axes[0].text(
        0.02, 0.98, txt,
        transform=axes[0].transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15)
    )
    axes[0].set_title(f"Full-pool projection histogram | {node['path']}")
    axes[0].set_xlabel("projection")
    axes[0].set_ylabel("count")

    pcs_full = node["pcs_full"]
    mask_left = node["mask_leftlike"]
    mask_right = ~mask_left

    axes[1].scatter(pcs_full[mask_left, 0], pcs_full[mask_left, 1], s=5, alpha=0.35, label=f".L leftlike (n={mask_left.sum()})")
    axes[1].scatter(pcs_full[mask_right, 0], pcs_full[mask_right, 1], s=5, alpha=0.35, label=f".R other (n={mask_right.sum()})")
    axes[1].set_title(f"PC1 vs PC2 | {node['path']} | best={node.get('best_dir_name', '?')}")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.show()

    fig, ax = plt.subplots(figsize=(20, 12))
    ei_left = node["summary_left"]["ei"]
    ei_right = node["summary_right"]["ei"]
    if ei_left is not None and ei_right is not None:
        pew.plot_ei_waveforms(
            [ei_left, ei_right],
            ei_pos,
            ref_channel=anchor_ch,
            scale=70.0,
            box_height=1.0,
            box_width=50.0,
            ax=ax,
            colors=["black", "red"],
            alpha=[0.65, 0.85],
            linewidth=[0.7, 0.9],
        )
        ax.set_title(
            f"{node['path']} | EI overlay | "
            f"black=.L (amp@{anchor_ch}={node['summary_left']['amp_anchor']:.1f}), "
            f"red=.R (amp@{anchor_ch}={node['summary_right']['amp_anchor']:.1f})"
        )
        plt.tight_layout()
        plt.show()

def make_leaf(idx_leaf, depth, path, reason):
    return {
        "type": "leaf",
        "path": path,
        "depth": depth,
        "n": int(idx_leaf.size),
        "reason": reason,
        "idx": np.asarray(idx_leaf, dtype=np.int64),
    }

def maybe_recurse_child(idx_child, depth, path, summary_child):
    if not summary_child["majority_left"]:
        frac = 100.0 * summary_child["frac_orig_left"]
        return make_leaf(
            idx_child,
            depth,
            path,
            f"not_pursued_right_heavy (orig-left={frac:.1f}% <= {100*left_recurse_thresh:.1f}%)"
        )
    return try_split(idx_child, depth, path)

def try_split(idx_node, depth, path):
    idx_node = np.asarray(idx_node, dtype=np.int64)
    n_full = idx_node.size

    if n_full < min_node_n:
        return make_leaf(idx_node, depth, path, f"too_small_for_split (<{min_node_n})")

    # --------------------------------------------------
    # 1) Discovery subset for this node
    # --------------------------------------------------
    idx_disc = build_node_discovery_subset(idx_node)
    n_disc = idx_disc.size
    if n_disc < min_node_n:
        return make_leaf(idx_node, depth, path, f"discovery_subset_too_small (<{min_node_n})")

    disc_times = full_times[idx_disc]

    # Extract discovery snippets on ALL channels so we can build EI and choose channels
    sn_disc_full = extract_snips_for_times(disc_times, np.arange(data_matrix.shape[1], dtype=np.int32))
    if sn_disc_full.shape[2] == 0:
        return make_leaf(idx_node, depth, path, "no_valid_discovery_snippets")

    idx_disc_valid = idx_disc.copy()
    side_disc_valid = full_side[idx_disc_valid]

    # Discovery EI + selected channels
    ei_disc = sn_disc_full.mean(axis=2).astype(np.float32)

    selected_channels, _ = select_template_channels(
        ei_disc,
        p2p_thr=p2p_thr,
        max_n=max_channels,
        min_n=min_channels,
        force_include_main=True
    )
    selected_channels = np.asarray(selected_channels, dtype=int)

    # Always include the detection channel explicitly, because the added
    # amplitude feature is defined on this channel.
    if anchor_ch not in selected_channels:
        selected_channels = np.concatenate([selected_channels, np.array([anchor_ch], dtype=int)])

    anchor_row = int(np.where(selected_channels == anchor_ch)[0][0])

    # Features for discovery subset
    sn_disc_sel = sn_disc_full[selected_channels, :, :]
    X_disc = sn_disc_sel.transpose(2, 0, 1).reshape(sn_disc_sel.shape[2], -1).astype(np.float32)

    n_pcs = int(min(n_search_pcs, X_disc.shape[0], X_disc.shape[1]))
    if n_pcs < 2:
        return make_leaf(idx_node, depth, path, "not_enough_rank_for_pca")

    pca = PCA(n_components=n_pcs)
    pcs_disc = pca.fit_transform(X_disc).astype(np.float32)

    # Detect-channel amplitude feature (robust trough around the snippet center)
    center_idx = int(-window_pp[0])
    amp_disc = detect_row_trough_feature(sn_disc_sel, anchor_row, center_idx)

    # Robust normalization so amplitude and PC scores live on comparable scales
    pcs_disc_z, pc_med, pc_scale = robust_zscore_cols_fit(pcs_disc)
    amp_disc_z, amp_med, amp_scale = robust_zscore_1d_fit(amp_disc)

    # Augmented feature space: [amp_det, PC1, PC2, ...]
    feat_disc = np.concatenate([amp_disc_z[:, None], pcs_disc_z], axis=1).astype(np.float32)
    feature_names = ["amp_det"] + [f"PC{i+1}" for i in range(n_pcs)]
    n_aug = int(feat_disc.shape[1])

    # Deterministic bank first
    candidate_dirs = build_biased_direction_bank(
        n_aug=n_aug,
        n_pcs=n_pcs,
        rng=rng,
        amp_pc1_weight=amp_pc1_weight,
        n_tail_dirs=n_biased_tail_dirs,
        tail_scale=biased_tail_scale,
    )

    # Then fully random directions in the augmented space
    W_rand = random_unit_directions(n_random_dirs, n_aug, rng)
    for i in range(n_random_dirs):
        candidate_dirs.append((f"random_{i:04d}", W_rand[i]))

    best_disc = None
    best_w = None
    best_proj_disc = None
    best_dir_name = None

    for dir_name, w in candidate_dirs:
        z_disc = feat_disc @ w
        res = score_projection_1d(
            z_disc,
            bins=hist_bins,
            sigma=smooth_sigma,
            min_child_n=min_child_n,
            min_group_frac=min_group_frac,
            target_frac=target_frac,
        )
        if res is None:
            continue
        if (best_disc is None) or (res["score"] > best_disc["score"]):
            best_disc = res
            best_w = w.copy()
            best_proj_disc = z_disc.copy()
            best_dir_name = str(dir_name)

    if best_disc is None:
        return make_leaf(idx_node, depth, path, "no_valid_valley_split_on_discovery_subset")

    # --------------------------------------------------
    # 2) Classify FULL node along the discovered axis
    # --------------------------------------------------
    node_times = full_times[idx_node]

    # Discovery positions in full node order
    pos_map_full = {int(i): pos for pos, i in enumerate(idx_node.tolist())}

    # Project discovery subset
    z_full = np.empty(n_full, dtype=np.float32)
    pcs_full = np.empty((n_full, n_pcs), dtype=np.float32)          # raw PCs, for diagnostics
    feat_full = np.empty((n_full, n_aug), dtype=np.float32)         # augmented normalized features

    z_disc_best = feat_disc @ best_w
    for gidx, pcs_row, feat_row, zval in zip(idx_disc_valid.tolist(), pcs_disc, feat_disc, z_disc_best):
        pos = pos_map_full[int(gidx)]
        pcs_full[pos] = pcs_row
        feat_full[pos] = feat_row
        z_full[pos] = zval

    # Leftovers: extract only selected channels
    disc_set = set(idx_disc_valid.tolist())
    idx_extra = np.array([i for i in idx_node.tolist() if i not in disc_set], dtype=np.int64)
    extra_times = full_times[idx_extra]

    if idx_extra.size > 0:
        sn_extra_sel = extract_snips_for_times(extra_times, selected_channels)
        if sn_extra_sel.shape[2] > 0:
            X_extra = sn_extra_sel.transpose(2, 0, 1).reshape(sn_extra_sel.shape[2], -1).astype(np.float32)
            pcs_extra = pca.transform(X_extra).astype(np.float32)
            pcs_extra_z = robust_zscore_cols_apply(pcs_extra, pc_med, pc_scale)

            amp_extra = detect_row_trough_feature(sn_extra_sel, anchor_row, center_idx)
            amp_extra_z = robust_zscore_1d_apply(amp_extra, amp_med, amp_scale)

            feat_extra = np.concatenate([amp_extra_z[:, None], pcs_extra_z], axis=1).astype(np.float32)
            z_extra = feat_extra @ best_w

            for gidx, pcs_row, feat_row, zval in zip(idx_extra.tolist(), pcs_extra, feat_extra, z_extra):
                pos = pos_map_full[int(gidx)]
                pcs_full[pos] = pcs_row
                feat_full[pos] = feat_row
                z_full[pos] = zval

    # Best threshold on FULL node along the fixed discovered axis
    best_full = score_projection_1d(
        z_full,
        bins=hist_bins,
        sigma=smooth_sigma,
        min_child_n=min_child_n,
        min_group_frac=min_group_frac,
        target_frac=target_frac,
    )
    if best_full is None:
        return make_leaf(idx_node, depth, path, "no_valid_valley_split_on_full_node")

    if best_full["score"] < min_score:
        return make_leaf(idx_node, depth, path, f"score_too_low ({best_full['score']:.3f} < {min_score})")
    if best_full["depth"] < min_depth:
        return make_leaf(idx_node, depth, path, f"depth_too_low ({best_full['depth']:.3f} < {min_depth})")
    if best_full["sep"] < min_sep:
        return make_leaf(idx_node, depth, path, f"sep_too_low ({best_full['sep']:.3f} < {min_sep})")

    # --------------------------------------------------
    # 3) Decide which side is the "left-like" child
    #     using discovery subset ancestry UNDER THE FULL threshold
    # --------------------------------------------------
    thr_full = best_full["thr"]
    proj_left_mask_full = z_full <= thr_full

    disc_pos_in_full = np.array([pos_map_full[int(i)] for i in idx_disc_valid.tolist()], dtype=np.int64)
    z_disc_under_full = z_full[disc_pos_in_full]
    disc_mask_proj_left = z_disc_under_full <= thr_full

    if np.any(disc_mask_proj_left):
        frac_left_proj_left = np.mean(side_disc_valid[disc_mask_proj_left] == 0)
    else:
        frac_left_proj_left = -np.inf

    if np.any(~disc_mask_proj_left):
        frac_left_proj_right = np.mean(side_disc_valid[~disc_mask_proj_left] == 0)
    else:
        frac_left_proj_right = -np.inf

    if frac_left_proj_left >= frac_left_proj_right:
        # projection-left is the left-like child
        mask_leftlike = proj_left_mask_full
        leftlike_is_proj_left = True
    else:
        # projection-right is the left-like child
        mask_leftlike = ~proj_left_mask_full
        leftlike_is_proj_left = False

    mask_other = ~mask_leftlike

    # Reorder children so .L is the left-like branch
    idx_left = idx_node[mask_leftlike]
    idx_right = idx_node[mask_other]
    times_left = full_times[idx_left]
    times_right = full_times[idx_right]

    summary_left = summarize_child(idx_left, times_left)
    summary_right = summarize_child(idx_right, times_right)

    node = {
        "type": "split",
        "path": path,
        "depth": depth,
        "n_full": int(n_full),
        "n_disc": int(n_disc),
        "idx": idx_node,
        "idx_disc": idx_disc_valid,
        "selected_channels": selected_channels,
        "pca": pca,
        "best_w": best_w,
        "pcs_full": pcs_full,
        "feat_full": feat_full,
        "feature_names": feature_names,
        "best_dir_name": best_dir_name,
        "best_disc": best_disc,
        "best_full": best_full,
        "best_proj_full": z_full,
        "mask_leftlike": mask_leftlike,
        "leftlike_is_proj_left": leftlike_is_proj_left,
        "idx_left": idx_left,
        "idx_right": idx_right,
        "summary_left": summary_left,
        "summary_right": summary_right,
    }

    print_node_summary(node)
    if plot_every_split:
        plot_split_diagnostics(node)

    if depth >= max_depth:
        node["left_child"] = make_leaf(idx_left, depth + 1, path + ".L", f"max_depth_reached ({max_depth})")
        node["right_child"] = make_leaf(idx_right, depth + 1, path + ".R", f"max_depth_reached ({max_depth})")
        return node

    node["left_child"] = maybe_recurse_child(idx_left, depth + 1, path + ".L", summary_left)
    node["right_child"] = maybe_recurse_child(idx_right, depth + 1, path + ".R", summary_right)
    return node

def collect_leaves(node, out=None):
    if out is None:
        out = []
    if node["type"] == "leaf":
        out.append(node)
    else:
        collect_leaves(node["left_child"], out)
        collect_leaves(node["right_child"], out)
    return out

def print_leaf_summary(leaves):
    print("\nFinal leaves")
    for leaf in leaves:
        idx = np.asarray(leaf["idx"], dtype=np.int64)
        n = idx.size
        if n > 0:
            n_left = int(np.sum(full_side[idx] == 0))
            n_right = int(np.sum(full_side[idx] == 1))
            frac_left = n_left / float(n)
            summary = summarize_child(idx, full_times[idx])
            amp_anchor = summary["amp_anchor"]
        else:
            n_left = 0
            n_right = 0
            frac_left = np.nan
            amp_anchor = np.nan

        print(
            f"  {leaf['path']:12s} | n={leaf['n']:4d} | "
            f"orig-left={n_left:4d} ({100*frac_left if np.isfinite(frac_left) else np.nan:5.1f}%) | "
            f"orig-right={n_right:4d} | "
            f"anchor amp={amp_anchor:7.2f} | "
            f"reason={leaf['reason']}"
        )

# -----------------------------
# Run recursion
# -----------------------------
root_idx = np.arange(N0, dtype=np.int64)
pp_tree = try_split(root_idx, depth=0, path="root")
pp_leaves = collect_leaves(pp_tree)
print_leaf_summary(pp_leaves)

lh_pp_tree_state = {
    "anchor_ch": anchor_ch,
    "window_pp": window_pp,
    "left_recurse_thresh": left_recurse_thresh,
    "tree": pp_tree,
    "leaves": pp_leaves,
    "full_times": full_times,
    "full_side": full_side,
}

# %%
# === Cell 4: finalize labels + build LH-style template bank ===
# Produces:
#   final_left_state
#   lh_template_bank

import numpy as np
from axolotl_utils_ram import extract_snippets_fast_ram
from collision_utils import median_ei_adaptive

if "lh_pp_tree_state" not in globals():
    raise RuntimeError("lh_pp_tree_state not found. Run Cell 3 first.")

data_matrix = raw_mod
anchor_ch = int(lh_pp_tree_state["anchor_ch"])
window_pp = tuple(lh_pp_tree_state["window_pp"])
pp_tree = lh_pp_tree_state["tree"]
pp_leaves = lh_pp_tree_state["leaves"]
full_times = np.asarray(lh_pp_tree_state["full_times"], dtype=np.int64)
full_side = np.asarray(lh_pp_tree_state["full_side"], dtype=np.int32)

margin_k = 0.40   # uncertainty margin = margin_k * pooled_sd at each split on the main path
left_leaf_thresh = 0.50

# template bank settings
template_reducer = "median"   # "median" or "mean"
template_n_bins = 6
template_min_bin_size = 100

# -----------------------------
# Helpers
# -----------------------------
def extract_full_snips(times):
    sn, valid_times = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=np.asarray(times, dtype=np.int64),
        window=window_pp,
        selected_channels=np.arange(data_matrix.shape[1], dtype=np.int32),
    )
    sn = sn.astype(np.float32, copy=False)
    valid_times = np.asarray(valid_times, dtype=np.int64)
    if valid_times.size != np.asarray(times).size:
        raise RuntimeError(
            f"Unexpected edge-drop while building full snippets: "
            f"requested {np.asarray(times).size}, got {valid_times.size}. "
            f"Cell 2/3/4 windows may be inconsistent."
        )
    return sn

def summarize_leaf(idx_leaf):
    idx_leaf = np.asarray(idx_leaf, dtype=np.int64)
    n = idx_leaf.size
    n_left = int(np.sum(full_side[idx_leaf] == 0))
    n_right = int(np.sum(full_side[idx_leaf] == 1))
    frac_left = n_left / float(n) if n > 0 else np.nan

    if n > 0:
        sn = extract_full_snips(full_times[idx_leaf])
        ei = sn.mean(axis=2).astype(np.float32)
        amp_anchor = float(-ei[anchor_ch].min())
    else:
        ei = None
        amp_anchor = np.nan

    return {
        "n": int(n),
        "n_left": n_left,
        "n_right": n_right,
        "frac_left": frac_left,
        "amp_anchor": amp_anchor,
        "ei": ei,
    }

def collect_majority_left_leaves(leaves, thresh=0.50):
    out = []
    for leaf in leaves:
        idx = np.asarray(leaf["idx"], dtype=np.int64)
        if idx.size == 0:
            continue
        frac_left = np.mean(full_side[idx] == 0)
        if frac_left > thresh:
            out.append(leaf)
    return out

def choose_main_left_leaf(leaves, thresh=0.50):
    cand = collect_majority_left_leaves(leaves, thresh=thresh)
    if len(cand) == 0:
        raise RuntimeError("No majority-left leaves found.")

    scored = []
    for leaf in cand:
        s = summarize_leaf(leaf["idx"])
        scored.append((leaf, s))

    # largest count first, then anchor amplitude, then left fraction
    scored.sort(
        key=lambda t: (t[1]["n"], t[1]["amp_anchor"], t[1]["frac_left"]),
        reverse=True
    )
    return scored[0][0], scored

def get_nodes_on_path(tree, leaf_path):
    """
    Return split nodes from root down to the parent of the leaf.
    """
    parts = leaf_path.split(".")
    if parts[0] != "root":
        raise ValueError(f"Unexpected leaf_path: {leaf_path}")

    nodes = []
    node = tree
    for branch in parts[1:]:
        if node["type"] != "split":
            raise RuntimeError(f"Path {leaf_path} hits a leaf too early.")
        nodes.append((node, branch))
        node = node["left_child"] if branch == "L" else node["right_child"]
    return nodes

def classify_main_leaf_core(main_leaf, margin_k=0.40):
    """
    For spikes in the main leaf:
      trusted_left = those confidently inside the chosen branch at EVERY split on the path
      uncertain    = remaining spikes in the main leaf
    """
    main_idx = np.asarray(main_leaf["idx"], dtype=np.int64)
    if main_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    trusted_mask = np.ones(main_idx.size, dtype=bool)
    path_nodes = get_nodes_on_path(pp_tree, main_leaf["path"])

    for node, branch in path_nodes:
        node_idx = np.asarray(node["idx"], dtype=np.int64)
        z_full = np.asarray(node["best_proj_full"], dtype=np.float32)
        thr = float(node["best_full"]["thr"])
        pooled_sd = float(node["best_full"]["pooled_sd"])
        margin = float(margin_k) * pooled_sd

        idx_to_local = {int(gidx): pos for pos, gidx in enumerate(node_idx.tolist())}
        local_pos = np.array([idx_to_local[int(gidx)] for gidx in main_idx], dtype=np.int64)
        z_main = z_full[local_pos]

        leftlike_is_proj_left = bool(node["leftlike_is_proj_left"])

        if branch == "L":
            # main leaf goes through the left-like child
            if leftlike_is_proj_left:
                cond = z_main <= (thr - margin)
            else:
                cond = z_main >= (thr + margin)
        elif branch == "R":
            # main leaf goes through the non-left-like child
            if leftlike_is_proj_left:
                cond = z_main >= (thr + margin)
            else:
                cond = z_main <= (thr - margin)
        else:
            raise RuntimeError(f"Unexpected branch: {branch}")

        trusted_mask &= cond

    trusted_idx = main_idx[trusted_mask]
    uncertain_idx = main_idx[~trusted_mask]
    return trusted_idx, uncertain_idx

def build_lh_style_template_bank(times, reducer="median", n_bins=6, min_bin_size=100):
    """
    Build templates from trusted-left spikes:
      1) provisional EI from all trusted-left spikes
      2) main channel = strongest trough channel in provisional EI
      3) sort spikes by trough amplitude on that channel
      4) split into equal-count bins
      5) build one template per bin
    """
    times = np.asarray(times, dtype=np.int64)
    if times.size == 0:
        raise RuntimeError("No trusted-left spikes to build template bank.")

    sn_full = extract_full_snips(times)   # [C, L, N]
    C, L, N = sn_full.shape

    if reducer == "median":
        provisional_ei = median_ei_adaptive(sn_full).astype(np.float32)
    elif reducer == "mean":
        provisional_ei = sn_full.mean(axis=2).astype(np.float32)
    else:
        raise ValueError("reducer must be 'median' or 'mean'")

    main_ch = int(np.argmin(provisional_ei.min(axis=1)))
    t0 = int(np.argmin(provisional_ei[main_ch]))
    lo = max(0, t0 - 1)
    hi = min(L - 1, t0 + 1)

    # more negative = stronger
    spike_amp = sn_full[main_ch, lo:hi + 1, :].min(axis=0).astype(np.float32)

    order = np.argsort(spike_amp)   # ascending => most negative first
    times_sorted = times[order]
    amp_sorted = spike_amp[order]
    sn_sorted = sn_full[:, :, order]

    n_bins_eff = min(int(n_bins), max(1, N // int(min_bin_size)))
    groups = np.array_split(np.arange(N), n_bins_eff)

    templates = []
    for bi, g in enumerate(groups):
        if g.size == 0:
            continue

        sn_bin = sn_sorted[:, :, g]
        if reducer == "median":
            tmpl = median_ei_adaptive(sn_bin).astype(np.float32)
        else:
            tmpl = sn_bin.mean(axis=2).astype(np.float32)

        templates.append({
            "bin_index": int(bi),
            "n_spikes": int(g.size),
            "times": times_sorted[g].astype(np.int64),
            "amp_min": float(np.min(amp_sorted[g])),
            "amp_max": float(np.max(amp_sorted[g])),
            "template": tmpl,
        })

    return {
        "window": tuple(window_pp),
        "reducer": reducer,
        "main_ch": int(main_ch),
        "t0_main": int(t0),
        "provisional_ei": provisional_ei,
        "times_all": times_sorted,
        "amp_all": amp_sorted,
        "templates": templates,
    }

def score_single_time_against_bank(t, template_bank, lag_radius=3, channels=None):
    """
    Score one candidate spike time against the provisional template bank.
    Lower score is better.

    Uses RMS residual on selected channels, allowing small integer lag shifts.
    """
    t = int(t)
    templates = template_bank["templates"]
    if len(templates) == 0:
        return np.inf, None, None

    if channels is None:
        # score on strongest channels of the provisional EI
        ei_ref = np.asarray(template_bank["provisional_ei"], dtype=np.float32)
        p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
        channels = np.argsort(p2p)[-20:]   # 20 strongest channels for scoring
        channels = np.asarray(channels, dtype=np.int32)
    else:
        channels = np.asarray(channels, dtype=np.int32)

    best_score = np.inf
    best_bin = None
    best_lag = None

    for lag in range(-int(lag_radius), int(lag_radius) + 1):
        tt = t + lag

        sn, valid_times = extract_snippets_fast_ram(
            raw_data=data_matrix,
            spike_times=np.array([tt], dtype=np.int64),
            window=window_pp,
            selected_channels=channels,
        )
        sn = sn.astype(np.float32, copy=False)

        if sn.shape[2] != 1:
            continue

        x = sn[:, :, 0]   # [Csel, L]

        for rec in templates:
            tmpl = np.asarray(rec["template"], dtype=np.float32)[channels, :]
            resid = x - tmpl
            score = float(np.sqrt(np.mean(resid ** 2)))
            if score < best_score:
                best_score = score
                best_bin = int(rec["bin_index"])
                best_lag = int(lag)

    return best_score, best_bin, best_lag


def collapse_close_times_by_template_fit(times, template_bank, close_dt=10, lag_radius=3):
    """
    Collapse groups of spike times with adjacent spacing < close_dt.
    Keep only the best-fitting candidate in each group.

    Returns:
      kept_times, dropped_times, group_records
    """
    times = np.asarray(times, dtype=np.int64)
    if times.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            [],
        )

    t_sorted = np.sort(times)

    # Build connected components where consecutive dt < close_dt
    groups = []
    start = 0
    for i in range(t_sorted.size - 1):
        if (t_sorted[i + 1] - t_sorted[i]) >= int(close_dt):
            groups.append(t_sorted[start:i + 1])
            start = i + 1
    groups.append(t_sorted[start:])

    kept = []
    dropped = []
    records = []

    # score on strongest provisional-EI channels
    ei_ref = np.asarray(template_bank["provisional_ei"], dtype=np.float32)
    p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
    score_channels = np.argsort(p2p)[-20:].astype(np.int32)

    for g in groups:
        if g.size == 1:
            kept.append(int(g[0]))
            records.append({
                "group_times": g.astype(np.int64),
                "winner_time": int(g[0]),
                "winner_score": np.nan,
                "winner_bin": None,
                "winner_lag": None,
                "n_group": 1,
            })
            continue

        cand_scores = []
        for t in g:
            sc, bi, lg = score_single_time_against_bank(
                t,
                template_bank,
                lag_radius=lag_radius,
                channels=score_channels,
            )
            cand_scores.append((float(sc), int(t), bi, lg))

        cand_scores.sort(key=lambda x: x[0])   # lowest residual wins
        winner_score, winner_time, winner_bin, winner_lag = cand_scores[0]

        kept.append(int(winner_time))
        for _, t, _, _ in cand_scores[1:]:
            dropped.append(int(t))

        records.append({
            "group_times": g.astype(np.int64),
            "winner_time": int(winner_time),
            "winner_score": float(winner_score),
            "winner_bin": winner_bin,
            "winner_lag": winner_lag,
            "n_group": int(g.size),
            "all_scores": cand_scores,
        })

    kept = np.array(sorted(set(kept)), dtype=np.int64)
    dropped = np.array(sorted(set(dropped)), dtype=np.int64)

    return kept, dropped, records


def count_isi_10_30(times):
    """
    Count adjacent sorted spike pairs with ISI in [10, 30] samples.
    Mirrors the notebook metric conceptually.
    """
    times = np.asarray(times, dtype=np.int64)
    if times.size < 2:
        return 0
    t_sorted = np.sort(times)
    d = np.diff(t_sorted)
    return int(np.sum((d >= 10) & (d <= 30)))

# -----------------------------
# Choose main leaf + extra-left leaves
# -----------------------------
main_leaf, scored_majority_left = choose_main_left_leaf(pp_leaves, thresh=left_leaf_thresh)
main_path = main_leaf["path"]
main_idx = np.asarray(main_leaf["idx"], dtype=np.int64)

majority_left_leaves = [leaf for leaf, _ in scored_majority_left]
extra_left_leaves = [leaf for leaf in majority_left_leaves if leaf["path"] != main_path]

extra_left_idx = np.concatenate([np.asarray(leaf["idx"], dtype=np.int64) for leaf in extra_left_leaves]) if len(extra_left_leaves) else np.array([], dtype=np.int64)

print(f"Main left leaf: {main_path}  (n={main_idx.size})")
print("Other majority-left leaves:")
if len(extra_left_leaves) == 0:
    print("  none")
else:
    for leaf in extra_left_leaves:
        s = summarize_leaf(leaf["idx"])
        print(f"  {leaf['path']:12s} | n={s['n']:4d} | orig-left={100*s['frac_left']:.1f}% | anchor amp={s['amp_anchor']:.2f}")

# -----------------------------
# Trusted-left vs uncertain inside the main leaf
# -----------------------------
trusted_left_idx, uncertain_idx = classify_main_leaf_core(main_leaf, margin_k=margin_k)

# Everything not in trusted_left / extra_left / uncertain is trusted_not_left
all_idx = np.arange(full_times.size, dtype=np.int64)

mask_trusted_left = np.zeros(full_times.size, dtype=bool)
mask_extra_left = np.zeros(full_times.size, dtype=bool)
mask_uncertain = np.zeros(full_times.size, dtype=bool)

mask_trusted_left[trusted_left_idx] = True
if extra_left_idx.size > 0:
    mask_extra_left[extra_left_idx] = True
mask_uncertain[uncertain_idx] = True

mask_trusted_not_left = ~(mask_trusted_left | mask_extra_left | mask_uncertain)

# Safety check: no overlaps
overlap_count = (
    np.sum(mask_trusted_left & mask_extra_left) +
    np.sum(mask_trusted_left & mask_uncertain) +
    np.sum(mask_extra_left & mask_uncertain)
)
if overlap_count != 0:
    raise RuntimeError(f"Label overlap detected: {overlap_count}")

trusted_left_times = full_times[mask_trusted_left]
extra_left_times = full_times[mask_extra_left]
uncertain_times = full_times[mask_uncertain]
trusted_not_left_times = full_times[mask_trusted_not_left]

print("\nFinal labels")
print(f"  trusted_left     : {trusted_left_times.size}")
print(f"  extra_left       : {extra_left_times.size}")
print(f"  uncertain        : {uncertain_times.size}")
print(f"  trusted_not_left : {trusted_not_left_times.size}")

# -----------------------------
# Check for near-duplicate spike times (<= 10 samples apart)
# -----------------------------
def report_close_pairs(name, times, max_dt=10, max_show=20):
    times = np.asarray(times, dtype=np.int64)
    if times.size < 2:
        print(f"  {name:16s}: too few spikes to check")
        return {
            "n_pairs": 0,
            "n_spikes_involved": 0,
            "pairs": np.empty((0, 3), dtype=np.int64),
        }

    t_sorted = np.sort(times)
    dt = np.diff(t_sorted)
    hit = np.where(dt <= int(max_dt))[0]

    if hit.size == 0:
        print(f"  {name:16s}: no spike pairs within {max_dt} samples")
        return {
            "n_pairs": 0,
            "n_spikes_involved": 0,
            "pairs": np.empty((0, 3), dtype=np.int64),
        }

    pairs = np.column_stack([
        t_sorted[hit],
        t_sorted[hit + 1],
        dt[hit]
    ]).astype(np.int64)

    spikes_involved = np.unique(np.concatenate([pairs[:, 0], pairs[:, 1]])).size

    print(f"  {name:16s}: {pairs.shape[0]} pairs within {max_dt} samples "
          f"({spikes_involved} spikes involved)")

    n_show = min(max_show, pairs.shape[0])
    for a, b, d in pairs[:n_show]:
        print(f"      {a:8d}  {b:8d}   dt={d:2d}")

    if pairs.shape[0] > n_show:
        print(f"      ... {pairs.shape[0] - n_show} more")

    return {
        "n_pairs": int(pairs.shape[0]),
        "n_spikes_involved": int(spikes_involved),
        "pairs": pairs,
    }

print("\nClose-time checks (<= 10 samples)")
close_trusted_left = report_close_pairs("trusted_left", trusted_left_times, max_dt=10)
close_extra_left = report_close_pairs("extra_left", extra_left_times, max_dt=10)
close_uncertain = report_close_pairs("uncertain", uncertain_times, max_dt=10)
close_trusted_not_left = report_close_pairs("trusted_not_left", trusted_not_left_times, max_dt=10)


# -----------------------------
# Build provisional bank from trusted-left
# -----------------------------
lh_template_bank_provisional = build_lh_style_template_bank(
    trusted_left_times,
    reducer=template_reducer,
    n_bins=template_n_bins,
    min_bin_size=template_min_bin_size,
)

print("\nProvisional template bank")
print(f"  trusted_left spikes used : {trusted_left_times.size}")
print(f"  main channel             : {lh_template_bank_provisional['main_ch']}")
print(f"  n templates              : {len(lh_template_bank_provisional['templates'])}")
for rec in lh_template_bank_provisional["templates"]:
    print(
        f"    bin {rec['bin_index']:2d} | n={rec['n_spikes']:4d} | "
        f"amp range [{rec['amp_min']:.1f}, {rec['amp_max']:.1f}]"
    )

# -----------------------------
# Collapse spuriously close spikes (<10 samples) in trusted-left
# -----------------------------
trusted_left_times_clean, trusted_left_times_dropped_close, close_group_records = collapse_close_times_by_template_fit(
    trusted_left_times,
    lh_template_bank_provisional,
    close_dt=10,      # your "within 10 samples" cleanup
    lag_radius=3,
)

print("\nClose-neighbor cleanup on trusted-left (<10 samples)")
print(f"  before cleanup           : {trusted_left_times.size}")
print(f"  after cleanup            : {trusted_left_times_clean.size}")
print(f"  dropped close-neighbors  : {trusted_left_times_dropped_close.size}")

# -----------------------------
# Rebuild FINAL template bank from cleaned trusted-left only
# -----------------------------
lh_template_bank = build_lh_style_template_bank(
    trusted_left_times_clean,
    reducer=template_reducer,
    n_bins=template_n_bins,
    min_bin_size=template_min_bin_size,
)

print("\nFinal template bank (after close-neighbor cleanup)")
print(f"  trusted_left_clean used  : {trusted_left_times_clean.size}")
print(f"  main channel             : {lh_template_bank['main_ch']}")
print(f"  n templates              : {len(lh_template_bank['templates'])}")
for rec in lh_template_bank["templates"]:
    print(
        f"    bin {rec['bin_index']:2d} | n={rec['n_spikes']:4d} | "
        f"amp range [{rec['amp_min']:.1f}, {rec['amp_max']:.1f}]"
    )

# -----------------------------
# Now compute the suspicious-ISI metric on the CLEANED trusted-left train
# -----------------------------
isi_10_30_trusted_left = count_isi_10_30(trusted_left_times_clean)
print("\nISI 10-30 samples on cleaned trusted-left")
print(f"  count: {isi_10_30_trusted_left}")

# -----------------------------
# Save final state
# -----------------------------
labels = np.empty(full_times.size, dtype=object)
labels[mask_trusted_left] = "trusted_left"
labels[mask_extra_left] = "extra_left"
labels[mask_uncertain] = "uncertain"
labels[mask_trusted_not_left] = "trusted_not_left"

final_left_state = {
    "anchor_ch": anchor_ch,
    "window_pp": tuple(window_pp),
    "main_leaf_path": main_path,
    "margin_k": float(margin_k),
    "trusted_left_idx": trusted_left_idx,
    "extra_left_idx": extra_left_idx,
    "uncertain_idx": uncertain_idx,
    "trusted_not_left_idx": np.where(mask_trusted_not_left)[0],
    "trusted_left_times_raw": trusted_left_times,
    "trusted_left_times": trusted_left_times_clean,
    "trusted_left_times_dropped_close": trusted_left_times_dropped_close,
    "close_group_records": close_group_records,
    "isi_10_30_trusted_left": isi_10_30_trusted_left,
    "extra_left_times": extra_left_times,
    "uncertain_times": uncertain_times,
    "trusted_not_left_times": trusted_not_left_times,
    "labels": labels,
    "majority_left_leaves": [leaf["path"] for leaf in majority_left_leaves],
    "extra_left_leaf_paths": [leaf["path"] for leaf in extra_left_leaves],
    "close_trusted_left": close_trusted_left,
    "close_extra_left": close_extra_left,
    "close_uncertain": close_uncertain,
    "close_trusted_not_left": close_trusted_not_left,
}

# %%
# === Ad hoc sanity check: plot the weakest trusted-left spike on the same 12 channels as Cell 5 ===

import numpy as np
import matplotlib.pyplot as plt
from axolotl_utils_ram import extract_snippets_fast_ram

if "final_left_state" not in globals():
    raise RuntimeError("final_left_state not found. Run Cell 4 first.")

if "lh_template_bank" not in globals():
    raise RuntimeError("lh_template_bank not found. Run Cell 4 first.")

data_matrix = raw_mod
window_pp = tuple(final_left_state["window_pp"])
trusted_left_times = np.asarray(final_left_state["trusted_left_times"], dtype=np.int64)

if trusted_left_times.size == 0:
    raise RuntimeError("No trusted_left spikes available.")

# -----------------------------
# Extract all trusted-left snippets
# -----------------------------
sn_full, valid_times = extract_snippets_fast_ram(
    raw_data=data_matrix,
    spike_times=trusted_left_times,
    window=window_pp,
    selected_channels=np.arange(data_matrix.shape[1], dtype=np.int32),
)
sn_full = sn_full.astype(np.float32, copy=False)
valid_times = np.asarray(valid_times, dtype=np.int64)

if valid_times.size != trusted_left_times.size:
    raise RuntimeError(
        f"Unexpected edge-drop here: requested {trusted_left_times.size}, got {valid_times.size}"
    )

C, L, N = sn_full.shape

# -----------------------------
# Sort by amplitude on final bank main channel
# -----------------------------
main_ch = int(lh_template_bank["main_ch"])
t0 = int(lh_template_bank["t0_main"])
lo = max(0, t0 - 1)
hi = min(L - 1, t0 + 1)

amp = sn_full[main_ch, lo:hi + 1, :].min(axis=0).astype(np.float32)   # negative values
order = np.argsort(amp)   # ascending => strongest first, weakest last

# pick which weak spike to inspect
weak_rank_from_end = 0   # 0 = weakest spike, 1 = second-weakest, etc.
idx_spike = order[-(weak_rank_from_end + 1)]

spike_time = int(valid_times[idx_spike])
spike_amp = float(amp[idx_spike])
spike_wf = sn_full[:, :, idx_spike].astype(np.float32)

print(f"Inspecting weak trusted-left spike")
print(f"  time               : {spike_time}")
print(f"  amplitude on main ch {main_ch}: {spike_amp:.1f}")
print(f"  weak_rank_from_end : {weak_rank_from_end}")

# -----------------------------
# Same top-500 / bottom-500 means as Cell 5
# -----------------------------
n_show = min(500, N // 2)
if n_show == 0:
    raise RuntimeError("Not enough trusted_left spikes to form top/bottom groups.")

idx_top = order[:n_show]          # strongest 500
idx_bottom = order[-n_show:]      # weakest 500

mean_top = sn_full[:, :, idx_top].mean(axis=2).astype(np.float32)
mean_bottom = sn_full[:, :, idx_bottom].mean(axis=2).astype(np.float32)

# -----------------------------
# Same 12 largest channels as Cell 5
# -----------------------------
ei_ref = np.asarray(lh_template_bank["provisional_ei"], dtype=np.float32)
p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
top12 = np.argsort(p2p)[-12:][::-1]

print("Top 12 channels by provisional EI P2P:")
print(top12)

# -----------------------------
# Plot
# -----------------------------
t = np.arange(window_pp[0], window_pp[1] + 1)

fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=False)
axes = axes.ravel()

for ax, ch in zip(axes, top12):
    ax.plot(t, mean_top[ch], label=f"strongest {n_show}", linewidth=2.0)
    ax.plot(t, mean_bottom[ch], label=f"weakest {n_show}", linewidth=2.0)
    ax.plot(t, spike_wf[ch], label="single weak spike", linewidth=2.0, alpha=0.9)
    ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"ch {int(ch)} | p2p={p2p[ch]:.1f}", fontsize=10)
    ax.grid(True, alpha=0.25)

axes[0].legend(fontsize=9)

for ax in axes[-4:]:
    ax.set_xlabel("Samples relative to center")
for ax in axes[::4]:
    ax.set_ylabel("ADC")

fig.suptitle(
    f"Weakest trusted-left spike vs top/bottom means | main ch {main_ch} amp={spike_amp:.1f} | time={spike_time}",
    y=0.98
)
plt.tight_layout()
plt.show()

# %%
# === Cell 5: top-500 vs bottom-500 mean waveforms on 12 largest channels ===
# Uses trusted_left only

import numpy as np
import matplotlib.pyplot as plt
from axolotl_utils_ram import extract_snippets_fast_ram

if "final_left_state" not in globals():
    raise RuntimeError("final_left_state not found. Run Cell 4 first.")

if "lh_template_bank" not in globals():
    raise RuntimeError("lh_template_bank not found. Run Cell 4 first.")

data_matrix = raw_mod
window_pp = tuple(final_left_state["window_pp"])
trusted_left_times = np.asarray(final_left_state["trusted_left_times"], dtype=np.int64)

if trusted_left_times.size == 0:
    raise RuntimeError("No trusted_left spikes available.")

# -----------------------------
# Extract all-channel snippets for trusted-left spikes
# -----------------------------
sn_full, valid_times = extract_snippets_fast_ram(
    raw_data=data_matrix,
    spike_times=trusted_left_times,
    window=window_pp,
    selected_channels=np.arange(data_matrix.shape[1], dtype=np.int32),
)
sn_full = sn_full.astype(np.float32, copy=False)
valid_times = np.asarray(valid_times, dtype=np.int64)

if valid_times.size != trusted_left_times.size:
    raise RuntimeError(
        f"Unexpected edge-drop in Cell 5: requested {trusted_left_times.size}, got {valid_times.size}. "
        f"Windows are inconsistent somewhere."
    )

C, L, N = sn_full.shape

# -----------------------------
# Sort spikes by amplitude on the provisional main channel
# strongest = most negative trough on main channel
# -----------------------------
main_ch = int(lh_template_bank["main_ch"])
t0 = int(lh_template_bank["t0_main"])
lo = max(0, t0 - 1)
hi = min(L - 1, t0 + 1)

amp = sn_full[main_ch, lo:hi + 1, :].min(axis=0).astype(np.float32)   # negative values
order = np.argsort(amp)   # ascending => strongest (most negative) first

n_show = min(500, N // 2) if N >= 2 else 0
if n_show == 0:
    raise RuntimeError("Not enough trusted_left spikes to form top/bottom groups.")

idx_top = order[:n_show]          # strongest 500
idx_bottom = order[-n_show:]      # weakest 500

mean_top = sn_full[:, :, idx_top].mean(axis=2).astype(np.float32)
mean_bottom = sn_full[:, :, idx_bottom].mean(axis=2).astype(np.float32)

# -----------------------------
# Pick 12 largest channels by P2P from provisional EI
# -----------------------------
ei_ref = np.asarray(lh_template_bank["provisional_ei"], dtype=np.float32)
p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)

top12 = np.argsort(p2p)[-12:][::-1]   # largest first

print(f"Trusted-left spikes: {N}")
print(f"Main channel: {main_ch}")
print(f"Comparing strongest {n_show} vs weakest {n_show} trusted-left spikes")
print("Top 12 channels by provisional EI P2P:")
print(top12)

# -----------------------------
# Plot 12 channels
# -----------------------------
t = np.arange(window_pp[0], window_pp[1] + 1)

fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=False)
axes = axes.ravel()

for ax, ch in zip(axes, top12):
    ax.plot(t, mean_top[ch], label=f"strongest {n_show}", linewidth=2.0)
    ax.plot(t, mean_bottom[ch], label=f"weakest {n_show}", linewidth=2.0)
    ax.axvline(0, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"ch {int(ch)} | p2p={p2p[ch]:.1f}", fontsize=10)
    ax.grid(True, alpha=0.25)

# one legend only
axes[0].legend(fontsize=9)

for ax in axes[-4:]:
    ax.set_xlabel("Samples relative to center")
for ax in axes[::4]:
    ax.set_ylabel("ADC")

fig.suptitle(
    f"Trusted-left mean waveforms | strongest {n_show} vs weakest {n_show} spikes on main ch {main_ch}",
    y=0.98
)
plt.tight_layout()
plt.show()

# %%
# === Cell 6: global multichannel template assignment + optional subtraction ===
# Uses trusted_left only.
# Assignment: best template by weighted multichannel residual RMS
# Lag: single global lag per spike
# Logs per-channel RMS deltas for the chosen fit
# Set APPLY_SUBTRACTION=True only when you want to actually modify raw_mod

import numpy as np
import matplotlib.pyplot as plt
from axolotl_utils_ram import extract_snippets_fast_ram

# -----------------------------
# Inputs
# -----------------------------
if "final_left_state" not in globals():
    raise RuntimeError("final_left_state not found. Run Cell 4 first.")

if "lh_template_bank" not in globals():
    raise RuntimeError("lh_template_bank not found. Run Cell 4 first.")

data_matrix = raw_mod
window_pp = tuple(final_left_state["window_pp"])
trusted_left_times = np.asarray(final_left_state["trusted_left_times"], dtype=np.int64)

if trusted_left_times.size == 0:
    raise RuntimeError("No trusted_left spikes available.")

# -----------------------------
# Knobs
# -----------------------------
APPLY_SUBTRACTION = True   # flip to True only when you want to subtract in-place from raw_mod

lag_radius = 3             # global lag search: -3..+3
n_score_channels = 20      # strongest channels used for assignment/scoring
min_improvement_frac = 0.00  # require post_score < pre_score * (1 - frac)
require_anchor_improvement = True

# -----------------------------
# Helpers
# -----------------------------
def roll_zero_2d(arr, shift):
    """
    Zero-padded shift along time axis for [C, T].
    Positive shift moves waveform to the right.
    """
    arr = np.asarray(arr)
    out = np.zeros_like(arr)
    if shift == 0:
        out[:] = arr
    elif shift > 0:
        out[:, shift:] = arr[:, :arr.shape[1] - shift]
    else:
        s = -shift
        out[:, :arr.shape[1] - s] = arr[:, s:]
    return out

def weighted_rms_score(x, tmpl, weights):
    """
    x, tmpl: [Csel, T]
    weights: [Csel], normalized
    Returns:
      score_scalar,
      rms_pre_ch,
      rms_post_ch,
      delta_rms_ch = rms_pre - rms_post
    """
    resid = x - tmpl

    rms_pre_ch = np.sqrt(np.mean(x ** 2, axis=1))
    rms_post_ch = np.sqrt(np.mean(resid ** 2, axis=1))
    delta_rms_ch = rms_pre_ch - rms_post_ch

    score_pre = float(np.sum(weights * rms_pre_ch))
    score_post = float(np.sum(weights * rms_post_ch))
    return score_pre, score_post, rms_pre_ch, rms_post_ch, delta_rms_ch

# -----------------------------
# Build score channels from the final trusted-left EI
# -----------------------------
ei_ref = np.asarray(lh_template_bank["provisional_ei"], dtype=np.float32)
main_ch = int(lh_template_bank["main_ch"])
t0_main = int(lh_template_bank["t0_main"])

p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
score_channels = np.argsort(p2p)[-n_score_channels:][::-1].astype(np.int32)

# ensure main channel present
if main_ch not in score_channels:
    score_channels[-1] = main_ch

# sort unique, then re-rank by p2p descending
score_channels = np.array(sorted(set(score_channels.tolist()), key=lambda c: p2p[c], reverse=True), dtype=np.int32)

weights = p2p[score_channels].astype(np.float32)
weights = weights / (weights.sum() + 1e-12)

print(f"Trusted-left spikes to score: {trusted_left_times.size}")
print(f"Main channel: {main_ch}")
print(f"Score channels ({score_channels.size}): {score_channels}")

# -----------------------------
# Extract scoring snippets once for all trusted-left spikes
# -----------------------------
sn_score, valid_times = extract_snippets_fast_ram(
    raw_data=data_matrix,
    spike_times=trusted_left_times,
    window=window_pp,
    selected_channels=score_channels,
)
sn_score = sn_score.astype(np.float32, copy=False)
valid_times = np.asarray(valid_times, dtype=np.int64)

if valid_times.size != trusted_left_times.size:
    raise RuntimeError(
        f"Unexpected edge-drop here: requested {trusted_left_times.size}, got {valid_times.size}. "
        f"Windows are inconsistent somewhere."
    )

Csel, L, N = sn_score.shape

# -----------------------------
# Precompute shifted templates on score channels and full channels
# -----------------------------
template_records = lh_template_bank["templates"]
n_templates = len(template_records)
lags = np.arange(-lag_radius, lag_radius + 1, dtype=int)

tmpl_score_bank = {}
tmpl_full_bank = {}

for rec in template_records:
    bi = int(rec["bin_index"])
    tmpl_full = np.asarray(rec["template"], dtype=np.float32)             # [C, L]
    tmpl_score = tmpl_full[score_channels, :]                             # [Csel, L]

    for lag in lags:
        tmpl_score_bank[(bi, int(lag))] = roll_zero_2d(tmpl_score, int(lag))
        tmpl_full_bank[(bi, int(lag))] = roll_zero_2d(tmpl_full, int(lag))

# -----------------------------
# Score all spikes against all templates/lags
# -----------------------------
best_bin = np.full(N, -1, dtype=np.int32)
best_lag = np.full(N, 0, dtype=np.int32)
pre_score = np.full(N, np.nan, dtype=np.float32)
post_score = np.full(N, np.nan, dtype=np.float32)
accepted = np.zeros(N, dtype=bool)

# diagnostic arrays
anchor_rms_pre = np.full(N, np.nan, dtype=np.float32)
anchor_rms_post = np.full(N, np.nan, dtype=np.float32)
delta_rms_ch = np.full((score_channels.size, N), np.nan, dtype=np.float32)

# which score-channel row corresponds to main_ch
anchor_row = np.where(score_channels == main_ch)[0]
anchor_row = int(anchor_row[0]) if anchor_row.size else 0

for i in range(N):
    x = sn_score[:, :, i]   # [Csel, L]

    best_post = np.inf
    best_pre = np.nan
    best_bin_i = -1
    best_lag_i = 0
    best_delta_ch = None
    best_rms_pre_ch = None
    best_rms_post_ch = None

    for rec in template_records:
        bi = int(rec["bin_index"])
        for lag in lags:
            tmpl = tmpl_score_bank[(bi, int(lag))]
            sc_pre, sc_post, rms_pre_ch, rms_post_ch, d_ch = weighted_rms_score(x, tmpl, weights)

            if sc_post < best_post:
                best_post = sc_post
                best_pre = sc_pre
                best_bin_i = bi
                best_lag_i = int(lag)
                best_delta_ch = d_ch
                best_rms_pre_ch = rms_pre_ch
                best_rms_post_ch = rms_post_ch

    best_bin[i] = best_bin_i
    best_lag[i] = best_lag_i
    pre_score[i] = best_pre
    post_score[i] = best_post
    delta_rms_ch[:, i] = best_delta_ch.astype(np.float32)

    anchor_rms_pre[i] = float(best_rms_pre_ch[anchor_row])
    anchor_rms_post[i] = float(best_rms_post_ch[anchor_row])

    improves_global = best_post < best_pre * (1.0 - float(min_improvement_frac))
    improves_anchor = anchor_rms_post[i] < anchor_rms_pre[i]

    if require_anchor_improvement:
        accepted[i] = bool(improves_global and improves_anchor)
    else:
        accepted[i] = bool(improves_global)

# -----------------------------
# Summary
# -----------------------------
print("\nAssignment summary")
print(f"Accepted for subtraction: {accepted.sum()} / {N}")
print(f"Rejected by gate        : {(~accepted).sum()} / {N}")

print("\nTemplate usage (all spikes before gate)")
for bi in sorted(np.unique(best_bin)):
    print(f"  bin {int(bi):2d}: {(best_bin == bi).sum():4d}")

print("\nTemplate usage (accepted spikes only)")
for bi in sorted(np.unique(best_bin[accepted])):
    print(f"  bin {int(bi):2d}: {((best_bin == bi) & accepted).sum():4d}")

# quick diagnostic plot: improvement histogram
improvement = pre_score - post_score
plt.figure(figsize=(6, 4))
plt.hist(improvement, bins=100)
plt.xlabel("Weighted RMS improvement (pre - post)")
plt.ylabel("Count")
plt.title("Trusted-left assignment improvement")
plt.tight_layout()
plt.show()

# -----------------------------
# Build subtraction plan
# -----------------------------
subtraction_plan = {
    "window_pp": tuple(window_pp),
    "main_ch": int(main_ch),
    "t0_main": int(t0_main),
    "score_channels": score_channels,
    "weights": weights,
    "times": valid_times,
    "best_bin": best_bin,
    "best_lag": best_lag,
    "pre_score": pre_score,
    "post_score": post_score,
    "improvement": improvement,
    "accepted": accepted,
    "anchor_rms_pre": anchor_rms_pre,
    "anchor_rms_post": anchor_rms_post,
    "delta_rms_ch": delta_rms_ch,   # [n_score_channels, N]
}

# -----------------------------
# Optional in-place subtraction on raw_mod
# -----------------------------
if APPLY_SUBTRACTION:
    pre, post = window_pp
    T_total, C_total = data_matrix.shape

    accepted_idx = np.where(accepted)[0]
    print(f"\nApplying subtraction in-place to raw_mod for {accepted_idx.size} spikes")

    for i in accepted_idx:
        t = int(valid_times[i])
        bi = int(best_bin[i])
        lag = int(best_lag[i])

        tmpl_full = tmpl_full_bank[(bi, lag)]   # [C, L]
        start = t + pre
        end = t + post + 1

        if start < 0 or end > T_total:
            continue

        # raw_mod is [T, C], template is [C, L]
        data_matrix[start:end, :] -= tmpl_full.T.astype(data_matrix.dtype, copy=False)

    print("Subtraction applied to raw_mod.")
else:
    print("\nAPPLY_SUBTRACTION=False, so raw_mod was not modified.")

# %% [markdown]
# # SOUP PURSUIT

# %%
# === One-shot PP scan on top-K spikes (no valley requirement, no recursion, no subtraction) ===

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

# -------------------------------------------------
# Config
# -------------------------------------------------
data_matrix = raw_mod
params_scan = LH_PP_LOOP_PARAMS

fs = int(params_scan["fs"])
duration_sec = float(params_scan["pp_duration_sec"])
stop = min(int(duration_sec * fs), data_matrix.shape[0])

window_pp = tuple(params_scan["window_pp"])
pre, post = int(window_pp[0]), int(window_pp[1])
center_idx = int(-pre)

topk_spikes_total = 10_000          # total strongest minima per channel
ei_subset_max = 500                 # subset used to estimate pooled EI and select channels
min_pool_spikes = 2000              # skip channels with fewer surviving minima than this

n_search_pcs = int(params_scan["pp_n_search_pcs"])
n_random_dirs = int(params_scan["pp_n_random_dirs"])
hist_bins = int(params_scan["pp_hist_bins"])
smooth_sigma = float(params_scan["pp_smooth_sigma"])
min_child_n = int(params_scan["pp_min_child_n"])
min_group_frac = float(params_scan["pp_min_group_frac"])
target_frac = float(params_scan["pp_target_frac"])

p2p_thr = float(params_scan["pp_p2p_thr"])
max_channels = int(params_scan["pp_max_channels"])
min_channels = int(params_scan["pp_min_channels"])

amp_pc1_weight = float(params_scan["pp_amp_pc1_weight"])
n_biased_tail_dirs = int(params_scan["pp_n_biased_tail_dirs"])
biased_tail_scale = float(params_scan["pp_biased_tail_scale"])

rng = np.random.default_rng(int(params_scan["pp_rng_seed"]))

# Optional limit for testing; set to None to scan all channels in skyline order
max_channels_to_process = None

# -------------------------------------------------
# Small local helpers
# -------------------------------------------------
def _local_maxima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    if y[0] > y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] > y[i - 1]) and (y[i] >= y[i + 1]):
            out.append(i)
    if y[-1] > y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=np.int64)

def _local_minima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n < 2:
        return np.array([], dtype=np.int64)
    if y[0] < y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] < y[i - 1]) and (y[i] <= y[i + 1]):
            out.append(i)
    if y[-1] < y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=np.int64)

def _normalize_direction(w):
    w = np.asarray(w, dtype=np.float32).ravel()
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)

def _robust_zscore_1d_fit(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32).ravel()
    med = float(np.median(x)) if x.size else 0.0
    mad = float(np.median(np.abs(x - med))) if x.size else 0.0
    scale = float(1.4826 * max(mad, eps))
    z = (x - med) / scale
    return z.astype(np.float32), med, scale

def _robust_zscore_cols_fit(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float32)
    med = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - med[None, :]), axis=0).astype(np.float32)
    scale = (1.4826 * np.maximum(mad, eps)).astype(np.float32)
    Z = (X - med[None, :]) / scale[None, :]
    return Z.astype(np.float32), med, scale

def _build_biased_direction_bank(
    n_aug,
    n_pcs,
    rng,
    amp_pc1_weight=0.5,
    n_tail_dirs=16,
    tail_scale=0.08,
):
    dirs = []

    if n_pcs >= 1:
        w = np.zeros(n_aug, dtype=np.float32)
        w[1] = 1.0
        dirs.append(("pc1_only", _normalize_direction(w)))

        w = np.zeros(n_aug, dtype=np.float32)
        w[0] = 1.0
        w[1] = float(amp_pc1_weight)
        dirs.append(("amp_plus_pc1", _normalize_direction(w)))

        n_tail = max(0, n_aug - 2)
        for i in range(int(n_tail_dirs)):
            w = np.zeros(n_aug, dtype=np.float32)
            w[0] = 1.0
            w[1] = float(amp_pc1_weight)
            if n_tail > 0:
                w[2:] = rng.uniform(
                    -float(tail_scale),
                    float(tail_scale),
                    size=n_tail,
                ).astype(np.float32)
            dirs.append((f"amp_plus_pc1_tail_{i:02d}", _normalize_direction(w)))

    return dirs

def _random_unit_directions(n_dirs, dim, rng):
    W = rng.normal(size=(n_dirs, dim)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W

def _score_projection_1d(z, bins, sigma, min_child_n, min_group_frac, target_frac):
    z = np.asarray(z, dtype=np.float32).ravel()
    N = z.size
    hist, edges = np.histogram(z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = gaussian_filter1d(hist.astype(np.float32), sigma=sigma)

    peaks = _local_maxima(hs)
    valleys = _local_minima(hs)

    best = None
    for v in valleys:
        left_peaks = peaks[peaks < v]
        right_peaks = peaks[peaks > v]
        if left_peaks.size == 0 or right_peaks.size == 0:
            continue

        pl = left_peaks[-1]
        pr = right_peaks[0]

        valley_h = float(hs[v])
        peak_l_h = float(hs[pl])
        peak_r_h = float(hs[pr])
        min_peak_h = min(peak_l_h, peak_r_h)
        if min_peak_h <= 0:
            continue

        thr = float(centers[v])
        mask_left = z <= thr
        n_left = int(mask_left.sum())
        n_right = int(N - n_left)
        n_minor = min(n_left, n_right)
        frac_minor = n_minor / float(N)

        if n_left < int(min_child_n) or n_right < int(min_child_n):
            continue
        if frac_minor < float(min_group_frac):
            continue

        zl = z[mask_left]
        zr = z[~mask_left]
        mu_l = float(zl.mean())
        mu_r = float(zr.mean())
        sd_l = float(zl.std(ddof=1)) if zl.size > 1 else 1e-12
        sd_r = float(zr.std(ddof=1)) if zr.size > 1 else 1e-12
        pooled_sd = np.sqrt(0.5 * (sd_l * sd_l + sd_r * sd_r) + 1e-12)

        depth = 1.0 - (valley_h / (min_peak_h + 1e-12))
        sep = abs(mu_r - mu_l) / pooled_sd
        size_bonus = 1.0 - np.exp(-frac_minor / float(target_frac))
        score = depth * sep * size_bonus

        cand = {
            "score": float(score),
            "thr": float(thr),
            "depth": float(depth),
            "sep": float(sep),
            "size_bonus": float(size_bonus),
            "pooled_sd": float(pooled_sd),
            "n_left_raw": int(n_left),
            "n_right_raw": int(n_right),
            "frac_minor": float(frac_minor),
            "mask_left_raw": mask_left,
            "hist": hist,
            "hist_smooth": hs,
            "edges": edges,
            "centers": centers,
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    return best

def _detect_channel_amp_from_snips(sn_sel, detect_row, center_idx):
    lo = max(0, int(center_idx) - 1)
    hi = min(sn_sel.shape[1] - 1, int(center_idx) + 1)
    return -sn_sel[int(detect_row), lo:hi + 1, :].min(axis=0).astype(np.float32)

# -------------------------------------------------
# Skyline order (with current exclusions)
# -------------------------------------------------
skyline_state_scan = skyline_scan_with_exclusions(data_matrix, channel_state, params_scan)
channel_order_scan = np.asarray(skyline_state_scan["channel_order"], dtype=np.int64)

if max_channels_to_process is not None:
    channel_order_scan = channel_order_scan[:int(max_channels_to_process)]

print(f"Scanning {len(channel_order_scan)} channels in skyline order")
print(f"stop={stop:,} samples ({stop / fs / 60:.1f} min), window={window_pp}, topk={topk_spikes_total}")

# -------------------------------------------------
# Main loop
# -------------------------------------------------
pp_topk_scan_results = []

for rank_idx, ch in enumerate(channel_order_scan):
    ch = int(ch)

    x = data_matrix[:stop, ch].astype(np.float32, copy=False)
    idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    if idx.size == 0:
        print(f"ch={ch:3d} rank={rank_idx:3d} SKIP no_minima")
        continue

    times_abs = idx.astype(np.int64)
    vals = x[idx].astype(np.float32, copy=False)

    ok = (times_abs + pre >= 0) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals = vals[ok]
    if times_abs.size == 0:
        print(f"ch={ch:3d} rank={rank_idx:3d} SKIP no_window_safe_minima")
        continue

    excl = None
    if channel_state.get(ch) is not None:
        excl = channel_state[ch].get("exclude_times", None)

    near = _times_near_exclusions(times_abs, excl, int(params_scan["exclude_radius_samples"]))
    times_keep = times_abs[~near]
    vals_keep = vals[~near]

    if times_keep.size < int(min_pool_spikes):
        print(
            f"ch={ch:3d} rank={rank_idx:3d} SKIP too_few_after_exclusion "
            f"n={times_keep.size}"
        )
        continue

    # top-K strongest minima
    order_amp = np.argsort(vals_keep)[:min(int(topk_spikes_total), int(times_keep.size))]
    pool_times = times_keep[order_amp].astype(np.int64, copy=False)
    pool_vals = vals_keep[order_amp].astype(np.float32, copy=False)

    n_pool = int(pool_times.size)
    if n_pool < 2 * int(min_child_n):
        print(
            f"ch={ch:3d} rank={rank_idx:3d} SKIP pool_too_small_for_split "
            f"n={n_pool}"
        )
        continue

    # Small subset for pooled EI / channel selection
    ei_subset_n = min(int(ei_subset_max), n_pool)
    ei_subset_idx = _sample_evenly_sorted_indices(n_pool, ei_subset_n)
    ei_subset_times = pool_times[ei_subset_idx]

    all_ch = np.arange(data_matrix.shape[1], dtype=np.int32)
    sn_ei, valid_ei = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=ei_subset_times,
        window=window_pp,
        selected_channels=all_ch,
    )
    sn_ei = sn_ei.astype(np.float32, copy=False)
    if sn_ei.shape[2] == 0:
        print(f"ch={ch:3d} rank={rank_idx:3d} SKIP no_valid_ei_subset_snips")
        continue

    ei_pool = sn_ei.mean(axis=2).astype(np.float32)

    selected_channels, _ = select_template_channels(
        ei_pool,
        p2p_thr=p2p_thr,
        max_n=max_channels,
        min_n=min_channels,
        force_include_main=True,
    )
    selected_channels = np.asarray(selected_channels, dtype=np.int64)
    if ch not in selected_channels:
        selected_channels = np.concatenate([selected_channels, np.array([ch], dtype=np.int64)])
    selected_channels = np.asarray(np.unique(selected_channels), dtype=np.int64)

    detect_row = int(np.where(selected_channels == ch)[0][0])

    sn_sel, valid_pool = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=pool_times,
        window=window_pp,
        selected_channels=selected_channels.astype(np.int32),
    )
    sn_sel = sn_sel.astype(np.float32, copy=False)
    valid_pool = np.asarray(valid_pool, dtype=np.int64)

    if valid_pool.size != pool_times.size:
        print(f"ch={ch:3d} rank={rank_idx:3d} SKIP edge_drop_in_selected_snips")
        continue

    X = sn_sel.transpose(2, 0, 1).reshape(sn_sel.shape[2], -1).astype(np.float32)
    n_pcs = int(min(n_search_pcs, X.shape[0], X.shape[1]))
    if n_pcs < 2:
        print(f"ch={ch:3d} rank={rank_idx:3d} SKIP not_enough_rank_for_pca n_pcs={n_pcs}")
        continue

    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(X).astype(np.float32)

    amp = _detect_channel_amp_from_snips(sn_sel, detect_row, center_idx)
    pcs_z, _, _ = _robust_zscore_cols_fit(pcs)
    amp_z, _, _ = _robust_zscore_1d_fit(amp)
    feat = np.concatenate([amp_z[:, None], pcs_z], axis=1).astype(np.float32)

    candidate_dirs = _build_biased_direction_bank(
        n_aug=feat.shape[1],
        n_pcs=n_pcs,
        rng=rng,
        amp_pc1_weight=amp_pc1_weight,
        n_tail_dirs=n_biased_tail_dirs,
        tail_scale=biased_tail_scale,
    )
    W_rand = _random_unit_directions(n_random_dirs, feat.shape[1], rng)
    for i_dir in range(n_random_dirs):
        candidate_dirs.append((f"random_{i_dir:04d}", W_rand[i_dir]))

    best = None
    best_name = None
    best_w = None
    best_z = None

    for dir_name, w in candidate_dirs:
        z = feat @ w
        res = _score_projection_1d(
            z,
            bins=hist_bins,
            sigma=smooth_sigma,
            min_child_n=min_child_n,
            min_group_frac=min_group_frac,
            target_frac=target_frac,
        )
        if res is None:
            continue
        if (best is None) or (res["score"] > best["score"]):
            best = res
            best_name = str(dir_name)
            best_w = w
            best_z = z.copy()

    if best is None:
        print(f"ch={ch:3d} rank={rank_idx:3d} SKIP no_valid_projection_split n={n_pool}")
        pp_topk_scan_results.append({
            "detect_ch": ch,
            "skyline_rank": rank_idx,
            "n_pool": n_pool,
            "status": "no_valid_projection_split",
        })
        continue

    # Orient leaves so LEFT = larger-amplitude mean leaf on detect channel
    mask_left_raw = np.asarray(best["mask_left_raw"], dtype=bool)
    amp_left_raw = amp[mask_left_raw]
    amp_right_raw = amp[~mask_left_raw]

    mean_amp_left_raw = float(np.mean(amp_left_raw)) if amp_left_raw.size else np.nan
    mean_amp_right_raw = float(np.mean(amp_right_raw)) if amp_right_raw.size else np.nan

    if mean_amp_left_raw >= mean_amp_right_raw:
        mask_left = mask_left_raw
        mask_right = ~mask_left_raw
        mean_amp_left = mean_amp_left_raw
        mean_amp_right = mean_amp_right_raw
    else:
        mask_left = ~mask_left_raw
        mask_right = mask_left_raw
        mean_amp_left = mean_amp_right_raw
        mean_amp_right = mean_amp_left_raw

    n_left = int(mask_left.sum())
    n_right = int(mask_right.sum())

    res_row = {
        "detect_ch": ch,
        "skyline_rank": int(rank_idx),
        "n_minima_total": int(idx.size),
        "n_after_window": int(times_abs.size),
        "n_after_exclusion": int(times_keep.size),
        "n_pool": int(n_pool),
        "n_sel_channels": int(len(selected_channels)),
        "dir_name": best_name,
        "score": float(best["score"]),
        "sep": float(best["sep"]),
        "depth": float(best["depth"]),
        "size_bonus": float(best["size_bonus"]),
        "pooled_sd": float(best["pooled_sd"]),
        "thr": float(best["thr"]),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "mean_amp_left": float(mean_amp_left),
        "mean_amp_right": float(mean_amp_right),
        "frac_minor": float(best["frac_minor"]),
        "status": "ok",
    }
    pp_topk_scan_results.append(res_row)

    print(
        "ch={ch:3d} rank={rk:3d} "
        "score={sc:6.3f} sep={sep:6.3f} depth={dep:6.3f} sb={sb:5.3f} "
        "psd={psd:6.3f} thr={thr:7.3f} "
        "nL={nL:5d} nR={nR:5d} "
        "ampL={aL:7.1f} ampR={aR:7.1f} "
        "selC={nc:2d} dir={dn}".format(
            ch=ch,
            rk=rank_idx,
            sc=res_row["score"],
            sep=res_row["sep"],
            dep=res_row["depth"],
            sb=res_row["size_bonus"],
            psd=res_row["pooled_sd"],
            thr=res_row["thr"],
            nL=res_row["n_left"],
            nR=res_row["n_right"],
            aL=res_row["mean_amp_left"],
            aR=res_row["mean_amp_right"],
            nc=res_row["n_sel_channels"],
            dn=best_name,
        )
    )

print("\nDone. Results saved in `pp_topk_scan_results`.")

# %%
# ==== Plots for pp_topk_scan_results ====

import numpy as np
import matplotlib.pyplot as plt

if "pp_topk_scan_results" not in globals():
    raise RuntimeError("pp_topk_scan_results not found. Run the one-shot PP scan cell first.")

rows = [r for r in pp_topk_scan_results if str(r.get("status", "")) == "ok"]
if len(rows) == 0:
    raise RuntimeError("No successful rows in pp_topk_scan_results.")

# Sort explicitly by skyline rank
rows = sorted(rows, key=lambda r: int(r["skyline_rank"]))

rank = np.array([int(r["skyline_rank"]) for r in rows], dtype=np.int64)
ch = np.array([int(r["detect_ch"]) for r in rows], dtype=np.int64)

score = np.array([float(r["score"]) for r in rows], dtype=np.float64)
sep = np.array([float(r["sep"]) for r in rows], dtype=np.float64)
depth = np.array([float(r["depth"]) for r in rows], dtype=np.float64)

ampL = np.array([float(r["mean_amp_left"]) for r in rows], dtype=np.float64)
ampR = np.array([float(r["mean_amp_right"]) for r in rows], dtype=np.float64)

nL = np.array([int(r["n_left"]) for r in rows], dtype=np.int64)
nR = np.array([int(r["n_right"]) for r in rows], dtype=np.int64)

# -----------------------------
# 1) sep vs depth scatter + correlation
# -----------------------------
mask_sd = np.isfinite(sep) & np.isfinite(depth)
if np.sum(mask_sd) >= 2:
    corr_sd = np.corrcoef(sep[mask_sd], depth[mask_sd])[0, 1]
else:
    corr_sd = np.nan

plt.figure(figsize=(6, 5))
plt.scatter(sep, depth, s=18, alpha=0.8)
plt.xlabel("sep")
plt.ylabel("depth")
plt.title(f"sep vs depth  |  Pearson r = {corr_sd:.3f}")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

print(f"Correlation sep vs depth: r = {corr_sd:.6f}")

# -----------------------------
# 2) ampL vs ampR as line plot (rank order)
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(rank, ampL, linewidth=1.5, label="ampL")
plt.plot(rank, ampR, linewidth=1.5, label="ampR")
plt.xlabel("skyline rank")
plt.ylabel("mean leaf amplitude on detect channel")
plt.title("ampL vs ampR by skyline rank")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# -----------------------------
# 3) N spikes in L vs R as line plot (rank order)
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(rank, nL, linewidth=1.5, label="nL")
plt.plot(rank, nR, linewidth=1.5, label="nR")
plt.xlabel("skyline rank")
plt.ylabel("number of spikes in leaf")
plt.title("nL vs nR by skyline rank")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# -----------------------------
# 4) score as line plot (rank order)
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(rank, score, linewidth=1.5)
plt.xlabel("skyline rank")
plt.ylabel("score")
plt.title("one-shot PP score by skyline rank")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# %%
# === EI overlays for channels with biggest ampL-ampR difference ===

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

# -----------------------------
# Config
# -----------------------------
data_matrix = raw_mod
params_scan = LH_PP_LOOP_PARAMS

fs = int(params_scan["fs"])
duration_sec = float(params_scan["pp_duration_sec"])
stop = min(int(duration_sec * fs), data_matrix.shape[0])

window_pp = tuple(params_scan["window_pp"])
pre, post = int(window_pp[0]), int(window_pp[1])
center_idx = int(-pre)

topk_spikes_total = 10_000
ei_subset_max = 500
top_n_channels = 10
max_ei_spikes_per_leaf = 1000

n_search_pcs = int(params_scan["pp_n_search_pcs"])
n_random_dirs = int(params_scan["pp_n_random_dirs"])
hist_bins = int(params_scan["pp_hist_bins"])
smooth_sigma = float(params_scan["pp_smooth_sigma"])
min_child_n = int(params_scan["pp_min_child_n"])
min_group_frac = float(params_scan["pp_min_group_frac"])
target_frac = float(params_scan["pp_target_frac"])

p2p_thr = float(params_scan["pp_p2p_thr"])
max_channels = int(params_scan["pp_max_channels"])
min_channels = int(params_scan["pp_min_channels"])

amp_pc1_weight = float(params_scan["pp_amp_pc1_weight"])
n_biased_tail_dirs = int(params_scan["pp_n_biased_tail_dirs"])
biased_tail_scale = float(params_scan["pp_biased_tail_scale"])

rng = np.random.default_rng(int(params_scan["pp_rng_seed"]))

# -----------------------------
# Helpers (same as one-shot scan)
# -----------------------------
def _local_maxima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    if y[0] > y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] > y[i - 1]) and (y[i] >= y[i + 1]):
            out.append(i)
    if y[-1] > y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=np.int64)

def _local_minima(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    out = []
    if n < 2:
        return np.array([], dtype=np.int64)
    if y[0] < y[1]:
        out.append(0)
    for i in range(1, n - 1):
        if (y[i] < y[i - 1]) and (y[i] <= y[i + 1]):
            out.append(i)
    if y[-1] < y[-2]:
        out.append(n - 1)
    return np.array(out, dtype=np.int64)

def _normalize_direction(w):
    w = np.asarray(w, dtype=np.float32).ravel()
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)

def _robust_zscore_1d_fit(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32).ravel()
    med = float(np.median(x)) if x.size else 0.0
    mad = float(np.median(np.abs(x - med))) if x.size else 0.0
    scale = float(1.4826 * max(mad, eps))
    z = (x - med) / scale
    return z.astype(np.float32), med, scale

def _robust_zscore_cols_fit(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float32)
    med = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - med[None, :]), axis=0).astype(np.float32)
    scale = (1.4826 * np.maximum(mad, eps)).astype(np.float32)
    Z = (X - med[None, :]) / scale[None, :]
    return Z.astype(np.float32), med, scale

def _build_biased_direction_bank(
    n_aug,
    n_pcs,
    rng,
    amp_pc1_weight=0.5,
    n_tail_dirs=16,
    tail_scale=0.08,
):
    dirs = []

    if n_pcs >= 1:
        w = np.zeros(n_aug, dtype=np.float32)
        w[1] = 1.0
        dirs.append(("pc1_only", _normalize_direction(w)))

        w = np.zeros(n_aug, dtype=np.float32)
        w[0] = 1.0
        w[1] = float(amp_pc1_weight)
        dirs.append(("amp_plus_pc1", _normalize_direction(w)))

        n_tail = max(0, n_aug - 2)
        for i in range(int(n_tail_dirs)):
            w = np.zeros(n_aug, dtype=np.float32)
            w[0] = 1.0
            w[1] = float(amp_pc1_weight)
            if n_tail > 0:
                w[2:] = rng.uniform(
                    -float(tail_scale),
                    float(tail_scale),
                    size=n_tail,
                ).astype(np.float32)
            dirs.append((f"amp_plus_pc1_tail_{i:02d}", _normalize_direction(w)))

    return dirs

def _random_unit_directions(n_dirs, dim, rng):
    W = rng.normal(size=(n_dirs, dim)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W

def _score_projection_1d(z, bins, sigma, min_child_n, min_group_frac, target_frac):
    z = np.asarray(z, dtype=np.float32).ravel()
    N = z.size
    hist, edges = np.histogram(z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = gaussian_filter1d(hist.astype(np.float32), sigma=sigma)

    peaks = _local_maxima(hs)
    valleys = _local_minima(hs)

    best = None
    for v in valleys:
        left_peaks = peaks[peaks < v]
        right_peaks = peaks[peaks > v]
        if left_peaks.size == 0 or right_peaks.size == 0:
            continue

        pl = left_peaks[-1]
        pr = right_peaks[0]

        valley_h = float(hs[v])
        peak_l_h = float(hs[pl])
        peak_r_h = float(hs[pr])
        min_peak_h = min(peak_l_h, peak_r_h)
        if min_peak_h <= 0:
            continue

        thr = float(centers[v])
        mask_left = z <= thr
        n_left = int(mask_left.sum())
        n_right = int(N - n_left)
        n_minor = min(n_left, n_right)
        frac_minor = n_minor / float(N)

        if n_left < int(min_child_n) or n_right < int(min_child_n):
            continue
        if frac_minor < float(min_group_frac):
            continue

        zl = z[mask_left]
        zr = z[~mask_left]
        mu_l = float(zl.mean())
        mu_r = float(zr.mean())
        sd_l = float(zl.std(ddof=1)) if zl.size > 1 else 1e-12
        sd_r = float(zr.std(ddof=1)) if zr.size > 1 else 1e-12
        pooled_sd = np.sqrt(0.5 * (sd_l * sd_l + sd_r * sd_r) + 1e-12)

        depth = 1.0 - (valley_h / (min_peak_h + 1e-12))
        sep = abs(mu_r - mu_l) / pooled_sd
        size_bonus = 1.0 - np.exp(-frac_minor / float(target_frac))
        score = depth * sep * size_bonus

        cand = {
            "score": float(score),
            "thr": float(thr),
            "depth": float(depth),
            "sep": float(sep),
            "size_bonus": float(size_bonus),
            "pooled_sd": float(pooled_sd),
            "mask_left_raw": mask_left,
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    return best

def _detect_channel_amp_from_snips(sn_sel, detect_row, center_idx):
    lo = max(0, int(center_idx) - 1)
    hi = min(sn_sel.shape[1] - 1, int(center_idx) + 1)
    return -sn_sel[int(detect_row), lo:hi + 1, :].min(axis=0).astype(np.float32)

def _sample_evenly_sorted_indices(n_total, n_keep):
    n_total = int(n_total)
    n_keep = int(min(n_keep, n_total))
    if n_keep <= 0:
        return np.array([], dtype=np.int64)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    idx = np.linspace(0, n_total - 1, n_keep)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, n_total - 1)
    idx = np.unique(idx)
    if idx.size == n_keep:
        return idx
    mask = np.ones(n_total, dtype=bool)
    mask[idx] = False
    extra = np.where(mask)[0]
    need = n_keep - idx.size
    if need > 0:
        idx = np.sort(np.concatenate([idx, extra[:need]]))
    return idx.astype(np.int64, copy=False)

# -----------------------------
# Pick top channels by amp difference
# -----------------------------
if "pp_topk_scan_results" not in globals():
    raise RuntimeError("pp_topk_scan_results not found. Run the one-shot PP scan cell first.")

rows = [r for r in pp_topk_scan_results if str(r.get("status", "")) == "ok"]
if len(rows) == 0:
    raise RuntimeError("No successful rows in pp_topk_scan_results.")

rows = sorted(rows, key=lambda r: float(r["mean_amp_left"] - r["mean_amp_right"]), reverse=True)
rows = rows[:int(top_n_channels)]

print(f"Plotting top {len(rows)} channels by (ampL - ampR)")

# -----------------------------
# Rebuild split and plot EI overlays
# -----------------------------
for i, row in enumerate(rows, start=1):
    ch = int(row["detect_ch"])
    rank_idx = int(row["skyline_rank"])

    x = data_matrix[:stop, ch].astype(np.float32, copy=False)
    idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    if idx.size == 0:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP no_minima")
        continue

    times_abs = idx.astype(np.int64)
    vals = x[idx].astype(np.float32, copy=False)

    ok = (times_abs + pre >= 0) & (times_abs + post < stop)
    times_abs = times_abs[ok]
    vals = vals[ok]
    if times_abs.size == 0:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP no_window_safe_minima")
        continue

    excl = None
    if channel_state.get(ch) is not None:
        excl = channel_state[ch].get("exclude_times", None)

    near = _times_near_exclusions(times_abs, excl, int(params_scan["exclude_radius_samples"]))
    times_keep = times_abs[~near]
    vals_keep = vals[~near]

    order_amp = np.argsort(vals_keep)[:min(int(topk_spikes_total), int(times_keep.size))]
    pool_times = times_keep[order_amp].astype(np.int64, copy=False)

    if pool_times.size < 2 * int(min_child_n):
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP pool too small")
        continue

    # EI subset for channel selection
    ei_subset_n = min(int(ei_subset_max), int(pool_times.size))
    ei_subset_idx = _sample_evenly_sorted_indices(pool_times.size, ei_subset_n)
    ei_subset_times = pool_times[ei_subset_idx]

    all_ch = np.arange(data_matrix.shape[1], dtype=np.int32)
    sn_ei, valid_ei = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=ei_subset_times,
        window=window_pp,
        selected_channels=all_ch,
    )
    sn_ei = sn_ei.astype(np.float32, copy=False)
    if sn_ei.shape[2] == 0:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP no EI subset")
        continue

    ei_pool = sn_ei.mean(axis=2).astype(np.float32)

    selected_channels, _ = select_template_channels(
        ei_pool,
        p2p_thr=p2p_thr,
        max_n=max_channels,
        min_n=min_channels,
        force_include_main=True,
    )
    selected_channels = np.asarray(selected_channels, dtype=np.int64)
    if ch not in selected_channels:
        selected_channels = np.concatenate([selected_channels, np.array([ch], dtype=np.int64)])
    selected_channels = np.asarray(np.unique(selected_channels), dtype=np.int64)
    detect_row = int(np.where(selected_channels == ch)[0][0])

    sn_sel, valid_pool = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=pool_times,
        window=window_pp,
        selected_channels=selected_channels.astype(np.int32),
    )
    sn_sel = sn_sel.astype(np.float32, copy=False)
    if valid_pool.size != pool_times.size:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP edge drop")
        continue

    X = sn_sel.transpose(2, 0, 1).reshape(sn_sel.shape[2], -1).astype(np.float32)
    n_pcs = int(min(n_search_pcs, X.shape[0], X.shape[1]))
    if n_pcs < 2:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP low rank")
        continue

    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(X).astype(np.float32)

    amp = _detect_channel_amp_from_snips(sn_sel, detect_row, center_idx)
    pcs_z, _, _ = _robust_zscore_cols_fit(pcs)
    amp_z, _, _ = _robust_zscore_1d_fit(amp)
    feat = np.concatenate([amp_z[:, None], pcs_z], axis=1).astype(np.float32)

    candidate_dirs = _build_biased_direction_bank(
        n_aug=feat.shape[1],
        n_pcs=n_pcs,
        rng=rng,
        amp_pc1_weight=amp_pc1_weight,
        n_tail_dirs=n_biased_tail_dirs,
        tail_scale=biased_tail_scale,
    )
    W_rand = _random_unit_directions(n_random_dirs, feat.shape[1], rng)
    for i_dir in range(n_random_dirs):
        candidate_dirs.append((f"random_{i_dir:04d}", W_rand[i_dir]))

    best = None
    best_name = None
    best_z = None

    for dir_name, w in candidate_dirs:
        z = feat @ w
        res = _score_projection_1d(
            z,
            bins=hist_bins,
            sigma=smooth_sigma,
            min_child_n=min_child_n,
            min_group_frac=min_group_frac,
            target_frac=target_frac,
        )
        if res is None:
            continue
        if (best is None) or (res["score"] > best["score"]):
            best = res
            best_name = str(dir_name)
            best_z = z.copy()

    if best is None:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP no valid split")
        continue

    mask_left_raw = np.asarray(best["mask_left_raw"], dtype=bool)
    amp_left_raw = amp[mask_left_raw]
    amp_right_raw = amp[~mask_left_raw]

    mean_amp_left_raw = float(np.mean(amp_left_raw)) if amp_left_raw.size else np.nan
    mean_amp_right_raw = float(np.mean(amp_right_raw)) if amp_right_raw.size else np.nan

    if mean_amp_left_raw >= mean_amp_right_raw:
        mask_left = mask_left_raw
        mask_right = ~mask_left_raw
        ampL = mean_amp_left_raw
        ampR = mean_amp_right_raw
    else:
        mask_left = ~mask_left_raw
        mask_right = mask_left_raw
        ampL = mean_amp_right_raw
        ampR = mean_amp_left_raw

    left_times = pool_times[mask_left]
    right_times = pool_times[mask_right]

    # limit to 1k spikes per leaf for EI
    left_keep_idx = _sample_evenly_sorted_indices(left_times.size, min(max_ei_spikes_per_leaf, left_times.size))
    right_keep_idx = _sample_evenly_sorted_indices(right_times.size, min(max_ei_spikes_per_leaf, right_times.size))

    left_times_ei = np.asarray(left_times[left_keep_idx], dtype=np.int64)
    right_times_ei = np.asarray(right_times[right_keep_idx], dtype=np.int64)

    sn_left, valid_left = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=left_times_ei,
        window=window_pp,
        selected_channels=all_ch,
    )
    sn_right, valid_right = extract_snippets_fast_ram(
        raw_data=data_matrix,
        spike_times=right_times_ei,
        window=window_pp,
        selected_channels=all_ch,
    )

    sn_left = sn_left.astype(np.float32, copy=False)
    sn_right = sn_right.astype(np.float32, copy=False)

    if sn_left.shape[2] == 0 or sn_right.shape[2] == 0:
        print(f"[{i}] ch={ch:3d} rank={rank_idx:3d} SKIP no valid EI snippets")
        continue

    ei_left = sn_left.mean(axis=2).astype(np.float32)
    ei_right = sn_right.mean(axis=2).astype(np.float32)

    print(
        f"[{i:2d}] ch={ch:3d} rank={rank_idx:3d} "
        f"score={best['score']:.3f} sep={best['sep']:.3f} depth={best['depth']:.3f} "
        f"nL={left_times.size} nR={right_times.size} "
        f"ampL={ampL:.1f} ampR={ampR:.1f} dir={best_name}"
    )

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_ei_waveforms(
        [ei_left, ei_right],
        ei_positions,
        ref_channel=ch,
        scale=70.0,
        box_height=1.0,
        box_width=50.0,
        ax=ax,
        colors=["black", "red"],
    )
    ax.set_title(
        f"ch {ch} (rank {rank_idx}) | one-shot PP top-K split | "
        f"score={best['score']:.3f}, sep={best['sep']:.3f}, depth={best['depth']:.3f}\n"
        f"left n={left_times.size}, amp={ampL:.1f}  |  right n={right_times.size}, amp={ampR:.1f}"
    )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # new loop

# %%
# === Stage-2 helpers for the looping PP ===

def _build_stage2_fallback_amp_split(all_times, all_vals, total_keep):
    """
    Take the `total_keep` most negative minima and split evenly by amplitude.
    First half (most negative) becomes faux-left, second half faux-right.

    Returns None if there are not enough minima to form two non-empty halves.
    """
    all_times = np.asarray(all_times, dtype=np.int64)
    all_vals = np.asarray(all_vals, dtype=np.float32)

    if all_times.size < 2:
        return None

    keep_n = min(int(total_keep), int(all_times.size))
    half_n = keep_n // 2
    keep_n = 2 * half_n
    if half_n == 0:
        return None

    amp_order = np.argsort(all_vals)[:keep_n]   # ascending => most negative first
    keep_times_amp = all_times[amp_order]
    keep_vals_amp = all_vals[amp_order]

    left_times_amp = keep_times_amp[:half_n]
    left_vals_amp = keep_vals_amp[:half_n]

    right_times_amp = keep_times_amp[half_n:]
    right_vals_amp = keep_vals_amp[half_n:]

    li = np.argsort(left_times_amp)
    ri = np.argsort(right_times_amp)

    return {
        "keep_n": int(keep_n),
        "half_n": int(half_n),
        "left_times": left_times_amp[li].astype(np.int64, copy=False),
        "left_vals": left_vals_amp[li].astype(np.float32, copy=False),
        "right_times": right_times_amp[ri].astype(np.int64, copy=False),
        "right_vals": right_vals_amp[ri].astype(np.float32, copy=False),
        "amp_cut_left_max": float(np.max(left_vals_amp)) if left_vals_amp.size else np.nan,
        "amp_cut_right_min": float(np.min(right_vals_amp)) if right_vals_amp.size else np.nan,
        "weakest_kept_val": float(np.max(keep_vals_amp)) if keep_vals_amp.size else np.nan,  # least-negative kept minimum
    }


def build_stage2_pool_for_channel(raw_mod, detect_ch, channel_state, params):
    """
    Stage-2 pool builder:
      - same strict LH valley locator as the single-channel notebook
      - same same-channel exclusion logic as the loop
      - fallback to top-K deepest minima if the left side is too small OR no valley is found

    Unlike stage 1, this does NOT require strict LH acceptance to proceed.
    """
    fs = int(params["fs"])
    duration_sec = float(params["pp_duration_sec"])
    stop = min(int(duration_sec * fs), raw_mod.shape[0])
    window_pp = tuple(params["window_pp"])
    exclusion_radius = int(params["exclude_radius_samples"])

    excl = np.array([], dtype=np.int64)
    if channel_state.get(int(detect_ch)) is not None:
        excl = np.asarray(
            channel_state[int(detect_ch)].get("exclude_times", np.array([], dtype=np.int64)),
            dtype=np.int64,
        )

    step1 = find_valley_and_times_excluding(
        raw_data=raw_mod,
        ch=int(detect_ch),
        window=window_pp,
        start=0,
        stop=stop,
        bin_width=float(params["valley_bin_width"]),
        valley_bins=int(params["valley_bins"]),
        min_valid_count=int(params["valley_min_valid_count"]),
        ratio_base=int(params["valley_ratio_base"]),
        ratio_step=int(params["valley_ratio_step"]),
        ratio_floor=int(params["valley_ratio_floor"]),
        ratio_cap=int(params["valley_ratio_cap"]),
        exclude_times=excl,
        exclude_radius=exclusion_radius,
    )

    all_times = np.asarray(step1.get("all_times", []), dtype=np.int64)
    all_vals = np.asarray(step1.get("all_vals", []), dtype=np.float32)
    if all_times.size == 0:
        return {
            "status": "fail",
            "reason": "empty_minima_after_exclusion",
            "step1": step1,
        }

    fallback_enable = bool(params.get("stage2_fallback_enable", True))
    fallback_left_min_count = int(params.get("stage2_fallback_left_min_count", 500))
    fallback_total_keep = int(params.get("stage2_fallback_total_keep", 10_000))

    valley_low = step1.get("valley_low", None)

    use_fallback = False
    if fallback_enable:
        if (valley_low is None) or (int(step1.get("left_count", 0)) < fallback_left_min_count):
            use_fallback = True

    if use_fallback:
        fb = _build_stage2_fallback_amp_split(
            all_times=all_times,
            all_vals=all_vals,
            total_keep=fallback_total_keep,
        )
        if fb is None:
            return {
                "status": "fail",
                "reason": "fallback_failed_not_enough_minima",
                "step1": step1,
            }

        left_times = fb["left_times"]
        right_times = fb["right_times"]
        left_vals = fb["left_vals"]
        right_vals = fb["right_vals"]

        full_times = np.concatenate([left_times, right_times]).astype(np.int64)
        full_side = np.concatenate([
            np.zeros(left_times.size, dtype=np.int32),
            np.ones(right_times.size, dtype=np.int32),
        ])

        effective_threshold_value = float(fb["weakest_kept_val"])
        effective_threshold_source = "fallback_weakest_kept"
        pool_builder = "stage2_fallback_topk_amp_split"
        fallback_info = fb
    else:
        valley_low = float(valley_low)

        left_mask = all_vals < valley_low
        right_mask = all_vals >= valley_low

        left_times = all_times[left_mask]
        left_vals = all_vals[left_mask]

        right_times_all = all_times[right_mask]
        right_vals_all = all_vals[right_mask]

        n_left = int(left_times.size)
        if n_left == 0:
            return {
                "status": "fail",
                "reason": "empty_left_pool",
                "step1": step1,
            }

        right_order = np.argsort(right_vals_all)   # ascending => strongest negative first
        right_keep = right_order[:min(n_left, right_times_all.size)]

        right_times = right_times_all[right_keep]
        right_vals = right_vals_all[right_keep]

        if right_times.size == 0:
            return {
                "status": "fail",
                "reason": "empty_right_pool",
                "step1": step1,
            }

        full_times = np.concatenate([left_times, right_times]).astype(np.int64)
        full_side = np.concatenate([
            np.zeros(left_times.size, dtype=np.int32),
            np.ones(right_times.size, dtype=np.int32),
        ])

        effective_threshold_value = float(valley_low)
        effective_threshold_source = "valley_low"
        pool_builder = "stage2_valley_match"
        fallback_info = None

    return {
        "status": "ok",
        "detect_ch": int(detect_ch),
        "stop": int(stop),
        "window_pp": tuple(window_pp),
        "step1": step1,
        "pool_builder": str(pool_builder),
        "fallback_used": bool(use_fallback),
        "fallback_info": fallback_info,
        "left_times": np.asarray(left_times, dtype=np.int64),
        "left_vals": np.asarray(left_vals, dtype=np.float32),
        "right_times": np.asarray(right_times, dtype=np.int64),
        "right_vals": np.asarray(right_vals, dtype=np.float32),
        "full_times": np.asarray(full_times, dtype=np.int64),
        "full_side": np.asarray(full_side, dtype=np.int32),
        "effective_threshold_value": float(effective_threshold_value),
        "effective_threshold_source": str(effective_threshold_source),
    }


def clone_stage2_recursive_params(params):
    """
    Clone the stage-1 PP params, then override only the stage-2 behavior:
      - depth up to 5
      - recurse while node has at least 100 spikes
      - no child-size gate
      - no minor-fraction gate
      - no AB/small-split detour
      - use the full node for discovery (effectively no subset cap)
    """
    p = dict(params)

    p["pp_max_depth"] = int(params.get("stage2_pp_max_depth", 5))
    p["pp_min_node_n"] = int(params.get("stage2_pp_min_node_n", 100))
    p["pp_min_child_n"] = int(params.get("stage2_pp_min_child_n", 1))
    p["pp_discover_max_per_side"] = int(params.get("stage2_pp_discover_max_per_side", 10_000_000))
    p["pp_min_group_frac"] = float(params.get("stage2_pp_min_group_frac", 0.0))
    p["pp_left_recurse_thresh"] = float(params.get("stage2_pp_left_recurse_thresh", 0.50))

    # Keep score/depth/separation thresholds explicit so stage-2 can be tuned separately if needed.
    p["pp_min_score"] = float(params.get("stage2_pp_min_score", params["pp_min_score"]))
    p["pp_min_depth"] = float(params.get("stage2_pp_min_depth", params["pp_min_depth"]))
    p["pp_min_sep"] = float(params.get("stage2_pp_min_sep", params["pp_min_sep"]))

    # Force every accepted split to be "scoreable", so the AB / tiny-split branch never runs.
    p["pp_scoreable_minor_frac"] = 0.0
    p["pp_scoreable_minor_n_cap"] = int(params.get("stage2_pp_scoreable_minor_n_cap", 10_000_000))

    return p


def collect_pp_child_summaries(pp_tree, out=None):
    """
    Map child path -> child summary, using the summaries already stored on split nodes.
    """
    if out is None:
        out = {}

    if pp_tree is None:
        return out
    if pp_tree.get("type", None) != "split":
        return out

    left_path = str(pp_tree["path"]) + ".L"
    right_path = str(pp_tree["path"]) + ".R"

    out[left_path] = dict(pp_tree.get("summary_left", {}))
    out[right_path] = dict(pp_tree.get("summary_right", {}))

    collect_pp_child_summaries(pp_tree.get("left_child", None), out)
    collect_pp_child_summaries(pp_tree.get("right_child", None), out)
    return out


def get_nodes_on_path_pp_tree(tree, leaf_path):
    parts = str(leaf_path).split(".")
    if len(parts) == 0 or parts[0] != "root":
        raise ValueError("Unexpected leaf_path={!r}".format(leaf_path))

    nodes = []
    node = tree
    for branch in parts[1:]:
        if node["type"] != "split":
            raise RuntimeError("Path hits a leaf too early")
        nodes.append((node, branch))
        node = node["left_child"] if branch == "L" else node["right_child"]
    return nodes


def build_lh_style_template_bank_from_times(raw_mod, times, window_pp, params):
    times = np.asarray(times, dtype=np.int64)
    if times.size == 0:
        raise RuntimeError("No spikes to build template bank")

    sn_full, valid_times = extract_snippets_fast_ram(
        raw_data=raw_mod,
        spike_times=times,
        window=tuple(window_pp),
        selected_channels=np.arange(raw_mod.shape[1], dtype=np.int32),
    )
    sn_full = sn_full.astype(np.float32, copy=False)
    valid_times = np.asarray(valid_times, dtype=np.int64)

    if valid_times.size != times.size:
        raise RuntimeError("Unexpected edge-drop while building template bank")

    template_reducer = str(params["short_bank_reducer"])
    template_n_bins = int(params["short_bank_n_bins"])
    template_min_bin_size = int(params["short_bank_min_bin_size"])

    if template_reducer == "median":
        provisional_ei = median_ei_adaptive(sn_full).astype(np.float32)
    elif template_reducer == "mean":
        provisional_ei = sn_full.mean(axis=2).astype(np.float32)
    else:
        raise ValueError("short_bank_reducer must be 'median' or 'mean'")

    main_ch = int(np.argmin(provisional_ei.min(axis=1)))
    t0 = int(np.argmin(provisional_ei[main_ch]))
    lo = max(0, t0 - 1)
    hi = min(sn_full.shape[1] - 1, t0 + 1)

    spike_amp = sn_full[main_ch, lo:hi + 1, :].min(axis=0).astype(np.float32)
    order = np.argsort(spike_amp)  # ascending => strongest negative first
    times_sorted = valid_times[order]
    amp_sorted = spike_amp[order]
    sn_sorted = sn_full[:, :, order]

    n_bins_eff = min(int(template_n_bins), max(1, int(sn_sorted.shape[2]) // int(template_min_bin_size)))
    groups = np.array_split(np.arange(sn_sorted.shape[2]), n_bins_eff)

    templates = []
    for bi, g in enumerate(groups):
        if g.size == 0:
            continue
        sn_bin = sn_sorted[:, :, g]
        if template_reducer == "median":
            tmpl = median_ei_adaptive(sn_bin).astype(np.float32)
        else:
            tmpl = sn_bin.mean(axis=2).astype(np.float32)

        templates.append({
            "bin_index": int(bi),
            "n_spikes": int(g.size),
            "times": times_sorted[g].astype(np.int64, copy=False),
            "amp_min": float(np.min(amp_sorted[g])),
            "amp_max": float(np.max(amp_sorted[g])),
            "template": tmpl,
        })

    return {
        "window": tuple(window_pp),
        "reducer": str(template_reducer),
        "main_ch": int(main_ch),
        "t0_main": int(t0),
        "provisional_ei": provisional_ei,
        "times_all": times_sorted.astype(np.int64, copy=False),
        "amp_all": amp_sorted.astype(np.float32, copy=False),
        "templates": templates,
    }


def score_single_time_against_template_bank(raw_mod, t, template_bank, lag_radius_local, channels=None):
    t = int(t)
    templates = template_bank["templates"]
    if len(templates) == 0:
        return np.inf, None, None

    if channels is None:
        ei_ref = np.asarray(template_bank["provisional_ei"], dtype=np.float32)
        p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
        channels = np.argsort(p2p)[-20:].astype(np.int32)
    else:
        channels = np.asarray(channels, dtype=np.int32)

    best_score = np.inf
    best_bin = None
    best_lag = None

    for lag in range(-int(lag_radius_local), int(lag_radius_local) + 1):
        tt = t + lag
        sn, valid_times = extract_snippets_fast_ram(
            raw_data=raw_mod,
            spike_times=np.array([tt], dtype=np.int64),
            window=tuple(template_bank["window"]),
            selected_channels=channels,
        )
        sn = sn.astype(np.float32, copy=False)
        if sn.shape[2] != 1:
            continue
        x = sn[:, :, 0]
        for rec in templates:
            tmpl = np.asarray(rec["template"], dtype=np.float32)[channels, :]
            resid = x - tmpl
            score = float(np.sqrt(np.mean(resid ** 2)))
            if score < best_score:
                best_score = score
                best_bin = int(rec["bin_index"])
                best_lag = int(lag)

    return best_score, best_bin, best_lag


def collapse_close_times_by_template_fit_from_bank(raw_mod, times, template_bank, params):
    times = np.asarray(times, dtype=np.int64)
    if times.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            [],
        )

    close_dt = int(params["trusted_close_dt"])
    lag_radius = int(params["assign_lag_radius"])

    t_sorted = np.sort(times)
    groups = []
    start = 0
    for i in range(t_sorted.size - 1):
        if (t_sorted[i + 1] - t_sorted[i]) >= int(close_dt):
            groups.append(t_sorted[start:i + 1])
            start = i + 1
    groups.append(t_sorted[start:])

    ei_ref = np.asarray(template_bank["provisional_ei"], dtype=np.float32)
    p2p = ei_ref.max(axis=1) - ei_ref.min(axis=1)
    score_channels = np.argsort(p2p)[-20:].astype(np.int32)

    kept = []
    dropped = []
    records = []

    for g in groups:
        if g.size == 1:
            kept.append(int(g[0]))
            records.append({
                "group_times": g.astype(np.int64, copy=False),
                "winner_time": int(g[0]),
                "winner_score": np.nan,
                "winner_bin": None,
                "winner_lag": None,
                "n_group": 1,
            })
            continue

        cand_scores = []
        for t in g:
            sc, bi, lg = score_single_time_against_template_bank(
                raw_mod=raw_mod,
                t=int(t),
                template_bank=template_bank,
                lag_radius_local=lag_radius,
                channels=score_channels,
            )
            cand_scores.append((float(sc), int(t), bi, lg))

        cand_scores.sort(key=lambda x: x[0])
        winner_score, winner_time, winner_bin, winner_lag = cand_scores[0]
        kept.append(int(winner_time))
        for _, t, _, _ in cand_scores[1:]:
            dropped.append(int(t))

        records.append({
            "group_times": g.astype(np.int64, copy=False),
            "winner_time": int(winner_time),
            "winner_score": float(winner_score),
            "winner_bin": winner_bin,
            "winner_lag": winner_lag,
            "n_group": int(g.size),
            "all_scores": cand_scores,
        })

    kept = np.array(sorted(set(kept)), dtype=np.int64)
    dropped = np.array(sorted(set(dropped)), dtype=np.int64)
    return kept, dropped, records


def finalize_stage2_main_leaf(raw_mod, stage2_tree_state, pool_state, params):
    """
    Stage-2 finalization:
      - pick the largest majority-left leaf
      - do NOT create uncertainty or extra-left deferrals
      - only keep close-neighbor cleanup before assignment
    """
    pp_tree = stage2_tree_state["tree"]
    pp_leaves = stage2_tree_state["leaves"]
    full_times = np.asarray(stage2_tree_state["full_times"], dtype=np.int64)
    full_side = np.asarray(stage2_tree_state["full_side"], dtype=np.int32)
    anchor_ch = int(stage2_tree_state["anchor_ch"])
    window_pp = tuple(stage2_tree_state["window_pp"])

    left_leaf_thresh = float(params.get("stage2_main_leaf_thresh", 0.50))

    summary_map = collect_pp_child_summaries(pp_tree)
    candidates = []

    for leaf in pp_leaves:
        idx = np.asarray(leaf.get("idx", []), dtype=np.int64)
        if idx.size == 0:
            continue

        summary = summary_map.get(str(leaf["path"]), None)
        if summary is None:
            n = int(idx.size)
            n_orig_left = int(np.sum(full_side[idx] == 0))
            n_orig_right = int(np.sum(full_side[idx] == 1))
            frac_orig_left = n_orig_left / float(n) if n > 0 else np.nan
            frac_orig_right = n_orig_right / float(n) if n > 0 else np.nan

            if not np.isfinite(frac_orig_left) or (frac_orig_left <= left_leaf_thresh):
                continue

            leaf_times = full_times[idx]
            sn_full, valid_times = extract_snippets_fast_ram(
                raw_data=raw_mod,
                spike_times=leaf_times,
                window=window_pp,
                selected_channels=np.arange(raw_mod.shape[1], dtype=np.int32),
            )
            sn_full = sn_full.astype(np.float32, copy=False)
            if sn_full.shape[2] == 0:
                continue

            ei_leaf = sn_full.mean(axis=2).astype(np.float32)
            amp_anchor = float(-ei_leaf[anchor_ch].min())

            summary = {
                "n": int(n),
                "n_orig_left": int(n_orig_left),
                "n_orig_right": int(n_orig_right),
                "frac_orig_left": float(frac_orig_left),
                "frac_orig_right": float(frac_orig_right),
                "amp_anchor": float(amp_anchor),
                "majority_left": bool(frac_orig_left > left_leaf_thresh),
                "ei": ei_leaf,
            }

        frac_left = float(summary.get("frac_orig_left", np.nan))
        if not np.isfinite(frac_left):
            continue
        if frac_left <= left_leaf_thresh:
            continue

        candidates.append((leaf, summary))

    if len(candidates) == 0:
        raise RuntimeError("No majority-left leaves found in stage 2.")

    candidates.sort(
        key=lambda t: (
            int(t[1]["n"]),
            int(t[1]["n_orig_left"]),
            float(t[1]["amp_anchor"]),
            float(t[1]["frac_orig_left"]),
        ),
        reverse=True,
    )

    main_leaf, main_summary = candidates[0]
    main_path = str(main_leaf["path"])

    path_nodes = get_nodes_on_path_pp_tree(pp_tree, main_path)
    if len(path_nodes) > 0:
        last_node = path_nodes[-1][0]
        main_path_score = float(last_node["best_full"]["score"])
        main_path_sep = float(last_node["best_full"]["sep"])
        main_path_depth = float(last_node["best_full"]["depth"])
        main_path_split_path = str(last_node["path"])

        last_path_score = float(main_path_score)
        last_path_sep = float(main_path_sep)
        last_path_depth = float(main_path_depth)
        last_path_split_path = str(main_path_split_path)
    else:
        main_path_score = np.nan
        main_path_sep = np.nan
        main_path_depth = np.nan
        main_path_split_path = None

        last_path_score = np.nan
        last_path_sep = np.nan
        last_path_depth = np.nan
        last_path_split_path = None

    trusted_left_times_raw = full_times[np.asarray(main_leaf["idx"], dtype=np.int64)].astype(np.int64, copy=False)
    if trusted_left_times_raw.size == 0:
        raise RuntimeError("Chosen stage-2 leaf has no spikes.")

    lh_template_bank_provisional = build_lh_style_template_bank_from_times(
        raw_mod=raw_mod,
        times=trusted_left_times_raw,
        window_pp=window_pp,
        params=params,
    )

    trusted_left_times_clean, trusted_left_times_dropped_close, close_group_records = (
        collapse_close_times_by_template_fit_from_bank(
            raw_mod=raw_mod,
            times=trusted_left_times_raw,
            template_bank=lh_template_bank_provisional,
            params=params,
        )
    )

    if trusted_left_times_clean.size == 0:
        raise RuntimeError("No stage-2 trusted_left spikes remain after close-neighbor cleanup.")

    lh_template_bank = build_lh_style_template_bank_from_times(
        raw_mod=raw_mod,
        times=trusted_left_times_clean,
        window_pp=window_pp,
        params=params,
    )

    final_left_state = {
        "anchor_ch": int(anchor_ch),
        "window_pp": tuple(window_pp),
        "step1": pool_state["step1"],
        "main_leaf_path": str(main_path),
        "margin_k": 0.0,
        "trusted_left_times_raw": trusted_left_times_raw.astype(np.int64, copy=False),
        "trusted_left_times": trusted_left_times_clean.astype(np.int64, copy=False),
        "trusted_left_times_dropped_close": trusted_left_times_dropped_close.astype(np.int64, copy=False),
        "close_group_records": close_group_records,
        "isi_10_30_trusted_left": int(_count_isi_10_30(trusted_left_times_clean)),
        "extra_left_times": np.array([], dtype=np.int64),
        "uncertain_times": np.array([], dtype=np.int64),
        "deferred_small_split_times": np.array([], dtype=np.int64),
        "trusted_not_left_times": np.array([], dtype=np.int64),
        "majority_left_leaves": [str(leaf["path"]) for leaf, _summary in candidates],
        "extra_left_leaf_paths": [],
        "deferred_small_split_leaf_paths": [],
        "uncertain_source_paths": [],
        "main_path_score": float(main_path_score),
        "main_path_sep": float(main_path_sep),
        "main_path_depth": float(main_path_depth),
        "main_path_split_path": main_path_split_path,
        "last_path_score": float(last_path_score),
        "last_path_sep": float(last_path_sep),
        "last_path_depth": float(last_path_depth),
        "last_path_split_path": last_path_split_path,
    }

    meta = {
        "main_leaf": main_leaf,
        "main_leaf_summary": main_summary,
        "summary_map": summary_map,
    }
    return final_left_state, lh_template_bank, meta


def stage2_amp_floor_guard(raw_mod, trusted_left_times, main_ch, t0_main, pool_state, params):
    """
    Weakest-spikes-vs-pool-threshold guard.

    We compare the mean amplitude of the weakest few spikes in the candidate leaf
    against the effective amplitude threshold that was used to build the full pool
    on this detect channel.

    Amplitudes are compared as positive trough magnitudes.
    """
    trusted_left_times = np.asarray(trusted_left_times, dtype=np.int64)
    if trusted_left_times.size == 0:
        return {
            "ok": False,
            "reason": "empty_trusted_left_times",
        }

    sn_main, valid_times = extract_snippets_fast_ram(
        raw_data=raw_mod,
        spike_times=trusted_left_times,
        window=tuple(pool_state["window_pp"]),
        selected_channels=np.array([int(main_ch)], dtype=np.int32),
    )
    sn_main = sn_main.astype(np.float32, copy=False)
    valid_times = np.asarray(valid_times, dtype=np.int64)

    if valid_times.size != trusted_left_times.size:
        return {
            "ok": False,
            "reason": "edge_drop_in_amp_floor_guard",
        }

    lo = max(0, int(t0_main) - 1)
    hi = min(sn_main.shape[1] - 1, int(t0_main) + 1)
    amp_pos = (-np.min(sn_main[0, lo:hi + 1, :], axis=0)).astype(np.float32)

    frac = float(params.get("stage2_smallest_frac", 0.05))
    smallest_min_n = int(params.get("stage2_smallest_min_n", 10))
    n_check = max(int(np.ceil(frac * amp_pos.size)), smallest_min_n)
    n_check = min(n_check, int(amp_pos.size))

    weakest_mean_amp = float(np.mean(np.sort(amp_pos)[:n_check]))

    thr_val = pool_state.get("effective_threshold_value", None)
    thr_src = pool_state.get("effective_threshold_source", None)

    if thr_val is None:
        return {
            "ok": True,
            "reason": "no_effective_threshold",
            "weakest_mean_amp": float(weakest_mean_amp),
            "threshold_amp": np.nan,
            "gap": np.nan,
            "min_gap": float(params.get("stage2_threshold_gap_min_adc", 15.0)),
            "n_check": int(n_check),
            "frac": float(frac),
            "threshold_source": thr_src,
        }

    threshold_amp = float(-float(thr_val))
    gap = float(weakest_mean_amp - threshold_amp)
    min_gap = float(params.get("stage2_threshold_gap_min_adc", 15.0))
    ok = bool(gap >= min_gap)

    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "gap_too_small",
        "weakest_mean_amp": float(weakest_mean_amp),
        "threshold_amp": float(threshold_amp),
        "gap": float(gap),
        "min_gap": float(min_gap),
        "n_check": int(n_check),
        "frac": float(frac),
        "threshold_source": thr_src,
        "threshold_value": float(thr_val),
        "main_ch": int(main_ch),
    }


def note_channel_skip_with_wait(
    channel_state,
    detect_ch,
    success_count,
    params,
    reason=None,
    step1=None,
    wait_for_main_ch=None,
):
    st = note_channel_skip(
        channel_state=channel_state,
        detect_ch=detect_ch,
        success_count=success_count,
        params=params,
        reason=reason,
        step1=step1,
    )
    if wait_for_main_ch is None:
        st["wait_for_main_ch"] = None
    else:
        st["wait_for_main_ch"] = int(wait_for_main_ch)
    return st


def channel_is_retry_eligible_two_stage(
    channel_state,
    detect_ch,
    success_count,
    params,
    processed_detect_channels=None,
):
    if not channel_is_retry_eligible(channel_state, detect_ch, success_count, params):
        return False

    st = _ensure_channel_state_entry(channel_state, detect_ch)
    wait_for_main_ch = st.get("wait_for_main_ch", None)
    if wait_for_main_ch is None:
        return True

    processed_detect_channels = set() if processed_detect_channels is None else {
        int(ch) for ch in processed_detect_channels
    }
    if int(wait_for_main_ch) in processed_detect_channels:
        st["wait_for_main_ch"] = None
        st["needs_retry"] = True
        return True

    return False


def wake_channels_waiting_on_main(channel_state, processed_detect_ch):
    processed_detect_ch = int(processed_detect_ch)
    woke = []

    for ch, st in channel_state.items():
        if st.get("wait_for_main_ch", None) == processed_detect_ch:
            st["wait_for_main_ch"] = None
            st["needs_retry"] = True
            woke.append(int(ch))

    return np.array(sorted(set(woke)), dtype=np.int64)


def attempt_one_channel_pp_unit_two_stage(
    raw_mod,
    ei_positions,
    detect_ch,
    channel_state,
    params,
    processed_detect_channels=None,
):
    """
    Two-stage channel attempt:
      - stage 1 = current loop PP
      - stage 2 = single-channel-like recursive fallback
    """
    detect_ch = int(detect_ch)
    processed_detect_channels = set() if processed_detect_channels is None else {
        int(ch) for ch in processed_detect_channels
    }

    stage1_res = attempt_one_channel_pp_unit(
        raw_mod=raw_mod,
        ei_positions=ei_positions,
        detect_ch=detect_ch,
        channel_state=channel_state,
        params=params,
    )

    if stage1_res["status"] == "ok":
        stage1_res["unit_record"]["pp_stage"] = "stage1"
        return stage1_res

    if not bool(params.get("stage2_enable", True)):
        return stage1_res

    stage1_reason = str(stage1_res.get("reason", "stage1_fail"))

    pool2 = build_stage2_pool_for_channel(
        raw_mod=raw_mod,
        detect_ch=detect_ch,
        channel_state=channel_state,
        params=params,
    )
    if pool2["status"] != "ok":
        return {
            "status": "fail",
            "reason": "{} [stage1={}]".format(pool2["reason"], stage1_reason),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", stage1_res.get("step1", None)),
            "stage1_reason": stage1_reason,
        }

    params2 = clone_stage2_recursive_params(params)
    pp_tree_state2 = run_recursive_pp_for_channel(
        raw_mod=raw_mod,
        ei_positions=ei_positions,
        pp_pool_state=pool2,
        params=params2,
    )

    if pp_tree_state2["tree"]["type"] != "split":
        return {
            "status": "fail",
            "reason": "stage2_{} [stage1={}]".format(pp_tree_state2["tree"]["reason"], stage1_reason),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "stage1_reason": stage1_reason,
        }

    try:
        final_left_state2, lh_template_bank2, stage2_meta = finalize_stage2_main_leaf(
            raw_mod=raw_mod,
            stage2_tree_state=pp_tree_state2,
            pool_state=pool2,
            params=params,
        )
    except Exception as exc:
        return {
            "status": "fail",
            "reason": "stage2_finalize_error: {} [stage1={}]".format(str(exc), stage1_reason),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "stage1_reason": stage1_reason,
        }

    stage2_main_ch = int(lh_template_bank2["main_ch"])
    if stage2_main_ch != detect_ch:
        if stage2_main_ch not in processed_detect_channels:
            return {
                "status": "fail",
                "reason": "stage2_wait_main_ch_{} [stage1={}]".format(stage2_main_ch, stage1_reason),
                "detect_ch": int(detect_ch),
                "wait_for_main_ch": int(stage2_main_ch),
                "step1": pool2.get("step1", None),
                "stage1_reason": stage1_reason,
            }
        return {
            "status": "fail",
            "reason": "stage2_main_ch_mismatch_processed(main={}) [stage1={}]".format(stage2_main_ch, stage1_reason),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "stage1_reason": stage1_reason,
        }

    amp_guard = stage2_amp_floor_guard(
        raw_mod=raw_mod,
        trusted_left_times=final_left_state2["trusted_left_times"],
        main_ch=lh_template_bank2["main_ch"],
        t0_main=lh_template_bank2["t0_main"],
        pool_state=pool2,
        params=params,
    )
    if not amp_guard["ok"]:
        gap = amp_guard.get("gap", np.nan)
        min_gap = amp_guard.get("min_gap", np.nan)
        return {
            "status": "fail",
            "reason": "stage2_amp_floor_guard(gap={:.1f}<min={:.1f}) [stage1={}]".format(
                float(gap) if np.isfinite(gap) else np.nan,
                float(min_gap) if np.isfinite(min_gap) else np.nan,
                stage1_reason,
            ),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "stage2_amp_guard": amp_guard,
            "stage1_reason": stage1_reason,
        }

    preview_res = assign_and_subtract_unit(
        raw_mod=raw_mod,
        final_left_state=final_left_state2,
        lh_template_bank=lh_template_bank2,
        detect_ch=detect_ch,
        params=params,
        dry_run=True,
    )
    if preview_res["status"] != "ok":
        return {
            "status": "fail",
            "reason": "stage2_{} [stage1={}]".format(preview_res["reason"], stage1_reason),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "stage2_amp_guard": amp_guard,
            "stage1_reason": stage1_reason,
        }

    preview_unit = preview_res["unit_record"]

    max_isi_10_30_main = int(params["max_isi_10_30_main"])
    isi_10_30_main = int(preview_unit.get("isi_10_30_main", 0))
    if isi_10_30_main > max_isi_10_30_main:
        return {
            "status": "fail",
            "reason": "stage2_isi_violation_skip(isi10_30={}>{}) [stage1={}]".format(
                int(isi_10_30_main),
                int(max_isi_10_30_main),
                stage1_reason,
            ),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "unit_record": preview_unit,
            "stage2_amp_guard": amp_guard,
            "stage1_reason": stage1_reason,
        }

    assign_res = assign_and_subtract_unit(
        raw_mod=raw_mod,
        final_left_state=final_left_state2,
        lh_template_bank=lh_template_bank2,
        detect_ch=detect_ch,
        params=params,
        dry_run=False,
    )
    if assign_res["status"] != "ok":
        return {
            "status": "fail",
            "reason": "stage2_{} [stage1={}]".format(assign_res["reason"], stage1_reason),
            "detect_ch": int(detect_ch),
            "step1": pool2.get("step1", None),
            "preflight": None,
            "stage2_amp_guard": amp_guard,
            "stage1_reason": stage1_reason,
        }

    unit_record = assign_res["unit_record"]
    unit_record["pp_stage"] = "stage2"
    unit_record["stage2_pool_builder"] = str(pool2["pool_builder"])
    unit_record["stage2_fallback_used"] = bool(pool2["fallback_used"])
    unit_record["stage2_effective_threshold_value"] = float(pool2["effective_threshold_value"])
    unit_record["stage2_effective_threshold_source"] = str(pool2["effective_threshold_source"])
    unit_record["stage2_amp_guard"] = dict(amp_guard)
    unit_record["stage2_main_leaf_path"] = str(final_left_state2["main_leaf_path"])
    unit_record["stage2_main_leaf_n"] = int(stage2_meta["main_leaf_summary"]["n"])
    unit_record["stage2_main_leaf_frac_left"] = float(stage2_meta["main_leaf_summary"]["frac_orig_left"])

    return {
        "status": "ok",
        "detect_ch": int(detect_ch),
        "unit_record": unit_record,
    }

# %%
# === Stage-2 params for the looping PP ===
LH_PP_LOOP_PARAMS.update({
    # Enable / disable the second stage entirely
    "stage2_enable": True,

    # Stage-2 pool building: single-channel-style fallback
    "stage2_fallback_enable": True,
    "stage2_fallback_left_min_count": 500,
    "stage2_fallback_total_keep": 10_000,

    # Stage-2 recursion: mimic the single-channel notebook
    "stage2_pp_max_depth": 5,
    "stage2_pp_min_node_n": 100,          # only rule: do not recurse below this node size
    "stage2_pp_min_child_n": 1,           # no child-size gate
    "stage2_pp_min_group_frac": 0.0,      # no child-fraction gate
    "stage2_pp_discover_max_per_side": 10_000_000,  # effectively "use the full node"
    "stage2_pp_left_recurse_thresh": 0.50,

    # Stage-2 split quality thresholds (kept explicit so they are easy to tune)
    "stage2_pp_min_score": 0.90,
    "stage2_pp_min_depth": 0.20,
    "stage2_pp_min_sep": 1.15,

    # Pick the largest leaf that is majority-left by at least this fraction
    "stage2_main_leaf_thresh": 0.50,

    # Amplitude-floor guard:
    # weakest 5% of spikes on the leaf main channel must still sit this far above
    # the effective pool threshold used on the detect channel
    "stage2_smallest_frac": 0.05,
    "stage2_smallest_min_n": 10,
    "stage2_threshold_gap_min_adc": 15.0,
})

print("Stage-2 PP enabled:", LH_PP_LOOP_PARAMS["stage2_enable"])
print("Stage-2 fallback keep:", LH_PP_LOOP_PARAMS["stage2_fallback_total_keep"])
print("Stage-2 max depth:", LH_PP_LOOP_PARAMS["stage2_pp_max_depth"])


