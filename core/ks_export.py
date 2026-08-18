"""
ks_export.py — Write Lighthouse QC results in Phy / Kilosort folder format.

Produces a directory openable by Phy and other tools that expect:
  spike_times.npy, spike_clusters.npy, spike_templates.npy, templates.npy,
  amplitudes.npy, channel_map.npy, channel_positions.npy,
  similar_templates.npy, params.py, cluster_group.tsv
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

import numpy as np


REQUIRED_RESULT_ATTRS = ("final_times", "channel")


def _as_col(arr: np.ndarray, dtype) -> np.ndarray:
    a = np.asarray(arr, dtype=dtype).reshape(-1)
    return a.reshape(-1, 1)


def _default_positions(n_channels: int) -> np.ndarray:
    pos = np.zeros((n_channels, 2), dtype=np.float64)
    pos[:, 1] = np.arange(n_channels, dtype=np.float64) * 50.0
    return pos


def _unit_template(
    result,
    n_channels: int,
    n_timepoints: int,
) -> np.ndarray:
    """Build [n_timepoints, n_channels] template for one unit."""
    ei = getattr(result, "final_ei", None)
    tmpl = np.zeros((n_timepoints, n_channels), dtype=np.float32)
    if ei is None:
        # Fall back to detect-channel mean waveform from KMeans if present
        means = getattr(getattr(result, "pca_km", None), "cluster_mean_waveforms", None)
        if means:
            w = np.asarray(means[0], dtype=np.float32).ravel()
            L = min(n_timepoints, w.size)
            ch = int(result.channel)
            if 0 <= ch < n_channels:
                tmpl[:L, ch] = w[:L]
        return tmpl

    ei = np.asarray(ei, dtype=np.float32)
    if ei.ndim != 2:
        return tmpl
    # final_ei is [n_channels, n_samples]
    C = min(n_channels, ei.shape[0])
    L = min(n_timepoints, ei.shape[1])
    tmpl[:L, :C] = ei[:C, :L].T
    return tmpl


def _unit_amplitudes(result, n_spikes: int) -> np.ndarray:
    """Per-spike amplitude estimates (abs trough on detect channel when possible)."""
    valley = getattr(result, "valley", None)
    times = np.asarray(result.final_times, dtype=np.int64)
    if valley is not None and times.size:
        left_t = np.asarray(getattr(valley, "left_times", []), dtype=np.int64)
        left_v = np.asarray(getattr(valley, "left_vals", []), dtype=np.float32)
        if left_t.size == left_v.size and left_t.size > 0:
            # Map final_times → abs(left_vals) where possible
            order = np.argsort(left_t)
            left_t_s = left_t[order]
            left_v_s = left_v[order]
            idx = np.searchsorted(left_t_s, times)
            ok = (idx < left_t_s.size) & (left_t_s[np.minimum(idx, left_t_s.size - 1)] == times)
            amps = np.full(times.size, np.nan, dtype=np.float64)
            amps[ok] = np.abs(left_v_s[idx[ok]].astype(np.float64))
            if np.isfinite(amps).any():
                fill = float(np.nanmedian(amps[np.isfinite(amps)]))
                amps[~np.isfinite(amps)] = fill
                return amps
    # Fallback: biophysics median amp or 1.0
    bio = getattr(result, "biophysics", {}) or {}
    med = float(bio.get("median_amp_adc", 1.0) or 1.0)
    return np.full(n_spikes, med, dtype=np.float64)


def collect_exportable_units(
    qc_results: Mapping[int, object],
    *,
    min_spikes: int = 1,
) -> list[tuple[int, object]]:
    """Return [(unit_id, result), ...] for accepted channels with final_times."""
    units: list[tuple[int, object]] = []
    unit_id = 0
    for ch in sorted(qc_results.keys()):
        r = qc_results[ch]
        if getattr(r, "reject_reason", None):
            continue
        ft = getattr(r, "final_times", None)
        if ft is None or np.asarray(ft).size < min_spikes:
            continue
        if getattr(r, "n_lh", 0) <= 0 and np.asarray(ft).size == 0:
            continue
        units.append((unit_id, r))
        unit_id += 1
    return units


def export_phy_folder(
    out_dir: str,
    qc_results: Mapping[int, object],
    *,
    n_channels: int,
    fs: float = 20_000.0,
    channel_positions: Optional[np.ndarray] = None,
    dat_path: Optional[str] = None,
    dtype: str = "int16",
    n_timepoints: Optional[int] = None,
    min_spikes: int = 1,
) -> dict:
    """
    Write a Phy/Kilosort-compatible folder from QC results.

    Parameters
    ----------
    out_dir : str
        Destination directory (created if needed).
    qc_results : {channel: QCResult}
        Results from batch or single-channel QC. Only accepted channels with
        non-empty ``final_times`` are exported (one unit per channel).
    n_channels : int
        Probe channel count (must match raw recording).
    fs : float
        Sample rate in Hz.
    channel_positions : optional [n_channels, 2]
        Electrode xy positions in microns.
    dat_path : optional str
        Path recorded in params.py for Phy.
    dtype : str
        Raw data dtype string for params.py.
    n_timepoints : optional int
        Template length. Inferred from final_ei / window when omitted.
    min_spikes : int
        Skip units with fewer than this many final spikes.

    Returns
    -------
    dict with keys: out_dir, n_units, n_spikes, unit_channels
    """
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")

    units = collect_exportable_units(qc_results, min_spikes=min_spikes)
    if not units:
        raise ValueError(
            "No exportable units: need accepted QC results with non-empty final_times."
        )

    # Infer template length
    if n_timepoints is None:
        n_timepoints = 61
        for _, r in units:
            ei = getattr(r, "final_ei", None)
            if ei is not None and np.asarray(ei).ndim == 2:
                n_timepoints = int(np.asarray(ei).shape[1])
                break
            means = getattr(getattr(r, "pca_km", None), "cluster_mean_waveforms", None)
            if means:
                n_timepoints = int(np.asarray(means[0]).size)
                break

    os.makedirs(out_dir, exist_ok=True)

    all_times: list[np.ndarray] = []
    all_clusters: list[np.ndarray] = []
    all_amps: list[np.ndarray] = []
    templates = np.zeros((len(units), n_timepoints, n_channels), dtype=np.float32)
    unit_channels: list[int] = []

    for unit_id, r in units:
        times = np.sort(np.unique(np.asarray(r.final_times, dtype=np.int64)))
        n = int(times.size)
        all_times.append(times)
        all_clusters.append(np.full(n, unit_id, dtype=np.int32))
        all_amps.append(_unit_amplitudes(r, n))
        templates[unit_id] = _unit_template(r, n_channels, n_timepoints)
        unit_channels.append(int(r.channel))

    spike_times = np.concatenate(all_times).astype(np.uint64)
    spike_clusters = np.concatenate(all_clusters).astype(np.int32)
    amplitudes = np.concatenate(all_amps).astype(np.float64)

    order = np.argsort(spike_times, kind="mergesort")
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]
    amplitudes = amplitudes[order]
    spike_templates = spike_clusters.copy()  # 1 template per unit

    n_units = templates.shape[0]
    similar = np.eye(n_units, dtype=np.float32)

    if channel_positions is None:
        positions = _default_positions(n_channels)
    else:
        positions = np.asarray(channel_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < n_channels:
            raise ValueError(
                f"channel_positions must be [n_channels, 2]; got {positions.shape}"
            )
        positions = positions[:n_channels, :2]

    channel_map = np.arange(n_channels, dtype=np.int32)

    np.save(os.path.join(out_dir, "spike_times.npy"), _as_col(spike_times, np.uint64))
    np.save(os.path.join(out_dir, "spike_clusters.npy"), _as_col(spike_clusters, np.int32))
    np.save(os.path.join(out_dir, "spike_templates.npy"), _as_col(spike_templates, np.int32))
    np.save(os.path.join(out_dir, "amplitudes.npy"), _as_col(amplitudes, np.float64))
    np.save(os.path.join(out_dir, "templates.npy"), templates)
    np.save(os.path.join(out_dir, "similar_templates.npy"), similar)
    np.save(os.path.join(out_dir, "channel_map.npy"), _as_col(channel_map, np.int32))
    np.save(os.path.join(out_dir, "channel_positions.npy"), positions)

    # Optional: template peak channel lookup used by some tools
    np.save(
        os.path.join(out_dir, "templates_ind.npy"),
        np.tile(channel_map, (n_units, 1)).astype(np.int32),
    )

    # cluster_group.tsv — mark all as good
    with open(os.path.join(out_dir, "cluster_group.tsv"), "w", encoding="utf-8") as f:
        f.write("cluster_id\tgroup\n")
        for uid in range(n_units):
            f.write(f"{uid}\tgood\n")

    # cluster_info.tsv with detect channel for convenience
    with open(os.path.join(out_dir, "cluster_info.tsv"), "w", encoding="utf-8") as f:
        f.write("cluster_id\tch\tn_spikes\tgroup\n")
        for uid, ch in enumerate(unit_channels):
            n_sp = int(np.sum(spike_clusters == uid))
            f.write(f"{uid}\t{ch}\t{n_sp}\tgood\n")

    # params.py (Phy / KS style)
    dat_path_repr = repr(dat_path) if dat_path else "None"
    params_text = (
        f"dat_path = {dat_path_repr}\n"
        f"n_channels_dat = {int(n_channels)}\n"
        f"dtype = '{dtype}'\n"
        f"offset = 0\n"
        f"sample_rate = {float(fs):.1f}\n"
        f"hp_filtered = True\n"
    )
    with open(os.path.join(out_dir, "params.py"), "w", encoding="utf-8") as f:
        f.write(params_text)

    # sidecar: map unit → original detect channel
    np.save(
        os.path.join(out_dir, "lh_unit_detect_channels.npy"),
        np.asarray(unit_channels, dtype=np.int32),
    )

    return {
        "out_dir": out_dir,
        "n_units": n_units,
        "n_spikes": int(spike_times.size),
        "unit_channels": unit_channels,
    }
