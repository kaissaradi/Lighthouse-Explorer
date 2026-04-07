"""
loader.py — raw data loading utilities for Lighthouse QC.

Adapted from axolotl/io.py. All functions are pure (no Qt, no side effects).
"""
from __future__ import annotations
from typing import Optional
import os
import numpy as np


def load_raw_readonly(
    dat_path: str,
    n_channels: int,
    dtype: str = "int16",
    start_min: float = 0.0,
    duration_min: float | None = None,
    fs: int = 20_000,
) -> np.memmap:
    """
    Memory-map a chunk of a raw binary file as READ-ONLY [T, C].

    Parameters
    ----------
    dat_path : str
        Path to .dat / .bin file (time-major, interleaved channels).
    n_channels : int
        Number of channels in the recording.
    dtype : str
        NumPy dtype string (default "int16").
    start_min : float
        Start offset in minutes.
    duration_min : float | None
        Duration to load in minutes. None = load entire file.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    np.memmap of shape (T, C), mode="r".
    """
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"Recording not found: {dat_path}")

    file_bytes = os.path.getsize(dat_path)
    itemsize = np.dtype(dtype).itemsize
    total_samples = file_bytes // (n_channels * itemsize)

    start_sample = int(start_min * 60.0 * fs)
    start_sample = max(0, min(start_sample, total_samples))

    if duration_min is not None and duration_min > 0:
        n_samples = int(duration_min * 60.0 * fs)
    else:
        n_samples = total_samples - start_sample

    n_samples = max(1, min(n_samples, total_samples - start_sample))

    offset = start_sample * n_channels * itemsize

    mem = np.memmap(
        dat_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=(n_samples, n_channels),
    )
    return mem


def load_channel_map(map_path: Optional[str], n_channels: int) -> np.ndarray:
    """
    Load electrode positions from .npy file.

    Parameters
    ----------
    map_path : Optional[str]
        Path to .npy file containing [C, 2] positions.
    n_channels : int
        Fallback channel count for linear layout.

    Returns
    -------
    np.ndarray of shape [C, 2] (microns).
    """
    if map_path and os.path.isfile(map_path):
        positions = np.load(map_path)
        if positions.ndim == 2 and positions.shape[1] >= 2:
            return positions[:, :2].astype(np.float64)

    # Fallback: linear layout — channels evenly spaced on a line
    positions = np.zeros((n_channels, 2), dtype=np.float64)
    positions[:, 1] = np.arange(n_channels) * 50.0  # 50 µm spacing on Y axis
    return positions


def load_sorter_spike_times(
    source_path: Optional[str], n_channels: int
) -> dict:
    """
    Load existing sorted spike times, keyed by channel index.

    Supports:
      - Kilosart output dir (spike_times.npy + spike_clusters.npy + channel_map.npy)
      - LH HDF5 file (peak_channel attr + spike_times per unit)

    Falls back to empty dict if path is None or unrecognized.

    Returns
    -------
    {channel_idx: spike_times_array}
    """
    if not source_path:
        return {}

    source_path = source_path.strip()
    if not os.path.exists(source_path):
        return {}

    # Try Kilosort dir
    if os.path.isdir(source_path):
        return _load_kilosort_times(source_path, n_channels)

    # Try LH HDF5
    if source_path.endswith((".h5", ".hdf5")):
        return _load_lh_hdf5_times(source_path, n_channels)

    return {}


def _load_kilosort_times(ks_dir: str, n_channels: int) -> dict[int, np.ndarray]:
    """Load spike times from Kilosort output directory."""
    try:
        spike_times = np.load(os.path.join(ks_dir, "spike_times.npy"))  # [N, 1]
        spike_clusters = np.load(os.path.join(ks_dir, "spike_clusters.npy"))  # [N]
        channel_map = np.load(os.path.join(ks_dir, "channel_map.npy"))  # [N_units]

        spike_times = spike_times.squeeze().astype(np.int64)
        spike_clusters = spike_clusters.squeeze()

        # Map each unit to its peak channel
        result: dict[int, list] = {}
        for unit_id in np.unique(spike_clusters):
            ch = int(channel_map[unit_id]) if unit_id < len(channel_map) else 0
            times = spike_times[spike_clusters == unit_id]
            result.setdefault(ch, []).extend(times.tolist())

        return {ch: np.array(times, dtype=np.int64) for ch, times in result.items()}
    except Exception:
        return {}


def _load_lh_hdf5_times(h5_path: str, n_channels: int) -> dict[int, np.ndarray]:
    """Load spike times from Lighthouse HDF5 output."""
    try:
        import h5py
    except ImportError:
        return {}

    try:
        result: dict[int, list] = {}
        with h5py.File(h5_path, "r") as f:
            for unit_name in f.keys():
                group = f[unit_name]
                if "spike_times" not in group:
                    continue
                times = group["spike_times"][()].astype(np.int64)
                ch = int(group.attrs.get("peak_channel", 0))
                result.setdefault(ch, []).extend(times.tolist())
        return {ch: np.array(times, dtype=np.int64) for ch, times in result.items()}
    except Exception:
        return {}


def compute_channel_firing_rates(
    sorter_spike_times: dict[int, np.ndarray], duration_s: float
) -> dict[int, float]:
    """
    Given sorter_spike_times {ch: times}, return {ch: FR_hz}.
    """
    if duration_s <= 0:
        return {}
    return {
        ch: len(times) / duration_s
        for ch, times in sorter_spike_times.items()
    }
