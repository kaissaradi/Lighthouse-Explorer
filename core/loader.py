"""
loader.py — raw data loading utilities for Lighthouse QC.

Adapted from axolotl/io.py. All functions are pure (no Qt, no side effects).

Added LitkeMultiFileArray to directly read a folder of .bin files (Litke format)
without joining, automatically stripping the TTL channel (index 0).
"""
from __future__ import annotations
from typing import Optional
import os
import numpy as np

# Try to import bin2py for Litke folder support
try:
    import bin2py
    _BIN2PY_AVAILABLE = True
except ImportError:
    _BIN2PY_AVAILABLE = False


class LitkeMultiFileArray:
    """
    A read‑only, array‑like wrapper over a folder of Litke .bin files.

    The folder must contain chronological .bin files (e.g., data000.bin,
    data001.bin, ...). Each file contains a TTL channel at index 0, followed
    by the real electrode channels. This wrapper automatically strips the
    TTL channel, so indexing uses 0‑based channel numbers corresponding to
    the real electrodes (0 = first electrode after TTL).

    Supports:
        raw[start:stop, channels]   -> (n_rows, n_channels) or (n_rows, len(channels))
        raw[sample, channel]        -> scalar
        len(raw)                    -> total samples
        raw.shape                   -> (total_samples, n_channels)
    """
    def __init__(self, folder_path: str, n_channels: int, dtype=np.int16):
        """
        Parameters
        ----------
        folder_path : str
            Path to the folder containing .bin files.
        n_channels : int
            Number of real electrode channels (excluding TTL). The underlying
            .bin files must have n_channels + 1 channels (TTL + electrodes).
        dtype : np.dtype
            Data type of the raw samples (default int16).
        """
        if not _BIN2PY_AVAILABLE:
            raise ImportError("bin2py is required for Litke folder support. pip install bin2py")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"{folder_path} is not a directory")

        self.folder_path = folder_path
        self.n_channels = n_channels
        self.dtype = np.dtype(dtype)
        self._reader = bin2py.PyBinFileReader(folder_path, is_row_major=True)
        self._n_samples = self._reader.length
        # The reader returns (n_total_channels, n_samples) with TTL at index 0.
        # We'll slice away the TTL when indexing.

    @property
    def shape(self):
        return (self._n_samples, self.n_channels)

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self._n_samples * self.n_channels

    def __len__(self):
        return self._n_samples

    def __getitem__(self, key):
        """
        Supports:
            - raw[start:stop, col]          where col is int, slice, or list
            - raw[sample_index, col]
            - raw[start:stop]               (all channels)
            - raw[sample_index]             (all channels)
        """
        # Parse key
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        # --- Single sample indexing ---
        if isinstance(row_key, int):
            # Read one time point (all electrodes including TTL)
            block = self._reader.get_data(row_key, 1)   # shape (n_total_elec, 1)
            # Strip TTL (first row) and convert to 1D
            data = block[1:, 0].astype(self.dtype)      # (n_channels,)
            if col_key == slice(None):
                return data
            elif isinstance(col_key, int):
                return data[col_key]
            else:
                return data[col_key]

        # --- Slicing along rows ---
        if isinstance(row_key, slice):
            start = row_key.start if row_key.start is not None else 0
            stop = row_key.stop if row_key.stop is not None else self._n_samples
            step = row_key.step if row_key.step is not None else 1
            if step != 1:
                raise NotImplementedError("Step != 1 not supported for multi‑file slicing")
            n_rows = stop - start
            if n_rows <= 0:
                # Return empty array of appropriate shape
                n_cols = self._get_col_count(col_key)
                return np.empty((0, n_cols), dtype=self.dtype)

            # Read the contiguous block from the reader
            block = self._reader.get_data(start, n_rows)   # (n_total_elec, n_rows)
            # Strip TTL (first row) -> (n_channels, n_rows)
            data = block[1:, :].astype(self.dtype)        # (n_channels, n_rows)

            if col_key == slice(None):
                # Return (n_rows, n_channels)
                return data.T
            elif isinstance(col_key, int):
                # Return (n_rows,) for a single channel
                return data[col_key, :]
            else:
                # Return (n_rows, len(col_key)) for multiple channels
                return data[col_key, :].T

        raise TypeError(f"Unsupported key type: {type(row_key)}")

    def _get_col_count(self, col_key):
        if col_key == slice(None):
            return self.n_channels
        if isinstance(col_key, int):
            return 1
        return len(col_key)


def load_litke_folder(
    folder_path: str,
    n_channels: int,
    dtype: str = "int16",
) -> LitkeMultiFileArray:
    """
    Load a folder of Litke .bin files as a single virtual array.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing chronological .bin files.
    n_channels : int
        Number of real electrode channels (excluding the TTL channel).
    dtype : str
        NumPy dtype string (default "int16").

    Returns
    -------
    LitkeMultiFileArray
        An object that behaves like a (T, C) numpy array, but reads data
        on‑the‑fly from the original files. No joining, no extra disk space.
    """
    return LitkeMultiFileArray(folder_path, n_channels, np.dtype(dtype))


def load_litke_as_writable_array(
    folder_path: str,
    n_channels: int,
    dtype: str = "int16",
    start_min: float = 0.0,
    duration_min: float | None = None,
    fs: int = 20_000,
    chunk_samples: int = 5000000,  # <-- FIX 1: Reduced from 5,000,000 to 100,000
    progress_cb=None,
) -> np.ndarray:
    """
    Read a Litke .bin folder into a contiguous, writable (T, C) int16 array.
    ... [docstring omitted for brevity] ...
    """
    if not _BIN2PY_AVAILABLE:
        raise ImportError("bin2py is required for Litke folder support.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a directory")

    reader = bin2py.PyBinFileReader(folder_path, is_row_major=True)
    total_samples = reader.length

    # Resolve the sample window
    start_sample = int(start_min * 60.0 * fs)
    start_sample = max(0, min(start_sample, total_samples))

    if duration_min is not None and duration_min > 0:
        n_samples = int(duration_min * 60.0 * fs)
    else:
        n_samples = total_samples - start_sample
    n_samples = max(1, min(n_samples, total_samples - start_sample))

    # Allocate the output buffer once
    out = np.empty((n_samples, n_channels), dtype=np.dtype(dtype))

    # Read in chunks: bin2py returns (n_total_channels, chunk_size) col-major
    n_loaded = 0
    while n_loaded < n_samples:
        this_chunk = min(chunk_samples, n_samples - n_loaded)
        block = reader.get_data(start_sample + n_loaded, this_chunk)
        
        # block shape: (n_total_elec, this_chunk)  — TTL is row 0
        # Strip TTL, transpose, and allow NumPy to cast the types implicitly
        # <-- FIX 2: Removed .astype(np.dtype(dtype))
        out[n_loaded : n_loaded + this_chunk, :] = block[1:, :].T 
        
        n_loaded += this_chunk
        if progress_cb is not None:
            progress_cb(n_loaded, n_samples)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Existing functions below – unchanged
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_readonly(
    dat_path: str,
    n_channels: int,
    dtype: str = "int16",
    start_min: float = 0.0,
    duration_min: float | None = None,
    fs: int = 20_000,
    writable: bool = False,
) -> np.memmap:
    """
    Memory-map a chunk of a raw binary file [T, C].

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
    writable : bool
        If True, open in "c" (copy-on-write) mode. Modifications are
        kept in memory and never written back to disk, preserving the
        original file.

    Returns
    -------
    np.memmap of shape (T, C).
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

    mode = "c" if writable else "r"
    mem = np.memmap(
        dat_path,
        dtype=dtype,
        mode=mode,
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
      - Kilosort output dir (spike_times.npy + spike_clusters.npy + channel_map.npy)
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


def load_sorter_spike_units(
    source_path: Optional[str], n_channels: int
) -> dict[int, dict[int, np.ndarray]]:
    """
    Load sorted spike times keyed by channel *and* unit ID.

    Returns
    -------
    {channel_idx: {unit_id: spike_times_array}}

    Only Kilosort directories are supported (requires spike_times.npy +
    spike_clusters.npy / spike_templates.npy + templates.npy).  Falls back to
    an empty dict for any other path or missing files.
    """
    if not source_path:
        return {}
    source_path = source_path.strip()
    if not os.path.exists(source_path):
        return {}
    if os.path.isdir(source_path):
        return _load_kilosort_units(source_path, n_channels)
    return {}


def _load_kilosort_times(ks_dir: str, n_channels: int) -> dict[int, np.ndarray]:
    """Load spike times from a Kilosort output directory.

    Works with KS2 / KS2.5 / KS3 / KS4.

    Files read
    ----------
    spike_times.npy      – [N] or [N,1] sample indices (required)
    spike_clusters.npy   – [N] post-merge unit IDs (preferred)
    spike_templates.npy  – [N] raw template indices (fallback)
    templates.npy        – [n_templates, T, C] waveforms → peak channel per template
    channel_map.npy      – [C] or [C,1] maps template-space channels to probe channels

    Returns {channel_idx: spike_times_array}
    """
    def _load(path):
        return np.load(path).squeeze()

    # ── spike_times (required) ────────────────────────────────────────────────
    spike_times_path = os.path.join(ks_dir, "spike_times.npy")
    if not os.path.isfile(spike_times_path):
        return {}
    spike_times = _load(spike_times_path).astype(np.int64)
    if spike_times.ndim != 1 or spike_times.size == 0:
        return {}

    # ── per-spike unit assignment ─────────────────────────────────────────────
    clusters_path  = os.path.join(ks_dir, "spike_clusters.npy")
    templates_path_spk = os.path.join(ks_dir, "spike_templates.npy")

    spike_clusters  = _load(clusters_path).astype(np.int64)  if os.path.isfile(clusters_path)      else None
    spike_templates = _load(templates_path_spk).astype(np.int64) if os.path.isfile(templates_path_spk) else None

    if spike_clusters is None and spike_templates is None:
        return {}

    # spike_clusters = post-merge IDs (preferred); spike_templates = raw template IDs
    spike_unit_ids = spike_clusters if spike_clusters is not None else spike_templates

    # ── channel_map ───────────────────────────────────────────────────────────
    channel_map = None
    cm_path = os.path.join(ks_dir, "channel_map.npy")
    if os.path.isfile(cm_path):
        cm = _load(cm_path)
        if cm.ndim == 2:
            cm = cm[:, 0]
        channel_map = cm.astype(int)

    # ── templates → peak channel per template ────────────────────────────────
    peak_ch_per_template = None
    tmpl_path = os.path.join(ks_dir, "templates.npy")
    if os.path.isfile(tmpl_path):
        tmpl = np.load(tmpl_path)          # [n_templates, T, C]
        if tmpl.ndim == 3 and tmpl.shape[0] > 0:
            # ptp removed in numpy 2.0 — use explicit max-min
            ptp = tmpl.max(axis=1) - tmpl.min(axis=1)   # [n_templates, C]
            peak_ch_per_template = ptp.argmax(axis=1).astype(int)  # [n_templates]
            if channel_map is not None:
                # map template-space channel indices → probe channel indices
                safe = np.clip(peak_ch_per_template, 0, len(channel_map) - 1)
                peak_ch_per_template = channel_map[safe]

    # ── map spikes → probe channels ───────────────────────────────────────────
    result: dict[int, list] = {}

    if peak_ch_per_template is not None:
        n_tmpl = len(peak_ch_per_template)

        # For spike_clusters (post-merge) we need to find each cluster's peak channel.
        # The cluster ID may not equal a template index directly, so we map via
        # spike_templates when available, otherwise fall back to treating cluster IDs
        # as template IDs (works for KS4 which doesn't rename them).
        if spike_clusters is not None and spike_templates is not None:
            # Build cluster → peak channel via the most common template for each cluster
            lookup = {}
            for uid in np.unique(spike_clusters):
                mask = spike_clusters == uid
                tmpl_ids = spike_templates[mask]
                # most common template for this cluster
                most_common = int(np.bincount(
                    np.clip(tmpl_ids, 0, n_tmpl - 1)
                ).argmax())
                lookup[int(uid)] = int(peak_ch_per_template[most_common])
            spike_channels = np.array([lookup[int(u)] for u in spike_unit_ids], dtype=int)
        else:
            # Only one of the two arrays exists — treat unit IDs as template IDs directly
            safe_ids = np.clip(spike_unit_ids, 0, n_tmpl - 1)
            spike_channels = peak_ch_per_template[safe_ids]

        for ch in np.unique(spike_channels):
            mask = spike_channels == ch
            result[int(ch)] = spike_times[mask].tolist()

    else:
        # No templates.npy — group by unit ID (channel mapping will be wrong
        # but at least spike counts per "unit" are preserved for miss-rate display)
        for uid in np.unique(spike_unit_ids):
            mask = spike_unit_ids == uid
            result.setdefault(int(uid), []).extend(spike_times[mask].tolist())

    return {ch: np.array(times, dtype=np.int64) for ch, times in result.items()}


def _load_kilosort_units(
    ks_dir: str, n_channels: int
) -> dict[int, dict[int, np.ndarray]]:
    """
    Like _load_kilosort_times but preserves the unit-ID dimension.

    Returns
    -------
    {channel_idx: {unit_id: spike_times_array}}

    Uses identical file-reading logic to _load_kilosort_times so that the
    channel assignment is consistent.  The extra cost is negligible — all the
    heavy numpy arrays are already in memory.
    """
    def _load(path):
        return np.load(path).squeeze()

    spike_times_path = os.path.join(ks_dir, "spike_times.npy")
    if not os.path.isfile(spike_times_path):
        return {}
    spike_times = _load(spike_times_path).astype(np.int64)
    if spike_times.ndim != 1 or spike_times.size == 0:
        return {}

    clusters_path = os.path.join(ks_dir, "spike_clusters.npy")
    templates_path_spk = os.path.join(ks_dir, "spike_templates.npy")

    spike_clusters = (
        _load(clusters_path).astype(np.int64) if os.path.isfile(clusters_path) else None
    )
    spike_templates = (
        _load(templates_path_spk).astype(np.int64)
        if os.path.isfile(templates_path_spk)
        else None
    )

    if spike_clusters is None and spike_templates is None:
        return {}

    spike_unit_ids = spike_clusters if spike_clusters is not None else spike_templates

    # ── channel_map ───────────────────────────────────────────────────────────
    channel_map = None
    cm_path = os.path.join(ks_dir, "channel_map.npy")
    if os.path.isfile(cm_path):
        cm = _load(cm_path)
        if cm.ndim == 2:
            cm = cm[:, 0]
        channel_map = cm.astype(int)

    # ── peak channel per template ─────────────────────────────────────────────
    peak_ch_per_template = None
    tmpl_path = os.path.join(ks_dir, "templates.npy")
    if os.path.isfile(tmpl_path):
        tmpl = np.load(tmpl_path)          # [n_templates, T, C]
        if tmpl.ndim == 3 and tmpl.shape[0] > 0:
            ptp = tmpl.max(axis=1) - tmpl.min(axis=1)
            peak_ch_per_template = ptp.argmax(axis=1).astype(int)
            if channel_map is not None:
                safe = np.clip(peak_ch_per_template, 0, len(channel_map) - 1)
                peak_ch_per_template = channel_map[safe]

    # ── assign each spike a probe channel ────────────────────────────────────
    if peak_ch_per_template is not None:
        n_tmpl = len(peak_ch_per_template)
        if spike_clusters is not None and spike_templates is not None:
            lookup: dict[int, int] = {}
            for uid in np.unique(spike_clusters):
                mask = spike_clusters == uid
                tmpl_ids = spike_templates[mask]
                most_common = int(
                    np.bincount(np.clip(tmpl_ids, 0, n_tmpl - 1)).argmax()
                )
                lookup[int(uid)] = int(peak_ch_per_template[most_common])
            spike_channels = np.array(
                [lookup[int(u)] for u in spike_unit_ids], dtype=int
            )
        else:
            safe_ids = np.clip(spike_unit_ids, 0, n_tmpl - 1)
            spike_channels = peak_ch_per_template[safe_ids]
    else:
        # No templates — use unit ID as a proxy for channel (degraded mode)
        spike_channels = spike_unit_ids.copy()

    # ── build nested dict {ch: {uid: times}} ─────────────────────────────────
    result: dict[int, dict[int, list]] = {}
    for ch in np.unique(spike_channels):
        ch_mask = spike_channels == ch
        ch_units = spike_unit_ids[ch_mask]
        ch_times = spike_times[ch_mask]
        unit_dict: dict[int, list] = {}
        for uid in np.unique(ch_units):
            u_mask = ch_units == uid
            unit_dict[int(uid)] = ch_times[u_mask].tolist()
        result[int(ch)] = unit_dict

    return {
        ch: {uid: np.array(t, dtype=np.int64) for uid, t in units.items()}
        for ch, units in result.items()
    }


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