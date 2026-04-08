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