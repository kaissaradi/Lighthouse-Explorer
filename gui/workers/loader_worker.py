"""
loader_worker.py — Runs raw data loading + baseline subtraction on a background thread.

Supports two recording formats transparently:
  - Flat binary (.dat / .bin):  memory-mapped (copy-on-write), fast seek.
  - Litke folder (.bin folder): read chunk-by-chunk into a contiguous writable
                                 ndarray, TTL channel stripped automatically.

Both paths emit the same finished(object) signal carrying a (T, C) int16 array
that is fully writable, so baseline subtraction works identically downstream.
"""
from __future__ import annotations
import os
from qtpy.QtCore import QObject, Signal

from core import loader
from lh_deps.axolotl_utils_ram import (
    compute_baselines_int16_deriv_robust,
    subtract_segment_baselines_int16,
)


class LoaderWorker(QObject):
    """
    Loads raw data and subtracts baselines on a background QThread.

    Pass either:
      - A path to a flat binary file (.dat / .bin)  → memory-mapped (COW)
      - A path to a Litke .bin *folder*             → materialised ndarray

    The object emitted by ``finished`` is always a writable (T, C) int16
    array-like with a ``.shape`` attribute, compatible with the rest of the
    pipeline (batch QC worker, snippet extraction, etc.).
    """

    progress = Signal(str)    # status message
    finished = Signal(object) # emits writable (T, C) array
    error = Signal(str)       # emits error message
    aborted = Signal()

    def __init__(
        self,
        dat_path: str,
        n_channels: int,
        dtype: str = "int16",
        start_min: float = 0.0,
        duration_min: float | None = None,
        fs: int = 20_000,
    ):
        super().__init__()
        self.dat_path = dat_path
        self.n_channels = n_channels
        self.dtype = dtype
        self.start_min = start_min
        self.duration_min = duration_min
        self.fs = fs
        self._abort = False

    def abort(self):
        """Signal the worker to stop."""
        self._abort = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_litke_folder(self) -> bool:
        """Return True when dat_path is a directory (Litke .bin folder)."""
        return os.path.isdir(self.dat_path)

    def _load_flat(self):
        """Memory-map a flat binary file (copy-on-write)."""
        self.progress.emit("Mapping recording…")
        raw_data = loader.load_raw_readonly(
            self.dat_path,
            n_channels=self.n_channels,
            dtype=self.dtype,
            start_min=self.start_min,
            duration_min=self.duration_min,
            fs=self.fs,
            writable=True,  # copy-on-write — safe for in-place baseline sub
        )
        return raw_data

    def _load_litke(self):
        """
        Read a Litke .bin folder into a contiguous writable ndarray.

        Reports chunk progress so the UI stays responsive during the read.
        Aborts early if self._abort is set between chunks.
        """
        self.progress.emit("Opening Litke folder…")

        # Peek at total length with the lazy virtual array (zero-copy).
        virtual = loader.load_litke_folder(
            self.dat_path, self.n_channels, self.dtype
        )
        total_samples = virtual.shape[0]

        # Resolve the window to report a meaningful denominator.
        start_sample = int(self.start_min * 60.0 * self.fs)
        start_sample = max(0, min(start_sample, total_samples))
        if self.duration_min is not None and self.duration_min > 0:
            n_samples = int(self.duration_min * 60.0 * self.fs)
        else:
            n_samples = total_samples - start_sample
        n_samples = max(1, min(n_samples, total_samples - start_sample))

        duration_s = n_samples / self.fs
        self.progress.emit(
            f"Reading {n_samples:,} samples ({duration_s:.1f} s) from Litke folder…"
        )

        def _progress_cb(n_loaded: int, n_total: int) -> None:
            # Called from within load_litke_as_writable_array after each chunk.
            # Qt signals are thread-safe so we can emit freely here.
            if self._abort:
                raise InterruptedError("aborted")
            pct = int(100 * n_loaded / max(n_total, 1))
            self.progress.emit(f"Reading Litke data… {pct}%")

        raw_data = loader.load_litke_as_writable_array(
            self.dat_path,
            n_channels=self.n_channels,
            dtype=self.dtype,
            start_min=self.start_min,
            duration_min=self.duration_min,
            fs=self.fs,
            chunk_samples=100_000,
            progress_cb=_progress_cb,
        )
        return raw_data

    # ------------------------------------------------------------------
    # Main entry point (called by QThread)
    # ------------------------------------------------------------------

    def run(self):
        try:
            # ── Step 1: Load raw data ──────────────────────────────────
            if self._is_litke_folder():
                try:
                    raw_data = self._load_litke()
                except InterruptedError:
                    self.aborted.emit()
                    return
            else:
                raw_data = self._load_flat()

            if self._abort:
                self.aborted.emit()
                return

            # ── Step 2: Compute baselines ──────────────────────────────
            self.progress.emit("Computing baselines…")
            baselines = compute_baselines_int16_deriv_robust(raw_data)

            if self._abort:
                self.aborted.emit()
                return

            # ── Step 3: Subtract baselines ─────────────────────────────
            # Works in-place on both the COW memmap and the Litke ndarray.
            self.progress.emit("Subtracting baselines…")
            subtract_segment_baselines_int16(raw_data, baselines)

            if self._abort:
                self.aborted.emit()
                return

            self.progress.emit("Ready.")
            self.finished.emit(raw_data)

        except Exception as e:
            self.error.emit(f"Load failed: {e}")