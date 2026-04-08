"""
loader_worker.py — Runs raw data loading + baseline subtraction on a background thread.
"""
from __future__ import annotations
import numpy as np
from qtpy.QtCore import QObject, Signal

from core import loader
from lh_deps.axolotl_utils_ram import (
    compute_baselines_int16_deriv_robust,
    subtract_segment_baselines_int16,
)


class LoaderWorker(QObject):
    """Loads raw data and subtracts baselines on a background QThread."""

    progress = Signal(str)  # status message
    finished = Signal(object)  # emits np.memmap (the loaded raw_data)
    error = Signal(str)  # emits error message
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

    def run(self):
        try:
            # Step 1: Memory-map the file (copy-on-write)
            self.progress.emit("Mapping recording…")
            raw_data = loader.load_raw_readonly(
                self.dat_path,
                n_channels=self.n_channels,
                dtype=self.dtype,
                start_min=self.start_min,
                duration_min=self.duration_min,
                fs=self.fs,
                writable=True,  # copy-on-write mode
            )

            if self._abort:
                self.aborted.emit()
                return

            # Step 2: Compute baselines
            self.progress.emit("Computing baselines…")
            baselines = compute_baselines_int16_deriv_robust(raw_data)

            if self._abort:
                self.aborted.emit()
                return

            # Step 3: Subtract baselines (in-place on the COW memmap)
            self.progress.emit("Subtracting baselines…")
            subtract_segment_baselines_int16(raw_data, baselines)

            if self._abort:
                self.aborted.emit()
                return

            self.progress.emit("Ready.")
            self.finished.emit(raw_data)

        except Exception as e:
            self.error.emit(f"Load failed: {e}")
