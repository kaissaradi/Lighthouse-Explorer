from __future__ import annotations

import os
from contextlib import nullcontext


NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_native_thread_environment(threads: int = 1) -> None:
    """
    Set process-level limits before NumPy/SciPy/scikit-learn are imported.

    These variables only reliably affect native libraries before they are loaded,
    so channel workers also use ``native_thread_limits`` at runtime.
    """
    value = str(max(1, int(threads)))
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = value


def native_thread_limits(threads: int = 1):
    """Return a context manager that limits already-loaded BLAS/OpenMP pools."""
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()

    return threadpool_limits(limits=max(1, int(threads)))
