from __future__ import annotations
import os
from contextlib import nullcontext

NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

NUMBA_THREADING_LAYER_ENV = "NUMBA_THREADING_LAYER"
LIGHTHOUSE_NUMBA_LAYER_ENV = "LIGHTHOUSE_NUMBA_THREADING_LAYER"
DEFAULT_NUMBA_THREADING_LAYER = "workqueue"

def configure_native_thread_environment(
    threads: int = 1,
    *,
    numba_threading_layer: str | None = None,
) -> None:
    value = str(max(1, int(threads)))
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = value

    layer = numba_threading_layer or "workqueue"
    os.environ[NUMBA_THREADING_LAYER_ENV] = layer

def native_thread_limits(threads: int = 1):
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=max(1, int(threads)))
