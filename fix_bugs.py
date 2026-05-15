import os
import glob

# 1. Create core/__init__.py with threading setup
core_init = """import os
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

    layer = numba_threading_layer
    if layer is None:
        layer = os.environ.get(
            LIGHTHOUSE_NUMBA_LAYER_ENV,
            DEFAULT_NUMBA_THREADING_LAYER,
        )
    os.environ[NUMBA_THREADING_LAYER_ENV] = layer

def native_thread_limits(threads: int = 1):
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=max(1, int(threads)))
"""
with open("core/__init__.py", "w") as f:
    f.write(core_init)

# 2. Remove threading setup from core/lh_qc_pipeline.py
with open("core/lh_qc_pipeline.py", "r") as f:
    lines = f.readlines()

with open("core/lh_qc_pipeline.py", "w") as f:
    skip = False
    for line in lines:
        if line.startswith('"""\n') and not skip and 'Process-level native library' in "".join(lines[:30]):
            pass # Keep logic simple, we'll just check line indices
            
    # Actually, simpler way: delete lines 15 to 85
    # Let's write lines 0-14 and 86-end
    f.writelines(lines[0:14])
    f.writelines(lines[85:])

# 3. Update imports in all python files
py_files = glob.glob("**/*.py", recursive=True)
for pf in py_files:
    if pf == "fix_bugs.py": continue
    with open(pf, "r") as f:
        content = f.read()
    orig = content
    content = content.replace("from core.lh_qc_pipeline import configure_native_thread_environment", "from core import configure_native_thread_environment")
    content = content.replace("from core.lh_qc_pipeline import native_thread_limits", "from core import native_thread_limits")
    content = content.replace("from core.lh_qc_pipeline import (\n    configure_native_thread_environment,\n    native_thread_limits,\n)", "from core import configure_native_thread_environment, native_thread_limits")
    
    # In case there's another format
    content = content.replace("from core.lh_qc_pipeline import configure_native_thread_environment, native_thread_limits", "from core import configure_native_thread_environment, native_thread_limits")
    content = content.replace("from core.lh_qc_pipeline import (\n    configure_native_thread_environment,\n)", "from core import configure_native_thread_environment")
    
    if content != orig:
        with open(pf, "w") as f:
            f.write(content)

# 4. Fix task.signals.deleteLater() in gui/qc_worker.py
with open("gui/qc_worker.py", "r") as f:
    content = f.read()

content = content.replace("task.signals.deleteLater()", "# task.signals.deleteLater()  # Removed to prevent wrapped C/C++ object deletion crash")
with open("gui/qc_worker.py", "w") as f:
    f.write(content)

# 5. Fix pyqtgraph overflow warning by filtering it
with open("run.py", "r") as f:
    content = f.read()

warning_code = """
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in cast", category=RuntimeWarning, module="pyqtgraph.*")
"""
if "import warnings" not in content:
    content = content.replace("import sys", "import sys\n" + warning_code)
    with open("run.py", "w") as f:
        f.write(content)

