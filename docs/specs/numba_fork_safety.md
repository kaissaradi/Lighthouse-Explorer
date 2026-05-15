# Specification: Numba / TBB Fork Safety in Batch QC

## Metadata

* **Status:** Partially mitigated — use **1 QC worker** on lab hardware until >1 worker path is verified clean
* **Agent handoff log:** [`docs/debug/batch_qc_freeze_handoff.md`](../debug/batch_qc_freeze_handoff.md)
* **Target Release:** v1.0 (stability / concurrency hardening)
* **Primary Developer/Agent:** TBD
* **Related issues:** Numba [#5973](https://github.com/numba/numba/issues/5973), [#7872](https://github.com/numba/numba/issues/7872); internal worker architecture in `docs/specs/refactor.md`

---

## Objective

Eliminate (or safely contain) the runtime warning:

```text
Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process.
```

This warning appears during **batch QC on real recordings**, typically around **channels 6–8**, not at application launch. The warning is emitted **four times** in the reported case, which matches the default **four concurrent `QThreadPool` workers**.

The immediate goal of this spec is to **document root cause, reproduction constraints, and a validation plan** so the issue can be confirmed and fixed on a machine with real data. **Implementation is out of scope until ACs below are met on that machine.**

---

## User Story

"As a researcher running batch QC on a full array recording, I want the pipeline to run without Numba/TBB fork warnings so I can trust that parallel Numba and Qt worker threads are not leaving child processes in an undefined state."

---

## Reported Behavior

| Observation | Detail |
|---|---|
| **When** | After loading a dataset and running batch QC — not on `python run.py` alone |
| **Where in progress** | Roughly **channels 6–8** (UI progress counter), not channel 0 |
| **Count** | **4 identical lines** printed to the terminal |
| **Environment** | `lighthouse_qc` conda env, Linux, interactive GUI (`python run.py`) |
| **Severity (current)** | Warning only — user has not reported crashes or wrong QC results yet |

---

## Investigation Summary

### 1. Architecture: where Numba runs

Numba-accelerated code lives in **`lh_deps/axolotl_utils_ram.py`** (READ-ONLY vendored). All hot paths use `@njit(parallel=True, cache=True)`:

| Function | Called from | Thread context |
|---|---|---|
| `compute_baselines_int16_deriv_robust` | `LoaderWorker.run()` | **Background `QThread`** (load path) |
| `subtract_segment_baselines_int16` | `LoaderWorker.run()` | **Background `QThread`** |
| `extract_snippets_fast_ram` | `core/lh_qc_pipeline.py` (multiple steps) | **`QThreadPool` worker** via `QCChannelTask.run()` |

Batch QC path:

```text
MainWindow → TaskManager.start_batch()
  → QThreadPool (default maxThreadCount = 4)
    → QCChannelTask.run()  [non-main thread]
      → run_qc_pipeline()
        → extract_snippets_fast_ram()  [Numba parallel=True, TBB]
        → sklearn PCA / KMeans         [native OpenMP, pinned to 1 thread per task]
```

Relevant defaults:

* `DEFAULT_BATCH_MAX_WORKERS = 4` in `gui/workers/qc_worker.py`
* Load panel **QC workers** spinbox defaults to **4** (`gui/panels/load_panel.py`)
* `native_thread_limits(1)` + `OMP_NUM_THREADS=1` etc. apply to **BLAS/OpenMP**, not to Numba’s internal `prange` pool

### 2. Why ~channel 6–8 and exactly four warnings

**Channel indexing vs. pool saturation:**

1. Batch queues **all channels** immediately; the pool runs at most **4** at a time.
2. Channels **0–3** occupy the pool first (first wave).
3. As those finish, channels **4–7** start (second wave). The UI progress label often reads **6–8 / N** while this second wave is active — consistent with user reports.
4. **Four warnings** align with **four pool threads**, each of which may initialize the Numba **TBB** task scheduler the first time it executes `parallel=True` code on that thread.

**Why not channel 0:**

* Early channels may **reject early** (valley not accepted, ISI, valley count) before heavy `extract_snippets_fast_ram` use, so TBB may not fully spin up until later channels with richer spike content.
* **JIT/cache warmup:** first compilations can occur on whichever thread hits a function first; warnings are tied to **fork + TBB on non-main thread**, not to JIT alone.

### 3. What Numba is telling us (upstream semantics)

Numba’s own tests (`numba/tests/test_parallel_backend.py`, class `TestTBBSpecificIssues`) document:

1. **`parallel=True`** starts the selected threading layer’s pool during compilation/execution (often **TBB** on Linux when available).
2. **TBB can recover from `fork()` on the main thread**; it **warns** (and child state may be invalid) if **`fork()` happens on a non-main thread** after TBB has been used on that thread.
3. A known trigger is **`multiprocessing.Pool` on Linux with `fork`**, where pool management threads call `fork(2)` while Numba parallel work has already run on worker threads (Numba #7872).

The message is printed from **native code (C stderr)**, not via Python’s `warnings` module — so `pytest.warns` and `warnings.catch_warnings` are **insufficient** for detection; tests must capture **process stderr** (subprocess or `capfd`).

### 4. What we ruled out or deprioritized

| Hypothesis | Assessment |
|---|---|
| Warning at `python run.py` with no data | **Not reproduced** — user clarified it happens mid-batch |
| Missing `configure_native_thread_environment()` | **Unlikely** — called from `run.py` and at `qc_worker` import |
| OpenMP/BLAS thread explosion | **Separate concern** — already mitigated; does not control Numba `prange` |
| `multiprocessing` / `Pool` in app code | **None found** in `core/`, `gui/`, or `lh_deps/` |
| sklearn `KMeans` forking | **Unlikely** in 1.6.x with default `n_jobs`; uses threaded/OpenMP paths — still needs confirmation under real load |

### 5. Local reproduction attempts (this machine / agent environment)

| Attempt | Result |
|---|---|
| Import + `MainWindow()` startup | No Numba lines |
| Synthetic 12-channel batch via `QCChannelTask` on real `QThreadPool` | No warnings; **no `os.fork()` observed** |
| Direct `extract_snippets_fast_ram` × 4 pool threads | No warnings |
| Forced `NUMBA_THREADING_LAYER=tbb` + `multiprocessing.Pool` after pool Numba | **Reproduced 2× warning** — confirms TBB + fork-from-worker pattern in principle |

**Conclusion:** Full reproduction needs **real memmap data**, **longer recordings**, and likely the same **TBB + channel count + spike density** as production. The sandbox/headless agent environment did not mirror the user’s interactive session closely enough.

### 6. Leading root-cause hypothesis (to validate on real hardware)

**Composite threading model:**

```text
Qt QThreadPool (Python threads, non-main)
  + Numba parallel=True (TBB thread pool per process)
  + [unknown] fork(2) from a non-main thread during or after snippet extraction / sklearn / library internals
```

Most plausible fork sources to trace on the real machine (in order):

1. **Incidental `multiprocessing` use** in the dependency stack (joblib, sklearn, numba cache, Qt plugins) triggered only under load.
2. **Second-wave pool threads** — all four workers have active TBB state when a fork occurs (explains **4 lines** at ~ch 6–8).
3. **LoaderWorker `QThread`** — baselines already ran Numba parallel on a loader thread before batch; batch QC adds pool threads; fork may correlate with overlap or GC of large memmaps (speculative).

**Risk if ignored:** Numba documents **invalid TBB state in the child process** after such a fork. Observed impact today is warnings only; risk is **subtle corruption or rare crashes** in code paths that fork after QC (not yet seen).

---

## Architecture & Technical Constraints

* **`lh_deps/` is READ-ONLY** — any change to `@njit(parallel=True)` requires vendoring policy or a thin wrapper in `core/` (future fix).
* **Zero UI freezing** — any fix must keep heavy work off the main thread (`docs/AGENTS.md`).
* **Batch concurrency** — `TaskManager` / `QCChannelTask` on `QThreadPool` remains the execution model unless this spec explicitly changes it.
* **Do not “fix” by silencing stderr** without addressing fork/TBB interaction.

---

## Proposed Fix Directions (future work — not in this spec)

Documented for planning only; **no implementation until validated.**

| Option | Idea | Tradeoff |
|---|---|---|
| **A. Main-thread Numba warmup** | JIT-compile / run one representative call on main thread before starting batch | May reduce but not eliminate fork-from-worker if fork happens later |
| **B. threading layer** | `NUMBA_THREADING_LAYER=workqueue` or `omp` with documented fork behavior | Performance vs TBB; must benchmark on real data |
| **C. Disable Numba `parallel` in workers** | Wrapper calling `parallel=False` variants from `core/` | Slower snippet/baseline paths; needs non-vendored copies |
| **D. Serialize Numba** | Global lock around all Numba entry points from pool threads | Safer but defeats batch parallelism for Numba sections |
| **E. Process isolation** | Run QC channel work in `spawn` subprocesses | Heavy refactor; memmap sharing care |

---

## Acceptance Criteria (Definition of Done — for the fix phase)

* **AC1:** Running batch QC on the **reference real dataset** (see Test Plan) produces **zero** lines matching `Attempted to fork from a non-main thread` on stderr.
* **AC2:** Batch QC on that dataset still completes with **identical QC verdicts** (or documented acceptable numeric tolerance) vs. pre-fix baseline export.
* **AC3:** `tests/test_numba_fork_safety.py` passes in CI (synthetic/regression) **and** the manual real-data checklist passes on the lab machine.
* **AC4:** No regression in `tests/test_batch_qc_worker.py`, `tests/test_memory_leaks.py`, or load-panel worker count behavior.

---

## Test Plan

### Phase 0 — Real-machine confirmation (required before coding a fix)

Run on a machine with the user’s dataset and display (not offscreen-only).

**Environment:**

```bash
conda activate lighthouse_qc
cd /path/to/Lighthouse-Explorer
export PYTHONPATH=.
```

**Capture stderr:**

```bash
python run.py 2>&1 | tee /tmp/lighthouse_qc_stderr.log
# Load recording → Run batch QC → note channel number when warnings appear
grep -n "Numba:" /tmp/lighthouse_qc_stderr.log
```

**Optional fork tracer** (append to a one-off script or temporary `run.py` hook — do not commit tracer to production):

```python
import os, threading, traceback
_real = os.fork
def _traced():
    if threading.current_thread().name != "MainThread":
        print("FORK from", threading.current_thread().name)
        traceback.print_stack(limit=12)
    return _real()
os.fork = _traced
```

Record:

* Dataset path, `n_channels`, `batch_max_workers` UI value
* Channel progress when warnings appear
* Whether warnings repeat on a **second** batch run (JIT/cache warm)
* Numba/threading env: `python -c "import numba; numba.config.THREADING_LAYER; print(numba.__version__)"`

**Reference dataset:** Use the same recording that triggered the report (if available: Litke folder or `.dat` path from lab). `tests/test_batch_qc_worker_real_data.py` points at `DATA000_PATH` on a dev machine — adapt path locally.

### Phase 1 — Automated regression (to add with fix)

New file: `tests/test_numba_fork_safety.py`

| Test | Purpose |
|---|---|
| `test_qc_channel_task_runs_off_main_thread` | Assert `QCChannelTask` executes pipeline on `QThread.currentThread() != main` |
| `test_batch_pool_default_is_four` | Document/default guard for `maxThreadCount == 4` |
| `test_tbb_fork_warning_regression_subprocess` | Linux-only: subprocess script warms Numba on pool threads, triggers fork, asserts stderr clean **after fix**; may `xfail` until fix lands |
| `test_real_data_batch_stderr_clean` | Opt-in: `@pytest.mark.real_data`, skips if `DATA000_PATH` missing; batch QC via `BatchQCWorker` + `qtbot`, assert no `Numba:` in captured stderr |

**Verification commands:**

```bash
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/test_numba_fork_safety.py -v
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/test_batch_qc_worker.py tests/test_memory_leaks.py -v
```

### Phase 2 — Post-fix benchmarking

* Wall time for full-array batch QC before/after on reference dataset
* Memory profile (`tests/test_memory_leaks.py`) unchanged

---

## Out of Scope / Anti-Goals

* Changing vendored `lh_deps/` without an explicit vendoring/update process
* Replacing `pyqtgraph` or rewriting the QC pipeline in this spec
* Suppressing Numba messages without fixing fork/TBB interaction
* Committing machine-specific absolute paths to the repo

---

## Files Involved (read-only for this spec)

| File | Role |
|---|---|
| `lh_deps/axolotl_utils_ram.py` | Numba `parallel=True` kernels |
| `gui/workers/qc_worker.py` | `QCChannelTask`, `TaskManager`, `LoaderWorker`, pool size |
| `core/lh_qc_pipeline.py` | Calls `extract_snippets_fast_ram` during QC steps |
| `gui/panels/load_panel.py` | UI `batch_max_workers` default |
| `docs/AGENTS.md` §3.6 | OpenMP/BLAS pinning (orthogonal to Numba TBB) |

---

## Decision Log

| Date | Decision |
|---|---|
| 2026-05-15 | Spec created from investigation; fix deferred pending real-data confirmation on lab machine |
| 2026-05-15 | User confirmed warnings occur mid-batch ~ch 6–8, not at startup |
| 2026-05-15 | Fix: `core/native_threading.py` sets `NUMBA_THREADING_LAYER=workqueue` before Numba import; override via `LIGHTHOUSE_NUMBA_THREADING_LAYER` |
| 2026-05-15 | workqueue without a lock freezes batch QC (concurrent pool threads); added `core/numba_kernels.py` global lock around all Numba entry points |
| 2026-05-15 | Lab: batch OK at 1 worker; >1 workers → 4× TBB fork warning + Killed; `NUMBA_NUM_THREADS=1` + TaskManager pool/fs fixes |
| 2026-05-15 | Default UI QC workers = 1; debug instrumentation removed |
