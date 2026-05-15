# Agent handoff: Batch QC freeze + Numba fork warnings

**Last updated:** 2026-05-15 (end of session)  
**Primary spec:** [`docs/specs/numba_fork_safety.md`](../specs/numba_fork_safety.md)  
**Project rules:** [`docs/AGENTS.md`](../AGENTS.md)

---

## Executive summary (read this first)

| What works (lab, May 2026) | What fails |
|----------------------------|------------|
| **QC workers = 1** — batch QC completes on Litke folder `data010` (519 ch) | **QC workers > 1** — four× TBB fork stderr lines, process **Killed** (OOM or abort) |
| **Litke bin folder** — load + baselines reasonably fast | **“Single Litke” path** (if distinct) — **computing baselines very slow** (needs path clarification) |
| Headless load+baselines ~40s for full ~10 GB file | Concurrent pool + Numba/TBB still unstable at >1 worker |

**Operational recommendation:** Use **QC workers = 1** until a follow-up fix allows safe parallelism (subprocess isolation, or confirmed workqueue+lock under all sklearn/joblib paths). Default UI spinbox is now **1**.

**Fixes landed in repo (keep):**

- `core/native_threading.py` — `NUMBA_THREADING_LAYER=workqueue`, `NUMBA_NUM_THREADS=1`, OMP/BLAS pinned
- `core/numba_kernels.py` — global lock around all Numba entry points
- `TaskManager.start_batch` — honors `batch_max_workers` from params; `_resolve_fs()` for sample rate
- `main_window` — passes `fs` into batch/single QC

**Removed (session cleanup):** `core/debug_trace.py`, `.cursor/debug-bec1d0.log`, all temporary debug instrumentation.

---

## 1. Symptoms

### Numba / TBB (stderr)

```text
Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process.
```

- Often **4 lines** when pool runs 4 channels (matches worker count).
- Still seen **once** even on successful runs with **1 worker** (dependency may load TBB despite `workqueue` env — investigate `numba.config.THREADING_LAYER` at runtime).
- With **>1 workers**: repeated warnings then **`Killed`**.

### GUI / UX

- Batch progress: `Running QC… (X/519)` — freeze/hang reported earlier; with **1 worker** batch completes.
- **pyqtgraph** `RuntimeWarning: overflow encountered in cast` in ViewBox — separate plotting/range bug, not Numba.
- On window close during batch: `RuntimeError: wrapped C/C++ object of type QThread/QCChannelTaskSignals has been deleted` — lifecycle bug in `closeEvent` vs running pool tasks (open).

---

## 2. Architecture

```text
Load → LoaderWorker (QThread)
  → load_litke_as_writable_array (full RAM)
  → numba_kernels: baselines + subtract
→ auto start_batch (TaskManager, QThreadPool)

Batch → QCChannelTask × N (queued at once, pool limits concurrency)
  → numba_kernels: extract_snippets_fast_ram
  → sklearn PCA/KMeans (threadpoolctl → 1 thread)
```

---

## 3. Lab measurements (`data010`, 519 ch)

```text
samples=9,680,000  (~8.1 min @ 20 kHz)
RAM ≈ 10 GB loaded
```

Headless (configure + numba_kernels):

| Phase | 5 min | Full |
|-------|-------|------|
| Read | 48 s | 38 s |
| Baselines | 0.6 s | 0.6 s |

GUI: **Litke bin folder** baselines OK; **single Litke variant** reported slow baselines (confirm which source type / code path).

---

## 4. Root causes (confirmed vs open)

| Issue | Status | Notes |
|-------|--------|-------|
| workqueue + concurrent Python threads without lock | **Fixed** | `numba_kernels` RLock |
| `NUMBA_NUM_THREADS` not set → workqueue used cpu_count internally | **Fixed** | In `native_threading.py` |
| `TaskManager` ignored `batch_max_workers` | **Fixed** | `setMaxThreadCount` in `start_batch` |
| `fs` not passed to batch | **Fixed** | `_resolve_fs`, `main_window` |
| TBB fork warning with **>1** QC workers | **Open** | Env may not bind before all Numba; sklearn/joblib may use TBB |
| >1 workers → Killed | **Open** | Thread/memory explosion or SIG from Numba |
| TBB warning with **1** worker | **Open** | Single channel still triggers fork from worker thread |
| Slow baselines on “single Litke” | **Open** | Compare `LoaderWorker` paths / materialization |
| pyqtgraph overflow | **Open** | QC view axis limits |
| closeEvent vs deleted Qt objects | **Open** | `main_window.closeEvent` / abort batch before thread teardown |

---

## 5. Key files

| Path | Role |
|------|------|
| `core/native_threading.py` | Process env before Numba import |
| `core/numba_kernels.py` | Locked Numba wrappers |
| `core/lh_qc_pipeline.py` | QC pipeline |
| `gui/workers/qc_worker.py` | TaskManager, LoaderWorker, QCChannelTask |
| `gui/main_window.py` | Auto batch after load |
| `gui/panels/load_panel.py` | `batch_max_workers` (default **1**) |
| `lh_deps/axolotl_utils_ram.py` | READ-ONLY Numba kernels |

---

## 6. Verification commands

```bash
conda activate lighthouse_qc
cd ~/Documents/Lighthouse-Explorer
export PYTHONPATH=.

python -c "
from core.native_threading import configure_native_thread_environment
import os
configure_native_thread_environment()
for k in ('NUMBA_THREADING_LAYER','NUMBA_NUM_THREADS','OMP_NUM_THREADS'):
    print(k, os.environ.get(k))
"

pytest tests/test_numba_fork_safety.py tests/test_batch_qc_worker.py -v
python -u run.py
```

---

## 7. Next session priorities

1. **Confirm threading layer at runtime** in GUI worker thread:  
   `numba.config.THREADING_LAYER`, `numba.get_num_threads()` after first Numba call.
2. **>1 workers:** subprocess-per-channel (spec option E) or enforce TBB-safe fork policy; do not re-enable default 4 until clean stderr on lab data.
3. **Single vs folder Litke** baseline slowness — trace which loader branch and array size.
4. **closeEvent:** `abort_batch()` + wait for pool before `deleteLater` on workers.
5. **pyqtgraph** overflow — clamp plot ranges in `qc_view_panel.py`.

---

## 8. Decision log

| Date | Event |
|------|--------|
| 2026-05-15 | Initial spec; workqueue hang without lock |
| 2026-05-15 | `numba_kernels`, `NUMBA_NUM_THREADS`, TaskManager pool + fs fixes |
| 2026-05-15 | Lab: 1 worker OK; >1 → TBB×4 + Killed; debug tooling removed |
| 2026-05-15 | Default QC workers → 1; handoff doc finalized |
