# Specification: Worker Unification & State Consolidation

## Metadata

* **Status:** Phase 1 ✅ Complete · Phase 2 ✅ Complete
* **Target Release:** v1.0
* **Primary Developer/Agent:** AI Assistant

## Objective

Strip redundant `QThread` boilerplate out of `main_window.py` and unify the execution of single-channel and batch QC into a single thread pool architecture. Extract UI styling to clean up application entry points. Consolidate duplicated logic across the `gui/workers/`, `gui/panels/`, and `core/` directories.

## User Story

"As a developer, I want all background tasks (loading, single QC, batch QC) to be managed by a single unified TaskManager so `main_window.py` only handles UI routing, preventing memory leaks and race conditions. I also want duplicated code consolidated into shared modules so bugs only need to be fixed in one place."

---

## Phase 1 — Worker Unification ✅ COMPLETE

*Completed on branch `refactor/worker-unification`.*

### Acceptance Criteria (all met)

* **AC1:** ✅ `gui/theme.py` created. `DARK_STYLESHEET` extracted from `app.py`.
* **AC2:** ✅ `main_window.py` no longer manually instantiates `QThread` for QC. All QC background execution delegated to `TaskManager`.
* **AC3:** ✅ `qc_worker.py` refactored into `TaskManager` — uses the exact same `QCChannelTask` (`QRunnable`) that `BatchQCWorker` uses.
* **AC4:** ✅ GUI launches, batch QC and single-channel QC both work. All existing tests pass.

### Files Changed

| File | Change |
|---|---|
| `gui/theme.py` | **NEW** — extracted `DARK_STYLESHEET` |
| `gui/app.py` | Slimmed to purely functional entry point (164→36 lines) |
| `gui/workers/qc_worker.py` | Old `QCWorker(QObject)` → `TaskManager` (unified dispatch) |
| `gui/workers/__init__.py` | Updated exports |
| `gui/main_window.py` | Removed `_single_thread`, `_single_worker`, `_batch_thread`, `_batch_worker` boilerplate |
| `requirements-test.txt` | **NEW** — test dependency manifest |

---

## Phase 2 — File Consolidation & Deduplication ✅ COMPLETE

### Objective

Reduce file count in `gui/workers/` and eliminate duplicated logic across `gui/panels/` and `core/loader.py`. After Phase 1, there's still redundancy:

- `batch_qc_worker.py` and `qc_worker.py` are tightly coupled (TaskManager imports everything from batch_qc_worker)
- `qc_view_panel.py` and `qc_summary_dialog.py` share duplicated spike-matching logic
- `core/loader.py` has two nearly identical internal functions (`_load_kilosort_times` and `_load_kilosort_units`) with ~80 lines of copy-pasted KiloSort parsing

---

### 2A: Merge `batch_qc_worker.py` into `qc_worker.py` ✅

**Problem:** After Phase 1, `qc_worker.py` (`TaskManager`) imports 6 symbols from `batch_qc_worker.py`. These two files are really one module split across two files. The legacy `BatchQCWorker` class is now only used by tests directly — `main_window.py` uses `TaskManager` exclusively.

**Plan:**
1. Move `QCChannelTask`, `QCChannelTaskSignals`, helper functions (`_positive_int`, `_resolve_worker_count`, constants), and `BatchQCWorker` into `qc_worker.py` alongside `TaskManager`.
2. Delete `batch_qc_worker.py`.
3. Update `gui/workers/__init__.py` to import from the single file.
4. Update `tests/test_batch_qc_worker.py` import: `from gui.workers import qc_worker as batch_module` (or alias to minimize test churn).
5. Update `tests/test_memory_leaks.py` import path.

**Files Modified:** `gui/workers/qc_worker.py`, `gui/workers/__init__.py`
**Files Deleted:** `gui/workers/batch_qc_worker.py`
**Tests Updated:** `tests/test_batch_qc_worker.py`, `tests/test_memory_leaks.py`

---

### 2B: Extract shared spike-matching utility from panels ✅

**Problem:** Both `qc_view_panel.py` (`_update_fr_plot`) and `qc_summary_dialog.py` (`_fragmentation_index`) implement their own spike coincidence matching with a ±1ms window. The two-pointer algorithm in `_update_fr_plot` and the `searchsorted`-based approach in `_fragmentation_index` do subtly different things but serve the same purpose. If the coincidence window or matching logic ever changes, you'd need to update it in two places.

**Plan:**
1. Create `core/spike_match.py` with a shared function:
   ```python
   def match_spikes(times_a, times_b, coincidence_samples):
       """Returns (n_matched, a_only_count, b_only_count, matched_indices_b)."""
   ```
2. Refactor `qc_view_panel.py::_update_fr_plot()` to call it.
3. Refactor `qc_summary_dialog.py::_fragmentation_index()` to call it.
4. Add a unit test in `tests/test_spike_match.py`.

**New Files:** `core/spike_match.py`, `tests/test_spike_match.py`
**Files Modified:** `gui/panels/qc_view_panel.py`, `gui/panels/qc_summary_dialog.py`

---

### 2C: Extract color constants from panels ✅

**Problem:** `qc_view_panel.py` defines a module-level `COLORS` dict. `qc_summary_dialog.py` hardcodes the same hex values inline (e.g., `#4CAF50`, `#2196F3`, `#FF9800`). `array_map_panel.py` also inlines color values. If the design palette changes, every file needs updating independently.

**Plan:**
1. Add a `COLORS` dict to `gui/theme.py` alongside `DARK_STYLESHEET`:
   ```python
   COLORS = {
       "lh": "#4CAF50",
       "soup": "#FF9800",
       "uncertain": "#9E9E9E",
       "cluster0": "#2196F3",
       "cluster1": "#FF9800",
       "sorter": "#2196F3",
       "error": "#F44336",
       ...
   }
   ```
2. Update `qc_view_panel.py`, `qc_summary_dialog.py`, and `array_map_panel.py` to import from `gui.theme`.

**Files Modified:** `gui/theme.py`, `gui/panels/qc_view_panel.py`, `gui/panels/qc_summary_dialog.py`, `gui/panels/array_map_panel.py`

---

### 2D: Deduplicate KiloSort loader internals in `core/loader.py` ✅

**Problem:** `_load_kilosort_times()` (lines 375–475) and `_load_kilosort_units()` (lines 478–577) share ~80 lines of identical logic: loading `.npy` files, resolving `spike_clusters` vs `spike_templates`, computing peak channels from `templates.npy`, and building the `spike_channels` array. The only difference is the final grouping step (flat `{ch: times}` vs nested `{ch: {uid: times}}`).

**Plan:**
1. Extract the shared loading and channel-assignment logic into a private helper:
   ```python
   def _load_kilosort_core(ks_dir):
       """Returns (spike_times, spike_unit_ids, spike_channels) or None."""
   ```
2. Rewrite `_load_kilosort_times()` and `_load_kilosort_units()` as thin wrappers that call `_load_kilosort_core()` and do only the final grouping.
3. This cuts ~80 lines of duplication and ensures channel-assignment logic stays in sync.

**Files Modified:** `core/loader.py`

> ⚠️ **NOTE:** `core/lh_qc_pipeline.py` is **frozen** for this refactor. The loader dedup is safe because `loader.py` is I/O infrastructure, not science logic.

---

### Phase 2 Summary

| Task | Effort | Risk | Files Changed |
|---|---|---|---|
| 2A: Merge worker files | Low | Low — mostly moving code | 4 files, 1 deleted |
| 2B: Spike-match utility | Medium | Low — pure function extraction | 4 files, 2 new |
| 2C: Color constants | Low | None — cosmetic | 4 files |
| 2D: Loader dedup | Medium | Low — internal refactor, public API unchanged | 1 file |

---

## Architecture & Technical Constraints

* **Memory Safety:** The unified task manager MUST retain the `.deleteLater()` and persistent list references to prevent Python GC from killing active threads.
* **Threading Rule:** `OMP_NUM_THREADS="1"` must still be enforced at the individual `QCChannelTask` level.
* **Science Code:** `core/lh_qc_pipeline.py` and `lh_deps/` are READ-ONLY for this spec.

## Test Plan

* **Unit:** Run `conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/ -v` after each sub-task.
* **Memory Safety:** Run `tests/test_memory_leaks.py` to ensure refactored routing doesn't orphan C++ objects.
* **App Launch:** `conda run -n lighthouse_qc env PYTHONPATH=. timeout 5 python run.py` (exit 124 = success).

## Anti-Goals

* Do NOT touch `lh_deps/`.
* Do NOT alter `core/lh_qc_pipeline.py`. (Science logic is frozen.)
* Do NOT add new biophysics metrics or headless CLI arguments.
* Do NOT refactor `gui/panels/load_panel.py` or `gui/panels/array_map_panel.py` internals — they are self-contained and appropriately sized.
