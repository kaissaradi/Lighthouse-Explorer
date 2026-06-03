# Lighthouse Explorer — Audit & Refactor Handoff Report

This report compiles the discrepancies, code debt, and refactoring tasks identified in the audit comparing the current codebase (branch `chore/remove-legacy-batch`) against the reference notebook [LH-UW.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py).

---

## 1. Pipeline Science Discrepancies (vs LH-UW.py)

The current implementation of the QC pipeline in [lh_qc_pipeline.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py) contains several differences from the reference notebook [LH-UW.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py). Some are critical bugs affecting scientific accuracy, while others are minor hyperparameter tuning choices.

### 🔴 Critical Issues

#### A. ISI Early Reject Window Mismatch
* **Location**: [lh_qc_pipeline.py:1378](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L1378) vs [LH-UW.py:638-644](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py#L638-L644) (and line 2593)
* **Current Code**:
  ```python
  isi_pairs = np.sum(diffs < 0.002 * fs)  # Counts all ISI < 2 ms (e.g. 40 samples at 20 kHz)
  ```
* **Reference Notebook**:
  ```python
  # Defined in compute_left_isi_pairs_10_30 and _count_isi_10_30
  isi_pairs = int(np.sum((diffs >= 10) & (diffs <= 30)))  # Two-sided refractory window: 10 to 30 samples (0.5–1.5 ms)
  ```
* **Impact**: The current code counts *all* ISI intervals shorter than 2 ms. The reference notebook specifically counts spikes falling inside the 10–30 sample (0.5–1.5 ms) refractory violation window. The current code is too strict and will over-reject channels.
* **Status**: User is aware of this and plans to apply the fix manually.

#### B. `n_total` Returns 0 on Successful Runs
* **Location**: [lh_qc_pipeline.py:152-154](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L152-L154) vs [lh_qc_pipeline.py:1243](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L1243)
* **Current Code**:
  ```python
  @property
  def n_total(self) -> int:
      return int(self.bltr.labels.size)
  ```
  However, on success, `run_bltr_support` returns an empty array for `labels`:
  ```python
  labels = np.array([], dtype=object)  # line 1243
  ```
* **Impact**: `n_total` is hardcoded to return `0` for every channel that completes QC successfully. This breaks downstream consumer code (such as miss-rate calculation and GUI displays) that depend on `n_total`.
* **Proposed Fix**: Sum counts directly from the `bltr.counts` dictionary:
  ```python
  @property
  def n_total(self) -> int:
      return sum(self.bltr.counts.values())
  ```

#### C. PCA Dimensions (`n_pcs` vs `N_PC`)
* **Location**: [lh_qc_pipeline.py:867](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L867) vs [LH-UW.py:1738](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py#L1738)
* **Current Code**: `n_pcs = 3`
* **Reference Notebook**: `N_PC = 2`
* **Impact**: The variance explained thresholds (`pc_var_thr=0.10`) were tuned against 2 PCs. Spreading variance across 3 PCs lowers the individual PC variance ratios, potentially causing valid channels to fail the precheck or changing KMeans cluster geometries.

#### D. Channel Selection for PCA
* **Location**: [lh_qc_pipeline.py:993-1002](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L993-L1002) vs [LH-UW.py:1948-1954](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py#L1948-L1954)
* **Current Code**: Always selects a fixed top-N channels (defaulting to 7, though comments mention 16) by sorting RMS computed from a subset of 100 spikes.
* **Reference Notebook**: Computes the mean EI (`ei_mean = snips.mean(axis=2)`) across all subsampled spikes, calculates the RMS of this mean EI per channel, and selects all channels where `RMS > 5.0` (falling back to the top 16 if no channels exceed the threshold).
* **Impact**: Current code uses a small, noisy sample (100 spikes) to rank channels and uses a fixed number of channels, which can miss high-variance channels on large multi-electrode arrays.

---

### 🟡 Medium/Minor Tuning Discrepancies

#### E. KMeans `n_init`
* **Location**: [lh_qc_pipeline.py:1032](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L1032) vs [LH-UW.py:1971](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py#L1971)
* **Current Code**: `n_init=10`
* **Reference Notebook**: `n_init=50`
* **Impact**: Fewer initializations make KMeans cluster assignments less stable across runs, especially on borderline channels.

#### F. PCA `svd_solver`
* **Location**: [lh_qc_pipeline.py:1028](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py#L1028) vs [LH-UW.py:1961](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/LH-UW.py#L1961)
* **Current Code**: `svd_solver="full"`
* **Reference Notebook**: `svd_solver="randomized"`
* **Impact**: `"full"` solver is deterministic but slower. The reference was built using `"randomized"`, which can cause minor numerical variations in edge cases.

#### G. Valley Bins and Minimum Valid Count
* **Location**: `valley_bins` and `min_valid_count` defaults
* **Current Defaults**: `valley_bins=3`, `min_valid_count=50`
* **Reference Notebook**: `valley_bins=5` (for amplitude histogram segmentation), `min_valid_count=900` (threshold crossings)
* **Impact**: Current code accepts very low-SNR channels (minimum 50 crossings) compared to the reference (900), and uses coarser bins to find amplitude valleys.

#### H. RNG Seed (`random_state`)
* **Current Code**: `random_state=42` uniformly.
* **Reference Notebook**: `random_state=0` for PCA/KMeans, and `np.random.RandomState(123)` for spike subsampling.

---

## 2. Legacy Batch Processing Cleanup (`chore/remove-legacy-batch`)

The pipeline has moved away from multi-file batch processing to simplify codebase maintenance and runtime stability. The following components are dead code and must be stripped:

### A. [gui/qc_worker.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/gui/qc_worker.py)
* **Batch Signals (Lines 136–141)**: Remove `batch_started`, `batch_progress`, `batch_channel_done`, `batch_finished`, `batch_error`, and `batch_aborted`.
* **State Variables (Lines 161–164)**: Remove `self._batch_files`, `self._batch_processed`, `self._batch_total`, and `self._batch_running`.
* **Properties & Methods (Lines 220–322)**:
  * Remove `is_batch_running` property.
  * Remove `start_batch(self, filepath, params)` method.
  * Remove `abort_batch(self)` method.
  * Remove `_on_batch_task_result(self, future)` handler.
  * Remove `_on_batch_task_error(self, future, exc)` handler.
  * Remove `_check_batch_finished(self)` helper.
* **Documentation**: Update class docstring (Lines 108–129) to remove mentions of batch mode.

### B. [gui/main_window.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/gui/main_window.py)
* **Signal Connections (Lines 238–243)**: Remove connections linking `TaskManager` batch signals to UI handlers.
* **Event Handlers (Lines 475–554)**:
  * Remove `_start_batch_qc()`.
  * Remove `_on_batch_progress()`.
  * Remove `_on_batch_channel_done()`.
  * Remove `_on_batch_finished()`.
  * Remove `_on_batch_error()`.
  * Remove `_on_batch_aborted()`.
* **Cleanup UI Calls**:
  * Line 359: Remove batch trigger call inside `_on_loader_finished`.
  * Lines 572–574: Remove `is_batch_running` guard in channel click events.
  * Line 633: Remove `abort_batch` call in close event handler.

### C. Documentation & Tests Cleanup
* **Delete File**: [docs/debug/batch_qc_freeze_handoff.md](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/debug/batch_qc_freeze_handoff.md) (irrelevant after batch removal).
* **Verify Tests**: Remove any remaining batch references from the `tests/` directory (e.g. references to parallel workers or multi-file batches).

---

## 3. Documentation Alignment & Correctness

Current documentation describes components that no longer exist or have been consolidated.

### A. Code Paths / Architecture Updates
The docs refer to a deep nested structure, but the project uses a flattened module organization:

| Document Path | Stale Reference | Actual Path / Status |
|---|---|---|
| `docs/specs/refactor.md` | `gui/workers/qc_worker.py` | [gui/qc_worker.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/gui/qc_worker.py) |
| `docs/specs/refactor.md` | `gui/workers/batch_qc_worker.py` | Deleted |
| `docs/specs/refactor.md` | `gui/panels/` (directory) | [gui/panels.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/gui/panels.py) (consolidated) |
| `docs/specs/refactor.md` | `gui/theme.py` | Styping is embedded / inline CSS |
| `docs/specs/refactor.md` | `core/spike_match.py` | Consolidated into [core/loader.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/loader.py) |
| `docs/specs/refactor.md` | `core/native_threading.py` | Threading logic defined in [core/__init__.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/__init__.py) |
| `docs/specs/refactor.md` | `tests/test_batch_qc_worker.py` | Deleted |

### B. Required Doc Updates
1. **[AGENTS.md](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/AGENTS.md)**: Full rewrite to document the flat `gui/` structure, single-threaded/fork-safe environment controls (Numba initialization), single-channel worker pattern, and testing protocols.
2. **[PLAN.md](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/PLAN.md)**: Rewrite roadmap. Remove completed worker unification/batch tasks, add parameter tuning notes, and highlight Biophysics expansion and Headless Mode as future items.
3. **`docs/specs/refactor.md`**: Move to `docs/archive/refactor.md` or delete since the architecture has changed.

---

## 4. Execution Plan for the Next Agent

### Step 1: Scientific & Pipeline Fixes
* [ ] Fix the `n_total` property in `QCResult` inside [lh_qc_pipeline.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/core/lh_qc_pipeline.py) using the `sum(self.bltr.counts.values())` fix.
* [ ] **Review parameters with the user** regarding whether to align PCA, KMeans, and selection parameters (`n_pcs=2`, `n_init=50`, `RMS_THRESH` thresholding, `valley_bins=5`, `min_valid_count=900`) with the reference notebook or document them as intentional divergences.

### Step 2: Remove Dead Batch Code
* [ ] Delete batch signals, state variables, and methods from [gui/qc_worker.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/gui/qc_worker.py).
* [ ] Delete batch signal handlers and UI guards from [gui/main_window.py](file:///home/fieldlab/Documents/Lighthouse-Explorer/gui/main_window.py).
* [ ] Run `grep -ri "batch" .` to clean up any remaining stale comments or unused imports.

### Step 3: Documentation Cleanup
* [ ] Delete `docs/debug/batch_qc_freeze_handoff.md`.
* [ ] Move `docs/specs/refactor.md` to `docs/archive/` or delete it.
* [ ] Rewrite [AGENTS.md](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/AGENTS.md) and [PLAN.md](file:///home/fieldlab/Documents/Lighthouse-Explorer/docs/PLAN.md) to match the final codebase structure.

### Step 4: Verification
* [ ] Run tests to ensure nothing was broken by batch cleanup:
  ```bash
  conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/ -v
  ```
* [ ] Verify that the application boots up successfully:
  ```bash
  conda run -n lighthouse_qc env PYTHONPATH=. python run.py
  ```
