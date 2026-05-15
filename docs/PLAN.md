# Lighthouse Explorer Master Development Plan

## Project Vision

Lighthouse Explorer is a standalone PyQt5/pyqtgraph application tailored for the visual curation and quality control of spike-sorted electrophysiology data. It supports dual modes: an interactive GUI for visual inspection and manual override, and a headless CLI mode for automated batch processing and CSV export.

---

## Current Milestone: v1.0 - Architecture Consolidation & Headless Export

*The focus of this milestone is solidifying the worker architecture, ensuring thread safety during batch QC, and expanding the core data structures to calculate biophysics for downstream transcriptomic comparisons.*

### 1. Active Priorities (In Order of Execution)

**Priority 1: Codebase Consolidation & Worker Unification**

* **Goal:** Streamline the existing architecture to reduce boilerplate and centralize state management *before* introducing new features.
* **Spec needed:** `docs/specs/worker_unification.md`
* **Architecture Change:** Extract common threading boilerplate, error handling, and signal emissions from `qc_worker.py` and `batch_qc_worker.py` into a shared base class. Ensure `main_window.py` acts as the sole state controller.

**Priority 2: Data Model & Biophysics Foundation**

* **Goal:** Expand `QCResult` to calculate and store physiological ground-truth data.
* **Spec needed:** `docs/specs/biophysics_expansion.md`
* **Focus:** Implement `compute_biophysics()` in `core/lh_qc_pipeline.py` for temporal metrics (Hz, ISI violations) and waveform metrics. Ensure tests pass using mock spike arrays without GUI dependencies.

**Priority 3: Headless Mode & CSV Export**

* **Goal:** Enable automated, UI-free batch processing of datasets.
* **Spec needed:** `docs/specs/headless_cli_export.md`
* **Architecture Change:** Update `run.py` to accept CLI arguments (`--headless`, `--export-csv`). Bypass `QApplication` to run the batch QC loop and flatten the expanded `QCResult` into a Pandas-ready CSV format.

**Priority 4: GUI Curation & Session Persistence**

* **Goal:** Allow users to save progress and manually override KMeans verdicts.
* **Spec needed:** `docs/specs/gui_curation_overrides.md`

---

## Testing & Infrastructure Initiatives

* **Memory Safety Regression:** Periodically run `tests/test_memory_leaks.py` with `psutil` to ensure UI interactions and canceled batch tasks properly garbage collect C++ bindings.
* **C++ Backend Guard:** Ensure `tests/test_batch_qc_worker.py` strictly verifies that a single channel failure does not halt the entire QThreadPool batch.
