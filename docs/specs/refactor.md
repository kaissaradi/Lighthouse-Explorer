# Specification: Worker Unification & State Consolidation

## Metadata

* **Status:** Ready for Dev
* **Target Release:** v1.0
* **Primary Developer/Agent:** AI Assistant

## Objective

Strip redundant `QThread` boilerplate out of `main_window.py` and unify the execution of single-channel and batch QC into a single thread pool architecture. Extract UI styling to clean up application entry points.

## User Story

"As a developer, I want all background tasks (loading, single QC, batch QC) to be managed by a single unified TaskManager so `main_window.py` only handles UI routing, preventing memory leaks and race conditions."

## Acceptance Criteria (Definition of Done)

* **AC1:** `gui/theme.py` is created to hold `DARK_STYLESHEET`, reducing `app.py` to a strictly functional entry point.
* **AC2:** `main_window.py` no longer manually instantiates `QThread` objects. All background execution is delegated to a unified worker class or task manager.
* **AC3:** `qc_worker.py` is either deleted or refactored to use the exact same `QCChannelTask` (`QRunnable`) that `BatchQCWorker` uses.
* **AC4:** The GUI launches, loads data, and runs both single and batch QC without any visual or functional regressions.

## Architecture & Technical Constraints

* **Files Modified:** `gui/main_window.py`, `gui/app.py`, `gui/workers/qc_worker.py`, `gui/workers/batch_qc_worker.py`.
* **New Files:** `gui/theme.py` (for CSS).
* **Memory Safety:** The unified task manager MUST retain the `.deleteLater()` and persistent list references to prevent Python GC from killing active threads.
* **Threading Rule:** `OMP_NUM_THREADS="1"` must still be enforced at the individual `QCChannelTask` level.

## Test Plan (TDD Requirements)

* **Unit:** Update `test_batch_qc_worker.py` if the class name/structure changes.
* **Memory Safety:** Run `conda run -n lighthouse_qc pytest tests/test_memory_leaks.py` to ensure the new routing doesn't leave orphaned C++ objects.
* **Verification Command:** `conda run -n lighthouse_qc pytest tests/`

## Anti-Goals

* Do NOT touch `lh_deps/`.
* Do NOT alter `core/lh_qc_pipeline.py`. (Science logic must remain frozen during this UI refactor).
* Do NOT add new biophysics metrics or headless CLI arguments yet.
