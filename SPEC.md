
# Lighthouse Explorer GUI Specification

## 1. Data Loading and State Management

### 1.1 Loading a New Recording
- **Requirement:** When a new recording is loaded, all previous state related to the old recording MUST be cleared.
- **State to Clear:**
    - `self.raw_data`
    - `self.qc_results`
    - `self.sorter_spike_times`
    - `self.sorter_unit_map`
    - `self.sorter_dom_channel`
    - Cached results in `ArrayMapPanel` and `QCViewPanel`.
- **Verification:** Automated test should load a mock dataset, then load another, and verify the state variables are reset.

### 1.2 Aborting Operations
- **Requirement:** Starting a new load or QC operation MUST abort any currently running background tasks (LoaderWorker, QCWorker, BatchQCWorker).
- **Verification:** Mock background workers and verify `abort()` is called.

## 2. Memory Efficiency

### 2.1 Worker Cleanup
- **Requirement:** Workers and their associated threads MUST be properly deleted after completion to prevent memory accumulation.
- **Verification:** Memory leak test using `psutil` over multiple load/QC cycles.

### 2.2 Large Data Handling
- **Requirement:** `raw_data` should be handled as `np.memmap` whenever possible for flat files to avoid loading entire recordings into RAM.
- **Verification:** Check `LoaderWorker` implementation and verify RSS memory usage remains stable for large file loads.

## 3. QC Pipeline Consistency

### 3.1 Result Integrity
- **Requirement:** `QCResult` must contain all necessary fields for biophysics and clustering analysis as defined in `core/result_types.py`.
- **Verification:** Unit tests for `run_qc_pipeline`.

## 4. UI Responsiveness

### 4.1 Background Execution
- **Requirement:** All heavy I/O and computations MUST run in background threads. The UI main thread MUST NOT freeze.
- **Verification:** GUI tests monitoring event loop responsiveness during QC runs.

### 4.2 Batch QC Thread Safety
- **Requirement:** Batch QC MUST keep native BLAS/OpenMP worker threads pinned to one thread per task while allowing the Qt task pool to process multiple channels concurrently. The GUI exposes a `QC workers` setting, and `BatchQCWorker` also accepts `max_workers` / `batch_max_workers` for tests and scripted runs.
- **Requirement:** Native thread limits MUST be applied both before GUI startup and around every channel task at runtime, because NumPy/SciPy/scikit-learn may already be imported before environment variables are set.
- **Requirement:** Batch QC MUST keep Python references to every queued `QRunnable` until the batch completes or is aborted, and one channel failure MUST be counted as completed without stopping the rest of the batch.
- **Verification:** Automated worker tests with a fake `QThreadPool` and mocked QC pipeline verify configurable Qt concurrency above one, native BLAS/OpenMP thread guards, task-reference retention, successful aggregation, per-channel error recovery, and abort cleanup.
