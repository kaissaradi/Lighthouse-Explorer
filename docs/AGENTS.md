# AI Developer Rulebook, Architecture, & Philosophy

Welcome to the Lighthouse Explorer repository. You are an AI acting as a core developer on this project. This project strictly follows **Spec-Driven Development (SDD)** and **Test-Driven Development (TDD)**.

Read this document in its entirety before modifying any code.

---

## 1. Guiding UX Philosophy

Lighthouse Explorer is a high-speed quality control tool used by researchers to analyze massive Retinal Ganglion Cell (RGC) recordings.

* **Memory Safety First:** We deal with massive files. Never load the entire `.dat` or `.bin` file into RAM. You MUST use `np.memmap(mode='r')`.
* **Zero UI Freezing:** The main thread must *never* be blocked. All heavy computations (baseline subtraction, PCA, KMeans) must run in background threads. UI interactions must feel instantaneous.
* **Pure Visualization:** The pipeline's output is for comparison and visualization against existing sorter calls. We do not rewrite or modify the raw data.

---

## 2. The Architecture & Data Pipeline

* **`core/loader.py` & `core/lh_qc_pipeline.py`:** The pure Python science backend. Parses raw files and runs the 4-step pipeline (valley detection → snippet extraction → PCA/KMeans → BL/TR labeling). No Qt code belongs here.
* **`run.py`:** The entry point of the application. Initializes `QApplication`, applies visual styles, and launches `MainWindow`.
* **`gui/main_window.py`:** The UI orchestration layer and single source of truth for application state. Delegates background QC work to `TaskManager`.
* **`gui/qc_worker.py`:** Contains `LoaderWorker` (File I/O worker using `QObject` + `QThread`) and `TaskManager` (orchestrates background QC tasks run as `QCChannelTask` (`QRunnable`) on a `QThreadPool`). Note: all legacy batch processing methods/signals are deprecated/removed.
* **`gui/panels.py`:** Consolidates UI visualization panels using `pyqtgraph`. These panels are thin visual layers and only react to state changes via Qt Signals.
* **`lh_deps/`:** Vendored upstream utilities. **READ-ONLY.** Do not modify these files.

---

## 3. Environment, Dependencies, & Execution Rules

### 3.1 The Conda Environment

Every terminal command you run MUST use the `lighthouse_qc` conda environment. There are two ways to do this:

```bash
# Option A: prefix every command (preferred for CI / one-shot commands)
conda run -n lighthouse_qc <command>

# Option B: activate in your shell first (preferred for interactive work)
conda activate lighthouse_qc
```

### 3.2 PYTHONPATH Requirement (CRITICAL)

This project does **not** have a `setup.py`, `pyproject.toml`, or `pip install -e .` configuration. The project root is **not** automatically on `sys.path`.

When running tests or scripts via `conda run`, you **MUST** set `PYTHONPATH` to the project root, otherwise you will get `ModuleNotFoundError: No module named 'gui'` or `'core'`:

```bash
# ✅ Correct — tests can find gui/, core/, etc.
conda run -n lighthouse_qc env PYTHONPATH=/path/to/Lighthouse-Explorer pytest tests/ -v

# ❌ Wrong — will fail with ModuleNotFoundError
conda run -n lighthouse_qc pytest tests/ -v
```

For `run.py`, the same applies:
```bash
conda run -n lighthouse_qc env PYTHONPATH=/path/to/Lighthouse-Explorer python run.py
```

### 3.3 Installing Test Dependencies

Test-only packages (pytest, pytest-qt, pytest-mock, psutil) are listed in `requirements-test.txt` at the project root. Install them once into the conda env:

```bash
conda run -n lighthouse_qc pip install -r requirements-test.txt
```

The main application dependencies are in `requirements.txt`. Test deps are separate to keep the production env lean.

### 3.4 Running Tests — Quick Reference

```bash
# Full test suite
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/ -v

# Individual test files
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/test_gui_state.py -v
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/test_spike_match.py -v
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/test_result_types.py -v

# Launch the app (blocks until window is closed; use timeout for smoke-test)
conda run -n lighthouse_qc env PYTHONPATH=. timeout 5 python run.py
# Exit code 124 = timeout killed it = app launched successfully
```

### 3.5 Qt Offscreen Mode

Tests use `QT_QPA_PLATFORM=offscreen` (set in `tests/conftest.py`) so they run headlessly without a display server. Do not remove this from conftest.

### 3.6 OpenMP/BLAS Concurrency Rule (CRITICAL)

We use Scikit-Learn (PCA/KMeans) inside worker threads. You MUST pin native threads (`OMP_NUM_THREADS="1"`, `OPENBLAS_NUM_THREADS="1"`, etc.) at the task level to prevent deadlocks and CPU thread explosion. This is enforced by:

1. `core/__init__.py` → `configure_native_thread_environment()` — sets environment variables at process startup.
2. `core/__init__.py` → `native_thread_limits(1)` — runtime context manager using `threadpoolctl` inside each `QCChannelTask.run()`.

Both layers are required. Environment variables only affect libraries loaded *after* they're set; `threadpoolctl` handles already-loaded libraries.

### 3.7 Garbage Collection (Memory Safety)

When writing `QRunnable` or `QThread` logic:

* You MUST keep a Python reference to active tasks (`self._tasks.append(task)`) to prevent the Python GC from destroying the C++ `QRunnable` while the thread pool is still executing it.
* You MUST call `.deleteLater()` on custom `QObject` signal emitters or workers upon task completion or abortion to ensure proper lifecycle management.

---

## 4. Git Protocol & Branching

Use descriptive, atomic commit messages and standard branch prefixes:

* `feat/` (e.g., `feat/headless-csv-export`)
* `fix/` (e.g., `fix/batch-qc-deadlock`)
* `test/` (e.g., `test/memory-leak-regression`)
* `refactor/` (e.g., `refactor/worker-unification`)

---

## 5. Multi-Agent Concurrency Rules

If operating as part of a multi-agent team, obey these isolation rules:

1. **One Spec = One Branch = One Agent:** Work only on your assigned `SPEC.md`. Stay on your dedicated branch.
2. **Domain Isolation:** If Agent A is refactoring `core/lh_qc_pipeline.py`, Agent B working on `gui/panels/array_map_panel.py` must not touch the core pipeline.
3. **The State Bottleneck:** If your spec modifies cross-panel state in `main_window.py`, rebase frequently (`git pull --rebase main`) to avoid overwriting another agent's UI routing.

---

## 6. The Prime Directives (Workflow)

1. **Never write implementation code without reading the corresponding spec in `docs/specs/` first.**
2. **Always write the failing test in `tests/` before modifying `core/` or `gui/`.**
3. **Qt Threading Rule:** Never update UI elements directly from a background thread. You must use Qt Signals.
4. **Plotting Rule:** Always use `pyqtgraph` for dynamic UI data. Matplotlib is forbidden due to performance overhead.
