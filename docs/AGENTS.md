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
* **`gui/app.py` & `gui/main_window.py`:** The UI orchestration layer. The single source of truth for the application's state.
* **`gui/workers/`:** All file I/O, heavy math, and batch processing executed via `QThread` or `QRunnable`.
* **`gui/panels/`:** Thin UI layers using `pyqtgraph`. They do no heavy lifting and only react to state changes via Qt Signals.
* **`lh_deps/`:** Vendored upstream utilities. **READ-ONLY.** Do not modify these files.

---

## 3. Environment, Data, & Execution Rules

* **The Conda Environment:** Every terminal command you run MUST use the `lighthouse_qc` conda environment (e.g., `conda run -n lighthouse_qc pytest tests/`).
* **OpenMP/BLAS Concurrency Rule (CRITICAL):** We use Scikit-Learn (PCA/KMeans) inside worker threads. You MUST pin native threads (`OMP_NUM_THREADS="1"`, `OPENBLAS_NUM_THREADS="1"`) at the task level before batch processing to prevent deadlocks and CPU thread explosion.
* **Garbage Collection (Memory Leaks):** When writing `QThread` or `QRunnable` logic, you MUST implement `.deleteLater()` to clean up C++ object references upon task completion or abortion. Long-running GUI sessions will otherwise leak RAM.

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
