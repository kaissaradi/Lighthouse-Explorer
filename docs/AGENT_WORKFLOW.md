# Agent Workflow: Testing & Verification

When making changes to the Lighthouse Explorer codebase, use the following workflow to ensure your changes are safe, correctly tested, and don't introduce regressions.

## 1. Environment Context
All commands must be executed within the `lighthouse_qc` conda environment. Because you are running non-interactively, you must prepend your commands with `conda run -n lighthouse_qc` and set `PYTHONPATH=.` from the project root.

## 2. Running the Test Suite
After making code changes, always run the full test suite. 

**Command:**
```bash
conda run -n lighthouse_qc env PYTHONPATH=. pytest tests/ -v
```

**Expected Output:**
- All tests should pass (or skip if real data is missing).
- *Note:* If you see an error in `test_gui_state.py` about the `mocker` fixture not being found, this is a known pre-existing issue caused by `pytest-mock` not being installed. You can safely ignore it unless your task specifically involves GUI state testing.

## 3. The App Launch Smoke Test
Because this is a PyQt5 GUI application, you cannot test the UI interactively. Instead, you must use a headless timeout smoke test to verify that the app launches successfully without crashing on startup (e.g., due to import errors or syntax errors).

**Command:**
```bash
conda run -n lighthouse_qc env PYTHONPATH=. timeout 5 python run.py 2>&1; echo "EXIT_CODE=$?"
```

**Expected Output:**
- The command will attempt to launch the app and automatically kill it after 5 seconds.
- You should look for `EXIT_CODE=124`. Exit code 124 indicates that the process was successfully terminated by the `timeout` command, meaning the app launched and stayed alive for 5 seconds without crashing.
- Any other exit code (like 1 or 139) indicates a crash or import failure that you must investigate.

## 4. Checking Memory Leaks
The test suite includes a memory leak regression test (`tests/test_memory_leaks.py`). If you touch worker threads, threading limits, or C++ references (like Qt signals or `QRunnable`), pay special attention to this test to ensure no memory leak was introduced.

## 5. Threading Rules (CRITICAL)
If you are writing background tasks, you **must** adhere to the native threading limits.
NumPy, SciPy, and Scikit-Learn will default to using all CPU cores. If multiple threads do this simultaneously, the C++ OpenMP backend will deadlock. 
- Always import and use the `native_thread_limits(1)` context manager from `gui.workers.qc_worker` inside your `QRunnable` or `QThread` `run()` methods.

## 6. Committing
Once tests and the smoke test pass:
1. Verify what files were modified/added: `git status --short`
2. Check your net file changes: `git diff --stat HEAD`
3. Stage and commit with a descriptive message.
