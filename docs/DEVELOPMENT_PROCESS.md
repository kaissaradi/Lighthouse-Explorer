# Development Process: TDD and SPEC-DD

To ensure the stability and correctness of the Lighthouse Explorer GUI, we follow a combined Specification-Driven (SPEC-DD) and Test-Driven Development (TDD) approach.

## 1. Specification First (SPEC-DD)
Before implementing any new feature or fix:
1.  **Define Requirements:** Update `SPEC.md` in the project root. Detail the expected behavior, state changes, and memory management goals.
2.  **Define Verification:** Explicitly state how the requirement will be verified (e.g., "Automated test in `tests/test_new_feature.py`").

## 2. Test-Driven Development (TDD)
Once the spec is defined:
1.  **Write Failing Tests:** Create or update a file in the `tests/` directory. Use `pytest-qt` for GUI interactions and `mocker` for isolating dependencies.
2.  **Run Tests:** Confirm they fail in the `lighthouse_qc` environment.
    ```bash
    conda run -n lighthouse_qc pytest tests/your_new_test.py
    ```
3.  **Implement:** Write the minimum code necessary to make the tests pass.
4.  **Refactor & Verify:** Clean up the code while ensuring all tests (including existing ones) continue to pass.

## 3. Memory Safety
A key goal is preventing memory leaks in the long-running GUI.
- **Check for Leaks:** Periodically run `tests/test_memory_leaks.py`.
- **Cleanup Patterns:** Always ensure background workers (QThreads/QRunnables) are properly disconnected and deleted (`deleteLater`) after finishing or being aborted.

## 4. Environment Consistency
Never run implementation scripts or tests in the `base` environment. Always use `lighthouse_qc`.

```bash
# Correct way to run any command
conda run -n lighthouse_qc [command]
```
