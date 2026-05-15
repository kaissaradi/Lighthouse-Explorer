# Environment Management

The project uses a dedicated Conda environment named `lighthouse_qc`. It is critical to maintain consistency by using this environment for all development and testing tasks.

## Activation
```bash
conda activate lighthouse_qc
```

## Running Commands
Prefer using `conda run` to ensure the correct context even if the environment isn't activated in your current shell:

```bash
# Run the app
conda run -n lighthouse_qc python run.py

# Run all tests
conda run -n lighthouse_qc pytest

# Install a new dependency
conda run -n lighthouse_qc pip install [package]
```

## Key Dependencies
- **PyQt5/qtpy:** GUI framework.
- **pyqtgraph:** High-performance plotting.
- **pytest-qt:** Extension for testing Qt applications.
- **psutil:** Used for monitoring memory usage in leak tests.
- **numba/numpy/scipy:** Computational backend.

## Troubleshooting
If you experience `ImportError` or `ModuleNotFoundError`, double-check that you are not in the `base` environment.

```bash
conda info --envs
```
The asterisk `*` should be next to `lighthouse_qc`.
