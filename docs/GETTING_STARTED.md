# Getting Started with Lighthouse Explorer

## Prerequisites
- Conda installed.
- Access to raw recording data (e.g., `.dat`, `.bin`, or Litke folders).

## Setup Environment
Ensure you are using the `lighthouse_qc` conda environment.

```bash
conda activate lighthouse_qc
```

If the environment is not set up, install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
pip install pytest-qt pytest-mock psutil
```

## Running the Application
Launch the GUI using the `run.py` entry point:

```bash
python run.py
```

Optional CLI arguments:
```bash
python run.py --dat /path/to/data.dat --n_channels 385
```

## Running Tests
Always run tests within the `lighthouse_qc` environment.

```bash
conda run -n lighthouse_qc pytest
```

To focus on the batch QC thread-pool and per-channel error behavior:

```bash
conda run -n lighthouse_qc pytest tests/test_batch_qc_worker.py
```

To run the real-data batch concurrency regression test against
`/home/localadmin/Documents/Development/data/raw/20260501A/data000`:

```bash
conda run -n lighthouse_qc pytest tests/test_batch_qc_worker_real_data.py
```

That test is skipped automatically when the local data folder is unavailable.

The test suite sets Qt to offscreen mode automatically for headless terminals and CI.

Batch QC can process multiple channels concurrently. In the GUI, use the
`QC workers` field under `Batch QC`; for scripted runs, pass
`batch_max_workers` in the worker params or set `LIGHTHOUSE_QC_BATCH_THREADS`.
Each channel task still pins BLAS/OpenMP to one native thread to avoid
PCA/KMeans oversubscription.

See [DEVELOPMENT_PROCESS.md](DEVELOPMENT_PROCESS.md) for more details on the TDD workflow.
