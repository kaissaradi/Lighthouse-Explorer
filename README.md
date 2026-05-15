# Lighthouse QC — Standalone GUI

A standalone PyQt5/qtpy application that loads a raw `.dat`/`.bin` recording or Litke folder, lets the user browse the electrode array spatially, picks a channel, and runs the full 4-step Lighthouse QC pipeline (valley detection → snippet extraction → PCA/KMeans → BL/TR labeling). 

For **Kilosort-format single files**, the app automatically subtracts baselines (using derivative-robust baseline estimation) before running the QC pipeline. For **Litke-format folders**, data is loaded as-is without baseline subtraction (since preprocessing is typically done upstream).

The output is a pure visualization: how many threshold crossings look like clean LH spikes vs. soup vs. uncertain, compared to what the sorter already called. No KS dependency. Standalone executable entry point.

## Setup

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda installed

### Environment Setup

1. Create a new conda environment:

```bash
conda create -n lighthouse_qc python=3.9
conda activate lighthouse_qc
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda install numpy scipy scikit-learn pyqtgraph qtpy pyqt h5py
```

### Running the Application

Set `PYTHONPATH` to the project root (there is no editable install):

```bash
export PYTHONPATH=.
python run.py
```

For power users, you can specify default paths:

```bash
python run.py --dat /path/to/data.dat --n_channels 512
```

### Known limitations (batch QC / threading)

See [`docs/debug/batch_qc_freeze_handoff.md`](docs/debug/batch_qc_freeze_handoff.md) for full context.

- **Use QC workers = 1** (default in the load panel) for stable batch QC on large Litke recordings. Values **> 1** can produce Numba/TBB fork warnings and the process may be killed.
- Mitigations in code: `core/native_threading.py` (`NUMBA_THREADING_LAYER=workqueue`, `NUMBA_NUM_THREADS=1`), `core/numba_kernels.py` (lock around Numba), `TaskManager` honors the worker spinbox and recording `fs`.
- A single TBB fork warning may still appear on stderr even with 1 worker; batch may still complete.

## Documentation

Detailed documentation on development and usage is available in the `docs/` folder:

- [Getting Started](docs/GETTING_STARTED.md)
- [File Tree](docs/FILE_TREE.md)
- [Development Process (TDD/SPEC-DD)](docs/DEVELOPMENT_PROCESS.md)
- [Environment Management](docs/ENVIRONMENT.md)
- [Batch QC / Numba handoff](docs/debug/batch_qc_freeze_handoff.md) (threading issues, lab findings)

## Repository Structure

```
lighthouse_qc/
│
├── README.md
├── requirements.txt              # Python dependencies
├── run.py                        # Entry point
│
├── core/                         # Pure-Python science code
│   ├── __init__.py
│   ├── loader.py                 # Raw data loading
│   ├── native_threading.py       # OMP/BLAS/Numba env (before Numba import)
│   ├── numba_kernels.py          # Thread-safe wrappers for lh_deps Numba
│   ├── lh_qc_pipeline.py         # QC pipeline implementation
│   └── result_types.py           # Data structures
│
├── gui/                          # Qt-based GUI
│   ├── __init__.py
│   ├── app.py                    # QApplication setup
│   ├── main_window.py            # Main window
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── load_panel.py         # Load panel
│   │   ├── array_map_panel.py    # Array map
│   │   └── qc_view_panel.py      # QC visualization
│   └── workers/
│       ├── __init__.py
│       └── qc_worker.py          # Background worker
│
└── lh_deps/                      # Vendored LH utilities
    ├── __init__.py
    ├── lighthouse_utils.py
    ├── axolotl_utils_ram.py
    ├── collision_utils.py
    └── joint_utils.py
```

## Dependencies

The application requires the following files to be copied from the axolotl codebase into `lh_deps/`:

- `lighthouse_utils.py`
- `axolotl_utils_ram.py`
- `collision_utils.py`
- `joint_utils.py`

These are proprietary utilities and must be obtained separately.