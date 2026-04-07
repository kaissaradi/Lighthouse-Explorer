# Lighthouse QC — Standalone GUI

A standalone PyQt5/qtpy application that loads a raw `.dat`/`.bin` recording, lets the user browse the electrode array spatially, picks a channel, and runs the full 4-step Lighthouse QC pipeline (valley detection → snippet extraction → PCA/KMeans → BL/TR labeling) **without any subtraction**. The output is a pure visualization: how many threshold crossings look like clean LH spikes vs. soup vs. uncertain, compared to what the sorter already called. No KS dependency. Standalone executable entry point.

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

```bash
python run.py
```

For power users, you can specify default paths:

```bash
python run.py --dat /path/to/data.dat --n_channels 512
```

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