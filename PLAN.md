pl# Lighthouse QC — Standalone GUI: Development Plan

## Overview

A standalone PyQt5/qtpy application that loads a raw `.dat`/`.bin` recording, lets the
user browse the electrode array spatially, picks a channel, and runs the full
4-step Lighthouse QC pipeline (valley detection → snippet extraction → PCA/KMeans →
BL/TR labeling) **without any subtraction or writing**. The output is a pure
visualization: how many threshold crossings look like clean LH spikes vs. soup vs.
uncertain, compared to what the sorter already called. No KS dependency. Standalone
executable entry point.

---

## Current Status

### Completed
- Directory structure created: `lighthouse_qc/` with all subdirectories (`core/`, `gui/`, `gui/panels/`, `gui/workers/`, `lh_deps/`)
- `requirements.txt`: Dependencies listed as per plan (numpy, scipy, scikit-learn, pyqtgraph, qtpy, PyQt5, h5py)
- `README.md`: Setup instructions for conda environment creation and dependency installation
- `run.py`: Entry point with CLI argument parsing for default paths
- All `__init__.py` files created (mostly empty)
- `core/result_types.py`: Full implementation of all dataclasses (ValleyResult, SnippetResult, PCAKMeansResult, BLTRResult, QCResult) with properties

### Remaining Tasks
- `core/loader.py`: Implement data loading functions (load_raw_readonly, load_channel_map, load_sorter_spike_times, compute_channel_firing_rates)
- `core/lh_qc_pipeline.py`: Implement the 4-step QC pipeline functions (run_valley_detection, run_snippet_extraction, run_pca_kmeans, run_bltr_labeling, run_qc_pipeline)
- `gui/app.py`: Implement QApplication setup, dark stylesheet, and run function
- `gui/main_window.py`: Implement the main window layout, signal connections, and QC workflow orchestration
- `gui/panels/load_panel.py`: Implement the load panel UI with file browsers and parameter inputs
- `gui/panels/array_map_panel.py`: Implement the electrode array map with clickable channels and coloring modes
- `gui/panels/qc_view_panel.py`: Implement the QC visualization panel with 4 pyqtgraph plots and summary stats
- `gui/workers/qc_worker.py`: Implement the background worker for QC pipeline execution
- `lh_deps/` files: Copy `lighthouse_utils.py`, `axolotl_utils_ram.py`, `collision_utils.py`, `joint_utils.py` from axolotl codebase (currently empty skeletons)

### Implementation Order (Updated)
1. ✅ `core/result_types.py` — dataclasses, no deps, test immediately
2. `core/loader.py` — test with a real `.dat` file
3. `lh_deps/` — copy utility files, verify imports work
4. `core/lh_qc_pipeline.py` — implement and unit-test each step function in isolation
5. `gui/app.py` + `gui/main_window.py` skeleton — get window to open
6. `gui/panels/load_panel.py` — get load flow working end to end
7. `gui/panels/array_map_panel.py` — get array rendering + click working
8. `gui/workers/qc_worker.py` — wire background execution
9. `gui/panels/qc_view_panel.py` — implement plots one at a time (hist → PCA → BLTR → waveforms)
10. Integration test: full run on a real dataset

---

## Repo Structure

```
lighthouse_qc/
│
├── README.md
├── requirements.txt              # pyqt5/qtpy, pyqtgraph, numpy, scipy, scikit-learn, h5py
├── run.py                        # entry point: `python run.py`
│
├── core/                         # pure-Python, NO Qt — all the science lives here
│   ├── __init__.py
│   ├── loader.py                 # raw data loading (adapted from axolotl/io.py)
│   ├── lh_qc_pipeline.py         # the 4-step QC pipeline for a single channel
│   └── result_types.py           # dataclasses for pipeline outputs
│
├── gui/
│   ├── __init__.py
│   ├── app.py                    # QApplication setup, theme, entry
│   ├── main_window.py            # top-level QMainWindow, layout orchestration
│   │
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── load_panel.py         # left sidebar: file paths, params, Load button
│   │   ├── array_map_panel.py    # electrode array spatial map (clickable channels)
│   │   └── qc_view_panel.py      # right side: 4 pyqtgraph plots + summary stats
│   │
│   └── workers/
│       ├── __init__.py
│       └── qc_worker.py          # QObject worker for background pipeline execution
│
└── lh_deps/                      # vendored copies of LH utility deps
    ├── __init__.py
    ├── lighthouse_utils.py       # COPY of lighthouse_utils.py (provided)
    ├── axolotl_utils_ram.py      # COPY — only extract_snippets_fast_ram needed
    ├── collision_utils.py        # COPY — only median_ei_adaptive needed
    └── joint_utils.py            # COPY — only cosine_two_eis needed
```

**Key decision on `lh_deps/`:** Rather than requiring the user to have the full axolotl
environment on `PYTHONPATH`, vendor the four utility files with a thin `__init__.py`
re-export. The dev should copy these files verbatim and only import the needed symbols.
If the user already has axolotl on path, the imports fall through naturally.

---

## Dependencies (`requirements.txt`)

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
pyqtgraph>=0.13
qtpy>=2.4
PyQt5>=5.15
h5py>=3.9
```

---

## Module-by-Module Spec

---

### `run.py`

Entry point. Parses optional CLI args for default paths so power users can launch
directly: `python run.py --dat /path/to/data.dat --n_channels 512`

```python
def main():
    """Parse CLI args, construct QApplication, show MainWindow, exec."""
    pass
```

---

### `core/result_types.py`

Dataclasses only. No Qt. No numpy ops.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ValleyResult:
    """Output of find_valley_and_times for one channel."""
    accepted: bool
    valley_low: float | None
    valley_high: float | None
    left_times: np.ndarray      # int64 [NL]
    left_vals: np.ndarray       # float32 [NL]
    valley_times: np.ndarray    # int64 [NV]
    valley_vals: np.ndarray     # float32 [NV]
    all_times: np.ndarray       # int64 — ALL threshold crossings
    all_vals: np.ndarray        # float32
    amp_hist_counts: np.ndarray
    amp_hist_edges: np.ndarray
    left_count: int
    valley_count: int

@dataclass
class SnippetResult:
    """Output of snippet extraction step."""
    snippets: np.ndarray        # float32 [C, L, N] — all channels, all spikes
    times: np.ndarray           # int64 [N] — valid spike times (boundary-filtered)
    n_channels: int
    snippet_len: int

@dataclass
class PCAKMeansResult:
    """Output of PCA + KMeans step."""
    pca_coords: np.ndarray      # float32 [N, n_pcs] — for scatter plot
    km_labels: np.ndarray       # int [N] — cluster 0 or 1
    cluster_mean_waveforms: list[np.ndarray]  # [2] each shape [L] on detect_ch
    explained_variance_ratio: np.ndarray
    n_pcs_used: int

@dataclass
class BLTRResult:
    """Output of BL/TR support labeling step."""
    labels: np.ndarray          # object dtype ['LH','soup','uncertain_boundary','uncertain_lowBL']
    bl_bulk: np.ndarray         # float32 [N]
    tr_bulk: np.ndarray         # float32 [N]
    counts: dict                # {'LH': int, 'soup': int, 'uncertain_boundary': int, ...}
    times: np.ndarray           # int64 [N] — same indexing as labels

@dataclass
class QCResult:
    """Full result for one channel QC run."""
    channel: int
    n_sorter_spikes: int        # how many spikes the existing sorter (KS/LH) found on this ch
    valley: ValleyResult
    snippets: SnippetResult
    pca_km: PCAKMeansResult
    bltr: BLTRResult

    @property
    def n_total(self) -> int:
        return int(self.bltr.labels.size)

    @property
    def n_lh(self) -> int:
        return int(self.bltr.counts.get('LH', 0))

    @property
    def n_soup(self) -> int:
        return int(self.bltr.counts.get('soup', 0))

    @property
    def n_uncertain(self) -> int:
        counts = self.bltr.counts
        return int(counts.get('uncertain_boundary', 0) + counts.get('uncertain_lowBL', 0))

    @property
    def miss_rate(self) -> float | None:
        """Fraction of LH-labeled spikes NOT found by sorter. None if n_sorter_spikes unknown."""
        if self.n_sorter_spikes < 0:
            return None
        if self.n_lh == 0:
            return 0.0
        missed = max(0, self.n_lh - self.n_sorter_spikes)
        return missed / self.n_lh
```

---

### `core/loader.py`

Adapted from `axolotl/io.py`. Responsible for:
- Memory-mapping the raw `.dat`/`.bin` file (read-only — we never modify)
- Loading a channel map (`.npy` electrode positions, shape `[C, 2]`)
- Loading optional existing spike times from KS output dir or LH HDF5 for miss-rate QC

```python
def load_raw_readonly(dat_path: str, n_channels: int, dtype: str = 'int16',
                      start_min: float = 0.0, duration_min: float | None = None,
                      fs: int = 20_000) -> np.ndarray:
    """
    Memory-map a chunk of a raw binary file as READ-ONLY [T, C] int16.
    No temp copy — we never write to it.
    Returns np.memmap shape (T, C).
    """
    pass

def load_channel_map(map_path: str) -> np.ndarray:
    """
    Load electrode positions from .npy file.
    Returns [C, 2] float array of (x, y) positions in microns.
    Falls back to a linear layout if path is None.
    """
    pass

def load_sorter_spike_times(source_path: str, n_channels: int) -> dict[int, np.ndarray]:
    """
    Load existing sorted spike times, keyed by channel index.
    Supports:
      - Kilosort output dir (reads spike_times.npy + spike_clusters.npy + templates.npy
        to find peak channel per unit)
      - LH HDF5 file (reads peak_channel attr + spike_times dataset per unit)
    Returns {channel_idx: spike_times_array} for quick miss-rate lookup.
    Falls back gracefully to empty dict if path is None or unrecognized format.
    """
    pass

def compute_channel_firing_rates(sorter_spike_times: dict[int, np.ndarray],
                                 duration_s: float) -> dict[int, float]:
    """
    Given sorter_spike_times {ch: times}, return {ch: FR_hz}.
    Used to color the array map by activity.
    """
    pass
```

---

### `core/lh_qc_pipeline.py`

The heart of the app. Runs the 4-step LH QC on a single channel against already-loaded
`raw_data`. All functions are pure (no Qt, no side effects). This module imports from
`lh_deps/` (or axolotl env if available).

```python
# Imports — try lh_deps first, fall back to direct axolotl imports
try:
    from lh_deps.lighthouse_utils import find_valley_and_times
    from lh_deps.axolotl_utils_ram import extract_snippets_fast_ram
    from lh_deps.collision_utils import median_ei_adaptive
    from lh_deps.joint_utils import cosine_two_eis
except ImportError:
    from lighthouse_utils import find_valley_and_times
    from axolotl_utils_ram import extract_snippets_fast_ram
    from collision_utils import median_ei_adaptive
    from joint_utils import cosine_two_eis

from .result_types import ValleyResult, SnippetResult, PCAKMeansResult, BLTRResult, QCResult


# ── DEFAULT PARAMS ────────────────────────────────────────────────────────────
# These mirror the LH_UW.py notebook defaults. Exposed here so the GUI can
# pass user-edited values through without touching this file.

DEFAULT_PARAMS = dict(
    # Valley detection
    window=(-40, 80),
    bin_width=10.0,
    valley_bins=5,
    min_valid_count=300,
    ratio_base=3,
    ratio_step=100,
    ratio_floor=2,
    ratio_cap=10,
    right_k=500,
    min_trough=None,

    # Snippet extraction
    snippet_window=(-20, 40),   # tighter window for PCA/BL/TR (km_win in LH_UW)
    max_snippets=5000,          # cap N for speed in QC mode

    # PCA / KMeans
    n_pcs=3,
    n_clusters=2,
    sample_for_pca=3000,
    random_state=42,

    # BL/TR
    cos_mask_adc=30.0,
    k_peak=(5, 10, 20),
    k_bulk=(50, 100, 200),
    min_bl_bulk=0.70,
    diag_eps=0.05,
)


def run_valley_detection(raw_data: np.ndarray, ch: int, params: dict) -> ValleyResult:
    """
    Step 1: Run find_valley_and_times on channel `ch`.
    Returns ValleyResult wrapping the raw dict output.
    Raises ValueError if raw_data is None or ch is out of range.
    """
    pass


def run_snippet_extraction(raw_data: np.ndarray, times: np.ndarray,
                           params: dict) -> SnippetResult:
    """
    Step 2: Extract snippets around all candidate times on ALL channels.
    Uses extract_snippets_fast_ram with params['snippet_window'].
    Caps at params['max_snippets'] by random downsampling if needed.
    Returns SnippetResult.
    """
    pass


def run_pca_kmeans(snippets: SnippetResult, detect_ch: int,
                   params: dict) -> PCAKMeansResult:
    """
    Step 3: PCA on flattened [C*L, N] snippet matrix → KMeans(k=2).
    Projects to n_pcs components, fits KMeans, computes per-cluster mean
    waveform on detect_ch only (for the overlay plot).
    Returns PCAKMeansResult.
    """
    pass


def run_bltr_labeling(snippets: SnippetResult, valley: ValleyResult,
                      params: dict) -> BLTRResult:
    """
    Step 4: BL/TR support labeling.
    Uses valley.left_times → sn_bl and valley.valley_times/rightk → sn_tr.
    Calls compute_bl_tr_support_decisions_from_groups (from LH_UW.py helpers,
    inlined or imported from lh_deps).
    Each spike in snippets.times gets a label and (BL_bulk, TR_bulk) coords.
    Returns BLTRResult.

    NOTE: The BL/TR labeling is defined over the LEFT vs RIGHT split from
    valley detection, not per-arbitrary-spike. Implementation must correctly
    align indices: sn_bl = snippets for left_times, sn_tr = snippets for
    valley_times (or rightk_times). See LH_UW.py lines ~280-430 for the
    exact compute_bl_tr_support_decisions_from_groups call signature.
    """
    pass


def run_qc_pipeline(raw_data: np.ndarray, ch: int,
                    n_sorter_spikes: int = -1,
                    params: dict | None = None) -> QCResult:
    """
    Top-level convenience: runs all 4 steps in sequence and returns QCResult.
    `n_sorter_spikes` = how many spikes the existing sorter found on this ch
    (pass -1 if unknown). Used only for miss_rate display.
    `params` defaults to DEFAULT_PARAMS if None.
    """
    pass
```

---

### `gui/workers/qc_worker.py`

QObject-based worker following the same pattern as axolotl's existing workers.
Runs `run_qc_pipeline` on a background QThread so the GUI stays responsive.

```python
from qtpy.QtCore import QObject, Signal
from core.lh_qc_pipeline import run_qc_pipeline, DEFAULT_PARAMS
from core.result_types import QCResult

class QCWorker(QObject):
    progress = Signal(str)          # status message for status bar
    finished = Signal(object)       # emits QCResult on success
    error = Signal(str)             # emits error message string

    def __init__(self, raw_data, channel: int, n_sorter_spikes: int,
                 params: dict):
        super().__init__()
        self.raw_data = raw_data
        self.channel = channel
        self.n_sorter_spikes = n_sorter_spikes
        self.params = params
        self._abort = False

    def abort(self):
        """Signal the worker to stop. Checked between pipeline steps."""
        self._abort = True

    def run(self):
        """
        Called by the QThread. Runs pipeline steps 1-4 sequentially,
        emitting progress() between steps and finished(result) at end.
        Catches all exceptions and emits error() instead of crashing.
        """
        pass
```

---

### `gui/panels/load_panel.py`

Left sidebar widget. Purely UI — emits signals upward, does no I/O itself.

```python
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import Signal

class LoadPanel(QWidget):
    # Emitted when user clicks "Load" with all fields filled
    load_requested = Signal(dict)   # dict of all params (paths, n_channels, fs, etc.)
    # Emitted when user clicks "Load Sorter" (optional, for miss-rate QC)
    sorter_load_requested = Signal(str)  # path to KS dir or LH HDF5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        """
        Build the sidebar layout:

        ┌─ Recording ──────────────────────┐
        │ .dat/.bin path    [Browse]        │
        │ n_channels        [512      ]     │
        │ dtype             [int16 ▼  ]     │
        │ fs (Hz)           [20000    ]     │
        │ start (min)       [0.0      ]     │
        │ duration (min)    [30.0     ]     │  ← None = full file
        ├─ Channel Map ────────────────────┤
        │ .npy path         [Browse]        │
        │ (optional — fallback: linear)    │
        ├─ Sorter Output (optional) ───────┤
        │ KS dir or LH .h5  [Browse]        │
        │ (for miss-rate display)          │
        ├─ LH Params ──────────────────────┤
        │ [Advanced ▼]  (collapsible)      │
        │  min_valid_count  [300      ]    │
        │  min_bl_bulk      [0.70     ]    │
        │  max_snippets     [5000     ]    │
        │  snippet_win pre  [-20      ]    │
        │  snippet_win post [40       ]    │
        ├──────────────────────────────────┤
        │         [ Load Recording ]       │
        └──────────────────────────────────┘
        """
        pass

    def get_params(self) -> dict:
        """Read all widget values and return as a params dict."""
        pass

    def set_loading_state(self, loading: bool):
        """Disable/enable the Load button during I/O."""
        pass
```

---

### `gui/panels/array_map_panel.py`

Interactive electrode array spatial map. Each electrode is a colored circle.
Click → triggers QC run on that channel.

```python
import pyqtgraph as pg
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel

class ArrayMapPanel(QWidget):
    channel_selected = Signal(int)   # emitted on channel click; int = channel index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ei_positions = None    # [C, 2] float, microns
        self._channel_colors = {}    # {ch: (r,g,b)} for current coloring mode
        self._selected_ch = None
        self._scatter = None         # pg.ScatterPlotItem
        self._build_ui()

    def _build_ui(self):
        """
        PlotWidget with a ScatterPlotItem for electrodes.
        Top toolbar: coloring mode selector (FR / QC miss-rate / flat).
        """
        pass

    def set_array(self, ei_positions: np.ndarray):
        """
        Load electrode positions and render all channels as gray dots.
        ei_positions: [C, 2] array of (x, y) microns.
        """
        pass

    def set_channel_colors(self, color_map: dict[int, tuple]):
        """
        Update dot colors. color_map: {ch_idx: (r, g, b, a)}.
        Called after data load (FR coloring) or after a QC run (miss-rate coloring).
        """
        pass

    def set_selected_channel(self, ch: int):
        """Highlight the selected channel with a ring/larger dot."""
        pass

    def set_qc_result_color(self, ch: int, miss_rate: float | None):
        """
        Color a single channel after its QC run completes.
        miss_rate 0.0=green ... 1.0=red. None=gray (no sorter data).
        """
        pass

    def _on_click(self, scatter, points):
        """Handle pyqtgraph scatter click — extract channel index, emit signal."""
        pass
```

**Coloring modes:**
- **Flat**: all channels same gray
- **FR**: color by sorter firing rate on that channel (blue→yellow heatmap)
- **QC**: color by miss_rate after runs (green=low miss, red=high miss, gray=not yet run)

---

### `gui/panels/qc_view_panel.py`

The main right-side panel. Four pyqtgraph plots arranged in a 2×2 grid plus a
summary stats bar at top.

```python
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from core.result_types import QCResult

class QCViewPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_result: QCResult | None = None
        self._build_ui()

    def _build_ui(self):
        """
        Layout:

        ┌─ Summary bar ───────────────────────────────────────────────────────┐
        │  CH: 42  |  Total: 1823  |  LH: 1204 (66%)  |  Soup: 412 (23%)    │
        │  Uncertain: 207 (11%)  |  Sorter found: 1150  |  Miss: ~4.5%       │
        └─────────────────────────────────────────────────────────────────────┘
        ┌─ Plot 1: Amp Histogram ──┬─ Plot 2: PCA Scatter ───────────────────┐
        │                          │                                          │
        │  bar chart of all_vals   │  PC1 vs PC2, colored by KMeans label    │
        │  valley_low marked       │  cluster 0=blue, cluster 1=orange       │
        │  left/right shading      │                                          │
        └──────────────────────────┴──────────────────────────────────────────┘
        ┌─ Plot 3: BL/TR Scatter ──┬─ Plot 4: Mean Waveforms ────────────────┐
        │                          │                                          │
        │  BL_bulk vs TR_bulk      │  mean waveform per KMeans cluster on    │
        │  LH=green, soup=orange   │  detect_ch only; amplitude envelope     │
        │  uncertain=gray          │  LH spikes overlay in green             │
        │  decision boundary lines │                                          │
        └──────────────────────────┴──────────────────────────────────────────┘
        """
        pass

    def show_result(self, result: QCResult):
        """
        Main entry point. Called after QCWorker.finished fires.
        Updates all 4 plots and the summary bar.
        """
        self._current_result = result
        self._update_summary_bar(result)
        self._update_amp_histogram(result)
        self._update_pca_scatter(result)
        self._update_bltr_scatter(result)
        self._update_waveforms(result)

    def show_loading(self, channel: int):
        """Replace plots with a 'Running QC on CH X...' placeholder."""
        pass

    def show_error(self, msg: str):
        """Show error message in place of plots."""
        pass

    def clear(self):
        """Reset all plots to empty state."""
        pass

    # ── private plot updaters ─────────────────────────────────────────────────

    def _update_summary_bar(self, result: QCResult):
        """Update the text labels in the summary bar from result properties."""
        pass

    def _update_amp_histogram(self, result: QCResult):
        """
        Plot 1: amplitude histogram.
        - Bar chart of result.valley.all_vals (all threshold crossings)
        - Vertical dashed line at result.valley.valley_low (threshold)
        - Left region (< valley_low) shaded blue = LEFT/clean candidates
        - Right region (>= valley_low) shaded orange = valley/soup candidates
        - X axis: ADC amplitude (negative). Y axis: count.
        """
        pass

    def _update_pca_scatter(self, result: QCResult):
        """
        Plot 2: PCA scatter.
        - result.pca_km.pca_coords[:, 0] vs [:, 1]
        - Colored by result.pca_km.km_labels (cluster 0=blue, 1=orange)
        - Axis labels: PC1, PC2, explained variance %
        """
        pass

    def _update_bltr_scatter(self, result: QCResult):
        """
        Plot 3: BL/TR scatter.
        - X = result.bltr.bl_bulk, Y = result.bltr.tr_bulk
        - Color by label: LH=#4CAF50, soup=#FF9800, uncertain=#9E9E9E
        - Diagonal identity line (BL=TR) as dashed reference
        - Horizontal/vertical lines at min_bl_bulk param
        - Diagonal band ± diag_eps shaded
        """
        pass

    def _update_waveforms(self, result: QCResult):
        """
        Plot 4: mean waveforms.
        - Two overlaid traces: cluster 0 and cluster 1 mean waveform on detect_ch
        - Thin semi-transparent envelope (10-90th percentile of LH-labeled spikes)
        - X axis: samples, Y axis: ADC amplitude
        - Legend: cluster 0 / cluster 1 / LH envelope
        """
        pass
```

---

### `gui/main_window.py`

Top-level `QMainWindow`. Orchestrates panels and worker lifecycle.

```python
from qtpy.QtWidgets import QMainWindow, QSplitter, QStatusBar, QProgressBar, QThread
from qtpy.QtCore import Qt
from .panels.load_panel import LoadPanel
from .panels.array_map_panel import ArrayMapPanel
from .panels.qc_view_panel import QCViewPanel
from .workers.qc_worker import QCWorker
from core import loader
from core.result_types import QCResult

class MainWindow(QMainWindow):

    def __init__(self, default_dat=None, default_n_channels=None):
        super().__init__()
        self.setWindowTitle("Lighthouse QC")
        self.setGeometry(100, 100, 1600, 900)

        # State
        self.raw_data = None          # np.memmap [T, C]
        self.ei_positions = None      # [C, 2]
        self.sorter_spike_times = {}  # {ch: times}
        self.qc_results = {}          # {ch: QCResult} — cache completed runs
        self.current_channel = None
        self.lh_params = {}           # from load_panel, merged with DEFAULT_PARAMS
        self._qc_thread = None
        self._qc_worker = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        """
        Layout:
        ┌─ LoadPanel (250px) ─┬─ ArrayMapPanel (350px) ─┬─ QCViewPanel (fill) ─┐
        │  file paths         │  electrode array          │  4 QC plots          │
        │  params             │  clickable channels       │  summary bar         │
        │  [Load] button      │  coloring modes           │                      │
        └─────────────────────┴───────────────────────────┴──────────────────────┘
        Status bar at bottom: progress bar + message
        """
        pass

    def _connect_signals(self):
        """Wire all inter-panel signals."""
        # load_panel.load_requested → self.on_load_requested
        # load_panel.sorter_load_requested → self.on_sorter_load_requested
        # array_map_panel.channel_selected → self.on_channel_selected
        pass

    # ── Data loading ──────────────────────────────────────────────────────────

    def on_load_requested(self, params: dict):
        """
        Slot for load_panel.load_requested.
        Calls loader.load_raw_readonly(), loader.load_channel_map(),
        updates array_map_panel, enables channel selection.
        Runs in main thread (fast — just mmapping).
        """
        pass

    def on_sorter_load_requested(self, path: str):
        """
        Slot for load_panel.sorter_load_requested.
        Calls loader.load_sorter_spike_times(), computes FR,
        updates array_map_panel color to FR mode.
        """
        pass

    # ── QC run lifecycle ──────────────────────────────────────────────────────

    def on_channel_selected(self, ch: int):
        """
        Slot for array_map_panel.channel_selected.
        If result cached in self.qc_results[ch]: show immediately.
        Else: abort any running worker, start new QCWorker for ch.
        """
        pass

    def _start_qc_worker(self, ch: int):
        """
        Spin up QThread + QCWorker for channel ch.
        Connects worker signals: progress→status bar, finished→on_qc_finished,
        error→on_qc_error.
        Stores refs in self._qc_thread, self._qc_worker.
        """
        pass

    def _abort_current_worker(self):
        """Call worker.abort(), wait for thread to finish, cleanup refs."""
        pass

    def on_qc_finished(self, result: QCResult):
        """
        Slot for QCWorker.finished.
        Caches result in self.qc_results[result.channel].
        Calls qc_view_panel.show_result(result).
        Calls array_map_panel.set_qc_result_color(ch, miss_rate).
        Updates status bar.
        """
        pass

    def on_qc_error(self, msg: str):
        """Show error in status bar and qc_view_panel."""
        pass

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _n_sorter_spikes_for_channel(self, ch: int) -> int:
        """
        Return count of sorter spikes on channel ch, or -1 if unknown.
        Sums all units whose peak channel == ch.
        """
        pass

    def closeEvent(self, event):
        """Abort worker, close memmap, accept."""
        pass
```

---

### `gui/app.py`

```python
from qtpy.QtWidgets import QApplication
from .main_window import MainWindow

DARK_STYLESHEET = """
/* minimal dark theme — mirrors axolotl RGC Viewer palette */
QMainWindow, QWidget { background: #111214; color: #F0F0F2; }
/* ... */
"""

def create_app(argv) -> tuple[QApplication, MainWindow]:
    """Build QApplication, apply stylesheet, return (app, window)."""
    pass

def run(argv, default_dat=None, default_n_channels=None):
    """Full entry: create_app → show → exec."""
    pass
```

---

## Data Flow (end to end)

```
User fills LoadPanel
        │
        ▼
on_load_requested()
  loader.load_raw_readonly()   → self.raw_data  [T,C] memmap (READ ONLY)
  loader.load_channel_map()    → self.ei_positions [C,2]
  loader.load_sorter_spike_times() → self.sorter_spike_times {ch: times}
  array_map_panel.set_array()
  array_map_panel.set_channel_colors(FR mode)
        │
        ▼
User clicks electrode on ArrayMapPanel
  array_map_panel.channel_selected(ch)
        │
        ▼
on_channel_selected(ch)
  check qc_results cache → hit: show_result() immediately
  miss: _start_qc_worker(ch)
        │
        ▼  [background QThread]
QCWorker.run()
  run_valley_detection(raw_data, ch, params)   → ValleyResult
  run_snippet_extraction(raw_data, times, params) → SnippetResult
  run_pca_kmeans(snippets, ch, params)         → PCAKMeansResult
  run_bltr_labeling(snippets, valley, params)  → BLTRResult
  emit finished(QCResult)
        │
        ▼  [main thread]
on_qc_finished(result)
  qc_results[ch] = result
  qc_view_panel.show_result(result)
  array_map_panel.set_qc_result_color(ch, miss_rate)
```

---

## What the Dev Still Needs to Supply / Copy

The following files need to be **copied verbatim** from the axolotl codebase into
`lh_deps/`. The plan cannot auto-generate these — they are existing proprietary
utility modules:

| File | Key symbols needed |
|---|---|
| `lighthouse_utils.py` | `find_valley_and_times`, `compute_bl_tr_support_decisions_from_groups`, `choose_adaptive_km_window` |
| `axolotl_utils_ram.py` | `extract_snippets_fast_ram`, `compute_baselines_int16_deriv_robust`, `subtract_segment_baselines_int16` |
| `collision_utils.py` | `median_ei_adaptive` |
| `joint_utils.py` | `cosine_two_eis` |

`lighthouse_utils.py` is already provided. The other three need to be obtained from
the axolotl repo.

Additionally, the dev should verify the **exact call signature** of
`compute_bl_tr_support_decisions_from_groups` in `lighthouse_utils.py` (lines 336-429
of the provided file) before implementing `run_bltr_labeling` — it takes `sn_bl` and
`sn_tr` as `[C, L, N]` snippet arrays, NOT pre-computed metrics.

---

## Notes / Gotchas

- **Read-only memmap**: use `mode='r'` in `load_raw_readonly`. We never call baseline
  subtraction or template subtraction. The pipeline reads raw_data only.
- **Snippet cap**: `max_snippets=5000` prevents the QC tool from being slow on channels
  with very high firing rates. Random downsample before PCA/BL-TR, but keep the valley
  result counts from ALL crossings for the histogram.
- **BL/TR requires N≥2 on each side**: guard in `run_bltr_labeling`. If `left_count < 2`
  or `valley_count < 2`, skip BL/TR and return NaN labels.
- **Thread safety**: `raw_data` is a memmap opened read-only — safe to pass to worker
  thread without copying. Never write to it.
- **Cancellation**: `QCWorker.abort()` sets a flag checked between steps 1/2/3/4.
  The individual LH functions are not interruptible, but steps are coarse enough (~0.5s
  each) that this is fine for QC use.
- **Cache invalidation**: if the user changes LH params in LoadPanel and reruns a
  channel, `qc_results` must be cleared for that channel. A "Clear Cache" button in
  LoadPanel is sufficient.
- **`lh_deps` import fallback**: the try/except import block in `lh_qc_pipeline.py`
  means the tool works both as a standalone repo (using vendored deps) and inside
  the existing axolotl environment (using the live modules).
