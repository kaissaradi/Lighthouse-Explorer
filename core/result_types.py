from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ValleyResult:
    """Output of find_valley_and_times for one channel."""
    accepted: bool
    valley_low: Optional[float]
    valley_high: Optional[float]
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
    rightk_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # added


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
    cluster_mean_waveforms: list  # [np.ndarray, np.ndarray] each shape [L] on detect_ch
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
    reject_reason: Optional[str] = None   # added for early rejections

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
    def miss_rate(self) -> Optional[float]:
        """Fraction of LH-labeled spikes NOT found by sorter. None if n_sorter_spikes unknown."""
        if self.n_sorter_spikes < 0:
            return None
        if self.n_lh == 0:
            return 0.0
        missed = max(0, self.n_lh - self.n_sorter_spikes)
        return missed / self.n_lh