from __future__ import annotations

import numpy as np

from core.result_types import (
    BLTRResult,
    PCAKMeansResult,
    QCResult,
    SnippetResult,
    ValleyResult,
)


def make_qc_result(
    channel: int = 0,
    *,
    n_sorter_spikes: int = -1,
    n_lh: int = 1,
    n_soup: int = 0,
) -> QCResult:
    """Build a tiny, structurally valid QCResult for worker and UI tests."""
    labels = np.array(["LH"] * n_lh + ["soup"] * n_soup, dtype=object)
    times = np.arange(labels.size, dtype=np.int64)
    snippet_len = 61

    valley = ValleyResult(
        accepted=True,
        valley_low=-100.0,
        valley_high=-50.0,
        left_times=times,
        left_vals=np.full(times.size, -120.0, dtype=np.float32),
        valley_times=np.array([], dtype=np.int64),
        valley_vals=np.array([], dtype=np.float32),
        all_times=times,
        all_vals=np.full(times.size, -120.0, dtype=np.float32),
        amp_hist_counts=np.array([1], dtype=np.int64),
        amp_hist_edges=np.array([-150.0, -50.0], dtype=np.float32),
        left_count=int(times.size),
        valley_count=0,
    )
    snippets = SnippetResult(
        snippets=np.empty((1, snippet_len, 0), dtype=np.float32),
        times=np.array([], dtype=np.int64),
        n_channels=1,
        snippet_len=snippet_len,
    )
    pca_km = PCAKMeansResult(
        pca_coords=np.empty((0, 3), dtype=np.float32),
        km_labels=np.array([], dtype=np.int64),
        cluster_mean_waveforms=[np.zeros(snippet_len), np.zeros(snippet_len)],
        explained_variance_ratio=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        n_pcs_used=3,
    )
    bltr = BLTRResult(
        labels=labels,
        bl_bulk=np.zeros(labels.size, dtype=np.float32),
        tr_bulk=np.zeros(labels.size, dtype=np.float32),
        counts={
            "LH": int(n_lh),
            "soup": int(n_soup),
            "uncertain_boundary": 0,
            "uncertain_lowBL": 0,
        },
        times=times,
    )

    return QCResult(
        channel=channel,
        n_sorter_spikes=n_sorter_spikes,
        valley=valley,
        snippets=snippets,
        pca_km=pca_km,
        bltr=bltr,
    )
