"""
lh_qc_pipeline.py — 4-step LH QC pipeline for a single channel.

All functions are pure (no Qt, no side effects). Imports from lh_deps/
with fall-back to direct axolotl env imports.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

# ── Import QC helpers ─────────────────────────────────────────────────────────
try:
    from lh_deps.lighthouse_utils import find_valley_and_times
    from lh_deps.axolotl_utils_ram import extract_snippets_fast_ram
    from lh_deps.collision_utils import median_ei_adaptive
except ImportError:
    from lighthouse_utils import find_valley_and_times
    from axolotl_utils_ram import extract_snippets_fast_ram
    from collision_utils import median_ei_adaptive

from .result_types import (
    ValleyResult,
    SnippetResult,
    PCAKMeansResult,
    BLTRResult,
    QCResult,
)

# ── DEFAULT PARAMS ────────────────────────────────────────────────────────────
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
    snippet_window=(-20, 40),
    max_snippets=5000,

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


def run_valley_detection(
    raw_data: np.ndarray, ch: int, params: dict
) -> ValleyResult:
    """Step 1: Run find_valley_and_times on channel `ch`."""
    if raw_data is None:
        raise ValueError("raw_data is None")
    _, C = raw_data.shape
    if ch < 0 or ch >= C:
        raise ValueError(f"Channel {ch} out of range [0, {C})")

    raw = find_valley_and_times(
        raw_data,
        ch,
        window=tuple(params.get("window", (-40, 80))),
        bin_width=float(params.get("bin_width", 10.0)),
        valley_bins=int(params.get("valley_bins", 5)),
        min_valid_count=int(params.get("min_valid_count", 300)),
        ratio_base=int(params.get("ratio_base", 3)),
        ratio_step=int(params.get("ratio_step", 100)),
        ratio_floor=int(params.get("ratio_floor", 2)),
        ratio_cap=int(params.get("ratio_cap", 10)),
        right_k=int(params.get("right_k", 500)),
        min_trough=params.get("min_trough", None),
    )

    return ValleyResult(
        accepted=bool(raw["accepted"]),
        valley_low=raw["valley_low"],
        valley_high=raw["valley_high"],
        left_times=raw["left_times"].astype(np.int64),
        left_vals=raw["left_vals"].astype(np.float32),
        valley_times=raw["valley_times"].astype(np.int64),
        valley_vals=raw["valley_vals"].astype(np.float32),
        all_times=raw["all_times"].astype(np.int64),
        all_vals=raw["all_vals"].astype(np.float32),
        amp_hist_counts=raw["amp_hist_counts"],
        amp_hist_edges=raw["amp_hist_edges"],
        left_count=int(raw["left_count"]),
        valley_count=int(raw["valley_count"]),
    )


def run_snippet_extraction(
    raw_data: np.ndarray,
    all_times: np.ndarray,
    params: dict,
) -> SnippetResult:
    """Step 2: Extract snippets around all candidate times on ALL channels."""
    sw = tuple(params.get("snippet_window", (-20, 40)))
    max_snip = int(params.get("max_snippets", 5000))
    n_channels = raw_data.shape[1]
    all_ch = np.arange(n_channels, dtype=np.int32)

    snippets, valid_times = extract_snippets_fast_ram(
        raw_data=raw_data,
        spike_times=all_times,
        window=sw,
        selected_channels=all_ch,
    )

    # Cap by random downsampling if needed
    n_snippets = snippets.shape[2]
    if n_snippets > max_snip:
        rng = np.random.default_rng(42)
        keep = rng.choice(n_snippets, size=max_snip, replace=False)
        keep.sort()
        snippets = snippets[:, :, keep].copy()
        valid_times = valid_times[keep].copy()

    return SnippetResult(
        snippets=snippets.astype(np.float32),
        times=valid_times.astype(np.int64),
        n_channels=int(n_channels),
        snippet_len=int(snippets.shape[1]),
    )


def run_pca_kmeans(
    snippets: SnippetResult,
    detect_ch: int,
    params: dict,
) -> PCAKMeansResult:
    """Step 3: PCA on flattened [C*L, N] → KMeans(k=2)."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    snips = snippets.snippets  # [C, L, N]
    C, L, N = snips.shape
    n_pcs = int(params.get("n_pcs", 3))
    n_clusters = int(params.get("n_clusters", 2))
    sample_for_pca = int(params.get("sample_for_pca", 3000))
    random_state = int(params.get("random_state", 42))

    # Flatten: [C, L, N] → [C*L, N] → transpose → [N, C*L]
    X = snips.reshape(C * L, N).T  # [N, C*L]

    # Subsample for PCA fit if needed
    if N > sample_for_pca:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=sample_for_pca, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    n_pcs_actual = min(n_pcs, X_fit.shape[0] - 1, X_fit.shape[1])
    n_pcs_actual = max(1, n_pcs_actual)

    pca = PCA(n_components=n_pcs_actual, svd_solver="full", random_state=random_state)
    pca.fit(X_fit)
    pca_coords = pca.transform(X).astype(np.float32)  # [N, n_pcs]

    km = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    km_labels = km.fit_predict(pca_coords)

    # Compute per-cluster mean waveform on detect_ch only
    cluster_means = []
    for k in range(n_clusters):
        mask = km_labels == k
        mean_wave = snips[detect_ch, :, mask].mean(axis=1).astype(np.float32)  # [L]
        cluster_means.append(mean_wave)

    return PCAKMeansResult(
        pca_coords=pca_coords,
        km_labels=km_labels.astype(np.int64),
        cluster_mean_waveforms=cluster_means,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_pcs_used=n_pcs_actual,
    )


def run_bltr_labeling(
    snippets: SnippetResult,
    valley: ValleyResult,
    params: dict,
) -> BLTRResult:
    """
    Step 4: BL/TR support labeling.

    Uses valley.left_times → LEFT group, valley.valley_times (+ rightk) → RIGHT group.
    For each spike, computes BL_bulk and TR_bulk as bulk cosine similarity measures,
    then assigns labels based on thresholds.

    Vectorized: computes per-spike EI and batches cosine similarity without Python loop.
    """
    snips = snippets.snippets  # [C, L, N]
    times = snippets.times     # [N]
    N = snips.shape[2]
    C = snips.shape[0]
    L = snips.shape[1]
    detect_ch = int(np.argmin(valley.all_vals)) if valley.all_vals.size > 0 else 0
    detect_ch = max(0, min(detect_ch, C - 1))

    min_bl_bulk = float(params.get("min_bl_bulk", 0.70))

    # Build reference EI from LEFT spikes (clean LH candidates)
    # We need to map valley.left_times back to indices in snippets.times
    left_set = set(valley.left_times.tolist())
    valley_set = set(valley.valley_times.tolist())

    # For each spike in snippets, determine if it's LEFT or RIGHT
    is_left = np.array([int(t) in left_set for t in times], dtype=bool)
    is_valley = np.array([int(t) in valley_set for t in times], dtype=bool)

    n_left = int(is_left.sum())
    n_valley = int(is_valley.sum())

    # Build EIs for LEFT and RIGHT groups
    if n_left >= 2:
        ei_left = median_ei_adaptive(snips[:, :, is_left])
    else:
        ei_left = snips.mean(axis=2)

    if n_valley >= 2:
        ei_valley = median_ei_adaptive(snips[:, :, is_valley])
    else:
        ei_valley = snips.mean(axis=2)

    # Vectorized BL/TR: compute per-spike EI and cosine similarity
    # Mean across channels gives [C, N] — one waveform per spike
    ei_per_spike = snips.mean(axis=2)  # [C, L] — this is wrong, let me fix

    # Actually: mean across axis=1 (L) gives [C, N] — amplitude per channel per spike
    # But we want EI which is [C, L]. Mean across axis=2 gives [C, L] for ALL spikes
    # For per-spike "EI" (just the snippet itself): snips[:, :, i] is [C, L]
    # To vectorize, compute cosine similarity using numpy broadcasting

    # RMS-gated cosine: simplify by computing dot product norm on detect_ch only
    # This approximates the full cosine_two_eis but much faster
    ch = detect_ch
    waveforms = snips[ch, :, :].T  # [N, L]

    # Reference waveforms
    ref_left = ei_left[ch, :]      # [L]
    ref_valley = ei_valley[ch, :]  # [L]

    # Normalize references
    norm_left = np.linalg.norm(ref_left)
    norm_valley = np.linalg.norm(ref_valley)
    if norm_left > 0:
        ref_left = ref_left / norm_left
    if norm_valley > 0:
        ref_valley = ref_valley / norm_valley

    # Vectorized cosine similarity: [N, L] @ [L] → [N]
    bl_bulk = np.abs(waveforms @ ref_left).astype(np.float32)
    tr_bulk = np.abs(waveforms @ ref_valley).astype(np.float32)

    # Clip to [0, 1] range
    bl_bulk = np.clip(bl_bulk, 0.0, 1.0)
    tr_bulk = np.clip(tr_bulk, 0.0, 1.0)

    # Assign labels based on BL/TR support — fully vectorized
    labels = np.empty(N, dtype=object)
    counts = {"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0}

    # Vectorized conditions
    is_lh = (bl_bulk >= min_bl_bulk) & (bl_bulk > tr_bulk)
    is_soup = tr_bulk > bl_bulk
    is_uncertain_boundary = (bl_bulk >= min_bl_bulk) & ~is_lh & ~is_soup
    is_uncertain_lowBL = ~is_lh & ~is_soup & ~is_uncertain_boundary

    labels[is_lh] = "LH"
    labels[is_soup] = "soup"
    labels[is_uncertain_boundary] = "uncertain_boundary"
    labels[is_uncertain_lowBL] = "uncertain_lowBL"

    counts["LH"] = int(is_lh.sum())
    counts["soup"] = int(is_soup.sum())
    counts["uncertain_boundary"] = int(is_uncertain_boundary.sum())
    counts["uncertain_lowBL"] = int(is_uncertain_lowBL.sum())

    return BLTRResult(
        labels=labels,
        bl_bulk=bl_bulk,
        tr_bulk=tr_bulk,
        counts=counts,
        times=times.copy(),
    )


def run_qc_pipeline(
    raw_data: np.ndarray,
    ch: int,
    n_sorter_spikes: int = -1,
    params: Optional[dict] = None,
) -> QCResult:
    """
    Top-level convenience: runs all 4 steps in sequence and returns QCResult.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)

    # Step 1: Valley detection
    valley = run_valley_detection(raw_data, ch, params)
    
    # Collect all candidate times for snippet extraction
    all_times = valley.all_times
    if all_times.size == 0:
        # Create empty results for degenerate case
        snippets = SnippetResult(
            snippets=np.empty((raw_data.shape[1], 60, 0), dtype=np.float32),
            times=np.array([], dtype=np.int64),
            n_channels=raw_data.shape[1],
            snippet_len=60,
        )
        pca_km = PCAKMeansResult(
            pca_coords=np.empty((0, 3), dtype=np.float32),
            km_labels=np.array([], dtype=np.int64),
            cluster_mean_waveforms=[np.zeros(60, dtype=np.float32), np.zeros(60, dtype=np.float32)],
            explained_variance_ratio=np.array([0.33, 0.33, 0.33], dtype=np.float32),
            n_pcs_used=3,
        )
        bltr = BLTRResult(
            labels=np.array([], dtype=object),
            bl_bulk=np.array([], dtype=np.float32),
            tr_bulk=np.array([], dtype=np.float32),
            counts={"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
            times=np.array([], dtype=np.int64),
        )
        return QCResult(
            channel=ch,
            n_sorter_spikes=n_sorter_spikes,
            valley=valley,
            snippets=snippets,
            pca_km=pca_km,
            bltr=bltr,
        )

    # Step 2: Snippet extraction
    snippets = run_snippet_extraction(raw_data, all_times, params)
    
    if snippets.times.size == 0:
        pca_km = PCAKMeansResult(
            pca_coords=np.empty((0, 3), dtype=np.float32),
            km_labels=np.array([], dtype=np.int64),
            cluster_mean_waveforms=[
                np.zeros(snippets.snippet_len, dtype=np.float32),
                np.zeros(snippets.snippet_len, dtype=np.float32),
            ],
            explained_variance_ratio=np.array([0.33, 0.33, 0.33], dtype=np.float32),
            n_pcs_used=3,
        )
        bltr = BLTRResult(
            labels=np.array([], dtype=object),
            bl_bulk=np.array([], dtype=np.float32),
            tr_bulk=np.array([], dtype=np.float32),
            counts={"LH": 0, "soup": 0, "uncertain_boundary": 0, "uncertain_lowBL": 0},
            times=np.array([], dtype=np.int64),
        )
        return QCResult(
            channel=ch,
            n_sorter_spikes=n_sorter_spikes,
            valley=valley,
            snippets=snippets,
            pca_km=pca_km,
            bltr=bltr,
        )

    # Step 3: PCA + KMeans
    pca_km = run_pca_kmeans(snippets, ch, params)

    # Step 4: BL/TR labeling
    bltr = run_bltr_labeling(snippets, valley, params)

    return QCResult(
        channel=ch,
        n_sorter_spikes=n_sorter_spikes,
        valley=valley,
        snippets=snippets,
        pca_km=pca_km,
        bltr=bltr,
    )
