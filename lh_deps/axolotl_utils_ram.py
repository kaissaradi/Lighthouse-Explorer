import numpy as np
import os

def compute_baselines_int16_deriv_robust(raw_data, segment_len=100_000, diff_thresh=50):
    """
    Compute mean baseline per channel over non-overlapping segments,
    using derivative masking (vectorized) to suppress spike influence.
    """
    total_samples, n_channels = raw_data.shape
    n_segments = (total_samples + segment_len - 1) // segment_len

    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len, total_samples)
        segment = raw_data[start:end, :]  # [S, C]

        if segment.shape[0] < 2:
            continue

        # Compute absolute derivative
        diff_segment = np.abs(np.diff(segment, axis=0))  # [S-1, C]
        diff_segment = np.vstack([diff_segment, diff_segment[-1]])  

        # Vectorized Masking: Keep low-derivative points, turn spikes into NaNs
        masked_segment = np.where(diff_segment < diff_thresh, segment, np.nan)

        # Compute the mean of the noise band across all channels simultaneously
        b_vals = np.nanmean(masked_segment, axis=0)
        
        # Handle the rare case where an entire segment for a channel is NaNs
        baselines[:, seg_idx] = np.nan_to_num(b_vals, nan=0.0)

    return baselines


def subtract_segment_baselines_int16(raw_data: np.ndarray,
                                     baselines_f32: np.ndarray,
                                     segment_len: int = 100_000) -> None:
    """
    In-place baseline removal for int16 raw traces.

    Parameters
    ----------
    raw_data      : [T, C]  int16   – entire recording in RAM
    baselines_f32 : [C, n_segments] float32 – from compute_baselines_int16_deriv_robust
    segment_len   : int               – same value that was passed to the baseline routine

    Returns
    -------
    None   (raw_data is modified in place)
    """

    T, C = raw_data.shape
    C_b, n_seg = baselines_f32.shape
    if C_b != C:
        raise ValueError("Channel count mismatch between raw_data and baselines")

    # Quantise baselines once; cost ≈ 1 kB
    baselines_i16 = np.rint(baselines_f32).astype(np.int16)

    for seg_idx in range(n_seg):
        start = seg_idx * segment_len
        end   = min(start + segment_len, T)          # handle last partial segment

        # Broadcast-subtract:  [end-start, C]  -=  [C]
        raw_data[start:end, :] -= baselines_i16[:, seg_idx]


def extract_snippets_fast_ram(
        raw_data: np.ndarray,           # [T, C]  int16
        spike_times: np.ndarray,        # [N]     int64 / int32
        window: tuple[int, int],        # (pre, post)  e.g. (-20, 40)
        selected_channels: np.ndarray   # [K]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return snippets [K, L, N] (float32) and the spike times that were
    inside bounds.  No Python loop, single memory copy.
    """
    pre, post = window
    win_len = post - pre + 1
    total_samples, _ = raw_data.shape

    m = (spike_times + pre >= 0) & (spike_times + post < total_samples)
    valid_times = spike_times[m]
    N = valid_times.size

    if selected_channels is None:
        selected_channels = np.arange(raw_data.shape[1], dtype=np.int32)
    K = len(selected_channels)

    if N == 0:
        return np.empty((K, win_len, 0), np.float32), valid_times

    offsets = np.arange(pre, post + 1, dtype=np.int64)

    # Preallocate output [K, L, N]
    K = int(len(selected_channels))
    out = np.empty((K, win_len, N), dtype=np.float32)

    # Heuristic memory budget for the per-batch temp (in MB); override via env if desired
    budget_mb = int(os.environ.get("AX_SNIP_BATCH_MB", "256"))
    budget_bytes = max(64, budget_mb) * (1024**2)

    # Estimate bytes per sample we need to hold at once:
    #   tmp int16 gather + float32 slice we will write into `out`
    bytes_per_elem = raw_data.dtype.itemsize + np.dtype(np.float32).itemsize  # 2 + 4 = 6
    # batch size B so that B*L*K*6 <= budget
    B = max(1, int(budget_bytes // (bytes_per_elem * win_len * max(1, K))))
    B = min(B, N)  # never exceed N
    if B <= 0:
        B = 1

    # Fast path when channels are a contiguous 1-step range (including "all channels")
    ch = np.asarray(selected_channels, dtype=np.int64)
    is_contig = (ch.size > 0) and np.all(np.diff(ch) == 1)
    if is_contig:
        ch0 = int(ch[0]); ch1 = int(ch[-1]) + 1

    for i0 in range(0, N, B):
        i1 = min(i0 + B, N)
        vt = valid_times[i0:i1]
        # rows for this batch
        rows = (vt[:, None] + offsets[None, :]).reshape(-1)               # [B*L]

        # Gather rows first (axis=0) to keep the big dimension linear in memory
        tmp_rows = np.take(raw_data, rows, axis=0)                         # [B*L, C] int16

        # Select channels
        if is_contig:
            tmp = tmp_rows[:, ch0:ch1]                                     # [B*L, K] view
        else:
            tmp = np.take(tmp_rows, ch, axis=1)                            # [B*L, K] copy

        # Reshape to [K, L, B] and write directly into the preallocated output
        tmp = tmp.reshape(i1 - i0, win_len, K).transpose(2, 1, 0)          # [K, L, B]
        out[:, :, i0:i1] = tmp.astype(np.float32, copy=False)

    return out, valid_times
