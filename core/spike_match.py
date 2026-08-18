"""LH ↔ KS coincidence comparison modes.

LH ``final_times`` are ground truth for evaluating sorter quality.
Modes:
  - per_unit (default): match each KS unit on the channel independently;
    pick a primary unit when one dominates LH coverage.
  - good_only: pool only Phy ``good`` units, then Venn vs LH.
  - all_pool: pool all non-noise units on the channel (legacy behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from core.loader import match_spikes

# Modes exposed in the UI
COMPARE_MODES = ("per_unit", "good_only", "all_pool")
DEFAULT_COMPARE_MODE = "per_unit"

# Confident primary-unit criteria (LH as GT)
# primary must match at least this fraction of LH spikes
_MIN_PRIMARY_RECALL = 0.40
# and at least this fraction of all unit-matched LH spikes (dominance)
_MIN_PRIMARY_DOMINANCE = 0.55
# absolute floor so tiny channels don't claim confidence
_MIN_PRIMARY_MATCHED = 20


@dataclass
class UnitMatchStats:
    """Per-unit coincidence stats vs LH ground truth."""

    unit_id: int
    label: str  # good / mua / unsorted / unknown
    n_unit: int
    n_matched: int
    n_ks_only: int
    recall: float  # n_matched / n_lh
    precision: float  # n_matched / n_unit


@dataclass
class CompareResult:
    """Result of comparing LH ground-truth times to KS units on one channel."""

    mode: str
    n_lh: int
    n_matched: int
    n_lh_only: int
    n_ks_only: int
    n_ks_total: int
    unit_stats: List[UnitMatchStats] = field(default_factory=list)
    primary_unit_id: Optional[int] = None
    confident: bool = False
    note: str = ""

    @property
    def recall(self) -> float:
        if self.n_lh <= 0:
            return 0.0
        return float(self.n_matched) / float(self.n_lh)

    @property
    def primary(self) -> Optional[UnitMatchStats]:
        if self.primary_unit_id is None:
            return None
        for u in self.unit_stats:
            if u.unit_id == self.primary_unit_id:
                return u
        return None


def _norm_label(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    if s in ("good", "mua", "noise", "unsorted"):
        return s
    return s or "unknown"


def _filter_unit_map(
    unit_map: Mapping[int, np.ndarray],
    unit_labels: Mapping[int, str],
    mode: str,
) -> Dict[int, np.ndarray]:
    """Select units for pooling modes."""
    out: Dict[int, np.ndarray] = {}
    for uid, times in unit_map.items():
        arr = np.asarray(times, dtype=np.int64)
        if arr.size == 0:
            continue
        lab = _norm_label(unit_labels.get(int(uid)))
        if mode == "good_only" and lab != "good":
            continue
        # all_pool / per_unit: keep everything already non-noise (noise was dropped at parse)
        out[int(uid)] = np.sort(arr)
    return out


def _pool_times(unit_map: Mapping[int, np.ndarray]) -> np.ndarray:
    if not unit_map:
        return np.array([], dtype=np.int64)
    parts = [np.asarray(v, dtype=np.int64) for v in unit_map.values() if np.asarray(v).size]
    if not parts:
        return np.array([], dtype=np.int64)
    if len(parts) == 1:
        return np.sort(parts[0])
    return np.sort(np.concatenate(parts))


def _per_unit_stats(
    lh_times: np.ndarray,
    unit_map: Mapping[int, np.ndarray],
    unit_labels: Mapping[int, str],
    coincidence: int,
) -> List[UnitMatchStats]:
    n_lh = int(lh_times.size)
    stats: List[UnitMatchStats] = []
    for uid in sorted(unit_map.keys()):
        ut = np.sort(np.asarray(unit_map[uid], dtype=np.int64))
        n_m, _, n_ks_only, _ = match_spikes(lh_times, ut, coincidence)
        n_u = int(ut.size)
        stats.append(
            UnitMatchStats(
                unit_id=int(uid),
                label=_norm_label(unit_labels.get(int(uid))),
                n_unit=n_u,
                n_matched=int(n_m),
                n_ks_only=int(n_ks_only),
                recall=float(n_m) / float(n_lh) if n_lh else 0.0,
                precision=float(n_m) / float(n_u) if n_u else 0.0,
            )
        )
    # Rank by matched descending for UI
    stats.sort(key=lambda s: (-s.n_matched, s.unit_id))
    return stats


def _pick_primary(stats: Sequence[UnitMatchStats], n_lh: int) -> tuple[Optional[int], bool]:
    """Return (primary_unit_id, confident)."""
    if not stats or n_lh <= 0:
        return None, False
    ranked = [s for s in stats if s.n_matched > 0]
    if not ranked:
        return None, False
    primary = ranked[0]
    total_matched_sum = sum(s.n_matched for s in ranked)
    # Note: total_matched_sum can exceed n_lh if units claim overlapping LH spikes;
    # dominance still useful as a relative measure.
    dominance = (
        primary.n_matched / float(total_matched_sum) if total_matched_sum > 0 else 0.0
    )
    confident = (
        primary.n_matched >= _MIN_PRIMARY_MATCHED
        and primary.recall >= _MIN_PRIMARY_RECALL
        and dominance >= _MIN_PRIMARY_DOMINANCE
    )
    return primary.unit_id, confident


def compare_lh_ks(
    lh_times: np.ndarray,
    unit_map: Optional[Mapping[int, np.ndarray]] = None,
    *,
    unit_labels: Optional[Mapping[int, str]] = None,
    pooled_times: Optional[np.ndarray] = None,
    mode: str = DEFAULT_COMPARE_MODE,
    fs: float = 20_000.0,
    coincidence_samples: Optional[int] = None,
) -> CompareResult:
    """Compare LH ground-truth spike times to KS units.

    Parameters
    ----------
    lh_times
        Sorted LH final_times (ground truth). Empty if rejected / no LH.
    unit_map
        {unit_id: spike_times} for units peaking on this channel.
    unit_labels
        {unit_id: 'good'|'mua'|'unsorted'|...} from cluster_group.tsv.
    pooled_times
        Optional pre-pooled channel times (fallback when unit_map empty).
    mode
        ``per_unit`` | ``good_only`` | ``all_pool``.
    """
    mode = (mode or DEFAULT_COMPARE_MODE).strip().lower()
    if mode not in COMPARE_MODES:
        mode = DEFAULT_COMPARE_MODE

    lh = np.sort(np.asarray(lh_times, dtype=np.int64).ravel()) if lh_times is not None else np.array([], dtype=np.int64)
    labels = {int(k): _norm_label(v) for k, v in (unit_labels or {}).items()}
    umap: Dict[int, np.ndarray] = {}
    if unit_map:
        for uid, t in unit_map.items():
            arr = np.asarray(t, dtype=np.int64).ravel()
            if arr.size:
                umap[int(uid)] = np.sort(arr)

    coincidence = (
        int(coincidence_samples)
        if coincidence_samples is not None
        else max(1, int(0.001 * float(fs)))
    )

    if lh.size == 0:
        filt_mode = "all_pool" if mode == "per_unit" else mode
        filtered = _filter_unit_map(umap, labels, filt_mode)
        ks = _pool_times(filtered) if filtered else (
            np.sort(np.asarray(pooled_times, dtype=np.int64).ravel())
            if pooled_times is not None and np.asarray(pooled_times).size
            else np.array([], dtype=np.int64)
        )
        # Still expose unit sizes for rejected-channel UI (matched=0)
        empty_stats: List[UnitMatchStats] = []
        for uid in sorted(filtered.keys()):
            n_u = int(filtered[uid].size)
            empty_stats.append(
                UnitMatchStats(
                    unit_id=int(uid),
                    label=_norm_label(labels.get(int(uid))),
                    n_unit=n_u,
                    n_matched=0,
                    n_ks_only=n_u,
                    recall=0.0,
                    precision=0.0,
                )
            )
        empty_stats.sort(key=lambda s: (-s.n_unit, s.unit_id))
        return CompareResult(
            mode=mode,
            n_lh=0,
            n_matched=0,
            n_lh_only=0,
            n_ks_only=int(ks.size),
            n_ks_total=int(ks.size),
            unit_stats=empty_stats,
            note="no LH ground truth",
        )

    # Per-unit stats always computed when we have a unit map (for breakdown UI)
    unit_stats = _per_unit_stats(lh, umap, labels, coincidence) if umap else []
    primary_id, confident = _pick_primary(unit_stats, int(lh.size))

    if mode == "per_unit":
        # Overall LH coverage uses the full non-noise pool (fair "did KS see it?")
        pool = _pool_times(umap)
        if pool.size == 0 and pooled_times is not None:
            pool = np.sort(np.asarray(pooled_times, dtype=np.int64).ravel())
        n_m, n_lh_only, n_ks_only, _ = match_spikes(lh, pool, coincidence)

        note = ""
        if confident and primary_id is not None:
            prim = next(s for s in unit_stats if s.unit_id == primary_id)
            note = (
                f"primary U{primary_id} ({prim.label}): "
                f"{prim.n_matched}/{lh.size} LH ({prim.recall:.0%})"
            )
        elif unit_stats:
            n_active = sum(1 for s in unit_stats if s.n_matched > 0)
            note = f"{n_active} KS unit(s) share LH; no confident primary"
        else:
            note = "no per-unit map — pooled only"

        return CompareResult(
            mode=mode,
            n_lh=int(lh.size),
            n_matched=int(n_m),
            n_lh_only=int(n_lh_only),
            n_ks_only=int(n_ks_only),
            n_ks_total=int(pool.size),
            unit_stats=unit_stats,
            primary_unit_id=primary_id,
            confident=confident,
            note=note,
        )

    # Pool modes: good_only or all_pool
    filtered = _filter_unit_map(umap, labels, mode)
    if filtered:
        pool = _pool_times(filtered)
        # Restrict unit_stats display to units in the filter
        allowed = set(filtered.keys())
        unit_stats = [s for s in unit_stats if s.unit_id in allowed]
        primary_id, confident = _pick_primary(unit_stats, int(lh.size))
    elif mode == "all_pool" and pooled_times is not None and np.asarray(pooled_times).size:
        pool = np.sort(np.asarray(pooled_times, dtype=np.int64).ravel())
        unit_stats = []
        primary_id, confident = None, False
    else:
        pool = np.array([], dtype=np.int64)

    n_m, n_lh_only, n_ks_only, _ = match_spikes(lh, pool, coincidence)
    if mode == "good_only" and not filtered and umap:
        note = "no 'good' units on this channel (all mua/unsorted?)"
    elif mode == "good_only":
        n_good = len(filtered)
        note = f"pooled {n_good} good unit(s)"
    else:
        note = f"pooled {len(filtered) if filtered else 0} unit(s)"

    return CompareResult(
        mode=mode,
        n_lh=int(lh.size),
        n_matched=int(n_m),
        n_lh_only=int(n_lh_only),
        n_ks_only=int(n_ks_only),
        n_ks_total=int(pool.size),
        unit_stats=unit_stats,
        primary_unit_id=primary_id,
        confident=confident,
        note=note,
    )
