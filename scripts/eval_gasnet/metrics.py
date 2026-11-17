"""Re-Identification Accuracy Metrics.

Implements standard scientific Re-ID metrics:
- CMC: Cumulative Matching Characteristics (Rank-1, Rank-5, Rank-10)
- mAP: mean Average Precision
- mINP: mean Inverse Negative Penalty
Includes protocol handling for cross-camera junk discarding (e.g. VeRi-776)
and single-gallery-per-ID random sampling (e.g. VRU / VehicleID).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def compute_distance_matrix(
    q_feats: Union[torch.Tensor, np.ndarray],
    g_feats: Union[torch.Tensor, np.ndarray],
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise distance matrix between query and gallery features."""
    if isinstance(q_feats, np.ndarray):
        q_feats = torch.from_numpy(q_feats)
    if isinstance(g_feats, np.ndarray):
        g_feats = torch.from_numpy(g_feats)

    q = F.normalize(q_feats.float(), p=2, dim=1)
    g = F.normalize(g_feats.float(), p=2, dim=1)

    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        dist = 1.0 - torch.mm(q, g.t())
    elif metric == "euclidean":
        dist = torch.cdist(q, g, p=2.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist.cpu().numpy()


def evaluate_rank_map_minp(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: Optional[np.ndarray] = None,
    g_camids: Optional[np.ndarray] = None,
    max_rank: int = 50,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Compute CMC, mAP, and mINP.

    Args:
        distmat: 2D array of shape [num_query, num_gallery].
        q_pids: 1D array of query vehicle IDs.
        g_pids: 1D array of gallery vehicle IDs.
        q_camids: Optional 1D array of query camera IDs.
        g_camids: Optional 1D array of gallery camera IDs.
        max_rank: Maximum rank for CMC curve.

    Returns:
        cmc: 1D array of CMC values up to max_rank.
        mAP: mean Average Precision (0.0 to 1.0).
        mINP: mean Inverse Negative Penalty (0.0 to 1.0).
        indices: 2D array of sorted gallery indices for each query.
    """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    if q_camids is not None:
        q_camids = np.asarray(q_camids)
    if g_camids is not None:
        g_camids = np.asarray(g_camids)

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)

    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        order = indices[q_idx]

        # Extract ordered gallery pids
        ordered_g_pids = g_pids[order]

        # Filter out junk images if camera ids are provided
        if q_camids is not None and g_camids is not None:
            q_cam = q_camids[q_idx]
            ordered_g_cams = g_camids[order]
            # Junk: same vehicle ID and same camera ID
            junk_mask = (ordered_g_pids == q_pid) & (ordered_g_cams == q_cam)
            keep_mask = ~junk_mask
            ordered_g_pids = ordered_g_pids[keep_mask]

        # Binary match vector: 1 if positive match, 0 otherwise
        matches = (ordered_g_pids == q_pid).astype(np.int32)
        total_positives = np.sum(matches)

        if total_positives == 0:
            continue

        num_valid_q += 1

        # 1. CMC
        raw_cmc = matches.cumsum()
        raw_cmc[raw_cmc > 1] = 1
        cmc_k = np.zeros(max_rank, dtype=np.float32)
        valid_len = min(len(raw_cmc), max_rank)
        cmc_k[:valid_len] = raw_cmc[:valid_len]
        if valid_len < max_rank and valid_len > 0:
            cmc_k[valid_len:] = raw_cmc[valid_len - 1]
        all_cmc.append(cmc_k)

        # 2. AP (Average Precision)
        pos_ranks = np.where(matches == 1)[0]
        precisions = (np.arange(1, total_positives + 1)) / (pos_ranks + 1.0)
        ap = np.sum(precisions) / total_positives
        all_ap.append(ap)

        # 3. INP (Inverse Negative Penalty)
        # Furthest positive match index (0-based) -> rank is max_pos_rank + 1
        max_pos_rank = pos_ranks[-1]
        inp = total_positives / (max_pos_rank + 1.0)
        all_inp.append(inp)

    if num_valid_q == 0:
        return np.zeros(max_rank, dtype=np.float32), 0.0, 0.0, indices

    cmc = np.mean(np.array(all_cmc), axis=0)
    mAP = float(np.mean(all_ap))
    mINP = float(np.mean(all_inp))

    return cmc, mAP, mINP, indices


def format_metrics_dict(cmc: np.ndarray, mAP: float, mINP: float) -> Dict[str, float]:
    """Format metrics into a clean dictionary with percentage strings."""
    return {
        "mAP": mAP * 100.0,
        "Rank-1": float(cmc[0]) * 100.0 if len(cmc) > 0 else 0.0,
        "Rank-5": float(cmc[4]) * 100.0 if len(cmc) > 4 else 0.0,
        "Rank-10": float(cmc[9]) * 100.0 if len(cmc) > 9 else 0.0,
        "mINP": mINP * 100.0,
    }
