"""Evaluation metrics for VRU/VRAI and UAV-Anti-UAV ReID."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_map_cmc(
    query_feats: torch.Tensor,
    query_ids: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_ids: torch.Tensor,
    topk: Tuple[int, int] = (1, 5),
    q_chunk_size: int = 2048,
    use_fp16_sim: bool = True,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """Chunked GPU-accelerated mAP and CMC (Rank-1, Rank-5) for large galleries (VRU/VRAI)."""
    if q_chunk_size < 1:
        raise ValueError("q_chunk_size must be >= 1")

    if query_feats.is_cuda:
        device = query_feats.device
    elif gallery_feats.is_cuda:
        device = gallery_feats.device
    else:
        device = query_feats.device

    q = F.normalize(query_feats.to(device=device, dtype=torch.float32, non_blocking=True), dim=1)
    g = F.normalize(gallery_feats.to(device=device, dtype=torch.float32, non_blocking=True), dim=1)
    query_ids = query_ids.to(device=device, non_blocking=True)
    gallery_ids = gallery_ids.to(device=device, non_blocking=True)

    num_q = q.size(0)
    num_g = g.size(0)
    max_k = max(topk)
    cmc = torch.zeros(max_k, device=device, dtype=torch.float64)
    ap_sum = 0.0
    ap_count = 0

    sim_dtype = None
    if device.type == "cuda" and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    needs_float_sim = sim_dtype is not None
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    positions = torch.arange(1, num_g + 1, device=device, dtype=torch.float32).unsqueeze(0)

    for start in range(0, num_q, q_chunk_size):
        end = min(start + q_chunk_size, num_q)
        sim = q_mm[start:end] @ g_mm.t()
        if needs_float_sim:
            sim = sim.float()

        order = torch.argsort(sim, dim=1, descending=True)
        del sim

        matches = (gallery_ids[order] == query_ids[start:end].unsqueeze(1)).int()
        has_match = matches.any(dim=1)
        if has_match.any():
            first_match = torch.argmax(matches, dim=1)[has_match]
            first_match_topk = first_match[first_match < max_k]
            if first_match_topk.numel() > 0:
                counts = torch.bincount(first_match_topk, minlength=max_k).to(cmc.dtype)
                cmc += torch.cumsum(counts, dim=0)

            valid_matches = matches[has_match].to(torch.float32)
            precision = torch.cumsum(valid_matches, dim=1) / positions
            ap = (precision * valid_matches).sum(dim=1) / valid_matches.sum(dim=1)
            ap_sum += ap.sum().item()
            ap_count += ap.numel()

        if verbose:
            print(f"[eval] processed queries: {end}/{num_q}")

    cmc = cmc / num_q
    mAP = (ap_sum / ap_count) if ap_count > 0 else 0.0
    r1 = float(cmc[topk[0] - 1].item())
    r5 = float(cmc[topk[1] - 1].item()) if topk[1] <= max_k else 0.0
    return mAP, r1, r5


def eval_uav_map_cmc(
    qf: torch.Tensor,
    gf: torch.Tensor,
    q_pids,
    g_pids,
    q_seq_ids=None,
    g_seq_ids=None,
    intra_sequence: bool = False,
):
    """Standard UAV-Anti-UAV evaluation (CMC curve up to 50, mAP, mINP, indices, matches)."""
    qf = F.normalize(qf.float(), p=2, dim=1)
    gf = F.normalize(gf.float(), p=2, dim=1)

    distmat = 1.0 - torch.mm(qf, gf.t())
    distmat = distmat.cpu().numpy()

    if intra_sequence and q_seq_ids is not None and g_seq_ids is not None:
        q_seqs = np.asarray(q_seq_ids)
        g_seqs = np.asarray(g_seq_ids)
        mask = q_seqs[:, np.newaxis] != g_seqs[np.newaxis, :]
        distmat[mask] = np.inf

    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)

    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.copy()
        num_valid_q += 1.0

        pos_indices = np.where(orig_cmc == 1)[0]
        max_pos_idx = np.max(pos_indices)
        inp = orig_cmc.sum() / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc = np.cumsum(cmc)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:50])

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        return np.zeros(50, dtype=np.float32), 0.0, 0.0, indices, matches

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    mINP = float(np.mean(all_INP))

    return all_cmc, mAP, mINP, indices, matches


def compute_tar_at_far(qf: torch.Tensor, gf: torch.Tensor, q_pids, g_pids, far_target: float = 0.001, threshold: float | None = None):
    """Compute TAR@FAR=far_target using pairwise cosine similarity (excluding i==j self-match)."""
    qf_n = F.normalize(qf.float(), p=2, dim=1).cpu().numpy()
    gf_n = F.normalize(gf.float(), p=2, dim=1).cpu().numpy()
    sim = np.dot(qf_n, gf_n.T)

    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)

    same = g_pids[None, :] == q_pids[:, None]
    valid = np.ones_like(same, dtype=bool)
    if sim.shape[0] == sim.shape[1]:
        np.fill_diagonal(valid, False)

    genuine_scores = sim[same & valid].astype(np.float32)
    impostor_scores = sim[(~same) & valid].astype(np.float32)

    if len(impostor_scores) == 0 or len(genuine_scores) == 0:
        return float("nan"), float("nan"), float("nan")

    t_star = (
        float(threshold)
        if threshold is not None
        else float(np.quantile(impostor_scores, 1.0 - far_target))
    )

    tar = float(np.mean(genuine_scores >= t_star))
    far = float(np.mean(impostor_scores >= t_star))
    return tar, far, t_star
