"""Feature extraction (PyTorch FP32/FP16 & TensorRT FP16) and chunked Re-ID metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .common import TensorRT10Runner
except ImportError:
    from common import TensorRT10Runner


@torch.no_grad()
def extract_features_pytorch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    precision: str = "fp32",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract concatenated `(bn_global, bn_fs)` features and L2-normalize in float32."""
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"Unsupported precision {precision!r} (expected 'fp32' or 'fp16')")
    model.eval()
    use_autocast = (device.type == "cuda") and (precision == "fp16")

    feats_list: List[torch.Tensor] = []
    pids_list: List[torch.Tensor] = []
    cams_list: List[torch.Tensor] = []
    paths_list: List[str] = []

    for imgs, pids, cams, paths in dataloader:
        imgs = imgs.to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                emb = torch.cat(list(outputs), dim=1)
            else:
                emb = outputs
        feats_list.append(emb.float().cpu())
        pids_list.append(pids.cpu())
        cams_list.append(cams.cpu())
        paths_list.extend(paths)

    feats = F.normalize(torch.cat(feats_list, dim=0), p=2, dim=1)
    return feats, torch.cat(pids_list, dim=0), torch.cat(cams_list, dim=0), paths_list


@torch.no_grad()
def extract_features_trt(
    runner: TensorRT10Runner,
    dataloader: DataLoader,
    max_batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract L2-normalized embeddings using a TensorRT FP16 engine."""
    feats_list: List[torch.Tensor] = []
    pids_list: List[torch.Tensor] = []
    cams_list: List[torch.Tensor] = []
    paths_list: List[str] = []

    for imgs, pids, cams, paths in dataloader:
        bs = imgs.shape[0]
        for start in range(0, bs, max_batch_size):
            sub = imgs[start : start + max_batch_size].to(device=runner.device, non_blocking=True)
            out = runner.infer(sub)
            runner.synchronize()
            feats_list.append(out.float().cpu().clone())
        pids_list.append(pids.cpu())
        cams_list.append(cams.cpu())
        paths_list.extend(paths)

    feats = F.normalize(torch.cat(feats_list, dim=0), p=2, dim=1)
    return feats, torch.cat(pids_list, dim=0), torch.cat(cams_list, dim=0), paths_list


@torch.no_grad()
def evaluate_retrieval_chunked(
    q_feats: torch.Tensor,
    q_pids: torch.Tensor,
    g_feats: torch.Tensor,
    g_pids: torch.Tensor,
    q_camids: Optional[torch.Tensor] = None,
    g_camids: Optional[torch.Tensor] = None,
    q_chunk_size: int = 1024,
    device: Optional[torch.device] = None,
    return_rank1_gallery_idx: bool = False,
) -> Dict[str, object]:
    """Compute exact CMC (R1/R5/R10), mAP, and mINP in percentage [0, 100].

    When `q_camids` and `g_camids` are provided, gallery entries with the same
    `vehicle_id` AND same `camera_id` are treated as junk and masked out before
    ranking. Queries left with zero positive gallery matches are excluded and
    tracked in `valid_queries_after_filter`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = F.normalize(q_feats.to(device=device, dtype=torch.float32), p=2, dim=1)
    g = F.normalize(g_feats.to(device=device, dtype=torch.float32), p=2, dim=1)
    q_p = q_pids.to(device=device, dtype=torch.long)
    g_p = g_pids.to(device=device, dtype=torch.long)

    use_cams = (
        q_camids is not None
        and g_camids is not None
        and bool((q_camids >= 0).any())
        and bool((g_camids >= 0).any())
    )
    q_c = q_camids.to(device=device, dtype=torch.long) if use_cams else None
    g_c = g_camids.to(device=device, dtype=torch.long) if use_cams else None

    num_q = q.shape[0]
    num_g = g.shape[0]
    g_t = g.t().contiguous()
    max_rank = min(10, num_g)

    cmc_sum = torch.zeros(10, dtype=torch.float64, device=device)
    ap_sum = 0.0
    inp_sum = 0.0
    valid_q = 0
    rank1_gal_list: Optional[List[torch.Tensor]] = [] if return_rank1_gallery_idx else None

    for start in range(0, num_q, q_chunk_size):
        end = min(start + q_chunk_size, num_q)
        sim = q[start:end] @ g_t  # [B, G]
        qp = q_p[start:end].unsqueeze(1)
        pos_mask = qp == g_p.unsqueeze(0)

        if use_cams and q_c is not None and g_c is not None:
            qc = q_c[start:end].unsqueeze(1)
            junk_mask = pos_mask & (qc == g_c.unsqueeze(0))
            sim = sim.masked_fill(junk_mask, -1e9)
            pos_mask = pos_mask & (~junk_mask)
            # Keep valid counts per row (number of non-junk gallery items)
            non_junk_counts = (~junk_mask).sum(dim=1)
        else:
            non_junk_counts = torch.full((end - start,), num_g, dtype=torch.long, device=device)

        order = torch.argsort(sim, dim=1, descending=True)
        ordered_matches = torch.gather(pos_mask.long(), 1, order)
        total_pos = ordered_matches.sum(dim=1)
        has_pos = total_pos > 0
        valid_count_chunk = int(has_pos.sum().item())

        if rank1_gal_list is not None:
            top1 = order[:, 0].clone()
            top1[~has_pos] = -1
            rank1_gal_list.append(top1.cpu())

        if valid_count_chunk == 0:
            continue

        valid_q += valid_count_chunk
        m = ordered_matches[has_pos]  # [V, G]
        tp = total_pos[has_pos].to(torch.float64)
        nj = non_junk_counts[has_pos]

        # Mask out any trailing junk items pushed to the end (sim = -1e9)
        col_idx = torch.arange(num_g, device=device).unsqueeze(0)
        valid_col = col_idx < nj.unsqueeze(1)
        m_clean = m * valid_col.long()

        # CMC at k in [1..10]
        cum_m = torch.cumsum(m_clean, dim=1)
        for k in range(1, max_rank + 1):
            cmc_sum[k - 1] += (cum_m[:, k - 1] > 0).to(torch.float64).sum()
        if max_rank < 10 and max_rank > 0:
            hit_max = (cum_m[:, max_rank - 1] > 0).to(torch.float64).sum()
            for k in range(max_rank + 1, 11):
                cmc_sum[k - 1] += hit_max

        # AP
        ranks = (col_idx + 1).to(torch.float64)
        precisions = cum_m.to(torch.float64) / ranks
        ap = (precisions * m_clean.to(torch.float64)).sum(dim=1) / tp
        ap_sum += float(ap.sum().item())

        # mINP: total_positives / (1-based rank of last positive match)
        pos_positions = torch.where(m_clean > 0, col_idx + 1, torch.zeros_like(col_idx))
        max_pos_rank = pos_positions.max(dim=1).values.to(torch.float64)
        inp = tp / max_pos_rank
        inp_sum += float(inp.sum().item())

    if valid_q == 0:
        out: Dict[str, object] = {
            "total_queries_before_filter": int(num_q),
            "valid_queries_after_filter": 0,
            "filtered_out_queries": int(num_q),
            "gallery_size": int(num_g),
            "mAP": 0.0,
            "R1": 0.0,
            "R5": 0.0,
            "R10": 0.0,
            "mINP": 0.0,
        }
    else:
        cmc_pct = (cmc_sum / float(valid_q)) * 100.0
        out = {
            "total_queries_before_filter": int(num_q),
            "valid_queries_after_filter": int(valid_q),
            "filtered_out_queries": int(num_q - valid_q),
            "gallery_size": int(num_g),
            "mAP": (ap_sum / float(valid_q)) * 100.0,
            "R1": float(cmc_pct[0].item()),
            "R5": float(cmc_pct[4].item()),
            "R10": float(cmc_pct[9].item()),
            "mINP": (inp_sum / float(valid_q)) * 100.0,
        }

    if rank1_gal_list is not None:
        out["rank1_gallery_idx"] = torch.cat(rank1_gal_list, dim=0).numpy()
    return out
