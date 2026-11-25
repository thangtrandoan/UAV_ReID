#!/usr/bin/env python3
"""Generate Chapter 5 visualizations (300 DPI PNG + vector PDF) with real hook attention checks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    load_eval_config,
    load_gasnet_checkpoint,
    reset_gpu_memory,
    resolve_checkpoint_path,
    resolve_efficiency_checkpoint,
    resolve_output_dir,
    write_csv,
    write_json,
)
from datasets import (  # noqa: E402
    ReIDSample,
    build_eval_transform,
    load_veri776,
    load_vrai_val,
    load_vru_split,
    make_eval_loader,
)
from metrics import evaluate_retrieval_chunked, extract_features_pytorch  # noqa: E402


def _save_fig_dual(fig: plt.Figure, stem_path: Path, dpi: int = 300) -> Tuple[Path, Path]:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem_path.with_suffix(".png")
    pdf_path = stem_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved visual: {png_path} & {pdf_path}")
    return png_path, pdf_path


def _resolve_model_ckpt_with_fallback(
    cfg: Dict[str, Any], domain: str, model_key: str
) -> Tuple[Path, str, bool]:
    domain_ckpt = resolve_checkpoint_path(cfg, domain, model_key)
    if domain_ckpt.is_file():
        return domain_ckpt, domain, False
    fb_domain, fb_ckpt = resolve_efficiency_checkpoint(cfg, model_key)
    print(
        f"[WARNING] {domain}/{model_key} checkpoint missing; using {fb_domain}/{model_key} "
        f"({fb_ckpt.name}) for visualization."
    )
    return fb_ckpt, fb_domain, True


@torch.no_grad()
def _capture_real_attention(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    gate_stage: str,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model.eval()
    model.set_attention_capture(True)
    try:
        x = img_tensor.unsqueeze(0).to(device)
        _ = model(x)
        gate_mod = getattr(model, gate_stage, None)
        assert gate_mod is not None, f"Missing gate module {gate_stage!r} on model"
        attn = getattr(gate_mod, "last_attention", None)
        assert attn is not None, (
            f"Attention hook returned None for {gate_stage} on {type(model).__name__}. "
            "Fake Gaussian fallback is strictly forbidden."
        )
        assert torch.isfinite(attn).all(), f"Non-finite attention tensor captured on {gate_stage}"
        attn_np = attn.squeeze().detach().float().cpu().numpy()
        if attn_np.ndim > 2:
            attn_np = attn_np.mean(axis=0)
        stats = {
            "min": float(np.min(attn_np)),
            "max": float(np.max(attn_np)),
            "mean": float(np.mean(attn_np)),
            "std": float(np.std(attn_np)),
            "shape": list(attn_np.shape),
            "hook_verified": True,
        }
        return attn_np, stats
    finally:
        model.set_attention_capture(False)


def _overlay_heatmap(pil_img: Image.Image, attn_2d: np.ndarray, alpha: float = 0.52) -> Image.Image:
    w, h = pil_img.size
    lo, hi = float(np.min(attn_2d)), float(np.max(attn_2d))
    norm_map = (attn_2d - lo) / (hi - lo + 1e-8)
    cmap = plt.get_cmap("jet")
    rgba = (cmap(norm_map)[:, :, :3] * 255).astype(np.uint8)
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    hm_pil = Image.fromarray(rgba).resize((w, h), bilinear)
    return Image.blend(pil_img.convert("RGBA"), hm_pil.convert("RGBA"), alpha=alpha).convert("RGB")


def _load_domain_q_g(cfg: Dict[str, Any], domain: str, tfm):
    ds_cfg = cfg.get("datasets", {})
    if domain == "veri":
        return load_veri776(ds_cfg["veri"]["dir"], tfm)
    if domain == "vrai":
        vcfg = ds_cfg["vrai"]
        return load_vrai_val(
            vcfg["dir"], num_ids=int(vcfg.get("num_val_ids", 1000)), seed=int(vcfg.get("seed", 42)), transform=tfm
        )
    if domain == "vru":
        vcfg = ds_cfg["vru"]
        return load_vru_split(vcfg["dir"], "Small", seed=int(vcfg.get("seed", 42)), transform=tfm)
    raise ValueError(f"Unsupported visualization domain: {domain!r}")


def _topk_for_query(
    sim_row: np.ndarray,
    q_sample: ReIDSample,
    g_samples: List[ReIDSample],
    use_cams: bool,
    k: int = 5,
) -> List[Tuple[ReIDSample, float, bool]]:
    order = np.argsort(-sim_row)
    items: List[Tuple[ReIDSample, float, bool]] = []
    for g_idx in order:
        gs = g_samples[int(g_idx)]
        if use_cams and gs.vehicle_id == q_sample.vehicle_id and gs.camera_id == q_sample.camera_id:
            continue
        is_match = gs.vehicle_id == q_sample.vehicle_id
        items.append((gs, float(sim_row[int(g_idx)]), is_match))
        if len(items) >= k:
            break
    return items


def _draw_retrieval_comparison(
    q_sample: ReIDSample,
    top_res: List[Tuple[ReIDSample, float, bool]],
    top_bcgg: List[Tuple[ReIDSample, float, bool]],
    title: str,
    stem_path: Path,
    dpi: int,
) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 5.8))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)

    q_img = Image.open(q_sample.img_path).convert("RGB").resize((192, 192))
    q_bordered = ImageOps.expand(q_img, border=6, fill="#1E90FF")
    q_cam_str = f" | Cam {q_sample.camera_id}" if q_sample.camera_id >= 0 else ""

    row_specs = [
        ("ResNet50-IBN + RGA", top_res, "#8B0000"),
        ("ConvNeXt + BCGG", top_bcgg, "#006400"),
    ]
    for r_idx, (row_label, top_items, label_color) in enumerate(row_specs):
        ax_q = axes[r_idx, 0]
        ax_q.imshow(q_bordered)
        ax_q.set_title(f"Query (ID:{q_sample.vehicle_id}{q_cam_str})\n[{row_label}]", fontsize=9, fontweight="bold", color=label_color)
        ax_q.axis("off")

        for c_idx in range(5):
            ax = axes[r_idx, c_idx + 1]
            if c_idx < len(top_items):
                gs, sim, is_match = top_items[c_idx]
                g_img = Image.open(gs.img_path).convert("RGB").resize((192, 192))
                color = "#28A745" if is_match else "#DC3545"
                status = "MATCH" if is_match else "MISS"
                ax.imshow(ImageOps.expand(g_img, border=6, fill=color))
                g_cam = f" | C{gs.camera_id}" if gs.camera_id >= 0 else ""
                ax.set_title(
                    f"Rank-{c_idx + 1}: {status}\nID:{gs.vehicle_id}{g_cam} | {sim:.3f}",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_fig_dual(fig, stem_path, dpi=dpi)


def generate_retrieval_examples(
    cfg: Dict[str, Any],
    out_dir: Path,
    tfm,
    device: torch.device,
    dpi: int,
) -> List[Dict[str, Any]]:
    ret_dir = out_dir / "retrieval_examples"
    ret_dir.mkdir(parents=True, exist_ok=True)
    ret_cfg = cfg.get("visualizations", {}).get("retrieval_domains", {"veri": 2, "vrai": 2, "vru": 1})
    batch_size = int(cfg.get("eval", {}).get("batch_size", 64))
    num_workers = int(cfg.get("eval", {}).get("num_workers", 4))

    selected_meta: List[Dict[str, Any]] = []
    failure_pool: List[Tuple[str, ReIDSample, List[Any], List[Any], bool, str]] = []

    for domain, count in ret_cfg.items():
        if int(count) <= 0:
            continue
        try:
            q_ds, g_ds, ds_meta = _load_domain_q_g(cfg, domain, tfm)
        except FileNotFoundError as exc:
            print(f"[WARNING] Skipping retrieval visual for {domain}: {exc}")
            continue

        res_ckpt, res_src, res_fb = _resolve_model_ckpt_with_fallback(cfg, domain, "resnet50ibn_rga")
        bcgg_ckpt, bcgg_src, bcgg_fb = _resolve_model_ckpt_with_fallback(cfg, domain, "convnext_bcgg")

        reset_gpu_memory(device)
        m_res, _ = load_gasnet_checkpoint(res_ckpt, expected_model_key="resnet50ibn_rga", deploy=False, device=device)
        q_loader = make_eval_loader(q_ds, batch_size=batch_size, num_workers=num_workers)
        g_loader = make_eval_loader(g_ds, batch_size=batch_size, num_workers=num_workers)
        qf_res, qp, qc, _ = extract_features_pytorch(m_res, q_loader, device=device, precision="fp32")
        gf_res, gp, gc, _ = extract_features_pytorch(m_res, g_loader, device=device, precision="fp32")
        del m_res
        reset_gpu_memory(device)

        m_bcgg, _ = load_gasnet_checkpoint(bcgg_ckpt, expected_model_key="convnext_bcgg", deploy=True, device=device)
        qf_bcgg, _, _, _ = extract_features_pytorch(m_bcgg, q_loader, device=device, precision="fp32")
        gf_bcgg, _, _, _ = extract_features_pytorch(m_bcgg, g_loader, device=device, precision="fp32")
        del m_bcgg
        reset_gpu_memory(device)

        use_cams = bool((qc >= 0).any() and (gc >= 0).any())
        sim_res = (qf_res @ gf_res.t()).numpy()
        sim_bcgg = (qf_bcgg @ gf_bcgg.t()).numpy()

        # Find (1) BCGG fixes ResNet Rank-1 error, (2) both correct fallback, (3) both fail pool
        bcgg_wins: List[int] = []
        both_ok: List[int] = []
        both_fail: List[int] = []
        seen_pids = set()

        for q_idx in range(len(q_ds)):
            qs = q_ds.samples[q_idx]
            if qs.vehicle_id in seen_pids:
                continue
            top_r = _topk_for_query(sim_res[q_idx], qs, g_ds.samples, use_cams, k=5)
            top_b = _topk_for_query(sim_bcgg[q_idx], qs, g_ds.samples, use_cams, k=5)
            if not top_r or not top_b:
                continue
            r1_r = top_r[0][2]
            r1_b = top_b[0][2]
            if (not r1_r) and r1_b:
                bcgg_wins.append(q_idx)
                seen_pids.add(qs.vehicle_id)
            elif r1_r and r1_b and len(both_ok) < int(count) + 3:
                both_ok.append(q_idx)
                seen_pids.add(qs.vehicle_id)
            elif (not r1_r) and (not r1_b) and len(both_fail) < 5:
                both_fail.append(q_idx)
                seen_pids.add(qs.vehicle_id)
            if len(bcgg_wins) >= int(count) + 1 and len(both_fail) >= 2:
                break

        chosen = (bcgg_wins + both_ok)[: int(count)]
        ck_note = f" | ckpt={res_src}/{bcgg_src}" if (res_fb or bcgg_fb) else ""

        for idx_num, q_idx in enumerate(chosen, start=1):
            qs = q_ds.samples[q_idx]
            top_r = _topk_for_query(sim_res[q_idx], qs, g_ds.samples, use_cams, k=5)
            top_b = _topk_for_query(sim_bcgg[q_idx], qs, g_ds.samples, use_cams, k=5)
            case_type = "bcgg_corrects_resnet_error" if (not top_r[0][2] and top_b[0][2]) else "comparison"
            title = f"{ds_meta['dataset']} ({case_type}) — Query ID {qs.vehicle_id}{ck_note}"
            stem = ret_dir / f"retrieval_{domain}_{idx_num}_id_{qs.vehicle_id}"
            png_p, pdf_p = _draw_retrieval_comparison(qs, top_r, top_b, title, stem, dpi)
            selected_meta.append(
                {
                    "domain": domain,
                    "dataset": ds_meta["dataset"],
                    "case_type": case_type,
                    "query_image": str(qs.img_path),
                    "query_id": qs.vehicle_id,
                    "resnet_rank1_match": bool(top_r[0][2]),
                    "bcgg_rank1_match": bool(top_b[0][2]),
                    "cross_domain_checkpoint": bool(res_fb or bcgg_fb),
                    "png_path": str(png_p),
                    "pdf_path": str(pdf_p),
                }
            )

        if both_fail:
            qf_idx = both_fail[0]
            qs = q_ds.samples[qf_idx]
            top_r = _topk_for_query(sim_res[qf_idx], qs, g_ds.samples, use_cams, k=5)
            top_b = _topk_for_query(sim_bcgg[qf_idx], qs, g_ds.samples, use_cams, k=5)
            failure_pool.append((domain, qs, top_r, top_b, bool(res_fb or bcgg_fb), ck_note))

    if failure_pool:
        domain, qs, top_r, top_b, is_fb, ck_note = failure_pool[0]
        title = f"{domain.upper()} Failure Case (Both Miss Rank-1) — Query ID {qs.vehicle_id}{ck_note}"
        stem = ret_dir / f"retrieval_failure_case_{domain}_id_{qs.vehicle_id}"
        png_p, pdf_p = _draw_retrieval_comparison(qs, top_r, top_b, title, stem, dpi)
        selected_meta.append(
            {
                "domain": domain,
                "dataset": domain.upper(),
                "case_type": "failure_case_both_miss_rank1",
                "query_image": str(qs.img_path),
                "query_id": qs.vehicle_id,
                "resnet_rank1_match": False,
                "bcgg_rank1_match": False,
                "cross_domain_checkpoint": is_fb,
                "png_path": str(png_p),
                "pdf_path": str(pdf_p),
            }
        )

    write_csv(selected_meta, ret_dir / "retrieval_cases.csv")
    write_json(selected_meta, ret_dir / "retrieval_cases.json")
    return selected_meta


def generate_attention_heatmaps(
    cfg: Dict[str, Any],
    out_dir: Path,
    tfm,
    device: torch.device,
    dpi: int,
) -> List[Dict[str, Any]]:
    hm_dir = out_dir / "attention_heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)
    hm_cfg = cfg.get("visualizations", {}).get("heatmap_domains", {"veri": 2, "vrai": 2})
    gate_stage = str(cfg.get("visualizations", {}).get("gate_stage", "ga3"))
    img_size = int(cfg.get("preprocess", {}).get("img_size", 224))

    stats_rows: List[Dict[str, Any]] = []

    for domain, count in hm_cfg.items():
        if int(count) <= 0:
            continue
        try:
            q_ds, _, ds_meta = _load_domain_q_g(cfg, domain, tfm)
        except FileNotFoundError as exc:
            print(f"[WARNING] Skipping heatmaps for {domain}: {exc}")
            continue

        # Pick distinct vehicle IDs
        chosen_samples: List[ReIDSample] = []
        seen_ids = set()
        for s in q_ds.samples:
            if s.vehicle_id not in seen_ids:
                chosen_samples.append(s)
                seen_ids.add(s.vehicle_id)
            if len(chosen_samples) >= int(count):
                break

        res_ckpt, res_src, res_fb = _resolve_model_ckpt_with_fallback(cfg, domain, "resnet50ibn_rga")
        crga_ckpt, crga_src, crga_fb = _resolve_model_ckpt_with_fallback(cfg, domain, "convnext_rga")
        cbcgg_ckpt, cbcgg_src, cbcgg_fb = _resolve_model_ckpt_with_fallback(cfg, domain, "convnext_bcgg")

        reset_gpu_memory(device)
        m_res, i_res = load_gasnet_checkpoint(res_ckpt, expected_model_key="resnet50ibn_rga", deploy=False, device=device)
        m_crga, i_crga = load_gasnet_checkpoint(crga_ckpt, expected_model_key="convnext_rga", deploy=False, device=device)
        m_cbcgg, i_cbcgg = load_gasnet_checkpoint(cbcgg_ckpt, expected_model_key="convnext_bcgg", deploy=False, device=device)

        try:
            for s_idx, sample in enumerate(chosen_samples, start=1):
                with Image.open(sample.img_path) as raw:
                    rgb_img = raw.convert("RGB")
                bicubic = getattr(Image, "Resampling", Image).BICUBIC
                pil_img = rgb_img.resize((img_size, img_size), bicubic)
                img_tensor = tfm(rgb_img)

                attn_res, st_res = _capture_real_attention(m_res, img_tensor, gate_stage, device)
                attn_crga, st_crga = _capture_real_attention(m_crga, img_tensor, gate_stage, device)
                attn_cbcgg, st_cbcgg = _capture_real_attention(m_cbcgg, img_tensor, gate_stage, device)

                ov_res = _overlay_heatmap(pil_img, attn_res)
                ov_crga = _overlay_heatmap(pil_img, attn_crga)
                ov_cbcgg = _overlay_heatmap(pil_img, attn_cbcgg)

                fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0))
                cols = [
                    (pil_img, f"Ảnh gốc ({ds_meta['dataset']})\nID: {sample.vehicle_id}", "#222222"),
                    (ov_res, f"ResNet50-IBN + RGA ({gate_stage})\nckpt={res_src}", "#8B0000"),
                    (ov_crga, f"ConvNeXt + RGA ({gate_stage})\nckpt={crga_src}", "#B8860B"),
                    (ov_cbcgg, f"ConvNeXt + BCGG ({gate_stage})\nckpt={cbcgg_src}", "#006400"),
                ]
                for ax, (im, lbl, color) in zip(axes, cols):
                    ax.imshow(im)
                    ax.set_title(lbl, fontsize=10, fontweight="bold", color=color)
                    ax.axis("off")

                plt.tight_layout()
                stem = hm_dir / f"heatmap_{domain}_{s_idx}_id_{sample.vehicle_id}"
                _save_fig_dual(fig, stem, dpi=dpi)

                for m_key, st, ck_info, is_fb in (
                    ("resnet50ibn_rga", st_res, i_res, res_fb),
                    ("convnext_rga", st_crga, i_crga, crga_fb),
                    ("convnext_bcgg", st_cbcgg, i_cbcgg, cbcgg_fb),
                ):
                    stats_rows.append(
                        {
                            "domain": domain,
                            "dataset": ds_meta["dataset"],
                            "image_path": str(sample.img_path),
                            "vehicle_id": sample.vehicle_id,
                            "gate_stage": gate_stage,
                            "model": m_key,
                            "checkpoint_sha256": ck_info["sha256"],
                            "cross_domain_checkpoint": is_fb,
                            **st,
                        }
                    )
        finally:
            del m_res, m_crga, m_cbcgg
            reset_gpu_memory(device)

    write_csv(stats_rows, hm_dir / "heatmap_stats.csv")
    write_json({"hook_verified": all(r["hook_verified"] for r in stats_rows), "samples": stats_rows}, hm_dir / "heatmap_stats.json")
    return stats_rows


def generate_metric_charts(out_root: Path, out_dir: Path, dpi: int) -> None:
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    acc_csv = out_root / "01_accuracy_benchmarks" / "accuracy_summary.csv"
    pt_csv = out_root / "03_efficiency_and_latency_orin" / "latency_pytorch_summary.csv"
    trt_csv = out_root / "03_efficiency_and_latency_orin" / "latency_tensorrt_summary.csv"
    e2e_csv = out_root / "03_efficiency_and_latency_orin" / "latency_e2e_summary.csv"

    # Chart 1: delta_map_all_datasets
    if acc_csv.is_file() and acc_csv.stat().st_size > 10:
        df_acc = pd.read_csv(acc_csv)
        if not df_acc.empty:
            df_acc["mAP"] = df_acc["mAP"].astype(float)
            piv = df_acc.pivot_table(index=["dataset", "split"], columns="model", values="mAP", aggfunc="first")
            split_order = {
                ("VeRi-776", "Full Test"): 0,
                ("VehicleID", "Test-800"): 1,
                ("VehicleID", "Test-1600"): 2,
                ("VehicleID", "Test-2400"): 3,
                ("VRU", "Small (1200)"): 4,
                ("VRU", "Medium (2400)"): 5,
                ("VRU", "Large (8000)"): 6,
                ("VRAI", "Val (1000 IDs)"): 7,
            }
            ordered_idx = sorted(
                piv.index,
                key=lambda item: (split_order.get((str(item[0]), str(item[1])), 99), str(item[0]), str(item[1])),
            )
            piv = piv.reindex(ordered_idx)
            labels = [f"{ds}\n({sp})" for ds, sp in piv.index]
            d_bb = []
            d_gate = []
            for idx in piv.index:
                r = piv.loc[idx]
                v_res = r.get("resnet50ibn_rga", np.nan)
                v_cr = r.get("convnext_rga", np.nan)
                v_cb = r.get("convnext_bcgg", np.nan)
                d_bb.append(float(v_cr - v_res) if np.isfinite(v_cr) and np.isfinite(v_res) else 0.0)
                d_gate.append(float(v_cb - v_cr) if np.isfinite(v_cb) and np.isfinite(v_cr) else 0.0)

            x = np.arange(len(labels))
            w = 0.36
            fig, ax = plt.subplots(figsize=(max(7.5, len(labels) * 1.35), 5.0))
            b1 = ax.bar(x - w / 2, d_bb, w, label="Backbone: ConvNeXt+RGA − ResNet50-IBN+RGA", color="#2B6CB0")
            b2 = ax.bar(x + w / 2, d_gate, w, label="Gate: ConvNeXt+BCGG − ConvNeXt+RGA", color="#2F855A")
            ax.axhline(0.0, color="#333333", linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Δ mAP (%)", fontsize=11, fontweight="bold")
            ax.set_title("Chênh lệch mAP theo Backbone và Cổng chú ý trên các tập dữ liệu", fontsize=12, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)
            for bars in (b1, b2):
                for bar in bars:
                    val = bar.get_height()
                    ax.annotate(
                        f"{val:+.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3 if val >= 0 else -12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
            plt.tight_layout()
            _save_fig_dual(fig, charts_dir / "delta_map_all_datasets", dpi=dpi)

    # Chart 2: latency_fp32_fp16_trt (BS=1 p5-p50-p95)
    if pt_csv.is_file() and pt_csv.stat().st_size > 10:
        df_pt = pd.read_csv(pt_csv)
        df_pt1 = df_pt[df_pt["batch_size"] == 1].copy()
        df_trt1 = pd.read_csv(trt_csv) if trt_csv.is_file() else pd.DataFrame()
        if not df_trt1.empty:
            df_trt1 = df_trt1[df_trt1["batch_size"] == 1].copy()
        df_e2e = pd.read_csv(e2e_csv) if e2e_csv.is_file() else pd.DataFrame()

        models_order = ["resnet50ibn_rga", "convnext_rga", "convnext_bcgg_folded"]
        runtimes = ["PyTorch FP32", "PyTorch FP16", "TensorRT FP16", "E2E TRT FP16"]
        colors = ["#718096", "#3182CE", "#2F855A", "#D69E2E"]

        def _lookup_quantiles(cfg_k: str, rt: str):
            if rt == "PyTorch FP32":
                sub = df_pt1[(df_pt1["config"] == cfg_k) & (df_pt1["precision"] == "FP32")]
                if not sub.empty:
                    r = sub.iloc[0]
                    return float(r["p5_ms"]), float(r["p50_ms"]), float(r["p95_ms"])
            elif rt == "PyTorch FP16":
                sub = df_pt1[(df_pt1["config"] == cfg_k) & (df_pt1["precision"] == "FP16")]
                if not sub.empty:
                    r = sub.iloc[0]
                    return float(r["p5_ms"]), float(r["p50_ms"]), float(r["p95_ms"])
            elif rt == "TensorRT FP16" and not df_trt1.empty:
                sub = df_trt1[df_trt1["config"] == cfg_k]
                if not sub.empty:
                    r = sub.iloc[0]
                    return float(r["p5_ms"]), float(r["p50_ms"]), float(r["p95_ms"])
            elif rt == "E2E TRT FP16" and not df_e2e.empty:
                sub = df_e2e[df_e2e["config"] == cfg_k]
                if not sub.empty:
                    r = sub.iloc[0]
                    return float(r["e2e_p5_ms"]), float(r["e2e_p50_ms"]), float(r["e2e_p95_ms"])
            return None

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        x = np.arange(len(models_order))
        w = 0.19
        for r_idx, (rt, color) in enumerate(zip(runtimes, colors)):
            p50s, low_err, high_err, xpos = [], [], [], []
            for m_idx, m_k in enumerate(models_order):
                q = _lookup_quantiles(m_k, rt)
                if q is not None:
                    p5, p50, p95 = q
                    xpos.append(x[m_idx] + (r_idx - 1.5) * w)
                    p50s.append(p50)
                    low_err.append(max(0.0, p50 - p5))
                    high_err.append(max(0.0, p95 - p50))
            if xpos:
                bars = ax.bar(xpos, p50s, w, label=rt, color=color, alpha=0.92)
                ax.errorbar(xpos, p50s, yerr=[low_err, high_err], fmt="none", ecolor="#1A202C", capsize=3, elinewidth=1.2)
                for b in bars:
                    ax.annotate(
                        f"{b.get_height():.1f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )

        ax.axhline(33.33, color="#E53E3E", linestyle="--", linewidth=1.3, label="30 FPS (33.33 ms)")
        ax.axhline(40.00, color="#DD6B20", linestyle=":", linewidth=1.3, label="25 FPS (40.00 ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in models_order], fontsize=10, fontweight="bold")
        ax.set_ylabel("Độ trễ BS=1 (ms) [p5–p50–p95]", fontsize=11, fontweight="bold")
        ax.set_title("Độ trễ suy luận BS=1 trên Jetson AGX Orin: PyTorch FP32/FP16 vs TensorRT FP16", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        _save_fig_dual(fig, charts_dir / "latency_fp32_fp16_trt", dpi=dpi)

    # Chart 3: accuracy_vs_latency_pareto
    if acc_csv.is_file() and acc_csv.stat().st_size > 10 and pt_csv.is_file() and pt_csv.stat().st_size > 10:
        df_acc = pd.read_csv(acc_csv)
        df_pt = pd.read_csv(pt_csv)
        df_trt = pd.read_csv(trt_csv) if trt_csv.is_file() else pd.DataFrame()
        if not df_acc.empty:
            df_acc["mAP"] = df_acc["mAP"].astype(float)
            mean_map_by_model = df_acc.groupby("model")["mAP"].mean().to_dict()
            pt1 = df_pt[(df_pt["batch_size"] == 1) & (df_pt["precision"] == "FP32")]
            trt1 = df_trt[df_trt["batch_size"] == 1] if not df_trt.empty else pd.DataFrame()

            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            palette = {"resnet50ibn_rga": "#E53E3E", "convnext_rga": "#3182CE", "convnext_bcgg": "#2F855A"}
            pts_trt = []
            for m_key, acc_key, lat_key in (
                ("resnet50ibn_rga", "resnet50ibn_rga", "resnet50ibn_rga"),
                ("convnext_rga", "convnext_rga", "convnext_rga"),
                ("convnext_bcgg", "convnext_bcgg", "convnext_bcgg_folded"),
            ):
                if acc_key not in mean_map_by_model:
                    continue
                y_val = float(mean_map_by_model[acc_key])
                sub_pt = pt1[pt1["config"] == lat_key]
                if not sub_pt.empty:
                    x_pt = float(sub_pt.iloc[0]["p50_ms"])
                    ax.scatter(x_pt, y_val, marker="o", s=95, color=palette[m_key], alpha=0.55)
                    ax.annotate(f"{MODEL_DISPLAY_NAMES[m_key]} (PT FP32)", (x_pt, y_val), xytext=(5, -10), textcoords="offset points", fontsize=8)
                if not trt1.empty:
                    sub_t = trt1[trt1["config"] == lat_key]
                    if not sub_t.empty:
                        x_t = float(sub_t.iloc[0]["p50_ms"])
                        ax.scatter(x_t, y_val, marker="*", s=180, color=palette[m_key])
                        ax.annotate(f"{MODEL_DISPLAY_NAMES[m_key]} (TRT FP16)", (x_t, y_val), xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold")
                        pts_trt.append((x_t, y_val))

            if len(pts_trt) >= 2:
                pts_trt.sort(key=lambda t: t[0])
                pareto = []
                best_y = -1e9
                for px, py in pts_trt:
                    if py >= best_y:
                        pareto.append((px, py))
                        best_y = py
                if len(pareto) >= 2:
                    ax.plot([p[0] for p in pareto], [p[1] for p in pareto], "k--", linewidth=1.2, label="Pareto frontier (TRT FP16)")

            ax.set_xlabel("Độ trễ BS=1 p50 (ms)", fontsize=11, fontweight="bold")
            ax.set_ylabel("mAP trung bình trên các tập đã đánh giá (%)", fontsize=11, fontweight="bold")
            ax.set_title("Đánh đổi Độ chính xác vs Độ trễ (Pareto Frontier) trên Jetson AGX Orin", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=9)
            plt.tight_layout()
            _save_fig_dual(fig, charts_dir / "accuracy_vs_latency_pareto", dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Generate Chapter 5 Visualizations (300 DPI + PDF)")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Regenerate only charts/ from existing CSVs without re-running retrieval/heatmap inference",
    )
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "05_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_cfg = cfg.get("visualizations", {})
    dpi = int(vis_cfg.get("dpi", 300))

    if not args.charts_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prep_cfg = cfg.get("preprocess", {})
        tfm, _ = build_eval_transform(
            img_size=int(prep_cfg.get("img_size", 224)),
            legacy_resize=bool(prep_cfg.get("legacy_resize", False)),
            mean=prep_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=prep_cfg.get("std", [0.229, 0.224, 0.225]),
        )
        generate_retrieval_examples(cfg, out_dir, tfm, device, dpi)
        generate_attention_heatmaps(cfg, out_dir, tfm, device, dpi)

    generate_metric_charts(out_root, out_dir, dpi)


if __name__ == "__main__":
    main()
