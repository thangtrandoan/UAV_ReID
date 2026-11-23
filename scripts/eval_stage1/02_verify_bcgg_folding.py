#!/usr/bin/env python3
"""Verify mathematical and retrieval equivalence of BCGG before vs after to_deploy()."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    load_eval_config,
    load_gasnet_checkpoint,
    reset_gpu_memory,
    resolve_checkpoint_path,
    resolve_output_dir,
    write_csv,
    write_json,
)
from datasets import (  # noqa: E402
    build_eval_transform,
    load_veri776,
    load_vrai_val,
    load_vru_split,
    make_eval_loader,
)
from metrics import evaluate_retrieval_chunked, extract_features_pytorch  # noqa: E402


def _load_domain_dataset(cfg: Dict[str, Any], domain: str, transform):
    ds_cfg = cfg.get("datasets", {})
    if domain == "veri":
        return load_veri776(ds_cfg["veri"]["dir"], transform)
    if domain == "vru":
        vru_cfg = ds_cfg["vru"]
        return load_vru_split(vru_cfg["dir"], "Small", seed=int(vru_cfg.get("seed", 42)), transform=transform)
    if domain == "vrai":
        vrai_cfg = ds_cfg["vrai"]
        return load_vrai_val(
            vrai_cfg["dir"],
            num_ids=int(vrai_cfg.get("num_val_ids", 1000)),
            seed=int(vrai_cfg.get("seed", 42)),
            transform=transform,
        )
    raise ValueError(f"Unsupported BCGG equivalence domain: {domain!r}")


@torch.no_grad()
def _forward_with_gate_hooks(
    model: torch.nn.Module,
    batch: torch.Tensor,
    precision: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    gate_outputs: Dict[str, torch.Tensor] = {}
    hooks = []
    for name in ("ga1", "ga2", "ga3", "ga4"):
        mod = getattr(model, name)

        def _make_hook(k: str):
            def _hook(_m, _inp, out):
                gate_outputs[k] = out.detach().float().cpu()

            return _hook

        hooks.append(mod.register_forward_hook(_make_hook(name)))

    use_autocast = (device.type == "cuda") and (precision == "fp16")
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            bn_global, bn_fs = model(batch)
            z = F.normalize(torch.cat([bn_global, bn_fs], dim=1).float(), p=2, dim=1)
    finally:
        for h in hooks:
            h.remove()
    return gate_outputs, z.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Verify BCGG Deploy Folding Equivalence")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if any configured domain checkpoint/dataset is missing")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "02_bcgg_deploy_equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eq_cfg = cfg.get("bcgg_equivalence", {})
    domains = list(eq_cfg.get("domains", ["vru", "veri", "vrai"]))
    batch_n = int(eq_cfg.get("tensor_batch_size", 32))
    fp32_thresh = float(eq_cfg.get("fp32_max_diff_threshold", 1e-4))

    prep_cfg = cfg.get("preprocess", {})
    tfm, _ = build_eval_transform(
        img_size=int(prep_cfg.get("img_size", 224)),
        legacy_resize=bool(prep_cfg.get("legacy_resize", False)),
        mean=prep_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=prep_cfg.get("std", [0.229, 0.224, 0.225]),
    )
    eval_bs = int(cfg.get("eval", {}).get("batch_size", 64))
    num_workers = int(cfg.get("eval", {}).get("num_workers", 4))

    tensor_rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    report_domains: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for domain in domains:
        ckpt_path = resolve_checkpoint_path(cfg, domain, "convnext_bcgg")
        if not ckpt_path.is_file():
            msg = f"Missing trained convnext_bcgg checkpoint for {domain}: {ckpt_path}"
            if args.strict:
                raise FileNotFoundError(f"[FAIL] {msg}")
            print(f"[WARNING] Skipping {domain}: {msg}")
            skipped.append({"domain": domain, "reason": msg})
            continue

        try:
            q_ds, g_ds, ds_meta = _load_domain_dataset(cfg, domain, tfm)
        except FileNotFoundError as exc:
            if args.strict:
                raise
            print(f"[WARNING] Skipping {domain} dataset: {exc}")
            skipped.append({"domain": domain, "reason": str(exc)})
            continue

        print("=" * 80)
        print(f"[INFO] Verifying BCGG folding equivalence on {domain}: {ckpt_path.name}")
        reset_gpu_memory(device)
        model_unfolded, ckpt_info = load_gasnet_checkpoint(
            ckpt_path, expected_model_key="convnext_bcgg", deploy=False, device=device
        )
        model_folded = model_unfolded.to_deploy().eval().to(device)
        assert not model_unfolded.ga3.deployed and model_folded.ga3.deployed

        # Collect 32 real images
        n_imgs = min(batch_n, len(q_ds))
        real_batch = torch.stack([q_ds[i][0] for i in range(n_imgs)], dim=0).to(device)

        domain_tensor_summary: Dict[str, Any] = {}
        for prec in ("fp32", "fp16"):
            gates_u, z_u = _forward_with_gate_hooks(model_unfolded, real_batch, prec, device)
            gates_f, z_f = _forward_with_gate_hooks(model_folded, real_batch, prec, device)

            for stage_name in ("ga1", "ga2", "ga3", "ga4"):
                diff = (gates_u[stage_name] - gates_f[stage_name]).abs()
                tensor_rows.append(
                    {
                        "domain": domain,
                        "checkpoint_sha256": ckpt_info["sha256"],
                        "precision": prec.upper(),
                        "batch_size": n_imgs,
                        "layer": stage_name,
                        "max_abs_diff": f"{diff.max().item():.6e}",
                        "mean_abs_diff": f"{diff.mean().item():.6e}",
                        "min_cosine_sim": "",
                        "mean_cosine_sim": "",
                    }
                )

            z_diff = (z_u - z_f).abs()
            cos_sim = F.cosine_similarity(z_u, z_f, dim=1)
            z_max = float(z_diff.max().item())
            z_mean = float(z_diff.mean().item())
            min_cos = float(cos_sim.min().item())
            mean_cos = float(cos_sim.mean().item())

            tensor_rows.append(
                {
                    "domain": domain,
                    "checkpoint_sha256": ckpt_info["sha256"],
                    "precision": prec.upper(),
                    "batch_size": n_imgs,
                    "layer": "z_out_960",
                    "max_abs_diff": f"{z_max:.6e}",
                    "mean_abs_diff": f"{z_mean:.6e}",
                    "min_cosine_sim": f"{min_cos:.8f}",
                    "mean_cosine_sim": f"{mean_cos:.8f}",
                }
            )
            print(
                f"  [{domain} | {prec.upper()}] z_out max_abs_diff={z_max:.3e} "
                f"mean_abs_diff={z_mean:.3e} min_cos={min_cos:.8f}"
            )
            if prec == "fp32" and z_max >= fp32_thresh:
                raise AssertionError(
                    f"[FAIL] FP32 BCGG folding max_abs_diff={z_max:.3e} >= threshold {fp32_thresh:.3e}"
                )
            domain_tensor_summary[prec] = {
                "z_max_abs_diff": z_max,
                "z_mean_abs_diff": z_mean,
                "z_min_cosine_sim": min_cos,
                "z_mean_cosine_sim": mean_cos,
            }

        # Retrieval-level equivalence (FP32)
        q_loader = make_eval_loader(q_ds, batch_size=eval_bs, num_workers=num_workers)
        g_loader = make_eval_loader(g_ds, batch_size=eval_bs, num_workers=num_workers)

        qf_u, qp_u, qc_u, _ = extract_features_pytorch(model_unfolded, q_loader, device=device, precision="fp32")
        gf_u, gp_u, gc_u, _ = extract_features_pytorch(model_unfolded, g_loader, device=device, precision="fp32")
        res_u = evaluate_retrieval_chunked(
            qf_u, qp_u, gf_u, gp_u, qc_u, gc_u, device=device, return_rank1_gallery_idx=True
        )

        qf_f, qp_f, qc_f, _ = extract_features_pytorch(model_folded, q_loader, device=device, precision="fp32")
        gf_f, gp_f, gc_f, _ = extract_features_pytorch(model_folded, g_loader, device=device, precision="fp32")
        res_f = evaluate_retrieval_chunked(
            qf_f, qp_f, gf_f, gp_f, qc_f, gc_f, device=device, return_rank1_gallery_idx=True
        )

        r1_u = np.asarray(res_u["rank1_gallery_idx"])
        r1_f = np.asarray(res_f["rank1_gallery_idx"])
        valid_mask = (r1_u >= 0) & (r1_f >= 0)
        rank1_flips = int(np.sum(r1_u[valid_mask] != r1_f[valid_mask]))

        d_map = float(res_f["mAP"]) - float(res_u["mAP"])
        d_r1 = float(res_f["R1"]) - float(res_u["R1"])
        d_minp = float(res_f["mINP"]) - float(res_u["mINP"])

        retrieval_rows.append(
            {
                "domain": domain,
                "dataset": ds_meta["dataset"],
                "split": ds_meta["split"],
                "checkpoint_sha256": ckpt_info["sha256"],
                "valid_queries": int(res_u["valid_queries_after_filter"]),
                "unfolded_mAP": f"{float(res_u['mAP']):.4f}",
                "folded_mAP": f"{float(res_f['mAP']):.4f}",
                "delta_mAP": f"{d_map:+.6f}",
                "unfolded_R1": f"{float(res_u['R1']):.4f}",
                "folded_R1": f"{float(res_f['R1']):.4f}",
                "delta_R1": f"{d_r1:+.6f}",
                "unfolded_mINP": f"{float(res_u['mINP']):.4f}",
                "folded_mINP": f"{float(res_f['mINP']):.4f}",
                "delta_mINP": f"{d_minp:+.6f}",
                "rank1_flips": rank1_flips,
            }
        )
        print(
            f"  [{domain} Retrieval] unfolded mAP={res_u['mAP']:.4f}% vs folded={res_f['mAP']:.4f}% "
            f"(Δ={d_map:+.6f}%) | rank1_flips={rank1_flips}"
        )

        report_domains.append(
            {
                "domain": domain,
                "dataset": ds_meta["dataset"],
                "split": ds_meta["split"],
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_info["sha256"],
                "tensor_equivalence": domain_tensor_summary,
                "retrieval_equivalence": {
                    "valid_queries": int(res_u["valid_queries_after_filter"]),
                    "unfolded": {k: float(res_u[k]) for k in ("mAP", "R1", "R5", "R10", "mINP")},
                    "folded": {k: float(res_f[k]) for k in ("mAP", "R1", "R5", "R10", "mINP")},
                    "delta": {"mAP": d_map, "R1": d_r1, "mINP": d_minp},
                    "rank1_flips": rank1_flips,
                },
            }
        )
        del model_unfolded, model_folded
        reset_gpu_memory(device)

    if not report_domains:
        raise RuntimeError("[FAIL] No trained convnext_bcgg checkpoints/datasets were verified!")

    write_csv(tensor_rows, out_dir / "bcgg_tensor_equivalence.csv")
    write_csv(retrieval_rows, out_dir / "bcgg_retrieval_equivalence.csv")
    write_json(
        {
            "fp32_max_diff_threshold": fp32_thresh,
            "verified_domains": report_domains,
            "skipped_domains": skipped,
        },
        out_dir / "bcgg_equivalence_report.json",
    )


if __name__ == "__main__":
    main()
