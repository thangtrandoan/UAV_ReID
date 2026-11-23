#!/usr/bin/env python3
"""Evaluate Stage 1 GASNet accuracy across VeRi-776, VehicleID (10 runs), VRU, and VRAI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_KEYS,
    TensorRT10Runner,
    build_or_load_trt_engine,
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
    load_vehicleid_split_pool,
    load_veri776,
    load_vrai_val,
    load_vru_split,
    make_eval_loader,
    split_gallery_query_indices,
)
from metrics import (  # noqa: E402
    evaluate_retrieval_chunked,
    extract_features_pytorch,
    extract_features_trt,
)


def _round2(val: float) -> str:
    return f"{val:.2f}"


def _eval_pair_pytorch(
    ckpt_path: Path,
    model_key: str,
    q_ds,
    g_ds,
    precision: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, Any]:
    reset_gpu_memory(device)
    deploy = "bcgg" in model_key
    model, ckpt_info = load_gasnet_checkpoint(
        ckpt_path, expected_model_key=model_key, deploy=deploy, device=device
    )
    q_loader = make_eval_loader(q_ds, batch_size=batch_size, num_workers=num_workers)
    g_loader = make_eval_loader(g_ds, batch_size=batch_size, num_workers=num_workers)
    q_feats, q_pids, q_cams, _ = extract_features_pytorch(model, q_loader, device=device, precision=precision)
    g_feats, g_pids, g_cams, _ = extract_features_pytorch(model, g_loader, device=device, precision=precision)
    res = evaluate_retrieval_chunked(q_feats, q_pids, g_feats, g_pids, q_cams, g_cams, device=device)
    res["checkpoint"] = str(ckpt_path)
    res["checkpoint_sha256"] = ckpt_info["sha256"]
    del model
    reset_gpu_memory(device)
    return res


def _eval_pair_trt(
    ckpt_path: Path,
    domain: str,
    model_key: str,
    q_ds,
    g_ds,
    trt_cache_dir: Path,
    cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, Any]:
    trt_cfg = cfg.get("efficiency", {}).get("trt", {})
    img_size = int(cfg.get("preprocess", {}).get("img_size", 224))
    max_bs = int(trt_cfg.get("max_bs", 32))
    engines_root = trt_cache_dir.parent
    _, engine_path = build_or_load_trt_engine(
        ckpt_path=ckpt_path,
        model_key=model_key,
        cache_dir=trt_cache_dir,
        tag=f"{domain}_{model_key}",
        trtexec_bin=str(trt_cfg.get("trtexec_bin", "/usr/src/tensorrt/bin/trtexec")),
        img_size=img_size,
        opset=int(trt_cfg.get("opset", 17)),
        min_bs=int(trt_cfg.get("min_bs", 1)),
        opt_bs=int(trt_cfg.get("opt_bs", 1)),
        max_bs=max_bs,
        timing_cache_file=engines_root / "trt_timing.cache",
        builder_opt_level=trt_cfg.get("builder_opt_level", 2),
        avg_timing=trt_cfg.get("avg_timing", 2),
        reuse_dirs=[engines_root, trt_cache_dir],
    )
    reset_gpu_memory(device)
    runner = TensorRT10Runner(engine_path, device=device)
    try:
        q_loader = make_eval_loader(q_ds, batch_size=batch_size, num_workers=num_workers)
        g_loader = make_eval_loader(g_ds, batch_size=batch_size, num_workers=num_workers)
        q_feats, q_pids, q_cams, _ = extract_features_trt(runner, q_loader, max_batch_size=max_bs)
        g_feats, g_pids, g_cams, _ = extract_features_trt(runner, g_loader, max_batch_size=max_bs)
        res = evaluate_retrieval_chunked(q_feats, q_pids, g_feats, g_pids, q_cams, g_cams, device=device)
    finally:
        runner.close()
        reset_gpu_memory(device)
    res["engine_path"] = str(engine_path)
    return res


def _append_summary_row(
    csv_rows: List[Dict[str, Any]],
    json_rows: List[Dict[str, Any]],
    dataset: str,
    split: str,
    model_key: str,
    precision: str,
    res: Dict[str, Any],
    stds: Optional[Dict[str, float]] = None,
) -> None:
    stds = stds or {}
    csv_rows.append(
        {
            "dataset": dataset,
            "split": split,
            "model": model_key,
            "precision": precision,
            "valid_queries": int(res["valid_queries_after_filter"]),
            "gallery_size": int(res["gallery_size"]),
            "mAP": _round2(float(res["mAP"])),
            "R1": _round2(float(res["R1"])),
            "R5": _round2(float(res["R5"])),
            "R10": _round2(float(res["R10"])),
            "mINP": _round2(float(res["mINP"])),
            "mAP_std": _round2(float(stds["mAP"])) if "mAP" in stds else "",
            "R1_std": _round2(float(stds["R1"])) if "R1" in stds else "",
            "R5_std": _round2(float(stds["R5"])) if "R5" in stds else "",
            "R10_std": _round2(float(stds["R10"])) if "R10" in stds else "",
            "mINP_std": _round2(float(stds["mINP"])) if "mINP" in stds else "",
        }
    )
    json_entry = {
        "dataset": dataset,
        "split": split,
        "model": model_key,
        "precision": precision,
        "total_queries_before_filter": int(res["total_queries_before_filter"]),
        "valid_queries_after_filter": int(res["valid_queries_after_filter"]),
        "gallery_size": int(res["gallery_size"]),
        "mAP": float(res["mAP"]),
        "R1": float(res["R1"]),
        "R5": float(res["R5"]),
        "R10": float(res["R10"]),
        "mINP": float(res["mINP"]),
    }
    if stds:
        json_entry["std"] = {k: float(v) for k, v in stds.items()}
    if "checkpoint" in res:
        json_entry["checkpoint"] = res["checkpoint"]
    if "checkpoint_sha256" in res:
        json_entry["checkpoint_sha256"] = res["checkpoint_sha256"]
    json_rows.append(json_entry)


def _update_precision_comparison(
    precision_records: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    fp32_lookup = {
        (r["dataset"], r["split"], r["model"]): r
        for r in precision_records
        if r["runtime_precision"] == "pytorch_fp32"
    }
    csv_rows: List[Dict[str, Any]] = []
    for r in precision_records:
        base = fp32_lookup.get((r["dataset"], r["split"], r["model"]))
        d_map = float(r["mAP"]) - float(base["mAP"]) if base else 0.0
        d_r1 = float(r["R1"]) - float(base["R1"]) if base else 0.0
        r["delta_mAP_vs_fp32"] = d_map
        r["delta_R1_vs_fp32"] = d_r1
        csv_rows.append(
            {
                "dataset": r["dataset"],
                "split": r["split"],
                "model": r["model"],
                "runtime_precision": r["runtime_precision"],
                "valid_queries": r["valid_queries_after_filter"],
                "gallery_size": r["gallery_size"],
                "mAP": _round2(float(r["mAP"])),
                "R1": _round2(float(r["R1"])),
                "R5": _round2(float(r["R5"])),
                "R10": _round2(float(r["R10"])),
                "mINP": _round2(float(r["mINP"])),
                "delta_mAP_vs_fp32": f"{d_map:+.2f}",
                "delta_R1_vs_fp32": f"{d_r1:+.2f}",
            }
        )
    write_csv(csv_rows, out_dir / "precision_comparison_fp32_fp16_trt.csv")
    write_json(precision_records, out_dir / "precision_comparison_fp32_fp16_trt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Accuracy Benchmarks across 4 Datasets")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        choices=["veri", "vehicleid", "vru", "vrai"],
        help="Optional subset of domains to evaluate",
    )
    parser.add_argument("--eval-trt", action="store_true", help="Also run TensorRT FP16 accuracy evaluation")
    parser.add_argument(
        "--trt-only",
        action="store_true",
        help="Only run TensorRT FP16 on precision_consistency_domains and merge into existing precision_comparison",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any dataset or checkpoint is missing")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "01_accuracy_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    trt_cache_dir = out_root / "03_efficiency_and_latency_orin" / "engines" / "accuracy_eval"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = cfg.get("eval", {})
    prep_cfg = cfg.get("preprocess", {})
    ds_cfg = cfg.get("datasets", {})
    batch_size = int(eval_cfg.get("batch_size", 64))
    num_workers = int(eval_cfg.get("num_workers", 4))
    primary_prec = str(eval_cfg.get("primary_precision", "fp32")).lower()
    cons_domains = set(eval_cfg.get("precision_consistency_domains", ["veri", "vrai"]))
    selected_domains = list(args.domains) if args.domains else ["veri", "vehicleid", "vru", "vrai"]

    tfm, tfm_meta = build_eval_transform(
        img_size=int(prep_cfg.get("img_size", 224)),
        legacy_resize=bool(prep_cfg.get("legacy_resize", False)),
        mean=prep_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=prep_cfg.get("std", [0.229, 0.224, 0.225]),
    )

    # TRT-only mode updates precision_comparison_fp32_fp16_trt.{csv,json}
    if args.trt_only:
        prec_json_path = out_dir / "precision_comparison_fp32_fp16_trt.json"
        if prec_json_path.is_file():
            existing = json.loads(prec_json_path.read_text(encoding="utf-8"))
            precision_records = [r for r in existing if r.get("runtime_precision") != "tensorrt_fp16"]
        else:
            precision_records = []

        for domain in selected_domains:
            if domain not in cons_domains:
                continue
            if domain == "veri":
                try:
                    q_ds, g_ds, ds_meta = load_veri776(ds_cfg["veri"]["dir"], tfm)
                except FileNotFoundError as exc:
                    if args.strict:
                        raise
                    print(f"[WARNING] Skipping TRT eval on veri: {exc}")
                    continue
                ds_label, split_label = "VeRi-776", "Full Test"
            elif domain == "vrai":
                try:
                    vcfg = ds_cfg["vrai"]
                    q_ds, g_ds, ds_meta = load_vrai_val(
                        vcfg["dir"], num_ids=int(vcfg.get("num_val_ids", 1000)), seed=int(vcfg.get("seed", 42)), transform=tfm
                    )
                except FileNotFoundError as exc:
                    if args.strict:
                        raise
                    print(f"[WARNING] Skipping TRT eval on vrai: {exc}")
                    continue
                ds_label, split_label = "VRAI", str(ds_meta["split"])
            else:
                continue

            for model_key in MODEL_KEYS:
                ckpt_path = resolve_checkpoint_path(cfg, domain, model_key)
                if not ckpt_path.is_file():
                    if args.strict:
                        raise FileNotFoundError(f"[FAIL] Missing checkpoint: {ckpt_path}")
                    print(f"[WARNING] Skipping TRT eval for {domain}/{model_key}: missing {ckpt_path}")
                    continue
                print(f"[EVAL] TensorRT FP16 on {ds_label} ({split_label}) | {model_key}")
                res = _eval_pair_trt(
                    ckpt_path, domain, model_key, q_ds, g_ds, trt_cache_dir, cfg, batch_size, num_workers, device
                )
                print(
                    f" -> mAP={res['mAP']:.2f}% R1={res['R1']:.2f}% R5={res['R5']:.2f}% "
                    f"R10={res['R10']:.2f}% mINP={res['mINP']:.2f}%"
                )
                precision_records.append(
                    {
                        "dataset": ds_label,
                        "split": split_label,
                        "model": model_key,
                        "runtime_precision": "tensorrt_fp16",
                        "total_queries_before_filter": int(res["total_queries_before_filter"]),
                        "valid_queries_after_filter": int(res["valid_queries_after_filter"]),
                        "gallery_size": int(res["gallery_size"]),
                        "mAP": float(res["mAP"]),
                        "R1": float(res["R1"]),
                        "R5": float(res["R5"]),
                        "R10": float(res["R10"]),
                        "mINP": float(res["mINP"]),
                        "checkpoint": str(ckpt_path),
                        "engine_path": res["engine_path"],
                    }
                )
        _update_precision_comparison(precision_records, out_dir)
        return

    summary_csv_rows: List[Dict[str, Any]] = []
    summary_json_rows: List[Dict[str, Any]] = []
    vehicleid_raw_rows: List[Dict[str, Any]] = []
    precision_records: List[Dict[str, Any]] = []
    skipped_items: List[Dict[str, str]] = []
    dataset_protocols: Dict[str, Any] = {}

    # 1. VeRi-776
    if "veri" in selected_domains:
        try:
            q_veri, g_veri, veri_meta = load_veri776(ds_cfg["veri"]["dir"], tfm)
            dataset_protocols["veri"] = veri_meta
            for model_key in MODEL_KEYS:
                ckpt_path = resolve_checkpoint_path(cfg, "veri", model_key)
                if not ckpt_path.is_file():
                    msg = f"Missing checkpoint: {ckpt_path}"
                    if args.strict:
                        raise FileNotFoundError(msg)
                    print(f"[WARNING] Skipping VeRi-776 / {model_key}: {msg}")
                    skipped_items.append({"domain": "veri", "model": model_key, "reason": msg})
                    continue

                print(f"[EVAL] VeRi-776 | {model_key} ({primary_prec.upper()})")
                res_fp32 = _eval_pair_pytorch(
                    ckpt_path, model_key, q_veri, g_veri, primary_prec, batch_size, num_workers, device
                )
                dataset_protocols["veri"]["valid_queries_after_filter"] = int(res_fp32["valid_queries_after_filter"])
                _append_summary_row(
                    summary_csv_rows, summary_json_rows, "VeRi-776", "Full Test", model_key, primary_prec.upper(), res_fp32
                )
                print(
                    f" -> valid_q={res_fp32['valid_queries_after_filter']}/{res_fp32['total_queries_before_filter']} | "
                    f"mAP={res_fp32['mAP']:.2f}% R1={res_fp32['R1']:.2f}% mINP={res_fp32['mINP']:.2f}%"
                )

                if "veri" in cons_domains:
                    precision_records.append(
                        {
                            "dataset": "VeRi-776",
                            "split": "Full Test",
                            "model": model_key,
                            "runtime_precision": f"pytorch_{primary_prec}",
                            "total_queries_before_filter": int(res_fp32["total_queries_before_filter"]),
                            "valid_queries_after_filter": int(res_fp32["valid_queries_after_filter"]),
                            "gallery_size": int(res_fp32["gallery_size"]),
                            "mAP": float(res_fp32["mAP"]),
                            "R1": float(res_fp32["R1"]),
                            "R5": float(res_fp32["R5"]),
                            "R10": float(res_fp32["R10"]),
                            "mINP": float(res_fp32["mINP"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    print(f"[EVAL] VeRi-776 | {model_key} (FP16 consistency)")
                    res_fp16 = _eval_pair_pytorch(
                        ckpt_path, model_key, q_veri, g_veri, "fp16", batch_size, num_workers, device
                    )
                    precision_records.append(
                        {
                            "dataset": "VeRi-776",
                            "split": "Full Test",
                            "model": model_key,
                            "runtime_precision": "pytorch_fp16",
                            "total_queries_before_filter": int(res_fp16["total_queries_before_filter"]),
                            "valid_queries_after_filter": int(res_fp16["valid_queries_after_filter"]),
                            "gallery_size": int(res_fp16["gallery_size"]),
                            "mAP": float(res_fp16["mAP"]),
                            "R1": float(res_fp16["R1"]),
                            "R5": float(res_fp16["R5"]),
                            "R10": float(res_fp16["R10"]),
                            "mINP": float(res_fp16["mINP"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    if args.eval_trt:
                        res_trt = _eval_pair_trt(
                            ckpt_path, "veri", model_key, q_veri, g_veri, trt_cache_dir, cfg, batch_size, num_workers, device
                        )
                        precision_records.append(
                            {
                                "dataset": "VeRi-776",
                                "split": "Full Test",
                                "model": model_key,
                                "runtime_precision": "tensorrt_fp16",
                                "total_queries_before_filter": int(res_trt["total_queries_before_filter"]),
                                "valid_queries_after_filter": int(res_trt["valid_queries_after_filter"]),
                                "gallery_size": int(res_trt["gallery_size"]),
                                "mAP": float(res_trt["mAP"]),
                                "R1": float(res_trt["R1"]),
                                "R5": float(res_trt["R5"]),
                                "R10": float(res_trt["R10"]),
                                "mINP": float(res_trt["mINP"]),
                                "checkpoint": str(ckpt_path),
                                "engine_path": res_trt["engine_path"],
                            }
                        )
        except FileNotFoundError as exc:
            if args.strict:
                raise
            print(f"[WARNING] Skipping VeRi-776 dataset: {exc}")
            skipped_items.append({"domain": "veri", "model": "*", "reason": str(exc)})

    # 2. VehicleID_V1.0 (3 splits x 10 seeds [42..51])
    if "vehicleid" in selected_domains:
        veh_cfg = ds_cfg["vehicleid"]
        veh_splits = [int(s) for s in veh_cfg.get("splits", [800, 1600, 2400])]
        veh_seeds = [int(s) for s in veh_cfg.get("seeds", list(range(42, 52)))]
        dataset_protocols["vehicleid"] = {"splits": veh_splits, "seeds": veh_seeds, "dir": str(veh_cfg["dir"])}

        for split_sz in veh_splits:
            try:
                pool_ds, idx_by_id, split_meta = load_vehicleid_split_pool(veh_cfg["dir"], split_sz, tfm)
            except FileNotFoundError as exc:
                if args.strict:
                    raise
                print(f"[WARNING] Skipping VehicleID test_list_{split_sz}: {exc}")
                skipped_items.append({"domain": "vehicleid", "model": f"split_{split_sz}", "reason": str(exc)})
                continue

            split_label = f"Test-{split_sz}"
            for model_key in MODEL_KEYS:
                ckpt_path = resolve_checkpoint_path(cfg, "vehicleid", model_key)
                if not ckpt_path.is_file():
                    msg = f"Missing checkpoint: {ckpt_path}"
                    if args.strict:
                        raise FileNotFoundError(msg)
                    print(f"[WARNING] Skipping VehicleID {split_label} / {model_key}: {msg}")
                    skipped_items.append({"domain": f"vehicleid_{split_sz}", "model": model_key, "reason": msg})
                    continue

                print(f"[EVAL] VehicleID {split_label} | {model_key} ({primary_prec.upper()}, {len(veh_seeds)} runs)")
                reset_gpu_memory(device)
                model, ckpt_info = load_gasnet_checkpoint(
                    ckpt_path, expected_model_key=model_key, deploy=("bcgg" in model_key), device=device
                )
                pool_loader = make_eval_loader(pool_ds, batch_size=batch_size, num_workers=num_workers)
                pool_feats, pool_pids, _, _ = extract_features_pytorch(
                    model, pool_loader, device=device, precision=primary_prec
                )
                del model
                reset_gpu_memory(device)

                run_metrics: List[Dict[str, Any]] = []
                for run_idx, seed in enumerate(veh_seeds, start=1):
                    q_idxs, g_idxs = split_gallery_query_indices(idx_by_id, seed=seed)
                    q_t = torch.tensor(q_idxs, dtype=torch.long)
                    g_t = torch.tensor(g_idxs, dtype=torch.long)
                    res = evaluate_retrieval_chunked(
                        pool_feats[q_t], pool_pids[q_t], pool_feats[g_t], pool_pids[g_t], device=device
                    )
                    run_metrics.append(res)
                    vehicleid_raw_rows.append(
                        {
                            "dataset": "VehicleID",
                            "split": split_label,
                            "model": model_key,
                            "precision": primary_prec.upper(),
                            "run_idx": run_idx,
                            "seed": seed,
                            "valid_queries": int(res["valid_queries_after_filter"]),
                            "gallery_size": int(res["gallery_size"]),
                            "mAP": f"{float(res['mAP']):.4f}",
                            "R1": f"{float(res['R1']):.4f}",
                            "R5": f"{float(res['R5']):.4f}",
                            "R10": f"{float(res['R10']):.4f}",
                            "mINP": f"{float(res['mINP']):.4f}",
                        }
                    )

                mean_res = {
                    "total_queries_before_filter": int(np.mean([r["total_queries_before_filter"] for r in run_metrics])),
                    "valid_queries_after_filter": int(np.mean([r["valid_queries_after_filter"] for r in run_metrics])),
                    "gallery_size": int(run_metrics[0]["gallery_size"]),
                    "mAP": float(np.mean([r["mAP"] for r in run_metrics])),
                    "R1": float(np.mean([r["R1"] for r in run_metrics])),
                    "R5": float(np.mean([r["R5"] for r in run_metrics])),
                    "R10": float(np.mean([r["R10"] for r in run_metrics])),
                    "mINP": float(np.mean([r["mINP"] for r in run_metrics])),
                    "checkpoint": str(ckpt_path),
                    "checkpoint_sha256": ckpt_info["sha256"],
                }
                stds = {
                    k: float(np.std([r[k] for r in run_metrics], ddof=0))
                    for k in ("mAP", "R1", "R5", "R10", "mINP")
                }
                _append_summary_row(
                    summary_csv_rows,
                    summary_json_rows,
                    "VehicleID",
                    split_label,
                    model_key,
                    primary_prec.upper(),
                    mean_res,
                    stds=stds,
                )
                print(
                    f" -> mAP={mean_res['mAP']:.2f}±{stds['mAP']:.2f}% | "
                    f"R1={mean_res['R1']:.2f}±{stds['R1']:.2f}%"
                )

    # 3. VRU (Small / Medium / Large, seed=42)
    if "vru" in selected_domains:
        vru_cfg = ds_cfg["vru"]
        vru_seed = int(vru_cfg.get("seed", 42))
        vru_splits = list(vru_cfg.get("splits", ["Small", "Medium", "Large"]))
        dataset_protocols["vru"] = {"splits": {}}
        for split_name in vru_splits:
            try:
                q_vru, g_vru, vru_meta = load_vru_split(vru_cfg["dir"], split_name, seed=vru_seed, transform=tfm)
                dataset_protocols["vru"]["splits"][split_name] = vru_meta
            except FileNotFoundError as exc:
                if args.strict:
                    raise
                print(f"[WARNING] Skipping VRU {split_name}: {exc}")
                skipped_items.append({"domain": "vru", "model": split_name, "reason": str(exc)})
                continue

            for model_key in MODEL_KEYS:
                ckpt_path = resolve_checkpoint_path(cfg, "vru", model_key)
                if not ckpt_path.is_file():
                    msg = f"Missing checkpoint: {ckpt_path}"
                    if args.strict:
                        raise FileNotFoundError(msg)
                    print(f"[WARNING] Skipping VRU {split_name} / {model_key}: {msg}")
                    skipped_items.append({"domain": f"vru_{split_name}", "model": model_key, "reason": msg})
                    continue

                print(f"[EVAL] VRU {vru_meta['split']} | {model_key} ({primary_prec.upper()})")
                res = _eval_pair_pytorch(
                    ckpt_path, model_key, q_vru, g_vru, primary_prec, batch_size, num_workers, device
                )
                _append_summary_row(
                    summary_csv_rows,
                    summary_json_rows,
                    "VRU",
                    str(vru_meta["split"]),
                    model_key,
                    primary_prec.upper(),
                    res,
                )
                print(f" -> mAP={res['mAP']:.2f}% R1={res['R1']:.2f}% mINP={res['mINP']:.2f}%")

    # 4. VRAI (1000 held-out IDs, cross-cam filter)
    if "vrai" in selected_domains:
        vrai_cfg = ds_cfg["vrai"]
        try:
            q_vrai, g_vrai, vrai_meta = load_vrai_val(
                vrai_cfg["dir"],
                num_ids=int(vrai_cfg.get("num_val_ids", 1000)),
                seed=int(vrai_cfg.get("seed", 42)),
                transform=tfm,
            )
            dataset_protocols["vrai"] = vrai_meta
            for model_key in MODEL_KEYS:
                ckpt_path = resolve_checkpoint_path(cfg, "vrai", model_key)
                if not ckpt_path.is_file():
                    msg = f"Missing checkpoint: {ckpt_path}"
                    if args.strict:
                        raise FileNotFoundError(msg)
                    print(f"[WARNING] Skipping VRAI / {model_key}: {msg}")
                    skipped_items.append({"domain": "vrai", "model": model_key, "reason": msg})
                    continue

                print(f"[EVAL] VRAI {vrai_meta['split']} | {model_key} ({primary_prec.upper()})")
                res_fp32 = _eval_pair_pytorch(
                    ckpt_path, model_key, q_vrai, g_vrai, primary_prec, batch_size, num_workers, device
                )
                dataset_protocols["vrai"]["valid_queries_after_filter"] = int(res_fp32["valid_queries_after_filter"])
                _append_summary_row(
                    summary_csv_rows,
                    summary_json_rows,
                    "VRAI",
                    str(vrai_meta["split"]),
                    model_key,
                    primary_prec.upper(),
                    res_fp32,
                )
                print(
                    f" -> before_filter={res_fp32['total_queries_before_filter']} "
                    f"valid_after_filter={res_fp32['valid_queries_after_filter']} | "
                    f"mAP={res_fp32['mAP']:.2f}% R1={res_fp32['R1']:.2f}% mINP={res_fp32['mINP']:.2f}%"
                )

                if "vrai" in cons_domains:
                    precision_records.append(
                        {
                            "dataset": "VRAI",
                            "split": str(vrai_meta["split"]),
                            "model": model_key,
                            "runtime_precision": f"pytorch_{primary_prec}",
                            "total_queries_before_filter": int(res_fp32["total_queries_before_filter"]),
                            "valid_queries_after_filter": int(res_fp32["valid_queries_after_filter"]),
                            "gallery_size": int(res_fp32["gallery_size"]),
                            "mAP": float(res_fp32["mAP"]),
                            "R1": float(res_fp32["R1"]),
                            "R5": float(res_fp32["R5"]),
                            "R10": float(res_fp32["R10"]),
                            "mINP": float(res_fp32["mINP"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    print(f"[EVAL] VRAI {vrai_meta['split']} | {model_key} (FP16 consistency)")
                    res_fp16 = _eval_pair_pytorch(
                        ckpt_path, model_key, q_vrai, g_vrai, "fp16", batch_size, num_workers, device
                    )
                    precision_records.append(
                        {
                            "dataset": "VRAI",
                            "split": str(vrai_meta["split"]),
                            "model": model_key,
                            "runtime_precision": "pytorch_fp16",
                            "total_queries_before_filter": int(res_fp16["total_queries_before_filter"]),
                            "valid_queries_after_filter": int(res_fp16["valid_queries_after_filter"]),
                            "gallery_size": int(res_fp16["gallery_size"]),
                            "mAP": float(res_fp16["mAP"]),
                            "R1": float(res_fp16["R1"]),
                            "R5": float(res_fp16["R5"]),
                            "R10": float(res_fp16["R10"]),
                            "mINP": float(res_fp16["mINP"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    if args.eval_trt:
                        res_trt = _eval_pair_trt(
                            ckpt_path, "vrai", model_key, q_vrai, g_vrai, trt_cache_dir, cfg, batch_size, num_workers, device
                        )
                        precision_records.append(
                            {
                                "dataset": "VRAI",
                                "split": str(vrai_meta["split"]),
                                "model": model_key,
                                "runtime_precision": "tensorrt_fp16",
                                "total_queries_before_filter": int(res_trt["total_queries_before_filter"]),
                                "valid_queries_after_filter": int(res_trt["valid_queries_after_filter"]),
                                "gallery_size": int(res_trt["gallery_size"]),
                                "mAP": float(res_trt["mAP"]),
                                "R1": float(res_trt["R1"]),
                                "R5": float(res_trt["R5"]),
                                "R10": float(res_trt["R10"]),
                                "mINP": float(res_trt["mINP"]),
                                "checkpoint": str(ckpt_path),
                                "engine_path": res_trt["engine_path"],
                            }
                        )
        except FileNotFoundError as exc:
            if args.strict:
                raise
            print(f"[WARNING] Skipping VRAI dataset: {exc}")
            skipped_items.append({"domain": "vrai", "model": "*", "reason": str(exc)})

    summary_fields = [
        "dataset",
        "split",
        "model",
        "precision",
        "valid_queries",
        "gallery_size",
        "mAP",
        "R1",
        "R5",
        "R10",
        "mINP",
        "mAP_std",
        "R1_std",
        "R5_std",
        "R10_std",
        "mINP_std",
    ]
    write_csv(summary_csv_rows, out_dir / "accuracy_summary.csv", fieldnames=summary_fields)
    write_json(summary_json_rows, out_dir / "accuracy_summary.json")

    raw_fields = [
        "dataset",
        "split",
        "model",
        "precision",
        "run_idx",
        "seed",
        "valid_queries",
        "gallery_size",
        "mAP",
        "R1",
        "R5",
        "R10",
        "mINP",
    ]
    write_csv(vehicleid_raw_rows, out_dir / "vehicleid_10runs_raw.csv", fieldnames=raw_fields)
    _update_precision_comparison(precision_records, out_dir)
    write_json(
        {
            "preprocess": tfm_meta,
            "primary_precision": primary_prec,
            "datasets": dataset_protocols,
            "skipped": skipped_items,
        },
        out_dir / "protocol_metadata.json",
    )


if __name__ == "__main__":
    main()
