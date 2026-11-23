#!/usr/bin/env python3
"""Export ONNX (opset 17), build TensorRT FP16 engines, and benchmark TRT + End-to-End latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    ROOT_DIR,
    TensorRT10Runner,
    build_or_load_trt_engine,
    load_eval_config,
    load_gasnet_checkpoint,
    maybe_lock_jetson_clocks,
    read_jetson_thermals,
    read_nvpmodel_status,
    reset_gpu_memory,
    resolve_efficiency_checkpoint,
    resolve_output_dir,
    summarize_latencies,
    write_csv,
    write_json,
)


def _find_sample_crop_bgr() -> Tuple[np.ndarray, str]:
    for rel in ("data/VeRi/image_query", "data/VRAI/images_train", "data/VRU/Pic"):
        d = ROOT_DIR / rel
        if d.is_dir():
            for p in sorted(d.glob("*.jpg"))[:5]:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    return img, str(p)
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(180, 240, 3), dtype=np.uint8), "synthetic_180x240_bgr"


@torch.no_grad()
def _verify_trt_vs_pytorch(
    ckpt_path: Path,
    model_key: str,
    deploy: bool,
    runner: TensorRT10Runner,
    img_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model, _ = load_gasnet_checkpoint(
        ckpt_path, expected_model_key=model_key, deploy=deploy, device=device
    )
    results = {}
    for bs in (1, 4):
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        bn_g, bn_f = model(x)
        pt_out = torch.cat([bn_g, bn_f], dim=1).float()
        trt_out = runner.infer(x).float()
        torch.cuda.synchronize(device)
        if not torch.isfinite(trt_out).all():
            raise RuntimeError(f"[FAIL] Non-finite TensorRT output for {model_key} at BS={bs}")
        cos = F.cosine_similarity(F.normalize(pt_out, p=2, dim=1), F.normalize(trt_out, p=2, dim=1), dim=1)
        max_diff = float((pt_out - trt_out).abs().max().item())
        min_cos = float(cos.min().item())
        if min_cos < 0.99:
            raise RuntimeError(
                f"[FAIL] TensorRT FP16 divergence for {model_key} BS={bs}: min_cos={min_cos:.6f}"
            )
        results[f"bs{bs}"] = {"min_cosine_sim": min_cos, "max_abs_diff": max_diff, "all_finite": True}
    del model
    reset_gpu_memory(device)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: ONNX Export & TensorRT FP16 + E2E Benchmark")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lock-clocks", action="store_true", help="Run sudo nvpmodel -m 0 && sudo jetson_clocks")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild ONNX and TensorRT engines")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "03_efficiency_and_latency_orin"
    engines_dir = out_dir / "engines"
    out_dir.mkdir(parents=True, exist_ok=True)
    engines_dir.mkdir(parents=True, exist_ok=True)

    eff_cfg = cfg.get("efficiency", {})
    trt_cfg = eff_cfg.get("trt", {})
    prep_cfg = cfg.get("preprocess", {})
    img_size = int(prep_cfg.get("img_size", 224))
    mean_np = np.array(prep_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(prep_cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, 3)

    batch_sizes = [int(b) for b in eff_cfg.get("batch_sizes", [1, 2, 4, 8, 32])]
    warmup = int(args.warmup if args.warmup is not None else trt_cfg.get("warmup_runs", 50))
    runs = int(args.runs if args.runs is not None else trt_cfg.get("measure_runs", 200))
    gallery_n = int(trt_cfg.get("e2e_gallery_size", 8000))
    topk = int(trt_cfg.get("e2e_topk", 10))

    hw_before = maybe_lock_jetson_clocks(args.lock_clocks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deploy_configs = [
        ("resnet50ibn_rga", "resnet50ibn_rga", False),
        ("convnext_rga", "convnext_rga", False),
        ("convnext_bcgg_folded", "convnext_bcgg", True),
    ]

    crop_bgr, crop_source = _find_sample_crop_bgr()

    trt_summary_rows: List[Dict[str, Any]] = []
    trt_raw_rows: List[Dict[str, Any]] = []
    e2e_summary_rows: List[Dict[str, Any]] = []
    e2e_raw_rows: List[Dict[str, Any]] = []
    parity_checks: Dict[str, Any] = {}

    for config_key, model_key, deploy in deploy_configs:
        domain, ckpt_path = resolve_efficiency_checkpoint(cfg, model_key)
        reset_gpu_memory(device)
        onnx_path, engine_path = build_or_load_trt_engine(
            ckpt_path=ckpt_path,
            model_key=model_key,
            cache_dir=engines_dir,
            tag=config_key,
            trtexec_bin=str(trt_cfg.get("trtexec_bin", "/usr/src/tensorrt/bin/trtexec")),
            img_size=img_size,
            opset=int(trt_cfg.get("opset", 17)),
            min_bs=int(trt_cfg.get("min_bs", 1)),
            opt_bs=int(trt_cfg.get("opt_bs", 1)),
            max_bs=int(trt_cfg.get("max_bs", 32)),
            force_rebuild=args.rebuild,
            timing_cache_file=engines_dir / "trt_timing.cache",
            builder_opt_level=trt_cfg.get("builder_opt_level", 2),
            avg_timing=trt_cfg.get("avg_timing", 2),
            reuse_dirs=[engines_dir],
        )
        engine_mb = engine_path.stat().st_size / (1024.0 * 1024.0)

        runner = TensorRT10Runner(engine_path, device=device)
        try:
            parity_checks[config_key] = _verify_trt_vs_pytorch(
                ckpt_path, model_key, deploy, runner, img_size, device
            )
            print(f"[OK] {config_key} TRT parity: {parity_checks[config_key]}")

            # 1. Pure TensorRT engine benchmark across batch sizes
            for bs in batch_sizes:
                dummy_gpu = torch.randn(bs, 3, img_size, img_size, device=device, dtype=torch.float32)
                for _ in range(warmup):
                    _ = runner.infer(dummy_gpu)
                runner.synchronize()
                torch.cuda.synchronize(device)

                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                latencies: List[float] = []
                for idx in range(1, runs + 1):
                    start_ev.record(runner.stream)
                    _ = runner.infer(dummy_gpu)
                    end_ev.record(runner.stream)
                    runner.synchronize()
                    dt_ms = float(start_ev.elapsed_time(end_ev))
                    latencies.append(dt_ms)
                    trt_raw_rows.append(
                        {
                            "config": config_key,
                            "batch_size": bs,
                            "iter": idx,
                            "latency_ms": f"{dt_ms:.6f}",
                        }
                    )

                stats = summarize_latencies(latencies, batch_size=bs)
                trt_summary_rows.append(
                    {
                        "config": config_key,
                        "display_name": MODEL_DISPLAY_NAMES[config_key],
                        "precision": "TRT_FP16",
                        "batch_size": bs,
                        "warmup_runs": warmup,
                        "measure_runs": runs,
                        "p5_ms": f"{stats['p5_ms']:.4f}",
                        "p50_ms": f"{stats['p50_ms']:.4f}",
                        "p95_ms": f"{stats['p95_ms']:.4f}",
                        "mean_ms": f"{stats['mean_ms']:.4f}",
                        "std_ms": f"{stats['std_ms']:.4f}",
                        "throughput_ips": f"{stats['throughput_ips']:.2f}",
                        "engine_size_MB": f"{engine_mb:.2f}",
                        "pct_under_33_33ms": f"{stats['pct_under_33_33ms']:.2f}",
                        "pct_under_40ms": f"{stats['pct_under_40ms']:.2f}",
                        "checkpoint_domain": domain,
                        "onnx_path": str(onnx_path),
                        "engine_path": str(engine_path),
                    }
                )
                print(
                    f"[OK] TRT {config_key} | BS={bs:<2d} | p50={stats['p50_ms']:.2f}ms "
                    f"p95={stats['p95_ms']:.2f}ms IPS={stats['throughput_ips']:.1f}"
                )

            # 2. End-to-End 1 cropped BGR image -> Resize+Norm -> H2D -> TRT -> D2H -> Top-10 match (8,000 gallery)
            feat_dim = int(runner.infer(torch.randn(1, 3, img_size, img_size, device=device)).shape[1])
            torch.cuda.synchronize(device)
            rng_g = np.random.default_rng(123)
            gallery_bank = rng_g.standard_normal((gallery_n, feat_dim), dtype=np.float32)
            gallery_bank /= np.linalg.norm(gallery_bank, axis=1, keepdims=True) + 1e-12

            def _run_e2e_once() -> Dict[str, float]:
                t0 = time.perf_counter()
                rgb = cv2.cvtColor(
                    cv2.resize(crop_bgr, (img_size, img_size), interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2RGB,
                )
                arr = (rgb.astype(np.float32) / 255.0 - mean_np) / std_np
                chw = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])
                inp_cpu = torch.from_numpy(chw)
                t1 = time.perf_counter()

                inp_gpu = inp_cpu.to(device)
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()

                out_gpu = runner.infer(inp_gpu)
                torch.cuda.synchronize(device)
                t3 = time.perf_counter()

                emb = out_gpu.float().cpu().numpy()[0]
                t4 = time.perf_counter()

                emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
                sims = gallery_bank @ emb_norm
                top_idx = np.argpartition(sims, -topk)[-topk:]
                _ = top_idx[np.argsort(-sims[top_idx])]
                t5 = time.perf_counter()

                return {
                    "preprocess_ms": (t1 - t0) * 1000.0,
                    "h2d_ms": (t2 - t1) * 1000.0,
                    "trt_infer_ms": (t3 - t2) * 1000.0,
                    "d2h_ms": (t4 - t3) * 1000.0,
                    "match_top10_ms": (t5 - t4) * 1000.0,
                    "e2e_total_ms": (t5 - t0) * 1000.0,
                }

            for _ in range(warmup):
                _ = _run_e2e_once()

            stage_samples: Dict[str, List[float]] = {
                k: [] for k in ("preprocess_ms", "h2d_ms", "trt_infer_ms", "d2h_ms", "match_top10_ms", "e2e_total_ms")
            }
            for idx in range(1, runs + 1):
                m = _run_e2e_once()
                for k, v in m.items():
                    stage_samples[k].append(v)
                e2e_raw_rows.append(
                    {"config": config_key, "iter": idx, **{k: f"{v:.6f}" for k, v in m.items()}}
                )

            tot_stats = summarize_latencies(stage_samples["e2e_total_ms"], batch_size=1)
            e2e_row = {
                "config": config_key,
                "display_name": MODEL_DISPLAY_NAMES[config_key],
                "gallery_size": gallery_n,
                "topk": topk,
                "preprocess_p50_ms": f"{np.percentile(stage_samples['preprocess_ms'], 50):.4f}",
                "h2d_p50_ms": f"{np.percentile(stage_samples['h2d_ms'], 50):.4f}",
                "trt_infer_p50_ms": f"{np.percentile(stage_samples['trt_infer_ms'], 50):.4f}",
                "d2h_p50_ms": f"{np.percentile(stage_samples['d2h_ms'], 50):.4f}",
                "match_top10_p50_ms": f"{np.percentile(stage_samples['match_top10_ms'], 50):.4f}",
                "e2e_p5_ms": f"{tot_stats['p5_ms']:.4f}",
                "e2e_p50_ms": f"{tot_stats['p50_ms']:.4f}",
                "e2e_p95_ms": f"{tot_stats['p95_ms']:.4f}",
                "e2e_mean_ms": f"{tot_stats['mean_ms']:.4f}",
                "e2e_std_ms": f"{tot_stats['std_ms']:.4f}",
                "e2e_fps": f"{tot_stats['throughput_ips']:.2f}",
                "pct_under_33_33ms": f"{tot_stats['pct_under_33_33ms']:.2f}",
                "pct_under_40ms": f"{tot_stats['pct_under_40ms']:.2f}",
                "crop_source": crop_source,
            }
            e2e_summary_rows.append(e2e_row)
            print(
                f"[OK] E2E {config_key} | p50={tot_stats['p50_ms']:.2f}ms "
                f"(prep={e2e_row['preprocess_p50_ms']}ms trt={e2e_row['trt_infer_p50_ms']}ms "
                f"match={e2e_row['match_top10_p50_ms']}ms) | FPS={tot_stats['throughput_ips']:.1f}"
            )
        finally:
            runner.close()
            reset_gpu_memory(device)

    hw_after = {"nvpmodel": read_nvpmodel_status(), "thermals_c": read_jetson_thermals()}
    write_csv(trt_raw_rows, out_dir / "latency_tensorrt_raw.csv")
    write_csv(trt_summary_rows, out_dir / "latency_tensorrt_summary.csv")
    write_csv(e2e_raw_rows, out_dir / "latency_e2e_raw.csv")
    write_csv(e2e_summary_rows, out_dir / "latency_e2e_summary.csv")
    write_json(
        {
            "hardware_before": hw_before,
            "hardware_after": hw_after,
            "parity_checks": parity_checks,
            "tensorrt_summary": trt_summary_rows,
        },
        out_dir / "latency_tensorrt_summary.json",
    )
    write_json(
        {
            "hardware_before": hw_before,
            "hardware_after": hw_after,
            "crop_source": crop_source,
            "e2e_summary": e2e_summary_rows,
        },
        out_dir / "latency_e2e_summary.json",
    )


if __name__ == "__main__":
    main()
