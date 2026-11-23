#!/usr/bin/env python3
"""Benchmark PyTorch FP32 and FP16 latency, throughput, and peak VRAM on Jetson AGX Orin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
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


@torch.no_grad()
def _time_callable_cuda(
    fn,
    dummy: torch.Tensor,
    precision: str,
    warmup: int,
    runs: int,
    device: torch.device,
) -> Tuple[List[float], float]:
    use_autocast = (device.type == "cuda") and (precision == "fp16")
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            _ = fn(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies: List[float] = []
    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            start_ev.record()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                _ = fn(dummy)
            end_ev.record()
            torch.cuda.synchronize(device)
            latencies.append(float(start_ev.elapsed_time(end_ev)))
        peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    else:
        import time

        for _ in range(runs):
            t0 = time.perf_counter()
            _ = fn(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        peak_vram_mib = 0.0
    return latencies, peak_vram_mib


def _make_pure_backbone_fn(model: torch.nn.Module):
    if hasattr(model, "convnext_backbone"):
        bb = model.convnext_backbone

        def _fn(x: torch.Tensor) -> torch.Tensor:
            x = bb.stem(x)
            x = bb.stage1(x)
            x = bb.stage2(x)
            x = bb.stage3(x)
            x = bb.stage4(x)
            return x

        return _fn
    if hasattr(model, "stem") and hasattr(model, "layer1"):

        def _fn(x: torch.Tensor) -> torch.Tensor:
            x = model.stem(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            return x

        return _fn
    raise RuntimeError(f"[FAIL] Unsupported model layout for backbone breakdown: {type(model)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Benchmark PyTorch Latency on Jetson Orin")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lock-clocks", action="store_true", help="Run sudo nvpmodel -m 0 && sudo jetson_clocks")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs-small", type=int, default=None)
    parser.add_argument("--runs-large", type=int, default=None)
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "03_efficiency_and_latency_orin"
    out_dir.mkdir(parents=True, exist_ok=True)

    eff_cfg = cfg.get("efficiency", {})
    img_size = int(cfg.get("preprocess", {}).get("img_size", 224))
    batch_sizes = [int(b) for b in eff_cfg.get("batch_sizes", [1, 2, 4, 8, 32])]
    precisions = [str(p).lower() for p in eff_cfg.get("precisions", ["fp32", "fp16"])]
    warmup = int(args.warmup if args.warmup is not None else eff_cfg.get("warmup_runs", 50))
    runs_small = int(
        args.runs_small if args.runs_small is not None else eff_cfg.get("measure_runs_small_bs", 200)
    )
    runs_large = int(
        args.runs_large if args.runs_large is not None else eff_cfg.get("measure_runs_large_bs", 100)
    )

    hw_before = maybe_lock_jetson_clocks(args.lock_clocks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("resnet50ibn_rga", "resnet50ibn_rga", False),
        ("convnext_rga", "convnext_rga", False),
        ("convnext_bcgg_unfolded", "convnext_bcgg", False),
        ("convnext_bcgg_folded", "convnext_bcgg", True),
    ]

    summary_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []

    for config_key, model_key, deploy in configs:
        domain, ckpt_path = resolve_efficiency_checkpoint(cfg, model_key)

        for precision in precisions:
            for bs in batch_sizes:
                runs = runs_large if bs >= 32 else runs_small
                reset_gpu_memory(device)
                model, ckpt_info = load_gasnet_checkpoint(
                    ckpt_path, expected_model_key=model_key, deploy=deploy, device=device
                )
                if config_key == "resnet50ibn_rga":
                    assert model.backbone_type == "resnet50_ibn"

                dummy = torch.randn(bs, 3, img_size, img_size, device=device)
                latencies, peak_vram_mib = _time_callable_cuda(
                    model, dummy, precision=precision, warmup=warmup, runs=runs, device=device
                )
                stats = summarize_latencies(latencies, batch_size=bs)

                for idx, val in enumerate(latencies, start=1):
                    raw_rows.append(
                        {
                            "config": config_key,
                            "precision": precision.upper(),
                            "batch_size": bs,
                            "iter": idx,
                            "latency_ms": f"{val:.6f}",
                        }
                    )

                bb_p50 = ""
                gfh_p50 = ""
                if bs == 1:
                    bb_fn = _make_pure_backbone_fn(model)
                    bb_lats, _ = _time_callable_cuda(
                        bb_fn, dummy, precision=precision, warmup=warmup, runs=runs, device=device
                    )
                    bb_med = float(np.percentile(np.asarray(bb_lats, dtype=np.float64), 50))
                    gfh_med = max(0.0, stats["p50_ms"] - bb_med)
                    bb_p50 = f"{bb_med:.4f}"
                    gfh_p50 = f"{gfh_med:.4f}"
                    breakdown_rows.append(
                        {
                            "config": config_key,
                            "display_name": MODEL_DISPLAY_NAMES[config_key],
                            "precision": precision.upper(),
                            "total_p50_ms": f"{stats['p50_ms']:.4f}",
                            "backbone_p50_ms": bb_p50,
                            "gates_fs_head_p50_ms": gfh_p50,
                            "gates_fs_head_pct": f"{(gfh_med / stats['p50_ms'] * 100.0) if stats['p50_ms'] > 0 else 0.0:.2f}",
                        }
                    )

                summary_rows.append(
                    {
                        "config": config_key,
                        "display_name": MODEL_DISPLAY_NAMES[config_key],
                        "precision": precision.upper(),
                        "batch_size": bs,
                        "warmup_runs": warmup,
                        "measure_runs": runs,
                        "p5_ms": f"{stats['p5_ms']:.4f}",
                        "p50_ms": f"{stats['p50_ms']:.4f}",
                        "p95_ms": f"{stats['p95_ms']:.4f}",
                        "mean_ms": f"{stats['mean_ms']:.4f}",
                        "std_ms": f"{stats['std_ms']:.4f}",
                        "throughput_ips": f"{stats['throughput_ips']:.2f}",
                        "peak_vram_MiB": f"{peak_vram_mib:.2f}",
                        "pct_under_33_33ms": f"{stats['pct_under_33_33ms']:.2f}",
                        "pct_under_40ms": f"{stats['pct_under_40ms']:.2f}",
                        "backbone_p50_ms": bb_p50,
                        "gates_fs_head_p50_ms": gfh_p50,
                        "checkpoint_domain": domain,
                        "checkpoint_sha256": ckpt_info["sha256"],
                    }
                )
                print(
                    f"[OK] {config_key} | {precision.upper()} | BS={bs:<2d} | "
                    f"p50={stats['p50_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms "
                    f"IPS={stats['throughput_ips']:.1f} VRAM={peak_vram_mib:.1f}MiB"
                )
                del model, dummy
                reset_gpu_memory(device)

    hw_after = {"nvpmodel": read_nvpmodel_status(), "thermals_c": read_jetson_thermals()}
    write_csv(raw_rows, out_dir / "latency_pytorch_raw.csv")
    write_csv(summary_rows, out_dir / "latency_pytorch_summary.csv")
    write_csv(breakdown_rows, out_dir / "latency_pytorch_breakdown_bs1.csv")
    write_json(
        {
            "hardware_before": hw_before,
            "hardware_after": hw_after,
            "summary": summary_rows,
            "breakdown_bs1": breakdown_rows,
        },
        out_dir / "latency_pytorch_summary.json",
    )


if __name__ == "__main__":
    main()
