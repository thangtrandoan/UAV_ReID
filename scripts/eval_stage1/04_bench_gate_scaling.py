#!/usr/bin/env python3
"""Benchmark isolated Stage-3 gate blocks (RGA vs BCGG) across resolutions, BS, and precisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ROOT_DIR,
    count_module_params,
    load_eval_config,
    maybe_lock_jetson_clocks,
    measure_exact_flops_macs,
    reset_gpu_memory,
    resolve_output_dir,
    summarize_latencies,
    write_csv,
    write_json,
)
from uav_reid.gasnet import BottleneckContextGroupGate  # noqa: E402
from uav_reid.gasnet.model import RGABlock  # noqa: E402


def _build_gate_module(variant: str, channels: int, feat_h: int, feat_w: int) -> torch.nn.Module:
    if variant == "rga_fixed_14x14":
        mod = RGABlock(channels=channels, spatial_size=(14, 14)).eval()
        return mod
    if variant == "rga_adapted":
        mod = RGABlock(channels=channels, spatial_size=(feat_h, feat_w)).eval()
        return mod
    if variant in ("bcgg_unfolded", "bcgg_folded"):
        mod = BottleneckContextGroupGate(
            channels=channels,
            hidden=16,
            groups=4,
            kernel_size=7,
            channel_kernel=5,
            inject_value=False,
        ).eval()
        if variant == "bcgg_folded":
            mod = mod.to_deploy().eval()
        return mod
    raise ValueError(f"Unknown gate variant: {variant!r}")


@torch.no_grad()
def _bench_gate_cell(
    variant: str,
    channels: int,
    feat_h: int,
    feat_w: int,
    bs: int,
    precision: str,
    warmup: int,
    runs: int,
    device: torch.device,
) -> Dict[str, Any]:
    reset_gpu_memory(device)
    mod = _build_gate_module(variant, channels, feat_h, feat_w).to(device)
    params_info = count_module_params(mod)
    input_shape = (bs, channels, feat_h, feat_w)
    dummy = torch.randn(*input_shape, device=device)

    # Probe compatibility (catches fixed 14x14 RGA on 16x16 / 24x24)
    try:
        use_autocast = (device.type == "cuda") and (precision == "fp16")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            _ = mod(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except (ValueError, RuntimeError) as exc:
        del mod, dummy
        reset_gpu_memory(device)
        return {
            "params_K": f"{params_info['params_K']:.3f}",
            "GFLOPs": "N/A",
            "p50_ms": "N/A",
            "p95_ms": "N/A",
            "peak_vram_MiB": "N/A",
            "status": f"N/A ({type(exc).__name__})",
        }

    flops_info = measure_exact_flops_macs(mod, input_shape=input_shape, device=device)

    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            _ = mod(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        lats: List[float] = []
        for _ in range(runs):
            start_ev.record()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                _ = mod(dummy)
            end_ev.record()
            torch.cuda.synchronize(device)
            lats.append(float(start_ev.elapsed_time(end_ev)))
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    else:
        import time

        lats = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = mod(dummy)
            lats.append((time.perf_counter() - t0) * 1000.0)
        peak_vram = 0.0

    stats = summarize_latencies(lats, batch_size=bs)
    del mod, dummy
    reset_gpu_memory(device)
    return {
        "params_K": f"{params_info['params_K']:.3f}",
        "GFLOPs": f"{flops_info['GFLOPs']:.6f}",
        "p50_ms": f"{stats['p50_ms']:.4f}",
        "p95_ms": f"{stats['p95_ms']:.4f}",
        "peak_vram_MiB": f"{peak_vram:.2f}",
        "status": "OK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Gate Scaling Benchmark (RGA vs BCGG)")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lock-clocks", action="store_true")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "04_gate_scaling_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    gs_cfg = cfg.get("gate_scaling", {})
    channels = int(gs_cfg.get("channels", 384))
    resolutions = [tuple(int(x) for x in r) for r in gs_cfg.get("resolutions", [[224, 14, 14], [256, 16, 16], [384, 24, 24]])]
    batch_sizes = [int(b) for b in gs_cfg.get("batch_sizes", [1, 8])]
    precisions = [str(p).lower() for p in gs_cfg.get("precisions", ["fp32", "fp16"])]
    warmup = int(args.warmup if args.warmup is not None else gs_cfg.get("warmup_runs", 50))
    runs = int(args.runs if args.runs is not None else gs_cfg.get("measure_runs", 200))

    hw_info = maybe_lock_jetson_clocks(args.lock_clocks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = ("rga_fixed_14x14", "rga_adapted", "bcgg_unfolded", "bcgg_folded")
    rows: List[Dict[str, Any]] = []

    for img_res, feat_h, feat_w in resolutions:
        spatial_str = f"{feat_h}x{feat_w}"
        for precision in precisions:
            for bs in batch_sizes:
                for variant in variants:
                    res = _bench_gate_cell(
                        variant=variant,
                        channels=channels,
                        feat_h=feat_h,
                        feat_w=feat_w,
                        bs=bs,
                        precision=precision,
                        warmup=warmup,
                        runs=runs,
                        device=device,
                    )
                    row = {
                        "spatial_size": spatial_str,
                        "precision": precision.upper(),
                        "batch_size": bs,
                        "gate_type": variant,
                        "params_K": res["params_K"],
                        "GFLOPs": res["GFLOPs"],
                        "p50_ms": res["p50_ms"],
                        "p95_ms": res["p95_ms"],
                        "peak_vram_MiB": res["peak_vram_MiB"],
                        "img_resolution": f"{img_res}x{img_res}",
                        "status": res["status"],
                    }
                    rows.append(row)
                    print(
                        f"[{res['status']}] {spatial_str} | {precision.upper()} | BS={bs} | {variant:<16s} | "
                        f"Params={res['params_K']}K GFLOPs={res['GFLOPs']} p50={res['p50_ms']}ms VRAM={res['peak_vram_MiB']}"
                    )

    write_csv(rows, out_dir / "gate_scaling_summary.csv")
    write_json({"hardware": hw_info, "channels": channels, "rows": rows}, out_dir / "gate_scaling_summary.json")


if __name__ == "__main__":
    main()
