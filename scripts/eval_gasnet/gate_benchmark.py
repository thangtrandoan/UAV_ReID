"""Gate Module Detailed Benchmark (Table 5.3).

Compares isolated Gate modules:
- RGA (Relation-Aware Global Attention)
- BCGG (Bottleneck Context Group Gate - Train mode)
- BCGG (Deploy mode - RepDW Folded)
Across different input image resolutions: 224x224, 256x256, 384x384.

Demonstrates:
- RGA's fixed-size spatial limitation (fails on mismatched resolutions)
- RGA's quadratic memory O((HW)^2) and latency explosion when adapted
- BCGG's resolution independence, linear complexity, and deploy fusion speedup
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from uav_reid.gasnet.bcgg import BottleneckContextGroupGate
from uav_reid.gasnet.model import _make_gate_block

from .efficiency import (
    benchmark_latency_and_fps,
    count_parameters,
    measure_flops_and_macs,
    measure_peak_gpu_memory,
)


def benchmark_single_gate(
    gate_module: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
    warmup_runs: int = 30,
    measure_runs: int = 100,
) -> Dict[str, Any]:
    """Profile an isolated gate module on a feature tensor."""
    gate_module = gate_module.to(device).eval()
    params = count_parameters(gate_module)

    # Measure FLOPs
    flops_info = measure_flops_and_macs(gate_module, input_shape=input_shape, device=device)

    # Measure Latency
    latency_info = benchmark_latency_and_fps(
        gate_module,
        input_shape=input_shape,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        device=device,
    )

    # Measure Peak VRAM
    vram = measure_peak_gpu_memory(gate_module, input_shape=input_shape, device=device)

    return {
        "params_K": params["total_params_M"] * 1000.0,
        "GFLOPs": flops_info["GFLOPs"],
        "latency_ms": latency_info["median_latency_ms"],
        "latency_std": latency_info["std_latency_ms"],
        "peak_vram_MB": vram,
    }


def run_table_5_3_benchmark(
    channels: int = 384,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Run full Table 5.3 Gate Module comparison across resolutions 224, 256, 384."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolutions = [
        (224, 14, 14),  # 224x224 img -> stage 3 feat is 14x14
        (256, 16, 16),  # 256x256 img -> stage 3 feat is 16x16
        (384, 24, 24),  # 384x384 img -> stage 3 feat is 24x24
    ]

    results: List[Dict[str, Any]] = []

    # 1. Base RGA (configured for 224x224, i.e., 14x14)
    rga_224 = _make_gate_block("rga", channels, (14, 14)).to(device).eval()

    # 2. BCGG instance (resolution-free)
    bcgg_train = BottleneckContextGroupGate(
        channels=channels,
        hidden=16,
        groups=4,
        kernel_size=7,
        channel_kernel=5,
    ).to(device).eval()

    bcgg_deploy = bcgg_train.to_deploy().to(device).eval()

    print("\n" + "=" * 90)
    print(f"BẢNG 5.3: SO SÁNH CHI PHÍ TÍNH TOÁN RGA vs BCGG (Channels={channels})")
    print("=" * 90)
    print(f"{'Gate Architecture':<32} {'Resolution':<12} {'#Params (K)':<12} {'GFLOPs':<10} {'Latency (ms)':<16} {'Peak VRAM (MB)':<14} {'Status':<15}")
    print("-" * 115)

    for img_res, feat_h, feat_w in resolutions:
        input_shape = (1, channels, feat_h, feat_w)
        res_str = f"{img_res}x{img_res}"

        # --- A. RGA (Static 14x14) ---
        if (feat_h, feat_w) == (14, 14):
            bench_rga = benchmark_single_gate(rga_224, input_shape, device)
            row_rga = {
                "gate": "RGA (Fixed 224)",
                "resolution": res_str,
                "params_K": bench_rga["params_K"],
                "GFLOPs": bench_rga["GFLOPs"],
                "latency_ms": bench_rga["latency_ms"],
                "latency_std": bench_rga["latency_std"],
                "vram_MB": bench_rga["peak_vram_MB"],
                "status": "OK",
            }
        else:
            # Test RGA failure with mismatched dimensions
            try:
                dummy = torch.randn(*input_shape, device=device)
                _ = rga_224(dummy)
                status = "Passed"
            except ValueError as e:
                status = "FAILED (Fixed Size)"
            row_rga = {
                "gate": "RGA (Fixed 224)",
                "resolution": res_str,
                "params_K": count_parameters(rga_224)["total_params_M"] * 1000.0,
                "GFLOPs": float("nan"),
                "latency_ms": float("nan"),
                "latency_std": float("nan"),
                "vram_MB": float("nan"),
                "status": status,
            }
        results.append(row_rga)
        print(f"{row_rga['gate']:<32} {row_rga['resolution']:<12} {row_rga['params_K']:<12.1f} {row_rga['GFLOPs']:<10.4f} {row_rga['latency_ms']:>6.2f} +/- {row_rga['latency_std']:<6.2f} {row_rga['vram_MB']:<14.1f} {row_rga['status']:<15}")

        # --- B. RGA (Re-instantiated / Adapted for this resolution) ---
        rga_adapted = _make_gate_block("rga", channels, (feat_h, feat_w)).to(device).eval()
        bench_rga_adapt = benchmark_single_gate(rga_adapted, input_shape, device)
        row_rga_adapt = {
            "gate": "RGA (Adapted resolution)",
            "resolution": res_str,
            "params_K": bench_rga_adapt["params_K"],
            "GFLOPs": bench_rga_adapt["GFLOPs"],
            "latency_ms": bench_rga_adapt["latency_ms"],
            "latency_std": bench_rga_adapt["latency_std"],
            "vram_MB": bench_rga_adapt["peak_vram_MB"],
            "status": "O(N^2) Explosion",
        }
        results.append(row_rga_adapt)
        print(f"{row_rga_adapt['gate']:<32} {row_rga_adapt['resolution']:<12} {row_rga_adapt['params_K']:<12.1f} {row_rga_adapt['GFLOPs']:<10.4f} {row_rga_adapt['latency_ms']:>6.2f} +/- {row_rga_adapt['latency_std']:<6.2f} {row_rga_adapt['vram_MB']:<14.1f} {row_rga_adapt['status']:<15}")

        # --- C. BCGG (Train mode) ---
        bench_bcgg = benchmark_single_gate(bcgg_train, input_shape, device)
        row_bcgg = {
            "gate": "BCGG (Train mode)",
            "resolution": res_str,
            "params_K": bench_bcgg["params_K"],
            "GFLOPs": bench_bcgg["GFLOPs"],
            "latency_ms": bench_bcgg["latency_ms"],
            "latency_std": bench_bcgg["latency_std"],
            "vram_MB": bench_bcgg["peak_vram_MB"],
            "status": "Adaptive O(N)",
        }
        results.append(row_bcgg)
        print(f"{row_bcgg['gate']:<32} {row_bcgg['resolution']:<12} {row_bcgg['params_K']:<12.1f} {row_bcgg['GFLOPs']:<10.4f} {row_bcgg['latency_ms']:>6.2f} +/- {row_bcgg['latency_std']:<6.2f} {row_bcgg['vram_MB']:<14.1f} {row_bcgg['status']:<15}")

        # --- D. BCGG (Deploy mode - RepDW Folded) ---
        bench_deploy = benchmark_single_gate(bcgg_deploy, input_shape, device)
        row_deploy = {
            "gate": "BCGG (Deploy - RepDW Folded)",
            "resolution": res_str,
            "params_K": bench_deploy["params_K"],
            "GFLOPs": bench_deploy["GFLOPs"],
            "latency_ms": bench_deploy["latency_ms"],
            "latency_std": bench_deploy["latency_std"],
            "vram_MB": bench_deploy["peak_vram_MB"],
            "status": "Optimal Fused",
        }
        results.append(row_deploy)
        print(f"{row_deploy['gate']:<32} {row_deploy['resolution']:<12} {row_deploy['params_K']:<12.1f} {row_deploy['GFLOPs']:<10.4f} {row_deploy['latency_ms']:>6.2f} +/- {row_deploy['latency_std']:<6.2f} {row_deploy['vram_MB']:<14.1f} {row_deploy['status']:<15}")
        print("-" * 115)

    return results
