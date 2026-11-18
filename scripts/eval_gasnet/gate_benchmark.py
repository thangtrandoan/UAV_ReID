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
                "Kiến trúc Gate": "RGA (Cố định 224x224)",
                "Độ phân giải": res_str,
                "Feature Map": f"{feat_h}x{feat_w}",
                "#Params (K)": f"{bench_rga['params_K']:.1f}",
                "GFLOPs": f"{bench_rga['GFLOPs']:.4f}",
                "Latency (ms)": f"{bench_rga['latency_ms']:.2f} ± {bench_rga['latency_std']:.2f}",
                "Peak VRAM (MB)": f"{bench_rga['peak_vram_MB']:.1f}",
                "Đặc tính / Khả năng mở rộng": "OK (Baseline)",
            }
        else:
            # Test RGA failure with mismatched dimensions
            try:
                dummy = torch.randn(*input_shape, device=device)
                _ = rga_224(dummy)
                status = "Passed"
            except (ValueError, RuntimeError) as e:
                status = "LỖI (Crash cố định kích thước)"
            row_rga = {
                "Kiến trúc Gate": "RGA (Cố định 224x224)",
                "Độ phân giải": res_str,
                "Feature Map": f"{feat_h}x{feat_w}",
                "#Params (K)": f"{count_parameters(rga_224)['total_params_M'] * 1000.0:.1f}",
                "GFLOPs": "N/A",
                "Latency (ms)": "N/A",
                "Peak VRAM (MB)": "N/A",
                "Đặc tính / Khả năng mở rộng": status,
            }
        results.append(row_rga)
        print(f"{row_rga['Kiến trúc Gate']:<28} {row_rga['Độ phân giải']:<10} {row_rga['#Params (K)']:<12} {row_rga['GFLOPs']:<10} {row_rga['Latency (ms)']:<18} {row_rga['Peak VRAM (MB)']:<14} {row_rga['Đặc tính / Khả năng mở rộng']:<25}")

        # --- B. RGA (Re-instantiated / Adapted for this resolution) ---
        rga_adapted = _make_gate_block("rga", channels, (feat_h, feat_w)).to(device).eval()
        bench_rga_adapt = benchmark_single_gate(rga_adapted, input_shape, device)
        row_rga_adapt = {
            "Kiến trúc Gate": "RGA (Adapted theo độ phân giải)",
            "Độ phân giải": res_str,
            "Feature Map": f"{feat_h}x{feat_w}",
            "#Params (K)": f"{bench_rga_adapt['params_K']:.1f}",
            "GFLOPs": f"{bench_rga_adapt['GFLOPs']:.4f}",
            "Latency (ms)": f"{bench_rga_adapt['latency_ms']:.2f} ± {bench_rga_adapt['latency_std']:.2f}",
            "Peak VRAM (MB)": f"{bench_rga_adapt['peak_vram_MB']:.1f}",
            "Đặc tính / Khả năng mở rộng": "Bùng nổ bộ nhớ O((HW)^2)",
        }
        results.append(row_rga_adapt)
        print(f"{row_rga_adapt['Kiến trúc Gate']:<28} {row_rga_adapt['Độ phân giải']:<10} {row_rga_adapt['#Params (K)']:<12} {row_rga_adapt['GFLOPs']:<10} {row_rga_adapt['Latency (ms)']:<18} {row_rga_adapt['Peak VRAM (MB)']:<14} {row_rga_adapt['Đặc tính / Khả năng mở rộng']:<25}")

        # --- C. BCGG (Train mode) ---
        bench_bcgg = benchmark_single_gate(bcgg_train, input_shape, device)
        row_bcgg = {
            "Kiến trúc Gate": "BCGG (Train mode)",
            "Độ phân giải": res_str,
            "Feature Map": f"{feat_h}x{feat_w}",
            "#Params (K)": f"{bench_bcgg['params_K']:.1f}",
            "GFLOPs": f"{bench_bcgg['GFLOPs']:.4f}",
            "Latency (ms)": f"{bench_bcgg['latency_ms']:.2f} ± {bench_bcgg['latency_std']:.2f}",
            "Peak VRAM (MB)": f"{bench_bcgg['peak_vram_MB']:.1f}",
            "Đặc tính / Khả năng mở rộng": "Thích ứng linh hoạt O(HW)",
        }
        results.append(row_bcgg)
        print(f"{row_bcgg['Kiến trúc Gate']:<28} {row_bcgg['Độ phân giải']:<10} {row_bcgg['#Params (K)']:<12} {row_bcgg['GFLOPs']:<10} {row_bcgg['Latency (ms)']:<18} {row_bcgg['Peak VRAM (MB)']:<14} {row_bcgg['Đặc tính / Khả năng mở rộng']:<25}")

        # --- D. BCGG (Deploy mode - RepDW Folded) ---
        bench_deploy = benchmark_single_gate(bcgg_deploy, input_shape, device)
        row_deploy = {
            "Kiến trúc Gate": "BCGG (Deploy - RepDW Folded)",
            "Độ phân giải": res_str,
            "Feature Map": f"{feat_h}x{feat_w}",
            "#Params (K)": f"{bench_deploy['params_K']:.1f}",
            "GFLOPs": f"{bench_deploy['GFLOPs']:.4f}",
            "Latency (ms)": f"{bench_deploy['latency_ms']:.2f} ± {bench_deploy['latency_std']:.2f}",
            "Peak VRAM (MB)": f"{bench_deploy['peak_vram_MB']:.1f}",
            "Đặc tính / Khả năng mở rộng": "Tối ưu hợp nhất (Zero-overhead)",
        }
        results.append(row_deploy)
        print(f"{row_deploy['Kiến trúc Gate']:<28} {row_deploy['Độ phân giải']:<10} {row_deploy['#Params (K)']:<12} {row_deploy['GFLOPs']:<10} {row_deploy['Latency (ms)']:<18} {row_deploy['Peak VRAM (MB)']:<14} {row_deploy['Đặc tính / Khả năng mở rộng']:<25}")
        print("-" * 115)

    return results
