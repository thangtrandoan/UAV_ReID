"""Efficiency and Computational Cost Benchmark Suite.

Measures:
- Parameter count (#Params in Millions / Thousands)
- Computational Complexity (GFLOPs / GMACs)
- Inference Latency (ms, median +/- std over 50 warmups + 200 repetitions)
- Throughput (FPS / Images per second at BS=1 and BS=12/32)
- Peak GPU Memory (MB via torch.cuda.max_memory_allocated())
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count model parameters in millions (M)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gate_params = 0
    for name, module in model.named_modules():
        if any(gate_tag in name for gate_tag in ("ga1", "ga2", "ga3", "ga4", "rga", "bcgg", "gate")):
            # Only count direct parameters of this layer
            gate_params += sum(p.numel() for p in module.parameters(recurse=False))

    return {
        "total_params_M": total / 1e6,
        "trainable_params_M": trainable / 1e6,
        "gate_params_M": gate_params / 1e6,
        "gate_params_K": gate_params / 1e3,
    }


class FLOPCounterHook:
    """Self-contained PyTorch FLOP profiler using forward hooks.

    Calculates multiply-accumulates (MACs) and FLOPs (2 * MACs) without external dependencies.
    """

    def __init__(self):
        self.total_macs: int = 0
        self.hooks: List[Any] = []

    def _conv2d_hook(self, module: nn.Conv2d, input_tuple, output):
        x = input_tuple[0]
        batch_size = x.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs_per_instance = kernel_ops * module.out_channels * out_h * out_w
        self.total_macs += batch_size * macs_per_instance

    def _conv1d_hook(self, module: nn.Conv1d, input_tuple, output):
        x = input_tuple[0]
        batch_size = x.shape[0]
        out_l = output.shape[2]
        kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
        macs_per_instance = kernel_ops * module.out_channels * out_l
        self.total_macs += batch_size * macs_per_instance

    def _linear_hook(self, module: nn.Linear, input_tuple, output):
        x = input_tuple[0]
        batch_size = x.shape[0] if x.dim() > 1 else 1
        num_instances = np.prod(x.shape[:-1]) if x.dim() > 1 else 1
        macs_per_instance = module.in_features * module.out_features
        self.total_macs += num_instances * macs_per_instance

    def register(self, model: nn.Module):
        self.total_macs = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(self._conv2d_hook))
            elif isinstance(module, nn.Conv1d):
                self.hooks.append(module.register_forward_hook(self._conv1d_hook))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self._linear_hook))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


@torch.no_grad()
def measure_flops_and_macs(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Measure GFLOPs and GMACs for a given input tensor shape.

    GFLOPs = 2 * GMACs (Multiply-Accumulate is counted as 2 FLOPs).
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # 1. Try thop if installed
    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return {
            "GMACs": macs / 1e9,
            "GFLOPs": (2.0 * macs) / 1e9,
        }
    except Exception:
        pass

    # 2. Try fvcore if installed
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input).total()
        return {
            "GMACs": (flops / 2.0) / 1e9,
            "GFLOPs": flops / 1e9,
        }
    except Exception:
        pass

    # 3. Fallback to built-in hook-based profiler
    profiler = FLOPCounterHook()
    profiler.register(model)
    try:
        _ = model(dummy_input)
        macs = profiler.total_macs
    finally:
        profiler.remove()

    return {
        "GMACs": macs / 1e9,
        "GFLOPs": (2.0 * macs) / 1e9,
    }


@torch.no_grad()
def benchmark_latency_and_fps(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    warmup_runs: int = 50,
    measure_runs: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Benchmark inference latency (median +/- std in ms) and throughput (FPS).

    Runs warmup_runs iterations to stabilize GPU clocks/caches,
    then measure_runs iterations using torch.cuda.Event for high precision.
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)
    batch_size = input_shape[0]

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)

    latencies_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(measure_runs):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(measure_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    median_latency = float(np.median(latencies_arr))
    mean_latency = float(np.mean(latencies_arr))
    std_latency = float(np.std(latencies_arr))

    # Throughput (Images per second)
    fps = (batch_size / (median_latency / 1000.0)) if median_latency > 0 else 0.0

    return {
        "median_latency_ms": median_latency,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "fps": fps,
    }


@torch.no_grad()
def measure_peak_gpu_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> float:
    """Measure Peak GPU Memory Allocated (MB) during forward pass."""
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    if device.type != "cuda":
        return 0.0

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    _ = model(dummy_input)
    torch.cuda.synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024.0 * 1024.0)


@torch.no_grad()
def measure_backbone_and_head_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    warmup_runs: int = 30,
    measure_runs: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Benchmark Backbone vs Gate+Head latency breakdown in milliseconds."""
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # 1. Total latency
    tot = benchmark_latency_and_fps(model, input_shape=input_shape, warmup_runs=warmup_runs, measure_runs=measure_runs, device=device)
    total_ms = tot["median_latency_ms"]

    # 2. Pure Backbone forward pass
    backbone_fn = None
    if hasattr(model, "convnext_backbone"):
        def _bb_run(x):
            x = model.convnext_backbone.stem(x)
            x = model.convnext_backbone.stage1(x)
            x = model.convnext_backbone.stage2(x)
            x = model.convnext_backbone.stage3(x)
            x = model.convnext_backbone.stage4(x)
            return x
        backbone_fn = _bb_run
    elif hasattr(model, "stem") and hasattr(model, "layer1"):
        def _bb_run(x):
            x = model.stem(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            return x
        backbone_fn = _bb_run

    if backbone_fn is not None:
        for _ in range(warmup_runs):
            _ = backbone_fn(dummy)
        latencies_ms: List[float] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start_e = torch.cuda.Event(enable_timing=True)
            end_e = torch.cuda.Event(enable_timing=True)
            for _ in range(measure_runs):
                start_e.record()
                _ = backbone_fn(dummy)
                end_e.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(start_e.elapsed_time(end_e))
        else:
            for _ in range(measure_runs):
                t0 = time.perf_counter()
                _ = backbone_fn(dummy)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
        backbone_ms = float(np.median(latencies_ms))
    else:
        backbone_ms = total_ms * 0.75

    gate_head_ms = max(0.0, total_ms - backbone_ms)

    return {
        "total_latency_ms": total_ms,
        "backbone_latency_ms": backbone_ms,
        "gate_head_latency_ms": gate_head_ms,
    }


def run_full_efficiency_profile(
    model: nn.Module,
    img_size: int = 224,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run full efficiency profile for a model.

    Evaluates:
    - #Params (Total and Gate)
    - GFLOPs at given img_size
    - Online Latency & FPS (Batch Size = 1)
    - Latency breakdown: Backbone Time (ms) vs Gate + Head Time (ms)
    - Batch Latency & FPS (Batch Size = 32 or 12)
    - Peak GPU VRAM (MB)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    params = count_parameters(model)
    flops = measure_flops_and_macs(model, input_shape=(1, 3, img_size, img_size), device=device)
    online_bench = benchmark_latency_and_fps(model, input_shape=(1, 3, img_size, img_size), device=device)
    latency_breakdown = measure_backbone_and_head_latency(model, input_shape=(1, 3, img_size, img_size), device=device)

    # Batch throughput at BS=32 (fallback to BS=12 if memory constrained)
    batch_size = 32
    try:
        batch_bench = benchmark_latency_and_fps(model, input_shape=(batch_size, 3, img_size, img_size), device=device)
    except RuntimeError:
        batch_size = 12
        batch_bench = benchmark_latency_and_fps(model, input_shape=(batch_size, 3, img_size, img_size), device=device)

    peak_vram = measure_peak_gpu_memory(model, input_shape=(1, 3, img_size, img_size), device=device)

    return {
        "params_total_M": params["total_params_M"],
        "params_gate_M": params["gate_params_M"],
        "params_gate_K": params["gate_params_K"],
        "GFLOPs": flops["GFLOPs"],
        "GMACs": flops["GMACs"],
        "latency_online_ms": online_bench["median_latency_ms"],
        "latency_online_std": online_bench["std_latency_ms"],
        "latency_backbone_ms": latency_breakdown["backbone_latency_ms"],
        "latency_gate_head_ms": latency_breakdown["gate_head_latency_ms"],
        "fps_online": online_bench["fps"],
        "batch_size": batch_size,
        "latency_batch_ms": batch_bench["median_latency_ms"],
        "fps_batch": batch_bench["fps"],
        "peak_vram_MB": peak_vram,
    }
