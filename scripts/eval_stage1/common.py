"""Shared helpers for Stage 1 (GASNet) Jetson AGX Orin evaluation."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uav_reid.gasnet import (  # noqa: E402
    GASNet,
    infer_backbone_name,
    infer_econvnext,
    infer_gate_config,
)
from uav_reid.gasnet.model import RGAChannel, RGASpatial  # noqa: E402
from uav_reid.utils import normalize_state_dict  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "eval_config.yaml"

MODEL_KEYS = ("resnet50ibn_rga", "convnext_rga", "convnext_bcgg")
MODEL_DISPLAY_NAMES = {
    "resnet50ibn_rga": "ResNet50-IBN + RGA",
    "convnext_rga": "ConvNeXt + RGA",
    "convnext_bcgg": "ConvNeXt + BCGG",
    "convnext_bcgg_unfolded": "ConvNeXt + BCGG (Unfolded)",
    "convnext_bcgg_folded": "ConvNeXt + BCGG (Folded)",
}
EXPECTED_ARCH = {
    "resnet50ibn_rga": ("resnet50_ibn", "rga"),
    "convnext_rga": ("dinov3_convnext", "rga"),
    "convnext_bcgg": ("dinov3_convnext", "bcgg"),
}


def resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def load_eval_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve() if (ROOT_DIR / path).exists() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"[FAIL] Evaluation config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_output_dir(cfg: Dict[str, Any], cli_output_dir: str | Path | None = None) -> Path:
    raw = cli_output_dir if cli_output_dir is not None else cfg.get("output_dir", "eval_results_stage1")
    out_dir = Path(raw)
    if not out_dir.is_absolute():
        out_dir = (ROOT_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reset_gpu_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        else:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def _run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return -1, "", str(exc)


def read_nvpmodel_status() -> Dict[str, Any]:
    rc, out, err = _run_cmd(["nvpmodel", "-q"])
    mode_line = ""
    for line in (out + "\n" + err).splitlines():
        if "Power Mode" in line:
            mode_line = line.strip()
            break
    return {
        "returncode": rc,
        "power_mode": mode_line or (out.splitlines()[0] if out else "unknown"),
        "raw_output": out,
        "raw_error": err,
    }


def read_jetson_thermals() -> Dict[str, float]:
    temps: Dict[str, float] = {}
    base = Path("/sys/devices/virtual/thermal")
    if not base.is_dir():
        return temps
    for zone in sorted(base.glob("thermal_zone*")):
        type_file = zone / "type"
        temp_file = zone / "temp"
        if not (type_file.is_file() and temp_file.is_file()):
            continue
        try:
            name = type_file.read_text(encoding="utf-8").strip()
            raw_val = temp_file.read_text(encoding="utf-8").strip()
            if raw_val:
                temps[name] = round(float(raw_val) / 1000.0, 2)
        except Exception:
            continue
    return temps


def maybe_lock_jetson_clocks(lock_clocks: bool) -> Dict[str, Any]:
    if lock_clocks:
        print("[INFO] Locking Jetson Orin clocks: sudo nvpmodel -m 0 && sudo jetson_clocks")
        rc1, out1, err1 = _run_cmd(["sudo", "-n", "nvpmodel", "-m", "0"], timeout=15)
        if rc1 != 0:
            print(f"[WARNING] `sudo -n nvpmodel -m 0` failed (rc={rc1}): {err1 or out1}")
        else:
            print("[OK] nvpmodel set to mode 0 (MAXN)")

        rc2, out2, err2 = _run_cmd(["sudo", "-n", "jetson_clocks"], timeout=15)
        if rc2 != 0:
            print(f"[WARNING] `sudo -n jetson_clocks` failed (rc={rc2}): {err2 or out2}")
        else:
            print("[OK] jetson_clocks enabled")
    else:
        print("[INFO] `--lock-clocks` not passed; recording current Jetson power mode and thermals.")

    nvp = read_nvpmodel_status()
    thermals = read_jetson_thermals()
    print(f"[INFO] Power mode: {nvp.get('power_mode')} | Thermals (C): {thermals}")
    return {"lock_requested": bool(lock_clocks), "nvpmodel": nvp, "thermals_c": thermals}


def collect_env_info() -> Dict[str, Any]:
    device_model = "unknown"
    dt_model = Path("/proc/device-tree/model")
    if dt_model.is_file():
        try:
            device_model = dt_model.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

    tegra_release = ""
    nv_rel = Path("/etc/nv_tegra_release")
    if nv_rel.is_file():
        tegra_release = nv_rel.read_text(encoding="utf-8", errors="ignore").strip()

    trt_ver = "missing"
    try:
        import tensorrt as trt

        trt_ver = str(trt.__version__)
    except Exception as exc:
        trt_ver = f"unavailable ({exc})"

    _, git_sha, _ = _run_cmd(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"])
    _, git_branch, _ = _run_cmd(["git", "-C", str(ROOT_DIR), "rev-parse", "--abbrev-ref", "HEAD"])
    _, git_status, _ = _run_cmd(["git", "-C", str(ROOT_DIR), "status", "--porcelain"])

    gpu_info: Dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_vram_MiB": round(props.total_memory / (1024.0 * 1024.0), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": int(props.multi_processor_count),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )

    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "device_tree_model": device_model,
        "l4t_tegra_release": tegra_release,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "tensorrt_version": trt_ver,
        "trtexec_exists": Path("/usr/src/tensorrt/bin/trtexec").is_file(),
        "gpu": gpu_info,
        "nvpmodel": read_nvpmodel_status(),
        "thermals_c": read_jetson_thermals(),
        "git": {
            "branch": git_branch,
            "commit_sha": git_sha,
            "dirty": bool(git_status),
        },
    }


def _unwrap_state_dict(raw_ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw_ckpt, dict):
        if "model_state_dict" in raw_ckpt and isinstance(raw_ckpt["model_state_dict"], dict):
            raw_ckpt = raw_ckpt["model_state_dict"]
        elif "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
            raw_ckpt = raw_ckpt["state_dict"]
    if not isinstance(raw_ckpt, dict):
        raise ValueError("Unsupported checkpoint format: expected dict/state_dict")
    return normalize_state_dict(raw_ckpt)


def _infer_ckpt_metadata(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    backbone = infer_backbone_name(state_dict, default="resnet50")
    econvnext = infer_econvnext(state_dict)
    gate_cfg = infer_gate_config(state_dict)
    cls_w = state_dict.get("classifier_global.weight")
    num_classes = int(cls_w.shape[0]) if cls_w is not None else 1000
    use_part_branch = any(k.startswith("part_reduce.") for k in state_dict)
    use_attention_local = any(k.startswith("attn_local.") for k in state_dict)
    return {
        "backbone": backbone,
        "econvnext": econvnext,
        "gate_cfg": gate_cfg,
        "num_classes": num_classes,
        "use_part_branch": use_part_branch,
        "use_attention_local": use_attention_local,
    }


def inspect_checkpoint(
    ckpt_path: Path,
    expected_model_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """Inspect a Stage 1 checkpoint and verify architecture/key compatibility."""
    if not ckpt_path.is_file():
        return (
            {
                "path": str(ckpt_path),
                "exists": False,
                "status": "MISSING",
                "sha256": None,
                "size_MB": None,
            },
            None,
        )

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(raw)
    meta = _infer_ckpt_metadata(state_dict)
    gate_cfg = meta["gate_cfg"]
    gate = gate_cfg["gate"]
    backbone = meta["backbone"]

    if expected_model_key in EXPECTED_ARCH:
        exp_bb, exp_gate = EXPECTED_ARCH[expected_model_key]
        if backbone != exp_bb:
            raise ValueError(
                f"[FAIL] Checkpoint {ckpt_path} has backbone={backbone!r}, expected {exp_bb!r} for {expected_model_key}"
            )
        if gate != exp_gate:
            raise ValueError(
                f"[FAIL] Checkpoint {ckpt_path} has gate={gate!r}, expected {exp_gate!r} for {expected_model_key}"
            )

    model = GASNet(
        num_classes=meta["num_classes"],
        use_pretrained=False,
        use_part_branch=meta["use_part_branch"],
        use_attention_local=meta["use_attention_local"],
        use_gem=True,
        backbone=backbone,
        econvnext=meta["econvnext"],
        **gate_cfg,
    )
    model_sd = model.state_dict()
    matched = {}
    shape_mismatches = []
    for k, v in state_dict.items():
        if k in model_sd:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                matched[k] = v
            else:
                shape_mismatches.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing_keys = [k for k in model_sd if k not in matched]
    unexpected_keys = [k for k in state_dict if k not in model_sd]
    strict_match = (len(missing_keys) == 0) and (len(unexpected_keys) == 0) and (len(shape_mismatches) == 0)

    non_cls_missing = [k for k in missing_keys if not k.startswith("classifier_")]
    if non_cls_missing or shape_mismatches:
        raise RuntimeError(
            f"[FAIL] Critical checkpoint incompatibility in {ckpt_path}: "
            f"non_cls_missing={non_cls_missing[:10]}, shape_mismatches={shape_mismatches[:5]}"
        )

    info = {
        "path": str(ckpt_path),
        "exists": True,
        "status": "OK",
        "sha256": sha256_file(ckpt_path),
        "size_MB": round(ckpt_path.stat().st_size / (1024.0 * 1024.0), 3),
        "backbone": backbone,
        "econvnext": meta["econvnext"],
        "gate": gate,
        "gate_hidden": gate_cfg.get("gate_hidden") if gate == "bcgg" else None,
        "gate_groups": gate_cfg.get("gate_groups") if gate == "bcgg" else None,
        "gate_kernel": gate_cfg.get("gate_kernel") if gate == "bcgg" else None,
        "gate_channel_kernel": gate_cfg.get("gate_channel_kernel") if gate == "bcgg" else None,
        "gate_inject_value": gate_cfg.get("gate_inject_value") if gate == "bcgg" else None,
        "num_classes": meta["num_classes"],
        "matched_keys": len(matched),
        "total_model_keys": len(model_sd),
        "matched_ratio": f"{len(matched)}/{len(model_sd)}",
        "missing_keys_count": len(missing_keys),
        "missing_keys": missing_keys,
        "unexpected_keys_count": len(unexpected_keys),
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "strict_match": strict_match,
    }
    return info, state_dict


def load_gasnet_checkpoint(
    ckpt_path: Path,
    expected_model_key: Optional[str] = None,
    deploy: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[GASNet, Dict[str, Any]]:
    """Build offline GASNet, validate strict parameter load, optionally fold BCGG."""
    info, state_dict = inspect_checkpoint(ckpt_path, expected_model_key=expected_model_key)
    if state_dict is None or not info["exists"]:
        raise FileNotFoundError(f"[FAIL] Checkpoint not found: {ckpt_path}")

    meta = _infer_ckpt_metadata(state_dict)
    model = GASNet(
        num_classes=meta["num_classes"],
        use_pretrained=False,
        use_part_branch=meta["use_part_branch"],
        use_attention_local=meta["use_attention_local"],
        use_gem=True,
        backbone=meta["backbone"],
        econvnext=meta["econvnext"],
        **meta["gate_cfg"],
    )
    model.load_state_dict(state_dict, strict=info["strict_match"])
    model.eval()
    if deploy:
        model = model.to_deploy().eval()
    if device is not None:
        model = model.to(device)
    return model, info


def resolve_checkpoint_path(cfg: Dict[str, Any], domain: str, model_key: str) -> Path:
    ckpts = cfg.get("checkpoints") or {}
    if domain not in ckpts or model_key not in ckpts[domain]:
        raise KeyError(f"[FAIL] Missing checkpoint mapping for domain={domain!r}, model={model_key!r} in eval_config.yaml")
    return resolve_path(ckpts[domain][model_key])


def resolve_efficiency_checkpoint(cfg: Dict[str, Any], model_key: str) -> Tuple[str, Path]:
    priority = cfg.get("efficiency", {}).get("checkpoint_priority_domains", ["vru", "veri", "vrai", "vehicleid"])
    for domain in priority:
        p = resolve_checkpoint_path(cfg, domain, model_key)
        if p.is_file():
            return domain, p
    raise FileNotFoundError(
        f"[FAIL] No trained checkpoint found for {model_key!r} across domains {priority}. "
        "Cannot benchmark with random weights."
    )


def count_gasnet_params(model: GASNet) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    gate_modules = [getattr(model, n, None) for n in ("ga1", "ga2", "ga3", "ga4")]
    gate_params = sum(p.numel() for m in gate_modules if m is not None for p in m.parameters())
    return {
        "total_params": int(total),
        "total_params_M": total / 1e6,
        "gate_params": int(gate_params),
        "gate_params_K": gate_params / 1e3,
    }


def count_module_params(module: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in module.parameters())
    return {
        "total_params": int(total),
        "total_params_M": total / 1e6,
        "params_K": total / 1e3,
    }


class ExactFLOPProfiler:
    """Count MACs across Conv2d, Conv1d, Linear, and RGA bmm operations."""

    def __init__(self) -> None:
        self.total_macs = 0
        self.hooks: List[Any] = []

    def _conv2d(self, mod: nn.Conv2d, inp, out) -> None:
        x = inp[0]
        b = int(x.shape[0])
        oh, ow = int(out.shape[2]), int(out.shape[3])
        kops = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        self.total_macs += b * kops * mod.out_channels * oh * ow

    def _conv1d(self, mod: nn.Conv1d, inp, out) -> None:
        x = inp[0]
        b = int(x.shape[0])
        ol = int(out.shape[2])
        kops = mod.kernel_size[0] * (mod.in_channels // mod.groups)
        self.total_macs += b * kops * mod.out_channels * ol

    def _linear(self, mod: nn.Linear, inp, out) -> None:
        x = inp[0]
        inst = int(np.prod(x.shape[:-1])) if x.dim() > 1 else 1
        self.total_macs += inst * mod.in_features * mod.out_features

    def _rga_spatial(self, mod: RGASpatial, inp, out) -> None:
        x = inp[0]
        b = int(x.shape[0])
        inter_ch = int(mod.theta[0].out_channels)
        self.total_macs += b * mod.num_nodes * mod.num_nodes * inter_ch

    def _rga_channel(self, mod: RGAChannel, inp, out) -> None:
        x = inp[0]
        b, c = int(x.shape[0]), int(x.shape[1])
        inter_sp = int(mod.theta[0].out_channels)
        self.total_macs += b * c * c * inter_sp

    def register(self, model: nn.Module) -> None:
        self.total_macs = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.hooks.append(m.register_forward_hook(self._conv2d))
            elif isinstance(m, nn.Conv1d):
                self.hooks.append(m.register_forward_hook(self._conv1d))
            elif isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(self._linear))
            elif isinstance(m, RGASpatial):
                self.hooks.append(m.register_forward_hook(self._rga_spatial))
            elif isinstance(m, RGAChannel):
                self.hooks.append(m.register_forward_hook(self._rga_channel))

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


@torch.no_grad()
def measure_exact_flops_macs(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    profiler = ExactFLOPProfiler()
    profiler.register(model)
    try:
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        macs = int(profiler.total_macs)
    finally:
        profiler.remove()
    return {
        "total_macs": macs,
        "GMACs": macs / 1e9,
        "GFLOPs": (2.0 * macs) / 1e9,
    }


def summarize_latencies(latencies_ms: List[float], batch_size: int) -> Dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p5 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    ips = (1000.0 * batch_size / p50) if p50 > 0 else 0.0
    return {
        "p5_ms": p5,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean,
        "std_ms": std,
        "throughput_ips": ips,
        "pct_under_33_33ms": float(np.mean(arr <= 33.33) * 100.0),
        "pct_under_40ms": float(np.mean(arr <= 40.0) * 100.0),
    }


class _FP16SafeGeMPool(nn.Module):
    """Scale-normalized GeM pooling mathematically identical to GeMPool but safe in FP16."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=self.eps)
        scale = x_clamped.amax(dim=(2, 3), keepdim=True)
        pooled = F.adaptive_avg_pool2d((x_clamped / scale).pow(self.p), (1, 1))
        return scale * pooled.pow(1.0 / self.p)


class _GASNetConcatWrapper(nn.Module):
    def __init__(self, gasnet_model: GASNet):
        super().__init__()
        self.model = copy.deepcopy(gasnet_model).cpu().eval()
        if hasattr(self.model, "gap") and hasattr(self.model.gap, "p"):
            self.model.gap = _FP16SafeGeMPool(
                p=float(self.model.gap.p), eps=float(getattr(self.model.gap, "eps", 1e-6))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            return torch.cat(list(out), dim=1)
        return out


def export_gasnet_onnx(
    model: GASNet,
    onnx_path: Path,
    img_size: int = 224,
    opset: int = 17,
) -> Path:
    import onnx

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _GASNetConcatWrapper(model).eval()
    # Trace with BS=2 so dynamic batch size is never specialized to constant 1.
    dummy = torch.randn(2, 3, img_size, img_size, device="cpu")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    return onnx_path


def build_trt_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    trtexec_bin: str = "/usr/src/tensorrt/bin/trtexec",
    img_size: int = 224,
    min_bs: int = 1,
    opt_bs: int = 1,
    max_bs: int = 32,
    timing_cache_file: Optional[Path] = None,
    builder_opt_level: Optional[int] = 2,
    avg_timing: Optional[int] = 2,
) -> Path:
    if not Path(trtexec_bin).is_file():
        raise FileNotFoundError(f"[FAIL] trtexec binary not found: {trtexec_bin}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=input:{min_bs}x3x{img_size}x{img_size}",
        f"--optShapes=input:{opt_bs}x3x{img_size}x{img_size}",
        f"--maxShapes=input:{max_bs}x3x{img_size}x{img_size}",
        "--skipInference",
    ]
    if timing_cache_file is not None:
        timing_cache_file.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--timingCacheFile={timing_cache_file}")
    if builder_opt_level is not None:
        cmd.append(f"--builderOptimizationLevel={int(builder_opt_level)}")
    if avg_timing is not None:
        cmd.append(f"--avgTiming={int(avg_timing)}")
    print(f"[INFO] Building TensorRT FP16 engine: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not engine_path.is_file():
        raise RuntimeError(
            f"[FAIL] trtexec failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    print(f"[OK] Built TensorRT engine: {engine_path} ({engine_path.stat().st_size / 1e6:.2f} MB)")
    return engine_path


class TensorRT10Runner:
    """TensorRT 10.3 Python runner backed by PyTorch CUDA tensors on a dedicated CUDA stream."""

    def __init__(self, engine_path: Path, device: Optional[torch.device] = None):
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        self.device = device or torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.device)
        logger = trt.Logger(trt.Logger.WARNING)
        with self.engine_path.open("rb") as f:
            engine_bytes = f.read()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"[FAIL] Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError(f"[FAIL] Could not identify I/O tensors in {engine_path}")

        self.output_dtype = self._to_torch_dtype(self.engine.get_tensor_dtype(self.output_name))
        self.input_dtype = self._to_torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        self._out_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    @staticmethod
    def _to_torch_dtype(trt_dtype) -> torch.dtype:
        import tensorrt as trt

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if hasattr(trt.DataType, "BF16"):
            mapping[trt.DataType.BF16] = torch.bfloat16
        return mapping.get(trt_dtype, torch.float32)

    def _get_output_buffer(self, shape: Tuple[int, ...]) -> torch.Tensor:
        buf = self._out_cache.get(shape)
        if buf is None:
            buf = torch.empty(shape, dtype=self.output_dtype, device=self.device)
            self._out_cache[shape] = buf
        return buf

    def synchronize(self) -> None:
        self.stream.synchronize()

    def infer(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.device != self.device:
            inp = inp.to(self.device)
        if inp.dtype != self.input_dtype:
            inp = inp.to(self.input_dtype)
        inp = inp.contiguous()

        cur_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(cur_stream)
        with torch.cuda.stream(self.stream):
            self.context.set_input_shape(self.input_name, tuple(inp.shape))
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            out = self._get_output_buffer(out_shape)
            self.context.set_tensor_address(self.input_name, int(inp.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(out.data_ptr()))
            ok = self.context.execute_async_v3(int(self.stream.cuda_stream))
            if not ok:
                raise RuntimeError(f"[FAIL] TensorRT execute_async_v3 failed on {self.engine_path.name}")
        cur_stream.wait_stream(self.stream)
        return out

    def close(self) -> None:
        self.synchronize()
        self._out_cache.clear()
        del self.context
        del self.engine
        del self.runtime


def _engine_meta_path(engine_path: Path) -> Path:
    return Path(f"{engine_path}.meta.json")


def _sig_matches(meta: Dict[str, Any], sig: Dict[str, Any]) -> bool:
    keys = ("ckpt_sha256", "deploy", "img_size", "min_bs", "opt_bs", "max_bs")
    return all(meta.get(k) == sig.get(k) for k in keys)


def build_or_load_trt_engine(
    ckpt_path: Path,
    model_key: str,
    cache_dir: Path,
    tag: str,
    trtexec_bin: str = "/usr/src/tensorrt/bin/trtexec",
    img_size: int = 224,
    opset: int = 17,
    min_bs: int = 1,
    opt_bs: int = 1,
    max_bs: int = 32,
    force_rebuild: bool = False,
    timing_cache_file: Optional[Path] = None,
    builder_opt_level: Optional[int] = 2,
    avg_timing: Optional[int] = 2,
    reuse_dirs: Optional[List[Path]] = None,
) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / f"{tag}.onnx"
    engine_path = cache_dir / f"{tag}_fp16.engine"
    meta_path = _engine_meta_path(engine_path)
    ckpt_sha = sha256_file(ckpt_path)
    sig = {
        "ckpt_sha256": ckpt_sha,
        "model_key": model_key,
        "deploy": ("bcgg" in model_key),
        "img_size": int(img_size),
        "min_bs": int(min_bs),
        "opt_bs": int(opt_bs),
        "max_bs": int(max_bs),
        "onnx_path": str(onnx_path),
    }

    if not force_rebuild:
        if engine_path.is_file() and onnx_path.is_file():
            if meta_path.is_file():
                try:
                    existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if _sig_matches(existing_meta, sig):
                        print(f"[OK] Reusing TensorRT engine: {engine_path}")
                        return onnx_path, engine_path
                except Exception:
                    pass
            else:
                meta_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")
                print(f"[OK] Reusing existing TensorRT engine (recorded SHA256 metadata): {engine_path}")
                return onnx_path, engine_path

        for rdir in reuse_dirs or []:
            if not Path(rdir).is_dir():
                continue
            for mfile in sorted(Path(rdir).glob("*.engine.meta.json")):
                cand_engine = Path(str(mfile)[: -len(".meta.json")])
                if not cand_engine.is_file():
                    continue
                try:
                    cand_meta = json.loads(mfile.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if _sig_matches(cand_meta, sig):
                    cand_onnx = Path(cand_meta.get("onnx_path", str(cand_engine).replace("_fp16.engine", ".onnx")))
                    print(f"[OK] Reusing TensorRT engine matching checkpoint SHA256: {cand_engine}")
                    return cand_onnx, cand_engine

    deploy = "bcgg" in model_key
    model, _ = load_gasnet_checkpoint(
        ckpt_path, expected_model_key=model_key, deploy=deploy, device=torch.device("cpu")
    )
    export_gasnet_onnx(model, onnx_path, img_size=img_size, opset=opset)
    del model
    build_trt_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        trtexec_bin=trtexec_bin,
        img_size=img_size,
        min_bs=min_bs,
        opt_bs=opt_bs,
        max_bs=max_bs,
        timing_cache_file=timing_cache_file or (cache_dir / "trt_timing.cache"),
        builder_opt_level=builder_opt_level,
        avg_timing=avg_timing,
    )
    meta_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")
    return onnx_path, engine_path


def write_csv(rows: Iterable[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> Path:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows_list[0].keys()) if rows_list else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
    print(f"[OK] Wrote CSV ({len(rows_list)} rows): {path}")
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote JSON: {path}")
    return path
