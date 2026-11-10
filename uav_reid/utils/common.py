"""Common utilities: seed, CUDA configuration, logging, atomic checkpoint I/O."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_for_speed(gpu_jetson: bool = False) -> None:
    if not torch.cuda.is_available() or gpu_jetson:
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def choose_amp_dtype(dtype_str: str = "auto") -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype_str == "bf16":
        print("[WARN] bf16 not supported on this GPU; falling back to fp16")
    return torch.float16


def normalize_state_dict(state_dict: dict) -> dict:
    """Strip DataParallel / torch.compile wrapper prefixes from checkpoint keys."""
    if not state_dict:
        return state_dict
    out = state_dict
    for prefix in ("module.", "_orig_mod."):
        if out and all(key.startswith(prefix) for key in out):
            out = {key[len(prefix) :]: value for key, value in out.items()}
    return out


class Logger:
    """Tee stdout/stderr to both console stream and log file."""

    def __init__(self, filename: str | Path, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(str(filename))), exist_ok=True)
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.stream.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def save_ckpt_atomic(obj: dict, path: str | Path, quiet: bool = False) -> None:
    """Write checkpoint atomically via .tmp + fsync + os.replace to prevent corruption."""
    path_str = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
    tmp = path_str + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path_str)
    size_mb = os.path.getsize(path_str) / 1e6
    if size_mb <= 0:
        raise RuntimeError(f"Ghi checkpoint thất bại (0 byte): {path_str}")
    if not quiet:
        stage_info = obj.get("stage", "-") if isinstance(obj, dict) else "-"
        epoch_info = obj.get("epoch", "-") if isinstance(obj, dict) else "-"
        print(
            f"    [OK] Saved {os.path.basename(path_str)}: {size_mb:.1f} MB | "
            f"stage={stage_info} epoch={epoch_info} | {path_str}"
        )
