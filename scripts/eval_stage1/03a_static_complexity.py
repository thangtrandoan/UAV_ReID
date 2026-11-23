#!/usr/bin/env python3
"""Measure static complexity (#Params, Gate #Params, GFLOPs, GMACs, Weight Size) for 4 GASNet configs."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    count_gasnet_params,
    load_eval_config,
    load_gasnet_checkpoint,
    measure_exact_flops_macs,
    reset_gpu_memory,
    resolve_efficiency_checkpoint,
    resolve_output_dir,
    write_csv,
    write_json,
)


def _serialized_state_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Static Complexity Profile (4 Configs)")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "03_efficiency_and_latency_orin"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = int(cfg.get("preprocess", {}).get("img_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("resnet50ibn_rga", "resnet50ibn_rga", False),
        ("convnext_rga", "convnext_rga", False),
        ("convnext_bcgg_unfolded", "convnext_bcgg", False),
        ("convnext_bcgg_folded", "convnext_bcgg", True),
    ]

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for config_key, model_key, deploy in configs:
        domain, ckpt_path = resolve_efficiency_checkpoint(cfg, model_key)
        reset_gpu_memory(device)
        model, ckpt_info = load_gasnet_checkpoint(
            ckpt_path, expected_model_key=model_key, deploy=deploy, device=device
        )
        if config_key == "resnet50ibn_rga":
            assert model.backbone_type == "resnet50_ibn", f"Expected resnet50_ibn, got {model.backbone_type}"

        params = count_gasnet_params(model)
        flops = measure_exact_flops_macs(model, input_shape=(1, 3, img_size, img_size), device=device)
        serialized_mb = _serialized_state_size_mb(model)
        ckpt_mb = float(ckpt_info["size_MB"])

        csv_rows.append(
            {
                "config": config_key,
                "display_name": MODEL_DISPLAY_NAMES[config_key],
                "backbone": model.backbone_type,
                "gate": model.gate_kind,
                "deployed": deploy,
                "source_domain": domain,
                "checkpoint_path": str(ckpt_path),
                "total_params_M": f"{params['total_params_M']:.4f}",
                "gate_params_K": f"{params['gate_params_K']:.3f}",
                "GFLOPs": f"{flops['GFLOPs']:.4f}",
                "GMACs": f"{flops['GMACs']:.4f}",
                "checkpoint_size_MB": f"{ckpt_mb:.2f}",
                "serialized_weights_MB": f"{serialized_mb:.2f}",
            }
        )
        json_rows.append(
            {
                "config": config_key,
                "display_name": MODEL_DISPLAY_NAMES[config_key],
                "backbone": model.backbone_type,
                "gate": model.gate_kind,
                "deployed": deploy,
                "source_domain": domain,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_info["sha256"],
                "input_shape": [1, 3, img_size, img_size],
                "total_params": params["total_params"],
                "total_params_M": params["total_params_M"],
                "gate_params": params["gate_params"],
                "gate_params_K": params["gate_params_K"],
                "total_macs": flops["total_macs"],
                "GMACs": flops["GMACs"],
                "GFLOPs": flops["GFLOPs"],
                "checkpoint_size_MB": ckpt_mb,
                "serialized_weights_MB": serialized_mb,
            }
        )
        print(
            f"[OK] {config_key}: params={params['total_params_M']:.3f}M "
            f"gate(ga1..ga4)={params['gate_params_K']:.2f}K "
            f"GFLOPs={flops['GFLOPs']:.3f} GMACs={flops['GMACs']:.3f} "
            f"ckpt={ckpt_mb:.2f}MB"
        )
        del model
        reset_gpu_memory(device)

    write_csv(csv_rows, out_dir / "static_complexity.csv")
    write_json(json_rows, out_dir / "static_complexity.json")


if __name__ == "__main__":
    main()
