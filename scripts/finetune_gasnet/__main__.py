#!/usr/bin/env python3
"""Stage 1: Config-Driven Finetuning for GASNet on VeRi-776 / VRAI / VRU.

Usage:
    # 1. Run directly from YAML config (identical workflow to train_gasnet.py):
    python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml

    # 2. Override any YAML key from CLI via --set:
    python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
        --set train.dataset=veri \
              paths.weight_path=weights/gasnet_convnext_rga.best.pth \
              paths.save_path=weights/veri/gasnet_convnext_rga.best.pth \
              model.backbone=dinov3_convnext \
              model.gate=rga
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.finetune_gasnet.common import finetune_gasnet_from_config
from uav_reid.config import apply_cli_overrides, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune GASNet from YAML Config (Stage 1)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "gasnet_finetune.yaml"),
        help="Path to GASNet finetune YAML config file (default: configs/gasnet_finetune.yaml)",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help=(
            "Override config values via section.key=value "
            "(e.g., --set train.dataset=veri paths.weight_path=weights/gasnet_resnet_rga.best.pth "
            "model.backbone=resnet50_ibn model.gate=rga)"
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    apply_cli_overrides(cfg, args.set)
    finetune_gasnet_from_config(cfg)


if __name__ == "__main__":
    main()
