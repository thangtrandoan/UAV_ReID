#!/usr/bin/env python3
"""Stage 1: Train or Evaluate GASNet on VRU / VRAI using YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uav_reid.config import apply_cli_overrides, load_yaml_config
from uav_reid.engine import evaluate_gasnet_checkpoint, train_gasnet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / Evaluate GASNet (Stage 1)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "gasnet_vru.yaml"),
        help="Path to GASNet YAML config file",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate a saved checkpoint and exit",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for --eval-only",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config values via section.key=value (e.g., --set train.epochs=60 train.batch_size=512)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    apply_cli_overrides(cfg, args.set)

    if args.eval_only:
        evaluate_gasnet_checkpoint(cfg, checkpoint_path=args.checkpoint)
    else:
        pc = cfg.get("paths", {})
        tc = cfg.get("train", {})
        has_weight_path = bool(pc.get("weight_path") or pc.get("pretrained_path") or pc.get("checkpoint"))
        dataset_name = str(tc.get("dataset", "vru")).strip().lower()
        if has_weight_path or dataset_name in ("veri", "veri776", "veri-776"):
            from scripts.finetune_gasnet.common import finetune_gasnet_from_config

            finetune_gasnet_from_config(cfg)
        else:
            train_gasnet(cfg)


if __name__ == "__main__":
    main()
