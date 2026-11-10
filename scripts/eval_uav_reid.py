#!/usr/bin/env python3
"""Evaluate UAVReIDNet on UAV-Anti-UAV test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uav_reid.config import apply_cli_overrides, load_yaml_config
from uav_reid.engine import evaluate_uav_reid


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UAVReIDNet on test set")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "uav_reid.yaml"),
        help="Path to UAVReIDNet YAML config file",
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default=None,
        help="Path to calibrated_threshold.json",
    )
    parser.add_argument(
        "--space",
        type=str,
        choices=["fused", "pre_bn"],
        default=None,
        help="Primary feature space for threshold evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual cosine similarity threshold override",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config values via section.key=value",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    apply_cli_overrides(cfg, args.set)
    evaluate_uav_reid(
        cfg,
        threshold_file=args.threshold_file,
        space_override=args.space,
        manual_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
