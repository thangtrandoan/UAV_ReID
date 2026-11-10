#!/usr/bin/env python3
"""Stage 2: Train UAVReIDNet (GASNet + Temporal Encoder) on UAV-Anti-UAV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uav_reid.config import apply_cli_overrides, load_yaml_config
from uav_reid.engine import train_uav_reid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UAVReIDNet (Stage 2)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "uav_reid.yaml"),
        help="Path to UAVReIDNet YAML config file",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config values via section.key=value (e.g., --set train.temporal_type=mamba)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    apply_cli_overrides(cfg, args.set)
    train_uav_reid(cfg)


if __name__ == "__main__":
    main()
