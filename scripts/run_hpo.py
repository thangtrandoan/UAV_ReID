#!/usr/bin/env python3
"""Run Optuna Hyperparameter Search (NAS-HPO) for GASNet or UAVReIDNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uav_reid.config import apply_cli_overrides, load_yaml_config
from uav_reid.search import run_hyperparameter_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna TPE + Hyperband Hyperparameter Optimization"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["gasnet", "uav_reid"],
        default="uav_reid",
        help="Which model stage to optimize ('gasnet' on VRU or 'uav_reid' on UAV-Anti-UAV)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to HPO YAML config file (defaults to configs/hpo_<stage>.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "hpo"),
        help="Directory to store Optuna SQLite DB, trial checkpoints, and best_config.yaml",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name (for resuming distributed/multi-GPU HPO)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional SQLite/PostgreSQL storage URL",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config values via section.key=value",
    )
    args = parser.parse_args()

    cfg_path = args.config or str(ROOT_DIR / "configs" / f"hpo_{args.stage}.yaml")
    cfg = load_yaml_config(cfg_path)
    apply_cli_overrides(cfg, args.set)

    n_trials = args.n_trials or int(cfg.get("hpo", {}).get("n_trials", 20))
    run_hyperparameter_search(
        base_cfg=cfg,
        stage=args.stage,
        n_trials=n_trials,
        output_dir=args.output_dir,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
