"""Hyperparameter Optimization (HPO / NAS-HPO) using Optuna with TPE + HyperbandPruner."""

from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

from uav_reid.config import save_yaml_config
from uav_reid.engine import train_gasnet, train_uav_reid


def _suggest_from_spec(trial, name: str, spec: Dict[str, Any]):
    kind = str(spec.get("type", "float")).lower()
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step)
    if kind in ("float", "loguniform"):
        log = bool(spec.get("log", kind == "loguniform"))
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]), log=log
        )
    raise ValueError(f"Unsupported search parameter type: {kind!r} for {name}")


def _set_nested_key(cfg: Dict[str, Any], dot_key: str, value: Any) -> None:
    keys = [k.strip() for k in dot_key.split(".") if k.strip()]
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


DEFAULT_GASNET_SEARCH_SPACE = {
    "train.backbone_lr": {"type": "float", "low": 1.0e-5, "high": 1.0e-4, "log": True},
    "train.head_lr": {"type": "float", "low": 1.0e-4, "high": 1.0e-3, "log": True},
    "train.convnext_weight_decay": {
        "type": "float",
        "low": 1.0e-3,
        "high": 1.0e-1,
        "log": True,
    },
    "train.triplet_margin": {"type": "float", "low": 0.2, "high": 0.5, "log": False},
    "train.triplet_weight": {"type": "float", "low": 0.25, "high": 1.5, "log": False},
    "train.label_smoothing": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
    "model.gem_p": {"type": "float", "low": 2.0, "high": 4.0, "log": False},
    "train.pk_k": {"type": "categorical", "choices": [4, 8]},
}

DEFAULT_UAV_REID_SEARCH_SPACE = {
    "train.stage1.lr": {"type": "float", "low": 1.0e-4, "high": 1.0e-3, "log": True},
    "train.stage1.weight_decay": {
        "type": "float",
        "low": 1.0e-4,
        "high": 5.0e-3,
        "log": True,
    },
    "train.stage2.lr_backbone": {
        "type": "float",
        "low": 3.0e-6,
        "high": 5.0e-5,
        "log": True,
    },
    "train.stage2.lr_temporal": {
        "type": "float",
        "low": 3.0e-5,
        "high": 5.0e-4,
        "log": True,
    },
    "train.stage2.lr_head": {
        "type": "float",
        "low": 3.0e-5,
        "high": 5.0e-4,
        "log": True,
    },
    "train.loss.lam1": {"type": "float", "low": 0.5, "high": 2.0, "log": False},
    "train.loss.lam3": {"type": "float", "low": 1.0e-4, "high": 1.0e-2, "log": True},
    "train.loss.triplet_margin": {
        "type": "float",
        "low": 0.2,
        "high": 0.5,
        "log": False,
    },
    "train.loss.label_smooth": {
        "type": "float",
        "low": 0.05,
        "high": 0.2,
        "log": False,
    },
    "train.temporal_dropout": {
        "type": "float",
        "low": 0.0,
        "high": 0.3,
        "log": False,
    },
}


def run_hyperparameter_search(
    base_cfg: Dict[str, Any],
    stage: str = "uav_reid",
    n_trials: int = 20,
    output_dir: str | Path = "outputs/hpo",
    study_name: str | None = None,
    storage: str | None = None,
) -> Dict[str, Any]:
    """Run Optuna TPE + Hyperband hyperparameter optimization for `gasnet` or `uav_reid`."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "[FAIL] Optuna chưa được cài đặt. Hãy chạy: pip install optuna"
        ) from exc

    stage = stage.lower()
    if stage not in ("gasnet", "uav_reid"):
        raise ValueError(f"stage must be 'gasnet' or 'uav_reid', got {stage!r}")

    out_dir = Path(output_dir) / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    hpo_cfg = base_cfg.get("hpo", {})
    n_trials = int(hpo_cfg.get("n_trials", n_trials))
    study_name = study_name or str(hpo_cfg.get("study_name", f"hpo_{stage}"))
    db_path = out_dir / f"{study_name}.db"
    storage = storage or hpo_cfg.get("storage") or f"sqlite:///{db_path.resolve()}"

    search_space = hpo_cfg.get("search_space") or (
        DEFAULT_GASNET_SEARCH_SPACE
        if stage == "gasnet"
        else DEFAULT_UAV_REID_SEARCH_SPACE
    )

    sampler = optuna.samplers.TPESampler(
        seed=int(hpo_cfg.get("seed", 42)), multivariate=True
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=int(hpo_cfg.get("min_resource", 5)),
        max_resource=int(hpo_cfg.get("max_resource", 30)),
        reduction_factor=int(hpo_cfg.get("reduction_factor", 3)),
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    print("=" * 72)
    print(f"  OPTUNA HYPERPARAMETER SEARCH ({stage.upper()})")
    print(f"  Study Name : {study_name}")
    print(f"  Storage    : {storage}")
    print(f"  Trials     : {n_trials}")
    print(f"  Output Dir : {out_dir}")
    print("=" * 72)

    def objective(trial: "optuna.Trial") -> float:
        trial_cfg = copy.deepcopy(base_cfg)
        trial_cfg["_OptunaQuietLog"] = True

        # Apply proxy epoch overrides if configured in `hpo.proxy_overrides`
        for k_ovr, v_ovr in (hpo_cfg.get("proxy_overrides") or {}).items():
            _set_nested_key(trial_cfg, k_ovr, v_ovr)

        sampled_params = {}
        for param_key, spec in search_space.items():
            val = _suggest_from_spec(trial, param_key, spec)
            sampled_params[param_key] = val
            _set_nested_key(trial_cfg, param_key, val)

        trial_dir = out_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        if stage == "gasnet":
            _set_nested_key(
                trial_cfg, "paths.save_path", str(trial_dir / "gasnet_trial.pth")
            )
            _set_nested_key(
                trial_cfg, "paths.log_path", str(trial_dir / "train.log")
            )
            # Ensure batch_size is divisible by pk_k
            bs = int(trial_cfg.get("train", {}).get("batch_size", 512))
            pk_k = int(trial_cfg.get("train", {}).get("pk_k", 4))
            if bs % pk_k != 0:
                _set_nested_key(trial_cfg, "train.batch_size", (bs // pk_k) * pk_k)
        else:
            _set_nested_key(
                trial_cfg, "paths.checkpoint_dir", str(trial_dir / "checkpoints")
            )
            _set_nested_key(trial_cfg, "paths.log_dir", str(trial_dir / "logs"))
            # Ensure batch_size is divisible by num_instances
            bs = int(trial_cfg.get("train", {}).get("batch_size", 12))
            ni = int(trial_cfg.get("train", {}).get("num_instances", 3))
            if bs % ni != 0:
                _set_nested_key(trial_cfg, "train.batch_size", max(ni * 2, (bs // ni) * ni))

        print(f"\n---> [Trial {trial.number}] Sampled hyperparameters:")
        for pk, pv in sampled_params.items():
            print(f"     {pk}: {pv}")

        def report_and_check_prune(step_epoch: int, score: float) -> bool:
            trial.report(score, step=step_epoch)
            return trial.should_prune()

        pruned = False
        try:
            if stage == "gasnet":
                res = train_gasnet(trial_cfg, trial_callback=report_and_check_prune)
                score = float(res.get("best_score", 0.0))
            else:
                res = train_uav_reid(trial_cfg, trial_callback=report_and_check_prune)
                score = float(res.get("best_val_rank1", 0.0))
        except Exception as exc:
            print(f"[WARNING] Trial {trial.number} failed with error: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"Trial errored: {exc}") from exc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best_trial = study.best_trial
    best_cfg = copy.deepcopy(base_cfg)
    best_cfg.pop("hpo", None)
    best_cfg.pop("_OptunaQuietLog", None)
    for k, v in best_trial.params.items():
        _set_nested_key(best_cfg, k, v)

    best_cfg_path = out_dir / f"best_config_{stage}.yaml"
    save_yaml_config(best_cfg, best_cfg_path)

    summary = {
        "stage": stage,
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_score": best_trial.value,
        "best_params": best_trial.params,
        "best_config_path": str(best_cfg_path),
        "n_trials_completed": len(study.trials),
    }
    summary_path = out_dir / "hpo_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 72)
    print(f"  [OK] HPO COMPLETED FOR {stage.upper()}")
    print(f"  Best Trial  : #{best_trial.number}")
    print(f"  Best Score  : {best_trial.value:.4f}")
    print(f"  Best Config : {best_cfg_path}")
    print(f"  Summary     : {summary_path}")
    print("=" * 72 + "\n")
    return summary
