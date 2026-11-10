"""Configuration loader and CLI override helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"[FAIL] Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def save_yaml_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_scalar(val_str: str) -> Any:
    """Parse string value from CLI into bool, int, float, list, or str."""
    low = val_str.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    if val_str.startswith("[") and val_str.endswith("]"):
        return yaml.safe_load(val_str)
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def apply_cli_overrides(cfg: Dict[str, Any], overrides: List[str] | None) -> Dict[str, Any]:
    """Apply dot-separated `key.subkey=value` CLI overrides in-place."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}. Expected format: section.key=value")
        key_path, raw_val = item.split("=", 1)
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        if not keys:
            continue
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = parse_scalar(raw_val)
    return cfg


def load_config(path: str | Path, overrides: List[str] | None = None) -> Dict[str, Any]:
    """Load YAML config and optionally apply CLI dot-notation overrides."""
    cfg = load_yaml_config(path)
    if overrides:
        cfg = apply_cli_overrides(cfg, overrides)
    return cfg


save_config = save_yaml_config
apply_overrides = apply_cli_overrides
