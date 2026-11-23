#!/usr/bin/env python3
"""Audit Jetson AGX Orin environment and all 12 Stage 1 GASNet checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MODEL_KEYS,
    collect_env_info,
    inspect_checkpoint,
    load_eval_config,
    resolve_checkpoint_path,
    resolve_output_dir,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Audit Orin Environment & 12 Checkpoints")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if any of the 12 checkpoints is missing")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "00_env_and_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] 00_audit_manifest: Collecting Jetson Orin environment & checkpoint manifest")
    print("=" * 80)

    env_info = collect_env_info()
    write_json(env_info, out_dir / "env_info.json")

    ckpts_cfg = cfg.get("checkpoints") or {}
    domains = list(ckpts_cfg.keys())
    csv_rows: List[Dict[str, Any]] = []
    json_entries: List[Dict[str, Any]] = []
    missing_paths: List[str] = []

    for domain in domains:
        for model_key in MODEL_KEYS:
            ckpt_path = resolve_checkpoint_path(cfg, domain, model_key)
            info, _ = inspect_checkpoint(ckpt_path, expected_model_key=model_key)
            info["domain"] = domain
            info["model_key"] = model_key
            json_entries.append(info)

            if not info["exists"]:
                missing_paths.append(str(ckpt_path))
                print(f"[WARNING] Missing checkpoint: {domain}/{model_key} -> {ckpt_path}")
            else:
                print(
                    f"[OK] {domain}/{model_key}: backbone={info['backbone']} gate={info['gate']} "
                    f"classes={info['num_classes']} keys={info['matched_ratio']} sha256={info['sha256'][:16]}"
                )

            csv_rows.append(
                {
                    "domain": domain,
                    "model_key": model_key,
                    "checkpoint_path": str(ckpt_path),
                    "exists": info["exists"],
                    "status": info["status"],
                    "size_MB": info.get("size_MB") if info["exists"] else "N/A",
                    "sha256": info.get("sha256") or "N/A",
                    "backbone": info.get("backbone", "N/A"),
                    "econvnext": info.get("econvnext", "N/A"),
                    "gate": info.get("gate", "N/A"),
                    "gate_hidden": info.get("gate_hidden") if info.get("gate_hidden") is not None else "N/A",
                    "gate_groups": info.get("gate_groups") if info.get("gate_groups") is not None else "N/A",
                    "gate_kernel": info.get("gate_kernel") if info.get("gate_kernel") is not None else "N/A",
                    "gate_channel_kernel": info.get("gate_channel_kernel")
                    if info.get("gate_channel_kernel") is not None
                    else "N/A",
                    "gate_inject_value": info.get("gate_inject_value")
                    if info.get("gate_inject_value") is not None
                    else "N/A",
                    "num_classes": info.get("num_classes", "N/A"),
                    "matched_ratio": info.get("matched_ratio", "N/A"),
                    "missing_keys_count": info.get("missing_keys_count", "N/A"),
                    "unexpected_keys_count": info.get("unexpected_keys_count", "N/A"),
                    "strict_match": info.get("strict_match", False),
                }
            )

    write_csv(csv_rows, out_dir / "checkpoints_manifest.csv")
    write_json(
        {
            "total_expected": len(csv_rows),
            "present_count": sum(1 for r in json_entries if r["exists"]),
            "missing_paths": missing_paths,
            "checkpoints": json_entries,
        },
        out_dir / "checkpoints_manifest.json",
    )

    if args.strict and missing_paths:
        raise FileNotFoundError(
            f"[FAIL] Strict mode enabled and {len(missing_paths)} checkpoint(s) are missing: {missing_paths}"
        )

    print(f"[OK] Audit complete: {sum(1 for r in json_entries if r['exists'])}/{len(csv_rows)} checkpoints present.")


if __name__ == "__main__":
    main()
