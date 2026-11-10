"""Pipeline lock & provenance tracking across train, eval, and calibration."""

from __future__ import annotations

import json
import os

SOURCES = {
    "num_frames": ("train", "num_frames"),
    "stride": ("data_pipeline", "frame_stride"),
    "backbone": ("train", "backbone"),
    "bbox_padding": ("data_pipeline", "bbox_padding"),
    "temporal_type": ("train", "temporal_type"),
    "temporal_pool": ("train", "temporal_pool"),
    "temporal_pe": ("train", "temporal_pe"),
}

OVERRIDABLE = {"num_frames"}
_SRC_LABEL = {k: f"{blk}.{sk}" for k, (blk, sk) in SOURCES.items()}
_DATA_KEYS = (
    "frame_stride",
    "num_before_frames",
    "num_after_frames",
    "bbox_padding",
    "crop_size",
)
_meta_seen = set()


def resolve_pipeline(cfg: dict, section: str, script: str = "", verbose: bool = True) -> dict:
    """Enforce pipeline parameters in `cfg[section]` from single source of truth."""
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        sec = {}
        cfg[section] = sec

    warnings = []
    notes = []
    resolved = {}

    for key, (blk, src_key) in SOURCES.items():
        src_label = _SRC_LABEL[key]
        source = (cfg.get(blk) or {}).get(src_key)
        own = sec.get(key)

        if key in OVERRIDABLE and own is not None:
            if source is not None and own != source:
                notes.append(f"{section}.{key} = {own!r} đè {src_label} = {source!r} (được phép)")
            sec[key] = own
            resolved[key] = (own, own, f"{section}.{key}")
            continue

        if source is None:
            if own is None:
                warnings.append(
                    f"thiếu nguồn {blk}.{src_key} và {section}.{key} cũng không có"
                )
            else:
                warnings.append(
                    f"thiếu nguồn {blk}.{src_key}; giữ nguyên {section}.{key} = {own!r}"
                )
            resolved[key] = (own, own, f"{section}.{key}")
            continue

        if own is not None and own != source:
            warnings.append(
                f"lệch {section}.{key} = {own!r} vs {blk}.{src_key} = {source!r} -> ép dùng {source!r}"
            )
        sec[key] = source
        resolved[key] = (source, own, src_label)

    if verbose:
        _print_banner(section, script, resolved, warnings, notes)

    check_data_meta(cfg, verbose=verbose)
    return sec


def check_data_meta(cfg: dict, verbose: bool = True) -> list[str]:
    """Compare `data_pipeline` config against `pipeline_meta.json` in `data_dir`."""
    dp = cfg.get("data_pipeline") or {}
    data_dir = (cfg.get("paths") or {}).get("data_dir")
    if not data_dir:
        return []

    meta_path = os.path.join(data_dir, "pipeline_meta.json")
    if not os.path.isfile(meta_path) or meta_path in _meta_seen:
        return []
    _meta_seen.add(meta_path)

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        msg = f"không đọc được {meta_path}: {e}"
        if verbose:
            print(f"  [WARNING] {msg}")
        return [msg]

    diffs = [
        f"{k}: dữ liệu sinh ở {meta[k]!r}, config đang là {dp[k]!r}"
        for k in _DATA_KEYS
        if k in meta and k in dp and meta[k] != dp[k]
    ]
    if diffs:
        msg = (
            f"config `data_pipeline` KHÁC `{meta_path}` — dữ liệu CHƯA được sinh lại, "
            f"train và eval/infer sẽ lệch nhau:"
        )
        if verbose:
            print("-" * 62)
            print(f"  [WARNING] {msg}")
            for d in diffs:
                print(f"      - {d}")
            print("      Sinh lại dữ liệu: python scripts/prepare_uav_data.py --config <config>")
            print("-" * 62)
        return [msg] + diffs
    return []


def _print_banner(section: str, script: str, resolved: dict, warnings: list, notes=()) -> None:
    line = "=" * 62
    print(line)
    print(f"  PIPELINE LOCK  script={script or '?'}  section={section}")
    print(line)
    for key, (val, own, label) in resolved.items():
        flag = f"   (config ghi {own!r})" if (own is not None and own != val) else ""
        print(f"  {key:<14} {val!r:<22} <- {label}{flag}")
    print("-" * 62)
    if warnings:
        for w in warnings:
            print(f"  [WARNING] {w}")
    elif notes:
        print("  [INFO] Không có lệch bị ép; xem các mục ghi đè hợp lệ bên dưới")
    else:
        print("  [OK] Không có lệch; mọi script dùng cùng một pipeline")
    for n in notes:
        print(f"  [INFO] {n}")
    print(line + "\n")


def provenance(cfg: dict) -> dict:
    """Extract provenance metadata for checkpoints and calibration reports."""
    tr = cfg.get("train") or {}
    dp = cfg.get("data_pipeline") or {}
    runtime_n = {
        s: cfg[s].get("num_frames")
        for s in ("eval", "infer", "infer_robustness", "infer_realworld")
        if isinstance(cfg.get(s), dict) and cfg[s].get("num_frames") is not None
    }
    return {
        "num_frames": tr.get("num_frames"),
        "num_frames_override": runtime_n or None,
        "frame_stride": dp.get("frame_stride"),
        "temporal_type": tr.get("temporal_type"),
        "temporal_pool": tr.get("temporal_pool"),
        "temporal_pe": tr.get("temporal_pe"),
        "backbone": tr.get("backbone"),
        "econvnext": tr.get("econvnext", False),
        "gate": tr.get("gate", "bcgg"),
        "n_frames_train": tr.get("n_frames_choices", "fixed"),
    }


def assert_frozen(cfg: dict) -> list[str]:
    """Check whether standard frozen pipeline constants were modified."""
    tr = cfg.get("train") or {}
    dp = cfg.get("data_pipeline") or {}
    loss = tr.get("loss") or {}
    out = []

    if tr.get("n_frames_choices"):
        out.append(
            "train.n_frames_choices đang bật nhưng pipeline chốt N cố định"
        )
    if not tr.get("temporal_type"):
        out.append(
            "thiếu train.temporal_type; nên ghi rõ 'mamba' hoặc 'attention'"
        )
    if dp.get("frame_stride") is None:
        out.append("thiếu data_pipeline.frame_stride -> mất đồng bộ toàn pipeline")

    n_max = max(list(tr.get("val_n_list") or []) + [tr.get("num_frames") or 0] or [0])
    for side in ("num_before_frames", "num_after_frames"):
        have = dp.get(side)
        if isinstance(have, int) and n_max > have:
            out.append(
                f"{side} = {have} < N lớn nhất cần dùng ({n_max}); clip sẽ bị pad"
            )
    if tr.get("temporal_pool") == "mean":
        out.append(
            "train.temporal_pool = 'mean'. Pipeline khuyến nghị 'attn' để bất biến theo N"
        )
    if loss.get("lam2") not in (0.0, 0, None):
        out.append(
            f"train.loss.lam2 = {loss.get('lam2')} != 0.0 (lưu ý kiểm tra ảnh hưởng của TemporalConsistencyLoss)"
        )
    return out
