#!/usr/bin/env python3
"""Master orchestrator for Stage 1 (GASNet) Jetson AGX Orin evaluation & verification."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_eval_config, resolve_output_dir  # noqa: E402


def _run_step(script_name: str, extra_args: List[str]) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *extra_args]
    print(f"\n>>> RUNNING: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"[FAIL] Step {script_name} exited with code {proc.returncode}")


def verify_checklist(out_root: Path) -> bool:
    print("\n" + "=" * 80)
    print("STAGE 1 CHECKLIST VERIFICATION")
    print("=" * 80)
    ok_all = True

    # 1. checkpoints_manifest.csv
    man_csv = out_root / "00_env_and_manifest" / "checkpoints_manifest.csv"
    c1 = man_csv.is_file()
    c1_detail = "missing file"
    if c1:
        with man_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        bad_resnet = [r for r in rows if r["model_key"] == "resnet50ibn_rga" and r["exists"] == "True" and r["backbone"] != "resnet50_ibn"]
        present = sum(1 for r in rows if r["exists"] == "True")
        c1 = (len(rows) == 12) and (len(bad_resnet) == 0)
        c1_detail = f"{len(rows)} entries ({present} present on disk, bad_resnet={len(bad_resnet)})"
    print(f"[{'OK' if c1 else 'FAIL'}] 1. Manifest: {man_csv} -> {c1_detail}")
    ok_all &= c1

    # 2. accuracy_summary.csv & protocol_metadata.json
    acc_csv = out_root / "01_accuracy_benchmarks" / "accuracy_summary.csv"
    proto_json = out_root / "01_accuracy_benchmarks" / "protocol_metadata.json"
    c2 = acc_csv.is_file() and proto_json.is_file()
    c2_detail = "missing file"
    if c2:
        with acc_csv.open("r", encoding="utf-8") as f:
            arows = list(csv.DictReader(f))
        c2 = len(arows) > 0
        c2_detail = f"{len(arows)} rows across datasets={sorted({r['dataset'] for r in arows})}"
    print(f"[{'OK' if c2 else 'FAIL'}] 2. Accuracy summary: {acc_csv} -> {c2_detail}")
    ok_all &= c2

    # 3. bcgg_equivalence_report.json
    bcgg_json = out_root / "02_bcgg_deploy_equivalence" / "bcgg_equivalence_report.json"
    c3 = bcgg_json.is_file()
    c3_detail = "missing file"
    if c3:
        rep = json.loads(bcgg_json.read_text(encoding="utf-8"))
        doms = rep.get("verified_domains", [])
        c3 = bool(doms) and all(
            bool(d.get("checkpoint_sha256")) and d["tensor_equivalence"]["fp32"]["z_max_abs_diff"] < 1e-4
            for d in doms
        )
        c3_detail = f"{len(doms)} domain(s) verified with real SHA256 and fp32 diff < 1e-4"
    print(f"[{'OK' if c3 else 'FAIL'}] 3. BCGG equivalence: {bcgg_json} -> {c3_detail}")
    ok_all &= c3

    # 4. Efficiency PyTorch + TRT + E2E
    pt_csv = out_root / "03_efficiency_and_latency_orin" / "latency_pytorch_summary.csv"
    trt_csv = out_root / "03_efficiency_and_latency_orin" / "latency_tensorrt_summary.csv"
    e2e_csv = out_root / "03_efficiency_and_latency_orin" / "latency_e2e_summary.csv"
    c4 = pt_csv.is_file() and trt_csv.is_file() and e2e_csv.is_file()
    print(f"[{'OK' if c4 else 'FAIL'}] 4. Orin Efficiency (PyTorch + TRT + E2E): files_present={c4}")
    ok_all &= c4

    # 5. Attention heatmaps & stats
    hm_json = out_root / "05_visualizations" / "attention_heatmaps" / "heatmap_stats.json"
    c5 = hm_json.is_file()
    c5_detail = "missing file"
    if c5:
        hdata = json.loads(hm_json.read_text(encoding="utf-8"))
        c5 = bool(hdata.get("hook_verified")) and len(hdata.get("samples", [])) > 0
        c5_detail = f"hook_verified={hdata.get('hook_verified')}, entries={len(hdata.get('samples', []))}"
    print(f"[{'OK' if c5 else 'FAIL'}] 5. Attention heatmaps: {hm_json} -> {c5_detail}")
    ok_all &= c5
    return ok_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Stage 1 (GASNet) Jetson evaluation steps")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_config.yaml"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if any dataset or checkpoint is missing")
    parser.add_argument("--lock-clocks", action="store_true", help="Run sudo nvpmodel -m 0 && sudo jetson_clocks")
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT export/benchmark & TRT accuracy")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        choices=["00", "01", "02", "03a", "03b", "03c", "01_trt", "04", "05"],
        help="Optional subset of stages to execute",
    )
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    common_args = ["--config", str(args.config), "--output-dir", str(out_root)]
    strict_args = [*common_args, "--strict"] if args.strict else common_args
    clock_args = [*common_args, "--lock-clocks"] if args.lock_clocks else common_args

    all_stages = ["00", "01", "02", "03a", "03b", "03c", "01_trt", "04", "05"]
    if args.skip_trt:
        all_stages = [s for s in all_stages if s not in ("03c", "01_trt")]
    selected = set(args.stages) if args.stages else set(all_stages)

    for st in all_stages:
        if st not in selected:
            continue
        if st == "00":
            _run_step("00_audit_manifest.py", strict_args)
        elif st == "01":
            _run_step("01_eval_accuracy.py", strict_args)
        elif st == "02":
            _run_step("02_verify_bcgg_folding.py", strict_args)
        elif st == "03a":
            _run_step("03a_static_complexity.py", common_args)
        elif st == "03b":
            _run_step("03b_bench_pytorch.py", clock_args)
        elif st == "03c":
            _run_step("03c_export_and_bench_trt.py", clock_args)
        elif st == "01_trt":
            _run_step("01_eval_accuracy.py", [*strict_args, "--trt-only"])
        elif st == "04":
            _run_step("04_bench_gate_scaling.py", clock_args)
        elif st == "05":
            _run_step("05_generate_visuals.py", common_args)

    if not args.stages:
        verify_checklist(out_root)


if __name__ == "__main__":
    main()
