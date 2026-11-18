#!/usr/bin/env python3
"""Master Evaluation Script for Stage 1: GASNet Improvements.

Generates all required experimental reports:
1. Thí nghiệm 1 (Bảng 5.1): So sánh GASNet dùng ResNet50-IBN vs DINO-ConvNeXt
   trên VeRi-776, VehicleID / VRU (Small/Medium/Large) (mAP, Rank-1, Rank-5, mINP).
2. Thí nghiệm 2 (Bảng 5.2): Ablation lũy tiến của GASNet (Độ chính xác & Tốc độ):
   - Hàng 1: GASNet gốc (ResNet-50 + RGA)
   - Hàng 2: GASNet + DINO-ConvNeXt + RGA
   - Hàng 3: GASNet + DINO-ConvNeXt + BCGG (Train mode)
   - Hàng 4: GASNet + DINO-ConvNeXt + BCGG (Deploy mode - RepDW Folded)
3. Thí nghiệm 2 (Bảng 5.3): So sánh chi tiết chi phí tính toán RGA vs BCGG
   ở các độ phân giải 224x224, 256x256, 384x384.
4. Trực quan hóa (Qualitative Analysis):
   - Attention Heatmaps (last_attention từ BCGG vs RGA)
   - Bảng ảnh truy vấn Top-5 (Query + Top-5 Gallery với viền xanh/đỏ).

Usage:
    python3 -m scripts.eval_gasnet.run_all --all
    python3 -m scripts.eval_gasnet.run_all --table 5.1
    python3 -m scripts.eval_gasnet.run_all --table 5.2
    python3 -m scripts.eval_gasnet.run_all --table 5.3
    python3 -m scripts.eval_gasnet.run_all --visualize
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_gasnet.datasets import (
    create_eval_dataloader,
    load_vehicleid,
    load_veri776,
    load_vrai_val,
    load_vru_split,
    resolve_dataset_path,
)
from scripts.eval_gasnet.efficiency import run_full_efficiency_profile
from scripts.eval_gasnet.gate_benchmark import run_table_5_3_benchmark
from scripts.eval_gasnet.metrics import (
    compute_distance_matrix,
    evaluate_rank_map_minp,
    format_metrics_dict,
)
from scripts.eval_gasnet.models import (
    extract_dataset_features,
    find_checkpoint,
    load_gasnet_variant,
)
from scripts.eval_gasnet.visualize import run_qualitative_evaluation


def evaluate_dataset_on_model(
    model: torch.nn.Module,
    q_dataset,
    g_dataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    dataset_name: str = "dataset",
) -> Dict[str, float]:
    """Run full feature extraction and compute CMC, mAP, mINP."""
    if device is None:
        device = next(model.parameters()).device

    q_loader = create_eval_dataloader(q_dataset, batch_size=batch_size)
    g_loader = create_eval_dataloader(g_dataset, batch_size=batch_size)

    q_feats, q_pids, q_cams, _ = extract_dataset_features(model, q_loader, device=device)
    g_feats, g_pids, g_cams, _ = extract_dataset_features(model, g_loader, device=device)

    distmat = compute_distance_matrix(q_feats, g_feats, metric="cosine")

    # If camera IDs are available (e.g. VeRi-776), apply cross-camera junk filtering
    has_cams = (q_cams >= 0).any() and (g_cams >= 0).any()
    q_c = q_cams.numpy() if has_cams else None
    g_c = g_cams.numpy() if has_cams else None

    cmc, mAP, mINP, _ = evaluate_rank_map_minp(
        distmat=distmat,
        q_pids=q_pids.numpy(),
        g_pids=g_pids.numpy(),
        q_camids=q_c,
        g_camids=g_c,
        max_rank=20,
    )
    return format_metrics_dict(cmc, mAP, mINP)


def save_markdown_and_csv(
    data: List[Dict[str, Any]],
    output_dir: Path,
    file_prefix: str,
    title: str,
) -> None:
    """Save a list of dictionaries as both Markdown table and CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / f"{file_prefix}.csv"
    md_file = output_dir / f"{file_prefix}.md"

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    md_content = f"# {title}\n\n"
    # Pure-Python Markdown table generator (no tabulate dependency required)
    headers = list(df.columns)
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join([":---:" if ("Rank" in h or "mAP" in h or "GFLOPs" in h or "FPS" in h or "ms" in h or "MB" in h or "#Params" in h) else ":---" for h in headers]) + " |\n"
    for _, row in df.iterrows():
        md_content += "| " + " | ".join(str(val) for val in row.values) + " |\n"
    md_content += "\n"
    md_file.write_text(md_content, encoding="utf-8")

    print(f"[OK] Saved: {csv_file}")
    print(f"[OK] Saved: {md_file}")


# --------------------------------------------------------------------------- #
# Experiment 1: Table 5.1 (Backbone Comparison)
# --------------------------------------------------------------------------- #
def run_experiment_1_backbone_comparison(
    data_root: Optional[str] = None,
    weights_dir: Optional[str] = None,
    output_dir: Path = Path("outputs/gasnet_report"),
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    legacy_resize: bool = False,
) -> List[Dict[str, Any]]:
    """Run Table 5.1: Compare ResNet50-IBN vs DINO-ConvNeXt on VeRi-776, VRU/VehicleID."""
    print("\n" + "=" * 90)
    print("THÍ NGHIỆM 1 — BẢNG 5.1: SO SÁNH HIỆU QUẢ THAY THẾ BACKBONE DINO-CONVNEXT")
    print(f"Preprocess: {'Resize 256 -> CenterCrop 224' if legacy_resize else 'Direct Resize 224x224 (Standard Vehicle Re-ID)'}")
    print("=" * 90)

    models_to_test = [
        ("ResNet50-IBN (RGA)", "resnet_rga"),
        ("DINO-ConvNeXt (RGA)", "convnext_rga"),
        ("DINO-ConvNeXt (BCGG)", "convnext_bcgg_deploy"),
    ]

    table_data: List[Dict[str, Any]] = []

    # 1. VeRi-776
    try:
        q_veri, g_veri = load_veri776(root_dir=data_root, legacy_resize=legacy_resize)
        print(f"[DATA] VeRi-776 loaded: {len(q_veri)} queries, {len(g_veri)} gallery images.")
        for bb_name, variant in models_to_test:
            print(f"[EVAL] Evaluating {bb_name} on VeRi-776...")
            model, meta = load_gasnet_variant(variant, device=device)
            metrics = evaluate_dataset_on_model(model, q_veri, g_veri, batch_size=batch_size, device=device)
            row = {
                "Dataset": "VeRi-776",
                "Split": "Full Test",
                "Backbone": bb_name,
                "mAP (%)": f"{metrics['mAP']:.2f}",
                "Rank-1 (%)": f"{metrics['Rank-1']:.2f}",
                "Rank-5 (%)": f"{metrics['Rank-5']:.2f}",
                "Rank-10 (%)": f"{metrics['Rank-10']:.2f}",
                "mINP (%)": f"{metrics['mINP']:.2f}",
            }
            table_data.append(row)
            print(f" -> mAP: {row['mAP (%)']} | Rank-1: {row['Rank-1 (%)']} | Rank-5: {row['Rank-5 (%)']} | Rank-10: {row['Rank-10 (%)']} | mINP: {row['mINP (%)']}")
    except Exception as exc:
        print(f"[WARNING] Skipping VeRi-776 evaluation: {exc}")

    # 2. VehicleID (Đánh giá riêng biệt từ data/VehicleID/)
    try:
        q_veh, g_veh = load_vehicleid(root_dir=data_root, legacy_resize=legacy_resize)
        print(f"[DATA] VehicleID (Brno) loaded: {len(q_veh)} queries (Cam A), {len(g_veh)} gallery (Cam B).")
        for bb_name, variant in models_to_test:
            print(f"[EVAL] Evaluating {bb_name} on VehicleID...")
            model, meta = load_gasnet_variant(variant, device=device)
            metrics = evaluate_dataset_on_model(model, q_veh, g_veh, batch_size=batch_size, device=device)
            row = {
                "Dataset": "VehicleID",
                "Split": "Pairs (1232 A↔B)",
                "Backbone": bb_name,
                "mAP (%)": f"{metrics['mAP']:.2f}",
                "Rank-1 (%)": f"{metrics['Rank-1']:.2f}",
                "Rank-5 (%)": f"{metrics['Rank-5']:.2f}",
                "Rank-10 (%)": f"{metrics['Rank-10']:.2f}",
                "mINP (%)": f"{metrics['mINP']:.2f}",
            }
            table_data.append(row)
            print(f" -> mAP: {row['mAP (%)']} | Rank-1: {row['Rank-1 (%)']} | Rank-5: {row['Rank-5 (%)']} | Rank-10: {row['Rank-10 (%)']} | mINP: {row['mINP (%)']}")
    except Exception as exc:
        print(f"[WARNING] Skipping VehicleID evaluation: {exc}")

    # 3. VRU (Đánh giá riêng biệt từ data/VRU/ theo 3 split Small, Medium, Large)
    vru_splits = [("Small (1200)", "Small"), ("Medium (2400)", "Medium"), ("Large (8000)", "Big")]
    for split_display, split_key in vru_splits:
        try:
            q_vru, g_vru = load_vru_split(root_dir=data_root, split_name=split_key, seed=42, legacy_resize=legacy_resize)
            print(f"[DATA] VRU {split_display} loaded: {len(q_vru)} queries, {len(g_vru)} gallery.")
            for bb_name, variant in models_to_test:
                print(f"[EVAL] Evaluating {bb_name} on VRU {split_display}...")
                model, meta = load_gasnet_variant(variant, device=device)
                metrics = evaluate_dataset_on_model(model, q_vru, g_vru, batch_size=batch_size, device=device)
                row = {
                    "Dataset": "VRU",
                    "Split": split_display,
                    "Backbone": bb_name,
                    "mAP (%)": f"{metrics['mAP']:.2f}",
                    "Rank-1 (%)": f"{metrics['Rank-1']:.2f}",
                    "Rank-5 (%)": f"{metrics['Rank-5']:.2f}",
                    "Rank-10 (%)": f"{metrics['Rank-10']:.2f}",
                    "mINP (%)": f"{metrics['mINP']:.2f}",
                }
                table_data.append(row)
                print(f" -> mAP: {row['mAP (%)']} | Rank-1: {row['Rank-1 (%)']} | Rank-5: {row['Rank-5 (%)']} | Rank-10: {row['Rank-10 (%)']} | mINP: {row['mINP (%)']}")
        except Exception as exc:
            print(f"[WARNING] Skipping VRU {split_display}: {exc}")

    # 4. VRAI (Đánh giá trên tập validation 1000 IDs ảnh UAV góc nhìn trên cao)
    try:
        q_vrai, g_vrai = load_vrai_val(root_dir=data_root, num_ids=1000, seed=42, legacy_resize=legacy_resize)
        print(f"[DATA] VRAI (Val 1000 IDs) loaded: {len(q_vrai)} queries, {len(g_vrai)} gallery.")
        for bb_name, variant in models_to_test:
            print(f"[EVAL] Evaluating {bb_name} on VRAI...")
            model, meta = load_gasnet_variant(variant, device=device)
            metrics = evaluate_dataset_on_model(model, q_vrai, g_vrai, batch_size=batch_size, device=device)
            row = {
                "Dataset": "VRAI",
                "Split": "Val (1000 IDs)",
                "Backbone": bb_name,
                "mAP (%)": f"{metrics['mAP']:.2f}",
                "Rank-1 (%)": f"{metrics['Rank-1']:.2f}",
                "Rank-5 (%)": f"{metrics['Rank-5']:.2f}",
                "Rank-10 (%)": f"{metrics['Rank-10']:.2f}",
                "mINP (%)": f"{metrics['mINP']:.2f}",
            }
            table_data.append(row)
            print(f" -> mAP: {row['mAP (%)']} | Rank-1: {row['Rank-1 (%)']} | Rank-5: {row['Rank-5 (%)']} | Rank-10: {row['Rank-10 (%)']} | mINP: {row['mINP (%)']}")
    except Exception as exc:
        print(f"[WARNING] Skipping VRAI evaluation: {exc}")

    if table_data:
        save_markdown_and_csv(
            table_data,
            output_dir,
            "table_5_1_backbone_comparison",
            "Bảng 5.1: So sánh GASNet dùng ResNet50-IBN và DINO-ConvNeXt",
        )
    return table_data


# --------------------------------------------------------------------------- #
# Experiment 2: Table 5.2 (Progressive Ablation of GASNet)
# --------------------------------------------------------------------------- #
def run_experiment_2_progressive_ablation(
    data_root: Optional[str] = None,
    weights_dir: Optional[str] = None,
    output_dir: Path = Path("outputs/gasnet_report"),
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    ablation_dataset: str = "vru",
    legacy_resize: bool = False,
) -> List[Dict[str, Any]]:
    """Run Table 5.2: Progressive Ablation of GASNet across 4 variants.

    Hàng 1: GASNet gốc (ResNet-50 + RGA)
    Hàng 2: GASNet + DINO-ConvNeXt + RGA
    Hàng 3: GASNet + DINO-ConvNeXt + BCGG (Train mode)
    Hàng 4: GASNet + DINO-ConvNeXt + BCGG (Deploy mode - RepDW Folded)
    """
    print("\n" + "=" * 90)
    print("THÍ NGHIỆM 2 — BẢNG 5.2: ABLATION LŨY TIẾN CỦA GASNet (ĐỘ CHÍNH XÁC & HIỆU NĂNG)")
    print(f"Preprocess: {'Resize 256 -> CenterCrop 224' if legacy_resize else 'Direct Resize 224x224 (Standard Vehicle Re-ID)'}")
    print("=" * 90)

    rows_config = [
        ("GASNet gốc (ResNet-50 + RGA)", "resnet_rga"),
        ("GASNet + DINO-ConvNeXt + RGA", "convnext_rga"),
        ("GASNet + DINO-ConvNeXt + BCGG (Train mode)", "convnext_bcgg_train"),
        ("GASNet + DINO-ConvNeXt + BCGG (Deploy - RepDW Folded)", "convnext_bcgg_deploy"),
    ]

    # Select evaluation dataset for accuracy: prefer specified ablation_dataset
    eval_dataset_q = None
    eval_dataset_g = None
    dataset_title = "VRU (Small)"

    if ablation_dataset.lower() == "vru":
        try:
            eval_dataset_q, eval_dataset_g = load_vru_split(root_dir=data_root, split_name="Small", seed=42, legacy_resize=legacy_resize)
            dataset_title = "VRU (Small)"
            print(f"[DATA] Selected {dataset_title} for Table 5.2 Ablation accuracy.")
        except Exception as e:
            print(f"[WARNING] Could not load VRU: {e}")
    elif ablation_dataset.lower() == "vehicleid":
        try:
            eval_dataset_q, eval_dataset_g = load_vehicleid(root_dir=data_root, legacy_resize=legacy_resize)
            dataset_title = "VehicleID"
            print(f"[DATA] Selected {dataset_title} for Table 5.2 Ablation accuracy.")
        except Exception as e:
            print(f"[WARNING] Could not load VehicleID: {e}")
    elif ablation_dataset.lower() == "veri":
        try:
            eval_dataset_q, eval_dataset_g = load_veri776(root_dir=data_root, legacy_resize=legacy_resize)
            dataset_title = "VeRi-776"
            print(f"[DATA] Selected {dataset_title} for Table 5.2 Ablation accuracy.")
        except Exception as e:
            print(f"[WARNING] Could not load VeRi-776: {e}")
    elif ablation_dataset.lower() == "vrai":
        try:
            eval_dataset_q, eval_dataset_g = load_vrai_val(root_dir=data_root, num_ids=1000, seed=42, legacy_resize=legacy_resize)
            dataset_title = "VRAI (Val 1000)"
            print(f"[DATA] Selected {dataset_title} for Table 5.2 Ablation accuracy.")
        except Exception as e:
            print(f"[WARNING] Could not load VRAI: {e}")

    # Fallback if preferred dataset failed to load
    if eval_dataset_q is None:
        try:
            eval_dataset_q, eval_dataset_g = load_vru_split(root_dir=data_root, split_name="Small", seed=42, legacy_resize=legacy_resize)
            dataset_title = "VRU (Small)"
        except Exception:
            try:
                eval_dataset_q, eval_dataset_g = load_veri776(root_dir=data_root, legacy_resize=legacy_resize)
                dataset_title = "VeRi-776"
            except Exception:
                pass

    table_data: List[Dict[str, Any]] = []

    for name, variant in rows_config:
        print(f"\n[BENCHMARK] Profiling row: {name}...")
        model, meta = load_gasnet_variant(variant, device=device)

        # 1. Efficiency profile
        eff = run_full_efficiency_profile(model, img_size=224, device=device)

        # 2. Accuracy evaluation
        map_val, r1_val, r5_val, r10_val, minp_val = "-", "-", "-", "-", "-"
        if eval_dataset_q is not None and eval_dataset_g is not None:
            print(f" -> Evaluating accuracy on {dataset_title}...")
            metrics = evaluate_dataset_on_model(
                model, eval_dataset_q, eval_dataset_g, batch_size=batch_size, device=device
            )
            map_val = f"{metrics['mAP']:.2f}"
            r1_val = f"{metrics['Rank-1']:.2f}"
            r5_val = f"{metrics['Rank-5']:.2f}"
            r10_val = f"{metrics['Rank-10']:.2f}"
            minp_val = f"{metrics['mINP']:.2f}"

        row = {
            "Mô hình / Biến thể": name,
            "#Params Tổng (M)": f"{eff['params_total_M']:.2f}",
            "Gate #Params (K)": f"{eff['params_gate_K']:.1f}",
            "GFLOPs": f"{eff['GFLOPs']:.2f}",
            "GMACs": f"{eff['GMACs']:.2f}",
            "Latency Tổng (ms)": f"{eff['latency_online_ms']:.2f} ± {eff['latency_online_std']:.2f}",
            "Backbone (ms)": f"{eff['latency_backbone_ms']:.2f}",
            "Gate+Head (ms)": f"{eff['latency_gate_head_ms']:.2f}",
            "Throughput BS=1 (FPS)": f"{eff['fps_online']:.1f}",
            "Throughput Batch (FPS)": f"{eff['fps_batch']:.1f}",
            "Peak VRAM (MB)": f"{eff['peak_vram_MB']:.1f}",
            f"mAP (%) [{dataset_title}]": map_val,
            f"Rank-1 (%) [{dataset_title}]": r1_val,
            f"Rank-5 (%) [{dataset_title}]": r5_val,
            f"Rank-10 (%) [{dataset_title}]": r10_val,
            f"mINP (%) [{dataset_title}]": minp_val,
        }
        table_data.append(row)
        print(f" -> #Params: {row['#Params Tổng (M)']}M (Gate: {row['Gate #Params (K)']}K) | GFLOPs: {row['GFLOPs']} | Latency: {row['Latency Tổng (ms)']} ms (BB: {row['Backbone (ms)']} ms, Gate+Head: {row['Gate+Head (ms)']} ms) | BS=1 FPS: {row['Throughput BS=1 (FPS)']} | Peak VRAM: {row['Peak VRAM (MB)']} MB")

    if table_data:
        save_markdown_and_csv(
            table_data,
            output_dir,
            "table_5_2_ablation_gasnet",
            "Bảng 5.2: Ablation Lũy tiến của GASNet (Độ chính xác & Tốc độ)",
        )
    return table_data


# --------------------------------------------------------------------------- #
# Main CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Unified GASNet Stage 1 Evaluation Suite")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing datasets")
    parser.add_argument("--weights-dir", type=str, default="weights", help="Directory containing checkpoint weights")
    parser.add_argument("--output-dir", type=str, default="outputs/gasnet_report", help="Directory for generated reports")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--table", type=str, choices=["5.1", "5.2", "5.3", "all"], default="all", help="Table(s) to benchmark")
    parser.add_argument("--ablation-dataset", type=str, choices=["vru", "veri", "vehicleid", "vrai"], default="vru", help="Dataset for Table 5.2 Ablation accuracy ('vru', 'veri', 'vehicleid', or 'vrai')")
    parser.add_argument("--legacy-resize", action="store_true", help="Use legacy ImageNet resize (Resize 256 then CenterCrop 224) instead of direct Resize 224")
    parser.add_argument("--visualize", action="store_true", help="Generate qualitative visualizations (Attention Heatmaps + Top-5)")
    parser.add_argument("--vis-dataset", type=str, choices=["veri", "vru", "vehicleid", "vrai"], default="veri", help="Dataset to sample queries for qualitative visualizations")
    parser.add_argument("--num-vis", type=int, default=5, help="Number of queries to visualize")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks and qualitative visualizations")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"GASNET STAGE 1 COMPREHENSIVE BENCHMARK SYSTEM")
    print(f"Device: {device}")
    print(f"Output Directory: {out_dir.resolve()}")
    print(f"=======================================================\n")

    run_all_flag = args.all or (args.table == "all" and not args.visualize)

    # 1. Table 5.1
    if run_all_flag or args.table == "5.1":
        run_experiment_1_backbone_comparison(
            data_root=args.data_root,
            weights_dir=args.weights_dir,
            output_dir=out_dir,
            batch_size=args.batch_size,
            device=device,
            legacy_resize=args.legacy_resize,
        )

    # 2. Table 5.2
    if run_all_flag or args.table == "5.2":
        run_experiment_2_progressive_ablation(
            data_root=args.data_root,
            weights_dir=args.weights_dir,
            output_dir=out_dir,
            batch_size=args.batch_size,
            device=device,
            ablation_dataset=args.ablation_dataset,
            legacy_resize=args.legacy_resize,
        )

    # 3. Table 5.3
    if run_all_flag or args.table == "5.3":
        gate_results = run_table_5_3_benchmark(channels=384, device=device)
        save_markdown_and_csv(
            gate_results,
            out_dir,
            "table_5_3_gate_comparison",
            "Bảng 5.3: So sánh Chi tiết Chi phí Tính toán RGA vs BCGG",
        )

    # 4. Qualitative Visualizations
    if run_all_flag or args.visualize:
        print("\n" + "=" * 90)
        print("TRỰC QUAN HÓA (QUALITATIVE ANALYSIS)")
        print(f"Dataset: {args.vis_dataset.upper()} | Samples: {args.num_vis}")
        print("=" * 90)
        try:
            model_bcgg, _ = load_gasnet_variant("convnext_bcgg_train", device=device)
            model_rga, _ = load_gasnet_variant("convnext_rga", device=device)

            # Load queries for visualization according to args.vis_dataset
            q_ds, g_ds = None, None
            if args.vis_dataset == "veri":
                try:
                    q_ds, g_ds = load_veri776(root_dir=args.data_root, legacy_resize=args.legacy_resize)
                except Exception as e:
                    print(f"[WARNING] Could not load VeRi-776 for visualization: {e}")
            elif args.vis_dataset == "vru":
                try:
                    q_ds, g_ds = load_vru_split(root_dir=args.data_root, split_name="Small", seed=42, legacy_resize=args.legacy_resize)
                except Exception as e:
                    print(f"[WARNING] Could not load VRU for visualization: {e}")
            elif args.vis_dataset == "vehicleid":
                try:
                    q_ds, g_ds = load_vehicleid(root_dir=args.data_root, legacy_resize=args.legacy_resize)
                except Exception as e:
                    print(f"[WARNING] Could not load VehicleID for visualization: {e}")
            elif args.vis_dataset == "vrai":
                try:
                    q_ds, g_ds = load_vrai_val(root_dir=args.data_root, num_ids=1000, seed=42, legacy_resize=args.legacy_resize)
                except Exception as e:
                    print(f"[WARNING] Could not load VRAI for visualization: {e}")

            # Fallback
            if q_ds is None or g_ds is None:
                try:
                    q_ds, g_ds = load_veri776(root_dir=args.data_root, legacy_resize=args.legacy_resize)
                except Exception:
                    try:
                        q_ds, g_ds = load_vru_split(root_dir=args.data_root, split_name="Small", seed=42, legacy_resize=args.legacy_resize)
                    except Exception:
                        pass

            if q_ds is not None and g_ds is not None:
                vis_dir = out_dir / "visualizations"
                run_qualitative_evaluation(
                    model=model_bcgg,
                    model_rga=model_rga,
                    query_dataset=q_ds,
                    gallery_dataset=g_ds,
                    output_dir=vis_dir,
                    num_queries_to_visualize=args.num_vis,
                    device=device,
                )
            else:
                print("[WARNING] Could not load datasets for visualization.")
        except Exception as exc:
            print(f"[WARNING] Error running qualitative visualizations: {exc}")

    print("\n" + "=" * 90)
    print(f"[DONE] Toàn bộ báo cáo và bảng số liệu đã được lưu tại: {out_dir.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()
