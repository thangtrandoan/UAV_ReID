#!/usr/bin/env python3
"""Standalone Finetuning Script for GASNet on PKU-VehicleID (V1.0).

Protocol:
- Train set: `data/VehicleID_V1.0/train_test_split/train_list.txt` (13,164 IDs, 113,346 images)
- Validation during training:
    * Small (`test_list_800.txt`: 800 gallery, 5,693 queries)
    * Medium (`test_list_1600.txt`: 1,600 gallery, 11,777 queries) -> Used to select the BEST checkpoint!
- Final held-out test after training completes:
    * Large (`test_list_2400.txt`: 2,400 gallery, 17,377 queries) -> Evaluated using the best Medium checkpoint.

Usage:
    # Finetune all 3 models (ResNet50-IBN+RGA, ConvNeXt+RGA, ConvNeXt+BCGG)
    python3 -m scripts.finetune_gasnet.finetune_vehicleid --variant all

    # Or finetune a single model
    python3 -m scripts.finetune_gasnet.finetune_vehicleid --variant convnext_bcgg --epochs 15

    # Or run via YAML config
    python3 -m scripts.finetune_gasnet.finetune_vehicleid --config configs/gasnet_finetune.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_gasnet.run_all import save_markdown_and_csv
from scripts.finetune_gasnet.common import (
    FINETUNE_VARIANTS,
    finetune_gasnet_from_config,
    prepare_vehicleid_splits,
    run_finetune_for_variant,
)
from uav_reid.config import apply_cli_overrides, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Finetuning for GASNet on PKU-VehicleID (Val: Small+Medium, Best: Medium, Final Test: Large)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (e.g. configs/gasnet_finetune.yaml). If provided, uses config-driven mode.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override YAML config values via section.key=value",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["resnet_rga", "convnext_rga", "convnext_bcgg", "all"],
        default="all",
        help="Model variant to finetune ('all' finetunes all 3 variants sequentially with the same recipe)",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing datasets")
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default="weights",
        help="Directory containing initial pretrained checkpoints",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="Explicit path to a single pretrained checkpoint (only used when --variant != 'all')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/vehicleid",
        help="Directory to save finetuned checkpoints (default: weights/vehicleid)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for 1-shot gallery selection on Small/Medium/Large (default: 42)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Total finetuning epochs (default: 15)")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=2,
        help="Initial epochs to warmup only the new 13164-class classifier heads (default: 2)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="PK training batch size (default: 64)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--pk-k", type=int, default=4, help="Instances per identity in PK batch (default: 4)")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Override backbone learning rate")
    parser.add_argument("--gate-lr", type=float, default=1.0e-4, help="Gate + FS + BNNeck learning rate")
    parser.add_argument("--head-lr", type=float, default=3.5e-4, help="Classifier head learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="AdamW weight decay")
    parser.add_argument("--triplet-weight", type=float, default=0.5, help="Triplet loss weight")
    parser.add_argument("--eval-every", type=int, default=3, help="Evaluate on Small & Medium every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--strong-aug", action="store_true", help="Enable stronger data augmentation")
    parser.add_argument("--no-gem", action="store_true", help="Use AdaptiveAvgPool2d instead of GeMPool (p=3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        cfg.setdefault("train", {})["dataset"] = "vehicleid"
        apply_cli_overrides(cfg, args.set)
        finetune_gasnet_from_config(cfg)
        return

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    (
        train_samples,
        _val_small_q,
        _val_small_g,
        val_med_q,
        val_med_g,
        _test_large_q,
        _test_large_g,
    ) = prepare_vehicleid_splits(
        root_dir=args.data_root,
        split_seed=args.split_seed,
    )

    variants = list(FINETUNE_VARIANTS) if args.variant == "all" else [args.variant]
    summary_rows = []

    for var in variants:
        ckpt_override = args.pretrained_ckpt if len(variants) == 1 else None
        res = run_finetune_for_variant(
            variant=var,
            train_samples=train_samples,
            val_q_dataset=val_med_q,
            val_g_dataset=val_med_g,
            dataset_label="VehicleID Medium (1600)",
            pretrained_dir=args.pretrained_dir,
            pretrained_ckpt=ckpt_override,
            output_dir=args.output_dir,
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            pk_k=args.pk_k,
            backbone_lr=args.backbone_lr,
            gate_lr=args.gate_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            triplet_weight=args.triplet_weight,
            eval_every=args.eval_every,
            num_workers=args.num_workers,
            strong_aug=args.strong_aug,
            use_gem=(not args.no_gem),
            seed=args.seed,
            device=device,
        )
        summary_rows.append(
            {
                "Variant": var,
                "Pooling": "AvgPool" if args.no_gem else "GeM (p=3.0)",
                "Best Epoch (by Medium)": res["best_epoch"],
                "Small (800) mAP (%)": f"{res.get('small_mAP', 0.0):.2f}",
                "Small (800) Rank-1 (%)": f"{res.get('small_Rank1', 0.0):.2f}",
                "Small (800) Rank-5 (%)": f"{res.get('small_Rank5', 0.0):.2f}",
                "Medium (1600) mAP (%)": f"{res['best_mAP']:.2f}",
                "Medium (1600) Rank-1 (%)": f"{res['best_Rank1']:.2f}",
                "Medium (1600) Rank-5 (%)": f"{res['best_Rank5']:.2f}",
                "Large (2400) Test mAP (%)": f"{res.get('large_mAP', 0.0):.2f}",
                "Large (2400) Test Rank-1 (%)": f"{res.get('large_Rank1', 0.0):.2f}",
                "Large (2400) Test Rank-5 (%)": f"{res.get('large_Rank5', 0.0):.2f}",
                "Checkpoint": res["best_checkpoint"],
            }
        )

    if summary_rows:
        out_dir = Path(args.output_dir)
        summary_name = (
            "finetune_vehicleid_avgpool_summary" if args.no_gem else "finetune_vehicleid_summary"
        )
        save_markdown_and_csv(
            summary_rows,
            out_dir,
            summary_name,
            "Kết quả Finetune GASNet trên PKU-VehicleID (Val: Small 800 & Medium 1600, Best: Medium 1600, Final Test: Large 2400)",
        )


if __name__ == "__main__":
    main()
