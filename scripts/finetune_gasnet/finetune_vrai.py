#!/usr/bin/env python3
"""Standalone Finetuning Script for GASNet on VRAI (Aerial Vehicle Re-ID).

Splits `data/VRAI/images_train` (6,302 identities, 66,113 images) into two strictly disjoint sets:
- Validation / Test split: 1,000 identities (identical to `load_vrai_val(seed=42)` in `eval_gasnet`)
- Finetune Train split: Remaining 5,302 identities (~55,600 images) — zero identity overlap!

Usage:
    # Finetune all 3 models (ResNet50-IBN+RGA, ConvNeXt+RGA, ConvNeXt+BCGG)
    python3 -m scripts.finetune_gasnet.finetune_vrai --variant all

    # Or finetune a single model
    python3 -m scripts.finetune_gasnet.finetune_vrai --variant convnext_bcgg --epochs 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_gasnet.datasets import SimpleReIDDataset, load_vrai_val, resolve_dataset_path
from scripts.eval_gasnet.run_all import save_markdown_and_csv
from scripts.finetune_gasnet.common import FINETUNE_VARIANTS, run_finetune_for_variant
from uav_reid.datasets.vrai import VRAITrainSample, read_vrai_train_samples


def prepare_vrai_train_and_val(
    root_dir: Optional[Union[str, Path]] = None,
    num_val_ids: int = 1000,
    split_seed: int = 42,
) -> Tuple[List[VRAITrainSample], SimpleReIDDataset, SimpleReIDDataset]:
    """Load VRAI train samples strictly excluding the `num_val_ids` validation identities."""
    vrai_dir = resolve_dataset_path("VRAI", root_dir)
    if not vrai_dir.exists():
        raise FileNotFoundError(f"VRAI directory not found at: {vrai_dir}")

    # 1. Load the exact 1,000-ID validation split used by eval_gasnet
    val_q, val_g = load_vrai_val(root_dir=root_dir, num_ids=num_val_ids, seed=split_seed)
    val_id_set = {s.vehicle_id for s in val_q.samples} | {s.vehicle_id for s in val_g.samples}

    # 2. Load all annotated VRAI samples and filter out all validation IDs
    all_samples = read_vrai_train_samples(vrai_dir)
    train_samples = [
        s for s in all_samples if (s.vehicle_id not in val_id_set) and s.img_path.is_file()
    ]
    train_id_set = {s.vehicle_id for s in train_samples}

    # Strict verification of zero identity leakage
    overlap = train_id_set & val_id_set
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} overlapping vehicle IDs!")

    print(
        f"[OK] VRAI Disjoint Split Verified (seed={split_seed}): "
        f"Train={len(train_id_set)} IDs ({len(train_samples)} imgs) | "
        f"Val={len(val_id_set)} IDs ({len(val_q)} queries, {len(val_g)} gallery) | "
        f"Overlap=0 IDs"
    )
    return train_samples, val_q, val_g


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Finetuning for GASNet on VRAI")
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
        default="weights/vrai",
        help="Directory to save finetuned checkpoints (default: weights/vrai)",
    )
    parser.add_argument("--val-ids", type=int, default=1000, help="Number of validation IDs held out (default: 1000)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for train/val ID split (must match eval_gasnet=42)")
    parser.add_argument("--epochs", type=int, default=12, help="Total finetuning epochs (default: 12)")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=2,
        help="Initial epochs to warmup only the new 5302-class classifier heads (default: 2)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="PK training batch size (default: 64)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--pk-k", type=int, default=4, help="Instances per identity in PK batch (default: 4)")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Override backbone learning rate")
    parser.add_argument("--gate-lr", type=float, default=1.0e-4, help="Gate + FS + BNNeck learning rate")
    parser.add_argument("--head-lr", type=float, default=3.5e-4, help="Classifier head learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="AdamW weight decay")
    parser.add_argument("--triplet-weight", type=float, default=0.5, help="Triplet loss weight")
    parser.add_argument("--eval-every", type=int, default=3, help="Evaluate on VRAI Val every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--strong-aug", action="store_true", help="Enable stronger data augmentation")
    parser.add_argument("--no-gem", action="store_true", help="Use AdaptiveAvgPool2d instead of GeMPool (p=3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_samples, val_q, val_g = prepare_vrai_train_and_val(
        root_dir=args.data_root,
        num_val_ids=args.val_ids,
        split_seed=args.split_seed,
    )

    variants = list(FINETUNE_VARIANTS) if args.variant == "all" else [args.variant]
    summary_rows = []

    for var in variants:
        ckpt_override = args.pretrained_ckpt if len(variants) == 1 else None
        res = run_finetune_for_variant(
            variant=var,
            train_samples=train_samples,
            val_q_dataset=val_q,
            val_g_dataset=val_g,
            dataset_label=f"VRAI (Val {args.val_ids} IDs)",
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
                "Initial mAP (%)": f"{res['init_mAP']:.2f}",
                "Finetuned mAP (%)": f"{res['best_mAP']:.2f}",
                "Rank-1 (%)": f"{res['best_Rank1']:.2f}",
                "Rank-5 (%)": f"{res['best_Rank5']:.2f}",
                "Rank-10 (%)": f"{res['best_Rank10']:.2f}",
                "mINP (%)": f"{res['best_mINP']:.2f}",
                "Best Epoch": res["best_epoch"],
                "Checkpoint": res["best_checkpoint"],
            }
        )

    if summary_rows:
        out_dir = Path(args.output_dir)
        summary_name = "finetune_vrai_avgpool_summary" if args.no_gem else "finetune_vrai_summary"
        save_markdown_and_csv(
            summary_rows,
            out_dir,
            summary_name,
            "Kết quả Finetune GASNet trên VRAI (5302 Train IDs -> 1000 Val IDs)",
        )


if __name__ == "__main__":
    main()
