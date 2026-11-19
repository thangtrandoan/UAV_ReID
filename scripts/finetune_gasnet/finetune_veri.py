#!/usr/bin/env python3
"""Standalone Finetuning Script for GASNet on VeRi-776.

Loads pretrained GASNet checkpoints (trained on VRU) from `weights/`,
finetunes on VeRi-776 training set (`data/VeRi/image_train`, 576 identities, 37,746 images),
and evaluates on VeRi-776 test set (`image_query` 1,678 vs `image_test` 11,579).

Usage:
    # Finetune all 3 models (ResNet50-IBN+RGA, ConvNeXt+RGA, ConvNeXt+BCGG)
    python3 -m scripts.finetune_gasnet.finetune_veri --variant all

    # Or finetune a single model
    python3 -m scripts.finetune_gasnet.finetune_veri --variant convnext_bcgg --epochs 15
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_gasnet.datasets import load_veri776, parse_veri_filename, resolve_dataset_path
from scripts.eval_gasnet.run_all import save_markdown_and_csv
from scripts.finetune_gasnet.common import (
    FINETUNE_VARIANTS,
    finetune_gasnet_from_config,
    load_veri_train_samples,
    run_finetune_for_variant,
)
from uav_reid.config import apply_cli_overrides, load_yaml_config
from uav_reid.datasets.vrai import VRAITrainSample

_XML_ITEM_RE = re.compile(
    r'imageName="([^"]+)"\s+vehicleID="([^"]+)"\s+cameraID="([^"]+)"\s+colorID="([^"]+)"\s+typeID="([^"]+)"'
)


def load_veri_train_samples(root_dir: Optional[Union[str, Path]] = None) -> List[VRAITrainSample]:
    """Load VeRi-776 training samples from `image_train/` and `train_label.xml`."""
    veri_dir = resolve_dataset_path("VeRi", root_dir)
    if not veri_dir.exists():
        raise FileNotFoundError(f"VeRi-776 directory not found at: {veri_dir}")

    train_img_dir = veri_dir / "image_train"
    if not train_img_dir.is_dir():
        raise FileNotFoundError(f"VeRi-776 image_train directory not found at: {train_img_dir}")

    xml_path = veri_dir / "train_label.xml"
    samples: List[VRAITrainSample] = []

    if xml_path.is_file():
        raw_text = xml_path.read_text(encoding="latin-1", errors="ignore")
        for match in _XML_ITEM_RE.finditer(raw_text):
            img_name, vid_str, cam_str, color_str, type_str = match.groups()
            img_path = train_img_dir / img_name
            if not img_path.is_file():
                continue
            samples.append(
                VRAITrainSample(
                    img_path=img_path,
                    image_name=img_name,
                    vehicle_id=int(vid_str),
                    camera=cam_str,
                    color=int(color_str) if color_str.isdigit() else -1,
                    vehicle_type=int(type_str) if type_str.isdigit() else -1,
                )
            )

    if not samples:
        # Fallback to name_train.txt or directory listing
        name_train_file = veri_dir / "name_train.txt"
        if name_train_file.is_file():
            names = [
                line.strip()
                for line in name_train_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            names = sorted(p.name for p in train_img_dir.glob("*.jpg"))

        for img_name in names:
            img_path = train_img_dir / img_name
            if not img_path.is_file():
                continue
            vid, cam_id = parse_veri_filename(img_name)
            samples.append(
                VRAITrainSample(
                    img_path=img_path,
                    image_name=img_name,
                    vehicle_id=vid,
                    camera=f"c{cam_id:03d}" if cam_id >= 0 else "",
                    color=-1,
                    vehicle_type=-1,
                )
            )

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Finetuning for GASNet on VeRi-776")
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
        default="weights/veri",
        help="Directory to save finetuned checkpoints (default: weights/veri)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Total finetuning epochs (default: 15)")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=2,
        help="Initial epochs to warmup only the new 576-class classifier heads (default: 2)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="PK training batch size (default: 64)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--pk-k", type=int, default=4, help="Instances per identity in PK batch (default: 4)")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Override backbone learning rate")
    parser.add_argument("--gate-lr", type=float, default=1.0e-4, help="Gate + FS + BNNeck learning rate")
    parser.add_argument("--head-lr", type=float, default=3.5e-4, help="Classifier head learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="AdamW weight decay")
    parser.add_argument("--triplet-weight", type=float, default=0.5, help="Triplet loss weight")
    parser.add_argument("--eval-every", type=int, default=3, help="Evaluate on VeRi test set every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--strong-aug", action="store_true", help="Enable stronger data augmentation")
    parser.add_argument("--no-gem", action="store_true", help="Use AdaptiveAvgPool2d instead of GeMPool (p=3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        cfg.setdefault("train", {})["dataset"] = "veri"
        apply_cli_overrides(cfg, args.set)
        finetune_gasnet_from_config(cfg)
        return

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_samples = load_veri_train_samples(root_dir=args.data_root)
    val_q, val_g = load_veri776(root_dir=args.data_root)
    num_train_ids = len({s.vehicle_id for s in train_samples})
    print(
        f"[DATA] VeRi-776 Ready | Train: {len(train_samples)} images ({num_train_ids} IDs) | "
        f"Test Query: {len(val_q)} | Test Gallery: {len(val_g)}"
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
            dataset_label="VeRi-776",
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
                "Zero-Shot mAP (%)": f"{res['init_mAP']:.2f}",
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
        summary_name = "finetune_veri_avgpool_summary" if args.no_gem else "finetune_veri_summary"
        save_markdown_and_csv(
            summary_rows,
            out_dir,
            summary_name,
            "Kết quả Finetune GASNet trên VeRi-776",
        )


if __name__ == "__main__":
    main()
