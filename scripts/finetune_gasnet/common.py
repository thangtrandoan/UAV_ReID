"""Shared Finetuning Engine for GASNet on VeRi-776 and VRAI.

Implements:
- Pretrained checkpoint loading with new classifier head (num_classes adaptation)
- Two-stage finetuning:
  1. Classifier Head Warmup (backbone & BNNeck frozen in eval mode to prevent feature drift)
  2. End-to-End Discriminative LR Finetuning (Backbone < Gate/Neck < Classifier Head)
- Zero-shot (Epoch 0) baseline evaluation and periodic mAP/CMC/mINP tracking
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.eval_gasnet.datasets import SimpleReIDDataset
from scripts.eval_gasnet.models import find_checkpoint
from scripts.eval_gasnet.run_all import evaluate_dataset_on_model
from uav_reid.datasets.vrai import VRAITrainDataset, VRAITrainSample, make_pk_loader
from uav_reid.gasnet.bcgg import infer_gate_config
from uav_reid.gasnet.model import GASNet, infer_backbone_name, infer_econvnext
from uav_reid.reidnet.losses import batch_hard_triplet_loss
from uav_reid.utils import choose_amp_dtype, configure_cuda_for_speed, normalize_state_dict, set_seed

FINETUNE_VARIANTS = ("resnet_rga", "convnext_rga", "convnext_bcgg")


def build_finetune_train_transform(strong_aug: bool = False) -> transforms.Compose:
    """Build training transform for GASNet finetuning (224x224)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if strong_aug:
        return transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.80, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)],
                p=0.6,
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
    ])


def load_pretrained_gasnet_for_finetune(
    variant: str,
    num_classes: int,
    pretrained_dir: Union[str, Path] = "weights",
    pretrained_ckpt: Optional[Union[str, Path]] = None,
    use_gem: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[GASNet, Path]:
    """Load a pretrained GASNet checkpoint and re-initialize classifiers for `num_classes`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variant == "resnet_rga":
        default_bb = "resnet50_ibn"
        gate = "rga"
        ckpt_key = "resnet_rga"
    elif variant == "convnext_rga":
        default_bb = "dinov3_convnext"
        gate = "rga"
        ckpt_key = "convnext_rga"
    elif variant == "convnext_bcgg":
        default_bb = "dinov3_convnext"
        gate = "bcgg"
        ckpt_key = "convnext_bcgg"
    else:
        raise ValueError(f"Unsupported variant {variant!r}. Choose from {FINETUNE_VARIANTS}")

    ckpt_file = (
        Path(pretrained_ckpt).resolve()
        if pretrained_ckpt is not None
        else find_checkpoint(ckpt_key, custom_weights_dir=pretrained_dir)
    )
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found at: {ckpt_file}")

    raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    state_dict = normalize_state_dict(raw)

    backbone = infer_backbone_name(state_dict, default=default_bb)
    econvnext = infer_econvnext(state_dict)
    gate_config = infer_gate_config(state_dict) if gate == "bcgg" else {"gate": "rga"}

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        backbone=backbone,
        econvnext=econvnext,
        use_gem=use_gem,
        **gate_config,
    )

    model_sd = model.state_dict()
    # Transfer all matching feature extractor & BNNeck parameters; skip mismatched classifier heads
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k in model_sd and v.shape == model_sd[k].shape and not k.startswith("classifier_")
    }
    # If classifier happens to have the exact same num_classes, we can also load it
    for cls_k in ("classifier_global.weight", "classifier_fs.weight"):
        if cls_k in state_dict and cls_k in model_sd and state_dict[cls_k].shape == model_sd[cls_k].shape:
            filtered[cls_k] = state_dict[cls_k]

    model.load_state_dict(filtered, strict=False)
    print(
        f"[OK] Loaded {len(filtered)}/{len(model_sd)} tensors from {ckpt_file.name} "
        f"| backbone={backbone} (econvnext={econvnext}) | gate={gate} | target_classes={num_classes}"
    )

    return model.to(device), ckpt_file


def set_warmup_freeze_state(model: GASNet, freeze_feature_extractor: bool) -> None:
    """Freeze or unfreeze feature extractor (backbone + gate + fs + bnneck) for head warmup."""
    for name, param in model.named_parameters():
        if name.startswith("classifier_"):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze_feature_extractor

    if freeze_feature_extractor:
        # Keep BatchNorm running stats frozen during classifier warmup
        model.eval()
        model.classifier_global.train()
        model.classifier_fs.train()
    else:
        model.train()


def build_finetune_optimizer_and_scheduler(
    model: GASNet,
    backbone_lr: float,
    gate_lr: float,
    head_lr: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int = 2,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create AdamW optimizer with 3 discriminative LR groups and Cosine scheduler."""
    backbone_params: List[nn.Parameter] = []
    gate_neck_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if name.startswith("classifier_"):
            head_params.append(param)
        elif any(
            name.startswith(prefix)
            for prefix in ("convnext_backbone.", "swin_backbone.", "stem.", "layer1.", "layer2.", "layer3.", "layer4.")
        ):
            backbone_params.append(param)
        else:
            gate_neck_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": gate_neck_params, "lr": gate_lr, "name": "gate_and_neck"},
        {"params": head_params, "lr": head_lr, "name": "classifiers"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    return optimizer, scheduler


def run_finetune_for_variant(
    variant: str,
    train_samples: List[VRAITrainSample],
    val_q_dataset: SimpleReIDDataset,
    val_g_dataset: SimpleReIDDataset,
    dataset_label: str,
    pretrained_dir: Union[str, Path] = "weights",
    pretrained_ckpt: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "weights/finetuned",
    epochs: int = 15,
    freeze_epochs: int = 2,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    pk_k: int = 4,
    backbone_lr: Optional[float] = None,
    gate_lr: float = 1.0e-4,
    head_lr: float = 3.5e-4,
    weight_decay: float = 0.02,
    triplet_weight: float = 0.5,
    triplet_margin: float = 0.3,
    label_smoothing: float = 0.1,
    eval_every: int = 2,
    num_workers: int = 4,
    strong_aug: bool = False,
    use_gem: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run complete finetuning pipeline for one GASNet variant."""
    set_seed(seed)
    configure_cuda_for_speed()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_suffix = "" if use_gem else "_avgpool"
    best_save_path = out_dir / f"gasnet_{variant}{pool_suffix}.best.pth"
    last_save_path = out_dir / f"gasnet_{variant}{pool_suffix}.last.pth"

    # Build label map for training IDs
    train_ids = sorted({s.vehicle_id for s in train_samples})
    label_map = {vid: idx for idx, vid in enumerate(train_ids)}
    num_classes = len(label_map)

    if batch_size % pk_k != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by pk_k ({pk_k})")
    pk_p = batch_size // pk_k

    train_tfm = build_finetune_train_transform(strong_aug=strong_aug)
    train_cameras = [s.camera for s in train_samples]
    train_colors = [s.color for s in train_samples]
    train_types = [s.vehicle_type for s in train_samples]
    has_valid_attrs = any(c >= 0 for c in train_colors) and any(t >= 0 for t in train_types)

    train_loader = make_pk_loader(
        samples=train_samples,
        transform=train_tfm,
        label_map=label_map,
        p=pk_p,
        k=pk_k,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else 2,
        cameras=train_cameras,
        colors=train_colors if has_valid_attrs else None,
        vehicle_types=train_types if has_valid_attrs else None,
        camera_balanced=True,
        attribute_hard_negative=has_valid_attrs,
        dataset_cls=VRAITrainDataset,
    )

    # Default backbone LR: smaller for DINOv3 ConvNeXt than ResNet50-IBN
    if backbone_lr is None:
        backbone_lr = 1.5e-5 if "convnext" in variant else 3.5e-5

    pool_label = "GeM (p=3.0)" if use_gem else "AvgPool"
    print("\n" + "=" * 90)
    print(f"[FINETUNE] Dataset: {dataset_label} | Variant: {variant} | Pooling: {pool_label}")
    print(
        f"[CONFIG] Train IDs: {num_classes} ({len(train_samples)} imgs) | "
        f"Epochs: {epochs} (Head warmup: {freeze_epochs}) | Batch: {batch_size} (P={pk_p}, K={pk_k})"
    )
    print(
        f"[LR] Backbone: {backbone_lr:.1e} | Gate+Neck: {gate_lr:.1e} | Classifier Head: {head_lr:.1e}"
    )
    print("=" * 90)

    model, src_ckpt = load_pretrained_gasnet_for_finetune(
        variant=variant,
        num_classes=num_classes,
        pretrained_dir=pretrained_dir,
        pretrained_ckpt=pretrained_ckpt,
        use_gem=use_gem,
        device=device,
    )

    # 0. Evaluate Zero-Shot Baseline (Epoch 0)
    print(f"[EVAL] Evaluating Epoch 0 (Pretrained Zero-Shot Baseline) on {dataset_label}...")
    model.eval()
    init_metrics = evaluate_dataset_on_model(
        model, val_q_dataset, val_g_dataset, batch_size=eval_batch_size, device=device
    )
    best_map = float(init_metrics["mAP"])
    best_metrics = dict(init_metrics)
    best_epoch = 0
    print(
        f" -> [Epoch 0 Baseline] mAP: {init_metrics['mAP']:.2f}% | "
        f"Rank-1: {init_metrics['Rank-1']:.2f}% | Rank-5: {init_metrics['Rank-5']:.2f}% | "
        f"Rank-10: {init_metrics['Rank-10']:.2f}% | mINP: {init_metrics['mINP']:.2f}%"
    )

    # Initialize best checkpoint with the pretrained model so best_save_path is never worse than baseline
    torch.save(model.state_dict(), best_save_path)

    optimizer, scheduler = build_finetune_optimizer_and_scheduler(
        model=model,
        backbone_lr=backbone_lr,
        gate_lr=gate_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_epochs=max(1, freeze_epochs),
    )

    use_amp = device.type == "cuda"
    amp_dtype = choose_amp_dtype("auto") if use_amp else torch.float32
    use_scaler = use_amp and (amp_dtype == torch.float16)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        is_frozen_phase = epoch <= freeze_epochs
        set_warmup_freeze_state(model, freeze_feature_extractor=is_frozen_phase)

        running_loss = 0.0
        num_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            imgs, labels, _, _, cameras, colors, vehicle_types = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            colors_t = colors.to(device, non_blocking=True) if has_valid_attrs else None
            types_t = vehicle_types.to(device, non_blocking=True) if has_valid_attrs else None

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                feats, logits = model(imgs)
                loss_id = sum(ce_loss(logit, labels) for logit in logits)
                if is_frozen_phase:
                    # During classifier warmup, feature extractor is frozen so only ID loss has gradients
                    loss = loss_id
                else:
                    loss_tri = sum(
                        batch_hard_triplet_loss(
                            feat,
                            labels,
                            margin=triplet_margin,
                            cameras=cameras,
                            colors=colors_t,
                            vehicle_types=types_t,
                        )
                        for feat in feats
                    )
                    loss = loss_id + triplet_weight * loss_tri

            if use_amp and use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            num_samples += bs

        scheduler.step()
        avg_loss = running_loss / max(1, num_samples)
        phase_tag = "HEAD-WARMUP" if is_frozen_phase else "FULL-FINETUNE"
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] ({phase_tag}) "
            f"loss={avg_loss:.4f} | time={elapsed:.1f}s"
        )

        if (eval_every > 0 and epoch % eval_every == 0) or epoch == epochs:
            model.eval()
            cur_metrics = evaluate_dataset_on_model(
                model, val_q_dataset, val_g_dataset, batch_size=eval_batch_size, device=device
            )
            cur_map = float(cur_metrics["mAP"])
            improved = cur_map > best_map
            marker = " *NEW BEST*" if improved else ""
            print(
                f" -> [Eval Epoch {epoch:02d}] mAP: {cur_map:.2f}% | "
                f"Rank-1: {cur_metrics['Rank-1']:.2f}% | Rank-5: {cur_metrics['Rank-5']:.2f}% | "
                f"Rank-10: {cur_metrics['Rank-10']:.2f}% | mINP: {cur_metrics['mINP']:.2f}%{marker}"
            )
            if improved:
                best_map = cur_map
                best_metrics = dict(cur_metrics)
                best_epoch = epoch
                torch.save(model.state_dict(), best_save_path)
                print(f"    [OK] Saved improved checkpoint -> {best_save_path}")

    torch.save(model.state_dict(), last_save_path)
    print("-" * 90)
    print(
        f"[SUMMARY] {variant} on {dataset_label}: "
        f"mAP {init_metrics['mAP']:.2f}% (Epoch 0) -> {best_map:.2f}% (Best @ Epoch {best_epoch}) | "
        f"Rank-1: {init_metrics['Rank-1']:.2f}% -> {best_metrics['Rank-1']:.2f}%"
    )
    print(f"[SAVED] Best checkpoint: {best_save_path}")
    print("-" * 90)

    return {
        "variant": variant,
        "dataset": dataset_label,
        "init_mAP": init_metrics["mAP"],
        "init_Rank1": init_metrics["Rank-1"],
        "best_mAP": best_map,
        "best_Rank1": best_metrics["Rank-1"],
        "best_Rank5": best_metrics["Rank-5"],
        "best_Rank10": best_metrics["Rank-10"],
        "best_mINP": best_metrics["mINP"],
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_save_path),
    }
