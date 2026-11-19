"""Config-driven & Standalone Finetuning Engine for GASNet (VeRi-776 / VRAI / VRU).

Supports two usage styles:
1. YAML Config style (just like `train_gasnet.py`):
   - Configure `paths.data_root`, `paths.weight_path`, `paths.save_path`,
     `model.backbone`, `model.gate`, `model.use_gem`, `train.dataset` (`veri` | `vrai` | `vru`), etc.
   - Run via `finetune_gasnet_from_config(cfg)`.
2. Programmatic / Variant CLI style via `run_finetune_for_variant(...)`.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms

from scripts.eval_gasnet.datasets import (
    SimpleReIDDataset,
    load_veri776,
    load_vrai_val,
    load_vru_split,
    parse_veri_filename,
    resolve_dataset_path,
)
from scripts.eval_gasnet.models import find_checkpoint
from scripts.eval_gasnet.run_all import evaluate_dataset_on_model
from uav_reid.datasets.vrai import VRAITrainDataset, VRAITrainSample, make_pk_loader, read_vrai_train_samples
from uav_reid.datasets.vru import read_split_file
from uav_reid.gasnet.bcgg import infer_gate_config
from uav_reid.gasnet.model import GASNet, infer_backbone_name, infer_econvnext
from uav_reid.reidnet.losses import batch_hard_triplet_loss
from uav_reid.utils import (
    Logger,
    choose_amp_dtype,
    configure_cuda_for_speed,
    normalize_state_dict,
    set_seed,
)

FINETUNE_VARIANTS = ("resnet_rga", "convnext_rga", "convnext_bcgg")

_XML_ITEM_RE = re.compile(
    r'imageName="([^"]+)"\s+vehicleID="([^"]+)"\s+cameraID="([^"]+)"\s+colorID="([^"]+)"\s+typeID="([^"]+)"'
)


# --------------------------------------------------------------------------- #
# Dataset Builders for VeRi-776, VRAI (Disjoint Split), and VRU
# --------------------------------------------------------------------------- #
def load_veri_train_samples(root_dir: Optional[Union[str, Path]] = None) -> List[VRAITrainSample]:
    """Load VeRi-776 training samples from `image_train/` and `train_label.xml`."""
    veri_dir = resolve_dataset_path("VeRi", root_dir)
    if not veri_dir.exists():
        raise FileNotFoundError(f"[FAIL] VeRi-776 directory not found at: {veri_dir}")

    train_img_dir = veri_dir / "image_train"
    if not train_img_dir.is_dir():
        raise FileNotFoundError(f"[FAIL] VeRi-776 image_train directory not found at: {train_img_dir}")

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


def prepare_vrai_train_and_val(
    root_dir: Optional[Union[str, Path]] = None,
    num_val_ids: int = 1000,
    split_seed: int = 42,
    legacy_resize: bool = False,
) -> Tuple[List[VRAITrainSample], SimpleReIDDataset, SimpleReIDDataset]:
    """Load VRAI train samples strictly excluding the `num_val_ids` validation identities."""
    vrai_dir = resolve_dataset_path("VRAI", root_dir)
    if not vrai_dir.exists():
        raise FileNotFoundError(f"[FAIL] VRAI directory not found at: {vrai_dir}")

    val_q, val_g = load_vrai_val(
        root_dir=root_dir, num_ids=num_val_ids, seed=split_seed, legacy_resize=legacy_resize
    )
    val_id_set = {s.vehicle_id for s in val_q.samples} | {s.vehicle_id for s in val_g.samples}

    all_samples = read_vrai_train_samples(vrai_dir)
    train_samples = [
        s for s in all_samples if (s.vehicle_id not in val_id_set) and s.img_path.is_file()
    ]
    train_id_set = {s.vehicle_id for s in train_samples}

    overlap = train_id_set & val_id_set
    if overlap:
        raise RuntimeError(f"[FAIL] Data leakage detected: {len(overlap)} overlapping vehicle IDs!")

    print(
        f"[OK] VRAI Disjoint Split Verified (seed={split_seed}): "
        f"Train={len(train_id_set)} IDs ({len(train_samples)} imgs) | "
        f"Val={len(val_id_set)} IDs ({len(val_q)} queries, {len(val_g)} gallery) | "
        f"Overlap=0 IDs"
    )
    return train_samples, val_q, val_g


def prepare_vru_train_and_val(
    root_dir: Optional[Union[str, Path]] = None,
    split_seed: int = 42,
    legacy_resize: bool = False,
) -> Tuple[List[VRAITrainSample], SimpleReIDDataset, SimpleReIDDataset]:
    """Load VRU train samples and VRU Small validation split."""
    vru_dir = resolve_dataset_path("VRU", root_dir)
    if not vru_dir.exists():
        raise FileNotFoundError(f"[FAIL] VRU directory not found at: {vru_dir}")

    raw_samples = read_split_file(vru_dir / "Pic", vru_dir / "train_test_split" / "train_list.txt")
    train_samples = [
        VRAITrainSample(
            img_path=s.img_path,
            image_name=s.img_path.name,
            vehicle_id=s.vehicle_id,
            camera="",
            color=-1,
            vehicle_type=-1,
        )
        for s in raw_samples
        if s.img_path.is_file()
    ]
    val_q, val_g = load_vru_split(
        root_dir=root_dir, split_name="Small", seed=split_seed, legacy_resize=legacy_resize
    )
    return train_samples, val_q, val_g


# --------------------------------------------------------------------------- #
# Transforms, Model Construction & Pretrained Checkpoint Loading
# --------------------------------------------------------------------------- #
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


def _resolve_weight_path_from_model_cfg(
    mc: Dict[str, Any],
    pc: Dict[str, Any],
) -> Optional[Path]:
    """Resolve `paths.weight_path` (or auto-select from `weights/` based on backbone & gate)."""
    raw_wp = pc.get("weight_path") or pc.get("pretrained_path") or pc.get("checkpoint")
    if raw_wp and str(raw_wp).strip().lower() not in ("none", "null", "false", "auto"):
        wp = Path(str(raw_wp))
        if not wp.is_file():
            raise FileNotFoundError(f"[FAIL] paths.weight_path does not exist: {wp}")
        return wp.resolve()

    if str(raw_wp).strip().lower() in ("none", "null", "false"):
        return None

    # Auto-resolve from weights/ based on model.backbone and model.gate
    backbone = str(mc.get("backbone", "dinov3_convnext")).lower()
    gate = str(mc.get("gate", "bcgg")).lower()
    if "resnet" in backbone:
        key = "resnet_rga"
    elif gate == "rga":
        key = "convnext_rga"
    else:
        key = "convnext_bcgg"
    return find_checkpoint(key)


def build_and_load_gasnet_from_config(
    mc: Dict[str, Any],
    weight_path: Optional[Path],
    num_classes: int,
    device: torch.device,
) -> GASNet:
    """Construct GASNet from `cfg['model']` and validate/load `weight_path`."""
    backbone = str(mc.get("backbone", "dinov3_convnext"))
    gate = str(mc.get("gate", "bcgg"))
    econvnext = bool(mc.get("econvnext", True))
    use_gem = bool(mc.get("use_gem", True))
    gem_p = float(mc.get("gem_p", 3.0))
    use_part_branch = bool(mc.get("use_part_branch", False))
    num_parts = int(mc.get("num_parts", 4))
    use_attention_local = bool(mc.get("use_attention_local", False))
    num_attention_heads = int(mc.get("num_attention_heads", 4))

    gate_kwargs = {
        "gate": gate,
        "gate_hidden": int(mc.get("gate_hidden", 16)),
        "gate_groups": int(mc.get("gate_groups", 4)),
        "gate_kernel": int(mc.get("gate_kernel", 7)),
        "gate_channel_kernel": int(mc.get("gate_channel_kernel", 5)),
        "gate_inject_value": bool(mc.get("gate_inject_value", False)),
    }

    state_dict = None
    if weight_path is not None:
        raw = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            raw = raw["model_state_dict"]
        state_dict = normalize_state_dict(raw)

        # Validate backbone compatibility between YAML config and checkpoint
        ckpt_bb = infer_backbone_name(state_dict, default=backbone)
        if backbone in ("resnet50", "resnet50_ibn") and ckpt_bb in ("resnet50", "resnet50_ibn"):
            if ckpt_bb != backbone:
                print(
                    f"[INFO] Adapting model.backbone '{backbone}' -> '{ckpt_bb}' to match checkpoint {weight_path.name}"
                )
                backbone = ckpt_bb
        elif ckpt_bb != backbone:
            raise ValueError(
                f"[FAIL] Config model.backbone='{backbone}' does not match checkpoint "
                f"{weight_path.name} (detected backbone='{ckpt_bb}'). "
                f"Please align model.backbone and paths.weight_path in your config."
            )

        # Validate gate compatibility
        ckpt_has_bcgg = any(".reduce." in k or ".context." in k for k in state_dict.keys())
        ckpt_has_rga = any(".rga_s." in k or ".rga_c." in k for k in state_dict.keys())
        ckpt_gate = "bcgg" if ckpt_has_bcgg else ("rga" if ckpt_has_rga else gate)
        if ckpt_gate != gate:
            raise ValueError(
                f"[FAIL] Config model.gate='{gate}' does not match checkpoint "
                f"{weight_path.name} (detected gate='{ckpt_gate}'). "
                f"Please align model.gate and paths.weight_path in your config."
            )

        ckpt_econv = infer_econvnext(state_dict)
        if backbone == "dinov3_convnext" and ckpt_econv != econvnext:
            print(
                f"[INFO] Adapting model.econvnext={econvnext} -> {ckpt_econv} to match checkpoint {weight_path.name}"
            )
            econvnext = ckpt_econv

        if gate == "bcgg":
            inferred_gate = infer_gate_config(state_dict)
            gate_kwargs.update(inferred_gate)

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=(False if state_dict is not None else bool(mc.get("pretrained", True))),
        use_part_branch=use_part_branch,
        num_parts=num_parts,
        use_gem=use_gem,
        gem_p=gem_p,
        backbone=backbone,
        use_attention_local=use_attention_local,
        num_attention_heads=num_attention_heads,
        econvnext=econvnext,
        **gate_kwargs,
    )

    if state_dict is not None:
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_sd and v.shape == model_sd[k].shape and not k.startswith("classifier_")
        }
        reinit_classifiers = []
        for cls_k in ("classifier_global.weight", "classifier_fs.weight"):
            if cls_k in state_dict and cls_k in model_sd and state_dict[cls_k].shape == model_sd[cls_k].shape:
                filtered[cls_k] = state_dict[cls_k]
            else:
                reinit_classifiers.append(cls_k)

        missing_non_cls = [
            k for k in model_sd.keys() if k not in filtered and not k.startswith("classifier_")
        ]
        if missing_non_cls:
            raise RuntimeError(
                f"[FAIL] Critical feature extractor keys missing from {weight_path.name}: {missing_non_cls[:10]}"
            )

        model.load_state_dict(filtered, strict=False)
        model._classifiers_loaded = len(reinit_classifiers) == 0
        print(
            f"[OK] Loaded {len(filtered)}/{len(model_sd)} tensors from {weight_path} "
            f"| backbone={backbone} (econvnext={econvnext}) | gate={gate} | use_gem={use_gem} | "
            f"re-initialized classifiers={reinit_classifiers} for num_classes={num_classes}"
        )

    return model.to(device)


def load_pretrained_gasnet_for_finetune(
    variant: str,
    num_classes: int,
    pretrained_dir: Union[str, Path] = "weights",
    pretrained_ckpt: Optional[Union[str, Path]] = None,
    use_gem: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[GASNet, Path]:
    """Load a pretrained GASNet variant by name (`resnet_rga`, `convnext_rga`, `convnext_bcgg`)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variant == "resnet_rga":
        mc = {"backbone": "resnet50_ibn", "gate": "rga", "use_gem": use_gem}
        ckpt_key = "resnet_rga"
    elif variant == "convnext_rga":
        mc = {"backbone": "dinov3_convnext", "gate": "rga", "use_gem": use_gem}
        ckpt_key = "convnext_rga"
    elif variant == "convnext_bcgg":
        mc = {"backbone": "dinov3_convnext", "gate": "bcgg", "use_gem": use_gem}
        ckpt_key = "convnext_bcgg"
    else:
        raise ValueError(f"Unsupported variant {variant!r}. Choose from {FINETUNE_VARIANTS}")

    ckpt_file = (
        Path(pretrained_ckpt).resolve()
        if pretrained_ckpt is not None
        else find_checkpoint(ckpt_key, custom_weights_dir=pretrained_dir)
    )
    model = build_and_load_gasnet_from_config(mc, ckpt_file, num_classes=num_classes, device=device)
    return model, ckpt_file


def set_warmup_freeze_state(model: nn.Module, freeze_feature_extractor: bool) -> None:
    """Freeze or unfreeze feature extractor (backbone + gate + fs + bnneck) for head warmup.

    Keeps top-level `model.training = True` and `base_model.training = True` so
    `GASNet.forward()` always takes the `if self.training:` branch and returns `(feats, logits)`.
    During head warmup, only BatchNorm and Dropout submodules are set to `.eval()`.
    """
    model.train()
    base_model = getattr(model, "_orig_mod", model)
    base_model.train()

    for name, param in base_model.named_parameters():
        if name.startswith("classifier_"):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze_feature_extractor

    if freeze_feature_extractor:
        for module in base_model.modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
                module.eval()


def build_finetune_optimizer_and_scheduler(
    model: nn.Module,
    backbone_lr: float,
    gate_lr: float,
    head_lr: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int = 2,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create AdamW optimizer with 3 discriminative LR groups and Cosine scheduler."""
    base_model = getattr(model, "_orig_mod", model)
    backbone_params: List[nn.Parameter] = []
    gate_neck_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for name, param in base_model.named_parameters():
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


def _resolve_best_and_last_paths(save_path: Path) -> Tuple[Path, Path]:
    """Derive `<stem>.best.pth` and `<stem>.last.pth` cleanly without doubling `.best.best.pth`."""
    name = save_path.name
    if name.endswith(".best.pth"):
        base_stem = name[: -len(".best.pth")]
        return save_path, save_path.with_name(f"{base_stem}.last.pth")
    if name.endswith(".last.pth"):
        base_stem = name[: -len(".last.pth")]
        return save_path.with_name(f"{base_stem}.best.pth"), save_path
    return (
        save_path.with_name(f"{save_path.stem}.best{save_path.suffix}"),
        save_path,
    )


# --------------------------------------------------------------------------- #
# Unified Config-Driven Finetuning Engine (mirrors `train_gasnet(cfg)`)
# --------------------------------------------------------------------------- #
def finetune_gasnet_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Finetune GASNet using a unified YAML configuration dictionary (`paths`, `model`, `train`, `eval`)."""
    pc = cfg.get("paths", {})
    mc = cfg.get("model", {})
    tc = cfg.get("train", {})
    ec = cfg.get("eval", {})

    dataset_name = str(tc.get("dataset", "veri")).strip().lower()
    backbone_cfg = str(mc.get("backbone", "dinov3_convnext"))
    gate_cfg = str(mc.get("gate", "bcgg"))
    use_gem = bool(mc.get("use_gem", True))

    default_save = f"weights/{dataset_name}/gasnet_{backbone_cfg}_{gate_cfg}{'' if use_gem else '_avgpool'}.best.pth"
    save_path = Path(pc.get("save_path", default_save))
    best_save_path, last_save_path = _resolve_best_and_last_paths(save_path)
    best_save_path.parent.mkdir(parents=True, exist_ok=True)

    log_path_raw = pc.get("log_path")
    if log_path_raw:
        log_path = Path(log_path_raw)
        try:
            sys.stdout = Logger(log_path, sys.stdout)
            sys.stderr = Logger(log_path, sys.stderr)
            print(f"[INFO] Log: {log_path}")
        except OSError as exc:
            print(f"[WARNING] Cannot open log {log_path}: {exc}")

    seed = int(tc.get("seed", 42))
    set_seed(seed)
    configure_cuda_for_speed()

    data_root = Path(pc.get("data_root", "data"))
    if not data_root.exists() and Path("data").exists():
        data_root = Path("data")

    legacy_resize = bool(ec.get("legacy_resize", False))
    split_seed = int(ec.get("split_seed", 42))
    vrai_val_ids = int(ec.get("vrai_val_ids", 1000))

    # 1. Load dataset according to `train.dataset`
    if dataset_name in ("veri", "veri776", "veri-776"):
        train_samples = load_veri_train_samples(root_dir=data_root)
        val_q_dataset, val_g_dataset = load_veri776(root_dir=data_root, legacy_resize=legacy_resize)
        dataset_label = "VeRi-776"
    elif dataset_name == "vrai":
        train_samples, val_q_dataset, val_g_dataset = prepare_vrai_train_and_val(
            root_dir=data_root,
            num_val_ids=vrai_val_ids,
            split_seed=split_seed,
            legacy_resize=legacy_resize,
        )
        dataset_label = f"VRAI (Val {vrai_val_ids} IDs)"
    elif dataset_name == "vru":
        train_samples, val_q_dataset, val_g_dataset = prepare_vru_train_and_val(
            root_dir=data_root,
            split_seed=split_seed,
            legacy_resize=legacy_resize,
        )
        dataset_label = "VRU (Small)"
    else:
        raise ValueError(
            f"[FAIL] Unsupported train.dataset={dataset_name!r}. Choose from 'veri', 'vrai', or 'vru'."
        )

    train_ids = sorted({s.vehicle_id for s in train_samples})
    label_map = {vid: idx for idx, vid in enumerate(train_ids)}
    num_classes = len(label_map)

    epochs = int(tc.get("epochs", 15))
    freeze_epochs = int(tc.get("freeze_epochs", 2))
    batch_size = int(tc.get("batch_size", 64))
    pk_k = int(tc.get("pk_k", 4))
    grad_accum = max(1, int(tc.get("grad_accum", 1)))
    num_workers = int(tc.get("num_workers", 4))
    prefetch_factor = int(tc.get("prefetch_factor", 2))
    strong_aug = bool(tc.get("strong_aug", False))
    triplet_weight = float(tc.get("triplet_weight", 0.5))
    triplet_margin = float(tc.get("triplet_margin", 0.3))
    label_smoothing = float(tc.get("label_smoothing", 0.1))
    attention_reg_weight = float(tc.get("attention_reg_weight", 0.0))
    attn_diversity_weight = float(tc.get("attn_diversity_weight", 0.0))
    log_every = int(tc.get("log_every", 20))

    if batch_size % pk_k != 0:
        raise ValueError(f"[FAIL] train.batch_size ({batch_size}) must be divisible by train.pk_k ({pk_k})")
    pk_p = batch_size // pk_k

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_channels_last = (device.type == "cuda") and bool(tc.get("channels_last", True))
    use_amp = (device.type == "cuda") and bool(tc.get("use_amp", True))
    amp_dtype = choose_amp_dtype(str(tc.get("amp_dtype", "auto"))) if use_amp else torch.float32
    use_scaler = use_amp and (amp_dtype == torch.float16)
    pin_memory = (device.type == "cuda") and bool(tc.get("pin_memory", True))
    persistent_workers = bool(tc.get("persistent_workers", True)) and (num_workers > 0)

    train_tfm = build_finetune_train_transform(strong_aug=strong_aug)
    train_cameras = [s.camera for s in train_samples]
    train_colors = [s.color for s in train_samples]
    train_types = [s.vehicle_type for s in train_samples]
    has_valid_cams = any(bool(c) for c in train_cameras)
    has_valid_attrs = any(c >= 0 for c in train_colors) and any(t >= 0 for t in train_types)

    train_loader = make_pk_loader(
        samples=train_samples,
        transform=train_tfm,
        label_map=label_map,
        p=pk_p,
        k=pk_k,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        cameras=train_cameras if has_valid_cams else None,
        colors=train_colors if has_valid_attrs else None,
        vehicle_types=train_types if has_valid_attrs else None,
        camera_balanced=has_valid_cams and bool(tc.get("camera_balanced_sampler", True)),
        attribute_hard_negative=has_valid_attrs and bool(tc.get("attribute_hard_negative_sampler", True)),
        dataset_cls=VRAITrainDataset,
    )

    # 2. Resolve weight_path & Construct Model
    weight_path = _resolve_weight_path_from_model_cfg(mc, pc)
    model = build_and_load_gasnet_from_config(
        mc=mc,
        weight_path=weight_path,
        num_classes=num_classes,
        device=device,
    )
    if getattr(model, "_classifiers_loaded", False) and freeze_epochs > 0:
        print(
            f"[INFO] Classifier heads matched num_classes={num_classes} (continuing finetune) -> "
            f"auto-skipping HEAD-WARMUP (freeze_epochs: {freeze_epochs} -> 0)"
        )
        freeze_epochs = 0

    if attention_reg_weight > 0:
        model.set_attention_capture(True)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if (
        device.type == "cuda"
        and hasattr(torch, "compile")
        and bool(tc.get("use_compile", False))
    ):
        print("[INFO] Compiling GASNet with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    # 3. Learning rates & Optimizer
    default_bb_lr = 1.5e-5 if "convnext" in model.backbone_type else float(tc.get("base_lr", 3.5e-5))
    backbone_lr = float(tc.get("backbone_lr", default_bb_lr))
    gate_lr = float(tc.get("gate_lr", 1.0e-4))
    head_lr = float(tc.get("head_lr", 3.5e-4))
    weight_decay = float(tc.get("convnext_weight_decay", tc.get("weight_decay", 0.02)))
    warmup_epochs = int(tc.get("convnext_warmup_epochs", max(1, freeze_epochs)))

    pool_label = f"GeM (p={mc.get('gem_p', 3.0)})" if use_gem else "AvgPool"
    print("\n" + "=" * 90)
    print(
        f"[FINETUNE CONFIG] Dataset: {dataset_label} | Backbone: {model.backbone_type} | "
        f"Gate: {gate_cfg} | Pooling: {pool_label}"
    )
    print(
        f"[PATHS] Weight In: {weight_path} -> Best Out: {best_save_path}"
    )
    print(
        f"[TRAIN] Train IDs: {num_classes} ({len(train_samples)} imgs) | "
        f"Epochs: {epochs} (Freeze Warmup: {freeze_epochs}) | Batch: {batch_size} (P={pk_p}, K={pk_k})"
    )
    print(
        f"[LR] Backbone: {backbone_lr:.1e} | Gate+Neck: {gate_lr:.1e} | Classifier Head: {head_lr:.1e}"
    )
    print("=" * 90)

    run_eval_flag = bool(ec.get("run_eval", True))
    eval_epoch_0 = bool(ec.get("eval_epoch_0", True)) and run_eval_flag
    eval_every = int(ec.get("eval_every", 3))
    eval_batch_size = int(ec.get("eval_batch_size", batch_size))
    save_best = bool(ec.get("save_best", True))

    # 4. Evaluate Epoch 0 (Zero-Shot Baseline)
    base_model_ref = getattr(model, "_orig_mod", model)
    init_metrics = {"mAP": 0.0, "Rank-1": 0.0, "Rank-5": 0.0, "Rank-10": 0.0, "mINP": 0.0}
    if eval_epoch_0:
        print(f"[EVAL] Evaluating Epoch 0 (Pretrained Zero-Shot Baseline) on {dataset_label}...")
        model.eval()
        init_metrics = evaluate_dataset_on_model(
            base_model_ref, val_q_dataset, val_g_dataset, batch_size=eval_batch_size, device=device
        )
        print(
            f" -> [Epoch 0 Baseline] mAP: {init_metrics['mAP']:.2f}% | "
            f"Rank-1: {init_metrics['Rank-1']:.2f}% | Rank-5: {init_metrics['Rank-5']:.2f}% | "
            f"Rank-10: {init_metrics['Rank-10']:.2f}% | mINP: {init_metrics['mINP']:.2f}%"
        )

    best_map = float(init_metrics["mAP"])
    best_metrics = dict(init_metrics)
    best_epoch = 0
    if save_best:
        torch.save(base_model_ref.state_dict(), best_save_path)

    optimizer, scheduler = build_finetune_optimizer_and_scheduler(
        model=model,
        backbone_lr=backbone_lr,
        gate_lr=gate_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_attention_local = bool(mc.get("use_attention_local", False))

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        is_frozen_phase = epoch <= freeze_epochs
        set_warmup_freeze_state(model, freeze_feature_extractor=is_frozen_phase)

        running_loss = 0.0
        num_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            imgs, labels, _, _, cameras, colors, vehicle_types = batch
            if use_channels_last:
                imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            cams_arg = cameras if has_valid_cams else None
            colors_t = colors.to(device, non_blocking=True) if has_valid_attrs else None
            types_t = vehicle_types.to(device, non_blocking=True) if has_valid_attrs else None

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                feats, logits = model(imgs)
                loss_id = sum(ce_loss(logit, labels) for logit in logits)
                if is_frozen_phase:
                    loss = loss_id
                else:
                    loss_tri = sum(
                        batch_hard_triplet_loss(
                            feat,
                            labels,
                            margin=triplet_margin,
                            cameras=cams_arg,
                            colors=colors_t,
                            vehicle_types=types_t,
                        )
                        for feat in feats
                    )
                    loss = loss_id + triplet_weight * loss_tri
                    if attention_reg_weight > 0:
                        loss = loss + attention_reg_weight * base_model_ref.attention_regularization_loss()
                    if use_attention_local and attn_diversity_weight > 0:
                        loss = loss + attn_diversity_weight * base_model_ref.attn_local.diversity_loss()

            loss_step = loss / grad_accum
            if use_amp and use_scaler:
                scaler.scale(loss_step).backward()
            else:
                loss_step.backward()

            if step % grad_accum == 0 or step == len(train_loader):
                if use_amp and use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            num_samples += bs

            if log_every > 0 and (step == 1 or step % log_every == 0 or step == len(train_loader)):
                print(f"  epoch {epoch:02d} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / max(1, num_samples)
        phase_tag = "HEAD-WARMUP" if is_frozen_phase else "FULL-FINETUNE"
        elapsed = time.time() - t0
        print(
            f"[INFO] Epoch {epoch:02d}/{epochs:02d} ({phase_tag}): "
            f"loss={avg_loss:.4f} | time={elapsed:.1f}s"
        )

        if run_eval_flag and ((eval_every > 0 and epoch % eval_every == 0) or epoch == epochs):
            model.eval()
            cur_metrics = evaluate_dataset_on_model(
                base_model_ref, val_q_dataset, val_g_dataset, batch_size=eval_batch_size, device=device
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
                if save_best:
                    torch.save(base_model_ref.state_dict(), best_save_path)
                    print(f"    [OK] Updated best checkpoint @ epoch {epoch} -> {best_save_path}")

    last_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_model_ref.state_dict(), last_save_path)
    print("-" * 90)
    print(
        f"[SUMMARY] {model.backbone_type} ({gate_cfg}, {pool_label}) on {dataset_label}: "
        f"mAP {init_metrics['mAP']:.2f}% (Epoch 0) -> {best_map:.2f}% (Best @ Epoch {best_epoch}) | "
        f"Rank-1: {init_metrics['Rank-1']:.2f}% -> {best_metrics['Rank-1']:.2f}%"
    )
    print(f"[OK] Saved Best Checkpoint: {best_save_path}")
    print(f"[OK] Saved Last Checkpoint: {last_save_path}")
    print("-" * 90)

    return {
        "backbone": model.backbone_type,
        "gate": gate_cfg,
        "use_gem": use_gem,
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
        "last_checkpoint": str(last_save_path),
    }


# --------------------------------------------------------------------------- #
# Convenience Wrapper for Variant-Based Finetuning
# --------------------------------------------------------------------------- #
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
    eval_every: int = 3,
    num_workers: int = 4,
    strong_aug: bool = False,
    use_gem: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run finetuning for a named variant (`resnet_rga`, `convnext_rga`, `convnext_bcgg`)."""
    if variant == "resnet_rga":
        backbone, gate, ckpt_key = "resnet50_ibn", "rga", "resnet_rga"
    elif variant == "convnext_rga":
        backbone, gate, ckpt_key = "dinov3_convnext", "rga", "convnext_rga"
    elif variant == "convnext_bcgg":
        backbone, gate, ckpt_key = "dinov3_convnext", "bcgg", "convnext_bcgg"
    else:
        raise ValueError(f"Unsupported variant {variant!r}. Choose from {FINETUNE_VARIANTS}")

    ckpt_file = (
        Path(pretrained_ckpt).resolve()
        if pretrained_ckpt is not None
        else find_checkpoint(ckpt_key, custom_weights_dir=pretrained_dir)
    )
    pool_suffix = "" if use_gem else "_avgpool"
    out_dir = Path(output_dir)
    save_path = out_dir / f"gasnet_{variant}{pool_suffix}.best.pth"

    ds_key = "veri" if "veri" in dataset_label.lower() else ("vrai" if "vrai" in dataset_label.lower() else "vru")
    cfg = {
        "paths": {
            "weight_path": str(ckpt_file),
            "save_path": str(save_path),
        },
        "model": {
            "backbone": backbone,
            "gate": gate,
            "use_gem": use_gem,
        },
        "train": {
            "dataset": ds_key,
            "seed": seed,
            "epochs": epochs,
            "freeze_epochs": freeze_epochs,
            "batch_size": batch_size,
            "pk_k": pk_k,
            "backbone_lr": backbone_lr if backbone_lr is not None else (1.5e-5 if "convnext" in variant else 3.5e-5),
            "gate_lr": gate_lr,
            "head_lr": head_lr,
            "convnext_weight_decay": weight_decay,
            "triplet_weight": triplet_weight,
            "triplet_margin": triplet_margin,
            "label_smoothing": label_smoothing,
            "num_workers": num_workers,
            "strong_aug": strong_aug,
        },
        "eval": {
            "run_eval": True,
            "eval_epoch_0": True,
            "eval_every": eval_every,
            "eval_batch_size": eval_batch_size,
            "save_best": True,
        },
    }
    res = finetune_gasnet_from_config(cfg)
    res["variant"] = variant
    return res
