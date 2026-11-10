"""Training and evaluation engine for Stage 1: GASNet on VRU / VRAI."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from uav_reid.datasets import (
    Sample,
    VRAITrainDataset,
    VRUDataset,
    build_query_gallery,
    make_pk_loader,
    read_split_file,
    read_vrai_train_samples,
)
from uav_reid.gasnet import GASNet, infer_backbone_name, infer_econvnext, infer_gate_config
from uav_reid.reidnet.losses import batch_hard_triplet_loss
from uav_reid.utils import (
    Logger,
    choose_amp_dtype,
    configure_cuda_for_speed,
    evaluate_map_cmc,
    normalize_state_dict,
    set_seed,
)

SPLIT_FILE_MAP: Dict[str, str] = {
    "Small": "test_list_1200.txt",
    "Medium": "test_list_2400.txt",
    "Big": "test_list_8000.txt",
}

BEST_METRIC_SPLIT_PRIORITY = {
    "big_map": ("Big", "Medium", "Small"),
    "medium_map": ("Medium", "Big", "Small"),
    "small_map": ("Small", "Medium", "Big"),
}


def build_gasnet_transforms(strong_aug: bool = False, legacy_resize: bool = False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if strong_aug:
        train_tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (224, 224), scale=(0.75, 1.0), ratio=(0.85, 1.15)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.35),
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.18, p=1.0)], p=0.25
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04
                        )
                    ],
                    p=0.75,
                ),
                transforms.RandomGrayscale(p=0.08),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"
                ),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if legacy_resize:
        test_tfms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        test_tfms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return train_tfms, test_tfms


def _make_eval_loader(
    samples: List[Sample],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    ds = VRUDataset(samples=samples, transform=transform, relabel=False, max_retries=3)
    loader_kwargs: Dict[str, Any] = dict(pin_memory=pin_memory, drop_last=False)
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **loader_kwargs,
    )


@torch.no_grad()
def _query_expansion_rerank_features(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    k1: int,
    alpha: float,
    q_chunk_size: int,
    use_fp16_sim: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k1 < 1:
        return q_feat, g_feat

    device = q_feat.device if q_feat.is_cuda else g_feat.device
    q = F.normalize(q_feat.to(device=device, dtype=torch.float32), dim=1)
    g = F.normalize(g_feat.to(device=device, dtype=torch.float32), dim=1)
    k = min(k1, g.size(0))

    sim_dtype = None
    if device.type == "cuda" and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    q_new_chunks = []
    for start in range(0, q.size(0), q_chunk_size):
        end = min(start + q_chunk_size, q.size(0))
        sim = q_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False).indices
        q_new_chunks.append(F.normalize(q[start:end] + alpha * g[idx].mean(dim=1), dim=1))

    g_new_chunks = []
    for start in range(0, g.size(0), q_chunk_size):
        end = min(start + q_chunk_size, g.size(0))
        sim = g_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False).indices
        g_new_chunks.append(F.normalize(g[start:end] + alpha * g[idx].mean(dim=1), dim=1))

    return torch.cat(q_new_chunks, dim=0), torch.cat(g_new_chunks, dim=0)


@torch.no_grad()
def _extract_vru_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
    output_device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    ids = []
    out_device = torch.device(output_device)
    if out_device.type == "cuda" and not torch.cuda.is_available():
        out_device = torch.device("cpu")

    for imgs, _, vehicle_ids, _ in loader:
        if use_channels_last:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)

        autocast_kwargs = dict(
            device_type=device.type, enabled=(use_amp and device.type == "cuda")
        )
        if device.type == "cuda":
            autocast_kwargs["dtype"] = amp_dtype

        with torch.autocast(**autocast_kwargs):
            outputs = model(imgs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            emb = torch.cat(list(outputs), dim=1)

        feats.append(emb.float().to(out_device, non_blocking=(out_device.type == "cuda")))
        ids.append(vehicle_ids.to(out_device, non_blocking=(out_device.type == "cuda")))

    return torch.cat(feats, dim=0), torch.cat(ids, dim=0)


@torch.no_grad()
def run_vru_eval(
    model: torch.nn.Module,
    data_root: Path,
    test_transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
    q_chunk_size: int = 2048,
    use_fp16_sim: bool = True,
    verbose_eval: bool = False,
    rerank: bool = False,
    rerank_k1: int = 20,
    rerank_alpha: float = 0.3,
    tta_flip: bool = False,
    split_seed: int | None = 42,
    gallery_mode: str = "random",
    splits_to_eval: Tuple[str, ...] = ("Small", "Medium", "Big"),
) -> Dict[str, Tuple[float, float, float]]:
    vru_dir = data_root / "VRU"
    pic_dir = vru_dir / "Pic"
    split_dir = vru_dir / "train_test_split"

    metrics_by_split: Dict[str, Tuple[float, float, float]] = {}

    was_training = model.training
    model.eval()

    for split_name in splits_to_eval:
        split_file = SPLIT_FILE_MAP[split_name]
        test_samples = read_split_file(pic_dir, split_dir / split_file)
        query_samples, gallery_samples = build_query_gallery(
            test_samples, seed=split_seed, gallery_mode=gallery_mode
        )

        q_loader = _make_eval_loader(
            query_samples,
            test_transform,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )
        g_loader = _make_eval_loader(
            gallery_samples,
            test_transform,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )

        feat_out_device = device
        q_feat, q_ids = _extract_vru_features(
            model,
            q_loader,
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            output_device=feat_out_device,
        )
        g_feat, g_ids = _extract_vru_features(
            model,
            g_loader,
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            output_device=feat_out_device,
        )
        if tta_flip:
            flip_transform = transforms.Compose(
                [test_transform, transforms.RandomHorizontalFlip(p=1.0)]
            )
            q_flip_loader = _make_eval_loader(
                query_samples,
                flip_transform,
                batch_size,
                num_workers,
                pin_memory,
                persistent_workers,
                prefetch_factor,
            )
            g_flip_loader = _make_eval_loader(
                gallery_samples,
                flip_transform,
                batch_size,
                num_workers,
                pin_memory,
                persistent_workers,
                prefetch_factor,
            )
            q_flip_feat, _ = _extract_vru_features(
                model,
                q_flip_loader,
                device,
                use_amp,
                amp_dtype,
                use_channels_last,
                output_device=feat_out_device,
            )
            g_flip_feat, _ = _extract_vru_features(
                model,
                g_flip_loader,
                device,
                use_amp,
                amp_dtype,
                use_channels_last,
                output_device=feat_out_device,
            )
            q_feat = F.normalize(
                F.normalize(q_feat, dim=1) + F.normalize(q_flip_feat, dim=1), dim=1
            )
            g_feat = F.normalize(
                F.normalize(g_feat, dim=1) + F.normalize(g_flip_feat, dim=1), dim=1
            )
        if rerank:
            q_feat, g_feat = _query_expansion_rerank_features(
                q_feat=q_feat,
                g_feat=g_feat,
                k1=rerank_k1,
                alpha=rerank_alpha,
                q_chunk_size=q_chunk_size,
                use_fp16_sim=use_fp16_sim,
            )

        m_ap, rank1, rank5 = evaluate_map_cmc(
            q_feat,
            q_ids,
            g_feat,
            g_ids,
            topk=(1, 5),
            q_chunk_size=q_chunk_size,
            use_fp16_sim=use_fp16_sim,
            verbose=(verbose_eval and split_name == "Big"),
        )
        metrics_by_split[split_name] = (m_ap, rank1, rank5)

    if was_training:
        model.train()

    return metrics_by_split


def print_vru_eval_report(
    metrics_by_split: Dict[str, Tuple[float, float, float]], title: str = "Evaluation"
) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Split':<10} {'mAP':>10} {'Rank-1':>10} {'Rank-5':>10}")
    for split_name in ("Small", "Medium", "Big"):
        if split_name in metrics_by_split:
            m_ap, rank1, rank5 = metrics_by_split[split_name]
            print(f"{split_name:<10} {m_ap:>10.4f} {rank1:>10.4f} {rank5:>10.4f}")


def evaluate_gasnet_checkpoint(cfg: Dict[str, Any], checkpoint_path: str | Path | None = None) -> Dict[str, Tuple[float, float, float]]:
    """Standalone evaluation of a saved GASNet checkpoint on VRU."""
    ec = cfg.get("eval", {})
    tc = cfg.get("train", {})
    mc = cfg.get("model", {})
    pc = cfg.get("paths", {})

    ckpt_path = Path(checkpoint_path or ec.get("checkpoint") or pc.get("save_path", "gasnet_vru.pth"))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"[FAIL] Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_for_speed()

    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    state = normalize_state_dict(raw)

    gate_config = infer_gate_config(state)
    backbone = infer_backbone_name(state, default=mc.get("backbone", "resnet50"))
    econvnext = infer_econvnext(state) or bool(mc.get("econvnext", False))
    classifier = state.get("classifier_global.weight")
    num_classes = int(classifier.shape[0]) if classifier is not None else 6302
    use_gem = bool(mc.get("use_gem", True))
    gem_p = float(mc.get("gem_p", 3.0))

    print(
        f"[INFO] eval-only | checkpoint={ckpt_path} | backbone={backbone} (econvnext={econvnext}) | "
        f"gate={gate_config['gate']} | num_classes={num_classes} | use_gem={use_gem}"
    )

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        backbone=backbone,
        econvnext=econvnext,
        use_gem=use_gem,
        gem_p=gem_p,
        **gate_config,
    )
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    print(f"[OK] Loaded {len(filtered)}/{len(model_state)} tensors")
    model.eval().to(device)

    _, test_tfms = build_gasnet_transforms(
        strong_aug=False, legacy_resize=bool(ec.get("legacy_resize", False))
    )
    num_workers = int(tc.get("num_workers", 8))
    metrics = run_vru_eval(
        model=model,
        data_root=Path(pc.get("data_root", "/data")),
        test_transform=test_tfms,
        batch_size=int(ec.get("batch_size", tc.get("batch_size", 512))),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=int(tc.get("prefetch_factor", 4)) if num_workers > 0 else None,
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        use_channels_last=False,
        q_chunk_size=int(ec.get("q_chunk_size", 2048)),
        use_fp16_sim=bool(ec.get("use_fp16_sim", True)),
        verbose_eval=bool(ec.get("verbose", False)),
        rerank=bool(ec.get("rerank", False)),
        rerank_k1=int(ec.get("rerank_k1", 20)),
        rerank_alpha=float(ec.get("rerank_alpha", 0.3)),
        tta_flip=bool(ec.get("tta_flip", False)),
        split_seed=int(ec.get("split_seed", 42)),
        gallery_mode=str(ec.get("gallery_mode", "random")),
    )
    print_vru_eval_report(metrics, title="GASNet Evaluation")
    return metrics


def _select_best_metric(
    metrics_by_split: dict, best_metric: str
) -> tuple[float, str] | tuple[None, None]:
    for split_name in BEST_METRIC_SPLIT_PRIORITY.get(best_metric, ("Big", "Medium", "Small")):
        split_metrics = metrics_by_split.get(split_name)
        if split_metrics is not None:
            return float(split_metrics[0]), split_name
    return None, None


def train_gasnet(
    cfg: Dict[str, Any],
    trial_callback: Callable[[int, float], bool] | None = None,
) -> Dict[str, Any]:
    """Train GASNet from a unified config dictionary.

    Args:
        cfg: Unified configuration dictionary (`paths`, `model`, `train`, `eval`).
        trial_callback: Optional callback `(epoch, metric_score) -> should_prune` for HPO/Optuna.

    Returns:
        Dictionary with `best_score`, `best_epoch`, `best_split`, `save_path`, `best_path`.
    """
    pc = cfg.get("paths", {})
    mc = cfg.get("model", {})
    tc = cfg.get("train", {})
    ec = cfg.get("eval", {})

    save_path = Path(pc.get("save_path", "outputs/gasnet/gasnet_vru.pth"))
    log_path = Path(pc.get("log_path") or (save_path.parent / "log.txt"))

    if not cfg.get("_OptunaQuietLog", False):
        try:
            sys.stdout = Logger(log_path, sys.stdout)
            sys.stderr = Logger(log_path, sys.stderr)
            print(f"[INFO] Log: {log_path}")
        except OSError as exc:
            print(f"[WARNING] Cannot open log {log_path}: {exc}")

    seed = int(tc.get("seed", 42))
    set_seed(seed)
    configure_cuda_for_speed()

    data_root = Path(pc.get("data_root", "/data"))
    dataset_name = str(tc.get("dataset", "vru")).lower()
    epochs = int(tc.get("epochs", 60))
    batch_size = int(tc.get("batch_size", 512))
    base_batch_size = int(tc.get("base_batch_size", 128))
    base_lr = float(tc.get("base_lr", 3.5e-4))
    grad_accum = int(tc.get("grad_accum", 1))
    num_workers = int(tc.get("num_workers", 8))
    prefetch_factor = int(tc.get("prefetch_factor", 4))
    log_every = int(tc.get("log_every", 20))
    pk_k = int(tc.get("pk_k", 4))
    strong_aug = bool(tc.get("strong_aug", False))
    triplet_weight = float(tc.get("triplet_weight", 0.5))
    triplet_margin = float(tc.get("triplet_margin", 0.3))
    label_smoothing = float(tc.get("label_smoothing", 0.1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_channels_last = torch.cuda.is_available() and bool(tc.get("channels_last", True))
    use_amp = torch.cuda.is_available() and bool(tc.get("use_amp", True))
    amp_dtype = choose_amp_dtype(str(tc.get("amp_dtype", "auto")))
    use_scaler = use_amp and (amp_dtype == torch.float16)
    pin_memory = torch.cuda.is_available() and bool(tc.get("pin_memory", True))
    persistent_workers = bool(tc.get("persistent_workers", True)) and num_workers > 0

    run_eval_flag = bool(ec.get("run_eval", True))
    eval_every = int(ec.get("eval_every", 10))
    save_best = bool(ec.get("save_best", True)) and run_eval_flag
    best_metric = str(ec.get("best_metric", "big_map"))
    eval_splits = tuple(ec.get("splits", ("Small", "Medium", "Big")))

    if dataset_name == "vrai":
        train_samples = read_vrai_train_samples(data_root / "VRAI")
        dataset_cls = VRAITrainDataset
        train_cameras = [s.camera for s in train_samples]
        train_colors = [s.color for s in train_samples]
        train_types = [s.vehicle_type for s in train_samples]
    else:
        vru_dir = data_root / "VRU"
        pic_dir = vru_dir / "Pic"
        split_dir = vru_dir / "train_test_split"
        train_samples = read_split_file(pic_dir, split_dir / "train_list.txt")
        dataset_cls = VRUDataset
        train_cameras = None
        train_colors = None
        train_types = None

    train_ids = sorted({s.vehicle_id for s in train_samples})
    train_label_map = {vid: i for i, vid in enumerate(train_ids)}
    num_classes = len(train_label_map)

    train_tfms, test_tfms = build_gasnet_transforms(
        strong_aug=strong_aug, legacy_resize=bool(ec.get("legacy_resize", False))
    )

    if batch_size % pk_k != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by pk_k ({pk_k})")
    pk_p = batch_size // pk_k

    train_loader = make_pk_loader(
        train_samples,
        train_tfms,
        label_map=train_label_map,
        p=pk_p,
        k=pk_k,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        cameras=train_cameras,
        colors=train_colors,
        vehicle_types=train_types,
        camera_balanced=(
            dataset_name == "vrai" and bool(tc.get("camera_balanced_sampler", True))
        ),
        attribute_hard_negative=(
            dataset_name == "vrai" and bool(tc.get("attribute_hard_negative_sampler", True))
        ),
        dataset_cls=dataset_cls,
    )

    backbone = str(mc.get("backbone", "dinov3_convnext"))
    use_convnext = backbone.startswith("dinov3_convnext")
    gate = str(mc.get("gate", "bcgg"))
    gate_hidden = int(mc.get("gate_hidden", 16))
    gate_groups = int(mc.get("gate_groups", 4))
    gate_kernel = int(mc.get("gate_kernel", 7))
    gate_channel_kernel = int(mc.get("gate_channel_kernel", 5))
    gate_inject_value = bool(mc.get("gate_inject_value", False))
    use_gem = bool(mc.get("use_gem", True))
    gem_p = float(mc.get("gem_p", 3.0))
    econvnext = bool(mc.get("econvnext", False))
    use_part_branch = bool(mc.get("use_part_branch", False))
    num_parts = int(mc.get("num_parts", 4))
    use_attention_local = bool(mc.get("use_attention_local", False))
    num_attention_heads = int(mc.get("num_attention_heads", 4))
    attention_reg_weight = float(tc.get("attention_reg_weight", 0.0))
    attn_diversity_weight = float(tc.get("attn_diversity_weight", 0.0))

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=bool(mc.get("pretrained", True)),
        use_part_branch=use_part_branch,
        num_parts=num_parts,
        use_gem=use_gem,
        gem_p=gem_p,
        backbone=backbone,
        use_attention_local=use_attention_local,
        num_attention_heads=num_attention_heads,
        econvnext=econvnext,
        gate=gate,
        gate_hidden=gate_hidden,
        gate_groups=gate_groups,
        gate_kernel=gate_kernel,
        gate_channel_kernel=gate_channel_kernel,
        gate_inject_value=gate_inject_value,
    ).to(device)

    if attention_reg_weight > 0:
        model.set_attention_capture(True)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if (
        torch.cuda.is_available()
        and hasattr(torch, "compile")
        and bool(tc.get("use_compile", True))
    ):
        print("[INFO] Compiling GASNet with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    if use_convnext:
        backbone_lr = float(tc.get("backbone_lr", 3e-5))
        head_lr = float(tc.get("head_lr", 3e-4))
        weight_decay = float(tc.get("convnext_weight_decay", 0.05))
        warmup_epochs = int(tc.get("convnext_warmup_epochs", 10))

        backbone_params = []
        head_params = []
        base_model = getattr(model, "_orig_mod", model)
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if "convnext_backbone." in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "convnext_backbone"},
            {"params": head_params, "lr": head_lr, "name": "head_classifier"},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        learning_rate = base_lr * (batch_size / base_batch_size)
        weight_decay = float(tc.get("weight_decay", 5e-4))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        warmup_epochs = int(tc.get("warmup_epochs", 5))
        milestones = list(tc.get("milestones", [40, 50]))

        def lr_lambda(epoch_idx: int) -> float:
            if epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / max(1, warmup_epochs)
            drops = sum(epoch_idx + 1 >= m for m in milestones)
            return 0.1**drops

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f"[INFO] GASNet Training | device={device} | amp={use_amp} ({amp_dtype}) | "
        f"backbone={backbone} (econvnext={econvnext}) | gate={gate} | batch={batch_size} (pk_k={pk_k})"
    )

    model.train()
    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_score: float | None = None
    best_epoch: int | None = None
    best_split: str | None = None
    best_path = save_path.with_name(f"{save_path.stem}.best{save_path.suffix}")
    if save_best:
        best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        running = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            if dataset_name == "vrai":
                imgs, labels, _, _, cameras, colors, vehicle_types = batch
                colors = colors.to(device, non_blocking=True)
                vehicle_types = vehicle_types.to(device, non_blocking=True)
            else:
                imgs, labels, _, _ = batch
                cameras = None
                colors = None
                vehicle_types = None

            if use_channels_last:
                imgs = imgs.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
            else:
                imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                feats, logits = model(imgs)
                loss_id = sum(ce_loss(logit, labels) for logit in logits)
                loss_tri = sum(
                    batch_hard_triplet_loss(
                        feat,
                        labels,
                        margin=triplet_margin,
                        cameras=cameras,
                        colors=colors,
                        vehicle_types=vehicle_types,
                    )
                    for feat in feats
                )
                loss = loss_id + triplet_weight * loss_tri
                if attention_reg_weight > 0:
                    base_model = getattr(model, "_orig_mod", model)
                    loss_attn = base_model.attention_regularization_loss()
                    loss = loss + attention_reg_weight * loss_attn
                if use_attention_local and attn_diversity_weight > 0:
                    base_model = getattr(model, "_orig_mod", model)
                    div_loss = base_model.attn_local.diversity_loss()
                    loss = loss + attn_diversity_weight * div_loss

            loss_to_backprop = loss / grad_accum
            if use_amp and use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if step % grad_accum == 0:
                if use_amp and use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * imgs.size(0)

            if step == 1 or step % log_every == 0 or step == len(train_loader):
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}"
                )

        if step % grad_accum != 0:
            if use_amp and use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = running / max(1, len(train_loader.dataset))
        print(
            f"[INFO] Epoch {epoch}/{epochs}: loss={epoch_loss:.4f}, time={(time.time() - t0)/60:.1f} min"
        )
        scheduler.step()

        should_eval = (
            run_eval_flag
            and dataset_name == "vru"
            and ((eval_every > 0 and epoch % eval_every == 0) or epoch == epochs)
        )
        if should_eval:
            metrics = run_vru_eval(
                model=model,
                data_root=data_root,
                test_transform=test_tfms,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
                q_chunk_size=int(ec.get("q_chunk_size", 2048)),
                use_fp16_sim=bool(ec.get("use_fp16_sim", True)),
                verbose_eval=bool(ec.get("verbose", False)),
                rerank=bool(ec.get("rerank", False)),
                rerank_k1=int(ec.get("rerank_k1", 20)),
                rerank_alpha=float(ec.get("rerank_alpha", 0.3)),
                tta_flip=bool(ec.get("tta_flip", False)),
                split_seed=int(ec.get("split_seed", 42)),
                gallery_mode=str(ec.get("gallery_mode", "random")),
                splits_to_eval=eval_splits,
            )
            print_vru_eval_report(metrics, title=f"Evaluation @ Epoch {epoch}")
            metric_val, metric_split = _select_best_metric(metrics, best_metric)
            if metric_val is not None:
                if best_score is None or metric_val > best_score:
                    best_score = metric_val
                    best_epoch = epoch
                    best_split = metric_split
                    if save_best:
                        torch.save(model.state_dict(), best_path)
                        print(
                            f"[OK] Updated best checkpoint @ epoch {epoch}: {metric_split} mAP={metric_val:.4f} -> {best_path}"
                        )
                if trial_callback is not None:
                    should_prune = trial_callback(epoch, metric_val)
                    if should_prune:
                        print(f"[INFO] Trial pruned at epoch {epoch} (mAP={metric_val:.4f})")
                        break

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Saved final GASNet model to {save_path}")

    return {
        "best_score": best_score if best_score is not None else 0.0,
        "best_epoch": best_epoch,
        "best_split": best_split,
        "save_path": str(save_path),
        "best_path": str(best_path) if save_best else None,
    }
