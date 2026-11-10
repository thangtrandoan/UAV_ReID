"""Training, evaluation, and Fixed-FAR calibration engine for Stage 2: UAVReIDNet."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from uav_reid.datasets import EvalDataset, ReIDBatchSampler, UAVReIDDataset
from uav_reid.reidnet import (
    CenterLoss,
    HardTripletLoss,
    LabelSmoothCrossEntropy,
    TemporalConsistencyLoss,
    UAVReIDNet,
    load_checkpoint_verbose,
)
from uav_reid.utils import (
    Logger,
    assert_frozen,
    check_data_meta,
    compute_tar_at_far,
    configure_cuda_for_speed,
    eval_uav_map_cmc,
    provenance,
    resolve_pipeline,
    save_ckpt_atomic,
    set_seed,
)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_uav_transforms():
    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform_train, transform_test


@torch.no_grad()
def validate_single_n(model: nn.Module, loader: DataLoader, tag: str = "") -> tuple[float, float]:
    print(f" --- Validation trên tập Test {tag} ---")
    model.eval()
    qf, gf, q_pids, g_pids = [], [], [], []
    start_t = time.time()
    for batch in loader:
        before, after, pids = batch[0], batch[1], batch[2]
        before, after = before.cuda(), after.cuda()
        bn_feat_g = model(before)
        bn_feat_q = model(after)
        gf.append(bn_feat_g)
        qf.append(bn_feat_q)
        g_pids.extend(pids.numpy())
        q_pids.extend(pids.numpy())
    qf_t = torch.cat(qf, dim=0)
    gf_t = torch.cat(gf, dim=0)
    cmc, mAP, _, _, _ = eval_uav_map_cmc(qf_t, gf_t, q_pids, g_pids)
    rank1 = float(cmc[0])
    print(
        f" -> Val Time: {time.time() - start_t:.2f}s | Rank-1: {rank1*100:.2f}% | mAP: {mAP*100:.2f}%"
    )
    model.train()
    return rank1, float(mAP)


def validate_multi_n(
    model: nn.Module, loaders: Dict[int, DataLoader], n_list: list[int], stage_name: str = ""
):
    per_n = {}
    map_per_n = {}
    for _n in n_list:
        r1, m_ap = validate_single_n(model, loaders[_n], tag=f"(N={_n})")
        per_n[_n] = r1
        map_per_n[_n] = m_ap
    spread = max(per_n.values()) - min(per_n.values())
    worst = min(per_n.values())
    mean_map = float(np.mean(list(map_per_n.values())))
    line = " | ".join(f"N={_n}:{per_n[_n]*100:6.2f}%" for _n in n_list)
    print(
        f" ==> [{stage_name}] {line} | LỆCH={spread*100:5.2f}% | MIN={worst*100:6.2f}% | Mean mAP={mean_map*100:6.2f}%"
    )
    return worst, per_n, spread, mean_map


def train_uav_reid(
    cfg: Dict[str, Any],
    trial_callback: Callable[[int, float], bool] | None = None,
) -> Dict[str, Any]:
    """Two-stage training for UAVReIDNet on UAV-Anti-UAV."""
    _frozen_warn = assert_frozen(cfg)
    if _frozen_warn and not cfg.get("_OptunaQuietLog", False):
        print("\n" + "!" * 66)
        print("  [WARNING] PIPELINE ĐANG MỞ KHÓA Ở CÁC ĐIỂM SAU:")
        for _w in _frozen_warn:
            print(f"     - {_w}")
        print("!" * 66 + "\n")

    check_data_meta(cfg, verbose=not cfg.get("_OptunaQuietLog", False))

    pc = cfg.get("paths", {})
    tc = cfg.get("train", {})
    dc = cfg.get("device", {})
    lc = tc.get("loss", {})

    seed = int(tc.get("seed", 42))
    set_seed(seed)

    data_dir = pc.get("data_dir", "processed")
    query_json = os.path.join(data_dir, "query_train.json")
    gallery_json = os.path.join(data_dir, "gallery_train.json")
    gasnet_weights = pc.get("gasnet_weights", "")
    checkpoint_dir = pc.get("checkpoint_dir", "outputs/reidnet/checkpoints")
    log_dir = pc.get("log_dir", "outputs/reidnet/logs")

    gpu_jetson = bool(dc.get("gpu_jetson", False))
    configure_cuda_for_speed(gpu_jetson=gpu_jetson)

    resume = tc.get("resume", "")
    resume_best_rank1 = float(tc.get("resume_best_rank1", -1.0))
    batch_size = int(tc.get("batch_size", 12))
    num_instances = int(tc.get("num_instances", 3))
    num_frames = int(tc.get("num_frames", 12))
    num_workers = int(tc.get("num_workers", 8))
    use_amp = bool(tc.get("use_amp", True))
    pin_memory = bool(tc.get("pin_memory", True))
    use_compile = bool(tc.get("use_compile", True))
    compile_dynamic = bool(tc.get("compile_dynamic", False))
    val_freq = int(tc.get("val_freq", 5))
    val_n_list = list(tc.get("val_n_list") or [num_frames])

    early_stop_patience = int(tc.get("early_stop_patience", 0))
    early_stop_min_delta = float(tc.get("early_stop_min_delta", 1e-4))
    stop_on_target = bool(tc.get("stop_on_target", False))
    target_rank1 = float(tc.get("target_rank1", 0.85))
    target_spread = float(tc.get("target_spread", 0.05))

    backbone = str(tc.get("backbone", "dinov3_convnext"))
    econvnext = bool(tc.get("econvnext", True))
    temporal_type = str(tc.get("temporal_type", "attention"))
    temporal_pool = str(tc.get("temporal_pool", "attn"))
    temporal_pe = bool(tc.get("temporal_pe", True))
    temporal_d_model = int(tc.get("temporal_d_model", 512))
    temporal_d_out = int(tc.get("temporal_d_out", 512))
    temporal_layers = int(tc.get("temporal_layers", 2))
    temporal_heads = int(tc.get("temporal_heads", 8))
    temporal_dropout = float(tc.get("temporal_dropout", 0.1))

    epochs_stage1 = int(tc.get("stage1", {}).get("epochs", 30))
    lr_stage1 = float(tc.get("stage1", {}).get("lr", 3.5e-4))
    wd_stage1 = float(tc.get("stage1", {}).get("weight_decay", 5e-4))

    epochs_stage2 = int(tc.get("stage2", {}).get("epochs", 30))
    lr_stage2_backbone = float(tc.get("stage2", {}).get("lr_backbone", 1e-5))
    lr_stage2_temporal = float(tc.get("stage2", {}).get("lr_temporal", 1e-4))
    lr_stage2_head = float(tc.get("stage2", {}).get("lr_head", 1e-4))
    wd_stage2 = float(tc.get("stage2", {}).get("weight_decay", 5e-4))

    lam1 = float(lc.get("lam1", 1.0))
    lam2 = float(lc.get("lam2", 0.0))
    lam3 = float(lc.get("lam3", 0.001))
    lr_center = float(lc.get("lr_center", 0.5))
    label_smooth = float(lc.get("label_smooth", 0.1))
    triplet_margin = float(lc.get("triplet_margin", 0.3))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not cfg.get("_OptunaQuietLog", False):
        log_path = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
        sys.stdout = Logger(log_path, sys.stdout)

    print(f"=== Bắt đầu huấn luyện UAVReIDNet lúc {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    transform_train, transform_test = build_uav_transforms()
    train_dir = os.path.join(data_dir, "train")
    n_values = tc.get("n_frames_choices") or [num_frames]

    dataset = UAVReIDDataset(
        train_dir,
        query_json,
        gallery_json,
        transform=transform_train,
        num_frames=num_frames,
        n_values=n_values,
    )
    num_identities = dataset.num_identities
    sampler = ReIDBatchSampler(
        dataset, batch_size=batch_size, num_instances=num_instances, n_values=n_values
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dir = os.path.join(data_dir, "test")
    val_loaders = {}
    has_val = False
    if os.path.exists(test_dir):
        val_query_json = query_json.replace("query_train", "query_test")
        val_gallery_json = gallery_json.replace("gallery_train", "gallery_test")
        for _n in val_n_list:
            _ds = EvalDataset(
                test_dir,
                val_query_json,
                val_gallery_json,
                transform=transform_test,
                num_frames=_n,
            )
            val_loaders[_n] = DataLoader(
                _ds, batch_size=batch_size, shuffle=False, num_workers=2
            )
        has_val = True
        print(f" [OK] Validation đo {len(val_n_list)} N: {val_n_list}")
    else:
        print(f" [WARNING] Không tìm thấy test_dir tại {test_dir}; bỏ qua validation.")

    model = UAVReIDNet(
        gasnet_weights_path=gasnet_weights or None,
        num_identities=num_identities,
        freeze_backbone=True,
        backbone=backbone,
        econvnext=econvnext,
        temporal_pool=temporal_pool,
        temporal_pe=temporal_pe,
        temporal_type=temporal_type,
        temporal_d_model=temporal_d_model,
        temporal_d_out=temporal_d_out,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        temporal_dropout=temporal_dropout,
    ).cuda()

    if not gpu_jetson and use_compile and hasattr(torch, "compile"):
        print(f" [INFO] Bật torch.compile(dynamic={compile_dynamic})...")
        for _attr in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, _attr):
                setattr(torch._dynamo.config, _attr, 64)
        for _attr in ("accumulated_cache_size_limit", "accumulated_recompile_limit"):
            if hasattr(torch._dynamo.config, _attr):
                setattr(torch._dynamo.config, _attr, 256)
        try:
            model = torch.compile(model, dynamic=compile_dynamic)
        except Exception as e:
            print(f" [WARNING] torch.compile() thất bại: {e}. Chạy eager mode.")

    criterion_id = LabelSmoothCrossEntropy(epsilon=label_smooth)
    criterion_triplet = HardTripletLoss(margin=triplet_margin)
    base_model = getattr(model, "_orig_mod", model)
    feat_dim = base_model.visual_dim + base_model.temporal_d_out
    criterion_center = CenterLoss(num_classes=num_identities, feat_dim=feat_dim).cuda()
    criterion_temporal = TemporalConsistencyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=wd_stage1,
    )
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=lr_center)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def train_epoch(epoch_idx: int, stage_name: str, optim, lr_sched=None):
        model.train()
        epoch_loss = 0.0
        epoch_loss_id = 0.0
        epoch_loss_tri = 0.0
        epoch_loss_center = 0.0
        epoch_loss_temp = 0.0

        from tqdm import tqdm

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"[{stage_name}] Epoch {epoch_idx}",
            leave=False,
            dynamic_ncols=True,
            disable=bool(cfg.get("_OptunaQuietLog", False)),
        )
        for i, (before, after, pids) in pbar:
            before, after, pids = before.cuda(), after.cuda(), pids.cuda()
            optim.zero_grad()
            optimizer_center.zero_grad()

            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                (feat_b, bn_b, logit_b), (feat_a, bn_a, logit_a), (t_seq_b, t_seq_a) = (
                    model(before, after)
                )
                loss_id = criterion_id(logit_b, pids) + criterion_id(logit_a, pids)
                bn_b_norm = F.normalize(bn_b, p=2, dim=1)
                bn_a_norm = F.normalize(bn_a, p=2, dim=1)
                loss_tri = criterion_triplet(bn_b_norm, pids) + criterion_triplet(
                    bn_a_norm, pids
                )
                loss_center = criterion_center(bn_b, pids) + criterion_center(
                    bn_a, pids
                )
                loss_temp = criterion_temporal(t_seq_b) + criterion_temporal(t_seq_a)
                loss = (
                    loss_id
                    + lam1 * loss_tri
                    + lam2 * loss_temp
                    + lam3 * loss_center
                )

            if not torch.isfinite(loss):
                print(f"[{i}] [WARNING] Bỏ qua batch (loss={loss.item():.3e} không finite).")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.step(optimizer_center)
            scaler.update()

            epoch_loss += loss.item()
            epoch_loss_id += loss_id.item()
            epoch_loss_tri += loss_tri.item()
            epoch_loss_center += loss_center.item()
            epoch_loss_temp += loss_temp.item()

        if lr_sched:
            lr_sched.step()

        num_batches = max(1, len(dataloader))
        print(
            f"---> TỔNG KẾT [{stage_name}] Epoch {epoch_idx}: "
            f"Tổng Loss = {epoch_loss/num_batches:.4f} | "
            f"ID = {epoch_loss_id/num_batches:.4f} | "
            f"Triplet = {epoch_loss_tri/num_batches:.4f} | "
            f"Temporal = {epoch_loss_temp/num_batches:.4f} | "
            f"Center = {epoch_loss_center/num_batches:.4f}"
        )
        return epoch_loss / num_batches

    start_epoch_stage1 = 1
    start_epoch_stage2 = 1
    best_val_rank1 = 0.0
    best_mean_map = 0.0
    _es = {1: {"best": -1.0, "bad": 0}, 2: {"best": -1.0, "bad": 0}}

    scheduler1 = get_warmup_cosine_scheduler(
        optimizer, warmup_epochs=5, total_epochs=epochs_stage1
    )

    if resume and os.path.isfile(resume):
        print(f"=> Loading checkpoint '{resume}'")
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        stage = checkpoint.get("stage", 2)
        best_val_rank1 = float(checkpoint.get("best_val_rank1", 0.0))
        if resume_best_rank1 >= 0:
            best_val_rank1 = resume_best_rank1
        if stage == 1:
            start_epoch_stage1 = checkpoint["epoch"] + 1
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler1.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            start_epoch_stage1 = epochs_stage1 + 1
            start_epoch_stage2 = checkpoint["epoch"] + 1

    global_step_epoch = 0

    # --- STAGE 1 ---
    if start_epoch_stage1 <= epochs_stage1:
        print("=== START STAGE 1: Train Temporal & Head (Freeze Backbone) ===")
        for epoch in range(start_epoch_stage1, epochs_stage1 + 1):
            global_step_epoch += 1
            avg_loss = train_epoch(epoch, "Stage 1", optimizer, scheduler1)

            checkpoint_data = {
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler1.state_dict(),
                "loss": avg_loss,
                "provenance": provenance(cfg),
                "best_val_rank1": best_val_rank1,
            }
            save_ckpt_atomic(
                checkpoint_data,
                os.path.join(checkpoint_dir, "last_model.pth"),
                quiet=True,
            )

            if has_val and (epoch % val_freq == 0 or epoch == epochs_stage1):
                score, per_n, spread, mean_map = validate_multi_n(
                    model, val_loaders, val_n_list, "Stage 1"
                )
                met = (min(per_n.values()) >= target_rank1) and (spread <= target_spread)
                if score > _es[1]["best"] + early_stop_min_delta:
                    _es[1]["best"] = score
                    _es[1]["bad"] = 0
                    if score > best_val_rank1:
                        best_val_rank1 = score
                        best_mean_map = mean_map
                        checkpoint_data["best_val_rank1"] = best_val_rank1
                        save_ckpt_atomic(
                            checkpoint_data,
                            os.path.join(checkpoint_dir, "best_model.pth"),
                        )
                        print(
                            f"[*] Best mới (MIN qua N) ở Stage 1, epoch {epoch}: {best_val_rank1*100:.2f}%"
                        )
                else:
                    _es[1]["bad"] += 1

                if trial_callback is not None:
                    if trial_callback(global_step_epoch, score):
                        print(f"[INFO] Trial pruned at Stage 1 epoch {epoch} (score={score:.4f})")
                        return {
                            "best_val_rank1": best_val_rank1,
                            "best_mean_map": best_mean_map,
                            "best_model_path": os.path.join(checkpoint_dir, "best_model.pth"),
                        }

                if met and stop_on_target:
                    break
                if early_stop_patience > 0 and _es[1]["bad"] >= early_stop_patience:
                    break

    # --- STAGE 2 ---
    if start_epoch_stage2 <= epochs_stage2:
        print("=== START STAGE 2: End-to-End Fine-tuning ===")
        if "optimizer" in locals():
            del optimizer
        if "scheduler1" in locals():
            del scheduler1
        gc.collect()
        torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if use_compile and hasattr(torch, "_dynamo"):
            try:
                torch._dynamo.reset()
            except Exception:
                pass

        _best_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.isfile(_best_path):
            _bst = torch.load(_best_path, map_location="cpu")
            model.load_state_dict(_bst["model_state_dict"])
            print(
                f"  [OK] Nạp lại best_model.pth làm điểm xuất phát Stage 2 "
                f"(best_val_rank1={_bst.get('best_val_rank1', 0.0)*100:.2f}%)"
            )

        base_model = getattr(model, "_orig_mod", model)
        base_model.unfreeze_backbone()

        n_bn_frozen = 0
        for _m in base_model.backbone.modules():
            if isinstance(_m, nn.BatchNorm2d):
                _m.weight.requires_grad = False
                _m.bias.requires_grad = False
                _m.momentum = 0.01
                n_bn_frozen += 1
        print(
            f"  [INFO] Stage 2: đóng băng weight/bias + momentum=0.01 cho {n_bn_frozen} BatchNorm2d"
        )

        pretrained_params = []
        random_params = []
        new_module_params = []
        NEW_TAGS = (".stem.", ".down.", ".conv1.", ".conv2.", ".conv3.", ".attn.", ".ese.")

        for name, param in base_model.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "convnext_backbone" in name and any(t in name for t in NEW_TAGS):
                new_module_params.append(param)
            elif any(
                tag in name
                for tag in ("convnext_backbone", "swin_backbone", "base", "layer", "ga", "fs")
            ):
                pretrained_params.append(param)
            else:
                random_params.append(param)

        param_groups = [
            {"params": pretrained_params, "lr": lr_stage2_backbone},
            {"params": new_module_params, "lr": lr_stage2_temporal},
            {"params": random_params, "lr": lr_stage2_temporal},
            {"params": base_model.temporal_encoder.parameters(), "lr": lr_stage2_temporal},
            {"params": base_model.head.parameters(), "lr": lr_stage2_head},
        ]
        optimizer2 = torch.optim.AdamW(param_groups, weight_decay=wd_stage2)
        scheduler2 = get_warmup_cosine_scheduler(
            optimizer2, warmup_epochs=5, total_epochs=epochs_stage2
        )

        for epoch in range(start_epoch_stage2, epochs_stage2 + 1):
            global_step_epoch += 1
            avg_loss = train_epoch(epoch, "Stage 2", optimizer2, scheduler2)

            checkpoint_data = {
                "epoch": epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer2.state_dict(),
                "scheduler_state_dict": scheduler2.state_dict(),
                "loss": avg_loss,
                "provenance": provenance(cfg),
                "best_val_rank1": best_val_rank1,
            }
            save_ckpt_atomic(
                checkpoint_data,
                os.path.join(checkpoint_dir, "last_model.pth"),
                quiet=True,
            )

            if has_val and (epoch % val_freq == 0 or epoch == epochs_stage2):
                score, per_n, spread, mean_map = validate_multi_n(
                    model, val_loaders, val_n_list, "Stage 2"
                )
                met = (min(per_n.values()) >= target_rank1) and (spread <= target_spread)
                if score > _es[2]["best"] + early_stop_min_delta:
                    _es[2]["best"] = score
                    _es[2]["bad"] = 0
                    if score > best_val_rank1:
                        best_val_rank1 = score
                        best_mean_map = mean_map
                        checkpoint_data["best_val_rank1"] = best_val_rank1
                        save_ckpt_atomic(
                            checkpoint_data,
                            os.path.join(checkpoint_dir, "best_model.pth"),
                        )
                        print(
                            f"[*] Best mới (MIN qua N) ở Stage 2, epoch {epoch}: {best_val_rank1*100:.2f}%"
                        )
                else:
                    _es[2]["bad"] += 1

                if trial_callback is not None:
                    if trial_callback(global_step_epoch, score):
                        print(f"[INFO] Trial pruned at Stage 2 epoch {epoch} (score={score:.4f})")
                        break

                if met and stop_on_target:
                    break
                if early_stop_patience > 0 and _es[2]["bad"] >= early_stop_patience:
                    break

    print("[OK] Training Complete!")
    return {
        "best_val_rank1": best_val_rank1,
        "best_mean_map": best_mean_map,
        "best_model_path": os.path.join(checkpoint_dir, "best_model.pth"),
    }


@torch.no_grad()
def evaluate_uav_reid(
    cfg: Dict[str, Any],
    threshold_file: str | None = None,
    space_override: str | None = None,
    manual_threshold: float | None = None,
) -> Dict[str, Any]:
    """Evaluate UAVReIDNet on test split (both fused and pre_bn spaces)."""
    resolve_pipeline(cfg, "eval", script="eval_uav_reid.py")
    ec = cfg.get("eval", {})
    tc = cfg.get("train", {})
    pc = cfg.get("paths", {})

    model_path = ec.get("model_path", "outputs/reidnet/checkpoints/best_model.pth")
    data_dir = pc.get("data_dir", "./processed")
    query_json = ec.get("query_json", os.path.join(data_dir, "query_test.json"))
    gallery_json = ec.get("gallery_json", os.path.join(data_dir, "gallery_test.json"))
    output_dir = ec.get("output_dir", "outputs/reidnet/eval_results")
    batch_size = int(ec.get("batch_size", 32))
    num_workers = int(ec.get("num_workers", 4))
    backbone_only = bool(ec.get("backbone_only", False))
    intra_sequence = bool(ec.get("intra_sequence", False))
    num_frames = int(ec.get("num_frames", tc.get("num_frames", 12)))
    backbone = str(ec.get("backbone", tc.get("backbone", "dinov3_convnext")))
    econvnext = bool(tc.get("econvnext", True))
    temporal_type = str(ec.get("temporal_type", tc.get("temporal_type", "attention")))
    temporal_pool = str(ec.get("temporal_pool", tc.get("temporal_pool", "attn")))
    temporal_pe = bool(ec.get("temporal_pe", tc.get("temporal_pe", True)))

    threshold_file = threshold_file or ec.get("threshold_file")
    space = space_override or ec.get("space", "fused")

    os.makedirs(output_dir, exist_ok=True)
    _, transform_test = build_uav_transforms()
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"[FAIL] Test directory not found: {test_dir}")

    dataset = EvalDataset(
        test_dir, query_json, gallery_json, transform=transform_test, num_frames=num_frames
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = UAVReIDNet(
        econvnext=econvnext,
        freeze_backbone=False,
        backbone=backbone,
        temporal_type=temporal_type,
        temporal_pool=temporal_pool,
        temporal_pe=temporal_pe,
        temporal_d_model=int(tc.get("temporal_d_model", 512)),
        temporal_d_out=int(tc.get("temporal_d_out", 512)),
        temporal_layers=int(tc.get("temporal_layers", 2)),
        temporal_heads=int(tc.get("temporal_heads", 8)),
        temporal_dropout=float(tc.get("temporal_dropout", 0.1)),
        use_pretrained_backbone=False,
    )
    if not backbone_only and os.path.exists(model_path):
        load_checkpoint_verbose(model, model_path, tag="eval")
    model.cuda().eval()

    spaces = ["backbone"] if backbone_only else (["pre_bn", "fused"] if space == "pre_bn" else ["fused", "pre_bn"])
    threshold_space = space if space in spaces else spaces[0]

    feats = {s: {"qf": [], "gf": []} for s in spaces}
    q_pids, g_pids, q_seq_ids, g_seq_ids = [], [], [], []

    for before, after, pids, _, _, _, seq_ids in dataloader:
        before, after = before.cuda(), after.cuda()
        if backbone_only:
            feats["backbone"]["gf"].append(model(before, backbone_only=True))
            feats["backbone"]["qf"].append(model(after, backbone_only=True))
        else:
            v_g_t, t_g, _ = model.extract_features(before)
            v_q_t, t_q, _ = model.extract_features(after)
            if "fused" in spaces:
                feats["fused"]["gf"].append(model.head(v_g_t, t_g))
                feats["fused"]["qf"].append(model.head(v_q_t, t_q))
            if "pre_bn" in spaces:
                feats["pre_bn"]["gf"].append(torch.cat([v_g_t, t_g], dim=-1))
                feats["pre_bn"]["qf"].append(torch.cat([v_q_t, t_q], dim=-1))

        g_pids.extend(pids.numpy())
        q_pids.extend(pids.numpy())
        q_seq_ids.extend(seq_ids)
        g_seq_ids.extend(seq_ids)

    for s in spaces:
        feats[s]["qf"] = torch.cat(feats[s]["qf"], dim=0)
        feats[s]["gf"] = torch.cat(feats[s]["gf"], dim=0)

    calibrated_threshold = None
    if manual_threshold is not None:
        calibrated_threshold = float(manual_threshold)
    elif threshold_file and os.path.exists(threshold_file):
        with open(threshold_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        thresholds_by_space = calib_data.get("thresholds", {}) or {}
        calibrated_threshold = thresholds_by_space.get(
            threshold_space, calib_data.get("threshold")
        )

    summary = {}
    print("\n" + "=" * 72)
    print(
        f"  {'Space':<10} {'Rank-1':>9} {'Rank-5':>9} {'Rank-10':>9} "
        f"{'mAP':>9} {'mINP':>9} {'TAR@0.1%':>10} {'FAR':>9} {'t*':>10}"
    )
    print("-" * 72)
    for s in spaces:
        cmc, mAP, mINP, _, _ = eval_uav_map_cmc(
            feats[s]["qf"],
            feats[s]["gf"],
            q_pids,
            g_pids,
            q_seq_ids=q_seq_ids,
            g_seq_ids=g_seq_ids,
            intra_sequence=intra_sequence,
        )
        thr = calibrated_threshold if s == threshold_space else None
        tar, far, t_star = compute_tar_at_far(
            feats[s]["qf"], feats[s]["gf"], q_pids, g_pids, far_target=0.001, threshold=thr
        )
        summary[s] = {
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]) if len(cmc) > 4 else 0.0,
            "rank10": float(cmc[9]) if len(cmc) > 9 else 0.0,
            "mAP": float(mAP),
            "mINP": float(mINP),
            "tar": float(tar),
            "far": float(far),
            "threshold": float(t_star),
        }
        print(
            f"  {s:<10} {cmc[0]*100:>8.2f}% {cmc[4]*100:>8.2f}% {cmc[9]*100:>8.2f}% "
            f"{mAP*100:>8.2f}% {mINP*100:>8.2f}% {tar*100:>9.2f}% {far*100:>8.4f}% {t_star:>10.6f}"
        )
    print("=" * 72 + "\n")

    out_json = os.path.join(output_dir, "eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Saved evaluation summary to {out_json}")
    return summary


@torch.no_grad()
def calibrate_uav_threshold(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Sequence-level split Fixed-FAR calibration on UAV-Anti-UAV test set."""
    resolve_pipeline(cfg, "eval", script="calibrate_threshold.py")
    ec = cfg.get("eval", {})
    cc = cfg.get("calibration", {})
    tc = cfg.get("train", {})
    pc = cfg.get("paths", {})

    data_dir = pc.get("data_dir", "./processed")
    query_json = ec.get("query_json", os.path.join(data_dir, "query_test.json"))
    gallery_json = ec.get("gallery_json", os.path.join(data_dir, "gallery_test.json"))
    model_path = ec.get("model_path", "outputs/reidnet/checkpoints/best_model.pth")
    backbone = str(ec.get("backbone", tc.get("backbone", "dinov3_convnext")))
    econvnext = bool(tc.get("econvnext", True))
    num_frames = int(tc.get("num_frames", 12))
    batch_size = int(ec.get("batch_size", 32))
    num_workers = int(ec.get("num_workers", 4))

    cal_ratio = float(cc.get("cal_ratio", 0.4))
    far_target = float(cc.get("far_target", 0.001))
    seed = int(cc.get("seed", 42))
    output_dir = cc.get("output_dir", "outputs/reidnet/calib_results")
    os.makedirs(output_dir, exist_ok=True)

    with open(query_json, "r", encoding="utf-8") as f:
        queries = json.load(f)
    seq_counts = defaultdict(int)
    for q in queries:
        seq_counts[q["sequence_id"]] += 1
    all_seqs = list(seq_counts.keys())
    rng = random.Random(seed)
    rng.shuffle(all_seqs)
    n_cal = max(1, int(len(all_seqs) * cal_ratio))
    cal_seqs = set(all_seqs[:n_cal])
    eval_seqs = set(all_seqs[n_cal:])

    model = UAVReIDNet(
        econvnext=econvnext,
        freeze_backbone=False,
        backbone=backbone,
        temporal_type=str(tc.get("temporal_type", "attention")),
        temporal_pool=str(tc.get("temporal_pool", "attn")),
        temporal_pe=bool(tc.get("temporal_pe", True)),
        temporal_d_model=int(tc.get("temporal_d_model", 512)),
        temporal_d_out=int(tc.get("temporal_d_out", 512)),
        temporal_layers=int(tc.get("temporal_layers", 2)),
        temporal_heads=int(tc.get("temporal_heads", 8)),
        temporal_dropout=float(tc.get("temporal_dropout", 0.1)),
        use_pretrained_backbone=False,
    )
    if os.path.exists(model_path):
        load_checkpoint_verbose(model, model_path, tag="calibrate")
    model.cuda().eval()

    _, transform_test = build_uav_transforms()
    test_dir = os.path.join(data_dir, "test")

    ds_cal = EvalDataset(
        test_dir, query_json, gallery_json, transform=transform_test, num_frames=num_frames, allowed_seq_ids=cal_seqs
    )
    ds_eval = EvalDataset(
        test_dir, query_json, gallery_json, transform=transform_test, num_frames=num_frames, allowed_seq_ids=eval_seqs
    )
    dl_cal = DataLoader(ds_cal, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    spaces = ["fused", "pre_bn", "visual", "temporal"]

    def _extract_spaces(loader):
        acc = {s: {"qf": [], "gf": []} for s in spaces}
        pids = []
        for before, after, pid, _, _, _, _ in loader:
            before, after = before.cuda(), after.cuda()
            v_g, t_g, _ = model.extract_features(before)
            v_q, t_q, _ = model.extract_features(after)
            acc["visual"]["gf"].append(v_g.cpu())
            acc["visual"]["qf"].append(v_q.cpu())
            acc["temporal"]["gf"].append(t_g.cpu())
            acc["temporal"]["qf"].append(t_q.cpu())
            acc["fused"]["gf"].append(model.head(v_g, t_g).cpu())
            acc["fused"]["qf"].append(model.head(v_q, t_q).cpu())
            acc["pre_bn"]["gf"].append(torch.cat([v_g, t_g], dim=-1).cpu())
            acc["pre_bn"]["qf"].append(torch.cat([v_q, t_q], dim=-1).cpu())
            pids.extend(pid.numpy().tolist())
        return {
            s: (torch.cat(acc[s]["qf"], dim=0), torch.cat(acc[s]["gf"], dim=0))
            for s in spaces
        }, np.array(pids)

    feats_cal, pids_cal = _extract_spaces(dl_cal)
    feats_eval, pids_eval = _extract_spaces(dl_eval)

    results = {}
    print(f"\n=== CALIBRATION REPORT (Fixed FAR <= {far_target*100:.2f}%) ===")
    print(f"{'Space':<10} {'t* (cal)':>10} {'TAR (cal)':>11} {'TAR (eval)':>12} {'FAR (eval)':>12}")
    for s in spaces:
        qf_c, gf_c = feats_cal[s]
        qf_e, gf_e = feats_eval[s]
        tar_cal, far_cal, t_star = compute_tar_at_far(qf_c, gf_c, pids_cal, pids_cal, far_target=far_target)
        tar_eval, far_eval, _ = compute_tar_at_far(
            qf_e, gf_e, pids_eval, pids_eval, far_target=far_target, threshold=t_star
        )
        results[s] = {
            "threshold": float(t_star),
            "cal_tar": float(tar_cal),
            "cal_far": float(far_cal),
            "eval_tar": float(tar_eval),
            "eval_far": float(far_eval),
        }
        print(
            f"{s:<10} {t_star:>10.6f} {tar_cal*100:>10.2f}% {tar_eval*100:>11.2f}% {far_eval*100:>11.4f}%"
        )

    threshold_out = os.path.join(output_dir, "calibrated_threshold.json")
    threshold_data = {
        "threshold": results["fused"]["threshold"],
        "space": "fused",
        "thresholds": {s: r["threshold"] for s, r in results.items()},
        "far_target": far_target,
        "cal_ratio": cal_ratio,
        "seed": seed,
        "model_path": model_path,
        "backbone": backbone,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(cfg),
        "results": results,
    }
    with open(threshold_out, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=4)
    print(f"[OK] Saved calibrated thresholds to {threshold_out}")
    return threshold_data
