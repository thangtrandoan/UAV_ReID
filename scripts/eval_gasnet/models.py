"""GASNet Model Factory and Feature Extraction.

Constructs and loads GASNet variants:
1. ResNet-50 + RGA (GASNet gốc)
2. DINO-ConvNeXt + RGA
3. DINO-ConvNeXt + BCGG (Train mode)
4. DINO-ConvNeXt + BCGG (Deploy mode - RepDW Folded)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_reid.gasnet.bcgg import infer_gate_config
from uav_reid.gasnet.model import GASNet, infer_backbone_name, infer_econvnext
from uav_reid.utils.common import normalize_state_dict


def find_checkpoint(model_key: str, custom_weights_dir: Optional[Union[str, Path]] = None) -> Path:
    """Find appropriate checkpoint path for a given model key.

    Searches in custom_weights_dir -> ./weights -> weights -> /weights.
    """
    candidates_dirs = []
    if custom_weights_dir is not None:
        candidates_dirs.append(Path(custom_weights_dir))
    candidates_dirs.extend([
        Path("./weights"),
        Path("weights"),
        Path("../weights"),
        Path("/weights"),
    ])

    patterns = {
        "resnet_rga": ["*resnet*rga*.pth", "*resnet50*rga*.pth"],
        "convnext_rga": ["*convnext*rga*.pth"],
        "convnext_bcgg": ["*convnext*bcgg*.pth"],
    }

    targets = patterns.get(model_key, [f"*{model_key}*.pth"])

    for wdir in candidates_dirs:
        if not wdir.is_dir():
            continue
        for pat in targets:
            matches = sorted(list(wdir.glob(pat)))
            if matches:
                return matches[0].resolve()

    # Fallback to direct naming
    fallback_name = f"gasnet_{model_key}.best.pth"
    return (Path("weights") / fallback_name).resolve()


def load_gasnet_variant(
    variant: str,
    weights_path: Optional[Union[str, Path]] = None,
    num_classes: int = 6302,
    device: Optional[torch.device] = None,
) -> Tuple[GASNet, Dict[str, str]]:
    """Instantiate and load weights for a specific GASNet variant.

    Variants:
    - 'resnet_rga': GASNet gốc (ResNet-50 + RGA)
    - 'convnext_rga': GASNet + DINO-ConvNeXt + RGA
    - 'convnext_bcgg_train': GASNet + DINO-ConvNeXt + BCGG (Train mode)
    - 'convnext_bcgg_deploy': GASNet + DINO-ConvNeXt + BCGG (Deploy mode - RepDW Folded)

    Returns:
        (model, metadata_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_deploy = False
    if variant == "resnet_rga":
        default_bb = "resnet50_ibn"
        gate = "rga"
        ckpt_key = "resnet_rga"
    elif variant == "convnext_rga":
        default_bb = "dinov3_convnext"
        gate = "rga"
        ckpt_key = "convnext_rga"
    elif variant == "convnext_bcgg_train":
        default_bb = "dinov3_convnext"
        gate = "bcgg"
        ckpt_key = "convnext_bcgg"
    elif variant == "convnext_bcgg_deploy":
        default_bb = "dinov3_convnext"
        gate = "bcgg"
        ckpt_key = "convnext_bcgg"
        is_deploy = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ckpt_file = Path(weights_path) if weights_path else find_checkpoint(ckpt_key)
    state_dict = None
    econvnext = False
    backbone = default_bb

    if ckpt_file.is_file():
        raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            raw = raw["model_state_dict"]
        state_dict = normalize_state_dict(raw)
        
        # 1. Infer num_classes from state_dict if available
        cls_w = state_dict.get("classifier_global.weight")
        if cls_w is not None:
            num_classes = int(cls_w.shape[0])

        # 2. Infer backbone and econvnext from state_dict dynamically
        backbone = infer_backbone_name(state_dict, default=default_bb)
        econvnext = infer_econvnext(state_dict)

    gate_config = infer_gate_config(state_dict or {}) if gate == "bcgg" else {"gate": "rga"}

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        backbone=backbone,
        econvnext=econvnext,
        use_gem=True,
        **gate_config,
    )

    loaded_status = "random_init"
    if state_dict is not None:
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        loaded_status = f"loaded_{len(filtered)}/{len(model_sd)}_keys"
        if len(filtered) == len(model_sd):
            print(f"[OK] Checkpoint {ckpt_file.name} matched 100% ({len(filtered)}/{len(model_sd)} keys) for {backbone} ({gate})")
        else:
            print(f"[INFO] Loaded {len(filtered)}/{len(model_sd)} parameters for {backbone} ({gate}) from {ckpt_file.name}")

    model.eval()

    if is_deploy:
        model = model.to_deploy()
        loaded_status += " (RepDW Folded)"

    model = model.to(device)

    meta = {
        "variant": variant,
        "backbone": backbone,
        "gate": gate,
        "deploy_mode": str(is_deploy),
        "checkpoint": str(ckpt_file) if ckpt_file.exists() else "None",
        "status": loaded_status,
    }
    return model, meta


@torch.no_grad()
def extract_dataset_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    use_fp16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract concatenated features [bn_global, bn_fs] for an entire DataLoader.

    Returns:
        (features [N, D], vehicle_ids [N], camera_ids [N], image_paths [N])
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    model.eval()
    all_feats: List[torch.Tensor] = []
    all_pids: List[torch.Tensor] = []
    all_cams: List[torch.Tensor] = []
    all_paths: List[str] = []

    use_autocast = device.type == "cuda" and use_fp16
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    for imgs, pids, cams, paths in dataloader:
        imgs = imgs.to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_autocast, dtype=dtype):
            outputs = model(imgs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            # In GASNet eval mode: outputs is (bn_global, bn_fs)
            emb = torch.cat(list(outputs), dim=1)

        all_feats.append(emb.float().cpu())
        all_pids.append(pids.cpu())
        all_cams.append(cams.cpu())
        all_paths.extend(paths)

    feats_tensor = torch.cat(all_feats, dim=0)
    pids_tensor = torch.cat(all_pids, dim=0)
    cams_tensor = torch.cat(all_cams, dim=0)

    # L2 normalize features
    feats_tensor = F.normalize(feats_tensor, p=2, dim=1)

    return feats_tensor, pids_tensor, cams_tensor, all_paths
