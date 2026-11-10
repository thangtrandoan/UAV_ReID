"""UAVReIDNet: Visual Backbone (GASNet) + Temporal Encoder (Mamba/Attention) + ReID Head."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from uav_reid.gasnet import GASNet, infer_backbone_name, infer_econvnext, infer_gate_config
from uav_reid.utils import normalize_state_dict

from .components import ReIDHead
from .registry import build_temporal_encoder
from .temporal_mamba import HAS_MAMBA


class UAVReIDNet(nn.Module):
    def __init__(
        self,
        gasnet_weights_path: str | None = None,
        num_identities: int = 1000,
        freeze_backbone: bool = True,
        backbone: str = "resnet50_ibn",
        temporal_pool: str = "attn",
        temporal_pe: bool = True,
        temporal_type: str = "mamba",
        econvnext: bool = False,
        gate: str | None = None,
        gate_hidden: int | None = None,
        gate_groups: int | None = None,
        gate_kernel: int | None = None,
        gate_channel_kernel: int | None = None,
        gate_inject_value: bool | None = None,
        use_gem: bool = True,
        gem_p: float = 3.0,
        temporal_d_model: int = 512,
        temporal_d_out: int = 512,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
        use_pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.temporal_type = temporal_type

        state_dict = None
        if gasnet_weights_path and os.path.exists(gasnet_weights_path):
            raw_ckpt = torch.load(gasnet_weights_path, map_location="cpu")
            if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
                raw_ckpt = raw_ckpt["model_state_dict"]
            elif isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
                raw_ckpt = raw_ckpt["state_dict"]
            state_dict = normalize_state_dict(raw_ckpt)

        # Auto-infer gate configuration from GASNet checkpoint if not explicitly overridden
        inferred_gate = infer_gate_config(state_dict or {}, default_kind=gate or "bcgg")
        effective_gate = gate or inferred_gate["gate"]
        effective_hidden = gate_hidden if gate_hidden is not None else inferred_gate["gate_hidden"]
        effective_groups = gate_groups if gate_groups is not None else inferred_gate["gate_groups"]
        effective_kernel = gate_kernel if gate_kernel is not None else inferred_gate["gate_kernel"]
        effective_ch_kernel = (
            gate_channel_kernel
            if gate_channel_kernel is not None
            else inferred_gate["gate_channel_kernel"]
        )
        effective_inject_val = (
            gate_inject_value
            if gate_inject_value is not None
            else inferred_gate["gate_inject_value"]
        )

        if state_dict is not None:
            detected_backbone = infer_backbone_name(state_dict, default=backbone)
            if detected_backbone != backbone:
                print(
                    f"  [WARNING] Config backbone={backbone!r} khác checkpoint={detected_backbone!r}. "
                    f"Giữ config backbone={backbone!r}."
                )
            if not econvnext and infer_econvnext(state_dict):
                print("  [INFO] Auto-detected E-ConvNeXt layout in GASNet checkpoint.")
                econvnext = True

        # Avoid downloading ImageNet/HF pretrained weights if we are loading a trained GASNet checkpoint
        need_online_pretrain = use_pretrained_backbone and (state_dict is None)

        self.backbone = GASNet(
            num_classes=num_identities,
            use_pretrained=need_online_pretrain,
            backbone=backbone,
            use_gem=use_gem,
            gem_p=gem_p,
            econvnext=econvnext,
            gate=effective_gate,
            gate_hidden=effective_hidden,
            gate_groups=effective_groups,
            gate_kernel=effective_kernel,
            gate_channel_kernel=effective_ch_kernel,
            gate_inject_value=effective_inject_val,
        )

        if state_dict is not None:
            model_state = self.backbone.state_dict()
            filtered = {
                k: v
                for k, v in state_dict.items()
                if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
            }
            missing = [k for k in model_state if k not in filtered]
            unexpected = [k for k in state_dict if k not in filtered]
            self.backbone.load_state_dict(filtered, strict=False)
            print(
                f"  [OK] Đã load GASNet weights từ {gasnet_weights_path} "
                f"({len(filtered)}/{len(model_state)} keys, gate={effective_gate})"
            )
            non_cls_missing = [
                k for k in missing if not k.startswith("classifier_")
            ]
            if non_cls_missing:
                print(
                    f"  [WARNING] {len(non_cls_missing)} feature keys KHÔNG tìm thấy / lệch shape:"
                )
                for mk in non_cls_missing[:10]:
                    print(f"    - {mk}")
            if any("convnext_backbone" in m for m in non_cls_missing):
                print(
                    "  [ERROR] CẢNH BÁO NGHIÊM TRỌNG: convnext_backbone KHÔNG được load đầy đủ từ GASNet weights!"
                )
        elif gasnet_weights_path:
            print(f"  [WARNING] Không tìm thấy pre-trained GASNet weights tại {gasnet_weights_path}")

        self.backbone.return_raw_features_eval = True

        if freeze_backbone:
            self.freeze_backbone()

        if backbone == "dinov3_convnext":
            visual_dim = 960  # 768 (Global) + 192 (FS)
        else:
            visual_dim = 2560  # 2048 (Global) + 512 (FS)
        self.visual_dim = visual_dim
        self.temporal_d_out = temporal_d_out

        self.temporal_encoder = build_temporal_encoder(
            temporal_type,
            d_in=visual_dim,
            d_model=temporal_d_model,
            d_out=temporal_d_out,
            max_seq_len=64,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=temporal_dropout,
            pool=temporal_pool,
            use_pe=temporal_pe,
        )
        print(
            f"  [INFO] Temporal encoder: {type(self.temporal_encoder).__name__}"
            + (
                ""
                if temporal_type != "mamba"
                else f" ({'mamba_ssm' if HAS_MAMBA else 'SimpleS6Block fallback'})"
            )
        )

        self.head = ReIDHead(
            in_dim=visual_dim + temporal_d_out, num_identities=num_identities
        )

    def freeze_backbone(self) -> None:
        print("  [INFO] Freezing Visual Backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        print("  [INFO] Unfreezing Visual Backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extract_features(self, clips: torch.Tensor):
        B, N, C, H, W = clips.shape
        clips = clips.view(B * N, C, H, W).contiguous(memory_format=torch.channels_last)

        if not next(self.backbone.parameters()).requires_grad:
            self.backbone.eval()

        grad_enabled = next(self.backbone.parameters()).requires_grad and self.training
        with torch.set_grad_enabled(grad_enabled):
            feats = self.backbone(clips)

        if isinstance(feats, tuple):
            if isinstance(feats[0], tuple):
                global_feat = feats[0][0]
                fs_feat = feats[0][1]
            else:
                global_feat = feats[0]
                fs_feat = feats[1]
            feats = torch.cat([global_feat, fs_feat], dim=-1)

        feats = feats.view(B, N, -1)
        visual_feat = feats.mean(dim=1)
        temporal_token, temporal_seq = self.temporal_encoder(feats)
        return visual_feat, temporal_token, temporal_seq

    def forward(self, before_clips: torch.Tensor, after_clips: torch.Tensor | None = None, backbone_only: bool = False):
        if not self.training:
            v_feat, t_token, _ = self.extract_features(before_clips)
            if backbone_only:
                return v_feat
            return self.head(v_feat, t_token)

        v_feat_before, t_token_before, t_seq_before = self.extract_features(before_clips)
        feat_b, bn_feat_b, logits_b = self.head(v_feat_before, t_token_before)

        if after_clips is not None:
            v_feat_after, t_token_after, t_seq_after = self.extract_features(after_clips)
            feat_a, bn_feat_a, logits_a = self.head(v_feat_after, t_token_after)
            return (
                (feat_b, bn_feat_b, logits_b),
                (feat_a, bn_feat_a, logits_a),
                (t_seq_before, t_seq_after),
            )

        return feat_b, bn_feat_b, logits_b
