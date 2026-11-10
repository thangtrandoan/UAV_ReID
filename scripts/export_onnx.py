#!/usr/bin/env python3
"""Unified ONNX Export for GASNet (Stage 1) and UAVReIDNet (Stage 2)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uav_reid.config import load_yaml_config
from uav_reid.gasnet import GASNet, infer_backbone_name, infer_econvnext, infer_gate_config
from uav_reid.reidnet import UAVReIDNet, load_checkpoint_verbose
from uav_reid.utils import normalize_state_dict


def export_gasnet(args: argparse.Namespace) -> None:
    ckpt_path = args.checkpoint
    state_dict = None
    if os.path.exists(ckpt_path):
        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "model_state_dict" in raw:
            raw = raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        state_dict = normalize_state_dict(raw)

    gate_config = infer_gate_config(state_dict or {})
    backbone = args.backbone or infer_backbone_name(state_dict or {}, default="dinov3_convnext")
    econvnext = infer_econvnext(state_dict or {}) or args.econvnext
    cls_w = (state_dict or {}).get("classifier_global.weight")
    num_classes = int(cls_w.shape[0]) if cls_w is not None else 6302

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        backbone=backbone,
        econvnext=econvnext,
        use_gem=True,
        **gate_config,
    )
    if state_dict is not None:
        ms = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in ms and v.shape == ms[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"[OK] Loaded {len(filtered)}/{len(ms)} tensors from {ckpt_path}")

    model.eval()
    if args.deploy:
        model = model.to_deploy()
        print("[OK] Folded BCGG Conv/BN and RepDW branches for deployment")

    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device="cpu")
    with torch.no_grad():
        test_out = model(dummy_input)
    output_names = (
        [f"output_{i}" for i in range(len(test_out))]
        if isinstance(test_out, (tuple, list))
        else ["output"]
    )

    out_path = args.output or "outputs/gasnet/gasnet.onnx"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
    )
    print(f"[OK] Saved GASNet ONNX to {out_path}")


def export_uav_reid(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config)
    tc = cfg.get("train", {})
    backbone = args.backbone or tc.get("backbone", "dinov3_convnext")
    econvnext = args.econvnext or bool(tc.get("econvnext", True))
    temporal_type = tc.get("temporal_type", "attention")
    temporal_pool = tc.get("temporal_pool", "attn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVReIDNet(
        freeze_backbone=False,
        backbone=backbone,
        temporal_type=temporal_type,
        temporal_pool=temporal_pool,
        econvnext=econvnext,
        use_pretrained_backbone=False,
    )
    if os.path.exists(args.checkpoint):
        load_checkpoint_verbose(model, args.checkpoint, tag="export")
    model.eval().to(device)

    if args.deploy and hasattr(model.backbone, "to_deploy"):
        model.backbone = model.backbone.to_deploy().to(device)
        print("[OK] Folded GASNet backbone BCGG branches for deployment")

    class BackboneWrapper(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb

        def forward(self, x):
            feats = self.bb(x)
            if isinstance(feats, tuple):
                if isinstance(feats[0], tuple):
                    return torch.cat([feats[0][0], feats[0][1]], dim=-1)
                return torch.cat([feats[0], feats[1]], dim=-1)
            return feats

    class TemporalWrapper(torch.nn.Module):
        def __init__(self, te, hd):
            super().__init__()
            self.te = te
            self.hd = hd

        def forward(self, seq_feats):
            vis = seq_feats.mean(dim=1)
            t_tok, _ = self.te(seq_feats)
            bn = self.hd(vis, t_tok)
            return torch.nn.functional.normalize(bn, p=2, dim=1)

    prefix = args.output or "outputs/reidnet/uav_reid_"
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    bb_wrapper = BackboneWrapper(model.backbone).eval()
    dummy_img = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    torch.onnx.export(
        bb_wrapper,
        dummy_img,
        prefix + "backbone.onnx",
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input_img"],
        output_names=["cnn_feat"],
        dynamic_axes={"input_img": {0: "batch_size"}, "cnn_feat": {0: "batch_size"}},
    )
    print(f"[OK] Saved {prefix}backbone.onnx")

    t_wrapper = TemporalWrapper(model.temporal_encoder, model.head).eval()
    dummy_seq = torch.randn(args.batch_size, args.num_frames, model.visual_dim, device=device)
    torch.onnx.export(
        t_wrapper,
        dummy_seq,
        prefix + "temporal.onnx",
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["seq_feats"],
        output_names=["reid_embed"],
        dynamic_axes={
            "seq_feats": {0: "batch_size", 1: "num_frames"},
            "reid_embed": {0: "batch_size"},
        },
    )
    print(f"[OK] Saved {prefix}temporal.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GASNet or UAVReIDNet to ONNX")
    parser.add_argument("--stage", choices=["gasnet", "uav_reid"], default="uav_reid")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "uav_reid.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default=None, help="Output file (gasnet) or prefix (uav_reid)")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--econvnext", action="store_true")
    parser.add_argument("--deploy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.stage == "gasnet":
        export_gasnet(args)
    else:
        export_uav_reid(args)


if __name__ == "__main__":
    main()
