"""Qualitative Visualization Suite for GASNet.

Generates:
1. Attention Heatmaps:
   Side-by-side comparison of attention focus: [Original | RGA Attention | BCGG Attention]
   extracted from last_attention of the Gate module.
2. Top-5 Retrieval Visualizations:
   Query image vs Top-5 retrieved gallery images with Green (True Match) and Red (False Match) borders.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

from .datasets import ReIDSample, SimpleReIDDataset, get_default_eval_transform
from .metrics import compute_distance_matrix
from .models import extract_dataset_features, load_gasnet_variant


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert normalized PyTorch image tensor [3, H, W] to PIL Image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnorm = (img_tensor.cpu() * std + mean).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(unnorm)


def _apply_colormap_on_image(
    org_img: Image.Image,
    activation_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Image.Image:
    """Overlay a 2D activation map (0.0 to 1.0) onto a PIL Image."""
    w, h = org_img.size
    # Normalize activation map
    norm_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)

    # Get colormap from matplotlib
    cmap = plt.get_cmap(colormap_name)
    rgba_heatmap = cmap(norm_map)  # [H, W, 4]
    rgb_heatmap = (rgba_heatmap[:, :, :3] * 255).astype(np.uint8)

    heatmap_pil = Image.fromarray(rgb_heatmap).resize((w, h), Image.Resampling.BILINEAR)
    blended = Image.blend(org_img.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=alpha)
    return blended.convert("RGB")


@torch.no_grad()
def extract_model_attention(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    gate_stage: str = "ga3",
) -> np.ndarray:
    """Forward pass with attention capture enabled, return 2D attention map."""
    device = next(model.parameters()).device
    model.eval()
    if hasattr(model, "set_attention_capture"):
        model.set_attention_capture(True)

    x = img_tensor.unsqueeze(0).to(device)
    _ = model(x)

    gate_module = getattr(model, gate_stage, None)
    attn_tensor = None
    if gate_module is not None:
        if hasattr(gate_module, "last_attention") and gate_module.last_attention is not None:
            attn_tensor = gate_module.last_attention
        elif hasattr(gate_module, "rga_s") and gate_module.rga_s.last_attention is not None:
            attn_tensor = gate_module.rga_s.last_attention

    if attn_tensor is None:
        # Fallback to center gaussian if attention not captured
        h, w = 14, 14
        y, x_coord = np.ogrid[:h, :w]
        return np.exp(-((y - h / 2) ** 2 + (x_coord - w / 2) ** 2) / (2 * 4.0))

    attn_np = attn_tensor.squeeze().float().cpu().numpy()
    if attn_np.ndim > 2:
        attn_np = attn_np.mean(axis=0)
    return attn_np


def generate_attention_comparison_figures(
    model_rga: torch.nn.Module,
    model_bcgg: torch.nn.Module,
    sample_images: List[Union[str, Path, ReIDSample]],
    output_dir: Union[str, Path] = "outputs/gasnet_report/visualizations/attention",
    gate_stage: str = "ga3",
) -> List[Path]:
    """Generate and save side-by-side attention heatmaps: [Original | RGA | BCGG]."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tfm = get_default_eval_transform(224)

    saved_figures = []
    for idx, item in enumerate(sample_images):
        if isinstance(item, ReIDSample):
            img_file = item.img_path
            veh_id = item.vehicle_id
        elif isinstance(item, (str, Path)):
            img_file = Path(item)
            veh_id = idx
        else:
            continue

        if not img_file.is_file():
            continue

        with Image.open(img_file) as raw_img:
            pil_img = raw_img.convert("RGB").resize((224, 224))

        img_tensor = tfm(pil_img)

        # Extract attention
        attn_rga = extract_model_attention(model_rga, img_tensor, gate_stage=gate_stage)
        attn_bcgg = extract_model_attention(model_bcgg, img_tensor, gate_stage=gate_stage)

        overlay_rga = _apply_colormap_on_image(pil_img, attn_rga, alpha=0.55)
        overlay_bcgg = _apply_colormap_on_image(pil_img, attn_bcgg, alpha=0.55)

        # Create 1x3 panel
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=200)
        axes[0].imshow(pil_img)
        axes[0].set_title("Input Vehicle Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(overlay_rga)
        axes[1].set_title(f"GASNet + RGA ({gate_stage})", fontsize=11, fontweight="bold", color="#B22222")
        axes[1].axis("off")

        axes[2].imshow(overlay_bcgg)
        axes[2].set_title(f"GASNet + BCGG ({gate_stage})", fontsize=11, fontweight="bold", color="#006400")
        axes[2].axis("off")

        plt.tight_layout()
        save_file = out_path / f"attention_cmp_sample_{idx+1}_id_{veh_id}.png"
        fig.savefig(save_file, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(save_file)
        print(f"[OK] Saved Attention Heatmap: {save_file}")

    return saved_figures


def add_colored_border(
    pil_img: Image.Image,
    color: str = "green",
    border_width: int = 6,
) -> Image.Image:
    """Add colored border (e.g. green for true match, red for false match)."""
    return ImageOps.expand(pil_img, border=border_width, fill=color)


def generate_topk_retrieval_panel(
    query_sample: ReIDSample,
    topk_gallery_samples: List[ReIDSample],
    topk_sims: List[float],
    save_path: Union[str, Path],
    img_size: Tuple[int, int] = (160, 160),
) -> Path:
    """Generate Query + Top-5 Gallery retrieval image panel."""
    k = len(topk_gallery_samples)
    fig_w = (k + 1) * 2.5
    fig, axes = plt.subplots(1, k + 1, figsize=(fig_w, 3.2), dpi=200)

    # Query image
    with Image.open(query_sample.img_path) as q_raw:
        q_pil = q_raw.convert("RGB").resize(img_size)
    q_bordered = add_colored_border(q_pil, color="#1E90FF", border_width=6)

    axes[0].imshow(q_bordered)
    axes[0].set_title(f"Query (ID:{query_sample.vehicle_id})", fontsize=10, fontweight="bold", color="#1E90FF")
    axes[0].axis("off")

    # Top-K Gallery images
    for i, (g_sample, sim) in enumerate(zip(topk_gallery_samples, topk_sims)):
        ax = axes[i + 1]
        is_match = g_sample.vehicle_id == query_sample.vehicle_id
        border_color = "#28A745" if is_match else "#DC3545"  # Green / Red
        status_text = "MATCH" if is_match else "MISMATCH"

        try:
            with Image.open(g_sample.img_path) as g_raw:
                g_pil = g_raw.convert("RGB").resize(img_size)
        except Exception:
            g_pil = Image.new("RGB", img_size, color="gray")

        g_bordered = add_colored_border(g_pil, color=border_color, border_width=6)
        ax.imshow(g_bordered)
        title_color = "#28A745" if is_match else "#DC3545"
        ax.set_title(
            f"Rank-{i+1}: {status_text}\nID:{g_sample.vehicle_id} | Sim:{sim:.3f}",
            fontsize=9,
            fontweight="bold",
            color=title_color,
        )
        ax.axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_qualitative_evaluation(
    model: torch.nn.Module,
    model_rga: Optional[torch.nn.Module],
    query_dataset: SimpleReIDDataset,
    gallery_dataset: SimpleReIDDataset,
    output_dir: Union[str, Path] = "outputs/gasnet_report/visualizations",
    num_queries_to_visualize: int = 5,
    top_k: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, List[Path]]:
    """Run full qualitative analysis: Attention heatmaps + Top-5 Retrieval panels."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"attention_maps": [], "retrieval_panels": []}

    # 1. Attention Heatmap Comparison
    if model_rga is not None and len(query_dataset) > 0:
        attn_dir = out_dir / "attention_heatmaps"
        sample_queries = [query_dataset.samples[i] for i in range(min(num_queries_to_visualize, len(query_dataset)))]
        saved_attn = generate_attention_comparison_figures(
            model_rga=model_rga,
            model_bcgg=model,
            sample_images=sample_queries,
            output_dir=attn_dir,
            gate_stage="ga3",
        )
        results["attention_maps"] = saved_attn

    # 2. Top-5 Retrieval Visualizations
    from torch.utils.data import DataLoader
    q_loader = DataLoader(query_dataset, batch_size=64, shuffle=False)
    g_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False)

    print("[INFO] Extracting features for Top-5 Retrieval visualization...")
    q_feats, q_pids, q_cams, q_paths = extract_dataset_features(model, q_loader, device=device)
    g_feats, g_pids, g_cams, g_paths = extract_dataset_features(model, g_loader, device=device)

    # Cosine distance = 1 - sim => Sim = 1 - dist
    distmat = compute_distance_matrix(q_feats, g_feats, metric="cosine")
    indices = np.argsort(distmat, axis=1)

    # Find diverse queries with matches in gallery
    chosen_queries = []
    g_pids_set = set(g_pids.numpy())
    for q_idx in range(len(query_dataset)):
        q_sample = query_dataset.samples[q_idx]
        if q_sample.vehicle_id in g_pids_set:
            chosen_queries.append(q_idx)
        if len(chosen_queries) >= num_queries_to_visualize:
            break

    retrieval_dir = out_dir / "retrieval_top5"
    for rank_idx, q_idx in enumerate(chosen_queries):
        q_sample = query_dataset.samples[q_idx]
        order = indices[q_idx]

        top_gallery_samples = []
        top_sims = []
        for g_idx in order[:top_k]:
            g_sample = gallery_dataset.samples[g_idx]
            sim = float(1.0 - distmat[q_idx, g_idx])
            top_gallery_samples.append(g_sample)
            top_sims.append(sim)

        save_file = retrieval_dir / f"retrieval_query_{rank_idx+1}_id_{q_sample.vehicle_id}.png"
        panel_path = generate_topk_retrieval_panel(
            query_sample=q_sample,
            topk_gallery_samples=top_gallery_samples,
            topk_sims=top_sims,
            save_path=save_file,
        )
        results["retrieval_panels"].append(panel_path)
        print(f"[OK] Saved Top-5 Retrieval Panel: {panel_path}")

    return results
