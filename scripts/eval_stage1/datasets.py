"""Dataset loaders for Stage 1 (VeRi-776, VehicleID_V1.0, VRU, VRAI)."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from .common import resolve_path
except ImportError:
    from common import resolve_path


@dataclass
class ReIDSample:
    img_path: Path
    vehicle_id: int
    camera_id: int = -1
    image_name: str = ""


class SimpleReIDDataset(Dataset):
    """Generic image Re-ID dataset yielding `(tensor_img, vehicle_id, camera_id, path_str)`."""

    def __init__(self, samples: List[ReIDSample], transform=None, max_retries: int = 3):
        self.samples = samples
        self.transform = transform
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> Image.Image:
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except OSError as exc:
                last_err = exc
        raise RuntimeError(f"[FAIL] Failed to read image {path}: {last_err}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        s = self.samples[idx]
        img = self._load_rgb(s.img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, s.vehicle_id, s.camera_id, str(s.img_path)


def build_eval_transform(
    img_size: int = 224,
    legacy_resize: bool = False,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> Tuple[transforms.Compose, Dict[str, object]]:
    norm = transforms.Normalize(mean=list(mean), std=list(std))
    if legacy_resize:
        tfm = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((img_size, img_size)),
                transforms.ToTensor(),
                norm,
            ]
        )
        desc = f"Resize(256,256,BICUBIC)+CenterCrop({img_size},{img_size})"
    else:
        tfm = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                norm,
            ]
        )
        desc = f"Resize({img_size},{img_size},BICUBIC)"
    meta = {
        "img_size": img_size,
        "legacy_resize": bool(legacy_resize),
        "description": desc,
        "mean": list(mean),
        "std": list(std),
    }
    return tfm, meta


def make_eval_loader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def parse_veri_filename(fname: str) -> Tuple[int, int]:
    parts = Path(fname).stem.split("_")
    vid = int(parts[0])
    cam_str = parts[1].replace("c", "") if len(parts) > 1 else "-1"
    cam_id = int(cam_str) if cam_str.isdigit() else -1
    return vid, cam_id


def load_veri776(
    veri_dir: Path | str,
    transform,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset, Dict[str, object]]:
    veri_path = resolve_path(veri_dir)
    if not veri_path.is_dir():
        raise FileNotFoundError(f"[FAIL] VeRi-776 directory not found: {veri_path}")

    q_dir = veri_path / "image_query"
    g_dir = veri_path / "image_test"
    if not q_dir.is_dir() or not g_dir.is_dir():
        raise FileNotFoundError(f"[FAIL] VeRi-776 image_query/image_test missing in {veri_path}")

    def _read_list(txt_file: Path, img_dir: Path) -> List[str]:
        if txt_file.is_file():
            return [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return sorted(p.name for p in img_dir.glob("*.jpg"))

    q_names = _read_list(veri_path / "name_query.txt", q_dir)
    g_names = _read_list(veri_path / "name_test.txt", g_dir)

    q_samples: List[ReIDSample] = []
    for name in q_names:
        p = q_dir / name
        if p.is_file():
            vid, cam = parse_veri_filename(name)
            q_samples.append(ReIDSample(img_path=p, vehicle_id=vid, camera_id=cam, image_name=name))

    g_samples: List[ReIDSample] = []
    for name in g_names:
        p = g_dir / name
        if p.is_file():
            vid, cam = parse_veri_filename(name)
            g_samples.append(ReIDSample(img_path=p, vehicle_id=vid, camera_id=cam, image_name=name))

    meta = {
        "dataset": "VeRi-776",
        "split": "Full Test",
        "dir": str(veri_path),
        "cross_camera_filter": True,
        "total_queries_before_filter": len(q_samples),
        "gallery_size": len(g_samples),
        "unique_query_ids": len({s.vehicle_id for s in q_samples}),
        "unique_gallery_ids": len({s.vehicle_id for s in g_samples}),
    }
    return SimpleReIDDataset(q_samples, transform=transform), SimpleReIDDataset(g_samples, transform=transform), meta


def load_vehicleid_split_pool(
    vehicleid_dir: Path | str,
    split_size: int,
    transform,
) -> Tuple[SimpleReIDDataset, Dict[int, List[int]], Dict[str, object]]:
    """Load VehicleID_V1.0 split (`test_list_{800,1600,2400}.txt`) into a single pool."""
    veh_path = resolve_path(vehicleid_dir)
    split_file = veh_path / "train_test_split" / f"test_list_{split_size}.txt"
    img_dir = veh_path / "image"
    if not split_file.is_file() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"[FAIL] VehicleID_V1.0 split {split_size} not found at {veh_path} "
            "(ensure VehicleID_V1.0.zip is extracted with test_list_*.txt and image/)."
        )

    samples: List[ReIDSample] = []
    indices_by_id: Dict[int, List[int]] = {}
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_id, vid = parts[0], int(parts[1])
            img_name = img_id if img_id.endswith(".jpg") else f"{img_id}.jpg"
            img_path = img_dir / img_name
            if not img_path.is_file():
                raise FileNotFoundError(f"[FAIL] Missing VehicleID image referenced in {split_file.name}: {img_path}")
            idx = len(samples)
            samples.append(ReIDSample(img_path=img_path, vehicle_id=vid, camera_id=-1, image_name=img_name))
            indices_by_id.setdefault(vid, []).append(idx)

    valid_ids = [vid for vid, idxs in indices_by_id.items() if len(idxs) >= 2]
    if not valid_ids:
        raise RuntimeError(f"[FAIL] No IDs with >= 2 images found in {split_file}")

    meta = {
        "dataset": "VehicleID",
        "split": str(split_size),
        "split_file": str(split_file),
        "total_images": len(samples),
        "total_ids": len(indices_by_id),
        "valid_ids_ge_2_imgs": len(valid_ids),
    }
    return SimpleReIDDataset(samples, transform=transform), indices_by_id, meta


def split_gallery_query_indices(
    indices_by_id: Dict[int, List[int]],
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Sample 1 random gallery index per valid ID (`>= 2` images), rest as queries."""
    rng = random.Random(seed)
    query_idxs: List[int] = []
    gallery_idxs: List[int] = []
    for vid in sorted(indices_by_id.keys()):
        group = indices_by_id[vid]
        if len(group) < 2:
            continue
        g_pos = rng.randrange(len(group))
        gallery_idxs.append(group[g_pos])
        query_idxs.extend(group[:g_pos])
        query_idxs.extend(group[g_pos + 1 :])
    return query_idxs, gallery_idxs


def load_vru_split(
    vru_dir: Path | str,
    split_name: str,
    seed: int,
    transform,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset, Dict[str, object]]:
    vru_path = resolve_path(vru_dir)
    split_map = {
        "Small": ("Small (1200)", "test_list_1200.txt"),
        "Medium": ("Medium (2400)", "test_list_2400.txt"),
        "Large": ("Large (8000)", "test_list_8000.txt"),
        "Big": ("Large (8000)", "test_list_8000.txt"),
    }
    if split_name not in split_map:
        raise ValueError(f"[FAIL] Unknown VRU split: {split_name!r}")
    split_display, fname = split_map[split_name]
    split_file = vru_path / "train_test_split" / fname
    pic_dir = vru_path / "Pic"
    if not split_file.is_file() or not pic_dir.is_dir():
        raise FileNotFoundError(f"[FAIL] VRU split {split_name} missing under {vru_path}")

    samples_by_id: Dict[int, List[ReIDSample]] = {}
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_id, vid = parts[0], int(parts[1])
            img_name = img_id if img_id.endswith(".jpg") else f"{img_id}.jpg"
            img_path = pic_dir / img_name
            if img_path.is_file():
                samples_by_id.setdefault(vid, []).append(
                    ReIDSample(img_path=img_path, vehicle_id=vid, camera_id=-1, image_name=img_name)
                )

    rng = random.Random(seed)
    q_samples: List[ReIDSample] = []
    g_samples: List[ReIDSample] = []
    for vid, group in samples_by_id.items():
        if len(group) < 2:
            continue
        g_idx = rng.randrange(len(group))
        g_samples.append(group[g_idx])
        q_samples.extend(group[:g_idx])
        q_samples.extend(group[g_idx + 1 :])

    meta = {
        "dataset": "VRU",
        "split": split_display,
        "split_file": str(split_file),
        "seed": seed,
        "cross_camera_filter": False,
        "total_queries_before_filter": len(q_samples),
        "gallery_size": len(g_samples),
        "unique_query_ids": len({s.vehicle_id for s in q_samples}),
        "unique_gallery_ids": len({s.vehicle_id for s in g_samples}),
    }
    return SimpleReIDDataset(q_samples, transform=transform), SimpleReIDDataset(g_samples, transform=transform), meta


def load_vrai_val(
    vrai_dir: Path | str,
    num_ids: int = 1000,
    seed: int = 42,
    transform=None,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset, Dict[str, object]]:
    """Load the exact 1,000-ID held-out validation split (disjoint from the 5,302 train IDs)."""
    vrai_path = resolve_path(vrai_dir)
    ann_path = vrai_path / "train_annotation.pkl"
    images_dir = vrai_path / "images_train"
    if not ann_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(f"[FAIL] VRAI files missing under {vrai_path}")

    with ann_path.open("rb") as f:
        ann = pickle.load(f)

    samples_by_id: Dict[int, List[ReIDSample]] = {}
    for name in ann.get("train_im_names", []):
        parts = Path(name).stem.split("_")
        if not parts[0].isdigit():
            continue
        vid = int(parts[0])
        cam_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        img_path = images_dir / name
        if img_path.is_file():
            samples_by_id.setdefault(vid, []).append(
                ReIDSample(img_path=img_path, vehicle_id=vid, camera_id=cam_id, image_name=name)
            )

    valid_ids = sorted(vid for vid, items in samples_by_id.items() if len(items) >= 2)
    rng = random.Random(seed)
    rng.shuffle(valid_ids)
    selected_ids = sorted(valid_ids[:num_ids])
    train_ids = set(valid_ids[num_ids:])
    overlap = set(selected_ids) & train_ids
    if overlap:
        raise RuntimeError(f"[FAIL] VRAI train/val ID overlap detected: {len(overlap)} IDs")

    q_samples: List[ReIDSample] = []
    g_samples: List[ReIDSample] = []
    for vid in selected_ids:
        group = samples_by_id[vid]
        g_idx = rng.randrange(len(group))
        g_samples.append(group[g_idx])
        q_samples.extend(group[:g_idx])
        q_samples.extend(group[g_idx + 1 :])

    meta = {
        "dataset": "VRAI",
        "split": f"Val ({num_ids} IDs)",
        "dir": str(vrai_path),
        "seed": seed,
        "train_disjoint_ids": len(train_ids),
        "val_ids": len(selected_ids),
        "overlap_with_train": len(overlap),
        "cross_camera_filter": True,
        "total_queries_before_filter": len(q_samples),
        "gallery_size": len(g_samples),
        "unique_query_ids": len({s.vehicle_id for s in q_samples}),
        "unique_gallery_ids": len({s.vehicle_id for s in g_samples}),
    }
    return SimpleReIDDataset(q_samples, transform=transform), SimpleReIDDataset(g_samples, transform=transform), meta
