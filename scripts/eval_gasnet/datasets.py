"""Dataset Loaders for VeRi-776, VRU / VehicleID, and VRAI.

Handles standard Re-ID dataset protocols:
- Path resolution across /data and ./data
- Parsing vehicle IDs and camera IDs
- Query and gallery split generation
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class ReIDSample:
    img_path: Path
    vehicle_id: int
    camera_id: int = -1
    image_name: str = ""


class SimpleReIDDataset(Dataset):
    """Generic Re-ID dataset yielding (tensor_image, vehicle_id, camera_id, path_str)."""

    def __init__(self, samples: List[ReIDSample], transform=None, max_retries: int = 3):
        self.samples = samples
        self.transform = transform
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        last_err = None
        for _ in range(self.max_retries):
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except OSError as err:
                last_err = err
        raise RuntimeError(f"Failed to open image at {path}: {last_err}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        sample = self.samples[idx]
        img = self._load_image(sample.img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.vehicle_id, sample.camera_id, str(sample.img_path)


def resolve_dataset_path(dataset_name: str, custom_root: Optional[Union[str, Path]] = None) -> Path:
    """Find dataset directory.

    Prioritizes the local workspace directory (./data/<dataset_name>) before
    checking system paths like /data, and safely catches any PermissionError.
    """
    project_root = Path(__file__).resolve().parents[2]

    candidates: List[Path] = []

    # 1. Custom root if provided and accessible
    if custom_root is not None:
        cr = Path(custom_root)
        try:
            candidates.append(cr / dataset_name)
            candidates.append(cr)
        except Exception:
            pass

    # 2. Local workspace candidates (preferred)
    candidates.extend([
        project_root / "data" / dataset_name,
        Path.cwd() / "data" / dataset_name,
        Path("./data") / dataset_name,
        Path("data") / dataset_name,
        Path("../data") / dataset_name,
    ])

    # 3. System paths (only if accessible)
    candidates.extend([
        Path("/data") / dataset_name,
        Path("/workspace/data") / dataset_name,
    ])

    for cand in candidates:
        try:
            if cand.is_dir() and os.access(cand, os.R_OK):
                return cand.resolve()
        except (PermissionError, OSError):
            continue

    # Safe fallback: always default to local workspace data folder
    return (project_root / "data" / dataset_name).resolve()


def get_default_eval_transform(
    img_size: Union[int, Tuple[int, int]] = 224,
    legacy_resize: bool = False,
) -> transforms.Compose:
    """Evaluation image transform for GASNet.

    Args:
        img_size: Target image size (default 224).
        legacy_resize:
            - False (Default / Standard Vehicle Re-ID): Trực tiếp Resize((224, 224)).
              Bảo toàn toàn bộ hộp bao xe, không bị cắt xén đèn, biển số, bánh xe.
            - True (Legacy ImageNet Style): Resize((256, 256)) rồi CenterCrop((224, 224)).
    """
    size = (img_size, img_size) if isinstance(img_size, int) else img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if legacy_resize:
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])


# --------------------------------------------------------------------------- #
# VeRi-776 Loader
# --------------------------------------------------------------------------- #
def parse_veri_filename(fname: str) -> Tuple[int, int]:
    """Parse vehicleID and cameraID from filename like '0001_c001_00016450_0.jpg'."""
    stem = Path(fname).stem
    parts = stem.split("_")
    vehicle_id = int(parts[0])
    # Camera format: c001 -> 1
    cam_str = parts[1].replace("c", "") if len(parts) > 1 else "-1"
    camera_id = int(cam_str) if cam_str.isdigit() else -1
    return vehicle_id, camera_id


def load_veri776(
    root_dir: Optional[Union[str, Path]] = None,
    transform=None,
    legacy_resize: bool = False,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset]:
    """Load VeRi-776 query and test (gallery) sets."""
    veri_dir = resolve_dataset_path("VeRi", root_dir)
    if not veri_dir.exists():
        raise FileNotFoundError(f"VeRi-776 directory not found at: {veri_dir}")

    q_dir = veri_dir / "image_query"
    g_dir = veri_dir / "image_test"

    # Use name_query.txt and name_test.txt if present, else listdir
    name_query_file = veri_dir / "name_query.txt"
    name_test_file = veri_dir / "name_test.txt"

    def read_names(txt_file: Path, fallback_dir: Path) -> List[str]:
        if txt_file.exists():
            with txt_file.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return sorted([f.name for f in fallback_dir.glob("*.jpg")])

    query_names = read_names(name_query_file, q_dir)
    gallery_names = read_names(name_test_file, g_dir)

    query_samples = []
    for name in query_names:
        img_path = q_dir / name
        if not img_path.exists():
            continue
        vid, cam = parse_veri_filename(name)
        query_samples.append(ReIDSample(img_path=img_path, vehicle_id=vid, camera_id=cam, image_name=name))

    gallery_samples = []
    for name in gallery_names:
        img_path = g_dir / name
        if not img_path.exists():
            continue
        vid, cam = parse_veri_filename(name)
        gallery_samples.append(ReIDSample(img_path=img_path, vehicle_id=vid, camera_id=cam, image_name=name))

    tfm = transform or get_default_eval_transform(224, legacy_resize=legacy_resize)
    return SimpleReIDDataset(query_samples, transform=tfm), SimpleReIDDataset(gallery_samples, transform=tfm)


# --------------------------------------------------------------------------- #
# PKU-VehicleID (V1.0) Loader (Small: 800, Medium: 1600, Large: 2400)
# --------------------------------------------------------------------------- #
def resolve_vehicleid_dir(custom_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve PKU-VehicleID directory (`VehicleID_V1.0` or `VehicleID`)."""
    for candidate_name in ("VehicleID_V1.0", "VehicleID"):
        cand = resolve_dataset_path(candidate_name, custom_root)
        if cand.exists() and (cand / "image").is_dir() and (cand / "train_test_split").is_dir():
            return cand
    # Fallback default path
    cand_v1 = resolve_dataset_path("VehicleID_V1.0", custom_root)
    if cand_v1.exists():
        return cand_v1
    return resolve_dataset_path("VehicleID", custom_root)


def load_vehicleid_split(
    root_dir: Optional[Union[str, Path]] = None,
    split_name: str = "Small",
    seed: int = 42,
    transform=None,
    legacy_resize: bool = False,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset]:
    """Load PKU-VehicleID (V1.0) dataset with Small (800), Medium (1600), or Large (2400) split.

    Protocol: Exactly 1 image per vehicle identity is sampled for the gallery (seeded),
    and all remaining images of those identities are queries.
    """
    veh_dir = resolve_vehicleid_dir(root_dir)
    if not veh_dir.exists():
        raise FileNotFoundError(f"PKU-VehicleID directory not found at: {veh_dir}")

    img_dir = veh_dir / "image"
    split_dir = veh_dir / "train_test_split"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"VehicleID image directory not found: {img_dir}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"VehicleID train_test_split directory not found: {split_dir}")

    split_file_map = {
        "small": "test_list_800.txt",
        "800": "test_list_800.txt",
        "medium": "test_list_1600.txt",
        "1600": "test_list_1600.txt",
        "large": "test_list_2400.txt",
        "big": "test_list_2400.txt",
        "2400": "test_list_2400.txt",
        "3200": "test_list_3200.txt",
        "6000": "test_list_6000.txt",
        "13164": "test_list_13164.txt",
        "full": "test_list_13164.txt",
    }
    key = str(split_name).strip().lower()
    fname = split_file_map.get(key, "test_list_800.txt")
    split_file = split_dir / fname
    if not split_file.exists():
        raise FileNotFoundError(f"VehicleID split file {fname} not found in {split_dir}")

    samples_by_id: Dict[int, List[ReIDSample]] = {}
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_id = parts[0]
            veh_id = int(parts[1])
            img_name = f"{img_id}.jpg" if not img_id.endswith(".jpg") else img_id
            img_path = img_dir / img_name
            if img_path.exists():
                sample = ReIDSample(
                    img_path=img_path,
                    vehicle_id=veh_id,
                    camera_id=-1,
                    image_name=img_name,
                )
                samples_by_id.setdefault(veh_id, []).append(sample)

    rng = random.Random(seed)
    query_samples: List[ReIDSample] = []
    gallery_samples: List[ReIDSample] = []

    for vid, group in samples_by_id.items():
        if len(group) < 2:
            continue
        g_idx = rng.randrange(len(group))
        gallery_samples.append(group[g_idx])
        query_samples.extend(group[:g_idx])
        query_samples.extend(group[g_idx + 1:])

    tfm = transform or get_default_eval_transform(224, legacy_resize=legacy_resize)
    return (
        SimpleReIDDataset(query_samples, transform=tfm),
        SimpleReIDDataset(gallery_samples, transform=tfm),
    )


def load_vehicleid(
    root_dir: Optional[Union[str, Path]] = None,
    split_name: str = "Small",
    seed: int = 42,
    transform=None,
    legacy_resize: bool = False,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset]:
    """Load PKU-VehicleID split (default Small=800; supports Small/800, Medium/1600, Large/2400)."""
    return load_vehicleid_split(
        root_dir=root_dir,
        split_name=split_name,
        seed=seed,
        transform=transform,
        legacy_resize=legacy_resize,
    )


# --------------------------------------------------------------------------- #
# VRU Loader (Small: 1200, Medium: 2400, Large: 8000)
# --------------------------------------------------------------------------- #
def load_vru_split(
    root_dir: Optional[Union[str, Path]] = None,
    split_name: str = "Small",
    seed: int = 42,
    transform=None,
    legacy_resize: bool = False,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset]:
    """Load VRU dataset with Small (1200), Medium (2400), or Large/Big (8000) split.

    Protocol: Exactly 1 image per vehicle identity is sampled for the gallery,
    and all remaining images of those identities are queries.
    """
    vru_dir = resolve_dataset_path("VRU", root_dir)
    if not vru_dir.exists():
        raise FileNotFoundError(f"VRU directory not found at: {vru_dir}")

    pic_dir = vru_dir / "Pic"
    split_dir = vru_dir / "train_test_split"

    split_file_map = {
        "Small": "test_list_1200.txt",
        "Medium": "test_list_2400.txt",
        "Large": "test_list_8000.txt",
        "Big": "test_list_8000.txt",
    }
    fname = split_file_map.get(split_name, "test_list_1200.txt")
    split_file = split_dir / fname
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {fname} not found in {split_dir}")

    samples_by_id: Dict[int, List[ReIDSample]] = {}
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_id = parts[0]
            veh_id = int(parts[1])
            img_name = f"{img_id}.jpg" if not img_id.endswith(".jpg") else img_id
            img_path = pic_dir / img_name
            if img_path.exists():
                sample = ReIDSample(img_path=img_path, vehicle_id=veh_id, camera_id=-1, image_name=img_name)
                samples_by_id.setdefault(veh_id, []).append(sample)

    rng = random.Random(seed)
    query_samples: List[ReIDSample] = []
    gallery_samples: List[ReIDSample] = []

    for vid, group in samples_by_id.items():
        if len(group) < 2:
            continue
        g_idx = rng.randrange(len(group))
        gallery_samples.append(group[g_idx])
        query_samples.extend(group[:g_idx])
        query_samples.extend(group[g_idx + 1:])

    tfm = transform or get_default_eval_transform(224, legacy_resize=legacy_resize)
    return SimpleReIDDataset(query_samples, transform=tfm), SimpleReIDDataset(gallery_samples, transform=tfm)


# --------------------------------------------------------------------------- #
# VRAI (Aerial Vehicle Re-ID) Loader (Offline Validation Split)
# --------------------------------------------------------------------------- #
def load_vrai_val(
    root_dir: Optional[Union[str, Path]] = None,
    num_ids: int = 1000,
    seed: int = 42,
    transform=None,
    legacy_resize: bool = False,
) -> Tuple[SimpleReIDDataset, SimpleReIDDataset]:
    """Load an offline validation split from VRAI (Vehicle Re-ID in Aerial Imagery).

    Extracts `num_ids` vehicle identities with at least 2 images.
    Puts 1 image per identity into the gallery, and remaining images into queries.
    """
    vrai_dir = resolve_dataset_path("VRAI", root_dir)
    if not vrai_dir.exists():
        raise FileNotFoundError(f"VRAI directory not found at: {vrai_dir}")

    images_dir = vrai_dir / "images_train"
    annotation_path = vrai_dir / "train_annotation.pkl"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    import pickle
    with annotation_path.open("rb") as f:
        annotation = pickle.load(f)

    im_names = annotation.get("train_im_names", [])
    samples_by_id: Dict[int, List[ReIDSample]] = {}

    for name in im_names:
        parts = Path(name).stem.split("_")
        if not parts[0].isdigit():
            continue
        vid = int(parts[0])
        cam_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        img_path = images_dir / name
        if img_path.exists():
            sample = ReIDSample(img_path=img_path, vehicle_id=vid, camera_id=cam_id, image_name=name)
            samples_by_id.setdefault(vid, []).append(sample)

    valid_ids = sorted([vid for vid, items in samples_by_id.items() if len(items) >= 2])
    rng = random.Random(seed)
    rng.shuffle(valid_ids)
    selected_ids = sorted(valid_ids[:num_ids])

    query_samples: List[ReIDSample] = []
    gallery_samples: List[ReIDSample] = []

    for vid in selected_ids:
        group = samples_by_id[vid]
        g_idx = rng.randrange(len(group))
        gallery_samples.append(group[g_idx])
        query_samples.extend(group[:g_idx])
        query_samples.extend(group[g_idx + 1:])

    tfm = transform or get_default_eval_transform(224, legacy_resize=legacy_resize)
    return SimpleReIDDataset(query_samples, transform=tfm), SimpleReIDDataset(gallery_samples, transform=tfm)


def create_eval_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create evaluation DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
