"""VRAI dataset & PK Batch Sampler for GASNet training."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .vru import VRUDataset


@dataclass
class VRAITrainSample:
    img_path: Path
    image_name: str
    vehicle_id: int
    camera: str
    color: int
    vehicle_type: int


def _parse_vrai_train_id(image_name: str) -> int:
    return int(Path(image_name).stem.split("_", 1)[0])


def _parse_vrai_train_camera(image_name: str) -> str:
    parts = Path(image_name).stem.split("_")
    return parts[1] if len(parts) > 1 else ""


def _lookup_attr(labels: dict, vehicle_id: int, image_name: str, default: int = -1) -> int:
    if vehicle_id in labels:
        return int(labels[vehicle_id])
    if str(vehicle_id) in labels:
        return int(labels[str(vehicle_id)])
    if image_name in labels:
        return int(labels[image_name])
    return default


def read_vrai_train_samples(vrai_dir: Path) -> List[VRAITrainSample]:
    annotation_path = vrai_dir / "train_annotation.pkl"
    images_dir = vrai_dir / "images_train"
    with annotation_path.open("rb") as f:
        annotation = pickle.load(f)
    image_names = annotation["train_im_names"]
    samples: List[VRAITrainSample] = []
    for name in image_names:
        vehicle_id = _parse_vrai_train_id(name)
        samples.append(
            VRAITrainSample(
                img_path=images_dir / name,
                image_name=name,
                vehicle_id=vehicle_id,
                camera=_parse_vrai_train_camera(name),
                color=_lookup_attr(annotation.get("color_label", {}), vehicle_id, name),
                vehicle_type=_lookup_attr(annotation.get("type_label", {}), vehicle_id, name),
            )
        )
    return samples


class VRAITrainDataset(Dataset):
    def __init__(
        self,
        samples: List[VRAITrainSample],
        transform=None,
        label_map=None,
        max_retries: int = 3,
    ):
        self.samples = samples
        self.transform = transform
        self.label_map = label_map or {}
        self.max_retries = max_retries

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path: Path):
        last_err = None
        for _ in range(self.max_retries):
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except OSError as exc:
                last_err = exc
        raise last_err

    def __getitem__(self, idx):
        tried = 0
        cur_idx = idx
        while tried < min(10, len(self.samples)):
            s = self.samples[cur_idx]
            try:
                img = self._load_rgb(s.img_path)
                if self.transform is not None:
                    img = self.transform(img)
                label = self.label_map[s.vehicle_id]
                return (
                    img,
                    label,
                    s.vehicle_id,
                    s.image_name,
                    s.camera,
                    s.color,
                    s.vehicle_type,
                )
            except OSError:
                tried += 1
                cur_idx = (cur_idx + 1) % len(self.samples)
        raise RuntimeError(
            f"Cannot read image after retries, start_idx={idx}, path={self.samples[idx].img_path}"
        )


class PKBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(
        self,
        labels: List[int],
        p: int,
        k: int,
        cameras: List[str] | None = None,
        colors: List[int] | None = None,
        vehicle_types: List[int] | None = None,
        camera_balanced: bool = False,
        attribute_hard_negative: bool = False,
    ):
        self.p = p
        self.k = k
        self.cameras = cameras
        self.camera_balanced = camera_balanced and cameras is not None
        self.attribute_hard_negative = (
            attribute_hard_negative and colors is not None and vehicle_types is not None
        )
        label_to_indices: dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)

        self.label_to_indices = label_to_indices
        self.labels = [label for label, idxs in label_to_indices.items() if len(idxs) >= k]
        if len(self.labels) < p:
            raise ValueError(
                f"Need at least {p} identities with >= {k} samples, got {len(self.labels)}"
            )
        self.attr_to_labels: dict[tuple[int, int], List[int]] = {}
        if self.attribute_hard_negative:
            label_to_attr = {}
            for idx, label in enumerate(labels):
                if label in self.labels and label not in label_to_attr:
                    label_to_attr[label] = (int(colors[idx]), int(vehicle_types[idx]))
            for label, attr in label_to_attr.items():
                self.attr_to_labels.setdefault(attr, []).append(label)
            self.attr_buckets = [
                bucket for bucket in self.attr_to_labels.values() if len(bucket) >= 2
            ]
        else:
            self.attr_buckets = []

    def __len__(self) -> int:
        return len(self.labels) // self.p

    def _sample_indices_for_label(self, label: int) -> List[int]:
        indices = self.label_to_indices[label]
        if not self.camera_balanced:
            return random.sample(indices, self.k)

        by_camera: dict[str, List[int]] = {}
        for idx in indices:
            by_camera.setdefault(self.cameras[idx], []).append(idx)
        for camera_indices in by_camera.values():
            random.shuffle(camera_indices)
        camera_keys = list(by_camera.keys())
        random.shuffle(camera_keys)
        selected: List[int] = []
        while len(selected) < self.k and any(by_camera[camera] for camera in camera_keys):
            for camera in camera_keys:
                if len(selected) >= self.k:
                    break
                if by_camera[camera]:
                    selected.append(by_camera[camera].pop())
        if len(selected) < self.k:
            remaining = [idx for idx in indices if idx not in set(selected)]
            selected.extend(
                random.sample(remaining, min(len(remaining), self.k - len(selected)))
            )
        return selected[: self.k]

    def _make_batch_labels(self, remaining: set[int]) -> List[int]:
        if self.attribute_hard_negative and self.attr_buckets and random.random() < 0.8:
            bucket = [
                label for label in random.choice(self.attr_buckets) if label in remaining
            ]
            random.shuffle(bucket)
            batch_labels = bucket[: self.p]
            if len(batch_labels) < self.p:
                pool = list(remaining.difference(batch_labels))
                random.shuffle(pool)
                batch_labels.extend(pool[: self.p - len(batch_labels)])
            return batch_labels
        pool = list(remaining)
        random.shuffle(pool)
        return pool[: self.p]

    def __iter__(self):
        remaining = set(self.labels)
        while len(remaining) >= self.p:
            batch_labels = self._make_batch_labels(remaining)
            batch: List[int] = []
            for label in batch_labels:
                batch.extend(self._sample_indices_for_label(label))
                remaining.discard(label)
            yield batch


def make_pk_loader(
    samples: List,
    transform,
    label_map: dict,
    p: int,
    k: int,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    cameras: List[str] | None = None,
    colors: List[int] | None = None,
    vehicle_types: List[int] | None = None,
    camera_balanced: bool = False,
    attribute_hard_negative: bool = False,
    dataset_cls=VRUDataset,
):
    labels = [label_map[s.vehicle_id] for s in samples]
    batch_sampler = PKBatchSampler(
        labels,
        p=p,
        k=k,
        cameras=cameras,
        colors=colors,
        vehicle_types=vehicle_types,
        camera_balanced=camera_balanced,
        attribute_hard_negative=attribute_hard_negative,
    )
    if dataset_cls is VRAITrainDataset:
        ds = dataset_cls(
            samples=samples, transform=transform, label_map=label_map, max_retries=3
        )
    else:
        ds = dataset_cls(
            samples=samples,
            transform=transform,
            relabel=True,
            label_map=label_map,
            max_retries=3,
        )
    loader_kwargs = dict(pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        **loader_kwargs,
    )
