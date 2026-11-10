"""Datasets and PK Batch Sampler for UAV-Anti-UAV video clip ReID."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class UAVReIDDataset(Dataset):
    """Training dataset for UAV-Anti-UAV before/after disappearance clip pairs."""

    def __init__(
        self,
        data_dir: str,
        query_json: str,
        gallery_json: str,
        transform=None,
        num_frames: int = 16,
        n_values=None,
    ):
        self.data_dir = data_dir
        with open(query_json, "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(gallery_json, "r", encoding="utf-8") as f:
            galleries = json.load(f)

        self.transform = transform
        self.num_frames = num_frames
        self.n_values = list(n_values) if n_values else [num_frames]

        g_dict = {(g["sequence_id"], g["event_index"]): g for g in galleries}
        self.valid_pairs = []
        for q in queries:
            key = (q["sequence_id"], q["event_index"])
            if key in g_dict:
                g = g_dict[key]
                if q.get("identity_id") is not None:
                    self.valid_pairs.append(
                        {
                            "identity_id": q["identity_id"],
                            "gallery_frames": g["frames"],
                            "gallery_dir": g["frame_dir"],
                            "query_frames": q["frames"],
                            "query_dir": q["frame_dir"],
                        }
                    )

        self.identities = sorted(list(set(p["identity_id"] for p in self.valid_pairs)))
        self.id_to_idx = {pid: i for i, pid in enumerate(self.identities)}
        self.num_identities = len(self.identities)

    def __len__(self) -> int:
        return len(self.valid_pairs) * len(self.n_values)

    def _load_clip(self, folder: str, frames: list, take_last: bool = False, n: int | None = None):
        n = n or self.num_frames
        frames = list(frames)
        if len(frames) > n:
            frames = frames[-n:] if take_last else frames[:n]
        elif len(frames) < n:
            if len(frames) == 0:
                return torch.zeros((n, 3, 224, 224))
            while len(frames) < n:
                frames.append(frames[-1])

        clip = []
        for fn in frames:
            path = os.path.join(self.data_dir, folder, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (256, 256), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)

    def __getitem__(self, idx: int):
        L = len(self.n_values)
        g, k = idx % L, idx // L
        n = self.n_values[g]
        pair = self.valid_pairs[k]

        before_clip = self._load_clip(
            pair["gallery_dir"], pair["gallery_frames"], take_last=True, n=n
        )
        after_clip = self._load_clip(
            pair["query_dir"], pair["query_frames"], take_last=False, n=n
        )

        pid = self.id_to_idx[pair["identity_id"]]
        return before_clip, after_clip, pid


class ReIDBatchSampler(Sampler):
    """PK Sampler ensuring P identities with K instances per batch and uniform N-choice."""

    def __init__(self, dataset: UAVReIDDataset, batch_size: int, num_instances: int = 4, n_values=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.n_values = (
            list(n_values)
            if n_values
            else list(getattr(dataset, "n_values", [dataset.num_frames]))
        )
        self.L = len(self.n_values)

        self.index_dic = defaultdict(list)
        for index, pair in enumerate(self.dataset.valid_pairs):
            pid = self.dataset.id_to_idx[pair["identity_id"]]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def _pk_batches(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = self.pids.copy()
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        batches = []
        for i in range(0, len(final_idxs), self.batch_size):
            batch = final_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
        return batches

    def __iter__(self):
        base = self._pk_batches()
        gs = (list(range(self.L)) * (len(base) // self.L + 1))[: len(base)]
        random.shuffle(gs)
        return iter([[k * self.L + g for k in b] for b, g in zip(base, gs)])

    def __len__(self) -> int:
        return self.length // self.batch_size


class EvalDataset(Dataset):
    """Evaluation / Calibration dataset for UAV-Anti-UAV."""

    def __init__(
        self,
        data_dir: str,
        query_json: str,
        gallery_json: str,
        transform=None,
        num_frames: int = 16,
        allowed_seq_ids=None,
    ):
        self.data_dir = data_dir
        with open(query_json, "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(gallery_json, "r", encoding="utf-8") as f:
            galleries = json.load(f)

        self.transform = transform
        self.num_frames = num_frames

        g_dict = {(g["sequence_id"], g["event_index"]): g for g in galleries}
        self.valid_pairs = []
        for q in queries:
            if allowed_seq_ids is not None and q["sequence_id"] not in allowed_seq_ids:
                continue
            key = (q["sequence_id"], q["event_index"])
            if key in g_dict:
                g = g_dict[key]
                if q.get("identity_id") is not None:
                    self.valid_pairs.append(
                        {
                            "identity_id": q["identity_id"],
                            "gallery_frames": g["frames"],
                            "gallery_dir": g["frame_dir"],
                            "query_frames": q["frames"],
                            "query_dir": q["frame_dir"],
                            "attributes": q.get("attributes", []),
                            "sequence_id": q["sequence_id"],
                            "event_index": q.get("event_index"),
                        }
                    )

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def _load_clip(self, folder: str, frames: list, take_last: bool = False):
        frames = list(frames)
        if len(frames) > self.num_frames:
            frames = frames[-self.num_frames :] if take_last else frames[: self.num_frames]
        elif len(frames) < self.num_frames:
            if len(frames) == 0:
                return torch.zeros((self.num_frames, 3, 224, 224))
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        clip = []
        for fn in frames:
            path = os.path.join(self.data_dir, folder, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (256, 256), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)

    def __getitem__(self, idx: int):
        pair = self.valid_pairs[idx]
        before_frames = pair["gallery_frames"]
        after_frames = pair["query_frames"]

        vis_path_b = (
            os.path.join(self.data_dir, pair["gallery_dir"], before_frames[len(before_frames) // 2])
            if before_frames
            else ""
        )
        vis_path_a = (
            os.path.join(self.data_dir, pair["query_dir"], after_frames[len(after_frames) // 2])
            if after_frames
            else ""
        )

        before_clip = self._load_clip(pair["gallery_dir"], before_frames, take_last=True)
        after_clip = self._load_clip(pair["query_dir"], after_frames, take_last=False)

        pid = pair["identity_id"]
        attrs = pair.get("attributes", [])
        seq_id = pair["sequence_id"]
        return before_clip, after_clip, pid, str(attrs), vis_path_a, vis_path_b, seq_id
