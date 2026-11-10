"""Loss functions for GASNet (VRU/VRAI) and UAVReIDNet (UAV-Anti-UAV)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_hard_triplet_loss(
    feat: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    cameras=None,
    colors: torch.Tensor | None = None,
    vehicle_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-hard triplet loss with optional cross-camera positive & attribute hard-negative mining."""
    dist = torch.cdist(feat, feat, p=2)
    labels = labels.view(-1, 1)
    mask_pos = labels.eq(labels.t())
    mask_neg = ~mask_pos
    if cameras is not None:
        cam_list = list(cameras)
        cam_equal = torch.tensor(
            [[a == b for b in cam_list] for a in cam_list],
            dtype=torch.bool,
            device=feat.device,
        )
        cross_camera_pos = mask_pos & (~cam_equal)
        has_cross_camera_pos = cross_camera_pos.sum(dim=1) > 0
        mask_pos = torch.where(has_cross_camera_pos.view(-1, 1), cross_camera_pos, mask_pos)

    if colors is not None and vehicle_types is not None:
        colors = colors.to(feat.device).view(-1, 1)
        vehicle_types = vehicle_types.to(feat.device).view(-1, 1)
        same_attr_neg = mask_neg & colors.eq(colors.t()) & vehicle_types.eq(vehicle_types.t())
        has_same_attr_neg = same_attr_neg.sum(dim=1) > 0
        mask_neg = torch.where(has_same_attr_neg.view(-1, 1), same_attr_neg, mask_neg)

    dist_pos = dist.clone()
    dist_pos[~mask_pos] = -1.0
    dist_pos.fill_diagonal_(-1.0)
    hardest_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist.clone()
    dist_neg[~mask_neg] = 1e9
    hardest_neg, _ = dist_neg.min(dim=1)

    valid = hardest_pos > -0.5
    if valid.sum() == 0:
        return feat.sum() * 0.0
    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        log_probs = self.log_softmax(inputs)
        targets_oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_oh = (1.0 - self.epsilon) * targets_oh + self.epsilon / num_classes
        return (-targets_oh * log_probs).sum(1).mean()


class HardTripletLoss(nn.Module):
    """Online Hard Mining Triplet Loss on L2-normalized features."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.normalize(inputs, p=2, dim=1)
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            neg_vals = dist[i][mask[i] == 0]
            if len(neg_vals) > 0:
                dist_an.append(neg_vals.min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor([0.0], device=dist.device))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        if torch.all(dist_an == 0):
            return torch.tensor(0.0, requires_grad=True, device=dist.device)

        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class CenterLoss(nn.Module):
    """Center loss pulling features of the same identity toward a learnable center."""

    def __init__(self, num_classes: int = 1000, feat_dim: int = 2816, lr_center: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim))
        self.lr_center = lr_center

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        return dist.clamp(min=1e-12, max=1e12).sum() / batch_size


class TemporalConsistencyLoss(nn.Module):
    """Temporal distance ranking loss: adjacent frames must be more similar than endpoints."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        if len(temporal_features.shape) == 2 or temporal_features.size(1) < 3:
            return torch.tensor(
                0.0, device=temporal_features.device, dtype=temporal_features.dtype
            )
        x = temporal_features.float()
        sim_near = F.cosine_similarity(x[:, :-1, :], x[:, 1:, :], dim=-1).mean()
        sim_far = F.cosine_similarity(x[:, 0, :], x[:, -1, :], dim=-1).mean()
        return F.relu(sim_far - sim_near + self.margin)
