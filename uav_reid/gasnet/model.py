"""GASNet visual backbone and stage modules (RGA, BCGG, E-ConvNeXt, ResNet-IBN, Swin)."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .bcgg import BottleneckContextGroupGate, infer_gate_config


def _concat_relation(relation: torch.Tensor) -> torch.Tensor:
    return torch.cat([relation, relation.transpose(1, 2)], dim=1)


class RGASpatial(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_nodes = spatial_size[0] * spatial_size[1]
        self.spatial_reduction = spatial_reduction
        inter_channels = max(1, channels // channel_reduction)
        relation_features = max(1, self.num_nodes // spatial_reduction)
        self.theta = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.g = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.relation = nn.Sequential(
            nn.Conv1d(self.num_nodes * 2, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(relation_features + 1, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(relation_features, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h * w != self.num_nodes:
            raise ValueError(
                f"Expected spatial dimensions {self.spatial_size} (total={self.num_nodes} nodes), "
                f"but got {(h, w)} (total={h * w} nodes)"
            )
        theta = self.theta(x).view(b, -1, self.num_nodes).transpose(1, 2)
        phi = self.phi(x).view(b, -1, self.num_nodes)
        spatial_relation = torch.bmm(theta, phi)
        relation = _concat_relation(spatial_relation)
        relation = self.relation(relation)
        channel_pooled = self.g(x).flatten(2).mean(dim=1, keepdim=True)
        attn = self.attn(torch.cat([relation, channel_pooled], dim=1))
        attn = self.sigmoid(attn).view(b, 1, h, w)
        self.last_attention = attn
        return x * attn


class RGAChannel(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_nodes = spatial_size[0] * spatial_size[1]
        inter_spatial = max(1, self.num_nodes // spatial_reduction)
        relation_features = max(1, channels // channel_reduction)
        self.theta = nn.Sequential(
            nn.Conv1d(self.num_nodes, inter_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_spatial),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Conv1d(self.num_nodes, inter_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_spatial),
            nn.ReLU(inplace=True),
        )
        self.relation = nn.Sequential(
            nn.Conv1d(channels * 2, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
        )
        self.channel_embed = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(relation_features + 1, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(relation_features, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h * w != self.num_nodes:
            raise ValueError(
                f"Expected spatial dimensions {self.spatial_size} (total={self.num_nodes} nodes), "
                f"but got {(h, w)} (total={h * w} nodes)"
            )
        x_flat = x.view(b, c, self.num_nodes)
        x_perm = x_flat.permute(0, 2, 1)
        theta = self.theta(x_perm).permute(0, 2, 1)
        phi = self.phi(x_perm)
        channel_relation = torch.bmm(theta, phi)
        relation = _concat_relation(channel_relation)
        relation = self.relation(relation)
        spatial_pooled = self.channel_embed(x).mean(dim=(2, 3)).unsqueeze(1)
        attn = self.attn(torch.cat([relation, spatial_pooled], dim=1))
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return x * attn


class RGABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.rga_s = RGASpatial(
            channels=channels,
            spatial_size=spatial_size,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction,
        )
        self.rga_c = RGAChannel(
            channels=channels,
            spatial_size=spatial_size,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction,
        )

    @property
    def last_attention(self) -> torch.Tensor | None:
        return self.rga_s.last_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rga_s(x)
        x = self.rga_c(x)
        return x


def _make_gate_block(
    kind: str,
    channels: int,
    spatial_size: tuple[int, int],
    hidden: int = 16,
    groups: int = 4,
    kernel_size: int = 7,
    channel_kernel: int = 5,
    inject_value: bool = False,
) -> nn.Module:
    if kind == "rga":
        return RGABlock(channels, spatial_size=spatial_size)
    if kind == "bcgg":
        return BottleneckContextGroupGate(
            channels=channels,
            hidden=hidden,
            groups=groups,
            kernel_size=kernel_size,
            channel_kernel=channel_kernel,
            inject_value=inject_value,
        )
    raise ValueError(f"Unsupported gate kind: {kind!r} (expected 'bcgg' or 'rga')")


def infer_backbone_name(state_dict: dict, default: str = "resnet50") -> str:
    keys = state_dict.keys()
    if any(k.startswith("convnext_backbone.") for k in keys):
        return "dinov3_convnext"
    if any(k.startswith("swin_backbone.") for k in keys):
        return "swin_t"
    if any(".bn1.in_norm." in k for k in keys):
        return "resnet50_ibn"
    if any(k.startswith("layer1.") for k in keys):
        return "resnet50"
    return default


def infer_econvnext(state_dict: dict) -> bool:
    """Detect whether a DINOv3 ConvNeXt checkpoint uses the E-ConvNeXt variant."""
    keys = state_dict.keys()
    return any("convnext_backbone.stage1.blocks." in k or "convnext_backbone.stem.conv1." in k for k in keys)


class Lite3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gate(x)


class OSBlockFS(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        mid_ch = out_ch

        self.stream1 = nn.Sequential(Lite3x3(in_ch, mid_ch))
        self.stream2 = nn.Sequential(Lite3x3(in_ch, mid_ch), Lite3x3(mid_ch, mid_ch))
        self.stream3 = nn.Sequential(
            Lite3x3(in_ch, mid_ch), Lite3x3(mid_ch, mid_ch), Lite3x3(mid_ch, mid_ch)
        )
        self.stream4 = nn.Sequential(
            Lite3x3(in_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch),
        )

        self.gate = ChannelGate(mid_ch, reduction)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)

        out = (
            self.gate(s1) * s1
            + self.gate(s2) * s2
            + self.gate(s3) * s3
            + self.gate(s4) * s4
        )

        out = out + self.shortcut(x)
        return self.relu(out)


class BNNeck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(
            1.0 / self.p
        )


class AttentionLocalBranch(nn.Module):
    """Attention-guided local feature learning for UAV Vehicle Re-ID."""

    def __init__(self, in_channels: int, num_heads: int = 4, reduction: int = 16):
        super().__init__()
        self.num_heads = num_heads
        mid = max(1, in_channels // reduction)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_heads, kernel_size=1),
            nn.Sigmoid(),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, feature_map: torch.Tensor, pool_fn: nn.Module) -> list[torch.Tensor]:
        attention = self.attention(feature_map)
        self.last_attention = attention

        local_features = []
        for i in range(self.num_heads):
            att = attention[:, i : i + 1, :, :]
            weighted = feature_map * (1 + att)
            pooled = pool_fn(weighted).flatten(1)
            local_features.append(pooled)
        return local_features

    def diversity_loss(self) -> torch.Tensor:
        if self.last_attention is None:
            return torch.tensor(0.0)
        att = self.last_attention
        B, K, _, _ = att.shape
        flat = att.view(B, K, -1)
        flat = F.normalize(flat, dim=2)
        gram = torch.bmm(flat, flat.transpose(1, 2))
        eye = torch.eye(K, device=gram.device).unsqueeze(0)
        return ((gram - eye) ** 2).mean()


class IBN(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.5):
        super().__init__()
        split = int(channels * ratio)
        self.split = split
        self.in_norm = nn.GroupNorm(split, split, affine=True)
        self.bn_norm = nn.BatchNorm2d(channels - split)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.split, x.size(1) - self.split], dim=1)
        return torch.cat([self.in_norm(x1), self.bn_norm(x2)], dim=1)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_ibn: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if use_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetIBN(nn.Module):
    def __init__(self, layers: tuple[int, int, int, int] = (3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], use_ibn=True)
        self.layer2 = self._make_layer(128, layers[1], stride=2, use_ibn=True)
        self.layer3 = self._make_layer(256, layers[2], stride=2, use_ibn=True)
        self.layer4 = self._make_layer(512, layers[3], stride=2, use_ibn=False)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _make_layer(
        self, planes: int, blocks: int, stride: int = 1, use_ibn: bool = True
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckIBN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * BottleneckIBN.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BottleneckIBN.expansion),
            )

        layers = [
            BottleneckIBN(
                self.inplanes, planes, stride=stride, downsample=downsample, use_ibn=use_ibn
            )
        ]
        self.inplanes = planes * BottleneckIBN.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckIBN(self.inplanes, planes, use_ibn=use_ibn))
        return nn.Sequential(*layers)


def _load_matching_resnet50_weights(model: nn.Module) -> None:
    weights = models.ResNet50_Weights.DEFAULT
    state = models.resnet50(weights=weights).state_dict()
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape
    }
    missing = len(model_state) - len(filtered)
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} matching ResNet50 weights into ResNet50-IBN; skipped {missing} tensors")


class SwinBackbone(nn.Module):
    SWIN_CHANNELS = (96, 192, 384, 768)
    TARGET_CHANNELS = (256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import Swin_T_Weights, swin_t

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        base = swin_t(weights=weights)
        if pretrained:
            print("Loaded Swin-T pretrained weights")

        self.stage1 = base.features[0:2]
        self.stage2 = base.features[2:4]
        self.stage3 = base.features[4:6]
        self.stage4 = base.features[6:8]
        self.norm = base.norm

        adapters = []
        for src_ch, tgt_ch in zip(self.SWIN_CHANNELS, self.TARGET_CHANNELS):
            adapters.append(
                nn.Sequential(
                    nn.Conv2d(src_ch, tgt_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(tgt_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.adapt1, self.adapt2, self.adapt3, self.adapt4 = adapters

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        feat1 = self.adapt1(x.permute(0, 3, 1, 2).contiguous())

        x = self.stage2(x)
        feat2 = self.adapt2(x.permute(0, 3, 1, 2).contiguous())

        x = self.stage3(x)
        feat3 = self.adapt3(x.permute(0, 3, 1, 2).contiguous())

        x = self.stage4(x)
        x = self.norm(x)
        feat4 = self.adapt4(x.permute(0, 3, 1, 2).contiguous())

        return feat1, feat2, feat3, feat4


# ============================================================================
# E-ConvNeXt (arXiv 2508.20955)
# ============================================================================
SIGMA_B = (0.173, 0.169, 0.127, 0.155)


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class EffectiveSELayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Hardsigmoid()
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean(dim=(2, 3), keepdim=True)
        return x * self.act(self.fc(x_se))


class EConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, use_ese: bool = True, sigma_b: float = 0.17):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pointwise_conv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.activation_fn = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.sigma_b = sigma_b
        if use_ese:
            self.norm2 = nn.BatchNorm2d(dim)
            self.ese = EffectiveSELayer(dim)
            self.gamma = None
        else:
            self.norm2 = None
            self.ese = None
            self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        if self.norm2 is not None:
            x = self.ese(self.norm2(x))
        else:
            x = x * self.gamma.view(1, -1, 1, 1)
        return residual + x


class ConvBNGELU(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class EConvNeXtStem(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = ConvBNGELU(3, mid, 2, stride=2, padding=0)
        self.conv2 = ConvBNGELU(mid, mid, 3, stride=1, padding=1)
        self.conv3 = ConvBNGELU(mid, out_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(x)))


class CSPStage(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        num_blocks: int,
        stride: int = 2,
        use_ese: bool = True,
        sigma_b: float = 0.17,
    ):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_mid = ch_mid
        self.block_dim = ch_mid // 2
        self.down = ConvBNGELU(ch_in, ch_mid, 2, stride=2) if stride == 2 else nn.Identity()
        self.conv1 = ConvBNGELU(ch_mid, ch_mid // 2, 1)
        self.conv2 = ConvBNGELU(ch_mid, ch_mid // 2, 1)
        self.blocks = nn.Sequential(
            *[
                EConvNeXtBlock(ch_mid // 2, use_ese=use_ese, sigma_b=sigma_b)
                for _ in range(num_blocks)
            ]
        )
        self.attn = EffectiveSELayer(ch_mid) if use_ese else None
        self.conv3 = ConvBNGELU(ch_mid, ch_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        return self.conv3(y)


class EConvNeXtStage(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_blocks: int, use_ese: bool = False):
        super().__init__()
        self.downsample_layers = nn.Sequential(
            LayerNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=2, stride=2),
        )
        self.layers = nn.Sequential(
            *[EConvNeXtBlock(ch_out, use_ese=use_ese) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.downsample_layers(x))


class DINOv3ConvNeXtBackbone(nn.Module):
    CNN_DIMS = (96, 192, 384, 768)
    CNN_DEPTHS = (3, 3, 27, 3)

    @staticmethod
    def _copy_block_(dst: EConvNeXtBlock, src, idx=None):
        dw = src.depthwise_conv.weight.data
        lnw, lnb = src.layer_norm.weight.data, src.layer_norm.bias.data
        p1w, p1b = src.pointwise_conv1.weight.data, src.pointwise_conv1.bias.data
        p2w, p2b = src.pointwise_conv2.weight.data, src.pointwise_conv2.bias.data
        gam = src.gamma.data

        if idx is None:
            dst.depthwise_conv.weight.data.copy_(dw)
            dst.depthwise_conv.bias.data.copy_(src.depthwise_conv.bias.data)
            dst.norm.weight.data.copy_(lnw)
            dst.norm.bias.data.copy_(lnb)
            dst.pointwise_conv1.weight.data.copy_(p1w[:, :, None, None])
            dst.pointwise_conv1.bias.data.copy_(p1b)
            dst.pointwise_conv2.weight.data.copy_(p2w[:, :, None, None])
            dst.pointwise_conv2.bias.data.copy_(p2b)
            if dst.gamma is not None:
                dst.gamma.data.copy_(gam)
            else:
                dst.norm2.weight.data.copy_(2.0 * gam * dst.sigma_b)
                dst.norm2.bias.data.zero_()
            return

        k = int(idx.numel())
        dst.depthwise_conv.weight.data.copy_(dw.index_select(0, idx))
        dst.depthwise_conv.bias.data.copy_(src.depthwise_conv.bias.data.index_select(0, idx))
        dst.norm.weight.data.copy_(lnw.index_select(0, idx))
        dst.norm.bias.data.copy_(lnb.index_select(0, idx))
        imp = p1w.norm(dim=1) * p2w.norm(dim=0)
        hid = torch.topk(imp, 4 * k).indices
        dst.pointwise_conv1.weight.data.copy_(
            p1w.index_select(0, hid).index_select(1, idx)[:, :, None, None]
        )
        dst.pointwise_conv1.bias.data.copy_(p1b.index_select(0, hid))
        dst.pointwise_conv2.weight.data.copy_(
            p2w.index_select(0, idx).index_select(1, hid)[:, :, None, None]
        )
        dst.pointwise_conv2.bias.data.copy_(p2b.index_select(0, idx))
        if dst.gamma is not None:
            dst.gamma.data.copy_(gam.index_select(0, idx))
        else:
            dst.norm2.weight.data.copy_(2.0 * gam.index_select(0, idx) * dst.sigma_b)
            dst.norm2.bias.data.zero_()

    @staticmethod
    def _stage_channel_index_(src_stage, keep: int) -> torch.Tensor:
        gam_mean = torch.stack([b.gamma.data.abs() for b in src_stage.layers]).mean(dim=0)
        return torch.topk(gam_mean, keep).indices

    def __init__(self, pretrained: bool = True, econvnext: bool = False):
        super().__init__()
        self.econvnext = bool(econvnext)

        if self.econvnext and not pretrained:
            # Build E-ConvNeXt directly without HuggingFace network call when pretrained=False
            self.stem = EConvNeXtStem(out_channels=64)
            self.stage1 = CSPStage(
                64, 96, num_blocks=3, stride=2, use_ese=True, sigma_b=SIGMA_B[0]
            )
            self.stage2 = CSPStage(
                96, 192, num_blocks=3, stride=2, use_ese=True, sigma_b=SIGMA_B[1]
            )
            self.stage3 = EConvNeXtStage(192, 384, num_blocks=27, use_ese=False)
            self.stage4 = EConvNeXtStage(384, 768, num_blocks=3, use_ese=False)
            return

        import transformers
        from huggingface_hub import get_token
        from packaging import version
        from transformers import AutoConfig, AutoModel

        model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"

        kwargs = {}
        hf_token = get_token() or os.environ.get("HF_TOKEN")
        if version.parse(transformers.__version__) >= version.parse("4.32.0"):
            kwargs["token"] = hf_token
        else:
            kwargs["use_auth_token"] = hf_token

        if pretrained:
            print(f"Loading pretrained ConvNeXt from HuggingFace: {model_name}...")
            full_model = AutoModel.from_pretrained(model_name, **kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name, **kwargs)
            full_model = AutoModel.from_config(config)

        if not self.econvnext:
            if hasattr(full_model, "embeddings"):
                self.stem = full_model.embeddings
                self.stage1 = full_model.encoder.stages[0]
                self.stage2 = full_model.encoder.stages[1]
                self.stage3 = full_model.encoder.stages[2]
                self.stage4 = full_model.encoder.stages[3]
            else:
                self.stem = nn.Identity()
                self.stage1 = full_model.model.stages[0]
                self.stage2 = full_model.model.stages[1]
                self.stage3 = full_model.model.stages[2]
                self.stage4 = full_model.model.stages[3]
            del full_model
            print("  DINOv3 ConvNeXt-Small backbone ready")
            return

        self.stem = EConvNeXtStem(out_channels=64)
        self.stage1 = CSPStage(
            64, 96, num_blocks=3, stride=2, use_ese=True, sigma_b=SIGMA_B[0]
        )
        self.stage2 = CSPStage(
            96, 192, num_blocks=3, stride=2, use_ese=True, sigma_b=SIGMA_B[1]
        )
        self.stage3 = EConvNeXtStage(192, 384, num_blocks=27, use_ese=False)
        self.stage4 = EConvNeXtStage(384, 768, num_blocks=3, use_ese=False)

        n_full, n_sliced = self._load_dinov3_weights_(full_model)
        del full_model
        print(
            f"  E-ConvNeXt backbone ready: {n_full} block copy nguyen, "
            f"{n_sliced} block cat bot kenh"
        )

    def _load_dinov3_weights_(self, full_model):
        if not hasattr(full_model, "model"):
            raise RuntimeError("Chi ho tro layout DINOv3 (full_model.model.stages)")
        src = full_model.model.stages
        n_full = n_sliced = 0

        for si, dst_stage in ((2, self.stage3), (3, self.stage4)):
            s = src[si]
            dst_stage.downsample_layers[0].weight.data.copy_(s.downsample_layers[0].weight.data)
            dst_stage.downsample_layers[0].bias.data.copy_(s.downsample_layers[0].bias.data)
            dst_stage.downsample_layers[1].weight.data.copy_(s.downsample_layers[1].weight.data)
            dst_stage.downsample_layers[1].bias.data.copy_(s.downsample_layers[1].bias.data)
            for i in range(len(dst_stage.layers)):
                self._copy_block_(dst_stage.layers[i], s.layers[i], idx=None)
                n_full += 1

        s0 = src[0]
        idx0 = self._stage_channel_index_(s0, self.stage1.block_dim)
        for i in range(len(self.stage1.blocks)):
            self._copy_block_(self.stage1.blocks[i], s0.layers[i], idx=idx0)
            n_sliced += 1

        s1 = src[1]
        idx1 = self._stage_channel_index_(s1, self.stage2.block_dim)
        for i in range(len(self.stage2.blocks)):
            self._copy_block_(self.stage2.blocks[i], s1.layers[i], idx=idx1)
            n_sliced += 1
        dst_down = self.stage2.down.conv
        dst_down.weight.data.copy_(s1.downsample_layers[1].weight.data[: self.stage2.ch_mid])
        src_bias = s1.downsample_layers[1].bias
        if src_bias is not None:
            dst_down.bias.data.copy_(src_bias.data[: self.stage2.ch_mid])
        else:
            dst_down.bias.data.zero_()

        return n_full, n_sliced


class GASNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool = True,
        use_part_branch: bool = False,
        num_parts: int = 4,
        use_gem: bool = False,
        gem_p: float = 3.0,
        backbone: str = "resnet50",
        use_attention_local: bool = False,
        num_attention_heads: int = 4,
        econvnext: bool = False,
        gate: str = "bcgg",
        gate_hidden: int = 16,
        gate_groups: int = 4,
        gate_kernel: int = 7,
        gate_channel_kernel: int = 5,
        gate_inject_value: bool = False,
    ):
        super().__init__()
        if num_parts < 1:
            raise ValueError("num_parts must be >= 1")
        if gate not in ("bcgg", "rga"):
            raise ValueError(f"Unsupported gate kind: {gate!r} (expected 'bcgg' or 'rga')")
        self.use_part_branch = use_part_branch
        self.use_attention_local = use_attention_local
        self.num_parts = num_parts
        self.backbone_type = backbone
        self.gate_kind = gate
        self._is_convnext = backbone == "dinov3_convnext"

        if backbone == "swin_t":
            self.swin_backbone = SwinBackbone(pretrained=use_pretrained)
        elif self._is_convnext:
            self.convnext_backbone = DINOv3ConvNeXtBackbone(
                pretrained=use_pretrained, econvnext=econvnext
            )
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            base = models.resnet50(weights=weights)
        elif backbone == "resnet50_ibn":
            base = ResNetIBN()
            if use_pretrained:
                _load_matching_resnet50_weights(base)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if self._is_convnext:
            c1, c2, c3, c4 = 96, 192, 384, 768
            dim_fs = 192
        else:
            c1, c2, c3, c4 = 256, 512, 1024, 2048
            dim_fs = 512

        if gate == "bcgg":
            if gate_groups <= 0 or gate_hidden <= 0:
                raise ValueError("gate_groups and gate_hidden must be positive")
            for stage, stage_channels in enumerate((c1, c2, c3, c4), start=1):
                if stage_channels % gate_groups:
                    raise ValueError(
                        f"gate_groups={gate_groups} must divide stage {stage} "
                        f"channels ({stage_channels})"
                    )

        gate_kwargs = dict(
            hidden=gate_hidden,
            groups=gate_groups,
            kernel_size=gate_kernel,
            channel_kernel=gate_channel_kernel,
            inject_value=gate_inject_value,
        )

        if backbone not in ("swin_t",) and not self._is_convnext:
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.ga1 = _make_gate_block(gate, c1, (56, 56), **gate_kwargs)
            self.ga2 = _make_gate_block(gate, c2, (28, 28), **gate_kwargs)
            self.ga3 = _make_gate_block(gate, c3, (14, 14), **gate_kwargs)
            self.ga4 = _make_gate_block(gate, c4, (7, 7), **gate_kwargs)

        if self._is_convnext:
            self.ga1 = _make_gate_block(gate, c1, (56, 56), **gate_kwargs)
            self.ga2 = _make_gate_block(gate, c2, (28, 28), **gate_kwargs)
            self.ga3 = _make_gate_block(gate, c3, (14, 14), **gate_kwargs)
            self.ga4 = _make_gate_block(gate, c4, (7, 7), **gate_kwargs)

        self.fs1 = OSBlockFS(c3, dim_fs)
        self.fs2 = OSBlockFS(dim_fs, dim_fs)

        self.gap = GeMPool(p=gem_p) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.bnneck_global = BNNeck(c4)
        self.bnneck_fs = BNNeck(dim_fs)
        if self.use_part_branch:
            self.part_reduce = nn.Sequential(
                nn.Linear(c4 * num_parts, dim_fs, bias=False),
                nn.BatchNorm1d(dim_fs),
                nn.ReLU(inplace=True),
            )
            self.bnneck_part = BNNeck(dim_fs)
            self.classifier_part = nn.Linear(dim_fs, num_classes, bias=False)
        if self.use_attention_local:
            self.attn_local = AttentionLocalBranch(
                in_channels=c4,
                num_heads=num_attention_heads,
            )
            self.attn_head_reduce = nn.ModuleList(
                [nn.Linear(c4, 256, bias=False) for _ in range(num_attention_heads)]
            )
            self.attn_local_reduce = nn.Sequential(
                nn.BatchNorm1d(256 * num_attention_heads),
                nn.ReLU(inplace=True),
                nn.Linear(256 * num_attention_heads, dim_fs, bias=False),
                nn.BatchNorm1d(dim_fs),
                nn.ReLU(inplace=True),
            )
            self.bnneck_attn_local = BNNeck(dim_fs)
            self.classifier_attn_local = nn.Linear(dim_fs, num_classes, bias=False)

        self.classifier_global = nn.Linear(c4, num_classes, bias=False)
        self.classifier_fs = nn.Linear(dim_fs, num_classes, bias=False)

    def _part_pool(self, x: torch.Tensor) -> torch.Tensor:
        stripes = torch.chunk(x, self.num_parts, dim=2)
        pooled = [self.gap(stripe).flatten(1) for stripe in stripes]
        return self.part_reduce(torch.cat(pooled, dim=1))

    def set_attention_capture(self, enabled: bool) -> None:
        for name in ("ga1", "ga2", "ga3", "ga4"):
            block = getattr(self, name, None)
            if block is None:
                continue
            if hasattr(block, "capture_attention"):
                block.capture_attention = bool(enabled)
            if not enabled:
                if hasattr(block, "rga_s"):
                    block.rga_s.last_attention = None
                elif hasattr(block, "last_attention"):
                    block.last_attention = None

    @torch.no_grad()
    def to_deploy(self) -> GASNet:
        if self.training:
            raise ValueError("Call eval() before to_deploy(); keep a training checkpoint")
        self.set_attention_capture(False)
        module = copy.deepcopy(self)
        module.eval()
        for name in ("ga1", "ga2", "ga3", "ga4"):
            block = getattr(module, name, None)
            if block is not None and hasattr(block, "to_deploy"):
                setattr(module, name, block.to_deploy())
        return module.eval()

    def attention_regularization_loss(
        self,
        target_mean: float = 0.55,
        std_margin: float = 0.08,
    ) -> torch.Tensor:
        if self.backbone_type == "swin_t":
            return next(self.parameters()).sum() * 0.0
        losses = []
        for block in (self.ga1, self.ga2, self.ga3, self.ga4):
            attn = getattr(block, "last_attention", None)
            if attn is None:
                continue
            flat = attn.flatten(1)
            mean_loss = (flat.mean(dim=1) - target_mean).pow(2)
            std_loss = F.relu(std_margin - flat.std(dim=1, unbiased=False)).pow(2)
            losses.append((mean_loss + std_loss).mean())
        if not losses:
            return next(self.parameters()).sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, x: torch.Tensor):
        if self.backbone_type == "swin_t":
            feat1, feat2, feat3, feat4 = self.swin_backbone(x)
            fs = self.fs1(feat3)
            fs = self.fs2(fs)
            x = feat4
        elif self._is_convnext:
            x = self.convnext_backbone.stem(x)
            x = self.convnext_backbone.stage1(x)
            x = self.ga1(x)
            x = self.convnext_backbone.stage2(x)
            x = self.ga2(x)
            x = self.convnext_backbone.stage3(x)
            fs = self.fs1(x)
            fs = self.fs2(fs)
            x = self.ga3(x)
            x = self.convnext_backbone.stage4(x)
            x = self.ga4(x)
        else:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.ga1(x)
            x = self.layer2(x)
            x = self.ga2(x)
            x = self.layer3(x)
            fs = self.fs1(x)
            fs = self.fs2(fs)
            x = self.ga3(x)
            x = self.layer4(x)
            x = self.ga4(x)

        global_feat = self.gap(x).flatten(1)
        fs_feat = self.gap(fs).flatten(1)
        bn_global = self.bnneck_global(global_feat)
        bn_fs = self.bnneck_fs(fs_feat)

        logits_global = self.classifier_global(bn_global)
        logits_fs = self.classifier_fs(bn_fs)

        if self.use_attention_local:
            local_feats = self.attn_local(x, self.gap)
            reduced = [
                self.attn_head_reduce[i](feat) for i, feat in enumerate(local_feats)
            ]
            attn_local_feat = self.attn_local_reduce(torch.cat(reduced, dim=1))
            bn_attn_local = self.bnneck_attn_local(attn_local_feat)
            logits_attn_local = self.classifier_attn_local(bn_attn_local)

        if self.use_part_branch and not self.use_attention_local:
            part_feat = self._part_pool(x)
            bn_part = self.bnneck_part(part_feat)
            logits_part = self.classifier_part(bn_part)

        if self.training:
            if self.use_attention_local:
                return (global_feat, fs_feat, attn_local_feat), (
                    logits_global,
                    logits_fs,
                    logits_attn_local,
                )
            if self.use_part_branch:
                return (global_feat, fs_feat, part_feat), (
                    logits_global,
                    logits_fs,
                    logits_part,
                )
            return (global_feat, fs_feat), (logits_global, logits_fs)

        if getattr(self, "return_raw_features_eval", False):
            if self.use_attention_local:
                return global_feat, fs_feat, attn_local_feat
            if self.use_part_branch:
                return global_feat, fs_feat, part_feat
            return global_feat, fs_feat

        if self.use_attention_local:
            return bn_global, bn_fs, bn_attn_local
        if self.use_part_branch:
            return bn_global, bn_fs, bn_part
        return bn_global, bn_fs
