"""Tiny YOLO-like detector sized for onboard / laptop inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


def _conv_bn_silu(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class CSPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv1 = _conv_bn_silu(channels, hidden)
        self.conv2 = _conv_bn_silu(hidden, hidden)
        self.conv3 = _conv_bn_silu(hidden * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        return self.conv3(torch.cat([y1, y2], dim=1))


@dataclass
class TinyYOLOConfig:
    num_classes: int
    anchors: Tuple[Tuple[int, int], ...] = ((10, 13), (16, 30), (33, 23))
    channels: Tuple[int, int, int] = (32, 64, 128)


class TinyYOLO(nn.Module):
    """Simplified YOLO head with three detection scales."""

    def __init__(self, config: TinyYOLOConfig) -> None:
        super().__init__()
        self.config = config
        c1, c2, c3 = config.channels
        self.stem = nn.Sequential(
            _conv_bn_silu(3, c1, stride=2),
            CSPBlock(c1),
            _conv_bn_silu(c1, c2, stride=2),
            CSPBlock(c2),
            _conv_bn_silu(c2, c3, stride=2),
            CSPBlock(c3),
        )
        self.pred = nn.Conv2d(c3, len(config.anchors) * (config.num_classes + 5), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        out = self.pred(feats)
        bs, _, h, w = out.shape
        out = out.view(bs, len(self.config.anchors), self.config.num_classes + 5, h, w)
        return out.permute(0, 1, 3, 4, 2)

    def decode(self, raw: torch.Tensor, conf_threshold: float = 0.5) -> List[torch.Tensor]:
        batch_detections = []
        for sample in raw:
            sample = sample.reshape(-1, self.config.num_classes + 5)
            scores = sample[:, 4].sigmoid()
            keep = scores > conf_threshold
            boxes = sample[keep, :4]
            class_scores = sample[keep, 5:].sigmoid()
            classes = class_scores.argmax(dim=-1)
            detections = torch.cat([boxes, scores[keep].unsqueeze(1), classes.float().unsqueeze(1)], dim=1)
            batch_detections.append(detections)
        return batch_detections


__all__ = ["TinyYOLO", "TinyYOLOConfig"]
