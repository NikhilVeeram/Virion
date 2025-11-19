"""Curiosity-driven GAN components for novelty estimation.

The module is intentionally lightweight so it can be trained on an Apple Silicon
M2 Max with <=32 GB of unified memory. The networks follow a standard DCGAN-like
layout but expose hooks for the reinforcement learning policy to request
novelty scores in real time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


class _ConvBlock(nn.Module):
    """A reusable Conv -> BN -> LeakyReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - pure delegation
        return self.net(x)


class CuriosityGenerator(nn.Module):
    """Small generator that synthesizes expected sensor observations."""

    def __init__(self, latent_dim: int = 64, base_channels: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class CuriosityDiscriminator(nn.Module):
    """Discriminator that scores how surprising an observation is."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            _ConvBlock(base_channels, base_channels * 2),
            _ConvBlock(base_channels * 2, base_channels * 4),
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


@dataclass
class CuriosityGANConfig:
    latent_dim: int = 64
    base_channels: int = 32
    lambda_gp: float = 10.0


class CuriosityGAN(nn.Module):
    """Wraps generator + discriminator with helper loss utilities."""

    def __init__(self, config: CuriosityGANConfig | None = None) -> None:
        super().__init__()
        self.config = config or CuriosityGANConfig()
        self.generator = CuriosityGenerator(
            latent_dim=self.config.latent_dim,
            base_channels=self.config.base_channels,
        )
        self.discriminator = CuriosityDiscriminator(
            base_channels=self.config.base_channels,
        )

    def sample(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(batch_size, self.config.latent_dim, device=device)
        return self.generator(z)

    def compute_losses(
        self,
        real_images: torch.Tensor,
        noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fake_images = self.generator(noise)
        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images.detach())
        d_loss = (fake_scores - real_scores).mean()

        gradients = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        d_loss = d_loss + self.config.lambda_gp * gradient_penalty

        g_loss = -self.discriminator(fake_images).mean()
        return {"d_loss": d_loss, "g_loss": g_loss, "real_scores": real_scores.mean(), "fake_scores": fake_scores.mean()}

    @torch.no_grad()
    def novelty_score(self, observation: torch.Tensor) -> torch.Tensor:
        """Higher scores correspond to more novel / surprising inputs."""

        observation = observation.to(next(self.parameters()).device)
        logits = self.discriminator(observation)
        return torch.sigmoid(logits)


__all__ = [
    "CuriosityGAN",
    "CuriosityGANConfig",
    "CuriosityGenerator",
    "CuriosityDiscriminator",
]
