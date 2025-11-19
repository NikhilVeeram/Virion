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
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class _ConvBlock(nn.Module):
    """A reusable Conv -> BN -> LeakyReLU block with optional spectral norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        apply_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=stride,
            padding=1,
            bias=False,
        )
        conv = spectral_norm(conv) if apply_spectral_norm else conv
        self.net = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - pure delegation
        return self.net(x)


class CuriosityGenerator(nn.Module):
    """Small generator that synthesizes expected sensor observations."""

    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 64,
        apply_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        convt = lambda in_c, out_c, stride, padding: nn.ConvTranspose2d(
            in_c, out_c, 4, stride, padding, bias=False
        )
        if apply_spectral_norm:
            convt = lambda in_c, out_c, stride, padding: spectral_norm(  # type: ignore[no-redef]
                nn.ConvTranspose2d(in_c, out_c, 4, stride, padding, bias=False)
            )

        self.net = nn.Sequential(
            convt(latent_dim, base_channels * 4, 1, 0),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            convt(base_channels * 4, base_channels * 2, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            convt(base_channels * 2, base_channels, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            convt(base_channels, 3, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer to reduce mode collapse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - lightweight math
        std = torch.sqrt(torch.var(x, dim=0, unbiased=False) + 1e-8)
        mean_std = std.mean().expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)


class CuriosityDiscriminator(nn.Module):
    """Discriminator that scores how surprising an observation is."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        apply_spectral_norm: bool = True,
        use_minibatch_std: bool = True,
    ) -> None:
        super().__init__()
        conv = lambda in_c, out_c, stride: nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
        if apply_spectral_norm:
            conv = lambda in_c, out_c, stride: spectral_norm(  # type: ignore[no-redef]
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
            )

        self.block1 = nn.Sequential(conv(in_channels, base_channels, 2), nn.LeakyReLU(0.2, inplace=True))
        self.block2 = _ConvBlock(base_channels, base_channels * 2, apply_spectral_norm=apply_spectral_norm)
        self.block3 = _ConvBlock(base_channels * 2, base_channels * 4, apply_spectral_norm=apply_spectral_norm)
        self.minibatch = MinibatchStdDev() if use_minibatch_std else None
        final_in = base_channels * 4 + (1 if use_minibatch_std else 0)
        self.final = conv(final_in, 1, 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        x = self.block1(x)
        features.append(x)
        x = self.block2(x)
        features.append(x)
        x = self.block3(x)
        features.append(x)
        if self.minibatch:
            x = self.minibatch(x)
        logits = self.final(x).view(-1)
        if return_features:
            return logits, features
        return logits


@dataclass
class CuriosityGANConfig:
    latent_dim: int = 64
    base_channels: int = 32
    use_spectral_norm: bool = True
    use_minibatch_std: bool = True
    lambda_gp: float = 10.0
    feature_matching_weight: float = 10.0


class CuriosityGAN(nn.Module):
    """Wraps generator + discriminator with helper loss utilities."""

    def __init__(self, config: CuriosityGANConfig | None = None) -> None:
        super().__init__()
        self.config = config or CuriosityGANConfig()
        self.generator = CuriosityGenerator(
            latent_dim=self.config.latent_dim,
            base_channels=self.config.base_channels,
            apply_spectral_norm=self.config.use_spectral_norm,
        )
        self.discriminator = CuriosityDiscriminator(
            base_channels=self.config.base_channels,
            apply_spectral_norm=self.config.use_spectral_norm,
            use_minibatch_std=self.config.use_minibatch_std,
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
        real_scores, real_features = self.discriminator(real_images, return_features=True)
        real_features = [feat.detach() for feat in real_features]
        fake_scores, _ = self.discriminator(fake_images.detach(), return_features=True)
        d_loss = (fake_scores - real_scores).mean()

        gradients = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        d_loss = d_loss + self.config.lambda_gp * gradient_penalty

        fake_logits_for_g, fake_features_for_g = self.discriminator(fake_images, return_features=True)
        feature_matching = sum(
            F.l1_loss(fake.mean(dim=[2, 3]), real.mean(dim=[2, 3]))
            for fake, real in zip(fake_features_for_g, real_features)
        )
        g_loss = -fake_logits_for_g.mean() + self.config.feature_matching_weight * feature_matching
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "real_scores": real_scores.mean(),
            "fake_scores": fake_scores.mean(),
        }

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
