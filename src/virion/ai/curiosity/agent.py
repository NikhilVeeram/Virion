"""PPO-style reinforcement learner that consumes curiosity rewards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from .gan import CuriosityGAN


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 128
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class CuriosityPolicy(nn.Module):
    """Gaussian policy head shared with the PPO update."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _mlp(config.obs_dim, config.hidden_dim, config.hidden_dim)
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        value = self.value_head(features).squeeze(-1)
        return dist, value


class CuriosityAgent:
    """Couples PPO with GAN-based intrinsic rewards."""

    def __init__(self, policy: CuriosityPolicy, gan: CuriosityGAN, lr: float = 3e-4) -> None:
        self.policy = policy
        self.gan = gan
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.policy(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def compute_intrinsic_reward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.unsqueeze(0) if observation.dim() == 3 else observation
        score = self.gan.novelty_score(observation)
        return score.detach()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        log_probs_old = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        dist, values = self.policy(obs)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        ratio = (log_probs - log_probs_old).exp()
        clip_adv = torch.clamp(ratio, 1 - self.policy.config.clip_epsilon, 1 + self.policy.config.clip_epsilon) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()

        loss = (
            policy_loss
            + self.policy.config.value_coef * value_loss
            - self.policy.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }


__all__ = ["CuriosityAgent", "CuriosityPolicy", "PPOConfig"]
