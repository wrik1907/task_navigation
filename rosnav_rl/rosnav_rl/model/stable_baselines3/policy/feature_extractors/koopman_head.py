"""Auxiliary next-latent prediction heads for PPO."""

import torch
from torch import nn


class KoopmanLinearHead(nn.Module):
    """Predict z_{t+1} = A z_t + B a_t."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()

        if latent_dim <= 0 or action_dim <= 0:
            raise ValueError("latent_dim and action_dim must be positive")

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Trainable Koopman operators.
        self.A = nn.Parameter(torch.eye(latent_dim))
        self.B = nn.Parameter(torch.zeros(latent_dim, action_dim))

    def forward(
        self,
        z_t: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {z_t.shape[-1]}"
            )

        if action_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, "
                f"got {action_t.shape[-1]}"
            )

        return z_t @ self.A.T + action_t @ self.B.T


class MLPNextLatentHead(nn.Module):
    """Two-layer MLP control for next-latent prediction."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if latent_dim <= 0 or action_dim <= 0:
            raise ValueError("latent_dim and action_dim must be positive")

        hidden_dim = latent_dim if hidden_dim is None else hidden_dim

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {z_t.shape[-1]}"
            )

        if action_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, "
                f"got {action_t.shape[-1]}"
            )

        inputs = torch.cat((z_t, action_t), dim=-1)
        return self.network(inputs)