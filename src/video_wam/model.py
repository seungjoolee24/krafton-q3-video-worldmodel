from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    grid_channels: int = 48
    hidden_channels: int = 64
    latent_action_dim: int = 1
    action_embedding: int = 16
    context: int = 32

    def __post_init__(self):
        if self.grid_channels != 48 or self.hidden_channels % 8:
            raise ValueError("Use 48 grid channels and a hidden width divisible by 8")
        if self.hidden_channels < 8 or self.latent_action_dim < 1 or self.action_embedding < 1:
            raise ValueError("Model dimensions must be positive")
        if not 1 <= self.context <= 32:
            raise ValueError("Context must be in [1,32]")


class ConvGRUCell(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.gates = nn.Conv2d(inputs + hidden, 2 * hidden, 3, padding=1)
        self.candidate = nn.Conv2d(inputs + hidden, hidden, 3, padding=1)

    def forward(self, observation, hidden):
        reset, update = torch.sigmoid(self.gates(torch.cat([observation, hidden], 1))).chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat([observation, reset * hidden], 1)))
        return (1.0 - update) * hidden + update * candidate


class GaussianHead(nn.Module):
    def __init__(self, inputs: int, width: int, dimensions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.output = nn.Linear(width, dimensions * 2)

    def forward(self, inputs):
        mean, log_std = self.output(self.features(inputs)).chunk(2, -1)
        return mean, log_std.clamp(-4.0, 1.0)


class VideoWorldModel(nn.Module):
    """A vision-only state model with a low-dimensional variational transition code.

    posterior: sees the true next observation, only during training/offline analysis.
    prior: sees only the available history, used for autonomous future prediction.
    The transition code is NOT calibrated to physical force; no action labels enter.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        c, h, d, e = (config.grid_channels, config.hidden_channels,
                       config.latent_action_dim, config.action_embedding)
        self.initial = nn.Conv2d(c, h, 3, padding=1)
        self.memory = ConvGRUCell(c, h)
        self.prior = GaussianHead(c + h, h, d)
        self.posterior = GaussianHead(2 * c + h, h, d)
        self.action_embed = nn.Sequential(nn.Linear(d, e), nn.SiLU(), nn.Linear(e, e))
        self.transition = nn.Sequential(
            nn.Conv2d(c + h + e, h, 3, padding=1), nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, c, 3, padding=1),
        )
        # Initial deterministic prediction exactly copies the last latent grid.
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def config_dict(self):
        return asdict(self.config)

    def encode_context(self, grids):
        if grids.ndim != 5 or grids.shape[1] < 1:
            raise ValueError("Expected (B,T,C,H,W), T >= 1")
        hidden = torch.tanh(self.initial(grids[:, 0]))
        for index in range(1, grids.shape[1]):
            hidden = self.memory(grids[:, index], hidden)
        return grids[:, -1], hidden

    def infer_prior(self, grid, hidden):
        return self.prior(torch.cat([grid, hidden], 1))

    def infer_posterior(self, grid, hidden, next_grid):
        return self.posterior(torch.cat([grid, hidden, next_grid - grid], 1))

    def predict_grid(self, grid, hidden, latent_action):
        embedding = self.action_embed(latent_action)[:, :, None, None]
        embedding = embedding.expand(-1, -1, grid.shape[-2], grid.shape[-1])
        delta = self.transition(torch.cat([grid, hidden, embedding], 1))
        return (grid + delta).clamp(-1.0, 1.0)

    def observe_grid(self, hidden, grid):
        # No past action input: this path works on genuinely video-only episodes.
        return self.memory(grid, hidden)

    def forecast(self, context, horizon: int, stochastic: bool = False,
                 generator: torch.Generator | None = None):
        if context.shape[1] != self.config.context or horizon < 1:
            raise ValueError("Forecast needs exactly the configured context and a positive horizon")
        grid, hidden = self.encode_context(context)
        predicted, codes = [], []
        for _ in range(horizon):
            mean, log_std = self.infer_prior(grid, hidden)
            if stochastic:
                noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                                    generator=generator)
                latent_action = torch.tanh(mean + log_std.exp() * noise)
            else:
                latent_action = torch.tanh(mean)
            grid = self.predict_grid(grid, hidden, latent_action)
            hidden = self.observe_grid(hidden, grid)
            predicted.append(grid)
            codes.append(latent_action)
        return torch.stack(predicted, 1), torch.stack(codes, 1)
