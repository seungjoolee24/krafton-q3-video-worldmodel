from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from video_wam.model import ConvGRUCell


@dataclass(frozen=True)
class ActionModelConfig:
    grid_channels: int = 48
    hidden_channels: int = 64
    action_embedding: int = 16
    context: int = 32
    difference_lags: tuple[int, ...] = (1, 4, 8)

    def __post_init__(self):
        # JSON checkpoints represent the tuple as a list. Normalize both forms.
        object.__setattr__(self, "difference_lags", tuple(self.difference_lags))
        if self.grid_channels != 48:
            raise ValueError("The provided visual boundary has 48 grid channels")
        if self.hidden_channels < 8 or self.hidden_channels % 8:
            raise ValueError("Hidden channels must be positive and divisible by 8")
        if self.action_embedding < 1 or not 1 <= self.context <= 32:
            raise ValueError("Action embedding must be positive; context must be in [1,32]")
        if (not self.difference_lags or
                any(type(lag) is not int or not 1 <= lag <= 31 for lag in self.difference_lags) or
                len(set(self.difference_lags)) != len(self.difference_lags)):
            raise ValueError("Difference lags must be distinct integers in [1,31]")


@dataclass(frozen=True)
class ActionState:
    """Bounded recurrent state; no raw RGB, future targets, or stochastic code."""

    hidden: torch.Tensor
    history: torch.Tensor  # (B,L,C,H,W), most recent frame last, L <= max_lag + 1

    @property
    def grid(self):
        return self.history[:, -1]


class ActionDifferenceWorldModel(nn.Module):
    """A deterministic action-conditioned residual transition with multiframe memory.

    Action a[t] is applied between image t and image t+1. When image t is
    observed, the memory receives a[t-1] with its observed feature differences.
    To predict image t+1, the transition separately receives known action a[t].
    Differences span several frame intervals, but EVERY intervening frame and
    action updates the recurrent memory; actions are never temporally skipped.
    """

    def __init__(self, config: ActionModelConfig = ActionModelConfig()):
        super().__init__()
        self.config = config
        c, h, e = config.grid_channels, config.hidden_channels, config.action_embedding
        self.initial = nn.Conv2d(c, h, 3, padding=1)
        self.action_embed = nn.Sequential(nn.Linear(1, e), nn.SiLU(), nn.Linear(e, e))
        self.observation = nn.Sequential(
            nn.Conv2d(c * (1 + len(config.difference_lags)) + e, h, 1),
            nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.GroupNorm(8, h), nn.SiLU(),
        )
        self.memory = ConvGRUCell(h, h)
        self.transition = nn.Sequential(
            nn.Conv2d(c + h + e, h, 3, padding=1), nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.GroupNorm(8, h), nn.SiLU(),
            nn.Conv2d(h, c, 3, padding=1),
        )
        # An untrained baseline copies the last grid instead of adding random motion.
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def config_dict(self):
        return asdict(self.config)

    def _validate_grids(self, grids):
        if (grids.ndim != 5 or grids.shape[1] < 1 or
                grids.shape[2] != self.config.grid_channels):
            raise ValueError("Expected grids (B,T,48,H,W), T >= 1")

    def _action(self, action, grid):
        if action.ndim == 1:
            action = action[:, None]
        if action.shape != (grid.shape[0], 1):
            raise ValueError("Each step requires one scalar action per batch item")
        return action.to(device=grid.device, dtype=grid.dtype)

    def _action_plane(self, action, grid):
        embedding = self.action_embed(self._action(action, grid))[:, :, None, None]
        return embedding.expand(-1, -1, grid.shape[-2], grid.shape[-1])

    def feature_differences(self, history):
        """Causal raw feature differences; unavailable preceding frames yield zeros.

        These are visual motion cues, not supervised or calibrated velocities.
        With 25 fps, the default lags span 0.04, 0.16 and 0.32 seconds.
        """
        self._validate_grids(history)
        current = history[:, -1]
        return tuple(current - history[:, -1 - lag] if history.shape[1] > lag
                     else torch.zeros_like(current)
                     for lag in self.config.difference_lags)

    def observe_grid(self, state: ActionState, grid, previous_action):
        """Observe image t with the action that produced it, a[t-1]."""
        if grid.shape != state.grid.shape:
            raise ValueError("Observed grid must have the same shape as the current grid")
        keep = max(self.config.difference_lags)
        history = torch.cat([state.history[:, -keep:], grid[:, None]], dim=1)
        observation = torch.cat(
            [grid, *self.feature_differences(history), self._action_plane(previous_action, grid)], 1)
        hidden = self.memory(self.observation(observation), state.hidden)
        return ActionState(hidden, history)

    def encode_context(self, grids, context_actions):
        self._validate_grids(grids)
        if context_actions.shape != (grids.shape[0], grids.shape[1] - 1):
            raise ValueError("T context frames require exactly T-1 aligned past actions")
        first = grids[:, 0]
        hidden = torch.tanh(self.initial(first))
        # Frame 0 has no preceding observed transition. Its action and deltas are zero.
        observation = torch.cat(
            [first, *[torch.zeros_like(first) for _ in self.config.difference_lags],
             self._action_plane(first.new_zeros(first.shape[0]), first)], 1)
        state = ActionState(self.memory(self.observation(observation), hidden), first[:, None])
        for t in range(1, grids.shape[1]):
            state = self.observe_grid(state, grids[:, t], context_actions[:, t - 1])
        return state

    def predict_grid(self, state: ActionState, action):
        """Predict image t+1 from state through t and the explicitly supplied a[t]."""
        grid = state.grid
        delta = self.transition(torch.cat([grid, state.hidden, self._action_plane(action, grid)], 1))
        return (grid + delta).clamp(-1.0, 1.0)

    def step(self, state: ActionState, action):
        grid = self.predict_grid(state, action)
        # During rollout, all new observations/differences come from predictions.
        return grid, self.observe_grid(state, grid, action)

    def forecast(self, context, context_actions, future_actions):
        self._validate_grids(context)
        if context.shape[1] != self.config.context:
            raise ValueError("Forecast requires exactly the configured number of context frames")
        if (future_actions.ndim != 2 or future_actions.shape[0] != context.shape[0] or
                future_actions.shape[1] < 1):
            raise ValueError("Expected future actions (B,H), H >= 1")
        state = self.encode_context(context, context_actions)
        predicted = []
        for action in future_actions.unbind(1):
            grid, state = self.step(state, action)
            predicted.append(grid)
        return torch.stack(predicted, 1)
