from __future__ import annotations

import torch
from torch.nn import functional as F


def gaussian_kl(mean_q, log_std_q, mean_p, log_std_p):
    # KL before the shared invertible tanh transformation; compute in float32.
    mean_q, log_std_q = mean_q.float(), log_std_q.float()
    mean_p, log_std_p = mean_p.float(), log_std_p.float()
    variance_ratio = torch.exp(2.0 * (log_std_q - log_std_p))
    squared_offset = (mean_q - mean_p).square() * torch.exp(-2.0 * log_std_p)
    return (log_std_p - log_std_q + 0.5 * (variance_ratio + squared_offset - 1)).sum(-1)


def motion_weighted_huber(predicted, target, previous, motion_weight: float):
    # A smooth motion weight, without ground-truth masks or action labels.
    motion = (target.float() - previous.float()).abs().mean(1, keepdim=True).detach()
    normalized = (motion / (motion.mean((-2, -1), keepdim=True) + 1e-3)).clamp(max=5)
    weights = 1.0 + motion_weight * normalized
    residual = F.smooth_l1_loss(predicted.float(), target.float(), reduction="none", beta=0.05)
    return (residual * weights).sum() / (weights.sum() * target.shape[1])


def training_objective(model, grids, horizon: int, beta: float, free_nats: float,
                       motion_weight: float, prior_weight: float):
    context = model.config.context
    if grids.shape[1] < context + horizon:
        raise ValueError("Training window shorter than context + prediction horizon")
    observed, history = model.encode_context(grids[:, :context])
    rollout_grid, rollout_history = observed, history
    posterior_loss = observed.new_zeros((), dtype=torch.float32)
    prior_loss = posterior_loss.clone()
    kl_loss = posterior_loss.clone()
    raw_kl = posterior_loss.clone()
    posterior_means = []
    posterior_stds = []
    for index in range(horizon):
        target = grids[:, context + index]
        mean_q, log_q = model.infer_posterior(observed, history, target)
        mean_p, log_p = model.infer_prior(observed, history)
        latent_action = torch.tanh(mean_q + log_q.exp() * torch.randn_like(mean_q))
        reconstructed = model.predict_grid(observed, history, latent_action)
        posterior_loss = posterior_loss + motion_weighted_huber(
            reconstructed, target, observed, motion_weight)
        kl = gaussian_kl(mean_q, log_q, mean_p, log_p)
        raw_kl = raw_kl + kl.mean()
        kl_loss = kl_loss + kl.clamp_min(free_nats).mean()
        posterior_means.append(torch.tanh(mean_q).detach())
        posterior_stds.append(log_q.exp().detach())

        # Autonomous rollout has NO true future observation or posterior code input.
        rollout_mean, _ = model.infer_prior(rollout_grid, rollout_history)
        rollout_grid = model.predict_grid(rollout_grid, rollout_history, torch.tanh(rollout_mean))
        prior_loss = prior_loss + motion_weighted_huber(
            rollout_grid, target, observed, motion_weight)
        rollout_history = model.observe_grid(rollout_history, rollout_grid)

        # Ground truth is used only in the training-only posterior/teacher branch.
        history = model.observe_grid(history, target)
        observed = target
    posterior_loss, prior_loss = posterior_loss / horizon, prior_loss / horizon
    kl_loss, raw_kl = kl_loss / horizon, raw_kl / horizon
    total = posterior_loss + prior_weight * prior_loss + beta * kl_loss
    metrics = {
        "loss": total.detach(), "posterior_loss": posterior_loss.detach(),
        "prior_rollout_loss": prior_loss.detach(), "kl": raw_kl.detach(),
        "posterior_code_std": torch.stack(posterior_means).float().std(unbiased=False),
        "posterior_uncertainty": torch.stack(posterior_stds).float().mean(),
    }
    return total, metrics
