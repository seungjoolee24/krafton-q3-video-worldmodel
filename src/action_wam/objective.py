from __future__ import annotations

import torch
from torch.nn import functional as F

from video_wam.objective import motion_weighted_huber


def _foreground_weights(target):
    """Color heuristic for emphasizing objects, not ground-truth segmentation."""
    maximum, minimum = target.amax(1, keepdim=True), target.amin(1, keepdim=True)
    # Match the evaluation heuristic's absolute channel spread. Normalizing by
    # brightness would wrongly emphasize weak chromatic sensor noise in shadows.
    foreground = (((maximum - minimum) > 0.15) & (maximum > 0.2)).float()
    return 1.0 + 4.0 * foreground


def _weighted_average(error, weights):
    return (error * weights).sum() / (weights.sum() * error.shape[1]).clamp_min(1e-6)


def _pixel_and_edge_loss(predicted, target):
    predicted, target = predicted.float(), target.float()
    weights = _foreground_weights(target).detach()
    pixel = _weighted_average(
        F.smooth_l1_loss(predicted, target, reduction="none", beta=0.02), weights)
    edges = []
    for dim in (-2, -1):
        if target.shape[dim] < 2:
            continue
        predicted_edge = predicted.diff(dim=dim)
        target_edge = target.diff(dim=dim)
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[dim], right[dim] = slice(None, -1), slice(1, None)
        edge_weights = 0.5 * (weights[tuple(left)] + weights[tuple(right)])
        edges.append(_weighted_average((predicted_edge - target_edge).abs(), edge_weights))
    edge = torch.stack(edges).mean() if edges else predicted.new_zeros(())
    return pixel, edge


def training_objective(model, grids, actions, horizon: int, motion_weight: float = 1.0,
                       teacher_weight: float = 0.25, stem=None, rgb_targets=None,
                       pixel_weight: float = 0.1, edge_weight: float = 0.05,
                       rgb_frames: int = 2):
    """Known-action autonomous rollout plus a separate teacher-forced auxiliary.

    grids: (B,context+horizon,48,H,W); actions: (B,context+horizon-1).
    actions[:,t] maps grids[:,t] to grids[:,t+1]. rgb_targets, when used,
    contains FUTURE frames only: (B,horizon,3,R,R). No future observation is
    supplied to the autonomous prediction branch; future targets enter losses.
    """
    context = model.config.context
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive integer")
    if grids.ndim != 5 or grids.shape[1] < context + horizon:
        raise ValueError("Training window shorter than context + prediction horizon")
    if actions.ndim != 2 or actions.shape != (grids.shape[0], grids.shape[1] - 1):
        raise ValueError("Actions must align with every adjacent pair in the training window")
    if min(motion_weight, teacher_weight, pixel_weight, edge_weight) < 0:
        raise ValueError("Loss weights cannot be negative")
    use_rgb = pixel_weight > 0 or edge_weight > 0
    if use_rgb:
        if stem is None or rgb_targets is None or rgb_frames < 1:
            raise ValueError("Positive RGB/edge weights require stem, future RGB targets and rgb_frames")
        if (rgb_targets.ndim != 5 or rgb_targets.shape[:3] != (grids.shape[0], horizon, 3)):
            raise ValueError("Expected FUTURE-only RGB targets (B,horizon,3,R,R)")
        # Sparse decoding contains the first and last predicted frames when count >= 2.
        count = min(int(rgb_frames), horizon)
        rgb_indices = set(torch.linspace(0, horizon - 1, count).round().long().tolist())
    else:
        rgb_indices = set()

    state = model.encode_context(grids[:, :context], actions[:, :context - 1])
    rollout_state, teacher_state = state, state
    zero = grids.new_zeros((), dtype=torch.float32)
    rollout_loss, teacher_loss, pixel_loss, edge_loss = zero, zero, zero, zero
    for index in range(horizon):
        # Last context is frame context-1; its outgoing action is also context-1.
        action = actions[:, context - 1 + index]
        target, previous = grids[:, context + index], grids[:, context - 1 + index]
        predicted, rollout_state = model.step(rollout_state, action)
        rollout_loss = rollout_loss + motion_weighted_huber(
            predicted, target, previous, motion_weight)
        if teacher_weight > 0:
            teacher_prediction = model.predict_grid(teacher_state, action)
            teacher_loss = teacher_loss + motion_weighted_huber(
                teacher_prediction, target, previous, motion_weight)
            teacher_state = model.observe_grid(teacher_state, target, action)
        if index in rgb_indices:
            # Freezing decoder parameters does not stop gradients into predicted grids.
            decoded = stem.decoder(predicted)
            target_rgb = rgb_targets[:, index].to(device=decoded.device, dtype=decoded.dtype)
            if decoded.shape != target_rgb.shape:
                raise ValueError("Decoded image and RGB target shapes differ")
            pixel, edge = _pixel_and_edge_loss(decoded, target_rgb)
            pixel_loss, edge_loss = pixel_loss + pixel, edge_loss + edge

    rollout_loss, teacher_loss = rollout_loss / horizon, teacher_loss / horizon
    if rgb_indices:
        pixel_loss, edge_loss = pixel_loss / len(rgb_indices), edge_loss / len(rgb_indices)
    total = (rollout_loss + teacher_weight * teacher_loss +
             pixel_weight * pixel_loss + edge_weight * edge_loss)
    metrics = {"loss": total.detach(), "rollout_feature_loss": rollout_loss.detach(),
               "teacher_feature_loss": teacher_loss.detach(), "pixel_loss": pixel_loss.detach(),
               "edge_loss": edge_loss.detach()}
    return total, metrics
