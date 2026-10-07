from __future__ import annotations

import torch

from .model import ModelConfig, VideoWorldModel
from .objective import gaussian_kl, training_objective
from .stem_adapter import load_reference_stem


def run_smoke(device, kit=None):
    """Two tiny synthetic updates, no dataset training and no saved learned weights."""
    torch.manual_seed(7)
    if device.type == "cpu":
        torch.set_num_threads(2)
    model = VideoWorldModel(ModelConfig(hidden_channels=16, context=4)).to(device)
    grids = torch.rand(2, 6, 48, 16, 16, device=device) * 1.4 - 0.7
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_objective(model, grids, 2, 0.001, 0.05, 1.0, 1.0)
        loss.backward()
        assert torch.isfinite(loss), "Non-finite loss"
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    for module in [model.prior, model.posterior, model.transition, model.memory]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), "Missing gradient path"
    model.eval()
    with torch.inference_mode():
        full, _ = model.forecast(grids[:, :4], 3)
        single, _ = model.forecast(grids[1:2, :4], 3)
        assert full.shape == (2, 3, 48, 16, 16)
        assert full.min() >= -1 and full.max() <= 1
        assert torch.allclose(full[1:2], single, atol=1e-4, rtol=1e-4), "Batch dependence"
        before = full.clone()
        def forbidden(*args, **kwargs):
            raise AssertionError("Inference used the future-dependent posterior")
        model.posterior.forward = forbidden
        after, _ = model.forecast(grids[:, :4], 3)
        assert torch.equal(before, after), "Forecast unexpectedly depends on posterior"
        zero = torch.zeros(2, 1, device=device)
        assert torch.allclose(gaussian_kl(zero, zero, zero, zero), torch.zeros(2, device=device))
    result = {"synthetic_updates": 2, "losses": losses, "finite_gradients": True,
              "prior_only_inference": True, "batch_independence": True,
              "latent_bounds": True, "device": str(device)}
    if kit is not None:
        stem = load_reference_stem(kit).to(device)
        with torch.inference_mode():
            frame = torch.rand(1, 3, 128, 128, device=device)
            encoded = stem.encoder(frame)
            decoded = stem.decoder(encoded)
        assert encoded.shape == (1, 48, 16, 16)
        assert decoded.shape == frame.shape and torch.isfinite(decoded).all()
        assert decoded.min() >= 0 and decoded.max() <= 1
        result["reference_stem"] = True
    return result
