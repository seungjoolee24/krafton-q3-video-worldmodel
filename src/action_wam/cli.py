from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from video_wam.stem_adapter import load_reference_stem
from video_wam.utils import device_from_string, file_hash
from .data import LabelledFeatureData, prepare_cache
from .evaluation import evaluate
from .model import ActionDifferenceWorldModel, ActionModelConfig
from .objective import training_objective
from .training import read_config, train


def run_smoke(device, kit=None):
    """Two tiny synthetic updates, never dataset training or persisted weights."""
    torch.manual_seed(42)
    model = ActionDifferenceWorldModel(ActionModelConfig(hidden_channels=16, context=4, difference_lags=(1, 2))).to(device)
    grids = torch.rand(2, 7, 48, 4, 4, device=device) * 1.6 - 0.8
    actions = torch.linspace(-0.8, 0.8, 12, device=device).reshape(2, 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_objective(model, grids, actions, horizon=3, pixel_weight=0, edge_weight=0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predicted = model.forecast(grids[:, :4], actions[:, :3], actions[:, 3:])
        single = model.forecast(grids[1:2, :4], actions[1:2, :3], actions[1:2, 3:])
        assert predicted.shape == (2, 3, 48, 4, 4) and torch.isfinite(predicted).all()
        assert torch.allclose(predicted[1:2], single, atol=1e-4, rtol=1e-4)
        state = model.encode_context(grids[:, :4], actions[:, :3])
        probe = actions.new_full((2,), 0.8)
        effect = float((model.predict_grid(state, probe) - model.predict_grid(state, -probe)).square().mean())
        assert effect > 0, "Updated transition ignored the explicit action input"
        if kit is not None:
            stem = load_reference_stem(kit).to(device)
            image = stem.decoder(predicted[:, 0])
            assert image.shape == (2, 3, 32, 32) and torch.isfinite(image).all()
    full = ActionDifferenceWorldModel()
    return {"synthetic_updates": 2, "loss": float(loss), "actions_used": True,
            "future_image_input": False, "batch_independent": True,
            "action_feature_effect": effect, "default_trainable_parameters": sum(p.numel() for p in full.parameters()),
            "device": str(device), "difference_lags": [1, 4, 8]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Labelled-only action-conditioned feature-difference world model")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for key in ["kit", "cache", "config"]:
        prep.add_argument(f"--{key}", type=Path, required=True)
    prep.add_argument("--encode-batch", type=int, default=64)
    prep.add_argument("--device", default="auto")
    training = commands.add_parser("train")
    for key in ["kit", "cache", "out", "config"]:
        training.add_argument(f"--{key}", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.add_argument("--persist", type=Path)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--max-seconds", type=float)
    training.add_argument("--device", default="auto")
    testing = commands.add_parser("evaluate")
    for key in ["kit", "cache", "checkpoint", "out"]:
        testing.add_argument(f"--{key}", type=Path, required=True)
    testing.add_argument("--horizon", type=int, default=32)
    testing.add_argument("--previews", type=int, default=3)
    testing.add_argument("--device", default="auto")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--kit", type=Path)
    smoke.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = device_from_string(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if args.command == "prepare":
        read_config(args.config)
        if args.encode_batch < 1:
            parser.error("--encode-batch must be positive")
        index = prepare_cache(args.kit, args.cache, load_reference_stem(args.kit), device, args.encode_batch)
        print(json.dumps({"train_episodes": len(index["train_ids"]), "dev_episodes": len(index["dev_ids"]),
                          "actions_used": index["actions_used"], "cache_fingerprint": index["fingerprint"]}))
    elif args.command == "train":
        if args.max_steps is not None and args.max_steps < 1:
            parser.error("--max-steps must be positive")
        train(args.kit, args.cache, args.out, read_config(args.config), device,
              args.resume, args.persist, args.max_steps, args.max_seconds)
    elif args.command == "evaluate":
        if not 1 <= args.horizon <= 32:
            parser.error("--horizon must be in [1,32]")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "action-difference-wam/1" or saved.get("actions_used") is not True:
            raise ValueError("Expected an action-conditioned checkpoint")
        data = LabelledFeatureData(args.cache, context=saved["model_config"]["context"])
        try:
            if saved["cache_fingerprint"] != data.index["fingerprint"]:
                raise ValueError("Checkpoint/cache/action labels differ")
            if saved["stem_sha256"] != file_hash(args.kit / "stem/reference_stem.pt"):
                raise ValueError("Checkpoint and visual stem differ")
            model = ActionDifferenceWorldModel(ActionModelConfig(**saved["model_config"])).to(device)
            model.load_state_dict(saved["model"])
            result = evaluate(model, load_reference_stem(args.kit).to(device), data, device, args.out, args.horizon, args.previews)
            print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))
        finally:
            data.close()
    elif args.command == "smoke":
        print(json.dumps(run_smoke(device, args.kit), indent=2))


if __name__ == "__main__":
    main()
