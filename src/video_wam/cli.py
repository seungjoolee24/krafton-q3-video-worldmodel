from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import FeatureData, prepare_cache
from .evaluation import evaluate
from .model import ModelConfig, VideoWorldModel
from .smoke import run_smoke
from .stem_adapter import load_reference_stem
from .training import read_config, train
from .utils import device_from_string, file_hash


def model_from_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "video-only-wam/1" or checkpoint.get("actions_used") is not False:
        raise ValueError("Expected a video-only-wam checkpoint")
    model = VideoWorldModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def check_checkpoint_cache(checkpoint, data):
    if checkpoint["cache_fingerprint"] != data.index["fingerprint"]:
        raise ValueError("Checkpoint cache fingerprint does not match")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Video-only latent-action world model")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Decode RGB videos and cache frozen stem features")
    prep.add_argument("--kit", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    prep.add_argument("--encode-batch", type=int, default=64)
    prep.add_argument("--device", default="auto")
    training = commands.add_parser("train", help="Train without reading any action sidecars")
    for item in ["kit", "cache", "out", "config"]:
        training.add_argument(f"--{item}", type=Path, required=True)
    start = training.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path)
    start.add_argument("--init-from", type=Path, help="Model weights only; starts a new run on a new data subset")
    training.add_argument("--persist", type=Path, help="Drive directory for checkpoint/report backups")
    training.add_argument("--max-steps", type=int, help="Absolute total update count, not additional steps")
    training.add_argument("--max-seconds", type=float, help="Wall-clock limit for training; saves and validates before exiting")
    training.add_argument("--device", default="auto")
    evaluation = commands.add_parser("evaluate", help="Autonomous prior rollout on held-out episodes")
    for item in ["kit", "cache", "checkpoint", "out"]:
        evaluation.add_argument(f"--{item}", type=Path, required=True)
    evaluation.add_argument("--limit", type=int, default=0)
    evaluation.add_argument("--horizon", type=int, default=32)
    evaluation.add_argument("--previews", type=int, default=3)
    evaluation.add_argument("--device", default="auto")
    smoke = commands.add_parser("smoke", help="Two tiny synthetic updates and inference checks")
    smoke.add_argument("--kit", type=Path)
    smoke.add_argument("--device", default="auto")
    codes = commands.add_parser("inspect-codes", help="Offline posterior codes; these are NOT physical forces")
    codes.add_argument("--cache", type=Path, required=True)
    codes.add_argument("--checkpoint", type=Path, required=True)
    codes.add_argument("--episode", type=int, required=True)
    codes.add_argument("--out", type=Path, required=True)
    codes.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)
    device = device_from_string(arguments.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if arguments.command == "prepare":
        if arguments.encode_batch < 1:
            parser.error("--encode-batch must be positive")
        config = read_config(arguments.config)
        stem = load_reference_stem(arguments.kit)
        prepare_cache(arguments.kit, arguments.cache, stem, device,
                      config["train_limit"], config["dev_limit"], config["selection_seed"],
                      arguments.encode_batch, config["model"]["context"] + config["eval_horizon"])
    elif arguments.command == "train":
        train(arguments.kit, arguments.cache, arguments.out, read_config(arguments.config), device,
              arguments.resume, arguments.persist, arguments.max_steps, arguments.max_seconds,
              init_from=arguments.init_from)
    elif arguments.command == "evaluate":
        if not 1 <= arguments.horizon <= 32:
            parser.error("--horizon must be in [1,32]")
        model, checkpoint = model_from_checkpoint(arguments.checkpoint, device)
        data = FeatureData(arguments.cache, model.config.context)
        check_checkpoint_cache(checkpoint, data)
        if checkpoint["stem_sha256"] != file_hash(arguments.kit / "stem" / "reference_stem.pt"):
            raise ValueError("Evaluation stem differs from training")
        stem = load_reference_stem(arguments.kit).to(device)
        result = evaluate(model, stem, data, device, arguments.out,
                          arguments.horizon, arguments.limit, arguments.previews)
        print(json.dumps({key: value for key, value in result.items() if key != "per_episode"}, indent=2))
    elif arguments.command == "smoke":
        print(json.dumps(run_smoke(device, arguments.kit), indent=2))
    elif arguments.command == "inspect-codes":
        model, checkpoint = model_from_checkpoint(arguments.checkpoint, device)
        data = FeatureData(arguments.cache, model.config.context)
        check_checkpoint_cache(checkpoint, data)
        row = next((row for row in data.train + data.dev if row["episode_id"] == arguments.episode), None)
        if row is None:
            raise ValueError("Episode is not in this cache")
        features = data.features(row)
        context = torch.from_numpy(np.array(features[:model.config.context], dtype=np.float32, copy=True)).unsqueeze(0).to(device)
        rows = []
        with torch.inference_mode():
            current, hidden = model.encode_context(context)
            for index in range(model.config.context - 1, len(features) - 1):
                next_grid = torch.from_numpy(np.array(features[index + 1], dtype=np.float32, copy=True)).unsqueeze(0).to(device)
                mean, log_std = model.infer_posterior(current, hidden, next_grid)
                code = torch.tanh(mean)[0].cpu().tolist()
                rows.append([index, index / 25, *code, *log_std.exp()[0].cpu().tolist()])
                hidden, current = model.observe_grid(hidden, next_grid), next_grid
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        with open(arguments.out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["transition_frame", "time_seconds"]
                            + [f"latent_code_{i}" for i in range(model.config.latent_action_dim)]
                            + [f"pre_tanh_std_{i}" for i in range(model.config.latent_action_dim)])
            writer.writerows(rows)
        print(f"Saved {len(rows)} offline latent codes. They are not calibrated physical forces.")


if __name__ == "__main__":
    main()
