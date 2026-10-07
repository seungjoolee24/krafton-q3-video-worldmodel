from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import FeatureData, read_manifest, select_rows
from .evaluation import evaluate
from .model import ModelConfig, VideoWorldModel
from .objective import training_objective
from .stem_adapter import load_reference_stem
from .utils import (atomic_json, atomic_torch_save, capture_rng, file_hash,
                    git_revision, persist_file, restore_rng, seed_everything)


def read_config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    ModelConfig(**config["model"])
    schedule = config["horizon_schedule"]
    if not schedule or schedule[0][0] != 0:
        raise ValueError("Horizon schedule must start at step zero")
    thresholds = [item[0] for item in schedule]
    if thresholds != sorted(set(thresholds)) or any(not 1 <= item[1] <= 32 for item in schedule):
        raise ValueError("Invalid horizon schedule")
    for key in ["batch_size", "total_steps", "log_every", "checkpoint_every", "eval_every", "eval_horizon"]:
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["eval_horizon"] > 32 or not 0 <= config["start_probability"] <= 1:
        raise ValueError("Invalid evaluation horizon or sampling probability")
    return config


def horizon_at(config, completed_steps: int):
    horizon = 1
    for threshold, candidate in config["horizon_schedule"]:
        if completed_steps >= threshold:
            horizon = candidate
    return horizon


def initialize_weights(model, path: Path, cache_index):
    """Transfer model weights across data subsets, without restoring training state."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "video-only-wam/1" or saved.get("actions_used") is not False:
        raise ValueError("Expected a video-only-wam initialization checkpoint")
    if saved["model_config"] != model.config_dict():
        raise ValueError("Initialization model architecture mismatch")
    for key in ["stem_sha256", "stem_architecture_sha256"]:
        if saved[key] != cache_index[key]:
            raise ValueError("Initialization stem differs from the current feature space")
    model.load_state_dict(saved["model"], strict=True)
    return {"checkpoint": str(path), "source_step": saved["step"],
            "source_git_revision": saved["git_revision"],
            "source_cache_fingerprint": saved["cache_fingerprint"],
            "optimizer_restored": False, "rng_restored": False}


def train(kit: Path, cache: Path, out: Path, config: dict, device,
          resume: Path | None = None, persist: Path | None = None,
          max_steps: int | None = None, max_seconds: float | None = None,
          init_from: Path | None = None):
    if resume is not None and init_from is not None:
        raise ValueError("Choose either resume or model-weight initialization")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    budget_started = time.perf_counter()
    seed_everything(config["seed"])
    data = FeatureData(cache, config["model"]["context"])
    manifest, manifest_hash = read_manifest(kit / "dataset")
    if manifest_hash != data.index["manifest_sha256"]:
        raise ValueError("Dataset manifest does not match the cache")
    if data.index["stem_sha256"] != file_hash(kit / "stem" / "reference_stem.pt"):
        raise ValueError("Stem weights do not match the feature cache")
    if data.index["stem_architecture_sha256"] != file_hash(kit / "wam" / "stem.py"):
        raise ValueError("Stem architecture does not match the feature cache")
    if data.index["selection_seed"] != config["selection_seed"]:
        raise ValueError("Cache and config use different selection seeds")
    for name in ["train", "dev"]:
        expected = [row["episode_id"] for row in select_rows(
            manifest, name, config[f"{name}_limit"], config["selection_seed"])]
        if expected != data.index[f"{name}_ids"]:
            raise ValueError(f"Cache {name} episodes differ from the configured split")
    out.mkdir(parents=True, exist_ok=True)
    stem = load_reference_stem(kit).to(device)
    model = VideoWorldModel(ModelConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    use_amp = bool(config["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(config["seed"])
    completed, best, recent_validation = 0, math.inf, None
    revision = git_revision()
    initialization = None
    if init_from is not None:
        initialization = initialize_weights(model, init_from, data.index)
        print(f"Initialize model weights from source step {initialization['source_step']}; "
              "new optimizer and update count start at zero.", flush=True)
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["cache_fingerprint"] != data.index["fingerprint"]:
            raise ValueError("Resume cache/split mismatch; rebuild the same selected cache")
        if saved["model_config"] != model.config_dict():
            raise ValueError("Resume model architecture mismatch")
        # Logging/budget changes are safe; changes to the actual learning recipe are not.
        ignored = {"total_steps", "log_every", "checkpoint_every", "eval_every", "eval_limit", "preview_episodes", "evaluate_at_start"}
        if {k: v for k, v in saved["config"].items() if k not in ignored} != {k: v for k, v in config.items() if k not in ignored}:
            raise ValueError("Resume recipe changed; preserve the original training configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        completed, best = saved["step"], saved["best_validation_mse"]
        recent_validation = saved.get("validation")
        initialization = saved.get("initialization")
        restore_rng(saved["rng"], rng)
        print(f"Resume step {completed}, original code {saved['git_revision']}, current code {revision}", flush=True)
    provenance = {"config": config, "git_revision": revision, "cache_fingerprint": data.index["fingerprint"],
                  "train_ids": data.index["train_ids"], "dev_ids": data.index["dev_ids"],
                  "torch": str(torch.__version__), "device": str(device), "actions_used": False,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "trainable_parameters": sum(p.numel() for p in model.parameters()),
                  "stem_parameters": sum(p.numel() for p in stem.parameters()),
                  "initialization": initialization}
    atomic_json(out / "run.json", provenance)
    persist_file(out / "run.json", persist)
    stop = max_steps if max_steps is not None else config["total_steps"]
    if stop <= completed:
        print(f"Already at step {completed}; requested total stop is {stop}. No updates.", flush=True)
        data.close()
        return model

    def save(name: str):
        checkpoint = {
            "format": "video-only-wam/1", "actions_used": False,
            "model": model.state_dict(), "model_config": model.config_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "rng": capture_rng(rng), "step": completed,
            "best_validation_mse": best, "validation": recent_validation,
            "config": config, "git_revision": revision,
            "cache_fingerprint": data.index["fingerprint"],
            "stem_sha256": data.index["stem_sha256"],
            "stem_architecture_sha256": data.index["stem_architecture_sha256"],
            "initialization": initialization,
        }
        path = out / name
        atomic_torch_save(path, checkpoint)
        persist_file(path, persist)
        if (out / "training.jsonl").exists():
            persist_file(out / "training.jsonl", persist)

    model.train()
    started = time.perf_counter()
    initial_step = completed
    last_validation_step = None
    stop_reason = "step_budget"

    def exhausted():
        return max_seconds is not None and time.perf_counter() - budget_started >= max_seconds

    def validate():
        nonlocal best, recent_validation, last_validation_step
        report = out / "validation" / f"step_{completed:06d}"
        recent_validation = evaluate(model, stem, data, device, report,
                                     config["eval_horizon"], config["eval_limit"],
                                     config["preview_episodes"])
        last_validation_step = completed
        score = recent_validation["mse"]["mean"]
        print(f"Validation step={completed} prior RGB MSE={score:.6f}, "
              f"copy-last={recent_validation['copy_last_mse']['mean']:.6f}", flush=True)
        if persist is not None:
            destination = persist / "validation" / report.name
            for artifact in report.iterdir():
                persist_file(artifact, destination)
        if score < best:
            best = score
            save("best.pt")

    try:
        if completed == 0 and config.get("evaluate_at_start", False):
            validate()
        while completed < stop:
            if exhausted():
                stop_reason = "time_budget"
                break
            horizon = horizon_at(config, completed)
            grids = data.sample_batch(rng, config["batch_size"], horizon,
                                      config["start_probability"]).to(device)
            beta = config["kl_beta"] * min(1.0, (completed + 1) / max(1, config["kl_warmup_steps"]))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, metrics = training_objective(model, grids, horizon, beta,
                                                   config["free_nats"], config["motion_weight"],
                                                   config["prior_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite objective before update at step {completed + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradients before update at step {completed + 1}")
            scaler.step(optimizer)
            scaler.update()
            completed += 1
            ending = completed == stop or exhausted()
            if exhausted() and completed < stop:
                stop_reason = "time_budget"
            if completed % config["log_every"] == 0 or ending:
                seconds = time.perf_counter() - started
                record = {"step": completed, "horizon": horizon, "beta": beta,
                          "elapsed_seconds": seconds,
                          "updates_per_second": (completed - initial_step) / max(seconds, 1e-9),
                          "gradient_norm": float(gradient_norm),
                          **{key: float(value) for key, value in metrics.items()}}
                with open(out / "training.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if completed % config["eval_every"] == 0 or ending:
                validate()
            if completed % config["checkpoint_every"] == 0 or ending:
                save("latest.pt")
            if ending:
                break
    except KeyboardInterrupt:
        save("latest.pt")
        print(f"Interrupted; last completed update {completed} saved.", flush=True)
        raise
    if last_validation_step != completed:
        validate()
    save("latest.pt")
    save("final.pt")
    completion = {"step": completed, "requested_steps": stop, "reason": stop_reason,
                  "wall_seconds": time.perf_counter() - budget_started,
                  "max_training_seconds": max_seconds,
                  "final_validation_mse": recent_validation["mse"]["mean"],
                  "copy_last_mse": recent_validation["copy_last_mse"]["mean"],
                  "best_validation_mse": best, "actions_used": False}
    completion["gpu_peak_memory_gb"] = (torch.cuda.max_memory_allocated(device) / 1e9
                                          if device.type == "cuda" else None)
    atomic_json(out / "completion.json", completion)
    persist_file(out / "completion.json", persist)
    data.close()
    return model
