from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from video_wam.data import read_manifest
from video_wam.stem_adapter import load_reference_stem
from video_wam.utils import (atomic_json, atomic_torch_save, capture_rng, file_hash,
                             git_revision, persist_file, restore_rng, seed_everything)
from .data import LabelledFeatureData, dataset_path, select_labelled_rows
from .evaluation import evaluate
from .model import ActionDifferenceWorldModel, ActionModelConfig
from .objective import training_objective


def read_config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    ActionModelConfig(**config["model"])
    if config.get("labelled_only") is not True:
        raise ValueError("This experiment requires labelled_only=true")
    if any(config.get(key) not in (None, 0) for key in ["train_limit", "dev_limit", "eval_limit"]):
        raise ValueError("All labelled episodes must be used; subset limits are unsupported")
    schedule = config["horizon_schedule"]
    if (not schedule or schedule[0][0] != 0 or
            [s[0] for s in schedule] != sorted(set(s[0] for s in schedule)) or
            any(not 1 <= s[1] <= 32 for s in schedule)):
        raise ValueError("Invalid horizon curriculum")
    for key in ["batch_size", "total_steps", "log_every", "checkpoint_every", "eval_every", "eval_horizon", "rgb_loss_frames"]:
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["eval_horizon"] > 32 or not 0 <= config["start_probability"] <= 1:
        raise ValueError("Invalid evaluation horizon or diagnostic-window probability")
    if config["learning_rate"] <= 0 or config["grad_clip"] <= 0:
        raise ValueError("Learning rate and gradient clipping must be positive")
    for key in ["teacher_weight", "motion_weight", "pixel_weight", "edge_weight", "weight_decay"]:
        if config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    return config


def horizon_at(config, step):
    horizon = 1
    for threshold, value in config["horizon_schedule"]:
        if step >= threshold:
            horizon = value
    return horizon


class EpisodeCoverageSampler:
    """Shuffle complete episode passes; every train episode is used before repeat.

    Keep the pass/cursor and RNG in checkpoints for exact continuation. Frames
    inside an episode are still sampled as contiguous original-rate windows.
    """
    def __init__(self, episode_ids, rng):
        self.ids = sorted(int(i) for i in episode_ids)
        self.rng, self.order, self.cursor, self.seen = rng, [], 0, set()
        if not self.ids:
            raise ValueError("No training episodes")

    def next_batch(self, size):
        selected = []
        while len(selected) < size:
            if self.cursor == len(self.order):
                self.order = [int(i) for i in self.rng.permutation(self.ids)]
                self.cursor = 0
            count = min(size - len(selected), len(self.order) - self.cursor)
            selected.extend(self.order[self.cursor:self.cursor + count])
            self.cursor += count
        self.seen.update(selected)
        return selected

    def state_dict(self):
        return {"ids": self.ids, "order": self.order, "cursor": self.cursor, "seen": sorted(self.seen)}

    def load_state_dict(self, state):
        if state["ids"] != self.ids or (state["order"] and sorted(state["order"]) != self.ids):
            raise ValueError("Episode coverage checkpoint has a different training split")
        if not 0 <= state["cursor"] <= len(state["order"]) or not set(state["seen"]).issubset(self.ids):
            raise ValueError("Invalid episode coverage checkpoint")
        self.order, self.cursor, self.seen = list(state["order"]), int(state["cursor"]), set(state["seen"])


def train(kit: Path, cache: Path, out: Path, config: dict, device,
          resume: Path | None = None, persist: Path | None = None,
          max_steps: int | None = None, max_seconds: float | None = None):
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    started = time.perf_counter()
    seed_everything(config["seed"])
    data = LabelledFeatureData(cache, context=config["model"]["context"])
    try:
        manifest, manifest_hash = read_manifest(kit / "dataset")
        if data.index["manifest_sha256"] != manifest_hash:
            raise ValueError("Manifest differs from prepared labelled cache")
        for key, path in [("stem_sha256", kit / "stem/reference_stem.pt"),
                          ("stem_architecture_sha256", kit / "wam/stem.py")]:
            if data.index[key] != file_hash(path):
                raise ValueError("Frozen visual stem differs from the cached feature space")
        for split in ["train", "dev"]:
            expected = [r["episode_id"] for r in select_labelled_rows(manifest, split)]
            if expected != data.index[f"{split}_ids"]:
                raise ValueError(f"Cache does not contain every labelled {split} episode")
        for row in select_labelled_rows(manifest, "all"):
            eid = str(row["episode_id"])
            if data.index["sidecar_sha256"][eid] != file_hash(dataset_path(kit / "dataset", row["sidecar"])):
                raise ValueError("Source action labels changed; rebuild the labelled cache")
        stem = load_reference_stem(kit).to(device)
        model = ActionDifferenceWorldModel(ActionModelConfig(**config["model"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        use_amp = bool(config["mixed_precision"] and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(config["seed"])
        coverage = EpisodeCoverageSampler(data.index["train_ids"], rng)
        completed, best, validation = 0, math.inf, None
        revision = git_revision()
        if resume is not None:
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            if saved.get("format") != "action-difference-wam/1" or saved.get("actions_used") is not True:
                raise ValueError("Resume requires this action model's checkpoint, not a video-only model")
            if saved["cache_fingerprint"] != data.index["fingerprint"]:
                raise ValueError("Resume data/actions/split fingerprint mismatch")
            if saved["model_config"] != model.config_dict():
                raise ValueError("Resume model architecture mismatch")
            harmless = {"total_steps", "log_every", "checkpoint_every", "eval_every", "preview_episodes", "evaluate_at_start"}
            if ({k: v for k, v in saved["config"].items() if k not in harmless} !=
                    {k: v for k, v in config.items() if k not in harmless}):
                raise ValueError("Resume training recipe changed; start a separately named experiment")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            restore_rng(saved["rng"], rng)
            coverage.load_state_dict(saved["episode_coverage"])
            completed, best, validation = saved["step"], saved["best_validation_mse"], saved.get("validation")
            print(f"Resume action-model step {completed}; code {revision}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        if resume is None and (out / "training.jsonl").is_file():
            raise ValueError("Existing run logs require explicit resume or a new run directory")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        atomic_json(out / "run.json", {
            "config": config, "git_revision": revision, "cache_fingerprint": data.index["fingerprint"],
            "train_ids": data.index["train_ids"], "dev_ids": data.index["dev_ids"],
            "actions_used": True, "labelled_only": True, "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "stem_parameters": sum(p.numel() for p in stem.parameters()),
            "initialization": "fresh action model; provided frozen visual stem",
            "action_alignment": "a[t] applies between frame[t] and frame[t+1]; memory frame[t] uses a[t-1]",
            "resume_source": str(resume) if resume else None, "starting_step": completed})
        persist_file(out / "run.json", persist)
        stop, starting_step = max_steps if max_steps is not None else config["total_steps"], completed
        if stop <= completed:
            print(f"Already at step {completed}; requested {stop}. No additional updates.", flush=True)
            return model
        if resume is not None:
            source_log = resume.parent / "training.jsonl"
            local_log = out / "training.jsonl"
            if not local_log.is_file() and source_log.is_file():
                # A fresh Colab runtime resumes from Drive. Preserve its earlier
                # log rather than overwriting the Drive history with only new rows.
                persist_file(source_log, out)
            if local_log.is_file():
                records = [json.loads(line) for line in local_log.read_text(encoding="utf-8").splitlines() if line.strip()]
                if records and max(row["step"] for row in records) > completed:
                    raise ValueError("Run log is newer than the resume checkpoint; use its latest checkpoint or a separate run directory")

        def save(name):
            atomic_torch_save(out / name, {
                "format": "action-difference-wam/1", "actions_used": True, "labelled_only": True,
                "model": model.state_dict(), "model_config": model.config_dict(), "config": config,
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": capture_rng(rng),
                "episode_coverage": coverage.state_dict(), "step": completed,
                "best_validation_mse": best, "validation": validation, "git_revision": revision,
                "cache_fingerprint": data.index["fingerprint"], "stem_sha256": data.index["stem_sha256"],
                "stem_architecture_sha256": data.index["stem_architecture_sha256"]})
            persist_file(out / name, persist)
            if (out / "training.jsonl").is_file():
                persist_file(out / "training.jsonl", persist)

        last_validation_step, stop_reason = None, "step_budget"

        def validate():
            nonlocal best, validation, last_validation_step
            report = out / "validation" / f"step_{completed:06d}"
            validation = evaluate(model, stem, data, device, report, config["eval_horizon"], config["preview_episodes"])
            last_validation_step = completed
            print(f"Validation step={completed} action RGB MSE={validation['mse']['mean']:.6f}; copy={validation['copy_last_mse']['mean']:.6f}", flush=True)
            if persist is not None:
                for path in report.iterdir():
                    persist_file(path, persist / "validation" / report.name)
            if validation["mse"]["mean"] < best:
                best = validation["mse"]["mean"]
                save("best.pt")

        def exhausted():
            return max_seconds is not None and time.perf_counter() - started >= max_seconds

        model.train()
        try:
            if completed == 0 and config.get("evaluate_at_start", False):
                validate()
            while completed < stop:
                if exhausted():
                    stop_reason = "time_budget"
                    break
                horizon = horizon_at(config, completed)
                ids = coverage.next_batch(config["batch_size"])
                batch = data.sample_batch(rng, config["batch_size"], horizon, config["start_probability"], episode_ids=ids)
                grids, actions = batch["grids"].to(device), batch["actions"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss, metrics = training_objective(
                        model, grids, actions, horizon, motion_weight=config["motion_weight"],
                        teacher_weight=config["teacher_weight"], stem=stem, rgb_targets=batch["future_rgb"],
                        pixel_weight=config["pixel_weight"], edge_weight=config["edge_weight"], rgb_frames=config["rgb_loss_frames"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss before update {completed + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Non-finite gradients before update {completed + 1}")
                scaler.step(optimizer)
                scaler.update()
                completed += 1
                ending = completed == stop or exhausted()
                if exhausted() and completed < stop:
                    stop_reason = "time_budget"
                if completed % config["log_every"] == 0 or ending:
                    elapsed = time.perf_counter() - started
                    record = {"step": completed, "horizon": horizon, "elapsed_seconds": elapsed,
                              "gradient_norm": float(norm), "train_episodes_seen": len(coverage.seen),
                              "updates_per_second": (completed - starting_step) / max(elapsed, 1e-9),
                              **{k: float(v) for k, v in metrics.items()}}
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
            print(f"Interrupted; completed update {completed} saved.", flush=True)
            raise
        except Exception as error:
            save("latest.pt")
            atomic_json(out / "error.json", {"step": completed, "error_type": type(error).__name__, "message": str(error)})
            persist_file(out / "error.json", persist)
            raise
        if last_validation_step != completed:
            validate()
        save("latest.pt")
        save("final.pt")
        completion = {
            "step": completed, "requested_steps": stop, "reason": stop_reason,
            "starting_step": starting_step, "actual_updates": completed - starting_step,
            "wall_seconds": time.perf_counter() - started, "max_training_seconds": max_seconds,
            "final_validation_mse": validation["mse"]["mean"], "best_validation_mse": best,
            "copy_last_mse": validation["copy_last_mse"]["mean"], "actions_used": True, "labelled_only": True,
            "train_episodes_seen": len(coverage.seen), "train_episodes_total": len(data.train),
            "dev_episodes": len(data.dev), "gpu_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None}
        atomic_json(out / "completion.json", completion)
        persist_file(out / "completion.json", persist)
        print(json.dumps(completion), flush=True)
        return model
    finally:
        data.close()
