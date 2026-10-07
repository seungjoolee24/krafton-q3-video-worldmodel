"""Behavioural tests for information flow, action exclusion, and exact CPU resume."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from video_wam.data import FeatureData, prepare_cache
from video_wam.model import ModelConfig, VideoWorldModel
from video_wam.smoke import run_smoke
from video_wam.training import train
from video_wam.utils import atomic_json


class TinyEncoder(torch.nn.Module):
    def forward(self, frame):
        small = torch.nn.functional.avg_pool2d(frame, 8)
        return small.repeat(1, 16, 1, 1) * 2 - 1


class TinyDecoder(torch.nn.Module):
    def forward(self, features):
        return torch.nn.functional.interpolate((features[:, :3] + 1) / 2, scale_factor=8)


class TinyStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = TinyEncoder(), TinyDecoder()


def fake_kit(root):
    for directory in ["dataset", "stem", "wam"]:
        (root / directory).mkdir(parents=True)
    (root / "stem" / "reference_stem.pt").write_bytes(b"test-weights")
    (root / "wam" / "stem.py").write_text("# test-only architecture\n")
    rows = []
    for eid in [0, 1]:
        (root / "dataset" / f"ep_{eid:06d}.mp4").write_bytes(b"test-video" + bytes([eid]))
        # This intentionally invalid NPZ must never be opened by video-only preparation.
        (root / "dataset" / f"ep_{eid:06d}.npz").write_bytes(b"must-not-read-actions")
        rows.append({"episode_id": eid, "length": 8, "labelled": bool(eid),
                     "video": f"ep_{eid:06d}.mp4", "sidecar": f"ep_{eid:06d}.npz"})
    atomic_json(root / "dataset" / "manifest.json",
                {"resolution": 128, "fps": 25, "dev_subset": [1], "episodes": rows})


def tiny_config():
    return {
        "model": {"hidden_channels": 16, "context": 4}, "train_limit": 0, "dev_limit": 0,
        "selection_seed": 42, "seed": 17, "batch_size": 2, "total_steps": 3,
        "horizon_schedule": [[0, 1], [1, 2]], "learning_rate": 0.0003,
        "weight_decay": 0.0001, "grad_clip": 1.0, "mixed_precision": False,
        "kl_beta": 0.001, "kl_warmup_steps": 2, "free_nats": 0.05,
        "motion_weight": 1.0, "prior_weight": 1.0, "start_probability": 0.5,
        "log_every": 1, "checkpoint_every": 1, "eval_every": 3,
        "eval_horizon": 2, "eval_limit": 0, "preview_episodes": 0,
    }


class VideoOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradients_and_prior_only_forecast(self):
        result = run_smoke(torch.device("cpu"))
        self.assertTrue(result["prior_only_inference"])

    def test_future_context_is_rejected(self):
        model = VideoWorldModel(ModelConfig(hidden_channels=16, context=4))
        with self.assertRaises(ValueError):
            model.forecast(torch.zeros(1, 6, 48, 16, 16), 2)

    def test_action_sidecars_ignored_and_cache_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, cache = root / "kit", root / "cache"
            fake_kit(kit)
            frames = np.random.default_rng(9).integers(0, 256, (8, 128, 128, 3), dtype=np.uint8)
            with patch("video_wam.data.decode_rgb", return_value=frames) as decoder:
                index = prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), rgb_prefix=8)
                self.assertEqual(decoder.call_count, 2)
                prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), rgb_prefix=8)
                self.assertEqual(decoder.call_count, 2, "Cached episodes decoded again")
            self.assertFalse(index["actions_used"])
            data = FeatureData(cache, context=4)
            self.assertEqual([row["episode_id"] for row in data.train], [0])
            self.assertEqual([row["episode_id"] for row in data.dev], [1])
            self.assertEqual(data.sample_batch(np.random.default_rng(7), 2, 2, 0.5).shape,
                             (2, 6, 48, 16, 16))
            data.close()

    def test_resume_matches_uninterrupted_cpu_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, cache = root / "kit", root / "cache"
            fake_kit(kit)
            frames = np.random.default_rng(4).integers(0, 256, (8, 128, 128, 3), dtype=np.uint8)
            config = tiny_config()
            with patch("video_wam.data.decode_rgb", return_value=frames), contextlib.redirect_stdout(io.StringIO()):
                prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), rgb_prefix=8)
            with patch("video_wam.training.load_reference_stem", side_effect=lambda kit: TinyStem()), contextlib.redirect_stdout(io.StringIO()):
                full = train(kit, cache, root / "full", config, torch.device("cpu"))
                train(kit, cache, root / "partial", config, torch.device("cpu"), max_steps=2)
                resumed = train(kit, cache, root / "resumed", config, torch.device("cpu"),
                                resume=root / "partial" / "latest.pt")
            for key, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            saved = torch.load(root / "resumed" / "final.pt", weights_only=True)
            self.assertEqual(saved["step"], 3)
            self.assertFalse(saved["actions_used"])

    def test_time_budget_still_saves_a_validated_checkpoint(self):
        import itertools
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, cache = root / "kit", root / "cache"
            fake_kit(kit)
            frames = np.random.default_rng(4).integers(0, 256, (8, 128, 128, 3), dtype=np.uint8)
            config = tiny_config()
            with patch("video_wam.data.decode_rgb", return_value=frames), contextlib.redirect_stdout(io.StringIO()):
                prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), rgb_prefix=8)
            clock = itertools.chain([0.0, 0.0, 0.0], itertools.repeat(2.0))
            with patch("video_wam.training.load_reference_stem", side_effect=lambda kit: TinyStem()), \
                    patch("video_wam.training.time.perf_counter", side_effect=clock), \
                    contextlib.redirect_stdout(io.StringIO()):
                train(kit, cache, root / "out", config, torch.device("cpu"), max_seconds=1.0)
            saved = torch.load(root / "out" / "final.pt", weights_only=True)
            completion = json.loads((root / "out" / "completion.json").read_text())
            self.assertEqual(saved["step"], 1)
            self.assertEqual(completion["reason"], "time_budget")
            self.assertTrue(np.isfinite(completion["final_validation_mse"]))
            self.assertTrue((root / "out" / "latest.pt").exists())

    def test_episode_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            atomic_json(cache / "index.json", {"complete": True, "actions_used": False,
                        "episodes": [{"episode_id": 3, "split": "train"},
                                     {"episode_id": 3, "split": "dev"}]})
            with self.assertRaisesRegex(ValueError, "leakage"):
                FeatureData(cache)


if __name__ == "__main__":
    unittest.main()
