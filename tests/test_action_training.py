"""Synthetic labelled-only training integration and exact continuation checks."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from action_wam.data import LabelledFeatureData, prepare_cache
from action_wam.evaluation import evaluate
from action_wam.model import ActionDifferenceWorldModel, ActionModelConfig
from action_wam.training import EpisodeCoverageSampler, train
from test_action_data import TinyStem as EncoderStem, fake_kit, frames_for_path


class TinyDecoder(torch.nn.Module):
    def forward(self, features):
        return torch.nn.functional.interpolate((features[:, :3] + 1) / 2, scale_factor=8)


class TinyStem(EncoderStem):
    def __init__(self):
        super().__init__()
        self.decoder = TinyDecoder()


def colored_frames_for_path(path):
    # Reuse the labelled-only fixture and give its mask a valid colored object.
    frames = frames_for_path(path)
    frames[:, 16:32, 16:32] = [255, 64, 16]
    return frames


def tiny_config():
    return {"model": {"hidden_channels": 16, "context": 4, "difference_lags": [1, 2]},
            "labelled_only": True, "seed": 17, "batch_size": 2, "total_steps": 3,
            "horizon_schedule": [[0, 1], [1, 2]], "learning_rate": 0.0003,
            "weight_decay": 0.0001, "grad_clip": 1.0, "mixed_precision": False,
            "teacher_weight": 0.25, "motion_weight": 1.0, "pixel_weight": 0.0,
            "edge_weight": 0.0, "rgb_loss_frames": 2, "start_probability": 0.5,
            "log_every": 1, "checkpoint_every": 1, "eval_every": 3,
            "eval_horizon": 2, "preview_episodes": 0, "evaluate_at_start": False}


class ActionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_nested_equal(self, first, second, path="state"):
        self.assertIs(type(first), type(second), path)
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second), path)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys(), path)
            for key in first:
                self.assert_nested_equal(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second), path)
            for index, (a, b) in enumerate(zip(first, second)):
                self.assert_nested_equal(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(first, second, path)

    def test_uninterrupted_and_resumed_updates_have_exact_model_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, cache = root / "kit", root / "cache"
            fake_kit(kit)
            config = tiny_config()
            with patch("action_wam.data.decode_rgb", side_effect=colored_frames_for_path), \
                    contextlib.redirect_stdout(io.StringIO()):
                index = prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), encode_batch=4)
            self.assertEqual(index["labelled_ids"], [2, 3])
            self.assertFalse((cache / "ep_000001_features.npy").exists())
            with patch("action_wam.training.load_reference_stem", side_effect=lambda kit: TinyStem()), \
                    contextlib.redirect_stdout(io.StringIO()):
                full = train(kit, cache, root / "full", config, torch.device("cpu"))
                train(kit, cache, root / "partial", config, torch.device("cpu"), max_steps=2,
                      persist=root / "drive")
                original_history = (root / "partial/training.jsonl").read_bytes()
                self.assertEqual((root / "drive/training.jsonl").read_bytes(), original_history)
                resumed = train(kit, cache, root / "resumed", config, torch.device("cpu"),
                                resume=root / "drive/latest.pt", persist=root / "drive")
            for key in full.state_dict():
                self.assertTrue(torch.equal(full.state_dict()[key], resumed.state_dict()[key]), key)
            full_saved = torch.load(root / "full/final.pt", weights_only=True)
            resumed_saved = torch.load(root / "resumed/final.pt", weights_only=True)
            for key in ["model", "optimizer", "scaler", "rng", "episode_coverage"]:
                self.assert_nested_equal(full_saved[key], resumed_saved[key], key)
            self.assertEqual(resumed_saved["step"], 3)
            self.assertTrue(resumed_saved["actions_used"])
            self.assertTrue(resumed_saved["labelled_only"])
            self.assertEqual(resumed_saved["episode_coverage"]["seen"], [2])
            report = json.loads((root / "resumed/completion.json").read_text())
            self.assertEqual(report["starting_step"], 2)
            self.assertEqual(report["actual_updates"], 1)
            self.assertEqual(report["reason"], "step_budget")
            self.assertEqual(report["train_episodes_seen"], report["train_episodes_total"])
            self.assertEqual(report["dev_episodes"], 1)
            metrics = json.loads((root / "resumed/validation/step_000003/metrics.json").read_text())
            self.assertIsNone(metrics["official_score"])
            resumed_history = (root / "resumed/training.jsonl").read_bytes()
            self.assertTrue(resumed_history.startswith(original_history))
            records = [json.loads(line) for line in resumed_history.splitlines()]
            self.assertEqual([row["step"] for row in records], [1, 2, 3])
            self.assertEqual((root / "drive/training.jsonl").read_bytes(), resumed_history)
            self.assertEqual((root / "partial/training.jsonl").read_bytes(), original_history)
            # Source labels changing after cache construction cannot silently train
            # with the stale cached action sequence, even when the NPZ stays valid.
            np.savez(kit / "dataset/ep_000002.npz", episode_id=np.int64(2),
                     length=np.int64(10), labelled=np.bool_(True),
                     actions=-np.arange(9, dtype=np.float32) / 10)
            with self.assertRaisesRegex(ValueError, "Source action labels changed"):
                train(kit, cache, root / "rejected", config, torch.device("cpu"))

    def test_episode_coverage_passes_and_restored_sampler_are_exact(self):
        ids = list(range(17))
        rng = np.random.default_rng(11)
        sampler = EpisodeCoverageSampler(ids, rng)
        emitted = sampler.next_batch(7) + sampler.next_batch(15) + sampler.next_batch(12)
        self.assertEqual(sorted(emitted[:17]), ids)
        self.assertEqual(sorted(emitted[17:34]), ids)
        self.assertEqual(sampler.seen, set(ids))
        state, random_state = sampler.state_dict(), copy.deepcopy(rng.bit_generator.state)
        expected = sampler.next_batch(23)
        resumed_rng = np.random.default_rng(0)
        resumed_rng.bit_generator.state = random_state
        resumed = EpisodeCoverageSampler(ids, resumed_rng)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.next_batch(23), expected)
        self.assertEqual(resumed.state_dict(), sampler.state_dict())
        with self.assertRaises(ValueError):
            EpisodeCoverageSampler(ids[:-1], np.random.default_rng(0)).load_state_dict(state)

    def test_evaluation_uses_only_observed_prefix_and_actual_future_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, cache = root / "kit", root / "cache"
            fake_kit(kit)
            with patch("action_wam.data.decode_rgb", side_effect=colored_frames_for_path), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), encode_batch=4)
            model = ActionDifferenceWorldModel(ActionModelConfig(
                hidden_channels=16, context=4, difference_lags=(1, 2)))
            model.train()
            calls = []
            forecast = model.forecast

            def record(context, past_actions, future_actions):
                calls.append((context.clone(), past_actions.clone(), future_actions.clone()))
                return forecast(context, past_actions, future_actions)

            with LabelledFeatureData(cache, context=4) as data:
                batch = data.dev_window(data.dev[0], horizon=2)
                with patch.object(model, "forecast", side_effect=record):
                    result = evaluate(model, TinyStem(), data, torch.device("cpu"),
                                      out=None, horizon=2, previews=0)
            self.assertTrue(model.training, "Evaluation should restore the previous training mode")
            self.assertEqual(len(calls), 2)
            context, past, future = calls[0]
            self.assertEqual(context.shape[1], 4)
            self.assertTrue(torch.equal(context, batch["grids"][:, :4]))
            self.assertTrue(torch.equal(past, batch["actions"][:, :3]))
            self.assertTrue(torch.equal(future, batch["actions"][:, 3:5]))
            self.assertTrue(torch.equal(calls[1][0], context))
            self.assertTrue(torch.equal(calls[1][1], past))
            self.assertFalse(torch.equal(calls[1][2], future))
            self.assertIsNone(result["official_score"])
            self.assertTrue(result["actions_used"])
            self.assertEqual(result["episode_ids"], [3])
            self.assertIn("counterfactual", result["note"])


if __name__ == "__main__":
    unittest.main()
