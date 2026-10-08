"""Causal action timing, multiframe differences and differentiable RGB supervision."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import torch

from action_wam.model import ActionDifferenceWorldModel, ActionModelConfig
from action_wam.objective import _foreground_weights, training_objective


class TinyFrozenStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Conv2d(48, 3, 1), torch.nn.Sigmoid())
        self.requires_grad_(False)


class ActionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.model = ActionDifferenceWorldModel(ActionModelConfig(
            hidden_channels=16, action_embedding=4, context=4, difference_lags=(1, 2, 3)))
        self.grids = torch.rand(2, 7, 48, 4, 4) * 0.5 - 0.25
        self.actions = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                     [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]])

    def enable_transition(self):
        # Copy-last initialization intentionally has no action effect. Once the output
        # layer learns, the causal tests must exercise a nontrivial transition.
        torch.nn.init.normal_(self.model.transition[-1].weight, std=0.01)

    def test_observed_differences_are_causal_and_cover_multiple_intervals(self):
        history = torch.arange(4.0).reshape(1, 4, 1, 1, 1).expand(1, 4, 48, 2, 2)
        differences = self.model.feature_differences(history)
        for difference, expected in zip(differences, (1.0, 2.0, 3.0)):
            self.assertTrue(torch.equal(difference, torch.full_like(difference, expected)))
        short = self.model.feature_differences(history[:, :2])
        self.assertTrue(torch.equal(short[0], torch.ones_like(short[0])))
        self.assertEqual(short[1].abs().sum().item(), 0)
        self.assertEqual(short[2].abs().sum().item(), 0)

    def test_context_observations_receive_the_preceding_action_without_skips(self):
        received = []
        observe = self.model.observe_grid

        def record(state, grid, previous_action):
            received.append((grid.detach().clone(), previous_action.detach().clone()))
            return observe(state, grid, previous_action)

        with patch.object(self.model, "observe_grid", side_effect=record):
            state = self.model.encode_context(self.grids[:, :4], self.actions[:, :3])
        self.assertEqual(len(received), 3)
        for t, (grid, action) in enumerate(received, start=1):
            self.assertTrue(torch.equal(grid, self.grids[:, t]))
            self.assertTrue(torch.equal(action, self.actions[:, t - 1]))
        self.assertEqual(state.history.shape[1], 4)
        self.assertTrue(torch.equal(state.grid, self.grids[:, 3]))

    def test_first_future_action_and_rollout_state_are_aligned(self):
        received = []
        step = self.model.step

        def record(state, action):
            predicted, next_state = step(state, action)
            received.append((action.detach().clone(), predicted.detach().clone(), next_state))
            return predicted, next_state

        with patch.object(self.model, "step", side_effect=record):
            training_objective(self.model, self.grids, self.actions, horizon=3,
                               teacher_weight=0, pixel_weight=0, edge_weight=0)
        self.assertEqual(len(received), 3)
        for i, (action, predicted, state) in enumerate(received):
            self.assertTrue(torch.equal(action, self.actions[:, 3 + i]))
            self.assertTrue(torch.equal(state.grid, predicted))
            self.assertLessEqual(state.history.shape[1], 4)

    def test_true_future_images_never_enter_autonomous_rollout(self):
        self.enable_transition()
        predictions = []
        step = self.model.step

        def record(state, action):
            output = step(state, action)
            predictions.append(output[0].detach().clone())
            return output

        altered = self.grids.clone()
        altered[:, 4:] = -altered[:, 4:] + 0.1
        losses = []
        with patch.object(self.model, "step", side_effect=record):
            for grids in (self.grids, altered):
                loss, _ = training_objective(self.model, grids, self.actions, horizon=3,
                                             teacher_weight=0.25, pixel_weight=0, edge_weight=0)
                losses.append(loss.item())
        for original, changed in zip(predictions[:3], predictions[3:]):
            self.assertTrue(torch.equal(original, changed))
        self.assertNotEqual(losses[0], losses[1], "Future targets should affect losses only")

    def test_future_actions_change_predictions_but_not_earlier_predictions(self):
        self.enable_transition()
        future = self.actions[:, 3:].clone()
        changed = future.clone()
        changed[:, 1] = -1
        baseline = self.model.forecast(self.grids[:, :4], self.actions[:, :3], future)
        counterfactual = self.model.forecast(self.grids[:, :4], self.actions[:, :3], changed)
        self.assertTrue(torch.equal(baseline[:, 0], counterfactual[:, 0]))
        self.assertGreater((baseline[:, 1] - counterfactual[:, 1]).abs().max().item(), 1e-6)
        self.assertGreater((baseline[:, 2] - counterfactual[:, 2]).abs().max().item(), 1e-6)
        self.assertTrue(torch.isfinite(counterfactual).all())
        self.assertLessEqual(counterfactual.abs().max().item(), 1)

    def test_past_actions_affect_recurrent_state(self):
        first = self.model.encode_context(self.grids[:, :4], self.actions[:, :3])
        second = self.model.encode_context(self.grids[:, :4], -self.actions[:, :3])
        self.assertGreater((first.hidden - second.hidden).abs().max().item(), 1e-6)

    def test_frozen_decoder_still_supplies_gradients_and_sparse_rgb_targets(self):
        self.enable_transition()
        stem = TinyFrozenStem()
        targets = torch.rand(2, 3, 3, 4, 4)
        with patch.object(stem.decoder, "forward", wraps=stem.decoder.forward) as decoder:
            loss, metrics = training_objective(self.model, self.grids, self.actions, horizon=3,
                                               stem=stem, rgb_targets=targets, rgb_frames=2)
        self.assertEqual(decoder.call_count, 2)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["pixel_loss"].item(), 0)
        self.assertGreater(metrics["edge_loss"].item(), 0)
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(self.model.action_embed[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.observation[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in stem.parameters()))

    def test_alignment_and_target_shape_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.encode_context(self.grids[:, :4], self.actions[:, :4])
        with self.assertRaises(ValueError):
            self.model.forecast(self.grids[:, :5], self.actions[:, :4], self.actions[:, 4:])
        with self.assertRaises(ValueError):
            training_objective(self.model, self.grids, self.actions[:, :-1], horizon=3,
                               pixel_weight=0, edge_weight=0)
        with self.assertRaises(ValueError):
            training_objective(self.model, self.grids, self.actions, horizon=3,
                               stem=TinyFrozenStem(), rgb_targets=torch.rand(2, 7, 3, 4, 4))

    def test_foreground_heuristic_excludes_gray_and_weak_color_noise(self):
        target = torch.tensor([[[[0.5, 0.21, 0.8]],
                                 [[0.5, 0.12, 0.2]],
                                 [[0.5, 0.12, 0.1]]]])
        self.assertTrue(torch.equal(_foreground_weights(target),
                                    torch.tensor([[[[1.0, 1.0, 5.0]]]])))

    def test_zero_initial_transition_copies_grid_and_config_survives_json(self):
        forecast = self.model.forecast(self.grids[:, :4], self.actions[:, :3], self.actions[:, 3:])
        self.assertTrue(torch.equal(forecast, self.grids[:, 3:4].expand_as(forecast)))
        restored = ActionModelConfig(**json.loads(json.dumps(self.model.config_dict())))
        self.assertEqual(restored, self.model.config)


if __name__ == "__main__":
    unittest.main()
