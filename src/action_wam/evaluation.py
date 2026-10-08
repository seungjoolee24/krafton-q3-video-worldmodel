from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw

from video_wam.evaluation import decode_grids, image_errors, summarize_per_horizon
from video_wam.utils import atomic_json


def write_comparison(path: Path, truth, prediction, baseline):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path), size=(512, 154), fps=25, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=8, ffmpeg_log_level="error", macro_block_size=2)
    writer.send(None)
    try:
        for actual, predicted, copied in zip(truth, prediction, baseline):
            canvas = Image.new("RGB", (512, 154), "#101722")
            difference = np.minimum(np.abs(actual.astype(np.int16) - predicted.astype(np.int16)) * 4, 255).astype(np.uint8)
            for column, (label, pixels) in enumerate(zip(
                    ["Actual", "Action model", "Copy last", "Difference x4"],
                    [actual, predicted, copied, difference])):
                canvas.paste(Image.fromarray(pixels), (column * 128, 26))
                ImageDraw.Draw(canvas).text((column * 128 + 5, 5), label, fill="white")
            writer.send(np.asarray(canvas))
    finally:
        writer.close()


@torch.inference_mode()
def evaluate(model, stem, data, device, out: Path | None = None,
             horizon: int = 32, previews: int = 3):
    """All labelled dev episodes, open-loop, with their actual future commands.

    Wrong-action and sensitivity diagnostics have no counterfactual ground truth:
    they measure input use, not certified physical response to another command.
    """
    was_training = model.training
    model.eval()
    stem.eval()
    arrays = {key: [] for key in ["mse", "foreground_mse_heuristic", "copy_last_mse",
                                  "copy_last_foreground_mse_heuristic", "stem_reconstruction_mse",
                                  "latent_mse", "wrong_action_mse"]}
    rows, sensitivities = [], []
    context_length = model.config.context
    try:
        for row in data.dev:
            batch = data.dev_window(row, horizon)
            grids = batch["grids"].to(device)
            actions = batch["actions"].to(device)
            context, targets = grids[:, :context_length], grids[:, context_length:]
            past_actions = actions[:, :context_length - 1]
            future_actions = actions[:, context_length - 1:context_length - 1 + horizon]
            predicted = model.forecast(context, past_actions, future_actions)
            wrong_actions = torch.roll(future_actions, shifts=max(1, horizon // 2), dims=1)
            if horizon == 1:
                wrong_actions = -future_actions
            wrong = model.forecast(context, past_actions, wrong_actions)
            state = model.encode_context(context, past_actions)
            positive, negative = future_actions.new_full((len(context),), 0.8), future_actions.new_full((len(context),), -0.8)
            sensitivities.append(float((model.predict_grid(state, positive) - model.predict_grid(state, negative)).square().mean()))
            pixels = decode_grids(stem, predicted)
            wrong_pixels = decode_grids(stem, wrong)
            baseline = np.repeat(decode_grids(stem, context[:, -1:]), horizon, axis=0)
            actual = data.raw_dev(row, horizon)
            mse, fg = image_errors(pixels, actual)
            copy_mse, copy_fg = image_errors(baseline, actual)
            reconstructed_mse, _ = image_errors(decode_grids(stem, targets), actual)
            wrong_mse, _ = image_errors(wrong_pixels, actual)
            arrays["mse"].append(mse)
            arrays["foreground_mse_heuristic"].append(fg)
            arrays["copy_last_mse"].append(copy_mse)
            arrays["copy_last_foreground_mse_heuristic"].append(copy_fg)
            arrays["stem_reconstruction_mse"].append(reconstructed_mse)
            arrays["wrong_action_mse"].append(wrong_mse)
            arrays["latent_mse"].append((predicted - targets).square().mean((0, 2, 3, 4)).cpu().numpy())
            rows.append({"episode_id": row["episode_id"], "mse": float(mse.mean()),
                         "copy_last_mse": float(copy_mse.mean()), "wrong_action_mse": float(wrong_mse.mean())})
            if out is not None and len(rows) <= previews:
                write_comparison(out / f"ep_{row['episode_id']:06d}.mp4", actual, pixels, baseline)
        if not rows:
            raise ValueError("No labelled validation episodes")
        summary = {key: summarize_per_horizon(np.stack(values)) for key, values in arrays.items()}
        summary.update({"mode": "action_conditioned_difference", "actions_used": True,
                        "labelled_only": True, "context": context_length, "horizon": horizon,
                        "episodes": len(rows), "episode_ids": [r["episode_id"] for r in rows],
                        "psnr_db": float(10 * np.log10(1.0 / max(summary["mse"]["mean"], 1e-12))),
                        "action_sensitivity": {"feature_mse_plus_vs_minus_0_8": float(np.mean(sensitivities))},
                        "official_score": None, "per_episode": rows,
                        "note": "Actual future action labels condition open-loop prediction; RGB MSE is a local diagnostic, not the official score. Wrong-action rollouts lack counterfactual ground-truth videos."})
        if out is not None:
            atomic_json(out / "metrics.json", summary)
        return summary
    finally:
        model.train(was_training)
