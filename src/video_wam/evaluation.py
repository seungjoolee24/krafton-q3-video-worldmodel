from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw

from .utils import atomic_json


def quantize(frames):
    # Same rounding as the kit; numpy float64 avoids boundary rounding differences.
    return (np.clip(np.asarray(frames, dtype=np.float64), 0, 1) * 255 + 0.5).astype(np.uint8)


@torch.inference_mode()
def decode_grids(stem, grids, batch: int = 16):
    flat = grids.reshape(-1, *grids.shape[-3:])
    decoded = []
    for start in range(0, len(flat), batch):
        decoded.append(stem.decoder(flat[start:start + batch].float()).cpu())
    images = torch.cat(decoded).permute(0, 2, 3, 1).numpy()
    return quantize(images)


def colored_object_mask(rgb):
    # Fixed object colours support a diagnostic heuristic, not a supplied ground-truth mask.
    image = np.asarray(rgb, dtype=np.float32) / 255
    saturation = image.max(-1) - image.min(-1)
    return (saturation > 0.15) & (image.max(-1) > 0.20)


def image_errors(predicted, target):
    residual = ((predicted.astype(np.float32) - target.astype(np.float32)) / 255) ** 2
    mse = residual.mean((1, 2, 3))
    mask = colored_object_mask(target)
    count = mask.sum((1, 2))
    foreground = (residual * mask[..., None]).sum((1, 2, 3)) / np.maximum(count * 3, 1)
    foreground = np.where(count > 0, foreground, np.nan)
    return mse, foreground


def write_comparison(path: Path, truth, prediction, baseline):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(path), size=(512, 154), fps=25,
                                        codec="libx264", pix_fmt_in="rgb24",
                                        pix_fmt_out="yuv420p", quality=8,
                                        ffmpeg_log_level="error", macro_block_size=2)
    writer.send(None)
    try:
        for index, (actual, predicted, copied) in enumerate(zip(truth, prediction, baseline)):
            canvas = Image.new("RGB", (512, 154), "#101722")
            difference = np.minimum(np.abs(actual.astype(np.int16) - predicted.astype(np.int16)) * 4, 255).astype(np.uint8)
            for column, (label, pixels) in enumerate(zip(
                    ["Actual", "Video prior", "Copy last", "Difference x4"],
                    [actual, predicted, copied, difference])):
                canvas.paste(Image.fromarray(pixels), (column * 128, 26))
                ImageDraw.Draw(canvas).text((column * 128 + 5, 5), label, fill="white")
            writer.send(np.asarray(canvas))
    finally:
        writer.close()


def summarize_per_horizon(values):
    result = {"mean": float(np.nanmean(values))}
    for horizon in [1, 8, 16, 32]:
        if horizon <= values.shape[1]:
            result[f"h{horizon}"] = float(np.nanmean(values[:, horizon - 1]))
    return result


@torch.inference_mode()
def evaluate(model, stem, data, device, out: Path | None = None,
             horizon: int = 32, limit: int = 0, previews: int = 0):
    previous_training = model.training
    model.eval()
    stem.eval()
    rows = data.dev[:limit] if limit > 0 else data.dev
    if not rows:
        raise ValueError("No validation episodes")
    per_episode, image_mse, image_fg, copy_mse, copy_fg, reconstruction = [], [], [], [], [], []
    latent_mse, latent_copy, sensitivities = [], [], []
    for row in rows:
        window = data.dev_window(row, horizon).to(device)
        context = window[:, :model.config.context]
        target = window[:, model.config.context:]
        # Only the prior sees available context. The posterior is never called here.
        prediction, codes = model.forecast(context, horizon)
        current, hidden = model.encode_context(context)
        probe = current.new_full((len(current), model.config.latent_action_dim), 0.8)
        sensitivities.append(float((model.predict_grid(current, hidden, probe)
                                    - model.predict_grid(current, hidden, -probe)).square().mean()))
        copied = context[:, -1:].expand_as(target)
        latent_mse.append((prediction - target).square().mean((0, 2, 3, 4)).cpu().numpy())
        latent_copy.append((copied - target).square().mean((0, 2, 3, 4)).cpu().numpy())
        pixels = decode_grids(stem, prediction)
        last = decode_grids(stem, context[:, -1:])
        copied_pixels = np.repeat(last, horizon, axis=0)
        reconstructed_target = decode_grids(stem, target)
        actual = data.raw_dev(row, horizon)
        mse, fg = image_errors(pixels, actual)
        copy_error, copy_object = image_errors(copied_pixels, actual)
        recon_error, _ = image_errors(reconstructed_target, actual)
        image_mse.append(mse)
        image_fg.append(fg)
        copy_mse.append(copy_error)
        copy_fg.append(copy_object)
        reconstruction.append(recon_error)
        per_episode.append({"episode_id": row["episode_id"], "mse": float(mse.mean()),
                            "copy_last_mse": float(copy_error.mean()),
                            "latent_codes": codes[0].cpu().tolist()})
        if out is not None and len(per_episode) <= previews:
            write_comparison(out / f"ep_{row['episode_id']:06d}.mp4", actual, pixels, copied_pixels)
    mse_array = np.stack(image_mse)
    summary = {
        "mode": "video_only_autonomous_prior", "actions_used": False,
        "horizon": horizon, "episodes": len(rows),
        "context": model.config.context,
        "mse": summarize_per_horizon(mse_array),
        "foreground_mse_heuristic": summarize_per_horizon(np.stack(image_fg)),
        "copy_last_mse": summarize_per_horizon(np.stack(copy_mse)),
        "copy_last_foreground_mse_heuristic": summarize_per_horizon(np.stack(copy_fg)),
        "stem_reconstruction_mse": summarize_per_horizon(np.stack(reconstruction)),
        "latent_mse": summarize_per_horizon(np.stack(latent_mse)),
        "copy_last_latent_mse": summarize_per_horizon(np.stack(latent_copy)),
        "psnr_db": float(10 * np.log10(1.0 / max(float(mse_array.mean()), 1e-12))),
        "latent_code_effect_mse": float(np.mean(sensitivities)),
        "official_score": None,
        "note": "Video-only prediction under inferred behaviour, not a test of response to supplied physical forces.",
        "per_episode": per_episode,
    }
    if out is not None:
        atomic_json(out / "metrics.json", summary)
    model.train(previous_training)
    return summary
