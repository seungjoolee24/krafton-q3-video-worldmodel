from __future__ import annotations

import json
import hashlib
import subprocess
from collections import OrderedDict
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from .utils import atomic_json, file_hash


CACHE_VERSION = 1


def read_manifest(dataset: Path):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["resolution"] != 128 or manifest["fps"] != 25:
        raise ValueError("This model expects the kit's 128x128, 25Hz videos")
    ids = [int(row["episode_id"]) for row in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs")
    dev = set(int(item) for item in manifest["dev_subset"])
    if not dev.issubset(ids):
        raise ValueError("Dev IDs are absent from the episode list")
    return manifest, file_hash(path)


def select_rows(manifest, split: str, limit: int, seed: int):
    dev = set(manifest["dev_subset"])
    rows = [row for row in manifest["episodes"]
            if (row["episode_id"] in dev) == (split == "dev")]
    rows = sorted(rows, key=lambda row: row["episode_id"])
    if limit > 0 and limit < len(rows):
        rng = np.random.default_rng(seed + (1 if split == "dev" else 0))
        selected = rng.choice(len(rows), size=limit, replace=False)
        rows = sorted((rows[int(index)] for index in selected), key=lambda row: row["episode_id"])
    return rows


def decode_rgb(path: Path, frames: int | None = None):
    arguments = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                 "-i", str(path)]
    if frames is not None:
        arguments.extend(["-frames:v", str(frames)])
    arguments.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    process = subprocess.run(arguments, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace")[:1000])
    pixels = np.frombuffer(process.stdout, dtype=np.uint8)
    if pixels.size % (128 * 128 * 3):
        raise ValueError(f"Unexpected decoded dimensions: {path}")
    return pixels.reshape(-1, 128, 128, 3)


def prepare_cache(kit: Path, cache: Path, stem, device, train_limit: int = 0,
                  dev_limit: int = 0, seed: int = 42, encode_batch: int = 64,
                  rgb_prefix: int = 64):
    """Read only MP4 and manifest. NPZ action files are never opened.

    Latents for all selected frames; raw RGB only for each dev episode's prefix.
    Per-episode metadata enables interrupted preparation to resume safely.
    """
    dataset = kit / "dataset"
    manifest, manifest_hash = read_manifest(dataset)
    stem_hash = file_hash(kit / "stem" / "reference_stem.pt")
    architecture_hash = file_hash(kit / "wam" / "stem.py")
    selected = [(split, row) for split, limit in [("train", train_limit), ("dev", dev_limit)]
                for row in select_rows(manifest, split, limit, seed)]
    provenance = {
        "version": CACHE_VERSION, "manifest_sha256": manifest_hash,
        "stem_sha256": stem_hash, "stem_architecture_sha256": architecture_hash,
        "dtype": "float16", "actions_used": False, "selection_seed": seed,
        "rgb_prefix_frames": rgb_prefix,
        "train_ids": [row["episode_id"] for split, row in selected if split == "train"],
        "dev_ids": [row["episode_id"] for split, row in selected if split == "dev"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    episodes = []
    total_bytes = sum(row["length"] * 48 * 16 * 16 * 2
                      + (min(row["length"], rgb_prefix) * 128 * 128 * 3 if split == "dev" else 0)
                      for split, row in selected)
    import shutil
    available = shutil.disk_usage(cache).free
    existing_bytes = sum((cache / f"ep_{int(row['episode_id']):06d}{suffix}.npy").stat().st_size
                         for split, row in selected
                         for suffix in (["", "_rgb"] if split == "dev" else [""])
                         if (cache / f"ep_{int(row['episode_id']):06d}{suffix}.npy").is_file())
    if available + existing_bytes < total_bytes + 128 * 1024 ** 2:
        raise OSError("Insufficient cache disk space. Use the pilot profile or a larger runtime disk.")
    print(f"Selected {len(selected)} videos; final cache ~{total_bytes / 1e9:.2f} GB; "
          f"disk free {available / 1e9:.2f} GB", flush=True)
    stem = stem.to(device).eval()
    for number, (split, row) in enumerate(selected, 1):
        eid, length = int(row["episode_id"]), int(row["length"])
        video = dataset / row["video"]
        if not video.is_file():
            raise FileNotFoundError(video)
        video_hash = file_hash(video)
        stem_name = f"ep_{eid:06d}"
        latent_path = cache / f"{stem_name}.npy"
        rgb_path = cache / f"{stem_name}_rgb.npy"
        metadata_path = cache / f"{stem_name}.json"
        expected = {
            "stem_sha256": stem_hash, "stem_architecture_sha256": architecture_hash,
            "video_sha256": video_hash, "length": length,
            "rgb_frames": min(length, rgb_prefix) if split == "dev" else 0,
        }
        ready = False
        if metadata_path.is_file() and latent_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
                array = np.load(latent_path, mmap_mode="r", allow_pickle=False)
                ready = metadata == expected and array.shape == (length, 48, 16, 16) and array.dtype == np.float16
                del array
                if split == "dev":
                    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
                    ready = ready and rgb.shape == (expected["rgb_frames"], 128, 128, 3) and rgb.dtype == np.uint8
                    del rgb
            except (ValueError, OSError, json.JSONDecodeError):
                ready = False
        if not ready:
            frames = decode_rgb(video)
            if len(frames) != length:
                raise ValueError(f"Episode {eid}: manifest {length} != decoded {len(frames)}")
            temporary = latent_path.with_name(latent_path.stem + "_tmp.npy")
            memory = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                             shape=(length, 48, 16, 16))
            with torch.inference_mode():
                for start in range(0, length, encode_batch):
                    batch = torch.from_numpy(np.array(frames[start:start + encode_batch], copy=True))
                    batch = batch.permute(0, 3, 1, 2).to(device, dtype=torch.float32).div_(255)
                    features = stem.encoder(batch)
                    if not torch.isfinite(features).all():
                        raise FloatingPointError("Non-finite stem feature")
                    memory[start:start + len(batch)] = features.cpu().numpy().astype(np.float16)
            memory.flush()
            del memory
            temporary.replace(latent_path)
            if split == "dev":
                np.save(rgb_path, frames[:rgb_prefix], allow_pickle=False)
            atomic_json(metadata_path, expected)
            del frames
        episodes.append({"episode_id": eid, "split": split, "length": length,
                         "latent": latent_path.name,
                         "rgb_prefix": rgb_path.name if split == "dev" else None})
        index = {**provenance, "fingerprint": fingerprint, "episodes": episodes,
                 "complete": number == len(selected)}
        atomic_json(cache / "index.json", index)
        print(f"[{number}/{len(selected)}] {split} episode {eid}: "
              f"{'cached' if ready else 'encoded'} {length} frames", flush=True)
    return index


class FeatureData:
    def __init__(self, cache: Path, context: int = 32, max_open: int = 8):
        self.cache = cache
        self.index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        if not self.index.get("complete") or self.index.get("actions_used") is not False:
            raise ValueError("Cache incomplete or not a video-only cache")
        self.context, self.max_open = context, max_open
        self.train = [row for row in self.index["episodes"] if row["split"] == "train"]
        self.dev = [row for row in self.index["episodes"] if row["split"] == "dev"]
        train_ids = {row["episode_id"] for row in self.train}
        dev_ids = {row["episode_id"] for row in self.dev}
        if train_ids & dev_ids:
            raise ValueError("Episode leakage between train and dev")
        if not self.train or not self.dev:
            raise ValueError("Need at least one train and one dev episode")
        self._open = OrderedDict()

    def close(self):
        for array in self._open.values():
            array._mmap.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def features(self, row):
        key = row["latent"]
        if key not in self._open:
            self._open[key] = np.load(self.cache / key, mmap_mode="r", allow_pickle=False)
        self._open.move_to_end(key)
        while len(self._open) > self.max_open:
            _, old = self._open.popitem(last=False)
            del old
        return self._open[key]

    def sample_batch(self, rng, batch: int, horizon: int, start_probability: float):
        window = self.context + horizon
        eligible = [row for row in self.train if row["length"] >= window]
        if not eligible:
            raise ValueError(f"No train episode has {window} frames")
        samples = []
        for _ in range(batch):
            row = eligible[int(rng.integers(len(eligible)))]
            start = 0 if rng.random() < start_probability else int(rng.integers(row["length"] - window + 1))
            samples.append(np.array(self.features(row)[start:start + window], dtype=np.float32, copy=True))
        return torch.from_numpy(np.stack(samples))

    def dev_window(self, row, horizon: int):
        stop = self.context + horizon
        if row["length"] < stop:
            raise ValueError("Dev episode too short")
        return torch.from_numpy(np.array(self.features(row)[:stop], dtype=np.float32, copy=True)).unsqueeze(0)

    def raw_dev(self, row, horizon: int):
        if not row["rgb_prefix"]:
            raise ValueError("Missing RGB validation prefix")
        frames = np.load(self.cache / row["rgb_prefix"], mmap_mode="r", allow_pickle=False)
        if len(frames) < self.context + horizon:
            raise ValueError("Rebuild cache with a longer --rgb-prefix")
        return np.array(frames[self.context:self.context + horizon], copy=True)
