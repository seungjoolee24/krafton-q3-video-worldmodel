"""Episode-isolated, action-labelled RGB/feature windows.

The action convention is a[t]: I[t] -> I[t+1]. Unlabelled episodes are
excluded before any MP4 or sidecar is opened. Raw RGB targets are retained
for every selected frame, so training windows need not start at frame zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from video_wam.data import decode_rgb, read_manifest
from video_wam.utils import atomic_json, file_hash


CACHE_VERSION = 1
CACHE_MODE = "action-conditioned-difference/1"


def select_labelled_rows(manifest: dict, split: str) -> list[dict]:
    """Select all labelled episodes using the kit's original episode split."""
    if split not in {"train", "dev", "all"}:
        raise ValueError("split must be train, dev or all")
    dev = {int(item) for item in manifest["dev_subset"]}
    selected = [row for row in manifest["episodes"]
                if row.get("labelled") is True
                and (split == "all" or (int(row["episode_id"]) in dev) == (split == "dev"))]
    return sorted(selected, key=lambda row: int(row["episode_id"]))


def dataset_path(dataset: Path, relative: str) -> Path:
    path = (dataset / relative).resolve()
    if not path.is_relative_to(dataset.resolve()):
        raise ValueError(f"Dataset path escapes its directory: {relative}")
    return path


def read_actions(path: Path, row: dict) -> np.ndarray:
    """Validate a sidecar against its manifest before accepting action labels."""
    if row.get("labelled") is not True:
        raise ValueError("Unlabelled episode cannot supply supervised actions")
    length, eid = int(row["length"]), int(row["episode_id"])
    if length < 2:
        raise ValueError(f"Episode {eid} must have at least two frames")
    with np.load(path, allow_pickle=False) as sidecar:
        required = {"episode_id", "length", "labelled", "actions"}
        if not required.issubset(sidecar.files):
            raise ValueError(f"Episode {eid}: missing labelled-sidecar fields")
        for key, expected in (("episode_id", eid), ("length", length), ("labelled", True)):
            value = sidecar[key]
            if value.shape != () or value.item() != expected:
                raise ValueError(f"Episode {eid}: sidecar {key} disagrees with manifest")
        actions = sidecar["actions"]
        if actions.shape != (length - 1,) or actions.dtype != np.float32:
            raise ValueError(f"Episode {eid}: actions must be float32 with shape ({length - 1},)")
        if not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
            raise ValueError(f"Episode {eid}: actions must be finite and in [-1, 1]")
        return np.array(actions, dtype=np.float32, copy=True)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _ready(metadata_path: Path, expected: dict, paths: dict, length: int) -> bool:
    try:
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            return False
        specification = {"latent": ((length, 48, 16, 16), np.float16),
                         "rgb": ((length, 128, 128, 3), np.uint8),
                         "actions": ((length - 1,), np.float32)}
        for key, (shape, dtype) in specification.items():
            array = np.load(paths[key], mmap_mode="r", allow_pickle=False)
            valid = array.shape == shape and array.dtype == dtype
            if key == "actions":
                valid = valid and np.isfinite(array).all() and not (np.abs(array) > 1).any() \
                    and hashlib.sha256(array.tobytes()).hexdigest() == expected["actions_array_sha256"]
            array._mmap.close()
            if not valid:
                return False
        return True
    except (OSError, ValueError, EOFError, json.JSONDecodeError):
        return False


def prepare_cache(kit: Path, cache: Path, stem, device, encode_batch: int = 64) -> dict:
    """Encode all and only labelled episodes; atomically resume per episode.

    No train/dev limits exist for this pipeline: all labelled episodes are
    represented. The fixed original dev episodes never become train windows.
    Cache identity includes every action sidecar and the complete split.
    """
    kit, cache = Path(kit), Path(cache)
    if encode_batch < 1:
        raise ValueError("encode_batch must be positive")
    dataset = kit / "dataset"
    manifest, manifest_hash = read_manifest(dataset)
    selected = [(split, row) for split in ("train", "dev")
                for row in select_labelled_rows(manifest, split)]
    if not any(split == "train" for split, _ in selected) or not any(split == "dev" for split, _ in selected):
        raise ValueError("Need labelled episodes in both train and dev splits")
    stem_hash = file_hash(kit / "stem" / "reference_stem.pt")
    architecture_hash = file_hash(kit / "wam" / "stem.py")
    sidecar_hashes = {}
    video_hashes = {}
    actions_by_id = {}
    # Validate all labels before spending GPU time or marking a cache complete.
    for _, row in selected:
        path = dataset_path(dataset, row["sidecar"])
        actions_by_id[int(row["episode_id"])] = read_actions(path, row)
        sidecar_hashes[str(int(row["episode_id"]))] = file_hash(path)
        video_hashes[str(int(row["episode_id"]))] = file_hash(dataset_path(dataset, row["video"]))
    provenance = {
        "version": CACHE_VERSION, "mode": CACHE_MODE, "actions_used": True, "labelled_only": True,
        "action_convention": "a[t] applies between I[t] and I[t+1]",
        "manifest_sha256": manifest_hash, "stem_sha256": stem_hash,
        "stem_architecture_sha256": architecture_hash,
        "sidecar_sha256": sidecar_hashes, "video_sha256": video_hashes, "feature_dtype": "float16",
        "rgb_dtype": "uint8", "action_dtype": "float32", "rgb_storage": "all_frames",
        "train_ids": [int(row["episode_id"]) for split, row in selected if split == "train"],
        "dev_ids": [int(row["episode_id"]) for split, row in selected if split == "dev"],
        "labelled_ids": sorted(int(row["episode_id"]) for _, row in selected),
    }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(int(row["length"]) * (48 * 16 * 16 * 2 + 128 * 128 * 3)
                      + (int(row["length"]) - 1) * 4 for _, row in selected)
    existing_bytes = sum(path.stat().st_size for _, row in selected
                         for suffix in ("_features.npy", "_rgb.npy", "_actions.npy")
                         if (path := cache / f"ep_{int(row['episode_id']):06d}{suffix}").is_file())
    available = shutil.disk_usage(cache).free
    largest_episode = max(int(row["length"]) for _, row in selected)
    reserve = max(256 * 1024 ** 2, largest_episode * (48 * 16 * 16 * 2 + 128 * 128 * 3))
    if available + existing_bytes < total_bytes + reserve:
        raise OSError(f"Labelled RGB+feature cache needs approximately {total_bytes / 1e9:.2f} GB")
    print(f"Labelled only: {len(provenance['train_ids'])} train + {len(provenance['dev_ids'])} dev; "
          f"cache ~{total_bytes / 1e9:.2f} GB; free {available / 1e9:.2f} GB", flush=True)
    episodes = []
    atomic_json(cache / "index.json", {**provenance, "fingerprint": fingerprint,
                                      "episodes": [], "complete": False})
    stem = stem.to(device).eval()
    for number, (split, row) in enumerate(selected, 1):
        eid, length = int(row["episode_id"]), int(row["length"])
        video = dataset_path(dataset, row["video"])
        names = {"latent": f"ep_{eid:06d}_features.npy", "rgb": f"ep_{eid:06d}_rgb.npy",
                 "actions": f"ep_{eid:06d}_actions.npy"}
        paths = {key: cache / name for key, name in names.items()}
        metadata_path = cache / f"ep_{eid:06d}.json"
        action_hash = hashlib.sha256(actions_by_id[eid].tobytes()).hexdigest()
        expected = {"version": CACHE_VERSION, "mode": CACHE_MODE,
                    "manifest_sha256": manifest_hash, "stem_sha256": stem_hash,
                    "stem_architecture_sha256": architecture_hash,
                    "video_sha256": video_hashes[str(eid)], "sidecar_sha256": sidecar_hashes[str(eid)],
                    "actions_array_sha256": action_hash,
                    "length": length, "episode_id": eid, "split": split, "rgb_frames": length}
        ready = _ready(metadata_path, expected, paths, length)
        if not ready:
            frames = decode_rgb(video)
            if len(frames) != length:
                raise ValueError(f"Episode {eid}: manifest length {length} != decoded {len(frames)}")
            temporary = paths["latent"].with_name(paths["latent"].name + ".tmp")
            memory = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                              shape=(length, 48, 16, 16))
            with torch.inference_mode():
                for start in range(0, length, encode_batch):
                    batch = torch.from_numpy(np.array(frames[start:start + encode_batch], copy=True))
                    batch = batch.permute(0, 3, 1, 2).to(device, dtype=torch.float32).div_(255)
                    features = stem.encoder(batch)
                    if features.shape != (len(batch), 48, 16, 16) or not torch.isfinite(features).all():
                        raise FloatingPointError("Invalid reference-stem feature")
                    stored = features.float().cpu().numpy().astype(np.float16)
                    if not np.isfinite(stored).all():
                        raise FloatingPointError("Feature overflows float16 cache")
                    memory[start:start + len(batch)] = stored
            memory.flush()
            del memory
            os.replace(temporary, paths["latent"])
            _atomic_npy(paths["rgb"], frames)
            _atomic_npy(paths["actions"], actions_by_id[eid])
            atomic_json(metadata_path, expected)
            del frames
        episodes.append({"episode_id": eid, "split": split, "labelled": True,
                         "length": length, "actions_sha256": action_hash, **names})
        index = {**provenance, "fingerprint": fingerprint, "episodes": episodes,
                 "complete": number == len(selected), "estimated_array_bytes": total_bytes}
        atomic_json(cache / "index.json", index)
        print(f"[{number}/{len(selected)}] {split} episode {eid}: "
              f"{'cached' if ready else 'encoded'} {length} frames, {length - 1} actions", flush=True)
    return index


class LabelledFeatureData:
    def __init__(self, cache: Path, context: int = 32, max_open: int = 8):
        self.cache = Path(cache)
        self.index = json.loads((self.cache / "index.json").read_text(encoding="utf-8"))
        if not self.index.get("complete") or self.index.get("actions_used") is not True \
                or self.index.get("labelled_only") is not True \
                or self.index.get("mode") != CACHE_MODE or self.index.get("rgb_storage") != "all_frames":
            raise ValueError("Need a complete labelled-action cache with all RGB frames")
        if context < 2 or max_open < 1:
            raise ValueError("context must be at least two and max_open must be positive")
        self.context, self.max_open = context, max_open
        self.train = [row for row in self.index["episodes"] if row["split"] == "train"]
        self.dev = [row for row in self.index["episodes"] if row["split"] == "dev"]
        ids = [int(row["episode_id"]) for row in self.index["episodes"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Episode leakage or duplicate IDs in cache")
        if not self.train or not self.dev or any(row.get("labelled") is not True for row in self.train + self.dev):
            raise ValueError("Need nonempty, entirely labelled train and dev splits")
        if sorted(int(row["episode_id"]) for row in self.train) != sorted(self.index["train_ids"]) \
                or sorted(int(row["episode_id"]) for row in self.dev) != sorted(self.index["dev_ids"]):
            raise ValueError("Cache episode split disagrees with provenance")
        excluded = {"fingerprint", "episodes", "complete", "estimated_array_bytes"}
        provenance = {key: value for key, value in self.index.items() if key not in excluded}
        actual_fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        if actual_fingerprint != self.index.get("fingerprint"):
            raise ValueError("Cache provenance fingerprint mismatch")
        if sorted(ids) != self.index["labelled_ids"]:
            raise ValueError("Cache does not contain every selected labelled episode")
        # Action payloads are small (~0.55 MB for the full kit), so validate all
        # at startup. Do not silently train against a finite but altered action.
        for row in self.train + self.dev:
            eid = int(row["episode_id"])
            metadata = json.loads((self.cache / f"ep_{eid:06d}.json").read_text(encoding="utf-8"))
            expected = {"version": CACHE_VERSION, "mode": CACHE_MODE,
                        "manifest_sha256": self.index["manifest_sha256"],
                        "stem_sha256": self.index["stem_sha256"],
                        "stem_architecture_sha256": self.index["stem_architecture_sha256"],
                        "sidecar_sha256": self.index["sidecar_sha256"][str(eid)],
                        "video_sha256": self.index["video_sha256"][str(eid)],
                        "length": int(row["length"]), "episode_id": eid,
                        "split": row["split"], "rgb_frames": int(row["length"]),
                        "actions_array_sha256": row["actions_sha256"]}
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Episode {eid}: cached metadata/provenance mismatch")
            actions = np.load(self.cache / row["actions"], allow_pickle=False)
            if actions.shape != (int(row["length"]) - 1,) or actions.dtype != np.float32 \
                    or not np.isfinite(actions).all() or (np.abs(actions) > 1).any() \
                    or hashlib.sha256(actions.tobytes()).hexdigest() != row["actions_sha256"]:
                raise ValueError(f"Episode {eid}: cached action payload integrity failure")
        self._open = OrderedDict()

    def close(self):
        for array in self._open.values():
            array._mmap.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _array(self, row, key):
        name = row[key]
        if name not in self._open:
            self._open[name] = np.load(self.cache / name, mmap_mode="r", allow_pickle=False)
        self._open.move_to_end(name)
        while len(self._open) > self.max_open:
            _, old = self._open.popitem(last=False)
            old._mmap.close()
        return self._open[name]

    def features(self, row):
        return self._array(row, "latent")

    def _window(self, row, start: int, horizon: int) -> dict:
        stop = start + self.context + horizon
        if horizon < 1 or start < 0 or stop > int(row["length"]):
            raise ValueError("Invalid action/frame window")
        grids = np.array(self._array(row, "latent")[start:stop], dtype=np.float32, copy=True)
        actions = np.array(self._array(row, "actions")[start:stop - 1], dtype=np.float32, copy=True)
        rgb = np.array(self._array(row, "rgb")[start + self.context:stop], copy=True)
        if grids.shape != (self.context + horizon, 48, 16, 16) \
                or actions.shape != (self.context + horizon - 1,) \
                or rgb.shape != (horizon, 128, 128, 3):
            raise ValueError("Corrupt cached window dimensions")
        return {"grids": torch.from_numpy(grids), "actions": torch.from_numpy(actions),
                "future_rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255),
                "episode_id": int(row["episode_id"]), "start": int(start)}

    @staticmethod
    def _stack(samples: list[dict]) -> dict:
        return {key: torch.stack([sample[key] for sample in samples])
                for key in ("grids", "actions", "future_rgb")} | {
                    "episode_ids": [sample["episode_id"] for sample in samples],
                    "starts": [sample["start"] for sample in samples]}

    def sample_batch(self, rng, batch: int, horizon: int, start_probability: float = 0.25,
                     episode_ids: list[int] | None = None) -> dict:
        if batch < 1 or horizon < 1 or not 0 <= start_probability <= 1:
            raise ValueError("Invalid batch, horizon or start_probability")
        window = self.context + horizon
        eligible = {int(row["episode_id"]): row for row in self.train if row["length"] >= window}
        if not eligible:
            raise ValueError(f"No labelled train episode has {window} frames")
        if episode_ids is not None and (len(episode_ids) != batch or any(int(eid) not in eligible for eid in episode_ids)):
            raise ValueError("Coverage episode_ids must contain one eligible train ID per batch slot")
        rows = list(eligible.values())
        samples = []
        for slot in range(batch):
            row = eligible[int(episode_ids[slot])] if episode_ids is not None else rows[int(rng.integers(len(rows)))]
            start = 0 if rng.random() < start_probability else int(rng.integers(row["length"] - window + 1))
            samples.append(self._window(row, start, horizon))
        return self._stack(samples)

    def dev_window(self, row, horizon: int = 32) -> dict:
        if row["split"] != "dev":
            raise ValueError("dev_window requires a held-out dev episode")
        return self._stack([self._window(row, 0, horizon)])

    def raw_dev(self, row, horizon: int = 32) -> np.ndarray:
        if row["split"] != "dev" or horizon < 1 or row["length"] < self.context + horizon:
            raise ValueError("Invalid held-out RGB window")
        return np.array(self._array(row, "rgb")[self.context:self.context + horizon], copy=True)
