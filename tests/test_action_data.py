"""Action/frame alignment, labelled-only access and cache provenance checks."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from action_wam.data import LabelledFeatureData, prepare_cache, read_actions, select_labelled_rows
from video_wam.utils import atomic_json


class TinyStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()


class TinyEncoder(torch.nn.Module):
    def forward(self, frames):
        return torch.nn.functional.avg_pool2d(frames, 8).repeat(1, 16, 1, 1) * 2 - 1


def fake_kit(root: Path) -> dict:
    for name in ("dataset", "wam", "stem"):
        (root / name).mkdir(parents=True)
    (root / "stem/reference_stem.pt").write_bytes(b"test-stem")
    (root / "wam/stem.py").write_text("# test architecture\n", encoding="utf-8")
    rows = []
    for eid in (1, 2, 3):
        labelled = eid != 1
        row = {"episode_id": eid, "length": 10, "labelled": labelled,
               "video": f"ep_{eid:06d}.mp4", "sidecar": f"ep_{eid:06d}.npz"}
        (root / "dataset" / row["video"]).write_bytes(b"must-not-decode" if not labelled else bytes([eid]))
        sidecar = root / "dataset" / row["sidecar"]
        if labelled:
            np.savez(sidecar, episode_id=np.int64(eid), length=np.int64(10),
                     labelled=np.bool_(True), actions=np.arange(9, dtype=np.float32) / 10)
        else:
            sidecar.write_bytes(b"must-not-open")
        rows.append(row)
    manifest = {"resolution": 128, "fps": 25, "n_labelled": 2,
                "dev_subset": [3], "episodes": rows}
    atomic_json(root / "dataset/manifest.json", manifest)
    return manifest


def frames_for_path(path: Path) -> np.ndarray:
    if path.name == "ep_000001.mp4":
        raise AssertionError("Unlabelled video was decoded")
    return np.broadcast_to(np.arange(10, dtype=np.uint8)[:, None, None, None],
                           (10, 128, 128, 3)).copy()


class ActionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def prepare(self, kit, cache):
        with patch("action_wam.data.decode_rgb", side_effect=frames_for_path), \
                contextlib.redirect_stdout(io.StringIO()):
            return prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), encode_batch=4)

    def test_only_labelled_media_read_and_windows_align_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            kit, cache = Path(folder) / "kit", Path(folder) / "cache"
            manifest = fake_kit(kit)
            self.assertEqual([row["episode_id"] for row in select_labelled_rows(manifest, "train")], [2])
            self.assertEqual([row["episode_id"] for row in select_labelled_rows(manifest, "dev")], [3])
            index = self.prepare(kit, cache)
            self.assertTrue(index["actions_used"])
            self.assertEqual(index["labelled_ids"], [2, 3])
            self.assertFalse((cache / "ep_000001_features.npy").exists())
            with LabelledFeatureData(cache, context=4, max_open=1) as data:
                # At start 2, observed I[2:6], future I[6:8], actions a[2:7].
                sample = data._window(data.train[0], start=2, horizon=2)
                self.assertEqual(sample["grids"].shape, (6, 48, 16, 16))
                np.testing.assert_allclose(sample["actions"].numpy(), np.arange(2, 7) / 10)
                np.testing.assert_allclose(sample["future_rgb"][:, 0, 0, 0].numpy(), np.arange(6, 8) / 255)
                # a[5] connects last observed frame I[5] to first future I[6].
                self.assertAlmostEqual(float(sample["actions"][3]), 0.5)
                batch = data.sample_batch(np.random.default_rng(3), 2, 2, 1, episode_ids=[2, 2])
                self.assertEqual(batch["episode_ids"], [2, 2])
                self.assertEqual(batch["actions"].shape, (2, 5))
                dev = data.dev_window(data.dev[0], horizon=2)
                self.assertEqual(dev["grids"].shape, (1, 6, 48, 16, 16))
                np.testing.assert_array_equal(data.raw_dev(data.dev[0], 2)[:, 0, 0, 0], [4, 5])
                with self.assertRaisesRegex(ValueError, "eligible train"):
                    data.sample_batch(np.random.default_rng(1), 1, 2, episode_ids=[3])

    def test_resumes_without_redecode_and_invalidates_changed_action_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            kit, cache = Path(folder) / "kit", Path(folder) / "cache"
            fake_kit(kit)
            with patch("action_wam.data.decode_rgb", side_effect=frames_for_path) as decode, \
                    contextlib.redirect_stdout(io.StringIO()):
                first = prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), encode_batch=4)
                again = prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), encode_batch=4)
                self.assertEqual(decode.call_count, 2)
                self.assertEqual(first["fingerprint"], again["fingerprint"])
                np.savez(kit / "dataset/ep_000002.npz", episode_id=np.int64(2), length=np.int64(10),
                         labelled=np.bool_(True), actions=-np.arange(9, dtype=np.float32) / 10)
                changed = prepare_cache(kit, cache, TinyStem(), torch.device("cpu"), encode_batch=4)
                self.assertEqual(decode.call_count, 3)
                self.assertNotEqual(first["fingerprint"], changed["fingerprint"])

    def test_split_change_changes_cache_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            kit, cache = Path(folder) / "kit", Path(folder) / "cache"
            manifest = fake_kit(kit)
            before = self.prepare(kit, cache)
            manifest["dev_subset"] = [2]
            atomic_json(kit / "dataset/manifest.json", manifest)
            after = self.prepare(kit, cache)
            self.assertNotEqual(before["fingerprint"], after["fingerprint"])
            with LabelledFeatureData(cache, context=4) as data:
                self.assertEqual([row["episode_id"] for row in data.train], [3])
                self.assertEqual([row["episode_id"] for row in data.dev], [2])

    def test_bad_action_shape_dtype_range_nan_and_label_flag_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fake_kit(root)
            row = manifest["episodes"][1]
            path = root / "dataset" / row["sidecar"]
            for actions in (np.zeros(10, np.float32), np.zeros(9, np.float64),
                            np.full(9, np.nan, np.float32), np.full(9, 1.1, np.float32)):
                np.savez(path, episode_id=np.int64(2), length=np.int64(10),
                         labelled=np.bool_(True), actions=actions)
                with self.assertRaises(ValueError):
                    read_actions(path, row)
            np.savez(path, episode_id=np.int64(2), length=np.int64(10),
                     labelled=np.bool_(False), actions=np.zeros(9, np.float32))
            with self.assertRaisesRegex(ValueError, "labelled disagrees"):
                read_actions(path, row)

    def test_modified_finite_cache_actions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            kit, cache = Path(folder) / "kit", Path(folder) / "cache"
            fake_kit(kit)
            self.prepare(kit, cache)
            np.save(cache / "ep_000002_actions.npy", np.zeros(9, np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "action payload integrity"):
                LabelledFeatureData(cache, context=4)

    def test_bundle_has_all_labelled_media_and_can_use_existing_safe_unpack(self):
        from scripts.pack_labelled_data import pack
        from scripts.unpack_kit import unpack
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, archive = root / "kit", root / "labelled.zip"
            fake_kit(kit)
            with contextlib.redirect_stdout(io.StringIO()):
                result = pack(kit, archive)
                extracted = unpack(archive, root / "extracted")
            self.assertEqual(result["labelled_episodes"], 2)
            self.assertEqual(result["train_ids"], [2])
            self.assertEqual(result["dev_ids"], [3])
            self.assertEqual(result["actions"], 18)
            with zipfile.ZipFile(archive) as stream:
                names = stream.namelist()
                self.assertNotIn("track3-kit/dataset/ep_000001.mp4", names)
                self.assertNotIn("track3-kit/dataset/ep_000001.npz", names)
                for eid in (2, 3):
                    for suffix in (".mp4", ".npz"):
                        self.assertIn(f"track3-kit/dataset/ep_{eid:06d}{suffix}", names)
            self.assertEqual((extracted / "dataset/ep_000002.npz").read_bytes(),
                             (kit / "dataset/ep_000002.npz").read_bytes())
            self.assertEqual(json.loads(archive.with_suffix(".zip.json").read_text())["sha256"],
                             result["sha256"])


if __name__ == "__main__":
    unittest.main()
