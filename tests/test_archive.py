"""Verify that the Drive chunk path preserves the original archive and rejects corruption."""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveTransferTest(unittest.TestCase):
    def test_split_reassemble_and_cached_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.zip"
            payload = bytes(range(256)) * 9000
            original.write_bytes(payload)
            parts = root / "drive"
            manifest = load_script("split_dataset").split(original, parts, chunk_mib=1)
            self.assertEqual(len(manifest["parts"]), 3)
            expected = hashlib.sha256(payload).hexdigest()
            local = root / "runtime" / "kit.zip"
            stage = load_script("unpack_kit").stage_archive
            stage(parts / "missing.zip", local, expected)
            self.assertEqual(local.read_bytes(), payload)
            # A staged, verified archive is reusable even when Drive is unavailable.
            stage(root / "unavailable" / "missing.zip", local, expected)
            self.assertEqual(local.read_bytes(), payload)

    def test_corrupt_part_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.zip"
            payload = b"test archive" * 20
            original.write_bytes(payload)
            parts = root / "drive"
            manifest = load_script("split_dataset").split(original, parts, chunk_mib=1)
            part = parts / manifest["parts"][0]["name"]
            part.write_bytes(b"x" + payload[1:])
            local = root / "runtime" / "kit.zip"
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_script("unpack_kit").stage_archive(
                    parts / "missing.zip", local, manifest["original_sha256"])
            self.assertFalse(local.exists())


if __name__ == "__main__":
    unittest.main()
