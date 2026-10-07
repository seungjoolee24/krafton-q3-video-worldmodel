"""Safely unpack the provided ZIP into the Colab runtime's local disk."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def stage_archive(source: Path, local: Path, expected_sha256: str):
    """Copy a full ZIP or reconstruct its upload-sized parts on the runtime disk."""
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file() and sha256(local) == expected_sha256:
        return local
    temporary = local.with_suffix(".zip.tmp")
    if source.is_file():
        shutil.copy2(source, temporary)
    else:
        manifest_path = source.parent / "track3-kit.parts.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Need {source} or {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["format"] != "split-zip/1" or manifest["original_sha256"] != expected_sha256:
            raise ValueError("Dataset parts manifest does not match the expected original archive")
        with open(temporary, "wb") as destination:
            for part in manifest["parts"]:
                name = part["name"]
                if Path(name).name != name:
                    raise ValueError("Part filename must not contain a path")
                path = source.parent / name
                if path.stat().st_size != part["size"]:
                    raise ValueError(f"Incomplete data part: {name}")
                digest = hashlib.sha256()
                with open(path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                if digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Data part checksum failed: {name}")
                print(f"Joined {name}", flush=True)
        if temporary.stat().st_size != manifest["original_size"]:
            raise ValueError("Reconstructed archive has the wrong size")
    if sha256(temporary) != expected_sha256:
        raise ValueError("Original dataset archive SHA-256 mismatch")
    temporary.replace(local)
    return local


def unpack(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(archive, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    marker = destination / ".unpacked.json"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        kit = Path(previous["kit"])
        if previous["sha256"] == digest.hexdigest() and (kit / "dataset" / "manifest.json").is_file() and (kit / "stem" / "reference_stem.pt").is_file():
            print(kit)
            return kit
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive path escapes destination: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are unsupported")
        stream.extractall(root)
    candidates = [path.parent.parent for path in root.rglob("dataset/manifest.json")]
    candidates = [path for path in candidates if (path / "wam" / "stem.py").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one Track 3 kit; found {len(candidates)}")
    kit = candidates[0]
    marker.write_text(json.dumps({"sha256": digest.hexdigest(), "kit": str(kit)}), encoding="utf-8")
    print(kit)
    return kit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    unpack(args.archive, args.out)
