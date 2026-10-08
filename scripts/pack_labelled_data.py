"""Bundle every labelled MP4/action sidecar and the common kit for Colab.

Unlabelled media are excluded; the unchanged full manifest preserves the
official episode split. ZIP_STORED is intentional: MP4 is already compressed
and the existing safe unpack_kit.py accepts ZIP archives.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

from action_wam.data import dataset_path, read_actions, select_labelled_rows
from video_wam.data import read_manifest
from video_wam.utils import atomic_json, file_hash


def bundle_paths(kit: Path, manifest: dict) -> list[Path]:
    paths = {Path("dataset/manifest.json")}
    # Shared starter APIs, runtime contract and stem weights remain available.
    for name in ("wam", "runtime", "stem"):
        directory = kit / name
        if directory.is_dir():
            paths.update(path.relative_to(kit) for path in directory.rglob("*")
                         if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    for name in ("README.md", "requirements.txt", "pyproject.toml"):
        if (kit / name).is_file():
            paths.add(Path(name))
    for row in select_labelled_rows(manifest, "all"):
        for key in ("video", "sidecar"):
            path = dataset_path(kit / "dataset", row[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.add(path.relative_to(kit.resolve()))
    return sorted(paths, key=lambda path: path.as_posix())


def describe(kit: Path) -> tuple[list[Path], dict]:
    kit = Path(kit).resolve()
    manifest, manifest_hash = read_manifest(kit / "dataset")
    splits = {split: select_labelled_rows(manifest, split) for split in ("train", "dev")}
    rows = select_labelled_rows(manifest, "all")
    if len(rows) != int(manifest.get("n_labelled", len(rows))):
        raise ValueError("Labelled manifest count disagrees with selected rows")
    for row in rows:
        read_actions(dataset_path(kit / "dataset", row["sidecar"]), row)
    paths = bundle_paths(kit, manifest)
    metadata = {
        "format": "labelled-action-bundle/1", "actions_included": True,
        "unlabelled_media_included": False, "manifest_sha256": manifest_hash,
        "action_convention": "a[t] applies between I[t] and I[t+1]",
        "labelled_ids": [int(row["episode_id"]) for row in rows],
        "train_ids": [int(row["episode_id"]) for row in splits["train"]],
        "dev_ids": [int(row["episode_id"]) for row in splits["dev"]],
        "labelled_episodes": len(rows), "frames": sum(int(row["length"]) for row in rows),
        "actions": sum(int(row["length"]) - 1 for row in rows),
        "payload_bytes": sum((kit / path).stat().st_size for path in paths),
        "files": len(paths),
    }
    return paths, metadata


def pack(kit: Path, archive: Path) -> dict:
    kit, archive = Path(kit).resolve(), Path(archive)
    paths, metadata = describe(kit)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(archive.parent).free < metadata["payload_bytes"] + 64 * 1024 ** 2:
        raise OSError("Insufficient space for labelled-only transfer archive")
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as output:
        for path in paths:
            output.write(kit / path, (Path("track3-kit") / path).as_posix())
        output.writestr("track3-kit/LABELLED_BUNDLE.json", json.dumps(metadata, indent=2))
    os.replace(temporary, archive)
    result = {**metadata, "archive": str(archive.resolve()), "size": archive.stat().st_size,
              "sha256": file_hash(archive)}
    atomic_json(archive.with_suffix(archive.suffix + ".json"), result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    if args.estimate:
        print(json.dumps(describe(args.kit)[1], indent=2), flush=True)
    elif args.out is not None:
        pack(args.kit, args.out)
    else:
        parser.error("Provide --out or --estimate")
