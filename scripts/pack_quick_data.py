"""Bundle only the original videos selected for the T4 first-validation profile."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from video_wam.data import read_manifest, select_rows
from video_wam.utils import file_hash


def pack(kit: Path, config_path: Path, archive: Path):
    config = json.loads(config_path.read_text())
    manifest, _ = read_manifest(kit / "dataset")
    splits = {split: select_rows(manifest, split, config[f"{split}_limit"], config["selection_seed"])
              for split in ["train", "dev"]}
    paths = [Path("dataset/manifest.json"), Path("wam/stem.py"), Path("stem/reference_stem.pt")]
    paths += [Path("dataset") / row["video"] for rows in splits.values() for row in rows]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for path in paths:
            output.write(kit / path, str(Path("track3-kit") / path).replace("\\", "/"))
        output.writestr("track3-kit/QUICK_BUNDLE.json", json.dumps({
            "profile": "t4_quick", "actions_included": False,
            "train_ids": [row["episode_id"] for row in splits["train"]],
            "dev_ids": [row["episode_id"] for row in splits["dev"]],
            "note": "Original MP4 bytes; full original manifest retained for deterministic selection."
        }, indent=2))
    result = {"archive": str(archive), "size": archive.stat().st_size, "sha256": file_hash(archive),
              "train": len(splits["train"]), "dev": len(splits["dev"])}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/t4_quick.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pack(args.kit, args.config, args.out)
