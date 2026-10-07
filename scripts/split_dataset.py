"""Split the original ZIP into upload-sized chunks without changing its bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split(archive: Path, out: Path, chunk_mib: int = 95):
    if chunk_mib < 1:
        raise ValueError("Chunk size must be positive")
    out.mkdir(parents=True, exist_ok=True)
    total, parts = hashlib.sha256(), []
    chunk_bytes = chunk_mib * 1024 * 1024
    with open(archive, "rb") as source:
        number = 1
        while True:
            first = source.read(min(1024 * 1024, chunk_bytes))
            if not first:
                break
            path = out / f"{archive.name}.part{number:03d}"
            digest, size = hashlib.sha256(), 0
            with open(path, "wb") as destination:
                block = first
                while block:
                    destination.write(block)
                    digest.update(block)
                    total.update(block)
                    size += len(block)
                    if size >= chunk_bytes:
                        break
                    block = source.read(min(1024 * 1024, chunk_bytes - size))
            parts.append({"name": path.name, "size": size, "sha256": digest.hexdigest()})
            print(path.name, size, flush=True)
            number += 1
    manifest = {"format": "split-zip/1", "original_name": archive.name,
                "original_size": archive.stat().st_size, "original_sha256": total.hexdigest(), "parts": parts}
    path = out / "track3-kit.parts.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=95)
    args = parser.parse_args()
    split(args.archive, args.out, args.chunk_mib)
