from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def persist_file(source: Path, directory: Path | None) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.resolve() == source.resolve():
        return
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(generator: np.random.Generator) -> dict:
    # Primitive containers + torch tensors remain compatible with weights_only=True.
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "sampler": generator.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict, generator: np.random.Generator) -> None:
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
    generator.bit_generator.state = state["sampler"]
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def device_from_string(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Select a GPU runtime in Colab.")
    return device
