from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def load_reference_stem(kit_root: str | Path):
    """Load the provided architecture and weights without copying kit assets into Git."""
    kit_root = Path(kit_root).resolve()
    source = kit_root / "wam" / "stem.py"
    weights = kit_root / "stem" / "reference_stem.pt"
    if not source.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Expected wam/stem.py and stem/reference_stem.pt under {kit_root}")
    module_name = "_video_wam_reference_stem"
    specification = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    stem = module.Stem()
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if "state" in payload:
        state = payload["state"]
    elif "stem" in payload:
        state = payload["stem"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    stem.load_state_dict(state, strict=True)
    stem.assert_scope()
    stem.requires_grad_(False)
    stem.eval()
    return stem
