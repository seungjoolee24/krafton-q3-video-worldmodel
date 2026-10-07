"""Dependency-free syntax and notebook/config validation; does not train a model."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    checked = []
    for directory in ["src", "scripts", "tests"]:
        for path in sorted((root / directory).rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            checked.append(str(path.relative_to(root)))
    notebooks = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        cells = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")
                assert not cell.get("outputs"), "Do not commit notebook output/credentials"
                cells += 1
        notebooks.append({"name": path.name, "python_cells": cells})
    assert notebooks, "Missing executable Colab notebook"
    for path in (root / "configs").glob("*.json"):
        config = json.loads(path.read_text())
        assert config["horizon_schedule"][0][0] == 0
        assert config["horizon_schedule"][-1][1] == 32
        assert config["model"]["context"] == 32
    report = {"python_files": len(checked), "notebooks": notebooks,
              "syntax": "passed", "model_runtime": "requires Torch smoke/unit checks"}
    (root / "work").mkdir(exist_ok=True)
    (root / "work" / "project-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
