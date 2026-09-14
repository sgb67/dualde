"""dualde: dual-graph deconvolution of enzyme fitness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__version__ = "0.1.0"


def repo_root() -> Path:
    """Repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not find the dualde repository root.")


def example_dir(name: str = "NUDT15_HUMAN") -> Path:
    """``examples/<name>/`` under the repository root."""
    path = repo_root() / "examples" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Example directory not found: {path}")
    return path


def load_example(name: str = "NUDT15_HUMAN") -> tuple[Path, dict[str, Any]]:
    """Return ``(examples/<name>, config.json)`` for a protein example."""
    root = example_dir(name)
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {root}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return root, cfg


__all__ = ["__version__", "example_dir", "load_example", "repo_root"]
