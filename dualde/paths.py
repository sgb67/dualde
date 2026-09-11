"""Locate the dualde repo root and example directories without hardcoded paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the dualde repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find dualde repo root (no pyproject.toml above dualde/)."
    )


def example_dir(name: str = "NUDT15_HUMAN") -> Path:
    """Return examples/<name>/ under the repo root."""
    path = repo_root() / "examples" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Example directory not found: {path}")
    return path
