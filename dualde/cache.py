"""Keep pip, Hugging Face, Torch, and temp files out of ``$HOME``.

Set ``DUALDE_CACHE`` to an absolute directory (fast scratch, not NFS home).
If unset, dualde uses ``<repo>/.cache``. Existing environment variables are
left unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

_APPLIED = False


def cache_root() -> Path:
    """Writable cache root: ``$DUALDE_CACHE`` or ``<repo>/.cache``."""
    env = os.environ.get("DUALDE_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".cache"
    raise FileNotFoundError(
        "Set DUALDE_CACHE to a writable directory outside $HOME "
        "(no dualde checkout with pyproject.toml on the path)."
    )


def apply_cache_env(*, create: bool = True) -> Path | None:
    """
    Point installer and model caches at ``cache_root()`` when unset.

    Returns the cache root, or ``None`` if it cannot be resolved (installed
    wheel with no ``DUALDE_CACHE`` and no checkout). Never writes under
    ``$HOME``.
    """
    global _APPLIED
    try:
        root = cache_root()
    except FileNotFoundError:
        return None
    if root.is_relative_to(Path.home()):
        raise RuntimeError(
            f"DUALDE_CACHE ({root}) resolves under $HOME. Point it at scratch "
            "or the clone, e.g. export DUALDE_CACHE=\"$PWD/.cache\"."
        )
    mapping = {
        "PIP_CACHE_DIR": root / "pip",
        "TMPDIR": root / "tmp",
        "TEMP": root / "tmp",
        "TMP": root / "tmp",
        "XDG_CACHE_HOME": root,
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": root / "huggingface" / "transformers",
        "TORCH_HOME": root / "torch",
        "MPLCONFIGDIR": root / "matplotlib",
    }
    for key, path in mapping.items():
        if os.environ.get(key):
            continue
        if create:
            path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    _APPLIED = True
    return root
