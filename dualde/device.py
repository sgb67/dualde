"""Device checks for optional GPU inference (ESM-2 650M, SPURS)."""

from __future__ import annotations


def cuda_is_available() -> bool:
    """True if PyTorch can see a CUDA GPU. False if torch is missing or CPU-only."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def require_cuda(task: str) -> str:
    """Return ``'cuda'`` or raise if no GPU is available."""
    if not cuda_is_available():
        raise RuntimeError(
            f"{task} requires a CUDA GPU. dualde will not run this model on CPU; "
            "use the precomputed CSV shipped with the example instead."
        )
    return "cuda"
