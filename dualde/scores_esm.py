"""ESM-2 650M single-mutant scans (GPU only).

CPU execution is intentionally unsupported. Use the precomputed
``scores/evolutionary.csv`` when a GPU is not available.

EVmutation scores and coupling graphs are not produced here; obtain them from
the EVcouplings webserver (https://evcouplings.org).
"""

from __future__ import annotations

import pandas as pd

from dualde.device import require_cuda

ESM2_650M_NAME = "esm2_t33_650M_UR50D"
ESM2_650M_COLUMN = "ESM2_650M"


def run_esm2_650m(
    sequence: str,
    protein_id: str = "protein",
    device: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Saturated single-mutant ESM-2 650M scan.

    Returns columns ``pos``, ``ref``, ``aa_to``, ``mutation``, ``ESM2_650M``.
    Requires the optional ``evo`` extra (PyPI ``evedesign``) and a CUDA GPU.
    """
    gpu = require_cuda("ESM-2 650M inference")
    if device is not None and not str(device).startswith("cuda"):
        raise RuntimeError("ESM-2 650M inference in dualde requires device='cuda'.")

    try:
        from evedesign.models import esm2
        from evedesign.system import EntityInstance, Protein, System, SystemInstance
    except ImportError as exc:
        raise ImportError(
            "ESM-2 extras are not installed. Use the cached evolutionary CSV, or "
            "`pip install -e \".[evo]\"` (published evedesign) on a GPU machine."
        ) from exc

    model = esm2.ESM2(
        model_name=ESM2_650M_NAME,
        model_dir_path=None,
        batch_size=batch_size,
        keep_model_after_build=True,
        device=gpu,
        num_sweeps=1000,
        init_strategy="system",
        scan_order="random",
    )
    system = System(Protein(id=protein_id, rep=sequence, first_index=1))
    model.build(system)
    scan = model.single_mutation_scan(SystemInstance(EntityInstance(rep=sequence)))
    return (
        scan.stack()
        .to_frame("score")
        .reset_index()
        .rename(columns={"level_3": "aa_to"})
        .assign(mutation=lambda df: df["ref"] + df["pos"].astype(str) + df["aa_to"])
        [["pos", "ref", "aa_to", "mutation", "score"]]
        .rename(columns={"score": ESM2_650M_COLUMN})
        .sort_values("pos")
        .reset_index(drop=True)
    )
