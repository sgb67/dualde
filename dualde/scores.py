"""Precomputed ESM-2 650M and SPURS tables.

Notebooks load these tables and do not run either model. The scan helpers
at the bottom of this module are unused by the notebooks.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from dualde.cache import apply_cache_env
from dualde.mapping import extract_sequence_from_pdb, pdb_to_uniprot_map

logger = logging.getLogger(__name__)

SPURS_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ESM2_650M_NAME = "esm2_t33_650M_UR50D"
ESM2_650M_COLUMN = "ESM2_650M"


def cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _require_cuda(task: str) -> str:
    if not cuda_is_available():
        raise RuntimeError(
            f"{task} requires a CUDA GPU. Use the precomputed table shipped with "
            "the example, or run on a GPU machine."
        )
    return "cuda"


def run_esm2_650m(
    sequence: str,
    protein_id: str = "protein",
    device: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Saturated single-mutant ESM-2 650M scan. CUDA required."""
    apply_cache_env()
    gpu = _require_cuda("ESM-2 650M inference")
    if device is not None and not str(device).startswith("cuda"):
        raise RuntimeError("ESM-2 650M inference requires device='cuda'.")
    try:
        from evedesign.models import esm2
        from evedesign.system import EntityInstance, Protein, System, SystemInstance
    except ImportError as exc:
        raise ImportError(
            "Install the [evo] extra (`pip install -e \".[evo]\"`) to run ESM-2, "
            "or load a precomputed score table."
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


def _pdb_resnum_from_label(label: object) -> int:
    text = str(label).strip()
    num = ""
    for i, c in enumerate(text):
        if c == "-" and i == 0:
            num += c
            continue
        if c.isdigit():
            num += c
            continue
        break
    if not num or num == "-":
        raise ValueError(f"Could not parse PDB residue number from {label!r}")
    return int(num)


def _looks_like_aa_seq(seq: str) -> bool:
    return bool(seq) and all(c in SPURS_ALPHABET + "X-U" for c in seq)


def _batch_seq_and_resnums(
    spurs_pdb: dict | None, pdb_path: str, chain: str
) -> tuple[str, list[int], list[str]]:
    seq = ""
    resn_list = None
    if spurs_pdb is not None:
        raw_seq = spurs_pdb.get("seq")
        if isinstance(raw_seq, (list, tuple)):
            raw_seq = raw_seq[0]
        if raw_seq is not None and hasattr(raw_seq, "detach"):
            seq = ""
        else:
            seq = str(raw_seq) if raw_seq is not None else ""
            if not _looks_like_aa_seq(seq):
                seq = ""
        if "spurs_resn_list" in spurs_pdb:
            resn_list = list(spurs_pdb["spurs_resn_list"])
        elif "resn_list" in spurs_pdb:
            resn_list = list(spurs_pdb["resn_list"])

    seq_bio, resnums_bio = extract_sequence_from_pdb(pdb_path, chain=chain)
    if not seq:
        seq = seq_bio
    elif seq != seq_bio:
        warnings.warn(
            "SPURS sequence differs from the PDB ATOM sequence; using SPURS letters "
            "with BioPython residue numbers.",
            stacklevel=2,
        )

    if resn_list is not None and (not seq or len(resn_list) == len(seq)):
        resnums = [_pdb_resnum_from_label(r) for r in resn_list]
        labels = [str(r).strip() for r in resn_list]
    elif seq:
        resnums = list(resnums_bio) if len(resnums_bio) == len(seq) else list(range(1, len(seq) + 1))
        labels = [str(r) for r in resnums]
    else:
        resnums, labels = [], []
    return seq, resnums, labels


def spurs_ddg_to_pdb_dataframe(
    ddg, pdb_path: str, chain: str = "A", spurs_pdb: dict | None = None
) -> pd.DataFrame:
    """PDB-numbered mutant table from SPURS logits (raw ΔΔG)."""
    if hasattr(ddg, "detach"):
        ddg = ddg.detach().cpu().numpy()
    ddg = np.asarray(ddg, dtype=float)
    if ddg.ndim != 2 or ddg.shape[1] != len(SPURS_ALPHABET):
        raise ValueError(f"ddg must have shape (n_positions, 20), got {ddg.shape}")

    seq, resnums, resn_labels = _batch_seq_and_resnums(spurs_pdb, pdb_path, chain)
    if len(seq) != ddg.shape[0]:
        raise ValueError(
            f"ddg has {ddg.shape[0]} positions but recovered sequence has {len(seq)}"
        )

    rows = []
    for i, wt in enumerate(seq):
        if wt not in SPURS_ALPHABET:
            continue
        pdb_pos = int(resnums[i])
        resn_label = resn_labels[i]
        for j, subs in enumerate(SPURS_ALPHABET):
            rows.append(
                {
                    "mutant": f"{wt}{resn_label}{subs}",
                    "pdb_position": pdb_pos,
                    "wt": wt,
                    "subs": subs,
                    "spurs": float(ddg[i, j]),
                }
            )
    return pd.DataFrame(rows)


def remap_spurs_to_uniprot(
    pdb_df: pd.DataFrame,
    mapping_entry: dict[str, Any],
    uniprot_sequence: str | None = None,
    first_index: int = 1,
) -> pd.DataFrame:
    """Rewrite mutant labels to UniProt numbering using mapping fragments."""
    pdb_to_uniprot = pdb_to_uniprot_map(mapping_entry)
    if not pdb_to_uniprot:
        raise ValueError("Mapping produced an empty PDB→UniProt map")

    out = pdb_df.copy()
    out["uniprot_position"] = out["pdb_position"].map(pdb_to_uniprot)
    out = out[out["uniprot_position"].notna()].copy()
    out["uniprot_position"] = out["uniprot_position"].astype(int)

    if uniprot_sequence:
        idx = out["uniprot_position"] - first_index
        valid = (idx >= 0) & (idx < len(uniprot_sequence))
        out = out.loc[valid].copy()
        idx = out["uniprot_position"] - first_index
        seq_wt = np.asarray([uniprot_sequence[i] for i in idx])
        mismatch = out["wt"].to_numpy() != seq_wt
        n_drop = int(mismatch.sum())
        if n_drop:
            logger.warning("Dropped %d SPURS rows where PDB WT ≠ UniProt sequence", n_drop)
            out = out.loc[~mismatch].copy()

    out["mutant"] = out["wt"] + out["uniprot_position"].astype(str) + out["subs"]
    return out[["mutant", "spurs", "wt", "subs", "uniprot_position", "pdb_position"]].reset_index(
        drop=True
    )


def load_spurs_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"mutant", "spurs"} - set(df.columns)
    if missing:
        raise ValueError(f"SPURS CSV missing columns: {sorted(missing)}")
    return df


def add_neg_spurs(df: pd.DataFrame, col: str = "spurs") -> pd.DataFrame:
    """Add −ΔΔG so stabilizing substitutions rank higher in deconvolution."""
    out = df.copy()
    out["neg_spurs"] = -pd.to_numeric(out[col], errors="coerce")
    return out


def run_spurs_scan(
    pdb_path: str,
    mapping_entry: dict[str, Any],
    chain: str = "A",
    device: str | None = None,
    uniprot_sequence: str | None = None,
) -> pd.DataFrame:
    """
    Raw SPURS scan remapped to UniProt via the mapping JSON.

    CUDA required. Install the [spurs] extra and the official SPURS package
    (GitHub ``luo-group/SPURS``, branch ``fix/py311-inference-beta``). Do not
    ``pip install spurs`` from PyPI.
    """
    apply_cache_env()
    gpu = _require_cuda("SPURS inference")
    if device is not None and not str(device).startswith("cuda"):
        raise RuntimeError("SPURS inference requires device='cuda'.")
    try:
        import torch
        from spurs.inference import get_SPURS_from_hub, parse_pdb_resolved_only
    except ImportError as exc:
        raise ImportError(
            "SPURS is not installed. Load a precomputed CSV, or follow the README "
            "[spurs] extra (not the PyPI project named spurs)."
        ) from exc

    pdb_id = str(mapping_entry.get("pdb_id", os.path.splitext(os.path.basename(pdb_path))[0]))
    model, cfg = get_SPURS_from_hub(device=gpu)
    model.eval()
    pdb_batch = parse_pdb_resolved_only(pdb_path, pdb_id, chain, cfg, device=gpu)
    with torch.no_grad():
        ddg = model(pdb_batch, return_logist=True)

    pdb_df = spurs_ddg_to_pdb_dataframe(ddg, pdb_path, chain=chain, spurs_pdb=pdb_batch)
    return remap_spurs_to_uniprot(pdb_df, mapping_entry, uniprot_sequence=uniprot_sequence)
