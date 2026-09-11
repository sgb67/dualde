"""Raw SPURS ddG tables, UniProt-remapped via a mapping JSON.

Does not import evedesign. SPURS inference (get_SPURS_from_hub +
parse_pdb_resolved_only) is an optional extra; cached CSVs work without it.

Cached / returned scores are raw ddG (wild-type diagonal 0). Deconvolution
uses -spurs so stabilizing mutations rank higher.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from dualde.device import require_cuda
from dualde.mapping import (
    extract_sequence_from_pdb,
    pdb_to_uniprot_map_from_position_mapping,
    seqres_to_pdb_offset,
)

logger = logging.getLogger(__name__)

SPURS_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _pdb_resnum_from_label(label: object) -> int:
    """Parse an integer PDB residue number, ignoring insertion codes (e.g. 162A)."""
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
    if not seq:
        return False
    allowed = set(SPURS_ALPHABET + "X-U")
    return all(c in allowed for c in seq)


def _batch_seq_and_resnums(spurs_pdb: dict | None, pdb_path: str, chain: str) -> tuple[str, list[int], list[str]]:
    """Sequence + PDB residue numbers in the same order as SPURS ddg rows."""
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
            "SPURS batch seq differs from BioPython seq on this PDB; "
            "using SPURS seq for wt letters and BioPython residue numbers for labels.",
            stacklevel=2,
        )

    if resn_list is not None and (not seq or len(resn_list) == len(seq)):
        resnums = [_pdb_resnum_from_label(r) for r in resn_list]
        labels = [str(r).strip() for r in resn_list]
    elif seq:
        resnums = list(resnums_bio) if len(resnums_bio) == len(seq) else list(range(1, len(seq) + 1))
        labels = [str(r) for r in resnums]
        if resn_list is not None:
            warnings.warn(
                f"SPURS resn_list length {len(resn_list)} != seq length {len(seq)}; "
                "falling back to PDB / 1..L labels",
                stacklevel=2,
            )
    else:
        resnums, labels = [], []
    return seq, resnums, labels


def spurs_ddg_to_pdb_dataframe(
    ddg,
    pdb_path: str,
    chain: str = "A",
    spurs_pdb: dict | None = None,
) -> pd.DataFrame:
    """Build a PDB-numbered mutant table from SPURS logit output (raw ddG)."""
    if hasattr(ddg, "detach"):
        ddg = ddg.detach().cpu().numpy()
    ddg = np.asarray(ddg, dtype=float)
    if ddg.ndim != 2 or ddg.shape[1] != len(SPURS_ALPHABET):
        raise ValueError(f"ddg must have shape (n_positions, 20), got {ddg.shape}")

    seq, resnums, resn_labels = _batch_seq_and_resnums(spurs_pdb, pdb_path, chain)
    if len(seq) != ddg.shape[0]:
        raise ValueError(
            f"ddg has {ddg.shape[0]} positions but recovered sequence has {len(seq)} residues"
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
    seqres_to_pdb_offset: int = 0,
    uniprot_sequence: str | None = None,
    first_index: int = 1,
) -> pd.DataFrame:
    """Rewrite mutant labels to UniProt numbering using JSON position_mapping."""
    pdb_to_uniprot = pdb_to_uniprot_map_from_position_mapping(
        mapping_entry, seqres_to_pdb_offset=seqres_to_pdb_offset
    )
    if not pdb_to_uniprot:
        raise ValueError("Mapping JSON produced an empty PDB→UniProt map")

    out = pdb_df.copy()
    out["uniprot_position"] = out["pdb_position"].map(pdb_to_uniprot)
    out = out[out["uniprot_position"].notna()].copy()
    out["uniprot_position"] = out["uniprot_position"].astype(int)

    if uniprot_sequence:
        idx = out["uniprot_position"] - first_index
        valid = (idx >= 0) & (idx < len(uniprot_sequence))
        out = out.loc[valid].copy()
        idx = out["uniprot_position"] - first_index
        seq_wt = [uniprot_sequence[i] for i in idx]
        mismatch = out["wt"].to_numpy() != np.asarray(seq_wt)
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
    need = {"mutant", "spurs"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"SPURS CSV missing columns: {sorted(missing)}")
    return df


def add_neg_spurs(df: pd.DataFrame, col: str = "spurs") -> pd.DataFrame:
    """Add -spurs so stabilizing mutations are higher (matches neg_spurs_multi_min)."""
    out = df.copy()
    out["neg_spurs"] = -pd.to_numeric(out[col], errors="coerce")
    return out


def run_spurs_scan(
    pdb_path: str,
    mapping_entry: dict[str, Any],
    chain: str = "A",
    device: str | None = None,
    uniprot_sequence: str | None = None,
    seqres_search_dirs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run raw SPURS (get_SPURS_from_hub + parse_pdb_resolved_only) and remap to UniProt.

    Requires a CUDA GPU and the optional ``spurs`` extra (local SPURS package
    from ``fix/py311-inference-beta``; not the PyPI name ``spurs``). CPU
    inference is not supported; use the cached CSV instead.
    """
    gpu = require_cuda("SPURS inference")
    if device is not None and not str(device).startswith("cuda"):
        raise RuntimeError("SPURS inference in dualde requires device='cuda'.")

    try:
        import torch
        from spurs.inference import get_SPURS_from_hub, parse_pdb_resolved_only
    except ImportError as exc:
        raise ImportError(
            "SPURS inference extras are not installed. Use the cached CSV, or "
            "install dualde with the [spurs] extra and the local SPURS package "
            "from branch fix/py311-inference-beta (do not `pip install spurs` from PyPI)."
        ) from exc

    device = gpu

    pdb_id = str(mapping_entry.get("pdb_id", os.path.splitext(os.path.basename(pdb_path))[0]))
    model, cfg = get_SPURS_from_hub(device=device)
    model.eval()
    pdb_batch = parse_pdb_resolved_only(pdb_path, pdb_id, chain, cfg, device=device)
    with torch.no_grad():
        ddg = model(pdb_batch, return_logist=True)

    pdb_df = spurs_ddg_to_pdb_dataframe(ddg, pdb_path, chain=chain, spurs_pdb=pdb_batch)
    offset = seqres_to_pdb_offset(
        pdb_path,
        pdb_id,
        chain,
        search_dirs=seqres_search_dirs,
    )
    return remap_spurs_to_uniprot(
        pdb_df,
        mapping_entry,
        seqres_to_pdb_offset=offset,
        uniprot_sequence=uniprot_sequence,
    )
