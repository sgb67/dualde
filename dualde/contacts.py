"""Ligand and protein contacts from one PDB plus a UniProt mapping JSON."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

from dualde.mapping import pdb_to_uniprot_map_from_position_mapping

NUDT15_LIGAND_NAMES = ("MKB", "SYW", "4DG", "UO8", "RQ2", "RTW", "RDZ", "71V")


def _residue_atoms(residue) -> np.ndarray:
    coords = [atom.coord for atom in residue if atom.element != "H"]
    if not coords:
        coords = [atom.coord for atom in residue]
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return float("nan")
    delta = coords_a[:, None, :] - coords_b[None, :, :]
    return float(np.sqrt((delta * delta).sum(axis=2)).min())


def _iter_protein_residues(structure, chain_id: str | None = None):
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                resname = residue.get_resname()
                if resname not in protein_letters_3to1:
                    continue
                yield chain.id, residue


def _iter_ligand_residues(structure, ligand_names: Sequence[str]):
    names = {n.strip().upper() for n in ligand_names}
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                if resname in names:
                    yield chain.id, residue


def residue_contacts(
    pdb_path: str,
    mapping_entry: dict[str, Any],
    chain: str = "A",
    ligand_names: Sequence[str] = NUDT15_LIGAND_NAMES,
    distance_cutoff: float = 8.0,
    seqres_to_pdb_offset: int = 0,
    include_protein: bool = True,
) -> pd.DataFrame:
    """
    All-atom min distances from mapped protein residues on ``chain`` to ligands
    and (optionally) other protein chains.

    Returns a long table with columns:
    target_pos, pdb_position, pdb_id, chain, residue_name, partner_chain,
    contact_type, minimum_distance.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    pdb_id = str(mapping_entry.get("pdb_id", "")).lower()
    pdb_to_uniprot = pdb_to_uniprot_map_from_position_mapping(
        mapping_entry, seqres_to_pdb_offset=seqres_to_pdb_offset
    )

    query_residues = []
    for chain_id, residue in _iter_protein_residues(structure, chain_id=chain):
        pdb_pos = residue.get_id()[1]
        uniprot_pos = pdb_to_uniprot.get(int(pdb_pos))
        if uniprot_pos is None:
            continue
        query_residues.append(
            {
                "chain": chain_id,
                "pdb_position": int(pdb_pos),
                "target_pos": int(uniprot_pos),
                "wt": protein_letters_3to1[residue.get_resname()],
                "coords": _residue_atoms(residue),
            }
        )

    partners: list[dict[str, Any]] = []
    for chain_id, residue in _iter_ligand_residues(structure, ligand_names):
        partners.append(
            {
                "residue_name": residue.get_resname().strip().upper(),
                "partner_chain": chain_id,
                "contact_type": "ligand",
                "coords": _residue_atoms(residue),
            }
        )
    if include_protein:
        for chain_id, residue in _iter_protein_residues(structure, chain_id=None):
            if chain_id == chain:
                continue
            partners.append(
                {
                    "residue_name": residue.get_resname().strip().upper(),
                    "partner_chain": chain_id,
                    "contact_type": "protein",
                    "coords": _residue_atoms(residue),
                }
            )

    rows = []
    for query in query_residues:
        for partner in partners:
            dist = _min_distance(query["coords"], partner["coords"])
            if not np.isfinite(dist) or dist > distance_cutoff:
                continue
            rows.append(
                {
                    "target_pos": query["target_pos"],
                    "pdb_position": query["pdb_position"],
                    "wt": query["wt"],
                    "pdb_id": pdb_id,
                    "chain": query["chain"],
                    "residue_name": partner["residue_name"],
                    "partner_chain": partner["partner_chain"],
                    "contact_type": partner["contact_type"],
                    "minimum_distance": dist,
                }
            )
    return pd.DataFrame(rows)


def closest_contacts(contacts: pd.DataFrame) -> pd.DataFrame:
    """One row per target_pos: the closest partner (any type)."""
    if contacts.empty:
        return contacts.copy()
    idx = contacts.groupby("target_pos")["minimum_distance"].idxmin()
    return contacts.loc[idx].sort_values("target_pos").reset_index(drop=True)


def functional_residues(
    contacts: pd.DataFrame,
    distance_cutoff: float = 3.5,
    contact_types: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Contacts below ``distance_cutoff``, keeping the closest partner of each type
    per UniProt position.

    Routing notebooks typically keep ligand rows whose residue_name is in the
    active-site ligand list.
    """
    if contacts.empty:
        return contacts.copy()
    work = contacts.loc[contacts["minimum_distance"] < distance_cutoff].copy()
    if contact_types is not None:
        types = {str(t) for t in contact_types}
        work = work.loc[work["contact_type"].isin(types)]
    if work.empty:
        return work
    keys = ["target_pos", "contact_type", "residue_name"]
    idx = work.groupby(keys)["minimum_distance"].idxmin()
    cols = [
        "target_pos",
        "minimum_distance",
        "pdb_id",
        "residue_name",
        "contact_type",
    ]
    extra = [c for c in ("partner_chain", "chain", "wt", "pdb_position") if c in work.columns]
    return (
        work.loc[idx, cols + extra]
        .sort_values(["target_pos", "contact_type", "minimum_distance"])
        .reset_index(drop=True)
    )


def attach_minimum_distance(scores: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    """Left-join the closest contact distance onto a mutant table with target_pos."""
    if contacts.empty:
        out = scores.copy()
        if "minimum_distance" not in out.columns:
            out["minimum_distance"] = np.nan
        return out
    closest = closest_contacts(contacts)
    keep = [
        c
        for c in (
            "target_pos",
            "minimum_distance",
            "residue_name",
            "pdb_id",
            "contact_type",
        )
        if c in closest.columns
    ]
    return scores.merge(closest[keep], on="target_pos", how="left")
