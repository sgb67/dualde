"""Ligand and protein contacts from PDB hits plus UniProt mapping JSON."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from Bio.PDB import is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

from dualde.mapping import homolog_chain_ids, load_structure, pdb_to_uniprot_map, preferred_assembly_chains

logger = logging.getLogger(__name__)


def _load_structure(pdb_path: str):
    return load_structure(pdb_path)


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
    ligand_names: Sequence[str],
    chain: str = "A",
    distance_cutoff: float = 8.0,
    include_protein: bool = True,
    mapped_chains: Iterable[str] | None = None,
    homolog_chains: Iterable[str] | None = None,
    preferred_chains: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    All-atom min distances from mapped protein residues on ``chain`` to ligands
    and (optionally) other protein chains.

    ``ligand_names`` are PDB residue names (HETATM), e.g. ``("71V", "MKB")``.
    UniProt numbering is taken from the mapping JSON (per-fragment PDB ranges).

    Protein partners must be in ``homolog_chains`` (sequence identity cutoff)
    **and**, when set, in ``preferred_chains`` (PDBe preferred biological
    assembly). Both the query chain and the partner must belong to that
    assembly. Those partners are labelled ``homodimer``.
    """
    structure = _load_structure(pdb_path)
    pdb_id = str(mapping_entry.get("pdb_id", "")).lower()
    pdb_to_uniprot = pdb_to_uniprot_map(mapping_entry)
    homologs = {str(c) for c in homolog_chains} if homolog_chains is not None else None
    preferred = {str(c) for c in preferred_chains} if preferred_chains is not None else None
    homodimer_chains = homologs if homologs is not None else {str(c) for c in (mapped_chains or [chain])}
    if preferred is not None:
        homodimer_chains = homodimer_chains & preferred

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
        query_in_assembly = preferred is None or chain in preferred
        for chain_id, residue in _iter_protein_residues(structure, chain_id=None):
            if chain_id == chain:
                continue
            if not query_in_assembly:
                continue
            if homologs is not None and chain_id not in homologs:
                continue
            if preferred is not None and chain_id not in preferred:
                continue
            partners.append(
                {
                    "residue_name": residue.get_resname().strip().upper(),
                    "partner_chain": chain_id,
                    "contact_type": (
                        "homodimer" if chain_id in homodimer_chains else "hetero_ppi"
                    ),
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


def residue_contacts_from_structures(
    entries: Sequence[dict[str, Any]],
    pdb_paths: Mapping[str, str],
    ligand_names: Sequence[str],
    distance_cutoff: float = 8.0,
    include_protein: bool = True,
    query_seq: str | None = None,
    min_identity: float = 0.6,
    use_preferred_assembly: bool = True,
) -> pd.DataFrame:
    """Contacts for every mapping hit that passed the identity filter (before RMSD).

    Protein–protein contacts use PDBe preferred-assembly chain pairs when
    ``use_preferred_assembly`` is true (crystal-packing copies are ignored).
    """
    chains_by_pdb: dict[str, set[str]] = {}
    entries_by_pdb: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        pid = str(entry.get("pdb_id", "")).lower()
        chains_by_pdb.setdefault(pid, set()).add(str(entry.get("pdb_chain", "")))
        entries_by_pdb.setdefault(pid, []).append(entry)

    homologs_by_pdb: dict[str, set[str]] = {}
    if query_seq:
        for pid, path in pdb_paths.items():
            homologs_by_pdb[pid] = homolog_chain_ids(
                path,
                query_seq,
                min_identity=min_identity,
                mapping_entries=entries_by_pdb.get(pid, []),
            )

    preferred_by_pdb: dict[str, set[str] | None] = {}
    if use_preferred_assembly:
        for pid in pdb_paths:
            try:
                preferred_by_pdb[pid] = preferred_assembly_chains(pid)
            except Exception as exc:
                logger.warning(
                    "Could not fetch PDBe preferred assembly for %s (%s); "
                    "skipping protein partners for this entry",
                    pid,
                    exc,
                )
                preferred_by_pdb[pid] = set()

    frames: list[pd.DataFrame] = []
    for entry in entries:
        pid = str(entry.get("pdb_id", "")).lower()
        path = pdb_paths.get(pid)
        if not path:
            continue
        chain = str(entry.get("pdb_chain", "A"))
        try:
            df = residue_contacts(
                path,
                entry,
                ligand_names=ligand_names,
                chain=chain,
                distance_cutoff=distance_cutoff,
                include_protein=include_protein,
                mapped_chains=chains_by_pdb.get(pid),
                homolog_chains=homologs_by_pdb.get(pid),
                preferred_chains=preferred_by_pdb.get(pid),
            )
        except Exception as exc:
            logger.warning(
                "Skipping contacts for %s chain %s (%s)", pid, chain, exc
            )
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "target_pos",
                "pdb_position",
                "wt",
                "pdb_id",
                "chain",
                "residue_name",
                "partner_chain",
                "contact_type",
                "minimum_distance",
            ]
        )
    return pd.concat(frames, ignore_index=True)


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
    Closest contact below ``distance_cutoff`` for each UniProt position,
    pooling homologous structures that passed the identity filter.

    Columns match the published table: ``target_pos``, ``minimum_distance``,
    ``pdb_id``, ``residue_name``, ``contact_type``.
    """
    if contacts.empty:
        return contacts.copy()
    work = contacts.loc[contacts["minimum_distance"] < distance_cutoff].copy()
    if contact_types is not None:
        types = {str(t) for t in contact_types}
        work = work.loc[work["contact_type"].isin(types)]
    if work.empty:
        return work
    idx = work.groupby("target_pos")["minimum_distance"].idxmin()
    cols = ["target_pos", "minimum_distance", "pdb_id", "residue_name", "contact_type"]
    return (
        work.loc[idx, cols]
        .sort_values("target_pos")
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
