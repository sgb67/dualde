"""Intra-chain contact graphs and FreeSASA relative solvent accessibility."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import Model, Structure, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

from dualde.mapping import load_structure, pdb_to_uniprot_map

# Reference areas (Å²) for RSA = FreeSASA SASA / max area.
# Same table used for max_rsa elsewhere in this project
# (Tien et al. 2013 empirical values, with Ile = 181 and Lys = 214).
MAX_SASA = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 181.0,
    "LEU": 191.0,
    "LYS": 214.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}

_FREESASA_OPTIONS = {
    "hetatm": False,
    "hydrogen": False,
    "join-models": False,
    "skip-unknown": True,
    "halt-at-unknown": False,
}


def _first_model(structure):
    models = list(structure.get_models())
    if len(models) <= 1:
        return structure
    trimmed = Structure.Structure(structure.id)
    trimmed.add(models[0].copy())
    return trimmed


def _chain_only(structure, chain_id: str):
    model = next(_first_model(structure).get_models())
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id!r} is not in the structure")
    out = Structure.Structure("chain")
    new_model = Model.Model(0)
    new_model.add(model[chain_id].copy())
    out.add(new_model)
    return out


def _iter_standard_residues(chain):
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        resname = residue.get_resname()
        if resname not in protein_letters_3to1:
            continue
        yield residue


def _heavy_coords(residue) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        element = (atom.element or "").strip().upper()
        if element == "H" or atom.get_name().strip().upper().startswith("H"):
            continue
        coords.append(atom.coord)
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _labels_for_chain(chain, mapping_entry: dict[str, Any] | None):
    pdb_to_uni = pdb_to_uniprot_map(mapping_entry) if mapping_entry else None
    residues = []
    labels = []
    seen = set()
    for residue in _iter_standard_residues(chain):
        pdb_num = int(residue.id[1])
        label = pdb_to_uni.get(pdb_num) if pdb_to_uni is not None else pdb_num
        if label is None or int(label) in seen:
            continue
        residues.append(residue)
        labels.append(int(label))
        seen.add(int(label))
    return residues, labels


def residue_distance_map(
    pdb_file: str,
    chain_id: str = "A",
    atom: str = "CA",
    mapping_entry: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Pairwise heavy-atom or Cα distances.

    Index and columns are UniProt positions when ``mapping_entry`` is given,
    otherwise PDB residue numbers.
    """
    structure = _first_model(load_structure(pdb_file))
    model = next(structure.get_models())
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id!r} is not in {pdb_file}")
    residues, labels = _labels_for_chain(model[chain_id], mapping_entry)
    if atom.upper() == "CA":
        keep = [(r, lab) for r, lab in zip(residues, labels) if "CA" in r]
        residues = [r for r, _ in keep]
        labels = [lab for _, lab in keep]
        coords = np.array([r["CA"].coord for r in residues], dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
    else:
        coords_list = [_heavy_coords(r) for r in residues]
        n = len(residues)
        dist = np.full((n, n), np.inf)
        for i in range(n):
            if coords_list[i].size == 0:
                continue
            for j in range(i, n):
                if coords_list[j].size == 0:
                    continue
                delta = coords_list[i][:, None, :] - coords_list[j][None, :, :]
                d = float(np.sqrt((delta * delta).sum(axis=-1)).min())
                dist[i, j] = dist[j, i] = d
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=labels, columns=labels)


def min_distance_across_structures(dist_maps: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Element-wise minimum distance across conformations.

    A pair missing from one structure does not contribute a distance from
    that structure. Residue numbers are the shared index.
    """
    if not dist_maps:
        raise ValueError("dist_maps is empty")
    residues = sorted(set().union(*[set(map(int, df.index)) for df in dist_maps]))
    out = np.full((len(residues), len(residues)), np.inf)
    for df in dist_maps:
        aligned = (
            df.copy()
            .rename(index=int, columns=int)
            .reindex(index=residues, columns=residues)
            .to_numpy(dtype=float, copy=True)
        )
        aligned[~np.isfinite(aligned)] = np.inf
        out = np.minimum(out, aligned)
    np.fill_diagonal(out, 0.0)
    return pd.DataFrame(out, index=residues, columns=residues)


def residue_distance_map_multi(
    pdb_files: Sequence[str],
    chain_id: str = "A",
    atom: str = "CA",
    chain_ids: Sequence[str] | None = None,
    mapping_entries: Sequence[dict[str, Any] | None] | None = None,
) -> pd.DataFrame:
    """Minimum inter-residue distance across PDB/mmCIF files."""
    if chain_ids is not None and len(chain_ids) != len(pdb_files):
        raise ValueError("chain_ids must have one entry per pdb file")
    if mapping_entries is not None and len(mapping_entries) != len(pdb_files):
        raise ValueError("mapping_entries must have one entry per pdb file")
    maps = []
    for i, pdb_file in enumerate(pdb_files):
        cid = chain_id if chain_ids is None else chain_ids[i]
        mapping = None if mapping_entries is None else mapping_entries[i]
        maps.append(
            residue_distance_map(
                pdb_file, chain_id=cid, atom=atom, mapping_entry=mapping
            )
        )
    return min_distance_across_structures(maps)


def contact_frame_from_distances(dist_df: pd.DataFrame, cutoff: float = 8.0) -> pd.DataFrame:
    """Thresholded contact map: 1 if 0 < d <= cutoff, else 0."""
    values = dist_df.to_numpy(dtype=float)
    contact = ((values <= cutoff) & (values > 0)).astype(float)
    np.fill_diagonal(contact, 0.0)
    return pd.DataFrame(contact, index=dist_df.index, columns=dist_df.columns)


def build_contact_graph(dist_df: pd.DataFrame, cutoff: float = 8.0) -> nx.Graph:
    """Undirected contact graph. Edge attribute ``distance`` is the input distance."""
    graph = nx.Graph()
    graph.add_nodes_from(int(i) for i in dist_df.index)
    labels = [int(i) for i in dist_df.index]
    values = dist_df.to_numpy(dtype=float)
    for i, j in combinations(range(len(labels)), 2):
        distance = float(values[i, j])
        if 0 < distance <= cutoff:
            graph.add_edge(labels[i], labels[j], distance=distance)
    return graph


def build_contact_graph_from_contact(contact_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    labels = [int(i) for i in contact_df.index]
    graph.add_nodes_from(labels)
    values = contact_df.to_numpy(dtype=float)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if values[i, j] > 0:
                graph.add_edge(labels[i], labels[j])
    return graph


def _freesasa_areas(structure, chain_id: str) -> dict[int, float]:
    try:
        import freesasa
    except ImportError as exc:
        raise ImportError(
            "RSA uses FreeSASA. Install it with `pip install freesasa`."
        ) from exc

    result, _classes = freesasa.calcBioPDB(structure, options=_FREESASA_OPTIONS)
    by_chain = result.residueAreas()
    if chain_id not in by_chain:
        raise KeyError(f"FreeSASA did not return chain {chain_id!r}")
    areas = by_chain[chain_id]
    model = next(structure.get_models())
    rsa: dict[int, float] = {}
    for residue in _iter_standard_residues(model[chain_id]):
        resname = residue.get_resname()
        if resname not in MAX_SASA:
            continue
        _het, resseq, icode = residue.id
        icode = "" if icode in (" ", "") else str(icode).strip()
        area = None
        for key in (f"{resseq}{icode}", str(resseq)):
            if key in areas:
                area = areas[key]
                break
        if area is None:
            continue
        rsa[int(resseq)] = float(area.total) / MAX_SASA[resname]
    return rsa


def residue_rsa_map(
    pdb_file: str,
    chain_id: str = "A",
    mapping_entry: dict[str, Any] | None = None,
    cap_at_monomer: bool = True,
) -> dict[int, float]:
    """Relative solvent accessibility from FreeSASA.

    SASA is calculated on every protein chain in the file so a partner
    chain buries the interface. Heteroatoms and hydrogens are omitted.
    RSA is that SASA divided by ``MAX_SASA``. When ``cap_at_monomer`` is
    set and a partner chain is present, RSA is capped at the value from
    the query chain alone (a partner can only shield solvent).

    Keys are UniProt positions when ``mapping_entry`` is given, otherwise
    PDB residue numbers.
    """
    structure = _first_model(load_structure(pdb_file))
    model = next(structure.get_models())
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id!r} is not in {pdb_file}")

    assembly = _freesasa_areas(structure, chain_id)
    partner = any(chain.id != chain_id for chain in model.get_chains())
    if cap_at_monomer and partner:
        monomer = _freesasa_areas(_chain_only(structure, chain_id), chain_id)
        pdb_rsa = {
            pos: min(value, monomer[pos]) if pos in monomer else value
            for pos, value in assembly.items()
        }
    else:
        pdb_rsa = assembly

    if mapping_entry is None:
        return pdb_rsa

    pdb_to_uni = pdb_to_uniprot_map(mapping_entry)
    mapped: dict[int, float] = {}
    for pdb_pos, rsa in pdb_rsa.items():
        uni = pdb_to_uni.get(int(pdb_pos))
        if uni is None:
            continue
        uni = int(uni)
        mapped[uni] = max(mapped.get(uni, rsa), float(rsa))
    return mapped


def attach_max_rsa(
    df: pd.DataFrame,
    rsa_map: Mapping[int, float],
    position_col: str = "target_pos",
) -> pd.DataFrame:
    """Add ``max_rsa`` by mapping ``position_col`` through ``rsa_map``."""
    out = df.copy()
    out["max_rsa"] = out[position_col].map(lambda pos: rsa_map.get(int(pos), np.nan))
    return out


def position_rsa_map(df: pd.DataFrame, col: str = "max_rsa") -> dict[int, float]:
    """One RSA per ``target_pos`` (maximum across rows)."""
    if col not in df.columns or "target_pos" not in df.columns:
        raise ValueError(f"df must contain 'target_pos' and {col!r}")
    tmp = (
        df[["target_pos", col]]
        .dropna()
        .assign(target_pos=lambda frame: frame["target_pos"].astype(int))
        .groupby("target_pos", as_index=False)[col]
        .max()
    )
    return dict(zip(tmp["target_pos"].astype(int), tmp[col].astype(float)))
