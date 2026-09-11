"""SIFTS / JSON mapping, PDB fetch, chain cleaning, and SEQRES offset."""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import urllib.request
from typing import Any, Sequence

import pandas as pd
from Bio.PDB import PDBIO, PDBList, PDBParser, Select, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

logger = logging.getLogger(__name__)


def read_fasta(path: str) -> tuple[str, str]:
    """Return (header, concatenated sequence) from a single-record FASTA."""
    header = ""
    chunks: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header or chunks:
                    break
                header = line[1:].split()[0]
            else:
                chunks.append(line)
    return header, "".join(chunks)

SIFTS_CSV_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"
)


def fetch_pdb_structure(pdb_id: str, output_dir: str) -> str:
    """Download a PDB entry from RCSB into output_dir. Returns the local path."""
    os.makedirs(output_dir, exist_ok=True)
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")
    if not pdb_file or not os.path.exists(pdb_file):
        raise FileNotFoundError(f"Failed to download PDB {pdb_id} into {output_dir}")
    logger.info("Downloaded %s to %s", pdb_id, pdb_file)
    return pdb_file


def clean_pdb_structure(
    pdb_file: str, chain: str = "A", output_dir: str | None = None
) -> str:
    """Keep standard amino acids from one chain; drop ligands, water, other chains."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", pdb_file)

    pdb_basename = os.path.basename(pdb_file)
    if pdb_basename.endswith((".pdb", ".PDB")):
        stem = pdb_basename[:-4]
    else:
        stem = pdb_basename

    if output_dir is None:
        output_file = os.path.join(os.path.dirname(pdb_file), f"{stem}_cleaned.pdb")
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stem}_cleaned.pdb")

    class ChainProteinSelect(Select):
        def accept_chain(self, chain_obj):
            return chain_obj.id == chain

        def accept_residue(self, residue):
            return is_aa(residue)

    io = PDBIO()
    io.set_structure(struct)
    io.save(output_file, ChainProteinSelect())
    logger.info("Cleaned PDB: kept chain %r -> %s", chain, output_file)
    return output_file


def extract_sequence_from_pdb(pdb_file: str, chain: str = "A") -> tuple[str, list[int]]:
    """ATOM-record sequence and residue numbers for one chain (standard amino acids)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    sequence: list[str] = []
    residue_numbers: list[int] = []
    for model in structure:
        for chain_obj in model:
            if chain_obj.id != chain:
                continue
            for residue in chain_obj:
                if not is_aa(residue):
                    continue
                res_name = residue.get_resname()
                if res_name not in protein_letters_3to1:
                    continue
                sequence.append(protein_letters_3to1[res_name])
                residue_numbers.append(residue.get_id()[1])
    return "".join(sequence), residue_numbers


def parse_seqres_one_letter(pdb_file: str, chain: str) -> str:
    """SEQRES 1-letter sequence for chain, or empty string if SEQRES is absent."""
    seq: list[str] = []
    try:
        with open(pdb_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("SEQRES"):
                    continue
                parts = line.split()
                if len(parts) < 5 or parts[2] != chain:
                    continue
                for code in parts[4:]:
                    seq.append(protein_letters_3to1.get(code, "X"))
    except OSError:
        return ""
    return "".join(seq)


def compute_seqres_to_pdb_offset(
    seqres_seq: str,
    modeled_seq: str,
    modeled_resnums: list[int],
    prefix_len: int = 10,
) -> int | None:
    """Offset such that 1-indexed SEQRES position s corresponds to PDB residue s + offset."""
    if not seqres_seq or not modeled_seq or not modeled_resnums:
        return None
    k = min(prefix_len, len(modeled_seq))
    needle = modeled_seq[:k]
    candidates: list[int] = []
    start = 0
    while True:
        idx = seqres_seq.find(needle, start)
        if idx < 0:
            break
        candidates.append(modeled_resnums[0] - (idx + 1))
        start = idx + 1
    if len(candidates) == 1:
        return candidates[0]
    return None


def find_seqres_source(
    pdb_path: str, pdb_id: str, chain: str, search_dirs: Sequence[str]
) -> tuple[str, str]:
    """Return (path, SEQRES sequence) from the first file that has SEQRES for chain."""
    pid = pdb_id.lower()
    candidates = [pdb_path]
    raw_names = [
        f"pdb{pid}.ent",
        f"{pid}.pdb",
        f"{pid}.ent",
        f"{pdb_id}.pdb",
        f"{pid}_cleaned.pdb",
    ]
    for directory in search_dirs:
        if not directory:
            continue
        for name in raw_names:
            candidates.append(os.path.join(directory, name))

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        seqres = parse_seqres_one_letter(path, chain)
        if seqres:
            return path, seqres
    return "", ""


def seqres_to_pdb_offset(
    pdb_path: str,
    pdb_id: str,
    chain: str,
    search_dirs: Sequence[str] | None = None,
) -> int:
    """Compute SEQRES→PDB residue-number offset, defaulting to 0 if it cannot be inferred."""
    dirs = list(search_dirs or [])
    dirs.insert(0, os.path.dirname(pdb_path))
    src, seqres = find_seqres_source(pdb_path, pdb_id, chain, dirs)
    if not seqres:
        logger.warning("No SEQRES for %s chain %s; using offset 0", pdb_id, chain)
        return 0
    modeled_seq, modeled_resnums = extract_sequence_from_pdb(pdb_path, chain)
    offset = compute_seqres_to_pdb_offset(seqres, modeled_seq, modeled_resnums)
    if offset is None:
        logger.warning(
            "Could not compute SEQRES→PDB offset from %s; using 0", src
        )
        return 0
    logger.info("SEQRES offset for %s chain %s: %+d (from %s)", pdb_id, chain, offset, src)
    return offset


def load_structural_mapping_json(path: str) -> list[dict[str, Any]]:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("chain_hits"), list):
        return data["chain_hits"]
    raise ValueError(
        f"Expected JSON array or object with 'chain_hits' at {path!r}, got {type(data).__name__}"
    )


def save_mapping_json(entries: Sequence[dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(entries), f, indent=2)
        f.write("\n")
    return path


def entry_matches_structure(
    entry: dict[str, Any],
    pdb_id: str,
    chain: str,
    uniprot_id: str | None,
) -> bool:
    if str(entry.get("pdb_id", "")).lower() != str(pdb_id).lower():
        return False
    if str(entry.get("pdb_chain", "")).strip() != str(chain).strip():
        return False
    if not uniprot_id:
        return True
    uid = str(uniprot_id).strip()
    return entry.get("uniprot_id") == uid or entry.get("uniprot_ac") == uid


def select_mapping_entry(
    entries: Sequence[dict[str, Any]],
    pdb_id: str,
    chain: str,
    uniprot_id: str | None = None,
) -> dict[str, Any]:
    matches = [
        e for e in entries if entry_matches_structure(e, pdb_id, chain, uniprot_id)
    ]
    if not matches and uniprot_id:
        matches = [
            e for e in entries if entry_matches_structure(e, pdb_id, chain, None)
        ]
    if not matches:
        raise ValueError(
            f"No JSON mapping entry for pdb_id={pdb_id!r} chain={chain!r} "
            f"uniprot={uniprot_id!r}"
        )
    return matches[0]


def pdb_to_uniprot_map_from_position_mapping(
    entry: dict[str, Any],
    seqres_to_pdb_offset: int = 0,
) -> dict[int, int]:
    """Build {pdb_residue_number -> UniProt position (1-based)} from one JSON record."""
    out: dict[int, int] = {}
    segments = entry.get("position_mapping") or []
    for seg in segments:
        ts = int(seg["target_start"])
        te = int(seg["target_end"])
        ss = int(seg["structure_start"])
        se = int(seg["structure_end"])
        len_t = te - ts + 1
        len_s = se - ss + 1
        if len_t != len_s:
            raise ValueError(
                f"Segment length mismatch in position_mapping: "
                f"target [{ts},{te}] len={len_t} vs structure [{ss},{se}] len={len_s} "
                f"(pdb={entry.get('pdb_id')} chain={entry.get('pdb_chain')})"
            )
        for k in range(len_t):
            out[ss + k + seqres_to_pdb_offset] = ts + k
    return out


def download_sifts_table(dest_path: str, url: str = SIFTS_CSV_URL) -> str:
    """Download the EBI SIFTS pdb_chain_uniprot CSV (gzipped) and gunzip it."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    gz_path = dest_path if dest_path.endswith(".gz") else dest_path + ".gz"
    logger.info("Downloading SIFTS table from %s", url)
    with urllib.request.urlopen(url) as resp, open(gz_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    if dest_path.endswith(".gz"):
        return dest_path
    with gzip.open(gz_path, "rb") as src, open(dest_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(gz_path)
    return dest_path


def mapping_entry_from_sifts(
    sifts_table_path: str,
    pdb_id: str,
    chain: str,
    uniprot_id: str | None,
) -> dict[str, Any]:
    """
    Build a distance-pipeline-style mapping record from a SIFTS CSV.

    structure_* values are SEQRES (resseq) indices, matching the JSON schema
    produced by search_protein in the distance pipeline notebooks.
    """
    if not os.path.isfile(sifts_table_path):
        raise FileNotFoundError(
            f"SIFTS table not found: {sifts_table_path}. "
            f"Download from {SIFTS_CSV_URL} or pass a mapping JSON."
        )

    usecols = [
        "pdb_id",
        "pdb_chain",
        "uniprot_ac",
        "resseq_start",
        "resseq_end",
        "uniprot_start",
        "uniprot_end",
    ]
    table = pd.read_csv(
        sifts_table_path,
        comment="#",
        usecols=lambda c: c in usecols or c == "uniprot_id",
    )
    pid = pdb_id.lower()
    hits = table[(table["pdb_id"] == pid) & (table["pdb_chain"] == chain)]
    if hits.empty:
        raise ValueError(
            f"No SIFTS rows for pdb_id={pdb_id!r} chain={chain!r} in {sifts_table_path}"
        )

    if uniprot_id:
        uid = str(uniprot_id).strip()
        ac_col = hits["uniprot_ac"].astype(str)
        if "uniprot_id" in hits.columns:
            id_col = hits["uniprot_id"].astype(str)
            filtered = hits[(ac_col == uid) | (id_col == uid)]
        else:
            filtered = hits[ac_col == uid]
        if not filtered.empty:
            hits = filtered
        else:
            logger.warning(
                "SIFTS uniprot filter %r matched no rows for %s chain %s; "
                "using pdb_id+chain hits",
                uid,
                pdb_id,
                chain,
            )

    unique_ac = hits["uniprot_ac"].unique()
    if len(unique_ac) > 1:
        raise ValueError(
            f"Multiple UniProt accessions on {pdb_id} chain {chain}: {list(unique_ac)}. "
            "Pass uniprot_id (accession) to disambiguate."
        )

    position_mapping = []
    for _, row in hits.iterrows():
        position_mapping.append(
            {
                "target_start": int(row["uniprot_start"]),
                "target_end": int(row["uniprot_end"]),
                "structure_start": int(row["resseq_start"]),
                "structure_end": int(row["resseq_end"]),
            }
        )

    first = hits.iloc[0]
    return {
        "pdb_id": pid,
        "pdb_chain": chain,
        "uniprot_ac": str(first["uniprot_ac"]),
        "uniprot_id": uniprot_id or str(first["uniprot_ac"]),
        "resseq_start": int(hits["resseq_start"].min()),
        "resseq_end": int(hits["resseq_end"].max()),
        "uniprot_start": int(hits["uniprot_start"].min()),
        "uniprot_end": int(hits["uniprot_end"].max()),
        "structure_type": "pdb",
        "position_mapping": position_mapping,
    }


def pdb_hits_from_mapping(entries: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Unique (pdb_id, pdb_chain) rows from a mapping JSON."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (str(entry.get("pdb_id", "")).lower(), str(entry.get("pdb_chain", "")).strip())
        if key in seen or not key[0]:
            continue
        seen.add(key)
        rows.append(
            {
                "pdb_id": key[0],
                "pdb_chain": key[1],
                "uniprot_ac": entry.get("uniprot_ac"),
                "uniprot_id": entry.get("uniprot_id"),
            }
        )
    return pd.DataFrame(rows)
