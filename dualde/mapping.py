"""PDB fetch/clean, UniProt mapping, and FASTA I/O.

Each ``position_mapping`` fragment is an ungapped correspondence among UniProt,
PDB ATOM residue numbers, and (optionally) SEQRES indices. Fragments may
differ; there is no global SEQRES→PDB offset.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import urllib.request
from typing import Any, Mapping, Sequence

import pandas as pd
from Bio.PDB import PDBIO, PDBList, PDBParser, Select, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

logger = logging.getLogger(__name__)

SIFTS_CSV_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"
)

_SIFTS_RENAME = {
    "PDB": "pdb_id",
    "CHAIN": "pdb_chain",
    "SP_PRIMARY": "uniprot_ac",
    "SP_BEG": "uniprot_start",
    "SP_END": "uniprot_end",
    "PDB_BEG": "pdb_start",
    "PDB_END": "pdb_end",
    "RES_BEG": "seqres_start",
    "RES_END": "seqres_end",
    "coord_start": "pdb_start",
    "coord_end": "pdb_end",
    "resseq_start": "seqres_start",
    "resseq_end": "seqres_end",
}


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


def fetch_pdb_structure(pdb_id: str, output_dir: str, file_format: str = "pdb") -> str:
    """Download a PDB or mmCIF entry from RCSB. Returns the local path."""
    os.makedirs(output_dir, exist_ok=True)
    pdb_file = PDBList().retrieve_pdb_file(
        pdb_id, pdir=output_dir, file_format=file_format
    )
    if not pdb_file or not os.path.exists(pdb_file):
        raise FileNotFoundError(f"Failed to download PDB {pdb_id} into {output_dir}")
    logger.info("Downloaded %s to %s", pdb_id, pdb_file)
    return pdb_file


def load_structure(path: str):
    """Parse a PDB or mmCIF file."""
    if str(path).lower().endswith((".cif", ".mmcif")):
        from Bio.PDB import MMCIFParser

        return MMCIFParser(QUIET=True).get_structure("prot", path)
    return PDBParser(QUIET=True).get_structure("prot", path)


def unique_pdb_ids(entries: Sequence[dict[str, Any]]) -> list[str]:
    """Unique PDB identifiers from mapping hits (all chains)."""
    return sorted({str(e.get("pdb_id", "")).lower() for e in entries if e.get("pdb_id")})


def _structure_chain_ids(path: str) -> set[str]:
    ids: set[str] = set()
    for model in load_structure(path):
        for chain in model:
            ids.add(chain.id)
    return ids


def _existing_structure_path(pdb_id: str, output_dir: str) -> str | None:
    pid = pdb_id.lower()
    candidates = [
        os.path.join(output_dir, f"{pid}.pdb"),
        os.path.join(output_dir, f"{pid}.cif"),
        os.path.join(output_dir, f"pdb{pid}.ent"),
        os.path.join(output_dir, f"{pid}.ent"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def select_structures(
    entries: Sequence[dict[str, Any]],
    output_dir: str,
    fetch_missing: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Resolve a local structure file for every homologous PDB hit.

    Returns ``(entries, {pdb_id: path})``. Hits are **not** filtered by RMSD;
    use :func:`dualde.rmsd.rmsd_heatmap_after_alignment` later to pick a
    representative for stability prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    needed_chains: dict[str, set[str]] = {}
    for entry in entries:
        pid = str(entry.get("pdb_id", "")).lower()
        needed_chains.setdefault(pid, set()).add(str(entry.get("pdb_chain", "")))

    paths: dict[str, str] = {}
    for pdb_id in unique_pdb_ids(entries):
        path = _existing_structure_path(pdb_id, output_dir)
        if path is None and fetch_missing:
            try:
                fetched = fetch_pdb_structure(pdb_id, output_dir, file_format="pdb")
            except Exception:
                fetched = fetch_pdb_structure(pdb_id, output_dir, file_format="mmCif")
            dest = os.path.join(
                output_dir,
                f"{pdb_id}.pdb" if fetched.endswith((".pdb", ".ent")) else f"{pdb_id}.cif",
            )
            if os.path.abspath(fetched) != os.path.abspath(dest):
                shutil.copy2(fetched, dest)
                if os.path.basename(fetched).startswith("pdb") and fetched.endswith(".ent"):
                    os.remove(fetched)
            path = dest
        if path is None:
            logger.warning("No structure file for %s in %s", pdb_id, output_dir)
            continue
        present = _structure_chain_ids(path)
        want = needed_chains.get(pdb_id, set())
        if want and not (want & present) and fetch_missing:
            try:
                fetched = fetch_pdb_structure(pdb_id, output_dir, file_format="mmCif")
                dest = os.path.join(output_dir, f"{pdb_id}.cif")
                if os.path.abspath(fetched) != os.path.abspath(dest):
                    shutil.copy2(fetched, dest)
                path = dest
            except Exception as exc:
                logger.warning("Could not fetch mmCIF for %s: %s", pdb_id, exc)
        paths[pdb_id] = path
    return list(entries), paths


def clean_pdb_structure(
    pdb_file: str, chain: str = "A", output_dir: str | None = None
) -> str:
    """Keep standard amino acids from one chain; drop ligands, water, other chains."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", pdb_file)
    pdb_basename = os.path.basename(pdb_file)
    stem = pdb_basename[:-4] if pdb_basename.endswith((".pdb", ".PDB")) else pdb_basename
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
    return output_file


def extract_sequence_from_pdb(pdb_file: str, chain: str = "A") -> tuple[str, list[int]]:
    """ATOM-record sequence and residue numbers for one chain."""
    structure = load_structure(pdb_file)
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


def query_identity(
    pdb_path: str,
    chain: str,
    query_seq: str,
    mapping_entry: dict[str, Any] | None = None,
    first_index: int = 1,
) -> float:
    """
    Sequence identity of a PDB chain to the query, over the **full query length**.

    Mapped residues that match the query count as identities; unmapped query
    positions count as mismatches (same definition as the original ≥60% filter).
    """
    if not query_seq:
        return 0.0
    seq, resnums = extract_sequence_from_pdb(pdb_path, chain=chain)
    if not seq:
        return 0.0
    if mapping_entry is not None:
        pdb2uni = pdb_to_uniprot_map(mapping_entry)
        matches = 0
        for aa, pdb_pos in zip(seq, resnums):
            uni = pdb2uni.get(int(pdb_pos))
            if uni is None:
                continue
            idx = int(uni) - first_index
            if 0 <= idx < len(query_seq) and query_seq[idx] == aa:
                matches += 1
        return matches / len(query_seq)

    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aln = aligner.align(query_seq, seq)[0]
    matches = sum(a == b and a != "-" for a, b in zip(str(aln[0]), str(aln[1])))
    return matches / len(query_seq)


def identity_table(
    entries: Sequence[dict[str, Any]],
    pdb_paths: Mapping[str, str],
    query_seq: str,
    first_index: int = 1,
) -> pd.DataFrame:
    """One row per mapping hit with query identity over the full sequence."""
    rows = []
    for entry in entries:
        pid = str(entry.get("pdb_id", "")).lower()
        chain = str(entry.get("pdb_chain", ""))
        path = pdb_paths.get(pid)
        ident = float("nan")
        if path:
            ident = query_identity(
                path, chain, query_seq, mapping_entry=entry, first_index=first_index
            )
        rows.append(
            {
                "pdb_id": pid,
                "pdb_chain": chain,
                "identity_percent": ident,
                "n_mapped": len(pdb_to_uniprot_map(entry)),
            }
        )
    return pd.DataFrame(rows)


def filter_hits_by_identity(
    entries: Sequence[dict[str, Any]],
    pdb_paths: Mapping[str, str],
    query_seq: str,
    min_identity: float = 0.6,
    first_index: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Keep mapping hits whose chain identity to the query is ≥ ``min_identity``."""
    table = identity_table(entries, pdb_paths, query_seq, first_index=first_index)
    keep: set[tuple[str, str]] = set()
    if not table.empty:
        passed = table.loc[table["identity_percent"] >= float(min_identity)]
        keep = set(zip(passed["pdb_id"].astype(str), passed["pdb_chain"].astype(str)))
    kept = [
        e
        for e in entries
        if (str(e.get("pdb_id", "")).lower(), str(e.get("pdb_chain", ""))) in keep
    ]
    return kept, table


_PDBE_JSON_CACHE: dict[str, Any] = {}
_PREFERRED_CHAIN_CACHE: dict[str, set[str]] = {}

PDBE_SUMMARY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"
PDBE_ASSEMBLY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/assembly/{pdb_id}"
PDBE_MOLECULES_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}"


def _pdbe_json(url: str) -> Any:
    if url in _PDBE_JSON_CACHE:
        return _PDBE_JSON_CACHE[url]
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    _PDBE_JSON_CACHE[url] = data
    return data


def _auth_chain_from_asym(pdb_id: str) -> dict[str, str]:
    """Map PDBe struct_asym IDs to author chain IDs used in local PDB/mmCIF files."""
    pid = pdb_id.lower()
    records = _pdbe_json(PDBE_MOLECULES_URL.format(pdb_id=pid)).get(pid, [])
    mapping: dict[str, str] = {}
    for mol in records:
        auth = [str(c) for c in (mol.get("in_chains") or [])]
        asym = [str(c) for c in (mol.get("in_struct_asyms") or [])]
        if len(auth) == len(asym) and auth:
            for a, c in zip(asym, auth):
                mapping[a] = c
        else:
            for c in auth:
                mapping.setdefault(c, c)
    return mapping


def preferred_assembly_chains(pdb_id: str) -> set[str]:
    """
    Author chain IDs of polymer copies in the PDBe **preferred** biological
    assembly (e.g. 5bon A+B, not the other crystal-packing dimers).
    """
    pid = str(pdb_id).lower()
    if pid in _PREFERRED_CHAIN_CACHE:
        return set(_PREFERRED_CHAIN_CACHE[pid])
    summary = _pdbe_json(PDBE_SUMMARY_URL.format(pdb_id=pid)).get(pid, [{}])[0]
    preferred_ids = {
        str(a.get("assembly_id"))
        for a in summary.get("assemblies") or []
        if a.get("preferred")
    }
    if not preferred_ids:
        preferred_ids = {"1"}
    assemblies = _pdbe_json(PDBE_ASSEMBLY_URL.format(pdb_id=pid)).get(pid, [])
    auth_from_asym = _auth_chain_from_asym(pid)
    chains: set[str] = set()
    for assembly in assemblies:
        if str(assembly.get("assembly_id")) not in preferred_ids:
            continue
        for ent in assembly.get("entities") or []:
            moltype = str(ent.get("molecule_type") or "").lower()
            if "polypeptide" not in moltype:
                continue
            for ch in ent.get("in_chains") or []:
                key = str(ch)
                chains.add(auth_from_asym.get(key, key))
    _PREFERRED_CHAIN_CACHE[pid] = chains
    logger.info("PDBe preferred assembly for %s: %s", pid, sorted(chains))
    return set(chains)


def homolog_chain_ids(
    pdb_path: str,
    query_seq: str,
    min_identity: float = 0.6,
    mapping_entries: Sequence[dict[str, Any]] | None = None,
    first_index: int = 1,
) -> set[str]:
    """PDB chain IDs whose identity to the query is ≥ ``min_identity``."""
    by_chain = {str(e.get("pdb_chain", "")): e for e in (mapping_entries or [])}
    homologs: set[str] = set()
    for chain in _structure_chain_ids(pdb_path):
        ident = query_identity(
            pdb_path,
            chain,
            query_seq,
            mapping_entry=by_chain.get(chain),
            first_index=first_index,
        )
        if ident >= float(min_identity):
            homologs.add(chain)
    return homologs


def load_structural_mapping_json(path: str) -> list[dict[str, Any]]:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("chain_hits"), list):
        return data["chain_hits"]
    raise ValueError(
        f"Expected a JSON array or an object with 'chain_hits' at {path!r}"
    )


def save_mapping_json(entries: Sequence[dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(entries), f, indent=2)
        f.write("\n")
    return path


def select_mapping_entry(
    entries: Sequence[dict[str, Any]],
    pdb_id: str,
    chain: str,
    uniprot_id: str | None = None,
) -> dict[str, Any]:
    def _match(entry, require_uniprot: bool) -> bool:
        if str(entry.get("pdb_id", "")).lower() != str(pdb_id).lower():
            return False
        if str(entry.get("pdb_chain", "")).strip() != str(chain).strip():
            return False
        if not require_uniprot or not uniprot_id:
            return True
        uid = str(uniprot_id).strip()
        return entry.get("uniprot_id") == uid or entry.get("uniprot_ac") == uid

    matches = [e for e in entries if _match(e, True)]
    if not matches and uniprot_id:
        matches = [e for e in entries if _match(e, False)]
    if not matches:
        raise ValueError(
            f"No mapping entry for pdb_id={pdb_id!r} chain={chain!r} uniprot={uniprot_id!r}"
        )
    return matches[0]


def _as_resnum(value: object) -> int:
    text = str(value).strip()
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
        raise ValueError(f"Could not parse residue number from {value!r}")
    return int(num)


def _first(seg: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in seg and seg[key] is not None and str(seg[key]).strip() != "":
            return _as_resnum(seg[key])
    raise KeyError(f"Segment missing any of {keys}: {seg}")


def fragment_table(entry: dict[str, Any]) -> pd.DataFrame:
    """One row per mapped fragment (UniProt, PDB ATOM, SEQRES)."""
    rows = []
    for i, seg in enumerate(entry.get("position_mapping") or [], start=1):
        u0 = _first(seg, "uniprot_start", "target_start")
        u1 = _first(seg, "uniprot_end", "target_end")
        p0 = _first(seg, "pdb_start", "coord_start", "structure_start")
        p1 = _first(seg, "pdb_end", "coord_end", "structure_end")
        row = {
            "fragment": i,
            "uniprot_start": u0,
            "uniprot_end": u1,
            "pdb_start": p0,
            "pdb_end": p1,
            "n_residues": u1 - u0 + 1,
        }
        try:
            row["seqres_start"] = _first(seg, "seqres_start", "resseq_start")
            row["seqres_end"] = _first(seg, "seqres_end", "resseq_end")
        except KeyError:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def pdb_to_uniprot_map(entry: dict[str, Any]) -> dict[int, int]:
    """``{PDB ATOM residue number → UniProt position}`` from mapping fragments."""
    out: dict[int, int] = {}
    for _, row in fragment_table(entry).iterrows():
        u0, u1 = int(row["uniprot_start"]), int(row["uniprot_end"])
        p0, p1 = int(row["pdb_start"]), int(row["pdb_end"])
        n_u, n_p = u1 - u0 + 1, p1 - p0 + 1
        if n_u != n_p:
            raise ValueError(
                f"Fragment length mismatch: UniProt [{u0},{u1}] vs PDB [{p0},{p1}] "
                f"(pdb={entry.get('pdb_id')} chain={entry.get('pdb_chain')})"
            )
        for k in range(n_u):
            out[p0 + k] = u0 + k
    return out


def download_sifts_table(dest_path: str, url: str = SIFTS_CSV_URL) -> str:
    """Download the EBI SIFTS pdb_chain_uniprot CSV and gunzip it."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    gz_path = dest_path if dest_path.endswith(".gz") else dest_path + ".gz"
    with urllib.request.urlopen(url) as resp, open(gz_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    if dest_path.endswith(".gz"):
        return dest_path
    with gzip.open(gz_path, "rb") as src, open(dest_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(gz_path)
    return dest_path


def _mapping_entry_from_hits(
    hits: pd.DataFrame,
    pdb_id: str,
    chain: str,
    uniprot_id: str | None,
) -> dict[str, Any]:
    pid = pdb_id.lower()
    if hits.empty:
        raise ValueError(f"No SIFTS rows for pdb_id={pdb_id!r} chain={chain!r}")

    if uniprot_id:
        uid = str(uniprot_id).strip()
        ac = hits["uniprot_ac"].astype(str)
        extra = hits["uniprot_id"].astype(str) if "uniprot_id" in hits.columns else ac
        filtered = hits[(ac == uid) | (extra == uid)]
        if not filtered.empty:
            hits = filtered

    unique_ac = hits["uniprot_ac"].unique()
    if len(unique_ac) > 1:
        raise ValueError(
            f"Multiple UniProt accessions on {pdb_id} chain {chain}: {list(unique_ac)}. "
            "Pass uniprot_id to disambiguate."
        )

    position_mapping = []
    for _, row in hits.iterrows():
        u0, u1 = _as_resnum(row["uniprot_start"]), _as_resnum(row["uniprot_end"])
        if "pdb_start" in hits.columns and pd.notna(row.get("pdb_start")):
            p0, p1 = _as_resnum(row["pdb_start"]), _as_resnum(row["pdb_end"])
        else:
            p0, p1 = _as_resnum(row["seqres_start"]), _as_resnum(row["seqres_end"])
        seg = {
            "uniprot_start": u0,
            "uniprot_end": u1,
            "pdb_start": p0,
            "pdb_end": p1,
        }
        if "seqres_start" in hits.columns and pd.notna(row.get("seqres_start")):
            seg["seqres_start"] = _as_resnum(row["seqres_start"])
            seg["seqres_end"] = _as_resnum(row["seqres_end"])
        position_mapping.append(seg)

    first = hits.iloc[0]
    pdb_nums = [_as_resnum(s["pdb_start"]) for s in position_mapping] + [
        _as_resnum(s["pdb_end"]) for s in position_mapping
    ]
    uni_nums = [_as_resnum(s["uniprot_start"]) for s in position_mapping] + [
        _as_resnum(s["uniprot_end"]) for s in position_mapping
    ]
    return {
        "pdb_id": pid,
        "pdb_chain": chain,
        "uniprot_ac": str(first["uniprot_ac"]),
        "uniprot_id": uniprot_id or str(first["uniprot_ac"]),
        "pdb_start": min(pdb_nums),
        "pdb_end": max(pdb_nums),
        "uniprot_start": min(uni_nums),
        "uniprot_end": max(uni_nums),
        "structure_type": "pdb",
        "position_mapping": position_mapping,
    }


def mapping_entry_from_sifts(
    sifts_table_path: str,
    pdb_id: str,
    chain: str,
    uniprot_id: str | None,
) -> dict[str, Any]:
    """Build a mapping record from a SIFTS chain–UniProt table (one row per fragment)."""
    if not os.path.isfile(sifts_table_path):
        raise FileNotFoundError(
            f"SIFTS table not found: {sifts_table_path}. Download from {SIFTS_CSV_URL}."
        )

    table = pd.read_csv(sifts_table_path, comment="#")
    table = table.rename(columns={c: _SIFTS_RENAME.get(c, c) for c in table.columns})
    pid = pdb_id.lower()
    hits = table[(table["pdb_id"].astype(str).str.lower() == pid) & (table["pdb_chain"] == chain)]
    return _mapping_entry_from_hits(hits, pdb_id, chain, uniprot_id)


def mapping_entries_from_sifts(
    sifts_table_path: str,
    uniprot_id: str,
    pdb_ids: Sequence[str] | None = None,
    uniprot_name: str | None = None,
) -> list[dict[str, Any]]:
    """One mapping record per PDB chain for a UniProt accession or identifier."""
    if not os.path.isfile(sifts_table_path):
        raise FileNotFoundError(
            f"SIFTS table not found: {sifts_table_path}. Download from {SIFTS_CSV_URL}."
        )
    table = pd.read_csv(sifts_table_path, comment="#")
    table = table.rename(columns={c: _SIFTS_RENAME.get(c, c) for c in table.columns})
    uid = str(uniprot_id).strip()
    ac = table["uniprot_ac"].astype(str)
    extra = table["uniprot_id"].astype(str) if "uniprot_id" in table.columns else ac
    hits = table[(ac == uid) | (extra == uid)]
    if pdb_ids:
        keep = {str(p).lower() for p in pdb_ids}
        hits = hits[hits["pdb_id"].astype(str).str.lower().isin(keep)]
    if hits.empty:
        raise ValueError(f"No SIFTS rows for uniprot={uniprot_id!r}")

    entries = []
    for (pid, chain), group in hits.groupby(["pdb_id", "pdb_chain"], sort=True):
        entry = _mapping_entry_from_hits(group, str(pid), str(chain), uniprot_id=uid)
        if uniprot_name:
            entry["uniprot_id"] = uniprot_name
        entries.append(entry)
    return entries
