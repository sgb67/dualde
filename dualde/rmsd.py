"""CA superposition RMSD heatmap and hierarchical clustering."""

from __future__ import annotations

import glob
import os
import re
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBIO, PDBParser, Select, Superimposer, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

from dualde.mapping import clean_pdb_structure, fetch_pdb_structure

_DEFAULT_HIGHLIGHT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def _parse_highlight_pdb_ids(
    highlight_pdb: Optional[Union[str, List[str]]],
    labels: List[str],
    max_highlights: int = 5,
) -> Dict[str, str]:
    if highlight_pdb is None:
        return {}

    if isinstance(highlight_pdb, str):
        requested = [
            token.strip().lower()
            for token in re.split(r"[,;\s]+", highlight_pdb)
            if token.strip()
        ]
    else:
        requested = [str(token).strip().lower() for token in highlight_pdb if str(token).strip()]

    if not requested:
        return {}
    if len(requested) > max_highlights:
        raise ValueError(f"At most {max_highlights} PDB IDs can be highlighted")

    label_lookup = {}
    for label in labels:
        label_str = str(label)
        label_lookup[label_str.lower()] = label_str
        pdb_match = re.search(r"([0-9][a-z0-9]{3})", label_str.lower())
        if pdb_match:
            label_lookup.setdefault(pdb_match.group(1), label_str)

    matched: Dict[str, str] = {}
    missing = []
    for pdb_id in requested:
        if pdb_id in label_lookup:
            matched[label_lookup[pdb_id]] = pdb_id
        else:
            missing.append(pdb_id)

    if missing:
        raise ValueError(
            f"Highlight PDB IDs not found in RMSD matrix labels: {missing}. "
            f"Available labels: {list(labels)}"
        )
    return matched


def _prepare_rmsd_distance_matrix(rmsd_df: pd.DataFrame) -> np.ndarray:
    mat = rmsd_df.to_numpy(dtype=float).copy()
    if mat.size == 0:
        return mat
    finite = mat[np.isfinite(mat)]
    fill_value = float(np.nanmax(finite)) if finite.size else 1.0
    mat = np.where(np.isfinite(mat), mat, fill_value * 1.5)
    np.fill_diagonal(mat, 0.0)
    return mat


def _cluster_order_rmsd_df(rmsd_df: pd.DataFrame, method: str = "average") -> pd.DataFrame:
    n = len(rmsd_df)
    if n <= 2:
        return rmsd_df.copy()

    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    mat_filled = _prepare_rmsd_distance_matrix(rmsd_df)
    order = dendrogram(
        linkage(squareform(mat_filled, checks=False), method=method),
        no_plot=True,
    )["leaves"]
    return rmsd_df.iloc[order, order]


def count_rmsd_clusters_at_cutoff(
    rmsd_df: pd.DataFrame,
    cutoff: float,
    linkage_method: str = "average",
) -> Tuple[int, np.ndarray]:
    """Count clusters in an RMSD distance matrix at a distance cutoff (Angstrom)."""
    labels = list(rmsd_df.index)
    n = len(labels)
    if n == 0:
        return 0, np.array([], dtype=int)
    if n == 1:
        return 1, np.array([1], dtype=int)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    mat_filled = _prepare_rmsd_distance_matrix(rmsd_df)
    condensed = squareform(mat_filled, checks=False)
    cluster_ids = fcluster(
        linkage(condensed, method=linkage_method),
        t=float(cutoff),
        criterion="distance",
    )
    return int(len(set(cluster_ids))), np.asarray(cluster_ids, dtype=int)


def summarize_rmsd_clusters_at_cutoff(
    rmsd_df: pd.DataFrame,
    cutoff: float,
    linkage_method: str = "average",
) -> pd.DataFrame:
    """Return one row per structure with its cluster id at the given RMSD cutoff."""
    n_clusters, cluster_ids = count_rmsd_clusters_at_cutoff(
        rmsd_df,
        cutoff=cutoff,
        linkage_method=linkage_method,
    )
    summary = pd.DataFrame(
        {
            "label": list(rmsd_df.index),
            "cluster": cluster_ids,
        }
    )
    summary.attrs["n_clusters"] = n_clusters
    summary.attrs["cutoff"] = cutoff
    return summary


def plot_rmsd_cluster_counts(
    rmsd_df: pd.DataFrame,
    cutoffs: Optional[Union[List[float], np.ndarray]] = None,
    cutoff: Optional[float] = None,
    max_cutoff: float = 5.0,
    step: float = 0.25,
    linkage_method: str = "average",
    ax=None,
    figsize: Tuple[float, float] = (8, 4),
):
    """Plot number of RMSD clusters versus distance cutoff."""
    import matplotlib.pyplot as plt

    if cutoffs is None:
        cutoffs = np.arange(step, max_cutoff + step * 0.5, step)
    cutoffs = np.asarray(cutoffs, dtype=float)

    cluster_counts = [
        count_rmsd_clusters_at_cutoff(rmsd_df, c, linkage_method=linkage_method)[0]
        for c in cutoffs
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(cutoffs, cluster_counts, marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("RMSD cutoff (A)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("RMSD cluster count vs cutoff")
    ax.grid(True, alpha=0.3)

    if cutoff is not None:
        n_at_cutoff, _cluster_ids = count_rmsd_clusters_at_cutoff(
            rmsd_df,
            cutoff=cutoff,
            linkage_method=linkage_method,
        )
        ax.axvline(cutoff, color="#e41a1c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.scatter([cutoff], [n_at_cutoff], color="#e41a1c", s=60, zorder=5)
        print(
            f"RMSD cutoff = {cutoff:.2f} A -> {n_at_cutoff} cluster(s) "
            f"across {len(rmsd_df)} structure(s)"
        )
        cluster_map = summarize_rmsd_clusters_at_cutoff(
            rmsd_df,
            cutoff=cutoff,
            linkage_method=linkage_method,
        )
        print(cluster_map.groupby("cluster")["label"].apply(list).to_string())

    plt.tight_layout()
    return ax, pd.DataFrame({"cutoff": cutoffs, "n_clusters": cluster_counts})


def plot_rmsd_heatmap(
    rmsd_df: pd.DataFrame,
    highlight_pdb: Optional[Union[str, List[str]]] = None,
    cluster: bool = True,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (10, 8),
    annot: bool = False,
    fmt: str = ".2f",
    linkage_method: str = "average",
    highlight_colors: Optional[List[str]] = None,
    ax=None,
):
    """Plot an RMSD heatmap with optional highlighting for up to five PDB IDs."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    if rmsd_df.shape[0] != rmsd_df.shape[1]:
        raise ValueError("rmsd_df must be square")
    if not rmsd_df.index.equals(rmsd_df.columns):
        raise ValueError("rmsd_df index and columns must match")

    plot_df = _cluster_order_rmsd_df(rmsd_df, method=linkage_method) if cluster else rmsd_df.copy()
    labels = [str(x) for x in plot_df.index]
    highlight_map = _parse_highlight_pdb_ids(highlight_pdb, labels)
    highlight_labels = list(highlight_map.keys())

    colors = highlight_colors or _DEFAULT_HIGHLIGHT_COLORS
    label_to_color = {
        label: colors[idx % len(colors)]
        for idx, label in enumerate(highlight_labels)
    }

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar_kws={"label": "RMSD (A)"},
        ax=ax,
    )

    for tick in ax.get_xticklabels():
        label = tick.get_text()
        if label in label_to_color:
            tick.set_color(label_to_color[label])
            tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        if label in label_to_color:
            tick.set_color(label_to_color[label])
            tick.set_fontweight("bold")

    n = len(plot_df)
    for label in highlight_labels:
        if label not in plot_df.index:
            continue
        idx = list(plot_df.index).index(label)
        color = label_to_color[label]
        ax.add_patch(Rectangle((0, idx), n, 1, fill=False, edgecolor=color, linewidth=2.5, clip_on=False))
        ax.add_patch(Rectangle((idx, 0), 1, n, fill=False, edgecolor=color, linewidth=2.5, clip_on=False))

    title = "Pairwise RMSD"
    if highlight_labels:
        highlighted = ", ".join(highlight_map[label] for label in highlight_labels)
        title += f" (highlighted: {highlighted})"
    ax.set_title(title)

    if highlight_labels:
        handles = [
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=label_to_color[label], linewidth=2.5)
            for label in highlight_labels
        ]
        ax.legend(
            handles,
            [highlight_map[label] for label in highlight_labels],
            title="Highlighted",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )

    plt.tight_layout()
    return ax, plot_df


def rmsd_heatmap_after_alignment(
    pdb_ids: Optional[List[str]] = None,
    pdb_files: Optional[List[str]] = None,
    chain_specs: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
    labels: Optional[List[str]] = None,
    cmap: str = "viridis",
    min_atoms: int = 10,
    cluster: bool = True,
    show: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare cleaned structures, compute pairwise RMSD after CA-based superposition.

    If chain_specs provides a chain for a structure, use only that chain.
    Otherwise test all eligible chains and report the minimum RMSD.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if pdb_ids is None and pdb_files is None:
        raise ValueError("Either pdb_ids or pdb_files must be provided")
    if pdb_ids is not None and pdb_files is not None:
        raise ValueError("Provide only one of pdb_ids or pdb_files")
    if chain_specs is None:
        chain_specs = {}

    def _align_atoms_for_rmsd(structure_a, chain_a: str, structure_b, chain_b: str):
        def _extract_residues_with_ca(structure, chain_id: str):
            for model in structure:
                if chain_id not in model:
                    return [], ""
                chain_obj = model[chain_id]
                residues = []
                seq = []
                for residue in chain_obj:
                    if not is_aa(residue):
                        continue
                    resname = residue.get_resname()
                    if resname not in protein_letters_3to1:
                        continue
                    if "CA" not in residue:
                        continue
                    residues.append(residue)
                    seq.append(protein_letters_3to1[resname])
                return residues, "".join(seq)
            return [], ""

        residues_a, seq_a = _extract_residues_with_ca(structure_a, chain_a)
        residues_b, seq_b = _extract_residues_with_ca(structure_b, chain_b)

        if len(seq_a) < min_atoms or len(seq_b) < min_atoms:
            return [], []

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5

        aln = aligner.align(seq_a, seq_b)[0]
        a_aln = str(aln[0])
        b_aln = str(aln[1])

        fixed_atoms = []
        mobile_atoms = []
        ia = 0
        ib = 0
        for aa, bb in zip(a_aln, b_aln):
            if aa != "-" and bb != "-":
                fixed_atoms.append(residues_a[ia]["CA"])
                mobile_atoms.append(residues_b[ib]["CA"])
                ia += 1
                ib += 1
            elif aa == "-":
                ib += 1
            else:
                ia += 1

        if len(fixed_atoms) < min_atoms:
            return [], []
        return fixed_atoms, mobile_atoms

    def _clean_pdb_all_chains(pdb_file: str, output_dir_local: Optional[str] = None) -> str:
        parser_local = PDBParser(QUIET=True)
        struct_local = parser_local.get_structure("prot", pdb_file)
        io = PDBIO()
        io.set_structure(struct_local)

        pdb_basename = os.path.basename(pdb_file)
        if pdb_basename.endswith((".pdb", ".PDB")):
            pdb_stem = pdb_basename[:-4]
        else:
            pdb_stem = pdb_basename

        if output_dir_local is None:
            output_file = os.path.join(os.path.dirname(pdb_file), f"{pdb_stem}_cleaned_allchains.pdb")
        else:
            os.makedirs(output_dir_local, exist_ok=True)
            output_file = os.path.join(output_dir_local, f"{pdb_stem}_cleaned_allchains.pdb")

        class ProteinOnlySelect(Select):
            def accept_residue(self, residue):
                return is_aa(residue)

        io.save(output_file, ProteinOnlySelect())
        return output_file

    def _reuse_existing_cleaned(input_path: str, search_dir: Optional[str], suffix: str) -> Optional[str]:
        if input_path.endswith(f"{suffix}.pdb") and os.path.exists(input_path):
            return input_path
        base_no_ext = os.path.splitext(os.path.basename(input_path))[0]
        candidates = [os.path.join(os.path.dirname(input_path), f"{base_no_ext}{suffix}.pdb")]
        if search_dir:
            candidates.append(os.path.join(search_dir, f"{base_no_ext}{suffix}.pdb"))
            candidates.extend(glob.glob(os.path.join(search_dir, f"*{base_no_ext}*{suffix}.pdb")))
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _available_protein_chains(structure_obj) -> List[str]:
        for model in structure_obj:
            chains = []
            for ch in model:
                ca_count = sum(1 for r in ch if is_aa(r) and ("CA" in r))
                if ca_count >= min_atoms:
                    chains.append(ch.id)
            return chains
        return []

    cleaned_files: List[str] = []
    chain_options: List[List[str]] = []
    names: List[str] = []

    if pdb_ids is not None:
        input_items = pdb_ids
        get_name = lambda x: x
        get_raw = lambda x: fetch_pdb_structure(x, output_dir=output_dir or ".")
    else:
        input_items = pdb_files
        get_name = lambda x: os.path.splitext(os.path.basename(x))[0]
        get_raw = lambda x: x

    for item in input_items:
        name = get_name(item)
        specified_chain = chain_specs.get(item) or chain_specs.get(name)
        raw_file = get_raw(item)
        if raw_file is None or not os.path.exists(str(raw_file)):
            print(f"  Skipping {name}: structure download/file unavailable")
            continue
        names.append(name)

        if specified_chain is not None:
            existing_cleaned = _reuse_existing_cleaned(raw_file, output_dir, "_cleaned")
            cleaned = existing_cleaned or clean_pdb_structure(
                raw_file, chain=specified_chain, output_dir=output_dir
            )
            chain_list = [specified_chain]
        else:
            existing_cleaned = _reuse_existing_cleaned(raw_file, output_dir, "_cleaned_allchains")
            cleaned = existing_cleaned or _clean_pdb_all_chains(raw_file, output_dir_local=output_dir)
            tmp_structure = PDBParser(QUIET=True).get_structure("tmp", cleaned)
            chain_list = _available_protein_chains(tmp_structure)
            if not chain_list:
                print(f"  Warning: no eligible protein chains found in {cleaned}")

        cleaned_files.append(cleaned)
        chain_options.append(chain_list)

    n = len(cleaned_files)
    parser = PDBParser(QUIET=True)
    structures = [parser.get_structure(f"s{i}", path) for i, path in enumerate(cleaned_files)]

    rmsd_matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        rmsd_matrix[i, i] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            best = np.inf
            for ci, cj in product(chain_options[i], chain_options[j]):
                fixed_atoms, mobile_atoms = _align_atoms_for_rmsd(structures[i], ci, structures[j], cj)
                if len(fixed_atoms) < min_atoms:
                    continue
                sup = Superimposer()
                sup.set_atoms(fixed_atoms, mobile_atoms)
                if sup.rms < best:
                    best = float(sup.rms)
            rmsd = best if best < np.inf else np.nan
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    if labels is None:
        labels = names
    rmsd_df = pd.DataFrame(rmsd_matrix, index=labels, columns=labels)

    plot_df = rmsd_df.copy()
    if cluster and n > 2:
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform

            mat = plot_df.to_numpy(dtype=float)
            max_val = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 1.0
            mat_filled = np.where(np.isnan(mat), max_val, mat)
            np.fill_diagonal(mat_filled, 0.0)
            order = dendrogram(linkage(squareform(mat_filled), method="average"), no_plot=True)["leaves"]
            plot_df = plot_df.iloc[order, order]
        except Exception as exc:
            print(f"  Clustering skipped: {exc}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        plot_df,
        annot=False,
        fmt=".3f",
        cmap=cmap,
        square=True,
        cbar_kws={"label": "RMSD (A)"},
        ax=ax,
    )
    ax.set_title("Pairwise RMSD (min over chain pairs, after structural alignment)")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return rmsd_df, cleaned_files


def unique_pdb_ids(hits: pd.DataFrame, chain: str | None = None) -> List[str]:
    """PDB IDs from a hits table, optionally restricted to one chain."""
    work = hits
    if chain is not None and "pdb_chain" in work.columns:
        work = work.loc[work["pdb_chain"].astype(str) == chain]
    return sorted(work["pdb_id"].astype(str).str.lower().unique())
