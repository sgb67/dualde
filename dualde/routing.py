"""Coupling-grid construction and Yen k-shortest paths for information transfer."""

from __future__ import annotations

import math
from itertools import islice, product

import networkx as nx
import numpy as np
import pandas as pd


def build_coupling_graph(coupling_grid, prob_threshold=0.2):
    """Undirected graph: nodes = residues, edges = couplings above threshold."""
    G = nx.Graph()
    G.add_nodes_from(coupling_grid.index.astype(int))

    edges = (
        coupling_grid.stack()
        .reset_index()
        .rename(columns={"level_0": "i", "level_1": "j", 0: "probability"})
    )
    edges = edges[(edges["probability"] >= prob_threshold) & (edges["i"] != edges["j"])]

    edges["a"] = edges[["i", "j"]].min(axis=1).astype(int)
    edges["b"] = edges[["i", "j"]].max(axis=1).astype(int)
    edges = edges.groupby(["a", "b"], as_index=False)["probability"].max()

    for _, row in edges.iterrows():
        p = float(row["probability"])
        G.add_edge(int(row["a"]), int(row["b"]), probability=p, cost=1.0 - p)

    return G


def _path_edge_probs(G, path):
    return [G[path[i]][path[i + 1]]["probability"] for i in range(len(path) - 1)]


def _path_cost(G, path):
    return sum(G[path[i]][path[i + 1]]["cost"] for i in range(len(path) - 1))


def find_coupling_paths(
    coupling_grid,
    func_positions,
    sbi_nonfunc_positions,
    prob_threshold=0.2,
    n_paths=5,
    weighted=True,
):
    """
    For each (source, target) pair, return up to n_paths shortest simple paths
    in the coupling graph (Yen). weight = 1 - probability if weighted=True.
    """
    func_positions = sorted({int(p) for p in func_positions})
    sbi_nonfunc_positions = sorted({int(p) for p in sbi_nonfunc_positions})

    G = build_coupling_graph(coupling_grid, prob_threshold=prob_threshold)
    graph_nodes = set(G.nodes())
    weight = "cost" if weighted else None

    records, pair_stats = [], []

    for src, tgt in product(func_positions, sbi_nonfunc_positions):
        if src == tgt:
            continue

        missing = [p for p in (src, tgt) if p not in graph_nodes]
        if missing:
            pair_stats.append(
                {
                    "source": src,
                    "target": tgt,
                    "n_paths_found": 0,
                    "status": f"missing node(s): {missing}",
                }
            )
            continue

        try:
            paths = list(islice(nx.shortest_simple_paths(G, src, tgt, weight=weight), n_paths))
        except nx.NetworkXNoPath:
            paths = []

        pair_stats.append(
            {
                "source": src,
                "target": tgt,
                "n_paths_found": len(paths),
                "status": "ok" if paths else "no path",
            }
        )

        for rank, path in enumerate(paths, start=1):
            edge_probs = _path_edge_probs(G, path)
            path_cost = _path_cost(G, path) if weighted else float(len(path) - 1)
            records.append(
                {
                    "source": src,
                    "target": tgt,
                    "path_rank": rank,
                    "path": path,
                    "path_str": " -> ".join(map(str, path)),
                    "n_edges": len(path) - 1,
                    "path_cost": path_cost,
                    "path_coupling_score": -path_cost,
                    "min_edge_prob": min(edge_probs) if edge_probs else np.nan,
                    "mean_edge_prob": float(np.mean(edge_probs)) if edge_probs else np.nan,
                    "edge_probs": edge_probs,
                }
            )

    paths_df = pd.DataFrame(records)
    pairs_df = pd.DataFrame(pair_stats)

    n_pairs = len(func_positions) * len(sbi_nonfunc_positions)
    summary = {
        "n_func_positions": len(func_positions),
        "n_sbi_nonfunc_positions": len(sbi_nonfunc_positions),
        "n_position_pairs": n_pairs,
        "max_possible_paths": n_pairs * n_paths,
        "n_paths_found": len(paths_df),
        "pairs_with_any_path": int((pairs_df["n_paths_found"] > 0).sum()) if len(pairs_df) else 0,
        "edge_count_overall": (
            paths_df["n_edges"].describe().to_dict() if len(paths_df) else {}
        ),
        "prob_threshold": prob_threshold,
        "n_paths": n_paths,
        "weighted": weighted,
        "n_graph_nodes": G.number_of_nodes(),
        "n_graph_edges": G.number_of_edges(),
    }
    return paths_df, pairs_df, summary, G


def prepare_scores_df(
    all_merged_df,
    activity_wt,
    abundance_wt,
    functional_dist_cutoff=3.5,
):
    """
    Annotate SBI membership and functional_site on a protein's merged DMS df.
    Expects columns: target_pos, activity, stability, minimum_distance.
    """
    df = all_merged_df.copy()
    required = {"target_pos", "activity", "stability", "minimum_distance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"all_merged_df missing columns: {sorted(missing)}")

    df["functional_site"] = df["minimum_distance"] < functional_dist_cutoff
    df["sbi_membership"] = np.where(
        (df["stability"] > abundance_wt) & (df["activity"] < activity_wt),
        "SBI",
        "non-SBI",
    )
    return df


def load_coupling_grid(
    couplings_file,
    dist_cutoff=4.5,
    seq_len=None,
):
    """
    Load EVcouplings Compared_all CSV (path or DataFrame), zero probability for
    dist > dist_cutoff, pivot to square probability grid.
    """
    if isinstance(couplings_file, pd.DataFrame):
        coup = couplings_file.copy()
        source_label = "<DataFrame>"
    else:
        coup = pd.read_csv(couplings_file)
        source_label = couplings_file

    need = {"i", "j", "probability", "dist"}
    missing = need - set(coup.columns)
    if missing:
        raise ValueError(f"couplings file missing columns: {sorted(missing)}")

    n_before = int((coup["probability"] > 0).sum())
    coup.loc[coup["dist"] > dist_cutoff, "probability"] = 0.0
    n_after = int((coup["probability"] > 0).sum())

    i_max = int(coup[["i", "j"]].max().max())
    if seq_len is None:
        seq_len = i_max
    positions = np.arange(1, int(seq_len) + 1)

    coupling_grid = (
        coup.pivot(index="i", columns="j", values="probability")
        .reindex(index=positions, columns=positions, fill_value=0.0)
        .fillna(0.0)
    )
    coupling_grid.index.name = "i"
    coupling_grid.columns.name = "j"

    meta = {
        "couplings_file": source_label,
        "dist_cutoff": dist_cutoff,
        "seq_len": int(seq_len),
        "n_raw_positive_prob": n_before,
        "n_positive_prob_after_dist_filter": n_after,
        "max_ij_in_file": i_max,
        "grid_shape": coupling_grid.shape,
        "n_nonzero_grid": int((coupling_grid.values > 0).sum()),
    }
    return coupling_grid, coup, meta


def load_functional_residues(functional_residues_file, functional_dist_cutoff=3.5):
    """Load a functional-residues table (path or DataFrame)."""
    if isinstance(functional_residues_file, pd.DataFrame):
        func_df = functional_residues_file.copy()
    else:
        func_df = pd.read_csv(functional_residues_file)

    need = {"target_pos", "minimum_distance", "residue_name"}
    missing = need - set(func_df.columns)
    if missing:
        raise ValueError(f"functional residues file missing: {sorted(missing)}")

    func_df["target_pos"] = func_df["target_pos"].astype(int)
    func_df = func_df.loc[func_df["minimum_distance"] < functional_dist_cutoff].copy()
    return func_df


def select_path_endpoints(
    df_scores,
    functional_residues_df,
    active_site_ligands,
    score_col,
    low_score_percentile=1.0,
):
    """
    Sources = functional positions contacting active_site_ligands.
    Targets = unique positions among the bottom low_score_percentile% of score_col.
    """
    active_site_ligands = set(active_site_ligands)

    if score_col not in df_scores.columns:
        raise ValueError(f"score_col '{score_col}' not in df_scores")

    ligands_present = sorted(functional_residues_df["residue_name"].dropna().unique())
    unknown = sorted(active_site_ligands - set(ligands_present))

    active_site_positions = sorted(
        {
            int(p)
            for p in functional_residues_df.loc[
                functional_residues_df["residue_name"].isin(active_site_ligands),
                "target_pos",
            ].unique()
        }
    )

    n_take = max(1, math.floor(len(df_scores) * float(low_score_percentile) / 100.0))
    bottom = df_scores.dropna(subset=[score_col]).nsmallest(n_take, score_col)

    target_positions = sorted({int(p) for p in bottom["target_pos"].unique()})

    checks = {
        "ligands_requested": sorted(active_site_ligands),
        "ligands_present_in_file": ligands_present,
        "ligands_requested_but_absent": unknown,
        "n_active_site_sources": len(active_site_positions),
        "n_bottom_mutants": len(bottom),
        "n_targets": len(target_positions),
        "active_site_positions": active_site_positions,
        "target_positions": target_positions,
        "bottom_mutants": bottom,
        "n_take": n_take,
        "score_col": score_col,
        "low_score_percentile": low_score_percentile,
    }
    return active_site_positions, target_positions, checks


def print_pipeline_checks(
    df_scores,
    coupling_meta,
    endpoint_checks,
    summary,
    activity_wt,
    abundance_wt,
    paths_df,
):
    print("=" * 60)
    print("PIPELINE CHECKS")
    print("=" * 60)

    print("\n[1] Scores / cutoffs")
    print(f"  n mutants:              {len(df_scores)}")
    print(f"  activity_wt:            {activity_wt}")
    print(f"  abundance_wt:           {abundance_wt}")
    print(f"  n SBI mutants:          {(df_scores['sbi_membership'] == 'SBI').sum()}")
    print(f"  n functional_site True: {df_scores['functional_site'].sum()}")
    print(f"  target_pos range:       {df_scores['target_pos'].min()}–{df_scores['target_pos'].max()}")

    print("\n[2] Couplings")
    for k, v in coupling_meta.items():
        print(f"  {k}: {v}")

    c = endpoint_checks
    print("\n[3] Endpoints")
    print(f"  score_col:              {c['score_col']}")
    print(f"  low_score_percentile:   {c['low_score_percentile']}% -> n_take={c['n_take']}")
    print(f"  ligands requested:      {c['ligands_requested']}")
    print(f"  ligands in file:        {c['ligands_present_in_file']}")
    if c["ligands_requested_but_absent"]:
        print(f"  WARNING absent ligands: {c['ligands_requested_but_absent']}")
    print(f"  active-site SOURCES:    {c['n_active_site_sources']}  {c['active_site_positions']}")
    print(f"  bottom mutants:         {c['n_bottom_mutants']}")
    print(f"  TARGETS (unique pos):   {c['n_targets']}  {c['target_positions']}")

    src_in = [p for p in c["active_site_positions"] if 1 <= p <= coupling_meta["seq_len"]]
    tgt_in = [p for p in c["target_positions"] if 1 <= p <= coupling_meta["seq_len"]]
    print("\n[4] Numbering sanity")
    print(f"  sources in [1, seq_len]: {len(src_in)}/{c['n_active_site_sources']}")
    print(f"  targets in [1, seq_len]: {len(tgt_in)}/{c['n_targets']}")
    if len(src_in) < c["n_active_site_sources"] or len(tgt_in) < c["n_targets"]:
        print("  WARNING: some endpoints outside coupling grid numbering")

    print("\n[5] Path-finding result")
    for k in [
        "prob_threshold",
        "n_paths",
        "weighted",
        "n_graph_nodes",
        "n_graph_edges",
        "n_paths_found",
        "max_possible_paths",
        "pairs_with_any_path",
        "n_position_pairs",
    ]:
        print(f"  {k}: {summary.get(k)}")
    if summary.get("edge_count_overall"):
        print("  n_edges describe:")
        print(pd.Series(summary["edge_count_overall"]))

    if len(paths_df):
        print("\n[6] paths_df preview")
        print(
            paths_df[
                [
                    "source",
                    "target",
                    "path_rank",
                    "n_edges",
                    "path_cost",
                    "min_edge_prob",
                    "mean_edge_prob",
                    "path_str",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        bad = paths_df["min_edge_prob"] < summary["prob_threshold"]
        print(f"\n  paths with min_edge_prob < threshold: {int(bad.sum())} (expect 0)")
        print(f"  unique sources used: {paths_df['source'].nunique()}")
        print(f"  unique targets used: {paths_df['target'].nunique()}")
    else:
        print("\n[6] paths_df is EMPTY — loosen prob_threshold or check endpoints/numbering")


def run_information_transfer_pipeline(
    all_merged_df,
    couplings_file,
    functional_residues_file,
    active_site_ligands,
    activity_wt,
    abundance_wt,
    score_col,
    low_score_percentile=1.0,
    prob_threshold=0.7,
    n_paths=5,
    weighted=True,
    dist_cutoff=5,
    functional_dist_cutoff=3.5,
    seq_len=None,
    verbose=True,
):
    """End-to-end: scores → couplings grid → endpoints → paths_df."""
    df_scores = prepare_scores_df(
        all_merged_df,
        activity_wt=activity_wt,
        abundance_wt=abundance_wt,
        functional_dist_cutoff=functional_dist_cutoff,
    )

    coupling_grid, coup_raw, coupling_meta = load_coupling_grid(
        couplings_file,
        dist_cutoff=dist_cutoff,
        seq_len=seq_len,
    )

    functional_residues_df = load_functional_residues(
        functional_residues_file,
        functional_dist_cutoff=functional_dist_cutoff,
    )

    sources, targets, endpoint_checks = select_path_endpoints(
        df_scores=df_scores,
        functional_residues_df=functional_residues_df,
        active_site_ligands=active_site_ligands,
        score_col=score_col,
        low_score_percentile=low_score_percentile,
    )

    if len(sources) == 0:
        raise ValueError(
            "No active-site source positions. Check active_site_ligands vs residue_name in file."
        )
    if len(targets) == 0:
        raise ValueError("No target positions. Check score_col / percentile.")

    paths_df, pairs_df, summary, coupling_G = find_coupling_paths(
        coupling_grid=coupling_grid,
        func_positions=sources,
        sbi_nonfunc_positions=targets,
        prob_threshold=prob_threshold,
        n_paths=n_paths,
        weighted=weighted,
    )

    if verbose:
        print_pipeline_checks(
            df_scores,
            coupling_meta,
            endpoint_checks,
            summary,
            activity_wt,
            abundance_wt,
            paths_df,
        )

    return {
        "paths_df": paths_df,
        "pairs_df": pairs_df,
        "summary": summary,
        "coupling_G": coupling_G,
        "coupling_grid": coupling_grid,
        "df_scores": df_scores,
        "sources": sources,
        "targets": targets,
        "endpoint_checks": endpoint_checks,
        "coupling_meta": coupling_meta,
    }
