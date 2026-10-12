"""CPU null tests for an observed residue path.

Three tests, each with a discrete or a continuous statistic:

* contact-graph decoys of the same length (Cα–Cα graph; ``cutoff`` is
  ``contact_distance_cutoff`` and is used only here)
* the same decoys, kept only when their mean RSA matches the path
* random residue sets matched residue-by-residue on RSA

Discrete statistic ``both_low`` is the mean, over positions, of the fraction
of variants with activity below ``activity_wt`` and stability below
``abundance_wt``. Continuous statistic ``stability`` is the mean (or median)
of per-position aggregates of ``score_col`` (default ``stability``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from dualde.structure_graph import (
    build_contact_graph,
    build_contact_graph_from_contact,
    position_rsa_map,
)

STATISTICS = {"both_low", "stability"}
ALTERNATIVES = {"less", "greater", "two-sided"}


def _as_path(path) -> list[int]:
    if isinstance(path, str):
        text = path.strip()
        if "->" in text:
            return [int(part) for part in text.split("->")]
        text = text.strip("[]")
        return [int(part) for part in text.split(",") if part.strip()]
    return [int(p) for p in path]


def _path_positions(path: Sequence[int], exclude_endpoints: bool) -> list[int]:
    positions = [int(p) for p in path]
    if exclude_endpoints and len(positions) > 2:
        positions = positions[1:-1]
    return sorted(set(positions))


def _mean_rsa(positions: Sequence[int], rsa_map: Mapping[int, float]) -> float:
    if any(int(p) not in rsa_map for p in positions):
        return float("nan")
    return float(np.mean([rsa_map[int(p)] for p in positions]))


def _resolve_alternative(statistic: str, alternative: str | None) -> str:
    if alternative is None:
        return "greater"
    alternative = alternative.lower()
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be 'less', 'greater', or 'two-sided'")
    return alternative


def _per_position_values(
    df: pd.DataFrame,
    statistic: str,
    score_col: str,
    agg: str,
    activity_wt: float | None,
    abundance_wt: float | None,
) -> dict[int, float]:
    if statistic == "both_low":
        if activity_wt is None or abundance_wt is None:
            raise ValueError("both_low requires activity_wt and abundance_wt")
        missing = {"target_pos", "activity", "stability"} - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {sorted(missing)}")
        work = df[["target_pos", "activity", "stability"]].dropna().copy()
        work["target_pos"] = work["target_pos"].astype(int)
        hit = (work["activity"] < float(activity_wt)) & (
            work["stability"] < float(abundance_wt)
        )
        return hit.groupby(work["target_pos"]).mean().astype(float).to_dict()

    if score_col not in df.columns or "target_pos" not in df.columns:
        raise ValueError(f"df must contain 'target_pos' and {score_col!r}")
    work = df[["target_pos", score_col]].dropna().copy()
    work["target_pos"] = work["target_pos"].astype(int)
    grouped = work.groupby("target_pos")[score_col]
    series = grouped.mean() if agg == "mean" else grouped.median()
    return series.astype(float).to_dict()


def _set_statistic(
    positions: Sequence[int],
    values: Mapping[int, float],
    agg: str,
) -> float:
    vals = [values[int(p)] for p in positions if int(p) in values and np.isfinite(values[int(p)])]
    if not vals:
        return float("nan")
    if agg == "mean":
        return float(np.mean(vals))
    return float(np.median(vals))


def _p_value(null_stats: np.ndarray, observed: float, alternative: str) -> tuple[float, str]:
    n = len(null_stats)
    if alternative == "less":
        return (float(np.sum(null_stats <= observed)) + 1.0) / (n + 1.0), "obs <= null"
    if alternative == "greater":
        return (float(np.sum(null_stats >= observed)) + 1.0) / (n + 1.0), "obs >= null"
    p_low = (float(np.sum(null_stats <= observed)) + 1.0) / (n + 1.0)
    p_high = (float(np.sum(null_stats >= observed)) + 1.0) / (n + 1.0)
    return min(1.0, 2.0 * min(p_low, p_high)), "two-sided"


def sample_decoy_paths(
    graph: nx.Graph,
    n_edges: int,
    n_samples: int = 500,
    seed: int = 0,
    fixed_start: int | None = None,
    fixed_end: int | None = None,
    allowed_starts: Sequence[int] | None = None,
    allowed_ends: Sequence[int] | None = None,
    max_tries_factor: int = 50,
) -> list[list[int]]:
    """Sample simple paths with exactly ``n_edges`` on ``graph``."""
    rng = np.random.default_rng(seed)
    n_edges = int(n_edges)
    nodes = [int(n) for n in graph.nodes()]
    if not nodes:
        raise ValueError("Empty contact graph")
    adjacency = {int(n): [int(v) for v in graph.neighbors(n)] for n in nodes}

    if fixed_start is not None:
        fixed_start = int(fixed_start)
        if fixed_start not in adjacency:
            raise ValueError(f"fixed_start {fixed_start} not in contact graph")
        start_pool = [fixed_start]
    else:
        allowed = None if allowed_starts is None else {int(s) for s in allowed_starts if int(s) in adjacency}
        start_pool = nodes if allowed is None else [n for n in nodes if n in allowed]
    if not start_pool:
        raise ValueError("No valid start nodes in graph")

    if fixed_end is not None:
        fixed_end = int(fixed_end)
        if fixed_end not in adjacency:
            raise ValueError(f"fixed_end {fixed_end} not in contact graph")
        end_set = {fixed_end}
    else:
        end_set = None if allowed_ends is None else {int(e) for e in allowed_ends if int(e) in adjacency}

    sampled: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    unconstrained = (
        fixed_start is None
        and fixed_end is None
        and allowed_starts is None
        and allowed_ends is None
    )
    max_tries = int(n_samples) * int(max_tries_factor)

    for _ in range(max_tries):
        if len(sampled) >= n_samples:
            break
        start = int(rng.choice(start_pool))
        path = [start]
        used = {start}
        ok = True
        for _step in range(n_edges):
            nbrs = [n for n in adjacency[path[-1]] if n not in used]
            if not nbrs:
                ok = False
                break
            nxt = int(rng.choice(nbrs))
            path.append(nxt)
            used.add(nxt)
        if not ok:
            continue
        if end_set is not None and path[-1] not in end_set:
            continue
        key = tuple(path)
        if unconstrained:
            if key in seen or tuple(reversed(path)) in seen:
                continue
        elif key in seen:
            continue
        seen.add(key)
        sampled.append(path)
    return sampled


def sample_rsa_matched_set(
    observed_positions: Sequence[int],
    pool_positions: Sequence[int],
    rsa_map: Mapping[int, float],
    rng: np.random.Generator,
    rsa_tol: float = 0.1,
) -> list[int]:
    """Sample a pool residue for each observed RSA, without replacement.

    A candidate must have ``|RSA - observed RSA| <= rsa_tol``. If none
    remain, the closest remaining residue is used. Observed positions are
    never included.
    """
    obs = [int(p) for p in observed_positions]
    obs_set = set(obs)
    pool = [int(p) for p in pool_positions if int(p) in rsa_map and int(p) not in obs_set]
    if len(pool) < len(obs):
        raise ValueError(
            "Pool too small for RSA matching after excluding observed: "
            f"need {len(obs)}, have {len(pool)}"
        )

    order = list(obs)
    rng.shuffle(order)
    available = set(pool)
    chosen = []
    for pos in order:
        target = float(rsa_map[int(pos)])
        cands = np.array(sorted(available), dtype=int)
        diffs = np.abs(np.array([rsa_map[int(c)] for c in cands], dtype=float) - target)
        within = cands[diffs <= float(rsa_tol)]
        pick = int(rng.choice(within)) if len(within) else int(cands[int(np.argmin(diffs))])
        chosen.append(pick)
        available.remove(pick)
    return sorted(chosen)


def _orient_path(
    path: Sequence[int],
    fixed_start: int | None = None,
    fixed_end: int | None = None,
    info_source: int | None = None,
) -> list[int]:
    observed = _as_path(path)
    if fixed_start is not None:
        fixed_start = int(fixed_start)
        if observed[0] != fixed_start:
            if observed[-1] == fixed_start:
                observed = observed[::-1]
            else:
                raise ValueError(
                    f"observed_path must start (or end) at fixed_start={fixed_start}"
                )
    if fixed_end is not None:
        fixed_end = int(fixed_end)
        if observed[-1] != fixed_end:
            if observed[0] == fixed_end and fixed_start is None:
                observed = observed[::-1]
            else:
                raise ValueError(f"observed_path must end at fixed_end={fixed_end}")
    if info_source is not None:
        info_source = int(info_source)
        if observed[-1] != info_source and observed[0] == info_source:
            observed = observed[::-1]
    return observed


def _check_statistic(statistic: str, agg: str) -> tuple[str, str]:
    statistic = statistic.lower()
    agg = agg.lower()
    if statistic not in STATISTICS:
        raise ValueError("statistic must be 'both_low' or 'stability'")
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    return statistic, agg


def _null_uniqueness(paths_or_sets) -> tuple[int, int, int]:
    keys = [tuple(p) for p in paths_or_sets]
    n_used = len(keys)
    n_unique = len(set(keys))
    return n_used, n_unique, n_used - n_unique


def _print_result(
    *,
    title: str,
    observed_path: Sequence[int],
    positions: Sequence[int],
    statistic: str,
    score_col: str,
    agg: str,
    alternative: str,
    alt_label: str,
    observed: float,
    null_stats: np.ndarray,
    p_value: float,
    n_null: int,
    n_unique: int,
    n_repeated: int,
    extra: list[str] | None = None,
) -> None:
    label = "inactive+unstable fraction" if statistic == "both_low" else score_col
    print(title)
    print(f"observed path: {' -> '.join(map(str, observed_path))}")
    print(f"positions (k={len(positions)}): {list(positions)}")
    print(f"statistic: {statistic} ({label}, {agg})")
    if extra:
        for line in extra:
            print(line)
    print(f"observed {agg}: {observed:.4f}")
    print(f"null sets used: {len(null_stats)} / requested {n_null}")
    print(f"null uniqueness: {n_unique}/{len(null_stats)} unique ({n_repeated} repeated)")
    print(f"null {agg} ± sd: {null_stats.mean():.4f} ± {null_stats.std():.4f}")
    print(f"p ({alt_label}): {p_value:.4g}")


def test_path_vs_contact_decoys(
    observed_path,
    df: pd.DataFrame,
    dist_df: pd.DataFrame | None = None,
    contact_df: pd.DataFrame | None = None,
    *,
    statistic: str = "stability",
    score_col: str = "stability",
    agg: str = "mean",
    alternative: str | None = None,
    activity_wt: float | None = None,
    abundance_wt: float | None = None,
    cutoff: float = 8.0,
    n_null: int = 1000,
    exclude_endpoints: bool = False,
    match_buriedness: bool = False,
    rsa_tol: float = 0.05,
    rsa_map: Mapping[int, float] | None = None,
    oversample: int = 30,
    fixed_start: int | None = None,
    fixed_end: int | None = None,
    allowed_starts: Sequence[int] | None = None,
    allowed_ends: Sequence[int] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Phenotype of a path versus same-length paths on a contact graph.

    ``match_buriedness=False`` keeps every simple physical decoy.
    ``match_buriedness=True`` keeps decoys whose mean RSA is within
    ``rsa_tol`` of the observed path (average buriedness). The RSA mean
    uses the same residues as the phenotype, including ``exclude_endpoints``.
    """
    statistic, agg = _check_statistic(statistic, agg)
    alternative = _resolve_alternative(statistic, alternative)
    if dist_df is None and contact_df is None:
        raise ValueError("Provide dist_df or contact_df")

    observed_path = _orient_path(observed_path, fixed_start=fixed_start, fixed_end=fixed_end)
    n_edges = len(observed_path) - 1
    if n_edges < 1:
        raise ValueError("observed_path must contain at least 2 residues")

    if contact_df is not None:
        contact_g = build_contact_graph_from_contact(contact_df)
    else:
        contact_g = build_contact_graph(dist_df, cutoff=cutoff)

    missing = [p for p in observed_path if p not in contact_g]
    if missing:
        raise ValueError(
            f"observed_path residues not in contact graph (numbering?): {missing}"
        )

    values = _per_position_values(
        df, statistic, score_col, agg, activity_wt, abundance_wt
    )
    positions = _path_positions(observed_path, exclude_endpoints)
    if not positions:
        raise ValueError("No positions left to test after exclude_endpoints.")
    observed = _set_statistic(positions, values, agg if statistic == "stability" else "mean")

    obs_rsa = float("nan")
    if match_buriedness:
        if rsa_map is None:
            rsa_map = position_rsa_map(df)
        obs_rsa = _mean_rsa(positions, rsa_map)
        if not np.isfinite(obs_rsa):
            raise ValueError("Observed path is missing max_rsa for one or more residues")

    n_draw = int(n_null if not match_buriedness else n_null * oversample)
    candidates = sample_decoy_paths(
        contact_g,
        n_edges=n_edges,
        n_samples=n_draw,
        seed=seed,
        fixed_start=None if fixed_start is None else int(fixed_start),
        fixed_end=None if fixed_end is None else int(fixed_end),
        allowed_starts=allowed_starts,
        allowed_ends=allowed_ends,
    )
    obs_key = tuple(observed_path)
    unconstrained = (
        fixed_start is None
        and fixed_end is None
        and allowed_starts is None
        and allowed_ends is None
    )
    if unconstrained:
        banned = {obs_key, tuple(reversed(observed_path))}
        candidates = [p for p in candidates if tuple(p) not in banned]
    else:
        candidates = [p for p in candidates if tuple(p) != obs_key]

    n_rsa_rejected = 0
    if match_buriedness:
        kept = []
        for path in candidates:
            mean_rsa = _mean_rsa(_path_positions(path, exclude_endpoints), rsa_map)
            if np.isfinite(mean_rsa) and abs(mean_rsa - obs_rsa) <= float(rsa_tol):
                kept.append(path)
            else:
                n_rsa_rejected += 1
            if len(kept) >= n_null:
                break
        null_paths = kept
    else:
        null_paths = candidates[:n_null]

    null_stats = np.array(
        [
            _set_statistic(
                _path_positions(path, exclude_endpoints),
                values,
                agg if statistic == "stability" else "mean",
            )
            for path in null_paths
        ],
        dtype=float,
    )
    if match_buriedness and null_paths:
        null_rsa = np.array(
            [
                _mean_rsa(_path_positions(path, exclude_endpoints), rsa_map)
                for path in null_paths
            ],
            dtype=float,
        )
    else:
        null_rsa = np.full(len(null_paths), np.nan)

    finite = np.isfinite(null_stats)
    null_stats = null_stats[finite]
    null_paths = [path for path, ok in zip(null_paths, finite) if ok]
    null_rsa = null_rsa[finite]
    if not np.isfinite(observed) or len(null_stats) == 0:
        raise ValueError(
            f"Empty null/observed (n_null={len(null_stats)}, obs={observed}). "
            "Try a larger cutoff, more n_null, a wider rsa_tol, or a larger oversample."
        )

    n_used, n_unique, n_repeated = _null_uniqueness(null_paths)
    p_value, alt_label = _p_value(null_stats, observed, alternative)
    extra = [
        f"n_edges: {n_edges}",
        f"exclude_endpoints: {exclude_endpoints}",
        f"fixed_start: {fixed_start} | fixed_end: {fixed_end}",
        (
            f"match_buriedness: {match_buriedness} | rsa_tol: {rsa_tol}"
            if match_buriedness
            else f"match_buriedness: {match_buriedness}"
        ),
    ]
    if match_buriedness:
        extra.extend(
            [
                f"observed mean RSA: {obs_rsa:.4f}",
                (
                    f"RSA filter: kept {n_used}, rejected {n_rsa_rejected} "
                    f"of {len(candidates)} physical candidates"
                ),
                f"null mean RSA ± sd: {null_rsa.mean():.4f} ± {null_rsa.std():.4f}",
            ]
        )
    if statistic == "both_low":
        extra.append(f"activity_wt: {activity_wt} | abundance_wt: {abundance_wt}")
    if verbose:
        title = (
            "contact-graph decoys, mean RSA matched"
            if match_buriedness
            else "contact-graph decoys"
        )
        _print_result(
            title=title,
            observed_path=observed_path,
            positions=positions,
            statistic=statistic,
            score_col=score_col,
            agg=agg if statistic == "stability" else "mean",
            alternative=alternative,
            alt_label=alt_label,
            observed=observed,
            null_stats=null_stats,
            p_value=p_value,
            n_null=n_null,
            n_unique=n_unique,
            n_repeated=n_repeated,
            extra=extra,
        )

    return {
        "test": "contact_rsa" if match_buriedness else "contact",
        "observed_path": observed_path,
        "positions": positions,
        "observed": observed,
        "observed_mean_rsa": obs_rsa,
        "null_paths": null_paths,
        "null_stats": null_stats,
        "null_mean_rsa": null_rsa,
        "n_rsa_rejected": n_rsa_rejected,
        "p_value": p_value,
        "statistic": statistic,
        "score_col": score_col if statistic == "stability" else None,
        "agg": agg if statistic == "stability" else "mean",
        "alternative": alternative,
        "exclude_endpoints": exclude_endpoints,
        "match_buriedness": match_buriedness,
        "rsa_tol": rsa_tol if match_buriedness else None,
        "activity_wt": activity_wt if statistic == "both_low" else None,
        "abundance_wt": abundance_wt if statistic == "both_low" else None,
        "fixed_start": fixed_start,
        "fixed_end": fixed_end,
        "n_null": n_used,
        "n_unique_null": n_unique,
        "n_repeated_draws": n_repeated,
        "contact_graph": contact_g,
    }


def test_path_vs_rsa_matched_contact_decoys(*args, **kwargs):
    """Contact-graph decoys restricted to the observed path's mean RSA."""
    kwargs["match_buriedness"] = True
    return test_path_vs_contact_decoys(*args, **kwargs)


def test_path_vs_rsa_matched_sets(
    observed_path,
    df: pd.DataFrame,
    *,
    statistic: str = "stability",
    score_col: str = "stability",
    agg: str = "mean",
    alternative: str | None = None,
    activity_wt: float | None = None,
    abundance_wt: float | None = None,
    info_source: int | None = None,
    n_null: int = 1000,
    exclude_endpoints: bool = False,
    pool_positions: Sequence[int] | None = None,
    rsa_tol: float = 0.1,
    rsa_map: Mapping[int, float] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Phenotype of a path versus residue sets matched one-to-one on RSA.

    Each observed residue is paired with a distinct pool residue whose RSA
    is within ``rsa_tol``. This null is not a path and does not use the
    contact graph.
    """
    statistic, agg = _check_statistic(statistic, agg)
    alternative = _resolve_alternative(statistic, alternative)
    observed_path = _orient_path(observed_path, info_source=info_source)
    positions = _path_positions(observed_path, exclude_endpoints)
    if not positions:
        raise ValueError("No positions left to test after exclude_endpoints.")

    if rsa_map is None:
        rsa_map = position_rsa_map(df)
    missing_rsa = [p for p in positions if p not in rsa_map]
    if missing_rsa:
        raise ValueError(f"Missing max_rsa for observed positions: {missing_rsa}")

    if pool_positions is None:
        pool = sorted(rsa_map)
    else:
        pool = sorted({int(p) for p in pool_positions if int(p) in rsa_map})
    pool = [p for p in pool if p not in set(positions)]
    if exclude_endpoints:
        endpoints = {int(observed_path[0]), int(observed_path[-1])}
        if info_source is not None:
            endpoints.add(int(info_source))
        pool = [p for p in pool if p not in endpoints]
    if len(pool) < len(positions):
        raise ValueError(
            "Pool too small after excluding observed"
            f"{' + endpoints' if exclude_endpoints else ''}: "
            f"need {len(positions)}, have {len(pool)}"
        )

    values = _per_position_values(
        df, statistic, score_col, agg, activity_wt, abundance_wt
    )
    use_agg = agg if statistic == "stability" else "mean"
    observed = _set_statistic(positions, values, use_agg)

    rng = np.random.default_rng(seed)
    null_sets = []
    null_stats = []
    for _ in range(int(n_null)):
        draw = sample_rsa_matched_set(positions, pool, rsa_map, rng, rsa_tol=rsa_tol)
        if set(draw) & set(positions):
            raise RuntimeError(f"Null set overlapped observed positions: {draw}")
        null_sets.append(draw)
        null_stats.append(_set_statistic(draw, values, use_agg))

    null_stats_arr = np.asarray(null_stats, dtype=float)
    finite = np.isfinite(null_stats_arr)
    null_stats_arr = null_stats_arr[finite]
    null_sets = [draw for draw, ok in zip(null_sets, finite) if ok]
    if not np.isfinite(observed) or len(null_stats_arr) == 0:
        raise ValueError("Could not compute observed/null statistics.")

    n_used, n_unique, n_repeated = _null_uniqueness(null_sets)
    p_value, alt_label = _p_value(null_stats_arr, observed, alternative)
    obs_rsa = np.array([rsa_map[p] for p in positions], dtype=float)
    null_rsa_means = np.array(
        [np.mean([rsa_map[p] for p in draw]) for draw in null_sets], dtype=float
    )
    if verbose:
        extra = [
            f"exclude_endpoints: {exclude_endpoints}",
            f"RSA matched residue-wise, tol={rsa_tol}",
            f"observed mean RSA: {obs_rsa.mean():.4f} (sd {obs_rsa.std():.4f})",
            f"null mean RSA ± sd: {null_rsa_means.mean():.4f} ± {null_rsa_means.std():.4f}",
        ]
        if statistic == "both_low":
            extra.append(f"activity_wt: {activity_wt} | abundance_wt: {abundance_wt}")
        _print_result(
            title="RSA-matched random residue sets",
            observed_path=observed_path,
            positions=positions,
            statistic=statistic,
            score_col=score_col,
            agg=use_agg,
            alternative=alternative,
            alt_label=alt_label,
            observed=observed,
            null_stats=null_stats_arr,
            p_value=p_value,
            n_null=n_null,
            n_unique=n_unique,
            n_repeated=n_repeated,
            extra=extra,
        )

    return {
        "test": "rsa_sets",
        "observed_path": observed_path,
        "positions": positions,
        "observed": observed,
        "null_stats": null_stats_arr,
        "null_sets": null_sets,
        "p_value": p_value,
        "statistic": statistic,
        "score_col": score_col if statistic == "stability" else None,
        "agg": use_agg,
        "alternative": alternative,
        "rsa_tol": rsa_tol,
        "activity_wt": activity_wt if statistic == "both_low" else None,
        "abundance_wt": abundance_wt if statistic == "both_low" else None,
        "observed_rsa": obs_rsa,
        "null_mean_rsa": null_rsa_means,
        "exclude_endpoints": exclude_endpoints,
        "n_null": n_used,
        "n_unique_null": n_unique,
        "n_repeated_draws": n_repeated,
        "has_repeated_nulls": n_repeated > 0,
    }


def result_record(res: dict) -> dict:
    """One flat row for a test result (no paths or graphs)."""
    return {
        "test": res["test"],
        "statistic": res["statistic"],
        "alternative": res["alternative"],
        "agg": res["agg"],
        "observed": res["observed"],
        "null_mean": float(np.mean(res["null_stats"])),
        "null_sd": float(np.std(res["null_stats"])),
        "p_value": res["p_value"],
        "n_null": res["n_null"],
        "n_unique_null": res["n_unique_null"],
        "n_repeated_draws": res["n_repeated_draws"],
        "activity_wt": res.get("activity_wt"),
        "abundance_wt": res.get("abundance_wt"),
        "score_col": res.get("score_col"),
        "rsa_tol": res.get("rsa_tol"),
    }
