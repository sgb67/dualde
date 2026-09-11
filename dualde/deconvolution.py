"""Rank-difference KDE deconvolution of evolutionary vs stability scores."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde

_MUTANT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*-])$")


def parse_mutant(mutant: str) -> tuple[str, int, str]:
    """Parse 'Q144A' into (wt, position, subs)."""
    match = _MUTANT_RE.match(str(mutant).strip())
    if not match:
        raise ValueError(f"Could not parse mutant label {mutant!r}")
    wt, pos, subs = match.groups()
    return wt, int(pos), subs


def add_target_pos(df: pd.DataFrame, mutant_col: str = "mutant") -> pd.DataFrame:
    out = df.copy()
    parsed = out[mutant_col].map(parse_mutant)
    out["wt"] = parsed.map(lambda t: t[0])
    out["target_pos"] = parsed.map(lambda t: t[1])
    out["subs"] = parsed.map(lambda t: t[2])
    return out


def assign_outlier_probability_model(
    df: pd.DataFrame, col_1: str, col_2: str
) -> pd.DataFrame:
    """
    Compute KDE-based outlier probabilities from rank(col_2) - rank(col_1),
    and write them to ``{col_1}_{col_2}_model``.

    Pandas port of the 260721 notebook helper (originally Polars).
    ``col_1`` is typically an evolutionary score; ``col_2`` is a stability
    proxy such as ``neg_spurs`` or DMS ``stability``.
    """
    rank_col = f"{col_1}_{col_2}_rank_difference"
    model_col = f"{col_1}_{col_2}_model"
    if df.empty:
        return df.copy()

    work = df.dropna(subset=[col_1, col_2]).copy()
    n = len(work)
    if n < 2:
        raise ValueError("Need at least two non-null points to fit KDE.")

    work["rank_x"] = work[col_1].rank(method="average") / n
    work["rank_y"] = work[col_2].rank(method="average") / n
    work[rank_col] = work["rank_y"] - work["rank_x"]

    rank_diff = work[rank_col].to_numpy(dtype=float)
    q1 = np.percentile(rank_diff, 25)
    q3 = np.percentile(rank_diff, 75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    bulk_data = rank_diff[rank_diff <= upper_fence]
    if bulk_data.size < 2:
        raise ValueError("Bulk data too small after IQR filtering for KDE fit.")

    kde = gaussian_kde(bulk_data)
    lower_bound = float(bulk_data.min())
    grid = np.linspace(lower_bound, float(rank_diff.max()), 10_000)
    pdf = kde(grid).squeeze()
    cdf_grid = cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf_scores = np.clip(np.interp(rank_diff, grid, cdf_grid, left=0.0), 0.0, 1.0)
    work[model_col] = 1.0 - cdf_scores
    return work


def merge_evo_spurs_dms(
    evo: pd.DataFrame,
    spurs: pd.DataFrame,
    activity: pd.DataFrame,
    stability: pd.DataFrame,
    evo_mutant_col: str = "mutation",
    activity_score_col: str = "DMS_score",
    stability_score_col: str = "DMS_score",
) -> pd.DataFrame:
    """Join evolutionary scores, raw SPURS, and Yan 2020 activity/stability DMS."""
    evo_df = evo.copy()
    if evo_mutant_col != "mutant":
        evo_df = evo_df.rename(columns={evo_mutant_col: "mutant"})

    act = activity.rename(columns={activity_score_col: "activity"})[["mutant", "activity"]]
    stab = stability.rename(columns={stability_score_col: "stability"})[["mutant", "stability"]]
    spurs_keep = [c for c in ("mutant", "spurs", "uniprot_position", "pdb_position", "wt", "subs") if c in spurs.columns]

    merged = (
        evo_df.merge(act, on="mutant", how="inner")
        .merge(stab, on="mutant", how="inner")
        .merge(spurs[spurs_keep], on="mutant", how="left")
    )
    if "target_pos" not in merged.columns:
        merged = add_target_pos(merged)
    if "spurs" in merged.columns:
        merged["neg_spurs"] = -pd.to_numeric(merged["spurs"], errors="coerce")
    return merged


def deconvolve_evo_vs_stability(
    merged: pd.DataFrame,
    evo_cols: Iterable[str],
    stability_cols: Iterable[str] = ("neg_spurs", "stability"),
) -> pd.DataFrame:
    """Run rank-difference KDE for each (evo, stability) pair that is present."""
    out = merged
    for evo_col in evo_cols:
        if evo_col not in out.columns:
            continue
        for stab_col in stability_cols:
            if stab_col not in out.columns:
                continue
            pair = out.dropna(subset=[evo_col, stab_col])
            if len(pair) < 2:
                continue
            scored = assign_outlier_probability_model(pair, evo_col, stab_col)
            extra = [c for c in scored.columns if c not in out.columns]
            out = out.merge(
                scored[["mutant", *extra]],
                on="mutant",
                how="left",
            )
    return out
