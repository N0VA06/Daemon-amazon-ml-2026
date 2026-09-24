"""Decision layer: one-to-one then global / relative / expected-F0.5."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.evaluate import f05_entity, macro_f05


def greedy_one_to_one(pairs: pd.DataFrame, score_col: str = "p") -> pd.DataFrame:
    """Assign each S2/S3 to at most one S1, greedily by descending score."""
    if pairs.empty:
        return pairs.assign(assigned=pd.Series(dtype=bool))
    ordered = pairs.sort_values(score_col, ascending=False)
    used_c: set[str] = set()
    keep = []
    # We do NOT consume an S1 after one assignment — an S1 may have many matches.
    # We only consume gallery IDs (one-to-one on S2/S3).
    for idx, cid in zip(ordered.index, ordered["cand_id"]):
        if cid in used_c:
            keep.append(False)
            continue
        used_c.add(cid)
        keep.append(True)
    mask = pd.Series(False, index=pairs.index)
    mask.loc[ordered.index] = keep
    out = pairs.copy()
    out["assigned"] = mask
    return out


def decide_global(pairs: pd.DataFrame, tau: float, score_col: str = "p") -> dict[str, list[str]]:
    pred: dict[str, list[str]] = defaultdict(list)
    for row in pairs.itertuples(index=False):
        if getattr(row, score_col) >= tau:
            pred[row.s1_id].append(row.cand_id)
    return dict(pred)


def decide_relative(
    pairs: pd.DataFrame,
    tau: float,
    alpha: float,
    score_col: str = "p",
) -> dict[str, list[str]]:
    pred: dict[str, list[str]] = {}
    for s1, g in pairs.groupby("s1_id"):
        if g.empty:
            pred[s1] = []
            continue
        mx = float(g[score_col].max())
        keep = g[(g[score_col] >= tau) & (g[score_col] >= alpha * mx)]
        pred[s1] = keep.sort_values(score_col, ascending=False)["cand_id"].tolist()
    return pred


def _expected_f05_prefix(scores: np.ndarray) -> int:
    """Return k in {0..n} maximising the expected F0.5 approximation."""
    n = len(scores)
    if n == 0:
        return 0
    p = np.clip(scores.astype(np.float64), 1e-9, 1 - 1e-9)
    p_none = float(np.prod(1.0 - p))
    e_truth = float(p.sum())
    best_k = 0
    best_e = p_none
    csum = np.cumsum(p)
    for k in range(1, n + 1):
        e_tp = float(csum[k - 1])
        e_f = 1.25 * e_tp / (0.25 * e_truth + k)
        if e_f > best_e:
            best_e = e_f
            best_k = k
    return best_k


def decide_expected_f05(pairs: pd.DataFrame, score_col: str = "p") -> dict[str, list[str]]:
    pred: dict[str, list[str]] = {}
    for s1, g in pairs.groupby("s1_id"):
        g = g.sort_values(score_col, ascending=False)
        k = _expected_f05_prefix(g[score_col].to_numpy())
        pred[s1] = g["cand_id"].tolist()[:k]
    return pred


def apply_decision(
    pairs: pd.DataFrame,
    s1_ids: list[str],
    rule: str,
    tau: float,
    alpha: float,
    one_to_one: bool,
    score_col: str = "p",
) -> dict[str, list[str]]:
    work = pairs
    if one_to_one:
        work = greedy_one_to_one(work, score_col)
        work = work[work["assigned"]]
    if rule == "global":
        pred = decide_global(work, tau, score_col)
    elif rule == "relative":
        pred = decide_relative(work, tau, alpha, score_col)
    else:
        pred = decide_expected_f05(work, score_col)
    return {s1: pred.get(s1, []) for s1 in s1_ids}


def sweep_global_threshold(
    pairs: pd.DataFrame,
    truth_map: dict[str, set[str]],
    s1_ids: list[str],
    score_col: str = "p",
    grid: np.ndarray | None = None,
    one_to_one: bool = True,
) -> tuple[float, dict]:
    grid = grid if grid is not None else np.linspace(0.15, 0.90, 31)
    best_t, best = 0.5, {"macro_f05": -1.0}
    for t in grid:
        pred = apply_decision(pairs, s1_ids, "global", float(t), 0.0, one_to_one, score_col)
        m = macro_f05(pred, truth_map, s1_ids)
        if m["macro_f05"] > best["macro_f05"]:
            best = m
            best_t = float(t)
    best["threshold"] = best_t
    return best_t, best


def compare_rules(
    pairs: pd.DataFrame,
    truth_map: dict[str, set[str]],
    s1_ids: list[str],
    tau: float,
    alpha: float,
    one_to_one: bool,
    score_col: str = "p",
) -> dict:
    out = {}
    for rule in ("global", "relative", "expected_f05"):
        pred = apply_decision(pairs, s1_ids, rule, tau, alpha, one_to_one, score_col)
        out[rule] = macro_f05(pred, truth_map, s1_ids)
        out[rule]["n_pred_links"] = sum(len(v) for v in pred.values())
        out[rule]["n_pred_singletons"] = sum(1 for v in pred.values() if not v)
    return out
