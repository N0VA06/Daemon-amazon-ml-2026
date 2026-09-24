"""S1-entity splits: hold-out, OOF folds, leave-one-country-out, val gallery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _stratum(s1: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    merged = s1.merge(
        gt[["source1_entity_id", "is_singleton"]],
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
    )
    merged["is_singleton"] = merged["is_singleton"].fillna(True)
    country = merged["country"].astype(str).fillna("UNK")
    flag = merged["is_singleton"].map({True: "singleton", False: "linked"})
    return country + "|" + flag


def make_holdout(
    s1: pd.DataFrame,
    gt: pd.DataFrame,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    strata = _stratum(s1, gt)
    # rare strata collapse
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, "RARE")
    idx = np.arange(len(s1))
    try:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_frac, random_state=seed, stratify=strata
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_frac, random_state=seed
        )
    return np.asarray(train_idx), np.asarray(val_idx)


def make_oof_folds(
    s1: pd.DataFrame,
    gt: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = _stratum(s1, gt)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= n_folds, "RARE")
    idx = np.arange(len(s1))
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(idx, strata))
    except ValueError:
        # unstratified fallback
        rng = np.random.RandomState(seed)
        perm = rng.permutation(idx)
        folds = np.array_split(perm, n_folds)
        splits = []
        for i in range(n_folds):
            val = folds[i]
            train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            splits.append((train, val))
    return [(np.asarray(tr), np.asarray(va)) for tr, va in splits]


def linked_ids(gt: pd.DataFrame, s1_ids: Iterable) -> set[str]:
    keep = set()
    subset = gt[gt["source1_entity_id"].isin(set(s1_ids))]
    for lst in subset["matched_list"]:
        keep.update(lst)
    return keep


def validation_gallery(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    val_s1_ids: list[str],
    train_s1_ids: list[str],
    unlinked_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All S2/S3 linked to val S1s + a proportional share of unlinked records."""
    val_linked = linked_ids(gt, val_s1_ids)
    train_linked = linked_ids(gt, train_s1_ids)
    all_linked = val_linked | train_linked

    def _pick(df: pd.DataFrame) -> pd.DataFrame:
        linked = df[df["entity_id"].isin(val_linked)]
        unlinked = df[~df["entity_id"].isin(all_linked)]
        n_take = int(round(len(unlinked) * unlinked_frac))
        if n_take > 0 and len(unlinked) > 0:
            unlinked = unlinked.sample(n=min(n_take, len(unlinked)), random_state=seed)
        else:
            unlinked = unlinked.iloc[0:0]
        return pd.concat([linked, unlinked], ignore_index=True)

    return _pick(s2), _pick(s3)


def loco_masks(s1: pd.DataFrame) -> dict[str, np.ndarray]:
    """country -> boolean mask of S1 rows with that country (open-set)."""
    out = {}
    for c in sorted(s1["country"].astype(str).unique()):
        out[c] = (s1["country"].astype(str) == c).to_numpy()
    return out


def save_split(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, dict):
            serialisable[k] = {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
            }
        else:
            serialisable[k] = v
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
