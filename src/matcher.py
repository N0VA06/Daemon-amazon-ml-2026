"""LightGBM binary matcher with monotone constraints and OOF training."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import feature_columns


def _monotone_vector(cols: list[str], mapping: dict) -> list[int]:
    return [int(mapping.get(c, 0)) for c in cols]


def subsample_negatives(
    df: pd.DataFrame,
    ratio: float,
    seed: int,
    hard_mask: pd.Series | None = None,
) -> pd.DataFrame:
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    if hard_mask is not None:
        hard = df[(df["label"] == 0) & hard_mask]
        easy = df[(df["label"] == 0) & ~hard_mask]
    else:
        hard = neg.iloc[0:0]
        easy = neg
    n_keep = int(len(pos) * ratio)
    n_easy = max(n_keep - len(hard), 0)
    if n_easy < len(easy):
        easy = easy.sample(n=n_easy, random_state=seed)
    return pd.concat([pos, hard, easy], ignore_index=True)


def train_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    cfg,
    feature_cols: list[str] | None = None,
) -> tuple[lgb.Booster, list[str]]:
    feature_cols = feature_cols or feature_columns(train_df)
    mono = _monotone_vector(feature_cols, dict(vars(cfg.matcher.monotone)))
    params = {
        "objective": cfg.matcher.objective,
        "metric": cfg.matcher.metric,
        "learning_rate": float(cfg.matcher.learning_rate),
        "num_leaves": int(cfg.matcher.num_leaves),
        "min_child_samples": int(cfg.matcher.min_child_samples),
        "subsample": float(cfg.matcher.subsample),
        "colsample_bytree": float(cfg.matcher.colsample_bytree),
        "verbosity": -1,
        "seed": int(cfg.seed),
        "n_jobs": int(cfg.matcher.n_jobs),
        "monotone_constraints": mono,
    }
    dtrain = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["label"].astype(int),
        feature_name=feature_cols,
        free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(100)]
    if valid_df is not None and len(valid_df):
        dvalid = lgb.Dataset(
            valid_df[feature_cols],
            label=valid_df["label"].astype(int),
            feature_name=feature_cols,
            free_raw_data=False,
        )
        valid_sets.append(dvalid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(int(cfg.matcher.early_stopping_rounds)))
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=int(cfg.matcher.n_estimators),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster, feature_cols


def predict_lgbm(booster: lgb.Booster, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return booster.predict(df[feature_cols])


def save_matcher(path: Path, booster: lgb.Booster, feature_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"booster": booster, "feature_cols": feature_cols}, path)


def load_matcher(path: Path) -> tuple[lgb.Booster, list[str]]:
    obj = joblib.load(path)
    return obj["booster"], obj["feature_cols"]


def oof_train_matcher(
    pairs: pd.DataFrame,
    fold_of_s1: dict[str, int],
    n_folds: int,
    cfg,
) -> tuple[pd.DataFrame, lgb.Booster, list[str]]:
    """Train one model per fold; write OOF p_gbm. Return last full-data model too."""
    df = pairs.copy()
    df["fold"] = df["s1_id"].map(fold_of_s1)
    df["p_gbm"] = np.nan
    feature_cols = feature_columns(df)
    last_model = None
    for k in range(n_folds):
        tr = df[df["fold"] != k]
        va = df[df["fold"] == k]
        if va.empty:
            continue
        hard = None
        if "house_num_mismatch" in tr.columns:
            hard = (tr["house_num_mismatch"] == 0) & (tr["name_rf_token_set"] > 0.85)
        tr_s = subsample_negatives(
            tr, float(cfg.matcher.negative_sample_ratio), int(cfg.seed) + k, hard
        )
        model, feature_cols = train_lgbm(tr_s, va, cfg, feature_cols)
        df.loc[va.index, "p_gbm"] = predict_lgbm(model, va, feature_cols)
        last_model = model
    # final model on all labelled pairs (subsampled)
    hard = None
    if "house_num_mismatch" in df.columns:
        hard = (df["house_num_mismatch"] == 0) & (df["name_rf_token_set"] > 0.85)
    all_s = subsample_negatives(
        df.dropna(subset=["label"]), float(cfg.matcher.negative_sample_ratio), int(cfg.seed), hard
    )
    final, feature_cols = train_lgbm(all_s, None, cfg, feature_cols)
    # fill any missing OOF with the final model
    miss = df["p_gbm"].isna()
    if miss.any():
        df.loc[miss, "p_gbm"] = predict_lgbm(final, df.loc[miss], feature_cols)
    return df, final, feature_cols
