"""Logistic stacker on [logit(p_gbm), logit(p_ce), rank features] + isotonic calibration."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.logging_utils import LOG, write_json


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log(1 - p)


def stacker_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = []
    mats = []
    if "p_gbm" in df.columns:
        mats.append(_logit(df["p_gbm"].fillna(0.0).to_numpy())[:, None])
        cols.append("logit_gbm")
    if "p_ce" in df.columns and df["p_ce"].notna().any():
        ce = df["p_ce"].to_numpy(dtype=np.float64)
        # missing CE (not in top-N) → 0.5 logit 0, plus a missing flag
        missing = np.isnan(ce)
        ce = np.where(missing, 0.5, ce)
        mats.append(_logit(ce)[:, None])
        mats.append(missing.astype(np.float64)[:, None])
        cols.extend(["logit_ce", "ce_missing"])
    for extra in (
        "p_gbm_rank",
        "cosine_ft_combined_rank",
        "cosine_zs_combined_rank",
        "score_faiss",
        "n_candidates",
        "mutual_best",
        "cand_margin_second_s1",
        "house_num_mismatch",
        "postcode_mismatch",
    ):
        if extra in df.columns:
            mats.append(df[extra].fillna(0.0).to_numpy(dtype=np.float64)[:, None])
            cols.append(extra)
    if not mats:
        raise ValueError("stacker has no input columns")
    return np.hstack(mats), cols


def train_stacker(df: pd.DataFrame, cfg) -> dict:
    labelled = df.dropna(subset=["label"]).copy()
    X, cols = stacker_matrix(labelled)
    y = labelled["label"].astype(int).to_numpy()
    lr = LogisticRegression(
        C=float(cfg.stacker.C),
        max_iter=int(cfg.stacker.max_iter),
        solver="lbfgs",
    )
    lr.fit(X, y)
    raw = lr.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y)
    cal = iso.predict(raw)
    brier_raw = float(brier_score_loss(y, raw))
    brier_cal = float(brier_score_loss(y, cal))
    return {
        "lr": lr,
        "iso": iso,
        "cols": cols,
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
    }


def apply_stacker(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X, _ = stacker_matrix(df)
    raw = bundle["lr"].predict_proba(X)[:, 1]
    return bundle["iso"].predict(raw)


def save_stacker(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: v for k, v in bundle.items() if k != "plot"}, path)


def load_stacker(path: Path) -> dict:
    return joblib.load(path)


def reliability_plot(y: np.ndarray, p: np.ndarray, path: Path, n_bins: int = 15) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(xs, ys, "o-", color="#1f4e79")
    ax.set_xlabel("predicted p")
    ax.set_ylabel("empirical positive rate")
    ax.set_title("Reliability")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"bin_pred": xs, "bin_true": ys, "bin_n": ns}


def run_stack_and_calibrate(df: pd.DataFrame, cfg, reports_dir: Path) -> tuple[pd.DataFrame, dict]:
    bundle = train_stacker(df, cfg)
    out = df.copy()
    out["p"] = apply_stacker(bundle, out)
    labelled = out.dropna(subset=["label"])
    rel = reliability_plot(
        labelled["label"].to_numpy(),
        labelled["p"].to_numpy(),
        reports_dir / "reliability.png",
    )
    metrics = {
        "brier_raw": bundle["brier_raw"],
        "brier_cal": bundle["brier_cal"],
        "reliability": rel,
        "n": int(len(labelled)),
    }
    write_json(reports_dir / "stacker.json", metrics)
    LOG.info("stacker  n=%s  brier_raw=%.4f  brier_cal=%.4f", metrics["n"], metrics["brier_raw"], metrics["brier_cal"])
    bundle["metrics"] = metrics
    return out, bundle
