"""CLI: python -m src.run --stage <stage> --config configs/default.yaml"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.blocking import run_blocking
from src.config import as_dict, load_config, set_seeds
from src.decision import apply_decision, compare_rules, sweep_global_threshold
from src.eda import run_eda
from src.embeddings import load_id_index, load_sentence_transformer
from src.evaluate import _self_test, macro_f05, per_country_f05
from src.features import (
    add_embedding_features,
    add_rank_context_features,
    build_pair_features,
    label_pairs,
    load_parquet,
    save_parquet,
)
from src.hardware import apply_auto_batch
from src.io_utils import (
    gt_to_map,
    read_id_list_tsv,
    write_candidate_pairs,
    write_matching_results,
)
from src.logging_utils import LOG, banner, log_experiment, setup_logging, write_json
from src.matcher import load_matcher, oof_train_matcher, predict_lgbm, save_matcher
from src.pipeline import (
    build_all_normalized,
    build_splits,
    cache_dir,
    concat_gallery_emb,
    encode_split_views,
    gallery_of,
    load_idf_if_any,
    load_or_build_normalized,
    load_test,
    load_train,
    load_view,
    norm_path,
)
from src.stacker import apply_stacker, load_stacker, run_stack_and_calibrate, save_stacker


STAGES = (
    "eda",
    "split",
    "normalize",
    "block",
    "train_biencoder",
    "features",
    "train_matcher",
    "train_crossencoder",
    "stack",
    "decide",
    "evaluate",
    "predict",
    "all",
)


def _log(cfg, stage: str, message: str, metrics=None):
    LOG.info("%s", message)
    log_experiment(cfg.paths.experiments_log, stage, message, metrics)


def _truth(gt) -> dict:
    return gt_to_map(gt)


def stage_eda(cfg):
    banner("eda")
    _self_test()
    LOG.info("F0.5 self-test passed  (pred {A,B,C} vs truth {A,C} → 0.714)")
    metrics = run_eda(cfg)
    LOG.info(
        "EDA  train_S1=%s  singleton_rate=%s  one_to_one=%s  same_country=%s  France_test_S1=%s",
        metrics.get("n_s1_train"),
        f"{metrics.get('singleton_rate', 0):.4f}" if metrics.get("singleton_rate") is not None else "?",
        metrics.get("one_to_one"),
        metrics.get("always_same_country"),
        metrics.get("n_france_test_s1"),
    )
    _log(cfg, "eda", "EDA written to reports/eda.md.", metrics)
    return metrics


def stage_split(cfg):
    banner("split")
    s1, s2, s3, gt = load_train(cfg)
    payload = build_splits(cfg, s1, gt)
    # materialise val gallery ids
    from src.split import validation_gallery

    val_s1 = set(payload["holdout_val"])
    train_s1 = set(payload["holdout_train"])
    g2, g3 = validation_gallery(
        s2, s3, gt, list(val_s1), list(train_s1),
        float(cfg.split.gallery_unlinked_frac), int(cfg.seed),
    )
    g2.to_parquet(cache_dir(cfg) / "val_gallery_s2.parquet", index=False)
    g3.to_parquet(cache_dir(cfg) / "val_gallery_s3.parquet", index=False)
    metrics = {
        "n_train_s1": len(payload["holdout_train"]),
        "n_val_s1": len(payload["holdout_val"]),
        "n_folds": len(payload["folds"]),
        "n_val_gallery": int(len(g2) + len(g3)),
        "loco_countries": list(payload["loco"].keys()),
    }
    write_json(Path(cfg.paths.reports_dir) / "split.json", metrics)
    LOG.info("hold-out train_S1=%s  val_S1=%s  val_gallery=%s",
             f"{metrics['n_train_s1']:,}", f"{metrics['n_val_s1']:,}", f"{metrics['n_val_gallery']:,}")
    _log(cfg, "split", "S1-entity hold-out + OOF folds.", metrics)
    return metrics


def stage_normalize(cfg):
    banner("normalize")
    frames = build_all_normalized(cfg)
    metrics = {k: int(len(v)) for k, v in frames.items()}
    metrics["idf"] = bool((cache_dir(cfg) / "idf.json").exists())
    _log(cfg, "normalize", "Normalised tables cached under cache/normalized/.", metrics)
    return metrics


def _holdout_frames(cfg):
    s1, s2, s3, gt = load_train(cfg)
    splits = build_splits(cfg, s1, gt)
    s1n = pd.read_parquet(norm_path(cfg, "train", "s1"))
    s2n = pd.read_parquet(norm_path(cfg, "train", "s2"))
    s3n = pd.read_parquet(norm_path(cfg, "train", "s3"))
    val_ids = set(splits["holdout_val"])
    train_ids = set(splits["holdout_train"])
    val_s1 = s1n[s1n["entity_id"].isin(val_ids)].reset_index(drop=True)
    train_s1 = s1n[s1n["entity_id"].isin(train_ids)].reset_index(drop=True)
    g2_p = cache_dir(cfg) / "val_gallery_s2.parquet"
    g3_p = cache_dir(cfg) / "val_gallery_s3.parquet"
    if g2_p.exists():
        g2 = pd.read_parquet(g2_p)
        g3 = pd.read_parquet(g3_p)
        # attach normalised columns
        s2n = s2n[s2n["entity_id"].isin(set(g2["entity_id"]))].reset_index(drop=True)
        s3n = s3n[s3n["entity_id"].isin(set(g3["entity_id"]))].reset_index(drop=True)
    val_gal = gallery_of(s2n, s3n)
    return train_s1, val_s1, val_gal, gt, splits, s1n, pd.read_parquet(norm_path(cfg, "train", "s2")), pd.read_parquet(norm_path(cfg, "train", "s3"))


def _ensure_embeddings(cfg, df, split, source, tag, model=None):
    return encode_split_views(cfg, df, split, source, tag, model=model)


def stage_block(cfg, tag: str | None = None):
    """Blocking on the hold-out validation split (test blocking is in predict)."""
    banner("block")
    train_s1, val_s1, val_gal, gt, splits, *_ = _holdout_frames(cfg)
    tmap = _truth(gt)

    # zero-shot embeddings
    model = None
    tag = tag or "zs"
    try:
        model = load_sentence_transformer(cfg)
        _ensure_embeddings(cfg, val_s1, "val", "s1", tag, model)
        # encode val gallery pieces: we stored filtered s2/s3 as val gallery
        # split the gallery back
        g2 = val_gal[val_gal["entity_id"].astype(str).str.startswith("S2-")]
        g3 = val_gal[val_gal["entity_id"].astype(str).str.startswith("S3-")]
        _ensure_embeddings(cfg, g2, "val", "s2", tag, model)
        _ensure_embeddings(cfg, g3, "val", "s3", tag, model)
        s1_emb = load_view(cfg, "val", "s1", "combined", tag)
        gal_emb, _ = concat_gallery_emb(cfg, "val", "combined", tag)
        # align gallery rows to embedding order (s2 then s3)
        gal_ids = []
        cache = Path(cfg.paths.cache_dir)
        from src.embeddings import ids_path

        for src in ("s2", "s3"):
            p = ids_path(cache, "val", src)
            if p.exists():
                gal_ids.extend(np.load(p, allow_pickle=True).tolist())
        gal_map = val_gal.set_index("entity_id")
        val_gal_aligned = gal_map.loc[[i for i in gal_ids if i in gal_map.index]].reset_index()
        # drop embeddings for missing
        keep = [i for i, eid in enumerate(gal_ids) if eid in gal_map.index]
        gal_emb = gal_emb[keep]
    except Exception as exc:
        LOG.warning("embedding path failed (%s); TF-IDF + keys only", exc)
        s1_emb = gal_emb = None
        val_gal_aligned = val_gal

    idf = load_idf_if_any(cfg)
    cand_map, pairs, metrics = run_blocking(
        cfg, val_s1, val_gal_aligned, s1_emb, gal_emb, idf,
        out_tsv=Path(cfg.paths.output_dir) / "val_candidate_pairs.tsv",
        tag=f"val_{tag}",
        truth_map=tmap,
    )
    save_parquet(pairs, cache_dir(cfg) / f"val_pairs_{tag}.parquet")
    _log(cfg, "block", f"Validation blocking ({tag}).", metrics)
    return metrics


def stage_train_biencoder(cfg):
    banner("train_biencoder")
    from src.biencoder import train_biencoder

    train_s1, val_s1, val_gal, gt, splits, s1n, s2n, s3n = _holdout_frames(cfg)
    tmap = _truth(gt)
    train_gt = gt[gt["source1_entity_id"].isin(set(train_s1["entity_id"]))]
    # zero-shot embs for hard-neg mining if present
    zs_s1 = zs_gal = None
    try:
        model = load_sentence_transformer(cfg)
        train_gal = gallery_of(s2n, s3n)
        if getattr(cfg.debug, "max_gallery", None):
            train_gal = train_gal.sample(
                n=min(int(cfg.debug.max_gallery), len(train_gal)), random_state=int(cfg.seed)
            )
        _ensure_embeddings(cfg, train_s1, "trainhold", "s1", "zs", model)
        g2t = train_gal[train_gal["entity_id"].astype(str).str.startswith("S2-")]
        g3t = train_gal[train_gal["entity_id"].astype(str).str.startswith("S3-")]
        _ensure_embeddings(cfg, g2t, "trainhold", "s2", "zs", model)
        _ensure_embeddings(cfg, g3t, "trainhold", "s3", "zs", model)
        zs_s1 = load_view(cfg, "trainhold", "s1", "combined", "zs")
        zs_gal, _ = concat_gallery_emb(cfg, "trainhold", "combined", "zs")
        train_gal = pd.concat([g2t, g3t], ignore_index=True)
    except Exception as exc:
        LOG.warning("zero-shot hard-neg mine skipped: %s", exc)
        train_gal = gallery_of(s2n, s3n)

    out = Path(cfg.paths.models_dir) / "biencoder"
    metrics = train_biencoder(
        cfg, train_s1, train_gal, train_gt, tmap, zs_s1, zs_gal, out,
        val_s1=val_s1, val_gallery=val_gal,
    )
    _log(cfg, "train_biencoder", "Phase A InfoNCE + Phase B remine/CoSENT.", metrics)
    return metrics


def _add_cosines(cfg, pairs, s1, gallery, split_s1, split_gal, tag, prefix):
    try:
        s1_emb = {"combined": load_view(cfg, split_s1, "s1", "combined", tag)}
        for v in ("name", "address"):
            p = Path(cfg.paths.cache_dir) / "embeddings" / f"{split_s1}_s1_{v}_{tag}.npy"
            if p.exists():
                s1_emb[v] = load_view(cfg, split_s1, "s1", v, tag)
        gal_combined, gal_index = concat_gallery_emb(cfg, split_gal, "combined", tag)
        gal_emb = {"combined": gal_combined}
        for v in ("name", "address"):
            try:
                mat, idx = concat_gallery_emb(cfg, split_gal, v, tag)
                gal_emb[v] = mat
            except FileNotFoundError:
                pass
        s1_index = load_id_index(Path(cfg.paths.cache_dir), split_s1, "s1")
        return add_embedding_features(pairs, s1_index, gal_index, s1_emb, gal_emb, prefix)
    except Exception as exc:
        LOG.warning("cosine %s/%s skipped: %s", prefix, tag, exc)
        return pairs


def stage_features(cfg):
    banner("features")
    train_s1, val_s1, val_gal, gt, splits, *_ = _holdout_frames(cfg)
    tmap = _truth(gt)
    pair_path = cache_dir(cfg) / "val_pairs_zs.parquet"
    if not pair_path.exists():
        stage_block(cfg, tag="zs")
    pairs = load_parquet(pair_path)
    idf = load_idf_if_any(cfg)
    feats = build_pair_features(pairs, val_s1, val_gal, idf, int(cfg.features.chunk_pairs))
    feats = label_pairs(feats, tmap)
    if cfg.features.compute_zero_shot_cosine:
        feats = _add_cosines(cfg, feats, val_s1, val_gal, "val", "val", "zs", "zs")
    ft_dir = Path(cfg.paths.models_dir) / "biencoder" / "merged"
    if cfg.features.compute_finetuned_cosine and ft_dir.exists():
        model = load_sentence_transformer(cfg, adapter_path=ft_dir)
        g2 = val_gal[val_gal["entity_id"].astype(str).str.startswith("S2-")]
        g3 = val_gal[val_gal["entity_id"].astype(str).str.startswith("S3-")]
        _ensure_embeddings(cfg, val_s1, "val", "s1", "ft", model)
        _ensure_embeddings(cfg, g2, "val", "s2", "ft", model)
        _ensure_embeddings(cfg, g3, "val", "s3", "ft", model)
        feats = _add_cosines(cfg, feats, val_s1, val_gal, "val", "val", "ft", "ft")
    score_cols = [c for c in feats.columns if c.startswith("score_") or c.startswith("cosine_")]
    feats = add_rank_context_features(feats, score_cols)
    save_parquet(feats, cache_dir(cfg) / "val_features.parquet")
    metrics = {"n_pairs": int(len(feats)), "n_pos": int(feats["label"].sum()), "n_feat_ready": True}
    _log(cfg, "features", "Pair features (string + numbers + cosine + rank).", metrics)
    return metrics


def stage_train_matcher(cfg):
    banner("train_matcher")
    feats = load_parquet(cache_dir(cfg) / "val_features.parquet")
    # 3 folds over S1 on the hold-out itself for OOF p_gbm
    s1_ids = feats["s1_id"].drop_duplicates().tolist()
    rng = np.random.RandomState(int(cfg.seed))
    rng.shuffle(s1_ids)
    folds = np.array_split(s1_ids, int(cfg.split.n_folds))
    fold_of = {}
    for i, chunk in enumerate(folds):
        for s in chunk:
            fold_of[s] = i
    out, model, cols = oof_train_matcher(feats, fold_of, int(cfg.split.n_folds), cfg)
    save_parquet(out, cache_dir(cfg) / "val_features.parquet")
    save_matcher(Path(cfg.paths.models_dir) / "lgbm.joblib", model, cols)
    metrics = {
        "n_pairs": int(len(out)),
        "mean_p_gbm_pos": float(out.loc[out["label"] == 1, "p_gbm"].mean()),
        "mean_p_gbm_neg": float(out.loc[out["label"] == 0, "p_gbm"].mean()),
        "n_features": len(cols),
    }
    LOG.info("LightGBM  pos_mean_p=%.4f  neg_mean_p=%.4f  n_features=%s",
             metrics["mean_p_gbm_pos"], metrics["mean_p_gbm_neg"], metrics["n_features"])
    write_json(Path(cfg.paths.reports_dir) / "matcher.json", metrics)
    _log(cfg, "train_matcher", "LightGBM OOF + final model.", metrics)
    return metrics


def stage_train_crossencoder(cfg):
    banner("train_crossencoder")
    from src.crossencoder import apply_ce_topn, train_crossencoder

    feats = load_parquet(cache_dir(cfg) / "val_features.parquet")
    _, val_s1, val_gal, *_ = _holdout_frames(cfg)
    bundle = train_crossencoder(
        cfg, feats, val_s1, val_gal, Path(cfg.paths.models_dir) / "crossencoder"
    )
    scored = apply_ce_topn(bundle["model"], feats, val_s1, val_gal, cfg, "p_gbm")
    save_parquet(scored, cache_dir(cfg) / "val_features.parquet")
    _log(cfg, "train_crossencoder", "BCE cross-encoder on top-N by p_gbm.", bundle["metrics"])
    return bundle["metrics"]


def stage_stack(cfg):
    banner("stack")
    feats = load_parquet(cache_dir(cfg) / "val_features.parquet")
    if "p_gbm" not in feats.columns:
        raise RuntimeError("run train_matcher before stack")
    if "p_ce" not in feats.columns:
        feats["p_ce"] = np.nan
    scored, bundle = run_stack_and_calibrate(feats, cfg, Path(cfg.paths.reports_dir))
    save_parquet(scored, cache_dir(cfg) / "val_features.parquet")
    save_stacker(Path(cfg.paths.models_dir) / "stacker.joblib", bundle)
    _log(cfg, "stack", "Logistic stacker + isotonic calibration.", bundle["metrics"])
    return bundle["metrics"]


def stage_decide(cfg):
    banner("decide")
    feats = load_parquet(cache_dir(cfg) / "val_features.parquet")
    _, val_s1, _, gt, *_ = _holdout_frames(cfg)
    tmap = _truth(gt)
    s1_ids = val_s1["entity_id"].astype(str).tolist()
    score_col = "p" if "p" in feats.columns else "p_gbm"
    tau, sweep = sweep_global_threshold(
        feats, tmap, s1_ids, score_col=score_col, one_to_one=bool(cfg.decision.one_to_one)
    )
    cfg.decision.global_threshold = tau
    comparison = compare_rules(
        feats, tmap, s1_ids, tau, float(cfg.decision.relative_alpha),
        bool(cfg.decision.one_to_one), score_col,
    )
    pred = apply_decision(
        feats, s1_ids, str(cfg.decision.rule), tau,
        float(cfg.decision.relative_alpha), bool(cfg.decision.one_to_one), score_col,
    )
    write_matching_results(s1_ids, pred, Path(cfg.paths.output_dir) / "val_matching_results.tsv")
    # candidate file already written by block; rewrite from features to be exact
    cand = feats.groupby("s1_id")["cand_id"].apply(list).to_dict()
    write_candidate_pairs(s1_ids, cand, Path(cfg.paths.output_dir) / "val_candidate_pairs.tsv")
    metrics = {
        "chosen_rule": cfg.decision.rule,
        "swept_tau": tau,
        "sweep": sweep,
        "comparison": comparison,
        "holdout": macro_f05(pred, tmap, s1_ids),
    }
    write_json(Path(cfg.paths.reports_dir) / "decision.json", metrics)
    LOG.info("decision rule=%s  tau=%.3f  holdout_F0.5=%.4f",
             cfg.decision.rule, tau, metrics["holdout"]["macro_f05"])
    _log(cfg, "decide", "Hold-out decision-rule comparison.", metrics)
    return metrics


def stage_evaluate(cfg):
    banner("evaluate")
    _self_test()
    feats = load_parquet(cache_dir(cfg) / "val_features.parquet")
    _, val_s1, _, gt, splits, s1n, *_ = _holdout_frames(cfg)
    tmap = _truth(gt)
    s1_ids = val_s1["entity_id"].astype(str).tolist()
    score_col = "p" if "p" in feats.columns else "p_gbm"
    pred = apply_decision(
        feats, s1_ids, str(cfg.decision.rule),
        float(cfg.decision.global_threshold), float(cfg.decision.relative_alpha),
        bool(cfg.decision.one_to_one), score_col,
    )
    country = dict(zip(val_s1["entity_id"].astype(str), val_s1["country"].astype(str)))
    metrics = {
        "holdout": macro_f05(pred, tmap, s1_ids),
        "by_country": per_country_f05(pred, tmap, country),
    }
    # leave-one-country-out proxy on the same scored table
    loco = {}
    for c, ids in splits["loco"].items():
        ids = [i for i in ids if i in set(s1_ids)]
        if not ids:
            continue
        loco[c] = macro_f05(pred, tmap, ids)
    metrics["loco_on_holdout"] = loco
    write_json(Path(cfg.paths.reports_dir) / "evaluate.json", metrics)
    LOG.info("holdout macro F0.5 = %.4f  over %s S1 entities",
             metrics["holdout"]["macro_f05"], metrics["holdout"]["n_entities"])
    _log(cfg, "evaluate", "Hold-out + per-country F0.5.", metrics)
    return metrics


def _predict_split(cfg, s1, gallery, split_name: str, out_match: Path, out_cand: Path):
    """Encode → block → features → score → decide. Writes PDF output TSVs."""
    banner(f"predict:{split_name}")
    LOG.info("predict  S1=%s  gallery=%s  → %s", f"{len(s1):,}", f"{len(gallery):,}", out_match)
    tag = "ft" if (Path(cfg.paths.models_dir) / "biencoder" / "merged").exists() else "zs"
    LOG.info("embedding tag=%s", tag)
    model = None
    try:
        adapter = Path(cfg.paths.models_dir) / "biencoder" / "merged"
        model = load_sentence_transformer(cfg, adapter_path=adapter if adapter.exists() else None)
        g2 = gallery[gallery["entity_id"].astype(str).str.startswith("S2-")]
        g3 = gallery[gallery["entity_id"].astype(str).str.startswith("S3-")]
        _ensure_embeddings(cfg, s1, split_name, "s1", tag, model)
        _ensure_embeddings(cfg, g2, split_name, "s2", tag, model)
        _ensure_embeddings(cfg, g3, split_name, "s3", tag, model)
        s1_emb = load_view(cfg, split_name, "s1", "combined", tag)
        gal_emb, _ = concat_gallery_emb(cfg, split_name, "combined", tag)
        from src.embeddings import ids_path

        gal_ids = []
        for src in ("s2", "s3"):
            p = ids_path(Path(cfg.paths.cache_dir), split_name, src)
            if p.exists():
                gal_ids.extend(np.load(p, allow_pickle=True).tolist())
        gal_map = gallery.set_index("entity_id")
        gal_aligned = gal_map.loc[[i for i in gal_ids if i in gal_map.index]].reset_index()
        keep = [i for i, eid in enumerate(gal_ids) if eid in gal_map.index]
        gal_emb = gal_emb[keep]
    except Exception as exc:
        LOG.warning("embeddings unavailable (%s); lexical blocking only", exc)
        s1_emb = gal_emb = None
        gal_aligned = gallery

    idf = load_idf_if_any(cfg)
    cand_map, pairs, bmetrics = run_blocking(
        cfg, s1, gal_aligned, s1_emb, gal_emb, idf,
        out_tsv=out_cand, tag=f"{split_name}_{tag}", truth_map=None,
    )
    feats = build_pair_features(pairs, s1, gal_aligned, idf, int(cfg.features.chunk_pairs))
    feats = _add_cosines(cfg, feats, s1, gal_aligned, split_name, split_name, tag, tag)
    score_cols = [c for c in feats.columns if c.startswith("score_") or c.startswith("cosine_")]
    feats = add_rank_context_features(feats, score_cols)

    lgbm_path = Path(cfg.paths.models_dir) / "lgbm.joblib"
    if lgbm_path.exists():
        booster, cols = load_matcher(lgbm_path)
        # missing columns → 0
        for c in cols:
            if c not in feats.columns:
                feats[c] = 0.0
        feats["p_gbm"] = predict_lgbm(booster, feats, cols)
    else:
        # lexical fallback score
        feats["p_gbm"] = feats.get("name_rf_token_set", pd.Series(0, index=feats.index)).fillna(0)

    ce_dir = Path(cfg.paths.models_dir) / "crossencoder"
    if ce_dir.exists() and (ce_dir / "crossencoder.pt").exists():
        from src.crossencoder import apply_ce_topn, load_crossencoder

        model_ce = load_crossencoder(cfg, ce_dir)
        feats = apply_ce_topn(model_ce, feats, s1, gal_aligned, cfg, "p_gbm")
    else:
        feats["p_ce"] = np.nan

    stack_path = Path(cfg.paths.models_dir) / "stacker.joblib"
    if stack_path.exists() and feats["p_ce"].notna().any():
        bundle = load_stacker(stack_path)
        feats["p"] = apply_stacker(bundle, feats)
    else:
        feats["p"] = feats["p_gbm"]

    s1_ids = s1["entity_id"].astype(str).tolist()
    pred = apply_decision(
        feats, s1_ids, str(cfg.decision.rule),
        float(cfg.decision.global_threshold), float(cfg.decision.relative_alpha),
        bool(cfg.decision.one_to_one), "p",
    )
    # exact candidate set the model scored
    scored_map = feats.groupby("s1_id")["cand_id"].apply(list).to_dict()
    write_candidate_pairs(s1_ids, scored_map, out_cand)
    write_matching_results(s1_ids, pred, out_match)
    LOG.info("wrote %s  and  %s  (rows=%s  predicted links=%s)",
             out_match, out_cand, f"{len(s1_ids):,}", f"{sum(len(v) for v in pred.values()):,}")
    save_parquet(feats, cache_dir(cfg) / f"{split_name}_scored.parquet")
    return {"blocking": bmetrics, "n_s1": len(s1_ids), "n_pred_links": sum(len(v) for v in pred.values())}


def stage_predict(cfg):
    banner("predict")
    t1 = pd.read_parquet(norm_path(cfg, "test", "s1"))
    t2 = pd.read_parquet(norm_path(cfg, "test", "s2"))
    t3 = pd.read_parquet(norm_path(cfg, "test", "s3"))
    gallery = gallery_of(t2, t3)
    out_dir = Path(cfg.paths.output_dir)
    metrics = _predict_split(
        cfg, t1, gallery, "test",
        out_dir / "matching_results.tsv",
        out_dir / "candidate_pairs.tsv",
    )
    # sanity: one row per test S1, France included
    s1_ids = t1["entity_id"].astype(str).tolist()
    pred = read_id_list_tsv(out_dir / "matching_results.tsv", "matched_entity_ids")
    cand = read_id_list_tsv(out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    n_fr = int((t1["country"].astype(str).str.lower() == "france").sum())
    fr_ids = t1.loc[t1["country"].astype(str).str.lower() == "france", "entity_id"].astype(str)
    fr_single = sum(1 for i in fr_ids if not pred.get(i))
    leak = sum(1 for s, ms in pred.items() if set(ms) - set(cand.get(s, [])))
    metrics.update(
        {
            "n_output_rows": len(pred),
            "n_test_s1": len(s1_ids),
            "n_france_s1": n_fr,
            "france_pred_singleton_rate": fr_single / max(n_fr, 1),
            "matches_not_in_candidates": leak,
        }
    )
    LOG.info(
        "sanity  output_rows=%s  test_S1=%s  France_S1=%s  France_singleton_rate=%.3f  matches_not_in_candidates=%s",
        metrics["n_output_rows"], metrics["n_test_s1"], metrics["n_france_s1"],
        metrics["france_pred_singleton_rate"], metrics["matches_not_in_candidates"],
    )
    if metrics["n_output_rows"] != metrics["n_test_s1"]:
        LOG.error("PDF rule failed: every test S1 must have exactly one output row")
    if metrics["matches_not_in_candidates"]:
        LOG.warning("matches not in candidate_pairs.tsv — validator will warn")
    _log(cfg, "predict", "Test inference → output/*.tsv.", metrics)

    # PDF validator (stdlib). Default omits --check-ids (memory); format rules still run.
    cmd = [
        sys.executable, "utils/validate_submission.py",
        "--matching", str(out_dir / "matching_results.tsv"),
        "--candidate", str(out_dir / "candidate_pairs.tsv"),
        "--test-dir", str(cfg.paths.test_dir),
    ]
    LOG.info("validator: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        LOG.info("%s", proc.stdout.strip())
    if proc.stderr:
        LOG.warning("%s", proc.stderr.strip())
    if proc.returncode == 0:
        LOG.info("validator PASS")
    else:
        LOG.error("validator FAIL (exit %s)", proc.returncode)
    metrics["validator_exit"] = proc.returncode
    metrics["validator_stdout"] = proc.stdout
    write_json(Path(cfg.paths.reports_dir) / "predict.json", metrics)
    return metrics


def stage_all(cfg):
    LOG.info("running full pipeline  eda → … → predict")
    stage_eda(cfg)
    stage_split(cfg)
    stage_normalize(cfg)
    stage_block(cfg)
    if cfg.biencoder.enabled and not cfg.debug.fast:
        try:
            stage_train_biencoder(cfg)
        except Exception:
            LOG.exception("train_biencoder failed — continuing with zero-shot embeddings")
    stage_features(cfg)
    stage_train_matcher(cfg)
    if cfg.cross_encoder.enabled and not cfg.debug.fast:
        try:
            stage_train_crossencoder(cfg)
        except Exception:
            LOG.exception("train_crossencoder failed — continuing with LightGBM scores only")
    try:
        stage_stack(cfg)
    except Exception:
        LOG.exception("stacker failed — decision will use p_gbm")
    stage_decide(cfg)
    stage_evaluate(cfg)
    return stage_predict(cfg)


DISPATCH = {
    "eda": stage_eda,
    "split": stage_split,
    "normalize": stage_normalize,
    "block": stage_block,
    "train_biencoder": stage_train_biencoder,
    "features": stage_features,
    "train_matcher": stage_train_matcher,
    "train_crossencoder": stage_train_crossencoder,
    "stack": stage_stack,
    "decide": stage_decide,
    "evaluate": stage_evaluate,
    "predict": stage_predict,
    "all": stage_all,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Business entity resolution pipeline")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--quiet", action="store_true", help="INFO only (hide DEBUG cache lines)")
    args = p.parse_args(argv)
    setup_logging(verbose=not args.quiet)
    cfg = load_config(args.config)
    set_seeds(int(cfg.seed))
    hw = apply_auto_batch(cfg)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.reports_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.models_dir).mkdir(parents=True, exist_ok=True)
    LOG.info("config=%s  seed=%s", cfg._path, cfg.seed)
    LOG.info("backbone=%s  repo=%s  prefix=%r", cfg.backbone.preset, cfg.backbone.repo, cfg.backbone.prefix)
    LOG.info("hardware %s", hw)
    _log(cfg, args.stage, f"start stage={args.stage}", {"hardware": hw, "backbone": as_dict(cfg)["backbone"]})
    DISPATCH[args.stage](cfg)
    LOG.info("finished --stage %s", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
