"""Shared load / cache helpers used by every CLI stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings import encode_frame_views, load_embeddings, load_id_index
from src.io_utils import load_ground_truth, load_sources
from src.logging_utils import LOG
from src.normalize import (
    attach_record_texts,
    compute_idf,
    load_idf,
    mine_abbreviations,
    normalize_frame,
    save_idf,
)
from src.split import (
    load_split,
    loco_masks,
    make_holdout,
    make_oof_folds,
    save_split,
)


def cache_dir(cfg) -> Path:
    p = Path(cfg.paths.cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_subsample_s1(s1: pd.DataFrame, cfg) -> pd.DataFrame:
    n = getattr(cfg.debug, "max_s1", None)
    if n:
        return s1.sample(n=min(int(n), len(s1)), random_state=int(cfg.seed)).reset_index(drop=True)
    return s1


def load_train(cfg):
    LOG.info("reading train TSVs from %s (sep=tab, dtype=str)", cfg.paths.train_dir)
    s1, s2, s3 = load_sources(cfg.paths.train_dir, "train")
    gt = load_ground_truth(Path(cfg.paths.train_dir) / "train_ground_truth.tsv")
    LOG.info("train rows  S1=%s  S2=%s  S3=%s  GT=%s", f"{len(s1):,}", f"{len(s2):,}", f"{len(s3):,}", f"{len(gt):,}")
    s1 = maybe_subsample_s1(s1, cfg)
    if getattr(cfg.debug, "max_s1", None):
        LOG.info("debug.max_s1=%s → S1 now %s", cfg.debug.max_s1, f"{len(s1):,}")
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))].copy()
    return s1, s2, s3, gt


def load_test(cfg):
    LOG.info("reading test TSVs from %s", cfg.paths.test_dir)
    t1, t2, t3 = load_sources(cfg.paths.test_dir, "test")
    LOG.info("test rows   S1=%s  S2=%s  S3=%s", f"{len(t1):,}", f"{len(t2):,}", f"{len(t3):,}")
    n_fr = int((t1["country"].astype(str).str.lower() == "france").sum())
    LOG.info("test S1 with country=France: %s (every test S1 must appear in the output)", f"{n_fr:,}")
    t1 = maybe_subsample_s1(t1, cfg)
    return t1, t2, t3


def norm_path(cfg, split: str, source: str) -> Path:
    return cache_dir(cfg) / "normalized" / f"{split}_{source}.parquet"


def parquet_nrows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.read_metadata(path).num_rows)
    except Exception:
        return -1


def normalized_sources() -> tuple[tuple[str, str], ...]:
    return (
        ("train", "s1"),
        ("train", "s2"),
        ("train", "s3"),
        ("test", "s1"),
        ("test", "s2"),
        ("test", "s3"),
    )


def all_normalized_cached(cfg) -> bool:
    for split, source in normalized_sources():
        if not norm_path(cfg, split, source).exists():
            return False
    if bool(getattr(cfg.normalize, "compute_idf", True)):
        if not (cache_dir(cfg) / "idf.json").exists():
            return False
    return True


def cached_normalized_counts(cfg) -> dict[str, int]:
    out = {}
    for split, source in normalized_sources():
        p = norm_path(cfg, split, source)
        out[f"{split}_{source}"] = parquet_nrows(p) if p.exists() else 0
    out["idf"] = int((cache_dir(cfg) / "idf.json").exists())
    return out


def load_or_build_normalized(cfg, df: pd.DataFrame, split: str, source: str, extra_abbrev=None):
    path = norm_path(cfg, split, source)
    if path.exists():
        n = parquet_nrows(path)
        LOG.info("cache hit  %s  (%s rows on disk, skip normalize)", path, f"{n:,}" if n >= 0 else "?")
        return pd.read_parquet(path)
    n_jobs = int(getattr(cfg.normalize, "n_jobs", -1))
    LOG.info(
        "normalizing %s/%s  n=%s  n_jobs=%s  → %s",
        split, source, f"{len(df):,}", n_jobs, path,
    )
    out = normalize_frame(df, extra_abbrev, n_jobs=n_jobs)
    out = attach_record_texts(out, cfg.backbone.prefix, cfg.backbone.record_template)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    mb = path.stat().st_size / (1024 * 1024)
    LOG.info("wrote %s  rows=%s  %.1f MB  (resume-safe)", path, f"{len(out):,}", mb)
    return out


def _load_or_mine_abbrev(cfg, s1, s2, s3, gt) -> dict:
    path = cache_dir(cfg) / "mined_abbrev.json"
    if path.exists():
        extra = json.loads(path.read_text(encoding="utf-8"))
        LOG.info("cache hit  mined abbreviations  %s  (%s entries)", path, len(extra))
        return extra
    if not cfg.normalize.mine_abbreviations:
        return {}
    LOG.info("mining abbreviations from matched train pairs (join, not row-loc)")
    s_all = pd.concat(
        [
            s1[["entity_id", "business_name", "business_address"]],
            s2[["entity_id", "business_name", "business_address"]],
            s3[["entity_id", "business_name", "business_address"]],
        ],
        ignore_index=True,
    ).drop_duplicates("entity_id")
    pairs = gt.copy()
    pairs["matched_id"] = pairs["matched_list"].map(lambda xs: xs[0] if xs else "")
    pairs = pairs[pairs["matched_id"].astype(str) != ""]
    cap = int(getattr(cfg.normalize, "max_abbrev_pairs", 0) or 0)
    if cap and len(pairs) > cap:
        pairs = pairs.sample(n=cap, random_state=int(cfg.seed))
        LOG.info("abbrev mine sampled to %s pairs (normalize.max_abbrev_pairs)", f"{cap:,}")
    left = pairs.merge(s_all, left_on="source1_entity_id", right_on="entity_id", how="inner")
    right = s_all.rename(
        columns={
            "entity_id": "matched_id",
            "business_name": "r_name",
            "business_address": "r_addr",
        }
    )
    aligned = left.merge(right, on="matched_id", how="inner")
    LOG.info("abbrev-aligned pairs=%s", f"{len(aligned):,}")
    extra = {}
    extra.update(
        mine_abbreviations(
            aligned["business_name"].tolist(),
            aligned["r_name"].tolist(),
            int(cfg.normalize.min_abbrev_count),
        )
    )
    extra.update(
        mine_abbreviations(
            aligned["business_address"].tolist(),
            aligned["r_addr"].tolist(),
            int(cfg.normalize.min_abbrev_count),
        )
    )
    path.write_text(json.dumps(extra, indent=2), encoding="utf-8")
    LOG.info("mined %s abbreviation substitutions → %s", len(extra), path)
    return extra


def build_all_normalized(cfg) -> dict:
    """Build or reuse cache/normalized/*.parquet. Each source is written as soon
    as it finishes, so a killed job continues from the next missing file.
    """
    if all_normalized_cached(cfg):
        counts = cached_normalized_counts(cfg)
        LOG.info("normalize cache complete — not re-reading TSVs")
        for k, n in counts.items():
            if k != "idf":
                LOG.info("  %s  %s rows", k, f"{n:,}" if n >= 0 else "?")
        return counts

    pending = [(sp, src) for sp, src in normalized_sources() if not norm_path(cfg, sp, src).exists()]
    need_train = any(sp == "train" for sp, _ in pending) or (
        bool(cfg.normalize.mine_abbreviations)
        and not (cache_dir(cfg) / "mined_abbrev.json").exists()
    )
    need_test = any(sp == "test" for sp, _ in pending)
    s1 = s2 = s3 = gt = t1 = t2 = t3 = None
    if need_train:
        s1, s2, s3, gt = load_train(cfg)
    if need_test:
        t1, t2, t3 = load_test(cfg)
    extra = {}
    if need_train and s1 is not None:
        extra = _load_or_mine_abbrev(cfg, s1, s2, s3, gt)
    elif (cache_dir(cfg) / "mined_abbrev.json").exists():
        extra = json.loads((cache_dir(cfg) / "mined_abbrev.json").read_text(encoding="utf-8"))

    raw = {
        ("train", "s1"): s1,
        ("train", "s2"): s2,
        ("train", "s3"): s3,
        ("test", "s1"): t1,
        ("test", "s2"): t2,
        ("test", "s3"): t3,
    }
    frames = {}
    LOG.info(
        "normalize plan  cached=%s  pending=%s  %s",
        6 - len(pending),
        len(pending),
        [f"{a}_{b}" for a, b in pending] or "none",
    )
    idf_path = cache_dir(cfg) / "idf.json"
    idf_ok = idf_path.exists() or not bool(cfg.normalize.compute_idf)
    for split, source in normalized_sources():
        key = f"{split}_{source}"
        path = norm_path(cfg, split, source)
        if path.exists() and idf_ok:
            n = parquet_nrows(path)
            LOG.info("cache hit  %s  (%s rows on disk, skip load)", path, f"{n:,}" if n >= 0 else "?")
            frames[key] = None
        else:
            df_raw = raw[(split, source)]
            if df_raw is None:
                # needed for IDF or missing parquet — load that split now
                if split == "train":
                    s1, s2, s3, gt = load_train(cfg)
                    raw[("train", "s1")], raw[("train", "s2")], raw[("train", "s3")] = s1, s2, s3
                    df_raw = raw[(split, source)]
                else:
                    t1, t2, t3 = load_test(cfg)
                    raw[("test", "s1")], raw[("test", "s2")], raw[("test", "s3")] = t1, t2, t3
                    df_raw = raw[(split, source)]
            frames[key] = load_or_build_normalized(cfg, df_raw, split, source, extra)

    if cfg.normalize.compute_idf:
        if idf_path.exists():
            LOG.info("cache hit  IDF  %s", idf_path)
        else:
            LOG.info("computing IDF over train+test token fields (unlabeled X only)")
            token_lists = []
            for df in frames.values():
                if df is None:
                    continue
                token_lists.extend(df["core_tokens"].astype(str).map(str.split).tolist())
                token_lists.extend(df["addr_tokens"].astype(str).map(str.split).tolist())
            cap = int(getattr(cfg.normalize, "max_idf_docs", 0) or 0)
            if cap and len(token_lists) > cap:
                n_docs = len(token_lists)
                rng = np.random.RandomState(int(cfg.seed))
                pick = rng.choice(n_docs, size=cap, replace=False)
                token_lists = [token_lists[i] for i in pick]
                LOG.info("IDF sampled %s / %s docs (normalize.max_idf_docs)", f"{cap:,}", f"{n_docs:,}")
            idf = compute_idf(token_lists)
            save_idf(idf, idf_path)
            LOG.info("IDF vocab size %s → %s", f"{len(idf):,}", idf_path)
    return {k: (v if v is not None else parquet_nrows(norm_path(cfg, *k.split("_", 1)))) for k, v in frames.items()}


def load_idf_if_any(cfg) -> dict | None:
    p = cache_dir(cfg) / "idf.json"
    if p.exists():
        return load_idf(p)
    return None


def build_splits(cfg, s1: pd.DataFrame, gt: pd.DataFrame) -> dict:
    path = cache_dir(cfg) / "splits.json"
    if path.exists():
        LOG.info("cache hit  splits %s", path)
        return load_split(path)
    LOG.info("building hold-out (val_frac=%s) and %s OOF folds on %s S1 entities",
             cfg.split.val_frac, cfg.split.n_folds, f"{len(s1):,}")
    tr, va = make_holdout(s1, gt, float(cfg.split.val_frac), int(cfg.seed))
    folds = make_oof_folds(s1, gt, int(cfg.split.n_folds), int(cfg.seed))
    ids = s1["entity_id"].astype(str).to_numpy()
    payload = {
        "s1_ids": ids.tolist(),
        "holdout_train": ids[tr].tolist(),
        "holdout_val": ids[va].tolist(),
        "folds": [[ids[a].tolist(), ids[b].tolist()] for a, b in folds],
        "loco": {c: ids[m].tolist() for c, m in loco_masks(s1).items()},
    }
    save_split(path, payload)
    return payload


def gallery_of(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([s2, s3], ignore_index=True)


def encode_split_views(cfg, df: pd.DataFrame, split: str, source: str, tag: str, model=None):
    return encode_frame_views(cfg, df, split, source, tag, model=model)


def load_view(cfg, split: str, source: str, view: str, tag: str) -> np.ndarray:
    from src.embeddings import embedding_path

    return load_embeddings(embedding_path(Path(cfg.paths.cache_dir), split, source, view, tag))


def concat_gallery_emb(cfg, split: str, view: str, tag: str) -> tuple[np.ndarray, dict[str, int]]:
    from src.embeddings import embedding_path, ids_path

    cache = Path(cfg.paths.cache_dir)
    mats, ids = [], []
    offset = 0
    index: dict[str, int] = {}
    for source in ("s2", "s3"):
        p = embedding_path(cache, split, source, view, tag)
        ip = ids_path(cache, split, source)
        if not p.exists():
            continue
        m = load_embeddings(p)
        id_arr = np.load(ip, allow_pickle=True)
        mats.append(m)
        for i, eid in enumerate(id_arr):
            index[str(eid)] = offset + i
        offset += len(id_arr)
    if not mats:
        raise FileNotFoundError(f"no gallery embeddings for {split}/{view}/{tag}")
    return np.vstack(mats), index
