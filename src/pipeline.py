"""Shared load / cache helpers used by every CLI stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings import encode_frame_views, load_embeddings, load_id_index, load_sentence_transformer
from src.io_utils import load_ground_truth, load_sources
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
    s1, s2, s3 = load_sources(cfg.paths.train_dir, "train")
    gt = load_ground_truth(Path(cfg.paths.train_dir) / "train_ground_truth.tsv")
    s1 = maybe_subsample_s1(s1, cfg)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))].copy()
    return s1, s2, s3, gt


def load_test(cfg):
    t1, t2, t3 = load_sources(cfg.paths.test_dir, "test")
    t1 = maybe_subsample_s1(t1, cfg)
    return t1, t2, t3


def norm_path(cfg, split: str, source: str) -> Path:
    return cache_dir(cfg) / "normalized" / f"{split}_{source}.parquet"


def load_or_build_normalized(cfg, df: pd.DataFrame, split: str, source: str, extra_abbrev=None):
    path = norm_path(cfg, split, source)
    if path.exists():
        return pd.read_parquet(path)
    out = normalize_frame(df, extra_abbrev)
    out = attach_record_texts(out, cfg.backbone.prefix, cfg.backbone.record_template)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def build_all_normalized(cfg) -> dict:
    s1, s2, s3, gt = load_train(cfg)
    t1, t2, t3 = load_test(cfg)
    extra = {}
    if cfg.normalize.mine_abbreviations:
        # align S1 name/addr with first match
        s_all = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
        left_n, right_n, left_a, right_a = [], [], [], []
        for row in gt.itertuples(index=False):
            if not row.matched_list or row.source1_entity_id not in s_all.index:
                continue
            mid = row.matched_list[0]
            if mid not in s_all.index:
                continue
            left_n.append(s_all.loc[row.source1_entity_id].business_name)
            right_n.append(s_all.loc[mid].business_name)
            left_a.append(s_all.loc[row.source1_entity_id].business_address)
            right_a.append(s_all.loc[mid].business_address)
        extra.update(mine_abbreviations(left_n, right_n, int(cfg.normalize.min_abbrev_count)))
        extra.update(mine_abbreviations(left_a, right_a, int(cfg.normalize.min_abbrev_count)))
        (cache_dir(cfg) / "mined_abbrev.json").write_text(
            __import__("json").dumps(extra, indent=2), encoding="utf-8"
        )

    frames = {
        "train_s1": load_or_build_normalized(cfg, s1, "train", "s1", extra),
        "train_s2": load_or_build_normalized(cfg, s2, "train", "s2", extra),
        "train_s3": load_or_build_normalized(cfg, s3, "train", "s3", extra),
        "test_s1": load_or_build_normalized(cfg, t1, "test", "s1", extra),
        "test_s2": load_or_build_normalized(cfg, t2, "test", "s2", extra),
        "test_s3": load_or_build_normalized(cfg, t3, "test", "s3", extra),
    }
    if cfg.normalize.compute_idf:
        idf_path = cache_dir(cfg) / "idf.json"
        if not idf_path.exists():
            token_lists = []
            for df in frames.values():
                token_lists.extend(df["core_tokens"].astype(str).map(str.split).tolist())
                token_lists.extend(df["addr_tokens"].astype(str).map(str.split).tolist())
            idf = compute_idf(token_lists)
            save_idf(idf, idf_path)
    return frames


def load_idf_if_any(cfg) -> dict | None:
    p = cache_dir(cfg) / "idf.json"
    if p.exists():
        return load_idf(p)
    return None


def build_splits(cfg, s1: pd.DataFrame, gt: pd.DataFrame) -> dict:
    path = cache_dir(cfg) / "splits.json"
    if path.exists():
        return load_split(path)
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
