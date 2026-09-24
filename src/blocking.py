"""Candidate generation: FAISS bi-encoder ∪ char TF-IDF ∪ key blocking.

Union, dedupe, cap K per S1. Optionally restrict to the same country string
(open-set: France is just another string; unknown countries still group).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import recall_at_k_curve
from src.io_utils import write_candidate_pairs
from src.logging_utils import write_json

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (x / n).astype(np.float32)


def _build_faiss(index_kind: str, dim: int, nlist: int, xb: np.ndarray):
    if faiss is None:
        return None
    n = xb.shape[0]
    if index_kind == "flat" or n < max(nlist * 4, 2048):
        index = faiss.IndexFlatIP(dim)
        index.add(xb)
        return index
    if index_kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        index.add(xb)
        return index
    quantizer = faiss.IndexFlatIP(dim)
    nlist = int(min(nlist, max(n // 39, 1)))
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(xb[np.random.choice(n, size=min(n, max(nlist * 40, 256)), replace=False)])
    index.add(xb)
    return index


def _faiss_search(index, xq: np.ndarray, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, index.ntotal)
    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)
    scores, idxs = index.search(xq, k)
    return scores, idxs


def _numpy_topk(xb: np.ndarray, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force inner-product top-k (CPU fallback)."""
    k = min(k, xb.shape[0])
    sims = xq @ xb.T
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row = np.arange(xq.shape[0])[:, None]
    top_scores = sims[row, part]
    order = np.argsort(-top_scores, axis=1)
    idxs = part[row, order]
    scores = top_scores[row, order]
    return scores, idxs


def search_topk(
    xb: np.ndarray,
    xq: np.ndarray,
    k: int,
    index_kind: str = "ivf",
    nlist: int = 4096,
    nprobe: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    xb = np.ascontiguousarray(xb.astype(np.float32))
    xq = np.ascontiguousarray(xq.astype(np.float32))
    if faiss is None:
        return _numpy_topk(xb, xq, k)
    index = _build_faiss(index_kind, xb.shape[1], nlist, xb)
    if index is None:
        return _numpy_topk(xb, xq, k)
    return _faiss_search(index, xq, k, nprobe)


def _group_by_country(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Open-set grouping: every distinct country string is a bucket."""
    out = {}
    countries = df["country"].astype(str).to_numpy()
    for c in np.unique(countries):
        out[str(c)] = np.flatnonzero(countries == c)
    return out


def block_faiss_by_country(
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    s1_emb: np.ndarray,
    gal_emb: np.ndarray,
    k: int,
    same_country: bool,
    index_kind: str,
    nlist: int,
    nprobe: int,
) -> dict[str, list[tuple[str, float]]]:
    """Return s1_id -> [(gal_id, score), ...] best-first."""
    results: dict[str, list[tuple[str, float]]] = defaultdict(list)
    s1_ids = s1["entity_id"].astype(str).to_numpy()
    gal_ids = gallery["entity_id"].astype(str).to_numpy()
    if not same_country:
        scores, idxs = search_topk(gal_emb, s1_emb, k, index_kind, nlist, nprobe)
        for i, s1_id in enumerate(s1_ids):
            for sc, j in zip(scores[i], idxs[i]):
                if j < 0:
                    continue
                results[s1_id].append((gal_ids[j], float(sc)))
        return results

    s1_groups = _group_by_country(s1)
    gal_groups = _group_by_country(gallery)
    for country, s1_pos in s1_groups.items():
        gal_pos = gal_groups.get(country)
        if gal_pos is None or len(gal_pos) == 0:
            continue
        xb = gal_emb[gal_pos]
        xq = s1_emb[s1_pos]
        kk = min(k, xb.shape[0])
        scores, idxs = search_topk(xb, xq, kk, index_kind, nlist, nprobe)
        for row_i, gi in enumerate(s1_pos):
            s1_id = s1_ids[gi]
            for sc, j in zip(scores[row_i], idxs[row_i]):
                if j < 0:
                    continue
                results[s1_id].append((gal_ids[gal_pos[j]], float(sc)))
    return results


def block_tfidf_name(
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    k: int,
    ngram: tuple[int, int],
    max_features: int,
    min_df: int,
    svd_dim: int,
    same_country: bool,
    index_kind: str,
    nlist: int,
    nprobe: int,
) -> dict[str, list[tuple[str, float]]]:
    texts_s1 = s1["name_exp"].astype(str).tolist()
    texts_g = gallery["name_exp"].astype(str).tolist()
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram,
        max_features=max_features,
        min_df=min_df,
        lowercase=False,
    )
    vec.fit(texts_s1 + texts_g)
    xs = vec.transform(texts_s1)
    xg = vec.transform(texts_g)
    dim = min(svd_dim, min(xs.shape[1], xs.shape[0] + xg.shape[0] - 1, 256))
    dim = max(dim, 8)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    svd.fit(xg if xg.shape[0] >= dim else xs)
    s1_emb = _l2_normalize(svd.transform(xs).astype(np.float32))
    gal_emb = _l2_normalize(svd.transform(xg).astype(np.float32))
    return block_faiss_by_country(
        s1, gallery, s1_emb, gal_emb, k, same_country, index_kind, nlist, nprobe
    )


def block_keys(
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    idf: dict[str, float] | None,
    idf_quantile: float,
    k: int,
    same_country: bool,
) -> dict[str, list[tuple[str, float]]]:
    """Shared rare name token OR (postcode + house number)."""
    results: dict[str, list[tuple[str, float]]] = defaultdict(list)
    # invert gallery
    token_post = defaultdict(list)
    key_post = defaultdict(list)
    gal_country = gallery["country"].astype(str).to_numpy()
    gal_ids = gallery["entity_id"].astype(str).to_numpy()
    gal_core = gallery["core_tokens"].astype(str).to_numpy()
    gal_pc = gallery["postcode"].astype(str).to_numpy()
    gal_hn = gallery["house_number"].astype(str).to_numpy()

    threshold = 0.0
    if idf:
        vals = np.array(list(idf.values()), dtype=np.float32)
        threshold = float(np.quantile(vals, idf_quantile)) if len(vals) else 0.0

    for i, eid in enumerate(gal_ids):
        for tok in str(gal_core[i]).split():
            if idf is None or idf.get(tok, 0.0) >= threshold:
                token_post[tok].append(i)
        if gal_pc[i] and gal_hn[i]:
            key_post[(gal_pc[i], gal_hn[i])].append(i)

    s1_ids = s1["entity_id"].astype(str).to_numpy()
    s1_country = s1["country"].astype(str).to_numpy()
    s1_core = s1["core_tokens"].astype(str).to_numpy()
    s1_pc = s1["postcode"].astype(str).to_numpy()
    s1_hn = s1["house_number"].astype(str).to_numpy()

    for i, s1_id in enumerate(s1_ids):
        hits: dict[str, float] = {}
        for tok in str(s1_core[i]).split():
            if idf is not None and idf.get(tok, 0.0) < threshold:
                continue
            for j in token_post.get(tok, ()):
                if same_country and gal_country[j] != s1_country[i]:
                    continue
                hits[gal_ids[j]] = hits.get(gal_ids[j], 0.0) + float(idf.get(tok, 1.0) if idf else 1.0)
        if s1_pc[i] and s1_hn[i]:
            for j in key_post.get((s1_pc[i], s1_hn[i]), ()):
                if same_country and gal_country[j] != s1_country[i]:
                    continue
                hits[gal_ids[j]] = hits.get(gal_ids[j], 0.0) + 5.0
        ranked = sorted(hits.items(), key=lambda x: -x[1])[:k]
        results[s1_id] = ranked
    return results


def union_cap(
    maps: list[dict[str, list[tuple[str, float]]]],
    s1_ids: list[str],
    k_cap: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s1 in s1_ids:
        seen = set()
        ordered: list[str] = []
        # round-robin across blockers so no single source starves the others
        iters = [m.get(s1, []) for m in maps]
        changed = True
        while changed and len(ordered) < k_cap:
            changed = False
            for seq in iters:
                for cid, _ in seq:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    ordered.append(cid)
                    changed = True
                    break
                if len(ordered) >= k_cap:
                    break
        out[s1] = ordered
    return out


def candidates_to_pairs(
    cand_map: dict[str, list[str]],
    score_maps: dict[str, dict[str, list[tuple[str, float]]]] | None = None,
) -> pd.DataFrame:
    rows = []
    for s1, cids in cand_map.items():
        for rank, cid in enumerate(cids):
            rec = {"s1_id": s1, "cand_id": cid, "block_rank": rank}
            if score_maps:
                for name, mp in score_maps.items():
                    lookup = {a: b for a, b in mp.get(s1, [])}
                    rec[f"score_{name}"] = lookup.get(cid, np.nan)
                    rec[f"rank_{name}"] = next(
                        (i for i, (a, _) in enumerate(mp.get(s1, [])) if a == cid),
                        -1,
                    )
            rows.append(rec)
    return pd.DataFrame(rows)


def run_blocking(
    cfg,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    s1_emb: np.ndarray | None,
    gal_emb: np.ndarray | None,
    idf: dict[str, float] | None,
    out_tsv: Path | None,
    tag: str,
    truth_map: dict[str, set[str]] | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    b = cfg.blocking
    same = bool(b.same_country_only)
    maps = []
    named: dict[str, dict] = {}

    if s1_emb is not None and gal_emb is not None:
        faiss_map = block_faiss_by_country(
            s1, gallery, s1_emb, gal_emb,
            k=int(b.k_faiss),
            same_country=same,
            index_kind=str(b.faiss_index),
            nlist=int(b.faiss_nlist),
            nprobe=int(b.faiss_nprobe),
        )
        maps.append(faiss_map)
        named["faiss"] = faiss_map

    tfidf_map = block_tfidf_name(
        s1, gallery,
        k=int(b.k_tfidf),
        ngram=(int(b.tfidf_ngram_min), int(b.tfidf_ngram_max)),
        max_features=int(b.tfidf_max_features),
        min_df=int(b.tfidf_min_df),
        svd_dim=int(b.tfidf_svd_dim),
        same_country=same,
        index_kind=str(b.faiss_index),
        nlist=int(b.faiss_nlist),
        nprobe=int(b.faiss_nprobe),
    )
    maps.append(tfidf_map)
    named["tfidf"] = tfidf_map

    key_map = block_keys(
        s1, gallery, idf, float(b.rare_token_idf_quantile), int(b.k_key), same
    )
    maps.append(key_map)
    named["key"] = key_map

    s1_ids = s1["entity_id"].astype(str).tolist()
    cand_map = union_cap(maps, s1_ids, int(b.k_cap))
    pairs = candidates_to_pairs(cand_map, named)

    metrics: dict = {
        "n_s1": len(s1_ids),
        "n_gallery": int(len(gallery)),
        "n_pairs": int(len(pairs)),
        "avg_candidates": (len(pairs) / max(len(s1_ids), 1)),
        "tag": tag,
    }
    if truth_map is not None:
        ranked = {s: [c for c, _ in named.get("faiss", {}).get(s, [])] for s in s1_ids}
        # union ranked by first appearance
        union_ranked = {s: cand_map[s] for s in s1_ids}
        ks = [1, 5, 10, 20, 30, 50, int(b.k_cap)]
        metrics["recall_at_k_union"] = recall_at_k_curve(union_ranked, truth_map, ks)
        if "faiss" in named:
            metrics["recall_at_k_faiss"] = recall_at_k_curve(ranked, truth_map, ks)
        from src.evaluate import blocking_recall

        metrics["blocking"] = blocking_recall(cand_map, truth_map, s1_ids)

    if out_tsv is not None:
        write_candidate_pairs(s1_ids, cand_map, out_tsv)
    write_json(Path(cfg.paths.reports_dir) / f"blocking_{tag}.json", metrics)
    return cand_map, pairs, metrics
