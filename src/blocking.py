"""Candidate generation: FAISS bi-encoder ∪ char TF-IDF ∪ key blocking.

Union, dedupe, cap K per S1. Optionally restrict to the same country string
(open-set: France is just another string; unknown countries still group).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import recall_at_k_curve
from src.io_utils import write_candidate_pairs
from src.logging_utils import LOG, write_json

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (x / n).astype(np.float32)


def _emb_fingerprint(mat: np.ndarray) -> str:
    if mat.size == 0:
        return "empty"
    head = np.ascontiguousarray(mat[0, : min(8, mat.shape[1])])
    tail = np.ascontiguousarray(mat[-1, : min(8, mat.shape[1])])
    h = hashlib.md5()
    h.update(str(mat.shape).encode())
    h.update(head.tobytes())
    h.update(tail.tobytes())
    return h.hexdigest()[:12]


def _safe_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))[:80]


def _faiss_paths(cache_dir: Path, tag: str, country: str, kind: str, nlist: int, xb: np.ndarray) -> tuple[Path, Path]:
    name = f"{_safe_token(country)}_{kind}_nlist{int(nlist)}_n{xb.shape[0]}_d{xb.shape[1]}_{_emb_fingerprint(xb)}.index"
    path = Path(cache_dir) / "faiss" / tag / name
    return path, path.with_name(path.name + ".ids.npy")


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


def _load_or_build_faiss(
    index_kind: str,
    nlist: int,
    xb: np.ndarray,
    cache_path: Path | None,
    cache_ids: np.ndarray | None,
):
    ids_path = cache_path.with_name(cache_path.name + ".ids.npy") if cache_path is not None else None
    if cache_path is not None and ids_path is not None and cache_path.exists() and ids_path.exists() and cache_ids is not None:
        saved = np.load(ids_path, allow_pickle=True)
        if saved.shape == cache_ids.shape and np.array_equal(saved, cache_ids):
            try:
                index = faiss.read_index(str(cache_path))
                LOG.info("FAISS cache hit  %s  ntotal=%s", cache_path, index.ntotal)
                return index
            except Exception as exc:
                LOG.warning("FAISS cache unreadable (%s); rebuilding", exc)
    index = _build_faiss(index_kind, xb.shape[1], nlist, xb)
    if index is not None and cache_path is not None and cache_ids is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(cache_path))
        np.save(ids_path, cache_ids)
        LOG.info("FAISS wrote  %s  ntotal=%s", cache_path, index.ntotal)
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
    cache_path: Path | None = None,
    cache_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xb = np.ascontiguousarray(xb.astype(np.float32))
    xq = np.ascontiguousarray(xq.astype(np.float32))
    if faiss is None:
        return _numpy_topk(xb, xq, k)
    index = _load_or_build_faiss(index_kind, nlist, xb, cache_path, cache_ids)
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
    cache_dir: Path | None = None,
    cache_tag: str | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Return s1_id -> [(gal_id, score), ...] best-first."""
    results: dict[str, list[tuple[str, float]]] = defaultdict(list)
    s1_ids = s1["entity_id"].astype(str).to_numpy()
    gal_ids = gallery["entity_id"].astype(str).to_numpy()

    def _search(xb, xq, kk, country: str, row_ids: np.ndarray):
        cache_path = None
        if cache_dir is not None and cache_tag:
            cache_path, _ = _faiss_paths(cache_dir, cache_tag, country, index_kind, nlist, xb)
        return search_topk(
            xb, xq, kk, index_kind, nlist, nprobe,
            cache_path=cache_path, cache_ids=row_ids,
        )

    if not same_country:
        scores, idxs = _search(gal_emb, s1_emb, k, "all", gal_ids)
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
        scores, idxs = _search(xb, xq, kk, country, gal_ids[gal_pos])
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
    cache_dir: Path | None = None,
    cache_tag: str | None = None,
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
        s1, gallery, s1_emb, gal_emb, k, same_country, index_kind, nlist, nprobe,
        cache_dir=cache_dir, cache_tag=cache_tag,
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
            idf_val = float(idf.get(tok, 0.0)) if idf is not None else 1.0
            if idf is not None and idf_val < threshold:
                continue
            for j in token_post.get(tok, ()):
                if same_country and gal_country[j] != s1_country[i]:
                    continue
                hits[gal_ids[j]] = hits.get(gal_ids[j], 0.0) + idf_val
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
    score_lu: dict[str, dict[str, dict[str, float]]] = {}
    rank_lu: dict[str, dict[str, dict[str, int]]] = {}
    names = list(score_maps.keys()) if score_maps else []
    if score_maps:
        for name, mp in score_maps.items():
            sm: dict[str, dict[str, float]] = {}
            rm: dict[str, dict[str, int]] = {}
            for s1, seq in mp.items():
                scores = {}
                ranks = {}
                for i, (cid, sc) in enumerate(seq):
                    scores[cid] = sc
                    ranks[cid] = i
                sm[s1] = scores
                rm[s1] = ranks
            score_lu[name] = sm
            rank_lu[name] = rm

    s1_col, cand_col, rank_col = [], [], []
    extra = {f"score_{n}": [] for n in names}
    extra.update({f"rank_{n}": [] for n in names})
    for s1, cids in cand_map.items():
        for rank, cid in enumerate(cids):
            s1_col.append(s1)
            cand_col.append(cid)
            rank_col.append(rank)
            for name in names:
                extra[f"score_{name}"].append(score_lu[name].get(s1, {}).get(cid, np.nan))
                extra[f"rank_{name}"].append(rank_lu[name].get(s1, {}).get(cid, -1))
    data = {"s1_id": s1_col, "cand_id": cand_col, "block_rank": rank_col}
    data.update(extra)
    return pd.DataFrame(data)


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
    LOG.info(
        "blocking tag=%s  S1=%s  gallery=%s  same_country=%s  k_faiss=%s k_tfidf=%s k_key=%s cap=%s",
        tag, f"{len(s1):,}", f"{len(gallery):,}", same,
        b.k_faiss, b.k_tfidf, b.k_key, b.k_cap,
    )

    if s1_emb is not None and gal_emb is not None:
        LOG.info("blocker A  FAISS %s  dim=%s", b.faiss_index, s1_emb.shape[1])
        faiss_map = block_faiss_by_country(
            s1, gallery, s1_emb, gal_emb,
            k=int(b.k_faiss),
            same_country=same,
            index_kind=str(b.faiss_index),
            nlist=int(b.faiss_nlist),
            nprobe=int(b.faiss_nprobe),
            cache_dir=Path(cfg.paths.cache_dir),
            cache_tag=f"{tag}_emb",
        )
        maps.append(faiss_map)
        named["faiss"] = faiss_map
        LOG.info("blocker A  done  (queries with hits=%s)", sum(1 for v in faiss_map.values() if v))

    LOG.info("blocker B  char TF-IDF %s–%s grams", b.tfidf_ngram_min, b.tfidf_ngram_max)
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
        cache_dir=Path(cfg.paths.cache_dir),
        cache_tag=f"{tag}_tfidf",
    )
    maps.append(tfidf_map)
    named["tfidf"] = tfidf_map
    LOG.info("blocker B  done")

    LOG.info("blocker C  rare name token | postcode+house")
    key_map = block_keys(
        s1, gallery, idf, float(b.rare_token_idf_quantile), int(b.k_key), same
    )
    maps.append(key_map)
    named["key"] = key_map
    LOG.info("blocker C  done")

    s1_ids = s1["entity_id"].astype(str).tolist()
    cand_map = union_cap(maps, s1_ids, int(b.k_cap))
    pairs = candidates_to_pairs(cand_map, named)
    LOG.info("union cap=%s  pairs=%s  avg/S1=%.1f", b.k_cap, f"{len(pairs):,}", len(pairs) / max(len(s1_ids), 1))

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
        LOG.info("blocking pair-recall=%.4f  entity-recall=%.4f  R@10_union=%s",
                 metrics["blocking"]["pair_recall"],
                 metrics["blocking"]["entity_recall"],
                 metrics["recall_at_k_union"].get(10))

    if out_tsv is not None:
        write_candidate_pairs(s1_ids, cand_map, out_tsv)
    write_json(Path(cfg.paths.reports_dir) / f"blocking_{tag}.json", metrics)
    return cand_map, pairs, metrics
