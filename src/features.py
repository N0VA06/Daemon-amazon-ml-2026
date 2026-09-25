"""Pair features for LightGBM. No country categorical — only same_country."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from tqdm.auto import tqdm

from src.io_utils import source_flag
from src.logging_utils import LOG, fmt_elapsed
from src.normalize import idf_token_jaccard, tokens


def _safe_ratio(a: str, b: str, fn) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(fn(a, b)) / 100.0


def _jw(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(JaroWinkler.similarity(a, b))


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a or ""), len(b or "")
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def _overlap(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _match_miss(a: str, b: str) -> tuple[int, int, int]:
    """match / mismatch / missing flags."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 0, 0, 1
    if not a or not b:
        return 0, 0, 1
    if a == b:
        return 1, 0, 0
    return 0, 1, 0


def _direction_conflict(a: str, b: str) -> int:
    da = set((a or "").split("|")) - {""}
    db = set((b or "").split("|")) - {""}
    if not da or not db:
        return 0
    opposites = {("e", "w"), ("w", "e"), ("n", "s"), ("s", "n")}
    for x in da:
        for y in db:
            if (x, y) in opposites:
                return 1
    return 0


def _acronym_match(a_acr: str, b_acr: str, a_core: str, b_core: str) -> int:
    compact = lambda s: "".join(ch for ch in (s or "") if ch.isalnum())
    ca, cb = compact(a_core), compact(b_core)
    if a_acr and (a_acr == cb or a_acr == b_acr):
        return 1
    if b_acr and (b_acr == ca or a_acr == b_acr):
        return 1
    if a_acr and b_acr and a_acr == b_acr:
        return 1
    return 0


def pair_string_features(a: pd.Series, b: pd.Series, idf: dict[str, float] | None) -> dict:
    an, bn = a["name_exp"], b["name_exp"]
    ac, bc = a["core_name"], b["core_name"]
    aa, ba = a["address_exp"], b["address_exp"]
    feats = {
        "name_rf_ratio": _safe_ratio(an, bn, fuzz.ratio),
        "name_rf_token_sort": _safe_ratio(an, bn, fuzz.token_sort_ratio),
        "name_rf_token_set": _safe_ratio(an, bn, fuzz.token_set_ratio),
        "name_rf_partial": _safe_ratio(an, bn, fuzz.partial_ratio),
        "name_jw": _jw(an, bn),
        "core_rf_ratio": _safe_ratio(ac, bc, fuzz.ratio),
        "core_rf_token_sort": _safe_ratio(ac, bc, fuzz.token_sort_ratio),
        "core_rf_token_set": _safe_ratio(ac, bc, fuzz.token_set_ratio),
        "core_rf_partial": _safe_ratio(ac, bc, fuzz.partial_ratio),
        "core_jw": _jw(ac, bc),
        "addr_rf_ratio": _safe_ratio(aa, ba, fuzz.ratio),
        "addr_rf_token_sort": _safe_ratio(aa, ba, fuzz.token_sort_ratio),
        "addr_rf_token_set": _safe_ratio(aa, ba, fuzz.token_set_ratio),
        "addr_rf_partial": _safe_ratio(aa, ba, fuzz.partial_ratio),
        "addr_jw": _jw(aa, ba),
        "name_len_ratio": _len_ratio(an, bn),
        "addr_len_ratio": _len_ratio(aa, ba),
        "legal_form_match": int(bool(a["legal_form"]) and a["legal_form"] == b["legal_form"]),
        "legal_form_mismatch": int(
            bool(a["legal_form"]) and bool(b["legal_form"]) and a["legal_form"] != b["legal_form"]
        ),
        "acronym_match": _acronym_match(a["acronym"], b["acronym"], ac, bc),
        "same_country": int(str(a["country_raw"]).strip() == str(b["country_raw"]).strip()),
        "source_s2": int(source_flag(str(b["entity_id"])) == "S2"),
        "source_s3": int(source_flag(str(b["entity_id"])) == "S3"),
    }
    hn_m, hn_x, hn_miss = _match_miss(a["house_number"], b["house_number"])
    pc_m, pc_x, pc_miss = _match_miss(a["postcode"], b["postcode"])
    un_m, un_x, un_miss = _match_miss(a["unit"], b["unit"])
    feats.update(
        {
            "house_num_match": hn_m,
            "house_num_mismatch": hn_x,
            "house_num_missing": hn_miss,
            "postcode_match": pc_m,
            "postcode_mismatch": pc_x,
            "postcode_missing": pc_miss,
            "unit_match": un_m,
            "unit_mismatch": un_x,
            "unit_missing": un_miss,
            "direction_conflict": _direction_conflict(a["directions"], b["directions"]),
            "landmark_overlap": _overlap(
                (a["landmarks"] or "").split("|"), (b["landmarks"] or "").split("|")
            ),
            "digit_jaccard": _overlap(
                (a["addr_digits"] or "").split("|"), (b["addr_digits"] or "").split("|")
            ),
            "addr_token_jaccard": _overlap(tokens(aa), tokens(ba)),
            "name_token_jaccard": _overlap(tokens(an), tokens(bn)),
            "core_token_jaccard": _overlap(tokens(ac), tokens(bc)),
            "s1_missing_addr": int(str(a.get("missing_addr", "0")) == "1"),
            "cand_missing_addr": int(str(b.get("missing_addr", "0")) == "1"),
            "s1_missing_postcode": int(str(a.get("missing_postcode", "0")) == "1"),
            "cand_missing_postcode": int(str(b.get("missing_postcode", "0")) == "1"),
            "s1_missing_house": int(str(a.get("missing_house", "0")) == "1"),
            "cand_missing_house": int(str(b.get("missing_house", "0")) == "1"),
        }
    )
    if idf is not None:
        feats["name_idf_jaccard"] = idf_token_jaccard(tokens(an), tokens(bn), idf)
        feats["core_idf_jaccard"] = idf_token_jaccard(tokens(ac), tokens(bc), idf)
        feats["addr_idf_jaccard"] = idf_token_jaccard(tokens(aa), tokens(ba), idf)
    else:
        feats["name_idf_jaccard"] = feats["name_token_jaccard"]
        feats["core_idf_jaccard"] = feats["core_token_jaccard"]
        feats["addr_idf_jaccard"] = feats["addr_token_jaccard"]
    return feats


def _cosine_from_index(
    s1_id: str,
    cand_id: str,
    s1_index: dict[str, int],
    gal_index: dict[str, int],
    s1_emb: np.ndarray | None,
    gal_emb: np.ndarray | None,
) -> float:
    if s1_emb is None or gal_emb is None:
        return np.nan
    i = s1_index.get(s1_id)
    j = gal_index.get(cand_id)
    if i is None or j is None:
        return np.nan
    return float(np.dot(s1_emb[i], gal_emb[j]))


def add_embedding_features(
    pairs: pd.DataFrame,
    s1_index: dict[str, int],
    gal_index: dict[str, int],
    emb_s1: dict[str, np.ndarray],
    emb_gal: dict[str, np.ndarray],
    prefix: str,
) -> pd.DataFrame:
    out = pairs
    i = pd.Series(s1_index).reindex(out["s1_id"].astype(str)).to_numpy()
    j = pd.Series(gal_index).reindex(out["cand_id"].astype(str)).to_numpy()
    valid = np.isfinite(i) & np.isfinite(j)
    ii = np.zeros(len(out), dtype=np.int64)
    jj = np.zeros(len(out), dtype=np.int64)
    ii[valid] = i[valid].astype(np.int64)
    jj[valid] = j[valid].astype(np.int64)
    for view, mat_s1 in emb_s1.items():
        mat_g = emb_gal.get(view)
        col = f"cosine_{prefix}_{view}" if prefix else f"cosine_{view}"
        vals = np.full(len(out), np.nan, dtype=np.float32)
        if mat_s1 is not None and mat_g is not None and valid.any():
            vals[valid] = np.einsum("ij,ij->i", mat_s1[ii[valid]], mat_g[jj[valid]])
        out[col] = vals
    return out


def add_rank_context_features(pairs: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Per-S1 rank, gap-to-best, n_candidates; per-candidate mutual-best vs S1s."""
    df = pairs.copy()
    df["n_candidates"] = df.groupby("s1_id")["cand_id"].transform("count")
    for col in score_cols:
        if col not in df.columns:
            continue
        best = df.groupby("s1_id")[col].transform("max")
        df[f"{col}_gap_to_best"] = best - df[col]
        df[f"{col}_rank"] = df.groupby("s1_id")[col].rank(ascending=False, method="first")
    # mutual-best: this S1 is the candidate's best S1 under the first available score
    primary = next((c for c in score_cols if c in df.columns), None)
    if primary is not None:
        best_s1_for_cand = df.loc[df.groupby("cand_id")[primary].idxmax()]
        mutual = set(zip(best_s1_for_cand["s1_id"], best_s1_for_cand["cand_id"]))
        df["mutual_best"] = [
            int((s1, c) in mutual) for s1, c in zip(df["s1_id"], df["cand_id"])
        ]
        # margin of this candidate to its second-best S1
        def _margin(g: pd.DataFrame) -> pd.Series:
            scores = g[primary].to_numpy()
            if len(scores) < 2:
                return pd.Series(np.zeros(len(g)), index=g.index)
            top2 = np.partition(scores, -2)
            second = top2[-2]
            return pd.Series(g[primary].to_numpy() - second, index=g.index)

        df["cand_margin_second_s1"] = (
            df.groupby("cand_id", group_keys=False).apply(_margin)
        )
    else:
        df["mutual_best"] = 0
        df["cand_margin_second_s1"] = 0.0
    return df


def _feat_cols() -> tuple[str, ...]:
    return (
        "name_exp",
        "core_name",
        "address_exp",
        "legal_form",
        "acronym",
        "country_raw",
        "house_number",
        "postcode",
        "unit",
        "directions",
        "landmarks",
        "addr_digits",
        "missing_addr",
        "missing_postcode",
        "missing_house",
        "entity_id",
    )


def _as_str_list(values) -> list[str]:
    out: list[str] = []
    for x in values:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            out.append("")
        else:
            out.append(str(x))
    return out


def _rf_lists(a, b, scorer, divide: float = 100.0, workers: int = -1) -> np.ndarray:
    """Aligned RapidFuzz over two string lists (cpdist, multi-worker)."""
    left, right = _as_str_list(a), _as_str_list(b)
    if not left:
        return np.zeros(0, dtype=np.float32)
    scores = np.asarray(
        cpdist(left, right, scorer=scorer, workers=workers, dtype=np.float32),
        dtype=np.float32,
    )
    if divide:
        scores = scores / np.float32(divide)
    return scores


def _join_pair_tables(pairs: pd.DataFrame, s1: pd.DataFrame, gallery: pd.DataFrame) -> pd.DataFrame:
    """pairs.join(s1).join(gallery) on entity ids — no per-row loc."""
    keep = [c for c in _feat_cols() if c != "entity_id"]
    s1_j = s1.set_index("entity_id")[[c for c in keep if c in s1.columns]].add_prefix("a_")
    g_j = gallery.set_index("entity_id")[[c for c in keep if c in gallery.columns]].add_prefix("b_")
    return pairs.join(s1_j, on="s1_id").join(g_j, on="cand_id")


def _len_ratio_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = np.fromiter((len("" if x is None else str(x)) for x in a), dtype=np.int32, count=len(a))
    lb = np.fromiter((len("" if x is None else str(x)) for x in b), dtype=np.int32, count=len(b))
    mx = np.maximum(la, lb)
    mn = np.minimum(la, lb)
    out = np.ones(len(a), dtype=np.float32)
    nz = mx > 0
    out[nz] = mn[nz] / mx[nz]
    return out


def _overlap_arr(a: np.ndarray, b: np.ndarray, sep: str = "|") -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sa = set(str(x).split(sep)) - {""} if x is not None and not (isinstance(x, float) and np.isnan(x)) else set()
        sb = set(str(y).split(sep)) - {""} if y is not None and not (isinstance(y, float) and np.isnan(y)) else set()
        if not sa and not sb:
            out[i] = 1.0
        elif not sa or not sb:
            out[i] = 0.0
        else:
            out[i] = len(sa & sb) / len(sa | sb)
    return out


def _token_overlap_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sa = set(tokens("" if x is None else str(x)))
        sb = set(tokens("" if y is None else str(y)))
        if not sa and not sb:
            out[i] = 1.0
        elif not sa or not sb:
            out[i] = 0.0
        else:
            out[i] = len(sa & sb) / len(sa | sb)
    return out


def _idf_jaccard_arr(a: np.ndarray, b: np.ndarray, idf: dict[str, float]) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = idf_token_jaccard(tokens("" if x is None else str(x)), tokens("" if y is None else str(y)), idf)
    return out


def _match_miss_arr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_s = np.array([("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)).strip() for x in a], dtype=object)
    b_s = np.array([("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)).strip() for x in b], dtype=object)
    a_empty = a_s == ""
    b_empty = b_s == ""
    missing = (a_empty | b_empty).astype(np.int8)
    match = ((a_s == b_s) & ~a_empty & ~b_empty).astype(np.int8)
    mismatch = ((a_s != b_s) & ~a_empty & ~b_empty).astype(np.int8)
    return match, mismatch, missing


def _direction_conflict_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    opposites = {("e", "w"), ("w", "e"), ("n", "s"), ("s", "n")}
    out = np.zeros(len(a), dtype=np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        da = set(str(x).split("|")) - {""} if x is not None else set()
        db = set(str(y).split("|")) - {""} if y is not None else set()
        if not da or not db:
            continue
        for p in da:
            for q in db:
                if (p, q) in opposites:
                    out[i] = 1
                    break
    return out


def _acronym_match_arr(a_acr, b_acr, a_core, b_core) -> np.ndarray:
    out = np.zeros(len(a_acr), dtype=np.int8)
    compact = lambda s: "".join(ch for ch in ("" if s is None else str(s)) if ch.isalnum())
    for i, (aa, ba, ac, bc) in enumerate(zip(a_acr, b_acr, a_core, b_core)):
        aa = "" if aa is None else str(aa)
        ba = "" if ba is None else str(ba)
        ca, cb = compact(ac), compact(bc)
        if aa and (aa == cb or aa == ba):
            out[i] = 1
        elif ba and (ba == ca or aa == ba):
            out[i] = 1
    return out


def _features_joined(
    sl: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    idf,
    workers: int = -1,
) -> pd.DataFrame:
    """pairs.join(s1).join(gallery), then RapidFuzz cpdist on the aligned lists."""
    joined = _join_pair_tables(sl, s1, gallery)
    an = joined["a_name_exp"].tolist()
    bn = joined["b_name_exp"].tolist()
    ac = joined["a_core_name"].tolist()
    bc = joined["b_core_name"].tolist()
    aa = joined["a_address_exp"].tolist()
    ba = joined["b_address_exp"].tolist()

    feats = pd.DataFrame({
        "s1_id": joined["s1_id"].to_numpy(),
        "cand_id": joined["cand_id"].to_numpy(),
        "name_rf_ratio": _rf_lists(an, bn, fuzz.ratio, workers=workers),
        "name_rf_token_sort": _rf_lists(an, bn, fuzz.token_sort_ratio, workers=workers),
        "name_rf_token_set": _rf_lists(an, bn, fuzz.token_set_ratio, workers=workers),
        "name_rf_partial": _rf_lists(an, bn, fuzz.partial_ratio, workers=workers),
        "name_jw": _rf_lists(an, bn, JaroWinkler.similarity, divide=1.0, workers=workers),
        "core_rf_ratio": _rf_lists(ac, bc, fuzz.ratio, workers=workers),
        "core_rf_token_sort": _rf_lists(ac, bc, fuzz.token_sort_ratio, workers=workers),
        "core_rf_token_set": _rf_lists(ac, bc, fuzz.token_set_ratio, workers=workers),
        "core_rf_partial": _rf_lists(ac, bc, fuzz.partial_ratio, workers=workers),
        "core_jw": _rf_lists(ac, bc, JaroWinkler.similarity, divide=1.0, workers=workers),
        "addr_rf_ratio": _rf_lists(aa, ba, fuzz.ratio, workers=workers),
        "addr_rf_token_sort": _rf_lists(aa, ba, fuzz.token_sort_ratio, workers=workers),
        "addr_rf_token_set": _rf_lists(aa, ba, fuzz.token_set_ratio, workers=workers),
        "addr_rf_partial": _rf_lists(aa, ba, fuzz.partial_ratio, workers=workers),
        "addr_jw": _rf_lists(aa, ba, JaroWinkler.similarity, divide=1.0, workers=workers),
        "name_len_ratio": _len_ratio_arr(joined["a_name_exp"].to_numpy(), joined["b_name_exp"].to_numpy()),
        "addr_len_ratio": _len_ratio_arr(joined["a_address_exp"].to_numpy(), joined["b_address_exp"].to_numpy()),
    })
    a_lf = joined["a_legal_form"].fillna("").astype(str)
    b_lf = joined["b_legal_form"].fillna("").astype(str)
    feats["legal_form_match"] = ((a_lf != "") & (a_lf == b_lf)).astype(np.int8)
    feats["legal_form_mismatch"] = ((a_lf != "") & (b_lf != "") & (a_lf != b_lf)).astype(np.int8)
    feats["acronym_match"] = _acronym_match_arr(
        joined["a_acronym"].to_numpy(), joined["b_acronym"].to_numpy(), ac, bc
    )
    feats["same_country"] = (
        joined["a_country_raw"].fillna("").astype(str).str.strip()
        == joined["b_country_raw"].fillna("").astype(str).str.strip()
    ).astype(np.int8)
    src = joined["cand_id"].astype(str).map(source_flag)
    feats["source_s2"] = (src == "S2").astype(np.int8)
    feats["source_s3"] = (src == "S3").astype(np.int8)

    hn_m, hn_x, hn_miss = _match_miss_arr(joined["a_house_number"].to_numpy(), joined["b_house_number"].to_numpy())
    pc_m, pc_x, pc_miss = _match_miss_arr(joined["a_postcode"].to_numpy(), joined["b_postcode"].to_numpy())
    un_m, un_x, un_miss = _match_miss_arr(joined["a_unit"].to_numpy(), joined["b_unit"].to_numpy())
    feats["house_num_match"] = hn_m
    feats["house_num_mismatch"] = hn_x
    feats["house_num_missing"] = hn_miss
    feats["postcode_match"] = pc_m
    feats["postcode_mismatch"] = pc_x
    feats["postcode_missing"] = pc_miss
    feats["unit_match"] = un_m
    feats["unit_mismatch"] = un_x
    feats["unit_missing"] = un_miss
    feats["direction_conflict"] = _direction_conflict_arr(
        joined["a_directions"].to_numpy(), joined["b_directions"].to_numpy()
    )
    feats["landmark_overlap"] = _overlap_arr(
        joined["a_landmarks"].to_numpy(), joined["b_landmarks"].to_numpy()
    )
    feats["digit_jaccard"] = _overlap_arr(
        joined["a_addr_digits"].to_numpy(), joined["b_addr_digits"].to_numpy()
    )
    feats["addr_token_jaccard"] = _token_overlap_arr(
        joined["a_address_exp"].to_numpy(), joined["b_address_exp"].to_numpy()
    )
    feats["name_token_jaccard"] = _token_overlap_arr(
        joined["a_name_exp"].to_numpy(), joined["b_name_exp"].to_numpy()
    )
    feats["core_token_jaccard"] = _token_overlap_arr(
        joined["a_core_name"].to_numpy(), joined["b_core_name"].to_numpy()
    )
    feats["s1_missing_addr"] = (joined["a_missing_addr"].astype(str) == "1").astype(np.int8)
    feats["cand_missing_addr"] = (joined["b_missing_addr"].astype(str) == "1").astype(np.int8)
    feats["s1_missing_postcode"] = (joined["a_missing_postcode"].astype(str) == "1").astype(np.int8)
    feats["cand_missing_postcode"] = (joined["b_missing_postcode"].astype(str) == "1").astype(np.int8)
    feats["s1_missing_house"] = (joined["a_missing_house"].astype(str) == "1").astype(np.int8)
    feats["cand_missing_house"] = (joined["b_missing_house"].astype(str) == "1").astype(np.int8)
    if idf is not None:
        feats["name_idf_jaccard"] = _idf_jaccard_arr(
            joined["a_name_exp"].to_numpy(), joined["b_name_exp"].to_numpy(), idf
        )
        feats["core_idf_jaccard"] = _idf_jaccard_arr(
            joined["a_core_name"].to_numpy(), joined["b_core_name"].to_numpy(), idf
        )
        feats["addr_idf_jaccard"] = _idf_jaccard_arr(
            joined["a_address_exp"].to_numpy(), joined["b_address_exp"].to_numpy(), idf
        )
    else:
        feats["name_idf_jaccard"] = feats["name_token_jaccard"]
        feats["core_idf_jaccard"] = feats["core_token_jaccard"]
        feats["addr_idf_jaccard"] = feats["addr_token_jaccard"]
    return feats


def build_pair_features(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    idf: dict[str, float] | None,
    chunk: int = 100000,
    checkpoint_dir: Path | None = None,
    name: str = "pairs",
    workers: int = -1,
) -> pd.DataFrame:
    """pairs.join(s1).join(gallery) + RapidFuzz cpdist. Chunks are resume-safe."""
    n = len(pairs)
    chunk = max(int(chunk), 1)
    LOG.info(
        "pair features  n=%s  chunk=%s  name=%s  rf_workers=%s  (join+cpdist, no row loc)",
        f"{n:,}", f"{chunk:,}", name, workers,
    )
    chunk_dir = None
    done: set[int] = set()
    if checkpoint_dir is not None:
        chunk_dir = Path(checkpoint_dir) / "features" / name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        man_p = chunk_dir / "manifest.json"
        if man_p.exists():
            try:
                man = json.loads(man_p.read_text(encoding="utf-8"))
                done = {int(x) for x in man.get("completed_starts", [])}
                LOG.info("feature-chunk resume  %s/%s chunks already on disk", f"{len(done):,}", f"{(n + chunk - 1) // chunk:,}")
            except Exception:
                done = set()

    t0 = time.perf_counter()
    starts = list(range(0, n, chunk))
    mem_chunks: list[pd.DataFrame] = []
    for i, start in enumerate(tqdm(starts, desc=f"pair features [{name}]", leave=True)):
        end = min(start + chunk, n)
        out_p = chunk_dir / f"chunk_{start:09d}.parquet" if chunk_dir else None
        if start in done and out_p is not None and out_p.exists():
            LOG.debug("  skip cached chunk %s–%s", f"{start:,}", f"{end:,}")
        else:
            sl = pairs.iloc[start:end]
            extra = _features_joined(sl, s1, gallery, idf, workers=workers)
            merged = sl.merge(extra, on=["s1_id", "cand_id"], how="left")
            if out_p is not None:
                merged.to_parquet(out_p, index=False)
                done.add(start)
                (chunk_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "n_pairs": n,
                            "chunk": chunk,
                            "name": name,
                            "completed_starts": sorted(done),
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                mem_chunks.append(merged)
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == len(starts):
            elapsed = time.perf_counter() - t0
            done_rows = min((i + 1) * chunk, n)
            rate = done_rows / max(elapsed, 1e-6)
            eta = (n - done_rows) / max(rate, 1e-6)
            LOG.info(
                "  features %s  chunk %s/%s  rows %s–%s  rate=%.0f pairs/s  elapsed=%s  eta=%s",
                name, i + 1, len(starts), f"{start:,}", f"{end:,}",
                rate, fmt_elapsed(elapsed), fmt_elapsed(eta),
            )

    if chunk_dir is not None:
        files = sorted(chunk_dir.glob("chunk_*.parquet"))
        LOG.info("concat %s feature chunks → table  (%s files)", name, len(files))
        parts = [pd.read_parquet(f) for f in files]
        out = pd.concat(parts, ignore_index=True) if parts else pairs.copy()
    else:
        out = pd.concat(mem_chunks, ignore_index=True) if mem_chunks else pairs.copy()
    LOG.info("pair features done  %s  rows=%s  elapsed=%s", name, f"{len(out):,}", fmt_elapsed(time.perf_counter() - t0))
    return out


def label_pairs(pairs: pd.DataFrame, truth_map: dict[str, set[str]]) -> pd.DataFrame:
    df = pairs.copy()
    df["label"] = [
        int(c in truth_map.get(s1, set())) for s1, c in zip(df["s1_id"], df["cand_id"])
    ]
    return df


FEATURE_EXCLUDE = {
    "s1_id",
    "cand_id",
    "label",
    "p_gbm",
    "p_ce",
    "p_stack",
    "p",
    "fold",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in FEATURE_EXCLUDE:
            continue
        if df[c].dtype == object:
            continue
        cols.append(c)
    return cols


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / (1024 * 1024)
    LOG.info("wrote %s  rows=%s  cols=%s  %.1f MB", path, f"{len(df):,}", len(df.columns), mb)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
