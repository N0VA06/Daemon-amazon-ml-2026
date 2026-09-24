"""Pair features for LightGBM. No country categorical — only same_country."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.io_utils import source_flag
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
    for view, mat_s1 in emb_s1.items():
        mat_g = emb_gal.get(view)
        col = f"cosine_{prefix}_{view}" if prefix else f"cosine_{view}"
        vals = [
            _cosine_from_index(s1, c, s1_index, gal_index, mat_s1, mat_g)
            for s1, c in zip(out["s1_id"], out["cand_id"])
        ]
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


def build_pair_features(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    idf: dict[str, float] | None,
    chunk: int = 100000,
) -> pd.DataFrame:
    s1_map = s1.set_index("entity_id", drop=False)
    g_map = gallery.set_index("entity_id", drop=False)
    chunks = []
    n = len(pairs)
    for start in range(0, n, chunk):
        sl = pairs.iloc[start : start + chunk]
        recs = []
        for row in sl.itertuples(index=False):
            a = s1_map.loc[row.s1_id]
            b = g_map.loc[row.cand_id]
            feats = pair_string_features(a, b, idf)
            feats["s1_id"] = row.s1_id
            feats["cand_id"] = row.cand_id
            recs.append(feats)
        extra = pd.DataFrame(recs)
        merged = sl.merge(extra, on=["s1_id", "cand_id"], how="left")
        chunks.append(merged)
    return pd.concat(chunks, ignore_index=True) if chunks else pairs.copy()


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


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
