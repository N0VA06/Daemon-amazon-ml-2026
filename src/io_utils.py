"""TSV I/O. Always tab-separated, dtype=str, keep_default_na=False."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

SOURCE_FILES = {
    "train": ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"),
    "test": ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"),
}


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")


def parse_id_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in text.split(","):
        eid = part.strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def join_id_list(ids: Iterable[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for eid in ids:
        eid = str(eid).strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return ",".join(out)


def load_sources(data_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    names = SOURCE_FILES[split]
    frames = [read_tsv(data_dir / name) for name in names]
    for df in frames:
        for col in ("entity_id", "business_name", "business_address", "country"):
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {split} source file")
            df[col] = df[col].fillna("").astype(str)
        df["source"] = df["entity_id"].str.slice(0, 2)
    return frames[0], frames[1], frames[2]


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    gt = read_tsv(path)
    if "source1_entity_id" not in gt.columns or "matched_entity_ids" not in gt.columns:
        raise ValueError("Ground truth must have source1_entity_id, matched_entity_ids")
    gt["matched_list"] = gt["matched_entity_ids"].map(parse_id_list)
    gt["n_matches"] = gt["matched_list"].map(len)
    gt["is_singleton"] = gt["n_matches"] == 0
    return gt


def gt_to_map(gt: pd.DataFrame) -> dict[str, set[str]]:
    return {
        row.source1_entity_id: set(row.matched_list)
        for row in gt.itertuples(index=False)
    }


def write_id_list_tsv(
    s1_ids: Iterable[str],
    mapping: Mapping[str, Iterable[str]],
    path: str | Path,
    id_col: str,
    list_col: str,
) -> None:
    rows = []
    for s1 in s1_ids:
        rows.append({id_col: s1, list_col: join_id_list(mapping.get(s1, []))})
    write_tsv(pd.DataFrame(rows, columns=[id_col, list_col]), path)


def write_matching_results(
    s1_ids: Iterable[str],
    mapping: Mapping[str, Iterable[str]],
    path: str | Path,
) -> None:
    write_id_list_tsv(
        s1_ids, mapping, path, "source1_entity_id", "matched_entity_ids"
    )


def write_candidate_pairs(
    s1_ids: Iterable[str],
    mapping: Mapping[str, Iterable[str]],
    path: str | Path,
) -> None:
    write_id_list_tsv(
        s1_ids, mapping, path, "source1_entity_id", "candidate_entity_ids"
    )


def read_id_list_tsv(path: str | Path, list_col: str) -> dict[str, list[str]]:
    df = read_tsv(path)
    return {
        row.source1_entity_id: parse_id_list(getattr(row, list_col))
        for row in df.itertuples(index=False)
    }


def source_flag(entity_id: str) -> str:
    if entity_id.startswith("S2-"):
        return "S2"
    if entity_id.startswith("S3-"):
        return "S3"
    if entity_id.startswith("S1-"):
        return "S1"
    return "UNK"
