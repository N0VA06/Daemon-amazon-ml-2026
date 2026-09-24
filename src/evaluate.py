"""Exact PDF F0.5 metric. Macro-average over every S1 entity, singletons included.

F_0.5 = (1.25 * P * R) / (0.25 * P + R)
      = 1.25 * TP / (0.25 * |truth| + |pred|)

Singleton: score 1.0 if pred is empty, else 0.0.
Unit test (PDF example): pred {A,B,C}, truth {A,C} → 0.714.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from src.io_utils import parse_id_list, read_tsv


def f05_from_counts(tp: int, n_pred: int, n_truth: int) -> float:
    if n_truth == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    return (1.25 * tp) / (0.25 * n_truth + n_pred)


def f05_entity(pred: Iterable[str], truth: Iterable[str]) -> float:
    pred_set = {p for p in pred if p}
    truth_set = {t for t in truth if t}
    tp = len(pred_set & truth_set)
    return f05_from_counts(tp, len(pred_set), len(truth_set))


def macro_f05(
    pred_map: Mapping[str, Iterable[str]],
    truth_map: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
) -> dict:
    if s1_ids is None:
        s1_ids = list(truth_map.keys())
    s1_ids = list(s1_ids)
    scores: list[float] = []
    n_tp = n_fp = n_fn = 0
    n_singleton_ok = n_singleton_bad = 0
    n_empty_pred = 0
    for s1 in s1_ids:
        pred = set(pred_map.get(s1, []))
        truth = set(truth_map.get(s1, []))
        score = f05_entity(pred, truth)
        scores.append(score)
        tp = len(pred & truth)
        n_tp += tp
        n_fp += len(pred - truth)
        n_fn += len(truth - pred)
        if not pred:
            n_empty_pred += 1
        if not truth:
            if pred:
                n_singleton_bad += 1
            else:
                n_singleton_ok += 1
    n = len(scores)
    return {
        "macro_f05": (sum(scores) / n) if n else 0.0,
        "n_entities": n,
        "micro_tp": n_tp,
        "micro_fp": n_fp,
        "micro_fn": n_fn,
        "n_empty_pred": n_empty_pred,
        "n_singleton_correct": n_singleton_ok,
        "n_singleton_false_merge": n_singleton_bad,
        "mean_pred_size": (
            sum(len(list(pred_map.get(s, []))) for s in s1_ids) / n if n else 0.0
        ),
        "mean_truth_size": (
            sum(len(list(truth_map.get(s, []))) for s in s1_ids) / n if n else 0.0
        ),
    }


def blocking_recall(
    cand_map: Mapping[str, Iterable[str]],
    truth_map: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
) -> dict:
    if s1_ids is None:
        s1_ids = list(truth_map.keys())
    s1_ids = list(s1_ids)
    n_pos = n_hit = 0
    n_pairs = 0
    entity_recalls: list[float] = []
    for s1 in s1_ids:
        truth = set(truth_map.get(s1, []))
        cand = set(cand_map.get(s1, []))
        n_pairs += len(cand)
        if not truth:
            entity_recalls.append(1.0)
            continue
        hit = len(truth & cand)
        n_pos += len(truth)
        n_hit += hit
        entity_recalls.append(hit / len(truth))
    n_s1 = max(len(s1_ids), 1)
    n_gallery_proxy = max(n_pairs, 1)
    return {
        "pair_recall": (n_hit / n_pos) if n_pos else 1.0,
        "entity_recall": sum(entity_recalls) / len(entity_recalls) if entity_recalls else 1.0,
        "n_candidate_pairs": n_pairs,
        "avg_candidates": n_pairs / n_s1,
        "reduction_vs_naive": None,
        "n_positive_pairs": n_pos,
        "n_recovered": n_hit,
        "n_gallery_proxy": n_gallery_proxy,
    }


def recall_at_k_curve(
    ranked_map: Mapping[str, list[str]],
    truth_map: Mapping[str, Iterable[str]],
    ks: Iterable[int],
) -> dict[int, float]:
    """ranked_map values must be best-first. Pair recall@K over non-singleton S1s."""
    ks = list(ks)
    hits = {k: 0 for k in ks}
    n_pos = 0
    # Only S1s that were actually queried (do not mix in train entities).
    for s1 in ranked_map:
        truth_set = {t for t in truth_map.get(s1, ()) if t}
        if not truth_set:
            continue
        n_pos += len(truth_set)
        ranked = ranked_map.get(s1, [])
        for k in ks:
            hits[k] += len(truth_set & set(ranked[:k]))
    if n_pos == 0:
        return {k: 1.0 for k in ks}
    return {k: hits[k] / n_pos for k in ks}


def evaluate_prediction_file(
    matching_path: str | Path,
    truth_map: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
) -> dict:
    df = read_tsv(matching_path)
    pred_map = {
        row.source1_entity_id: parse_id_list(row.matched_entity_ids)
        for row in df.itertuples(index=False)
    }
    return macro_f05(pred_map, truth_map, s1_ids=s1_ids)


def per_country_f05(
    pred_map: Mapping[str, Iterable[str]],
    truth_map: Mapping[str, Iterable[str]],
    s1_country: Mapping[str, str],
) -> dict[str, dict]:
    by_c: dict[str, list[str]] = defaultdict(list)
    for s1, country in s1_country.items():
        by_c[str(country)].append(s1)
    return {c: macro_f05(pred_map, truth_map, ids) for c, ids in sorted(by_c.items())}


def _self_test() -> None:
    """PDF unit test: pred {A,B,C}, truth {A,C} must give 0.714."""
    score = f05_entity({"A", "B", "C"}, {"A", "C"})
    expected = 1.25 * 2 / (0.25 * 2 + 3)
    assert abs(score - expected) < 1e-12, (score, expected)
    assert abs(score - 0.714) < 1e-3, score
    assert f05_entity([], []) == 1.0
    assert f05_entity(["X"], []) == 0.0
    assert f05_entity([], ["X"]) == 0.0


if __name__ == "__main__":
    _self_test()
    print("evaluate.py self-test PASS  (pred {A,B,C} vs truth {A,C} → 0.714)")
