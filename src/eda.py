"""Stage 0: EDA. Writes reports/eda.md. Never looks at test labels (there are none)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.io_utils import load_ground_truth, load_sources
from src.logging_utils import LOG, write_json


def _counts_by(df: pd.DataFrame, col: str) -> dict:
    return df[col].astype(str).value_counts().to_dict()


def _source_summary(name: str, df: pd.DataFrame) -> dict:
    return {
        "name": name,
        "n": int(len(df)),
        "n_countries": int(df["country"].nunique()),
        "by_country": _counts_by(df, "country"),
        "empty_name": int((df["business_name"].astype(str).str.strip() == "").sum()),
        "empty_address": int((df["business_address"].astype(str).str.strip() == "").sum()),
        "empty_country": int((df["country"].astype(str).str.strip() == "").sum()),
    }


def run_eda(cfg) -> dict:
    reports = Path(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    LOG.info("EDA  reading train + test TSVs")

    s1, s2, s3 = load_sources(cfg.paths.train_dir, "train")
    t1, t2, t3 = load_sources(cfg.paths.test_dir, "test")
    gt = load_ground_truth(Path(cfg.paths.train_dir) / "train_ground_truth.tsv")

    train_sum = [
        _source_summary("train_source1", s1),
        _source_summary("train_source2", s2),
        _source_summary("train_source3", s3),
    ]
    test_sum = [
        _source_summary("test_source1", t1),
        _source_summary("test_source2", t2),
        _source_summary("test_source3", t3),
    ]

    # singleton / matches-per-S1
    match_dist = gt["n_matches"].value_counts().sort_index().to_dict()
    n_s1 = len(gt)
    n_single = int(gt["is_singleton"].sum())
    singleton_rate = n_single / n_s1 if n_s1 else 0.0

    # Does any S2/S3 match more than one S1?
    rev = defaultdict(set)
    for row in gt.itertuples(index=False):
        for mid in row.matched_list:
            rev[mid].add(row.source1_entity_id)
    multi_s1 = {k: sorted(v) for k, v in rev.items() if len(v) > 1}
    one_to_one = len(multi_s1) == 0

    # Do matched pairs always share a country?
    id_country = {}
    for df in (s1, s2, s3):
        id_country.update(zip(df["entity_id"], df["country"].astype(str)))
    n_cross = 0
    n_same = 0
    n_missing = 0
    for row in gt.itertuples(index=False):
        c1 = id_country.get(row.source1_entity_id)
        for mid in row.matched_list:
            c2 = id_country.get(mid)
            if c1 is None or c2 is None:
                n_missing += 1
            elif c1 == c2:
                n_same += 1
            else:
                n_cross += 1
    always_same_country = n_cross == 0

    # singleton rate by country (train)
    s1_gt = s1.merge(
        gt[["source1_entity_id", "is_singleton", "n_matches"]],
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
    )
    by_country_single = (
        s1_gt.groupby(s1_gt["country"].astype(str))["is_singleton"]
        .mean()
        .to_dict()
    )

    # 30 matched pairs and 30 near-miss non-pairs (same country, similar name length)
    examples_pos = []
    s2_idx = s2.set_index("entity_id", drop=False)
    s3_idx = s3.set_index("entity_id", drop=False)
    s1_idx = s1.set_index("entity_id", drop=False)
    linked = gt[~gt["is_singleton"]].head(40)
    for row in linked.itertuples(index=False):
        if len(examples_pos) >= 30:
            break
        if row.source1_entity_id not in s1_idx.index:
            continue
        a = s1_idx.loc[row.source1_entity_id]
        mid = row.matched_list[0]
        b = s2_idx.loc[mid] if mid in s2_idx.index else s3_idx.loc[mid] if mid in s3_idx.index else None
        if b is None:
            continue
        examples_pos.append(
            {
                "s1": row.source1_entity_id,
                "s1_name": a.business_name,
                "s1_addr": a.business_address,
                "s1_country": a.country,
                "match": mid,
                "m_name": b.business_name,
                "m_addr": b.business_address,
                "m_country": b.country,
            }
        )

    # near-miss: S2 records in the same country whose name shares a token but are not GT
    examples_neg = []
    gt_map = {r.source1_entity_id: set(r.matched_list) for r in gt.itertuples(index=False)}
    rng_s1 = s1.sample(n=min(2000, len(s1)), random_state=int(cfg.seed))
    s2_by_c = {c: g for c, g in s2.groupby(s2["country"].astype(str))}
    for row in rng_s1.itertuples(index=False):
        if len(examples_neg) >= 30:
            break
        toks = set(str(row.business_name).lower().split())
        if len(toks) < 2:
            continue
        pool = s2_by_c.get(str(row.country))
        if pool is None or pool.empty:
            continue
        sample = pool.sample(n=min(40, len(pool)), random_state=abs(hash(row.entity_id)) % 10_000)
        truth = gt_map.get(row.entity_id, set())
        for cand in sample.itertuples(index=False):
            if cand.entity_id in truth:
                continue
            ct = set(str(cand.business_name).lower().split())
            if toks & ct:
                examples_neg.append(
                    {
                        "s1": row.entity_id,
                        "s1_name": row.business_name,
                        "s1_addr": row.business_address,
                        "neg": cand.entity_id,
                        "neg_name": cand.business_name,
                        "neg_addr": cand.business_address,
                        "country": row.country,
                    }
                )
                break

    # raw French test records (inputs only — no labels)
    fr_mask = t1["country"].astype(str).str.lower() == "france"
    n_fr_s1 = int(fr_mask.sum())
    n_fr_s2 = int((t2["country"].astype(str).str.lower() == "france").sum())
    n_fr_s3 = int((t3["country"].astype(str).str.lower() == "france").sum())
    fr_samples = t1.loc[fr_mask, ["entity_id", "business_name", "business_address", "country"]].head(25)
    train_countries = sorted(set(s1["country"].astype(str)) | set(s2["country"].astype(str)) | set(s3["country"].astype(str)))
    test_countries = sorted(set(t1["country"].astype(str)) | set(t2["country"].astype(str)) | set(t3["country"].astype(str)))
    unseen_test_countries = sorted(set(test_countries) - set(train_countries))

    metrics = {
        "train": train_sum,
        "test": test_sum,
        "n_s1_train": n_s1,
        "singleton_rate": singleton_rate,
        "n_singletons": n_single,
        "match_dist": {str(k): int(v) for k, v in match_dist.items()},
        "one_to_one": one_to_one,
        "n_s2s3_linked_to_multiple_s1": len(multi_s1),
        "always_same_country": always_same_country,
        "n_same_country_pairs": n_same,
        "n_cross_country_pairs": n_cross,
        "n_missing_country_pairs": n_missing,
        "singleton_rate_by_country": by_country_single,
        "n_france_test_s1": n_fr_s1,
        "n_france_test_s2": n_fr_s2,
        "n_france_test_s3": n_fr_s3,
        "train_countries": train_countries,
        "test_countries": test_countries,
        "unseen_test_countries": unseen_test_countries,
    }
    write_json(reports / "eda.json", metrics)

    md = _render_eda_md(
        metrics, examples_pos, examples_neg, fr_samples, multi_s1
    )
    (reports / "eda.md").write_text(md, encoding="utf-8")
    return metrics


def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(none)_\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/")[:80] for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _render_eda_md(metrics, examples_pos, examples_neg, fr_samples, multi_s1) -> str:
    lines = [
        "# Stage 0 — EDA",
        "",
        "Generated by `python -m src.run --stage eda`. Country is treated as an open set.",
        "",
        "## Counts",
        "",
        "### Train",
    ]
    for s in metrics["train"]:
        lines.append(f"- **{s['name']}**: {s['n']:,} rows, countries={s['by_country']}, "
                     f"empty name={s['empty_name']}, empty address={s['empty_address']}")
    lines += ["", "### Test", ""]
    for s in metrics["test"]:
        lines.append(f"- **{s['name']}**: {s['n']:,} rows, countries={s['by_country']}, "
                     f"empty name={s['empty_name']}, empty address={s['empty_address']}")
    lines += [
        "",
        f"- Train countries: `{metrics['train_countries']}`",
        f"- Test countries: `{metrics['test_countries']}`",
        f"- Unseen in test (not in train): `{metrics['unseen_test_countries']}`",
        f"- France in test: S1={metrics['n_france_test_s1']:,}, "
        f"S2={metrics['n_france_test_s2']:,}, S3={metrics['n_france_test_s3']:,}",
        "",
        "## Singletons and matches-per-S1",
        "",
        f"- Train S1 entities: {metrics['n_s1_train']:,}",
        f"- Singleton rate: **{metrics['singleton_rate']:.4f}** "
        f"({metrics['n_singletons']:,} / {metrics['n_s1_train']:,})",
        f"- Singleton rate by country: `{metrics['singleton_rate_by_country']}`",
        f"- Matches-per-S1 distribution: `{metrics['match_dist']}`",
        "",
        "## One-to-one constraint",
        "",
        f"- S2/S3 records linked to more than one S1: **{metrics['n_s2s3_linked_to_multiple_s1']}**",
        f"- Hard one-to-one is valid: **{metrics['one_to_one']}**",
        "",
    ]
    if multi_s1:
        lines.append(f"Examples of multi-S1 gallery IDs (first 10): `{list(multi_s1.items())[:10]}`")
        lines.append("")
    lines += [
        "## Same-country matches",
        "",
        f"- Same-country matched pairs: {metrics['n_same_country_pairs']:,}",
        f"- Cross-country matched pairs: {metrics['n_cross_country_pairs']:,}",
        f"- Missing-country matched pairs: {metrics['n_missing_country_pairs']:,}",
        f"- Block within country: **{metrics['always_same_country']}**",
        "",
        "## 30 matched pairs (raw)",
        "",
        _md_table(examples_pos, ["s1", "s1_name", "s1_addr", "match", "m_name", "m_addr", "s1_country"]),
        "",
        "## 30 near-miss non-pairs (same country, shared name token, not in GT)",
        "",
        _md_table(examples_neg, ["s1", "s1_name", "s1_addr", "neg", "neg_name", "neg_addr", "country"]),
        "",
        "## Raw French test S1 records (first 25, inputs only)",
        "",
        _md_table(fr_samples.to_dict("records"), ["entity_id", "business_name", "business_address", "country"]),
        "",
        "## Implications for the pipeline",
        "",
        "- Treat `country` as an open string; never one-hot to {US, India}.",
        "- Every test S1 (France included) must appear in both output TSVs.",
        "- If one-to-one holds, apply it as a hard constraint before the decision rule.",
        "- If matches always share a country, block within country (France still works).",
        "- Generic names and near-duplicate house numbers need exact number / token checks.",
        "",
    ]
    return "\n".join(lines)
