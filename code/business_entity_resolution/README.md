# Business Entity Resolution

Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the challenge TSVs. Run every command from `student_resource/`
(the folder with `dataset/`, `src/`, `configs/`, `utils/`).

Keep `dataset/` on disk in the shipped layout. Do not commit the TSVs.

```
dataset/train/train_source{1,2,3}.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Task (problem statement)

Source 1 is the deduplicated reference. For every S1 entity, list matching
S2/S3 records (zero, one, or many). Country is an **open string** — test
includes France, which is not in train. Score is **macro F0.5** over every
S1, singletons included (empty pred on a singleton = 1.0).

| file | columns | rule |
| --- | --- | --- |
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | one row per test S1; empty list = singleton; S2/S3 IDs only; no duplicates |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | exact set the matcher scored; matches ⊆ candidates |

TSVs are read with `sep="\t", dtype=str, keep_default_na=False`. No external
registries, geocoders, or scraped lists.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Needs `transformers>=5.1.0`, `torch>=2.8.0`, `peft>=0.15.2`. GPU recommended;
batch sizes follow VRAM (`src/hardware.py`). L4 24 GB → encode 256, InfoNCE 1024.

F0.5 self-test (must print 0.714):

```bash
python -m src.evaluate
```

## Run

Logs go to stdout (`HH:MM:SS | INFO | …`). Caches under `cache/` are reused.

```bash
python -m src.run --stage all --config configs/default.yaml
```

Snowflake backbone (same code, prefix `query: `):

```bash
python -m src.run --stage all --config configs/snowflake.yaml
```

One stage:

```bash
python -m src.run --stage {eda,split,normalize,block,train_biencoder,features,train_matcher,train_crossencoder,stack,decide,evaluate,predict} \
    --config configs/default.yaml
```

Dry run: `debug.max_s1: 5000` and `debug.fast: true` in the YAML.

## Validate and zip

`--stage predict` already runs the problem-statement validator. To run it
yourself (PDF command; `--check-ids` is optional and heavy):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

bash make_zip.sh team_jina_er
```

Zip layout: `output/` (both TSVs), `code/business_entity_resolution/`,
`Documentation_template.md`.

## Pipeline

```
TSVs → normalize → block (FAISS ∪ TF-IDF ∪ keys)
     → LightGBM → cross-encoder (top-N) → stacker
     → one-to-one → expected-F0.5 → matching_results.tsv
```

`candidate_pairs.tsv` is rewritten from the scored pair table.

| config | meaning |
| --- | --- |
| `backbone.preset` / `repo` / `prefix` | backbone swap |
| `blocking.k_cap` | union cap per S1 |
| `blocking.same_country_only` | group by country string |
| `decision.rule` | `global` / `relative` / `expected_f05` |
