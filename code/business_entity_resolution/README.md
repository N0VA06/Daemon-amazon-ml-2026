# Business Entity Resolution

Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the challenge TSVs. Run every command from this directory
(`student_resource/`: it contains `dataset/`, `src/`, `configs/`, `utils/`).

Do **not** commit or copy `dataset/`. Keep the shipped layout:

```
dataset/train/train_source{1,2,3}.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## What this solves

Source 1 is the deduplicated reference. For every S1 entity, list the S2/S3
records that refer to the same business (zero, one, or many). Country is an
**open string** (test has France; train does not). Scoring is macro-F0.5 over
every S1, singletons included.

Outputs match the problem statement:

| file | role |
| --- | --- |
| `output/matching_results.tsv` | leaderboard file: one row per test S1, empty list = singleton |
| `output/candidate_pairs.tsv` | exact candidate set the matcher scored (matches ⊆ candidates) |

Read TSVs with `sep="\t", dtype=str, keep_default_na=False`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install flash-attn
```

Needs `transformers>=5.1.0`, `torch>=2.8.0`, `peft>=0.15.2`. A GPU is strongly
recommended; batch sizes are set from VRAM in `src/hardware.py`.

Metric self-test (PDF example must be 0.714):

```bash
python -m src.evaluate
python -m pytest src/tests/test_evaluate.py -q
```

Licence tags (writes `reports/licences.md`):

```bash
python3 check_licences.py --config configs/default.yaml
```

## Run

Default backbone is `jinaai/jina-embeddings-v5-text-small-text-matching`
(prefix `Document: ` on both sides). Swap to Apache-2.0 Snowflake Arctic
(`query: `) with the other config — same code.

```bash
python -m src.run --stage all --config configs/default.yaml
python -m src.run --stage all --config configs/snowflake.yaml
```

Stage by stage (each appends to `experiments.md`):

```bash
python -m src.run --stage eda                 --config configs/default.yaml
python -m src.run --stage split               --config configs/default.yaml
python -m src.run --stage normalize           --config configs/default.yaml
python -m src.run --stage block               --config configs/default.yaml
python -m src.run --stage train_biencoder     --config configs/default.yaml
python -m src.run --stage features            --config configs/default.yaml
python -m src.run --stage train_matcher       --config configs/default.yaml
python -m src.run --stage train_crossencoder  --config configs/default.yaml
python -m src.run --stage stack               --config configs/default.yaml
python -m src.run --stage decide              --config configs/default.yaml
python -m src.run --stage evaluate            --config configs/default.yaml
python -m src.run --stage predict             --config configs/default.yaml
```

Dry run: set `debug.max_s1: 5000` and `debug.fast: true` in the YAML.

## Validate and zip

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

bash make_zip.sh team_jina_er
```

`--stage predict` already runs the validator. The zip layout is the one in
the problem statement: `output/`, `code/business_entity_resolution/`,
`Documentation_template.md`.

## Pipeline

```
TSVs → normalize (3 text views + parsed numbers/postcodes/legal form)
     → block (jina FAISS ∪ char TF-IDF ∪ rare-token / postcode+house)
     → LightGBM → optional jina cross-encoder (top-N) → stacker
     → one-to-one on S2/S3 → expected-F0.5 prefix → matching_results.tsv
```

`candidate_pairs.tsv` is rewritten from the scored pair table, so it is the
set the model actually ran on.

| config | meaning |
| --- | --- |
| `backbone.preset` / `repo` / `prefix` | only backbone swap |
| `blocking.k_cap` | union cap per S1 (default 80) |
| `blocking.same_country_only` | group by country *string* (France is just another label) |
| `decision.rule` | `global` / `relative` / `expected_f05` |

No country one-hot to `{US, India}`. No external registries or geocoders.

## Layout

```
src/        pipeline (`python -m src.run` is the entry point)
configs/    default.yaml (jina), snowflake.yaml
dataset/    local data, not committed
cache/      normalised parquet, embeddings, pair tables
models/     LoRA, merged encoder, LightGBM, cross-encoder, stacker
reports/    eda, blocking, decision, licences
output/     the two submission TSVs
```

After a backbone swap, delete `cache/embeddings/`.
