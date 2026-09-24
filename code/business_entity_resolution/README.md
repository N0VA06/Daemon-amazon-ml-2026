# Business Entity Resolution

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from
the challenge train/test TSVs. Run every command from `student_resource/`
(the directory that contains `dataset/`, `utils/`, `src/`, `configs/`).

Backbone is a **single config key**. Default is jina text-matching
(`Document: ` on both sides). Swap to Snowflake Arctic (Apache-2.0,
`query: `) with `configs/snowflake.yaml` — the rest of the code is unchanged.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install flash-attn
```

Floors required by jina-embeddings-v5-text-small: `transformers>=5.1.0`,
`torch>=2.8.0`, `peft>=0.15.2`. GPU is strongly recommended. Batch sizes
are set from detected VRAM in `src/hardware.py`.

Self-test the PDF metric (must print `0.714`):

```bash
python -m src.evaluate
python -m pytest src/tests/test_evaluate.py -q
```

Print Hugging Face licence tags and write `reports/licences.md`:

```bash
python3 check_licences.py --config configs/default.yaml
```

## End-to-end

```bash
# full pipeline (EDA → split → normalize → block → train → decide → test)
python -m src.run --stage all --config configs/default.yaml

# Apache-2.0 drop-in backbone (same flags, same stages)
python -m src.run --stage all --config configs/snowflake.yaml
```

Or stage-by-stage (do not skip; each stage logs to `experiments.md`):

```bash
python -m src.run --stage eda              --config configs/default.yaml
python -m src.run --stage split            --config configs/default.yaml
python -m src.run --stage normalize        --config configs/default.yaml
python -m src.run --stage block            --config configs/default.yaml
python -m src.run --stage train_biencoder  --config configs/default.yaml
python -m src.run --stage features         --config configs/default.yaml
python -m src.run --stage train_matcher    --config configs/default.yaml
python -m src.run --stage train_crossencoder --config configs/default.yaml
python -m src.run --stage stack            --config configs/default.yaml
python -m src.run --stage decide           --config configs/default.yaml
python -m src.run --stage evaluate         --config configs/default.yaml
python -m src.run --stage predict          --config configs/default.yaml
```

Dry-run on a slice (no behaviour change, just fewer S1 rows):

```bash
# in configs/default.yaml set debug.max_s1: 5000 and debug.fast: true
python -m src.run --stage all --config configs/default.yaml
```

## Validate and zip

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

bash make_zip.sh team_jina_er
# → team_jina_er_submission.zip
```

`predict` already runs the validator. `candidate_pairs.tsv` is the exact set
the final model scored (matches ⊆ candidates).

## Layout

```
src/           pipeline (run.py is the only entry point)
configs/       default.yaml (jina) and snowflake.yaml
cache/         normalised tables, .npy embeddings, FAISS, pair parquet
models/        LoRA adapters, merged ST model, LightGBM, CE, stacker
reports/       eda.md, licences.md, blocking_*.json, decision.json, …
output/        matching_results.tsv, candidate_pairs.tsv
```

Caches are keyed by split / source / view / model tag (`zs` or `ft`). Delete
`cache/embeddings/` after a backbone swap.

## Config keys that matter

| key | meaning |
| --- | --- |
| `backbone.preset` / `repo` / `prefix` | the only backbone swap |
| `blocking.k_cap` | union cap per S1 (default 80) |
| `blocking.same_country_only` | open-set country string equality |
| `biencoder.phase_b_aux_loss` | `cosent` or `online_contrastive` |
| `biencoder.full_finetune` | LoRA vs full FT ablation |
| `decision.rule` | `global` / `relative` / `expected_f05` |
