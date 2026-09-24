# Model licences

Printed by `python check_licences.py` (re-run that script after install to
replace the expected tags below with the live Hugging Face card values).

The PDF asks for MIT / Apache-2.0. The user-chosen primary backbone
`jinaai/jina-embeddings-v5-text-small-text-matching` is **CC-BY-NC-4.0**.
The backbone is a single config key so
`Snowflake/snowflake-arctic-embed-l-v2.0` (Apache-2.0, prefix `query: `)
is a tested drop-in: `python -m src.run --stage all --config configs/snowflake.yaml`.

| repo | expected HF licence tag | notes |
| --- | --- | --- |
| `jinaai/jina-embeddings-v5-text-small-text-matching` | `cc-by-nc-4.0` | Primary. Text-matching LoRA merged into Qwen3-0.6B. Prefix `Document: ` on both sides. |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | `apache-2.0` | Drop-in. Prefix `query: ` on both sides. Same encode / LoRA / CE code path. |

LightGBM, TF-IDF, RapidFuzz and the logistic/isotonic stacker are not HF models.
No external business-registry, geocoder, or scraped gazetteer is used.
