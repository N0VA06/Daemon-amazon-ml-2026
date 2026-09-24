# Experiments log

Append-only. Every `python -m src.run --stage …` call appends a timestamped
block. This preamble is the **design log** for Stages 0–7 (written with the
code; numeric cells are filled by subsequent stage runs into `reports/*.json`).

---

## Stage 0 — EDA (design)

- Counts per source and country, train and test, including French test S1/S2/S3.
- Singleton rate and matches-per-S1 histogram.
- Reverse-index S2/S3 → S1. If empty collision set, one-to-one is a hard constraint.
- Count same-country vs cross-country matched pairs. If zero cross-country, block on the country **string** (open set; France is just another label).
- Dump 30 matched pairs, 30 near-miss non-pairs, 25 raw French test S1 rows.
- Output: `reports/eda.md`, `reports/eda.json`.

## Stage 1 — Validation protocol (design)

- Hold out 20% of S1 entities, stratified by `country|singleton`.
- Validation gallery = all S2/S3 linked to those S1s + `gallery_unlinked_frac` of unlinked S2/S3.
- OOF: K=3 folds over S1. Bi-encoder / GBM trained per fold so no feature sees that pair.
- Unseen-country proxy: LOCO masks (train US → score India, and the reverse). France is not in train so this is the generalisation probe.
- Metric: PDF F0.5, unit-tested (`pred {A,B,C}` vs `truth {A,C}` → 0.714).
- Output: `cache/splits.json`, `reports/split.json`.

## Stage 2 — Normalisation (design)

- NFKC, lowercase, accent-fold (Latin combining marks only; Indic kept), `&`→`and`.
- Legal form extracted (US / India / France lists; unknown countries still try the union).
- Address: house / unit / postcode (ZIP 5, PIN 6, French 5 + CEDEX), landmarks, directions.
- Hand-written abbrev maps (US rd/st/ave/…, India marg/ngr/nr/…, France r/av/bd/chem/bis/ter/…).
- Optional mined substitutions from aligned tokens of matched train pairs (`normalize.mine_abbreviations`).
- IDF over **all** source files when `normalize.compute_idf` is true.
- Output: `cache/normalized/*.parquet`, `cache/idf.json`, `cache/mined_abbrev.json`.

## Stage 3 — Baseline end-to-end (design)

- Blocking: zero-shot jina FAISS (same country) ∪ char TF-IDF 2–4g ∪ key (rare name token | postcode+house).
- Union, dedupe, cap `k_cap=80`. Write that set as `candidate_pairs`.
- Matcher: RapidFuzz + number/postcode/direction features + LightGBM (monotone: house/postcode mismatch ↓, name sim ↑). No country categorical.
- Decision: swept global threshold on hold-out, then validator on test.
- Output: first valid `output/*.tsv`.

## Stage 4 — Bi-encoder (design)

- LoRA r=32, α=64, dropout 0.05, on q/k/v/o/gate/up/down. Full FT is `biencoder.full_finetune` ablation only.
- Phase A: InfoNCE / Cached-MNRL style, τ=0.05 (`scale=20`), effective batch 512–1024, mini-batch 32–64, entity-aware sampler (no two records of the same S1 in a batch), `NO_DUPLICATES` intent. Hard negatives from zero-shot same-country top-k plus synthetic one-digit / W↔E / token-swap.
- Symmetric pairs: S1↔S2/S3 and S2↔S3 sharing an S1.
- Optional Matryoshka dims [1024, 512, 256] (config flag).
- Phase B: ANCE remine + multi-task InfoNCE + CoSENT (λ=20). Ablate vs OnlineContrastive margin 0.5 (`phase_b_aux_loss`).
- GIST fallback: `use_gist_fallback` keeps the frozen zero-shot guide available if the entity-aware sampler is turned off.
- Eval R@1/10/50 each epoch; keep best adapter; merge for inference.
- Output: `models/biencoder/{best_adapter,last_adapter,merged}`, `reports/biencoder.json`.

## Stage 5 — Rank features, CE, stacker (design)

- Rank / gap-to-best / n_candidates / mutual-best / candidate-to-second-S1 margin.
- Cross-encoder: same jina/Qwen3 backbone, last-token → linear → sigmoid, LoRA r=16 + full head, BCE (not focal; calibrated p). Top-N≈10 by `p_gbm`. Input prefixed with `Document: ` (or `query: `).
- Stacker: logistic on [logit(p_gbm), logit(p_ce), rank features]; isotonic on OOF; Brier + reliability plot.
- Keep a stage only if hold-out macro F0.5 rises (`reports/evaluate.json` vs previous).

## Stage 6 — Decision (design)

Compare on OOF, confirm on hold-out:

- (a) global τ (swept)
- (b) τ and p ≥ α · max_p
- (c) per-entity expected F0.5: sort by p, pick prefix k∈{0..n} maximising 1.25·E[TP] / (0.25·E[|truth|]+k), with k=0 worth Π(1−p_i)

One-to-one on gallery IDs is applied **before** the prefix choice when EDA says no S2/S3 matches two S1s.

## Stage 7 — Final / package (design)

- Retrain on all train S1s (`--stage predict` after a full-data matcher retrain; `all` already trains on the hold-out protocol then scores test).
- Sanity: one row per test S1; France present; matches ⊆ candidates; singleton rate vs train-by-country.
- `utils/validate_submission.py` must print PASS.
- `bash make_zip.sh team_jina_er` → `team_jina_er_submission.zip`.

---

## Milestone table (filled by stage JSON after a run)

| milestone | where | macro F0.5 |
| --- | --- | --- |
| baseline (zs + LGBM + global τ) | `reports/decision.json` after Stage 3 | _run_ |
| + fine-tuned bi-encoder | `reports/biencoder.json` + re-block | _run_ |
| + rank/context features | `reports/matcher.json` | _run_ |
| + cross-encoder | `reports/crossencoder.json` | _run_ |
| + stacker / isotonic | `reports/stacker.json` | _run_ |
| decision (a) / (b) / (c) | `reports/decision.json` `comparison` | _run_ |
| LOCO US↔India | `reports/evaluate.json` `loco_on_holdout` | _run_ |

## Loss ablation knobs

| ablation | config |
| --- | --- |
| InfoNCE only | `phase_b_epochs: 0` |
| + CoSENT | `phase_b_aux_loss: cosent` (default) |
| + OnlineContrastive | `phase_b_aux_loss: online_contrastive` |
| LoRA vs full FT | `full_finetune: false \| true` |
| entity-aware vs GIST | `use_entity_aware_sampler` / `use_gist_fallback` |

---
