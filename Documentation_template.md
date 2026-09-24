# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** team_jina_er
**Team Members:** [fill]
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We resolve Source-1 businesses against noisy S2/S3 records with a
precision-heavy stack: multi-view **jina-embeddings-v5-text-small** (text-matching)
plus char-TF-IDF and key blocking, a monotone LightGBM pair matcher, a jina
cross-encoder on the top-N, a logistic+isotonic stacker, and a per-entity
expected-F0.5 decision under a one-to-one gallery constraint. Country is an
open string (France is unseen in train). The backbone is one YAML key —
`Snowflake/snowflake-arctic-embed-l-v2.0` (Apache-2.0, prefix `query: `) is a
tested drop-in for the PDF licence constraint.

---

## 2. Methodology

### 2.1 Problem Analysis

Source 1 is the deduplicated reference. Each S1 may match zero, one, or many
S2/S3 records. Training countries observed in the source files are **US** and
**India**; the test set additionally contains **France** (legal forms SAS/SARL/SASU,
`Rue` / `Boulevard`, `bis`/`ter`). File-size inspection of the shipped TSVs:

| file | ~rows (header excluded) |
| --- | ---: |
| `dataset/train/train_source1.tsv` / `train_ground_truth.tsv` | ~2,206,822 |
| `dataset/train/train_source2.tsv` | ~5,034,617 |
| `dataset/train/train_source3.tsv` | ~5,285,604 |
| `dataset/test/test_source1.tsv` | ~1,732,545 |

Exact counts, singleton rate, one-to-one collisions, same-country rate, and
French test counts are computed by `python -m src.run --stage eda` and written
to `reports/eda.md` / `reports/eda.json`.

**Noise (from the PDF and raw samples):**

- Names: Pvt/Private, Corp/Corporation, legal-suffix drift, DBA, `&`/`and`,
  token reorder, typos, Shree/Shri, Thiruvananthapuram/Trivandrum, Devanagari
  and Tamil/Gujarati/Punjabi source-2/3 strings paired with Latin S1 names.
- Addresses: Rd/Road, missing PIN/state, landmarks (“Near Fortis Hospital”),
  component reorder (“OH, Columbus, 5559 Orville Avenue”), `19 1/2`, `5 bis`,
  Ghatkopar (W) vs (E), empty addresses on some S3 rows.
- Near-duplicate negatives that embeddings collapse: 174 vs 147 Stover Road,
  house-number ±1 (`1108` vs `1109` Bonhomme Lake), generic names
  (“Eye Group”, “Primary Care Medicine”, “Jai Trading”).

**Scoring.** Macro F0.5 over **every** S1, singletons included.
`F0.5 = 1.25·TP / (0.25·|truth| + |pred|)`. Unit test (PDF): pred {A,B,C},
truth {A,C} → **0.714**. A false merge is more expensive than a miss, so
blocking maximises recall and the decision layer maximises precision.

**Traps handled in code:**

- `country` is never one-hot to {US, India}. Blocking groups by the country
  *string*; unknown labels still form a bucket. Every test S1 is written out.
- Generic names rely on house number, postcode, direction, and landmark overlap.
- Number/token hard checks (`house_num_mismatch` monotone decreasing in LightGBM).

### 2.2 Solution Strategy

**Approach Type:** Hybrid blocking + pairwise classifier + cross-encoder stacker
+ expected-F0.5 decision.

**Core Innovation:** (i) swappable symmetric bi-encoder (jina text-matching
LoRA or Snowflake) trained with entity-aware InfoNCE then CoSENT so a *global*
cosine is meaningful for singletons; (ii) precision layer that combines
monotone LightGBM, a last-token CE head, and per-entity expected F0.5 under
one-to-one.

```
TSVs ─► Normalizer ─► 3 text views (combined / name / address)
          │            + parsed fields (numbers, postcodes, legal form)
          │
          ├─► BLOCKING  (maximise recall)
          │     A. jina bi-encoder (LoRA), FAISS top-K, same country string
          │     B. char TF-IDF (2–4 grams) on name, top-K
          │     C. key: shared rare name token | postcode + house number
          │     union, dedupe, cap K  ──► candidate_pairs.tsv
          │
          ├─► SCORING   (maximise precision)
          │     S1. LightGBM  → p_gbm
          │     S2. jina CE on top-N by p_gbm → p_ce
          │     S3. logistic stacker + isotonic → p
          │
          └─► DECISION  (maximise macro F0.5)
                one-to-one on S2/S3 ─► expected-F0.5 prefix
                ──► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **A.** Fine-tuned (or zero-shot) backbone, 1024-d L2-normalised combined view,
    FAISS IVF-IP **within the same country string**.
  - **B.** Character TF-IDF (2–4 grams) on expanded name → TruncatedSVD(256) → FAISS.
  - **C.** Inverted index on rare `core_name` tokens (IDF quantile) **or**
    `(postcode, house_number)`.
- **Union / cap:** round-robin merge, `k_cap=80` (config).
- **Candidate pairs generated:** `reports/blocking_*.json` → `n_pairs`,
  `avg_candidates`. Test write-up is the exact set scored by the matcher
  (`output/candidate_pairs.tsv`).
- **How you ensured true matches were not lost:** three complementary blockers
  (semantic / surface / exact-key); same-country only after EDA confirms no
  cross-country positives; recall@K curve logged for K ∈ {1,5,10,20,30,50,80}
  on the hold-out (`recall_at_k_union`, `recall_at_k_faiss`). Chosen operating
  point is `k_cap=80` unless the curve saturates earlier.

**Recall vs K (fill from `reports/blocking_val_zs.json` and `blocking_val_ft.json`):**

| K | R@K zero-shot jina (union) | R@K fine-tuned jina (union) |
| ---: | ---: | ---: |
| 1 | _run_ | _run_ |
| 10 | _run_ | _run_ |
| 20 | _run_ | _run_ |
| 50 | _run_ | _run_ |
| 80 | _run_ | _run_ |

Reduction ratio ≈ `avg_candidates / n_gallery_in_country` (logged).

---

## 4. Matching Model

**Features used:**

- **Name:** RapidFuzz ratio / token_sort / token_set / partial, Jaro–Winkler,
  char-TF-IDF cosine (via blocking score), IDF-weighted token Jaccard; the same
  on `core_name` (legal form stripped); legal-form match/mismatch; acronym;
  length ratio.
- **Address:** the same string similarities; house-number match / mismatch /
  missing; unit; postcode match / mismatch / missing; W↔E direction conflict;
  landmark overlap; digit-set Jaccard; city/state token overlap via address
  token Jaccard; missing-component flags on each side.
- **Embeddings:** fine-tuned and zero-shot jina cosine on combined / name /
  address views.
- **Rank / context:** rank and gap-to-best under each scorer; `n_candidates`;
  mutual-best; candidate margin to its second-best S1.
- **Source flag** S2/S3. **`same_country` only** — `country` is not a
  LightGBM categorical (France must generalise).

**Model type:**

1. LightGBM `binary` log-loss, early stopping on OOF, monotone constraints
   (house-number mismatch ↓, postcode mismatch ↓, name similarity ↑).
2. Cross-encoder on the same jina/Qwen3-0.6B backbone: last-token hidden →
   linear → sigmoid. LoRA r=16 + full head. **BCE** (not focal) so probabilities
   stay calibratable. Top-N ≈ 10 by `p_gbm`. A/B order swapped at train time.
3. Logistic regression on `[logit(p_gbm), logit(p_ce), rank features]`, then
   isotonic calibration on OOF. Reliability plot + Brier in `reports/stacker.json`
   / `reports/reliability.png`.

**Bi-encoder losses (why these):**

| loss | role |
| --- | --- |
| InfoNCE / Cached-MNRL, τ=0.05 | ranking *within one query*; large batches (512–1024) via GradCache-style mini-batches |
| CoSENT, λ=20 | every positive cosine above every negative *across* queries → one global threshold, needed for singletons |
| OnlineContrastive, margin 0.5 | ablation against CoSENT (`phase_b_aux_loss`) |
| Entity-aware sampler | no two records of the same entity in a batch (kills false in-batch negatives). GIST-guide fallback if disabled |
| LoRA r=32 | keeps multilingual weights intact for unseen France; full FT is an ablation |

Augmentation (on the fly, both sides): abbrev swap, drop/change legal suffix,
token reorder, address-component reorder, drop PIN/state, landmark add/remove,
1–2 char typos. Hard-neg generator: one-digit house change, W↔E, swap a
core-name token.

**Threshold selection method:** Stage 6 compares (a) swept global τ,
(b) τ + relative `p ≥ α·max_p`, (c) per-entity expected F0.5
`E[1.25·TP/(0.25·|truth|+k)]` with `k=0` = `Π(1−p_i)`. Default rule is (c).
One-to-one on gallery IDs is applied first when EDA reports no S2/S3→multi-S1.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** see `reports/evaluate.json` `holdout.macro_f05` after
  `--stage evaluate`. Milestone ladder (each stage kept only if F0.5 rises):

| milestone | macro F0.5 |
| --- | --- |
| baseline zs + LGBM + global τ | `reports/decision.json` |
| + fine-tuned bi-encoder | re-block `ft` |
| + rank/context features | `reports/matcher.json` |
| + cross-encoder | `reports/crossencoder.json` |
| + stacker / isotonic | `reports/stacker.json` |
| decision (a) / (b) / (c) | `reports/decision.json` `comparison` |

- **Leave-one-country-out:** `reports/evaluate.json` `loco_on_holdout`
  (train-on-complement, score US vs India on the hold-out slice).
- **Common false positives (wrong merges):** expected from the shipped
  benchmark errors — same brand / generic name in a different city; house
  number ±1 (`1108` vs `1109`); (W) vs (E); “Jai Industries” vs “Jain First
  Industries”. Mitigations: monotone house/postcode features, direction
  conflict, expected-F0.5 preferring k=0 on low p.
- **Common false negatives (missed matches):** heavy transliteration
  (Latin S1 ↔ Devanagari S2), empty S3 addresses, DBA / legal-suffix-only
  overlap. Mitigations: key blocking on postcode+house, name-only and
  address-only embedding views, mined abbrev maps, CoSENT so a moderate
  cosine still ranks above junk.

**Sanity (Stage 7):** `n_output_rows == n_test_s1`; every France S1 present;
`matches_not_in_candidates == 0`; predicted singleton rate per country vs
train rates (`reports/predict.json`).

**Validator:**

```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

Must print `PASS — no blocking issues found. Safe to submit.`
(`--stage predict` runs this.)

---

## 6. Conclusion

The pipeline is a recall-first union blocker and a precision-first stack
(LightGBM + jina CE + isotonic) with a singleton-aware expected-F0.5 rule.
LoRA on a multilingual 0.6B encoder is the inductive bias for France; nothing
in the model is hard-coded to {US, India}. The same binary runs on
Snowflake Arctic by changing `backbone.preset`.

---

## Appendix

### A. Code Artefacts

Runnable copy lives in the zip at `code/business_entity_resolution/`
(`src/`, `configs/`, `README.md`, `requirements.txt`). Development entry
point from `student_resource/`:

```bash
python -m src.run --stage {eda,split,normalize,block,train_biencoder,train_crossencoder,features,train_matcher,stack,decide,evaluate,predict,all} \
    --config configs/default.yaml
```

Pinned env: `requirements.txt` (`transformers==5.1.0`, `torch==2.8.0`,
`peft==0.17.1`, `sentence-transformers==5.1.1`, `faiss-cpu`, `rapidfuzz`,
`lightgbm`, `scikit-learn`, `pandas`, `pyarrow`).

Caches: `cache/normalized/*.parquet`, `cache/embeddings/*.npy`, pair
`*.parquet`, FAISS built on the fly. LoRA adapters are saved under
`models/biencoder/{best_adapter,last_adapter}` and merged to
`models/biencoder/merged` for inference.

Package:

```bash
bash make_zip.sh team_jina_er
# → team_jina_er_submission.zip
```

Zip layout matches the PDF: `output/` (both TSVs), `code/business_entity_resolution/`,
`Documentation_template.md`.

### B. Additional Results

- Architecture and losses: this document, §2–4.
- Blocking recall / reduction: `reports/blocking_*.json`.
- Reliability / Brier: `reports/stacker.json`, `reports/reliability.png`.
- Licences: `reports/licences.md` (from `python3 check_licences.py`).
  jina = CC-BY-NC-4.0 (user-chosen primary); Snowflake Arctic-embed-l-v2.0 =
  Apache-2.0 drop-in.
- Experiment chronology: `experiments.md`.
- EDA: `reports/eda.md`.

### C. Fair-play

No entity-resolution API, business registry, geocoder, or scraped city/PIN
list. Abbreviation maps are hand-written in `src/normalize.py` or mined from
aligned tokens of **provided** matched train pairs. Test records are never
hand-labelled. TSVs are read with
`pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.

### D. Model licences (HF tags)

| repo | expected tag |
| --- | --- |
| `jinaai/jina-embeddings-v5-text-small-text-matching` | cc-by-nc-4.0 |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | apache-2.0 |

`check_licences.py` overwrites `reports/licences.md` with the live card values.

---

**Note:** Numeric cells marked `_run_` are produced by the corresponding
stage and stored as JSON under `reports/`. Re-run `Documentation` assembly
from those files after a full ` --stage all ` on your hardware.
