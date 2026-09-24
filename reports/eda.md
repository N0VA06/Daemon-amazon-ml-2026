# Stage 0 — EDA

This file is **overwritten** by `python -m src.run --stage eda` with exact
counts. Notes below are from reading the shipped TSVs (no pipeline run).

## What the EDA stage measures

- Row counts and country histograms for every train/test source file
- French test S1/S2/S3 counts (France is absent from train)
- Singleton rate and matches-per-S1 histogram
- Whether any S2/S3 ID is linked to more than one S1 (one-to-one constraint)
- Whether every matched pair shares a `country` string (same-country blocking)
- 30 raw matched pairs and 30 near-miss non-pairs
- 25 raw French test S1 rows (inputs only)

## Observed before a run

- Columns: `entity_id`, `business_name`, `business_address`, `country`
- ID prefixes `S1-` / `S2-` / `S3-`; ground truth is
  `source1_entity_id`, `matched_entity_ids` (comma list, empty = singleton)
- Approximate sizes (from file length): train S1 / GT ~2.21M, train S2 ~5.03M,
  train S3 ~5.29M, test S1 ~1.73M
- Train samples are US and India; test samples include France
  (`<< Team Ecole`, `ZNB Club SARL`, `Thermal & Fils SASU`, Bordeaux / Dunkerque)
- Address noise already visible: reordered US components, Indian landmarks,
  empty S3 addresses, Devanagari S2 names, `19 1/2`, `Unit APARTMENT G`
- Benchmark error file shows the classic near-duplicate traps (house ±1,
  generic “Jai/Shree Trading”, (East) vs other city)

Run the stage to replace this document with the full tables.
