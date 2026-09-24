#!/usr/bin/env bash
# Build <team_name>_submission.zip from student_resource/.
# Usage: bash make_zip.sh [team_name]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
TEAM="${1:-team_jina_er}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/output"
mkdir -p "$STAGE/code/business_entity_resolution/src"
mkdir -p "$STAGE/code/business_entity_resolution/configs"

if [[ -f output/matching_results.tsv ]]; then
  cp output/matching_results.tsv "$STAGE/output/"
  cp output/candidate_pairs.tsv "$STAGE/output/"
else
  echo "WARNING: output/*.tsv missing — zip will not be a valid final package until you run --stage predict" >&2
fi

cp -R src/. "$STAGE/code/business_entity_resolution/src/"
cp -R configs/. "$STAGE/code/business_entity_resolution/configs/"
cp requirements.txt "$STAGE/code/business_entity_resolution/"
if [[ -f code/business_entity_resolution/README.md ]]; then
  cp code/business_entity_resolution/README.md "$STAGE/code/business_entity_resolution/"
else
  cp PIPELINE.md "$STAGE/code/business_entity_resolution/README.md"
fi
cp Documentation_template.md "$STAGE/"
# keep licence + experiment logs for reviewers
mkdir -p "$STAGE/code/business_entity_resolution/reports"
[[ -f reports/licences.md ]] && cp reports/licences.md "$STAGE/code/business_entity_resolution/reports/"
[[ -f experiments.md ]] && cp experiments.md "$STAGE/code/business_entity_resolution/"

ZIP="${TEAM}_submission.zip"
rm -f "$ZIP"
(cd "$STAGE" && zip -rq "$ROOT/$ZIP" output code Documentation_template.md)
echo "wrote $ROOT/$ZIP"
echo "contents:"
unzip -l "$ZIP" | head -n 40
