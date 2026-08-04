#!/bin/bash
# Validate UNIDOS_13_18 vs classification_20260730 for one year (default 2017).
#
#   cd ~/fire
#   git checkout feat/auxiliares-to-gee && git pull
#   bash validation/run_unidos_validation_year.sh
#
# Calendar product (preferred if present):
#   PREFER_CALENDAR=1 bash validation/run_unidos_validation_year.sh
#
# Force season folder only:
#   PREFER_CALENDAR=0 CLASS_DIR=~/classification_20260730 \
#     bash validation/run_unidos_validation_year.sh
#
# Other year:
#   VALIDATE_YEAR=2016 bash validation/run_unidos_validation_year.sh

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"
SCRIPT="${FIRE_REPO}/validation/run_unidos_classification_validation.py"

YEAR="${VALIDATE_YEAR:-2017}"
REFERENCE_SHP="${REFERENCE_SHP:-${HOME}/validation/UNIDOS_13_18.shp}"
OUTPUT_ROOT="${VALIDATE_OUTPUT_ROOT:-${HOME}/validation/unidos_vs_20260730}"
YEAR_COLUMN="${YEAR_COLUMN:-Season}"
WORKERS="${VALIDATE_WORKERS:-4}"
PREFER_CALENDAR="${PREFER_CALENDAR:-1}"
CLASS_DIR="${CLASS_DIR:-}"
SKIP_EXISTING="${VALIDATE_SKIP_EXISTING:-0}"

echo "============================================="
echo "UNIDOS vs classification validation"
echo "  Year:       ${YEAR}"
echo "  Reference:  ${REFERENCE_SHP}"
echo "  Output:     ${OUTPUT_ROOT}"
echo "  Calendar?:  ${PREFER_CALENDAR}"
echo "  Class dir:  ${CLASS_DIR:-auto}"
echo "============================================="

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT} — pull feat/auxiliares-to-gee" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_SHP}" ]]; then
  echo "ERROR: missing reference: ${REFERENCE_SHP}" >&2
  exit 1
fi

"${PYTHON}" -c "import geopandas, rasterio, numpy, pandas; print('deps OK')"

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --year "${YEAR}"
  --reference-shp "${REFERENCE_SHP}"
  --output-root "${OUTPUT_ROOT}"
  --year-column "${YEAR_COLUMN}"
  --workers "${WORKERS}"
  --python "${PYTHON}"
)

if [[ "${PREFER_CALENDAR}" == "1" ]]; then
  cmd+=(--prefer-calendar)
fi
if [[ -n "${CLASS_DIR}" ]]; then
  cmd+=(--classification-dir "${CLASS_DIR}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

cd "${FIRE_REPO}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo ""
echo "Results:"
ls -lh "${OUTPUT_ROOT}/year_${YEAR}/05_jaccard/" 2>/dev/null || true
ls -lh "${OUTPUT_ROOT}/year_${YEAR}/04_hits/" 2>/dev/null || true
echo "Manifest: ${OUTPUT_ROOT}/year_${YEAR}/run_manifest.json"
