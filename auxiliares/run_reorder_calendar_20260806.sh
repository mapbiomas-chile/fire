#!/bin/bash
# Season (temp_10 remap) -> calendar year for classification_20260806.
#
# Input (flat):
#   ~/classification_20260806/burned_area_chile_temp_10_remap_{year}.tif
# Output (subdir of same campaign folder):
#   ~/classification_20260806/calendar/burned_area_chile_calendar_{year}.tif
#
#   cd ~/fire && git checkout feat/auxiliares-to-gee && git pull
#   conda activate mb_fuego
#   bash auxiliares/run_reorder_calendar_20260806.sh
#
# Optional:
#   SEASON_CLASS_DIR=~/classification_20260806 \
#   CALENDAR_SUBDIR=calendar \
#   REORDER_FROM_YEAR=2013 REORDER_TO_YEAR=2025 \
#     bash auxiliares/run_reorder_calendar_20260806.sh
#
# Dry-run:
#   DRY_RUN=1 bash auxiliares/run_reorder_calendar_20260806.sh

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
SCRIPT="${FIRE_REPO}/auxiliares/reorder_season_to_calendar.py"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

INPUT_DIR="${SEASON_CLASS_DIR:-${HOME}/classification_20260806}"
SUBDIR="${CALENDAR_SUBDIR:-calendar}"
OUTPUT_DIR="${CALENDAR_OUTPUT_DIR:-${INPUT_DIR}/${SUBDIR}}"
FROM_YEAR="${REORDER_FROM_YEAR:-2013}"
TO_YEAR="${REORDER_TO_YEAR:-2025}"
SKIP_EXISTING="${REORDER_SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================="
echo "Season temp_10_remap -> calendar"
echo "  Input:   ${INPUT_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Years:   ${FROM_YEAR}-${TO_YEAR}"
echo "  Names:   burned_area_chile_temp_10_remap_YYYY.tif"
echo "============================================="

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT} — pull feat/auxiliares-to-gee" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: missing input dir: ${INPUT_DIR}" >&2
  exit 1
fi

n_in=$(ls -1 "${INPUT_DIR}"/burned_area_chile_temp_10_remap_*.tif 2>/dev/null | wc -l || true)
echo "Season remap TIFs found: ${n_in}"
if [[ "${n_in}" -lt 1 ]]; then
  echo "ERROR: no burned_area_chile_temp_10_remap_*.tif in ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" -c "import numpy, pandas, rasterio; print('deps OK')"

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --from-year "${FROM_YEAR}"
  --to-year "${TO_YEAR}"
  --output-pattern "burned_area_chile_calendar_{year}.tif"
  --stats-csv "${OUTPUT_DIR}/season_to_calendar_stats.csv"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

cd "${FIRE_REPO}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo ""
echo "Results:"
ls -lh "${OUTPUT_DIR}"/*.tif 2>/dev/null | head -20 || true
echo "Stats: ${OUTPUT_DIR}/season_to_calendar_stats.csv"
