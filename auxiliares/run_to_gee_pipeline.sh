#!/usr/bin/env bash
# MapBiomas Fire — auxiliares: enmascarar TIF con polígonos finales + mosaico Chile por año
#
#   cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
#   source auxiliares/cluster_paths.env
#   bash auxiliares/run_to_gee_pipeline.sh
#
# STEPS (comma-separated, or "all"): mask_tiles, merge_years

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"

_preserve_STEPS="${STEPS:-}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

[[ -n "${_preserve_STEPS}" ]] && STEPS="${_preserve_STEPS}"

PYTHON="${PYTHON:-}"
WORK_ROOT="${WORK_ROOT:-}"
CLASSIFIED_INPUT_DIR="${CLASSIFIED_INPUT_DIR:-${WORK_ROOT}/classified_filtered}"
POLYGON_INPUT_DIR="${POLYGON_INPUT_DIR:-${WORK_ROOT}/polygons_filtered_min20ha_p25}"
TO_GEE_ROOT="${TO_GEE_ROOT:-${HOME}/toGEE}"
OUTPUT_BY_TILE="${OUTPUT_BY_TILE:-${TO_GEE_ROOT}/by_tile}"
OUTPUT_BY_YEAR="${OUTPUT_BY_YEAR:-${TO_GEE_ROOT}/by_year_chile}"
MERGE_OUTPUT_STEM="${MERGE_OUTPUT_STEM:-chile_burn_p25}"
MASK_WORKERS="${MASK_WORKERS:-4}"
POLYGON_SUFFIX="${POLYGON_SUFFIX:-auto}"
STEPS="${STEPS:-all}"

log() { echo "[$(date -Iseconds)] $*"; }

step_enabled() {
  local step="$1"
  [[ "${STEPS}" == "all" ]] && return 0
  [[ ",${STEPS}," == *",${step},"* ]]
}

if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not set. Copy auxiliares/cluster_paths.to_gee.env.leftraru to cluster_paths.env" >&2
  exit 1
fi

for d in "${CLASSIFIED_INPUT_DIR}" "${POLYGON_INPUT_DIR}"; do
  if [[ ! -d "${d}" ]]; then
    echo "ERROR: Directory not found: ${d}" >&2
    exit 1
  fi
done

cd "${REPO_ROOT}"
mkdir -p "${TO_GEE_ROOT}" "${OUTPUT_BY_TILE}" "${OUTPUT_BY_YEAR}"

if step_enabled "mask_tiles"; then
  log "=== Mask classified rasters by polygon layer ==="
  log "Input TIF:  ${CLASSIFIED_INPUT_DIR}"
  log "Polygons:   ${POLYGON_INPUT_DIR}"
  log "Output:     ${OUTPUT_BY_TILE}"
  "${PYTHON}" auxiliares/mask_classified_by_polygons.py \
    --input-dir "${CLASSIFIED_INPUT_DIR}" \
    --polygon-dir "${POLYGON_INPUT_DIR}" \
    --output-dir "${OUTPUT_BY_TILE}" \
    --polygon-suffix "${POLYGON_SUFFIX}" \
    --workers "${MASK_WORKERS}" \
    --stats-json "${TO_GEE_ROOT}/logs/mask_by_polygons_stats.json"
fi

if step_enabled "merge_years"; then
  log "=== Merge regions into one Chile raster per year ==="
  "${PYTHON}" validation/merge_reprojected_tiles_by_year.py \
    --input-dir "${OUTPUT_BY_TILE}" \
    --output-dir "${OUTPUT_BY_YEAR}" \
    --output-stem "${MERGE_OUTPUT_STEM}" \
    --method first
  log "Yearly mosaics: ${OUTPUT_BY_YEAR}/${MERGE_OUTPUT_STEM}_<year>.tif"
fi

log "=== toGEE pipeline finished ==="
log "By tile:  ${OUTPUT_BY_TILE}"
log "By year:  ${OUTPUT_BY_YEAR}"
