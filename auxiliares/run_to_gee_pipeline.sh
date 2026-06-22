#!/usr/bin/env bash
# MapBiomas Fire — auxiliares: enmascarar TIF con polígonos finales + mosaico Chile por año
#
#   cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
#   source auxiliares/cluster_paths.env
#   bash auxiliares/run_to_gee_pipeline.sh
#
# STEPS (comma-separated):
#   all              mask_tiles + merge_years
#   all_with_fill    mask + merge + fill_tiles + fill_merge_years
#   fill_tiles, fill_merge_years, mask_tiles, merge_years

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
OUTPUT_BY_TILE_FILLED="${OUTPUT_BY_TILE_FILLED:-${TO_GEE_ROOT}/by_tile_filled}"
OUTPUT_BY_YEAR_FILLED="${OUTPUT_BY_YEAR_FILLED:-${TO_GEE_ROOT}/by_year_chile_filled}"
MERGE_OUTPUT_STEM="${MERGE_OUTPUT_STEM:-chile_burn_p25}"
MERGE_FILLED_OUTPUT_STEM="${MERGE_FILLED_OUTPUT_STEM:-chile_burn_p25_filled}"
REFERENCE_SCARS_SHP="${REFERENCE_SCARS_SHP:-/home/flepin/validation/UNIDOS_13_18.shp}"
REFERENCE_YEAR_COLUMN="${REFERENCE_YEAR_COLUMN:-Season}"
FILL_REQUIRE_OVERLAP="${FILL_REQUIRE_OVERLAP:-1}"
FILL_SKIP_EXISTING="${FILL_SKIP_EXISTING:-0}"
FILL_PATTERN="${FILL_PATTERN:-*.tif}"
FILL_FROM_YEAR="${FILL_FROM_YEAR:-2013}"
FILL_TO_YEAR="${FILL_TO_YEAR:-2018}"
PROGRESS_HEARTBEAT_SEC="${PROGRESS_HEARTBEAT_SEC:-30}"
MASK_WORKERS="${MASK_WORKERS:-4}"
FILL_WORKERS="${FILL_WORKERS:-${MASK_WORKERS}}"
POLYGON_SUFFIX="${POLYGON_SUFFIX:-auto}"
STEPS="${STEPS:-all}"

log() { echo "[$(date -Iseconds)] $*"; }

step_enabled() {
  local step="$1"
  if [[ "${STEPS}" == "all" ]]; then
    case "${step}" in
      fill_tiles|fill_merge_years) return 1 ;;
      *) return 0 ;;
    esac
  fi
  if [[ "${STEPS}" == "all_with_fill" ]]; then
    return 0
  fi
  [[ ",${STEPS}," == *",${step},"* ]]
}

if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not set. Copy auxiliares/cluster_paths.to_gee.env.leftraru to cluster_paths.env" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${TO_GEE_ROOT}" "${TO_GEE_ROOT}/logs"

log "Tip: open another terminal and run: bash auxiliares/watch_to_gee_progress.sh"

if step_enabled "mask_tiles"; then
  for d in "${CLASSIFIED_INPUT_DIR}" "${POLYGON_INPUT_DIR}"; do
    if [[ ! -d "${d}" ]]; then
      echo "ERROR: Directory not found: ${d}" >&2
      exit 1
    fi
  done
  mkdir -p "${OUTPUT_BY_TILE}"
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
    --heartbeat-sec "${PROGRESS_HEARTBEAT_SEC}" \
    --stats-json "${TO_GEE_ROOT}/logs/mask_by_polygons_stats.json"
fi

if step_enabled "merge_years"; then
  if [[ ! -d "${OUTPUT_BY_TILE}" ]]; then
    echo "ERROR: Tile output not found: ${OUTPUT_BY_TILE}" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_BY_YEAR}"
  log "=== Merge regions into one Chile raster per year ==="
  "${PYTHON}" validation/merge_reprojected_tiles_by_year.py \
    --input-dir "${OUTPUT_BY_TILE}" \
    --output-dir "${OUTPUT_BY_YEAR}" \
    --output-stem "${MERGE_OUTPUT_STEM}" \
    --method first
  log "Yearly mosaics: ${OUTPUT_BY_YEAR}/${MERGE_OUTPUT_STEM}_<year>.tif"
fi

if step_enabled "fill_tiles"; then
  if [[ ! -f "${REFERENCE_SCARS_SHP}" ]]; then
    echo "ERROR: Reference shapefile not found: ${REFERENCE_SCARS_SHP}" >&2
    exit 1
  fi
  if [[ ! -d "${OUTPUT_BY_TILE}" ]]; then
    echo "ERROR: Tile input not found: ${OUTPUT_BY_TILE}" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_BY_TILE_FILLED}"
  FILL_ARGS=(
    auxiliares/fill_raster_from_reference_scars.py
    --input-dir "${OUTPUT_BY_TILE}"
    --output-dir "${OUTPUT_BY_TILE_FILLED}"
    --reference-shp "${REFERENCE_SCARS_SHP}"
    --year-column "${REFERENCE_YEAR_COLUMN}"
    --pattern "${FILL_PATTERN}"
    --from-year "${FILL_FROM_YEAR}"
    --to-year "${FILL_TO_YEAR}"
    --workers "${FILL_WORKERS}"
    --heartbeat-sec "${PROGRESS_HEARTBEAT_SEC}"
    --stats-json "${TO_GEE_ROOT}/logs/fill_reference_tiles_stats.json"
  )
  if [[ "${FILL_REQUIRE_OVERLAP}" == "1" ]]; then
    FILL_ARGS+=(--require-overlap)
  else
    FILL_ARGS+=(--no-require-overlap)
  fi
  if [[ "${FILL_SKIP_EXISTING}" == "1" ]]; then
    FILL_ARGS+=(--skip-existing)
  fi
  log "=== Fill tile rasters from reference scars ==="
  log "Input TIF:  ${OUTPUT_BY_TILE}"
  log "Reference:  ${REFERENCE_SCARS_SHP}"
  log "Fill years: ${FILL_FROM_YEAR}..${FILL_TO_YEAR}"
  log "Output:     ${OUTPUT_BY_TILE_FILLED}"
  "${PYTHON}" "${FILL_ARGS[@]}"
fi

if step_enabled "fill_merge_years"; then
  if [[ ! -d "${OUTPUT_BY_TILE_FILLED}" ]]; then
    echo "ERROR: Filled tile output not found: ${OUTPUT_BY_TILE_FILLED}" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_BY_YEAR_FILLED}"
  log "=== Merge filled tiles into Chile yearly mosaics ==="
  "${PYTHON}" validation/merge_reprojected_tiles_by_year.py \
    --input-dir "${OUTPUT_BY_TILE_FILLED}" \
    --output-dir "${OUTPUT_BY_YEAR_FILLED}" \
    --output-stem "${MERGE_FILLED_OUTPUT_STEM}" \
    --method first
  log "Yearly mosaics: ${OUTPUT_BY_YEAR_FILLED}/${MERGE_FILLED_OUTPUT_STEM}_<year>.tif"
fi

log "=== toGEE pipeline finished ==="
log "By tile:         ${OUTPUT_BY_TILE}"
log "By year:         ${OUTPUT_BY_YEAR}"
log "By tile filled:  ${OUTPUT_BY_TILE_FILLED}"
log "By year filled:  ${OUTPUT_BY_YEAR_FILLED}"
