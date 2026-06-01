#!/usr/bin/env bash
# =============================================================================
# MapBiomas Fire — class-mask filtering pipeline (cluster-ready)
#
# LULC input (choose one):
#   LULC_DIR   — GEE tiles per year (e.g. lulc_chile_collection02_2013*.tif)
#   LULC_STACK — single multi-band LULC stack (legacy)
#
# Usage (GEE tiles):
#   export LULC_DIR=/home/flepin/lulc_collection02
#   export CLASSIFIED_DIR=/home/flepin/classi_v2
#   export WORK_ROOT=/home/flepin/filtering_work
#   bash filtering/run_filtering_pipeline.sh
#
# NLHPC: sbatch filtering/run_filtering_pipeline_slurm.sh
#
# STEPS: lulc_mosaic, masks_accumulated, masks_yearly, masks_total, filter
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

LULC_DIR="${LULC_DIR:-}"
LULC_STACK="${LULC_STACK:-}"
LULC_TILE_PATTERN="${LULC_TILE_PATTERN:-lulc_chile_collection02_*.tif}"

CLASSIFIED_DIR="${CLASSIFIED_DIR:-/data/mapbiomas/fire/classified}"
WORK_ROOT="${WORK_ROOT:-/data/mapbiomas/fire/filtering_work}"

LULC_MOSAICS_DIR="${LULC_MOSAICS_DIR:-${WORK_ROOT}/lulc_mosaics}"
MASCARAS_ROOT="${MASCARAS_ROOT:-${WORK_ROOT}/mascaras}"
ACCUMULATED_DIR="${ACCUMULATED_DIR:-${MASCARAS_ROOT}/acumuladas}"
YEARLY_MASKS_DIR="${YEARLY_MASKS_DIR:-${MASCARAS_ROOT}/by_year}"
TOTAL_MASKS_DIR="${TOTAL_MASKS_DIR:-${MASCARAS_ROOT}/totales}"
FILTERED_DIR="${FILTERED_DIR:-${WORK_ROOT}/classified_filtered}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2025}"
START_YEAR_BAND1="${START_YEAR_BAND1:-2000}"
WORKERS="${WORKERS:-16}"
FILL_VALUE="${FILL_VALUE:-0}"
STEPS="${STEPS:-all}"

DEFAULT_STEPS_GEE="lulc_mosaic,masks_accumulated,masks_yearly,masks_total,filter"
DEFAULT_STEPS_STACK="masks_accumulated,masks_yearly,masks_total,filter"

log() { echo "[$(date -Iseconds)] $*"; }

use_gee_lulc() {
  [[ -n "${LULC_DIR}" ]]
}

resolve_default_steps() {
  if use_gee_lulc; then
    echo "${DEFAULT_STEPS_GEE}"
  else
    echo "${DEFAULT_STEPS_STACK}"
  fi
}

step_enabled() {
  local name="$1"
  local defaults
  defaults="$(resolve_default_steps)"
  if [[ "${STEPS}" == "all" ]]; then
    [[ ",${defaults}," == *",${name},"* ]]
    return
  fi
  [[ ",${STEPS}," == *",${name},"* ]]
}

run_py() {
  log "RUN: $*"
  "${PYTHON}" "$@"
}

validate_lulc_config() {
  if use_gee_lulc; then
    if [[ ! -d "${LULC_DIR}" ]]; then
      echo "ERROR: LULC_DIR not found: ${LULC_DIR}" >&2
      exit 1
    fi
    return
  fi
  if [[ -n "${LULC_STACK}" && -e "${LULC_STACK}" ]]; then
    return
  fi
  echo "ERROR: Set LULC_DIR (GEE tiles) or LULC_STACK (multiband file)." >&2
  exit 1
}

mkdir -p "${WORK_ROOT}/logs" "${ACCUMULATED_DIR}" "${YEARLY_MASKS_DIR}" "${TOTAL_MASKS_DIR}" "${FILTERED_DIR}"
cd "${REPO_ROOT}"

if step_enabled "lulc_mosaic"; then
  if ! use_gee_lulc; then
    log "=== Skip lulc_mosaic (LULC_DIR not set) ==="
  else
    log "=== Step 0: mosaic GEE LULC tiles by year ==="
    run_py filtering/mosaic_gee_lulc_tiles.py \
      --input-dir "${LULC_DIR}" \
      --output-dir "${LULC_MOSAICS_DIR}" \
      --pattern "${LULC_TILE_PATTERN}" \
      --from-year "${FROM_YEAR}" \
      --to-year "${TO_YEAR}" \
      --workers "${WORKERS}"
  fi
fi

if step_enabled "masks_accumulated" || step_enabled "masks_yearly" || step_enabled "masks_total"; then
  validate_lulc_config
fi

if step_enabled "masks_accumulated"; then
  log "=== Step 1a: accumulated class masks ==="
  if use_gee_lulc; then
    run_py filtering/create_accumulated_class_masks.py \
      --yearly-dir "${LULC_MOSAICS_DIR}" \
      --yearly-pattern "lulc_*.tif" \
      --from-year "${FROM_YEAR}" \
      --to-year "${TO_YEAR}" \
      --output-dir "${ACCUMULATED_DIR}"
  else
    run_py filtering/create_accumulated_class_masks.py \
      --input-tif "${LULC_STACK}" \
      --output-dir "${ACCUMULATED_DIR}"
  fi
fi

if step_enabled "masks_yearly"; then
  log "=== Step 1b: yearly thematic masks ==="
  if use_gee_lulc; then
    run_py filtering/create_yearly_masks.py \
      --yearly-dir "${LULC_MOSAICS_DIR}" \
      --yearly-pattern "lulc_*.tif" \
      --output-dir "${YEARLY_MASKS_DIR}" \
      --from-year "${FROM_YEAR}" \
      --to-year "${TO_YEAR}" \
      --workers "${WORKERS}"
  else
    run_py filtering/create_yearly_masks.py \
      --input-tif "${LULC_STACK}" \
      --output-dir "${YEARLY_MASKS_DIR}" \
      --start-year-in-band-1 "${START_YEAR_BAND1}" \
      --from-year "${FROM_YEAR}" \
      --to-year "${TO_YEAR}" \
      --workers "${WORKERS}"
  fi
fi

if step_enabled "masks_total"; then
  log "=== Step 1c: mascara_total_<year>.tif ==="
  run_py filtering/create_total_masks_by_year.py \
    --mascaras-root "${MASCARAS_ROOT}" \
    --from-year "${FROM_YEAR}" \
    --to-year "${TO_YEAR}" \
    --workers "${WORKERS}"
fi

if step_enabled "filter"; then
  log "=== Step 2: apply year masks to classified rasters ==="
  run_py filtering/filter_classified_parallel.py \
    --input-dir "${CLASSIFIED_DIR}" \
    --masks-dir "${TOTAL_MASKS_DIR}" \
    --output-dir "${FILTERED_DIR}" \
    --workers "${WORKERS}" \
    --fill-value "${FILL_VALUE}"
fi

log "=== Class-filter pipeline finished ==="
if use_gee_lulc; then
  log "LULC mosaics:    ${LULC_MOSAICS_DIR}/lulc_<year>.tif"
fi
log "Total masks:     ${TOTAL_MASKS_DIR}/mascara_total_<year>.tif"
log "Filtered tiles:  ${FILTERED_DIR}"
