#!/usr/bin/env bash
# =============================================================================
# MapBiomas Fire — class-mask filtering pipeline (cluster-ready)
#
# Builds land-cover masks and applies them to classified burned-area rasters.
# Stops before polygonize / area thresholding (those steps are under evaluation).
#
# See filtering/README.md for script details.
#
# Usage:
#   export LULC_STACK=/path/to/lulc_multiband.tif
#   export CLASSIFIED_DIR=/path/to/classified_tiles
#   export WORK_ROOT=/path/to/work/fire_filtering
#   bash filtering/run_filtering_pipeline.sh
#
# NLHPC SLURM: sbatch filtering/run_filtering_pipeline_slurm.sh  (see classification/run_classify_fire_model_slurm_v2.sh)
#
# STEPS (comma-separated, or "all" = full class-filter chain):
#   masks_accumulated, masks_yearly, masks_total, filter
#
# Example — masks already built, only filter rasters:
#   STEPS=filter bash filtering/run_filtering_pipeline.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

LULC_STACK="${LULC_STACK:-/data/mapbiomas/lulc_chile_multiband.tif}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-/data/mapbiomas/fire/classified}"
WORK_ROOT="${WORK_ROOT:-/data/mapbiomas/fire/filtering_work}"

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

# Default: full class-mask pipeline (no polygonize / area threshold)
STEPS="${STEPS:-all}"

DEFAULT_STEPS="masks_accumulated,masks_yearly,masks_total,filter"

log() { echo "[$(date -Iseconds)] $*"; }

step_enabled() {
  local name="$1"
  if [[ "${STEPS}" == "all" ]]; then
    [[ ",${DEFAULT_STEPS}," == *",${name},"* ]]
    return
  fi
  [[ ",${STEPS}," == *",${name},"* ]]
}

run_py() {
  log "RUN: $*"
  "${PYTHON}" "$@"
}

mkdir -p "${WORK_ROOT}/logs" "${ACCUMULATED_DIR}" "${YEARLY_MASKS_DIR}" "${TOTAL_MASKS_DIR}" "${FILTERED_DIR}"
cd "${REPO_ROOT}"

if step_enabled "masks_accumulated"; then
  log "=== Step 1a: accumulated class masks ==="
  run_py filtering/create_accumulated_class_masks.py \
    --input-tif "${LULC_STACK}" \
    --output-dir "${ACCUMULATED_DIR}"
fi

if step_enabled "masks_yearly"; then
  log "=== Step 1b: yearly thematic masks (rio_lago, agricultura, ...) ==="
  run_py filtering/create_yearly_masks.py \
    --input-tif "${LULC_STACK}" \
    --output-dir "${YEARLY_MASKS_DIR}" \
    --start-year-in-band-1 "${START_YEAR_BAND1}" \
    --from-year "${FROM_YEAR}" \
    --to-year "${TO_YEAR}" \
    --workers "${WORKERS}"
fi

if step_enabled "masks_total"; then
  log "=== Step 1c: combine → mascara_total_<year>.tif ==="
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
log "Total masks:     ${TOTAL_MASKS_DIR}/mascara_total_<year>.tif"
log "Filtered tiles:  ${FILTERED_DIR}"
log "Next (manual, when area rules are defined): polygonize → histograms → threshold — see filtering/README.md"
