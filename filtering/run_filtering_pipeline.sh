#!/usr/bin/env bash
# =============================================================================
# MapBiomas Fire — filtrado por clases LULC (solo orquestación; scripts Python sin cambios)
#
# Configuración: filtering/cluster_paths.env (copiar desde cluster_paths.env.example)
#   source filtering/cluster_paths.env
#   bash filtering/run_filtering_pipeline.sh
#
# NLHPC: sbatch filtering/run_filtering_pipeline_slurm.sh
#
# STEPS: masks_accumulated | masks_yearly | masks_total | filter | all
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${SCRIPT_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/cluster_paths.env"
fi

PYTHON="${PYTHON:-python3}"

LULC_STACK="${LULC_STACK:-}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-}"
WORK_ROOT="${WORK_ROOT:-}"

MASCARAS_ROOT="${MASCARAS_ROOT:-${WORK_ROOT}/mascaras}"
ACCUMULATED_DIR="${ACCUMULATED_DIR:-${MASCARAS_ROOT}/acumuladas}"
YEARLY_MASKS_DIR="${YEARLY_MASKS_DIR:-${MASCARAS_ROOT}/by_year}"
TOTAL_MASKS_DIR="${TOTAL_MASKS_DIR:-${MASCARAS_ROOT}/totales}"
FILTERED_DIR="${FILTERED_DIR:-${WORK_ROOT}/classified_filtered}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2025}"
START_YEAR_BAND1="${START_YEAR_BAND1:-1999}"
WORKERS="${WORKERS:-16}"
FILL_VALUE="${FILL_VALUE:-0}"
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

if [[ -z "${LULC_STACK}" || ! -e "${LULC_STACK}" ]]; then
  echo "ERROR: LULC_STACK not set or missing: ${LULC_STACK:-<empty>}" >&2
  echo "Set it in filtering/cluster_paths.env" >&2
  exit 1
fi

if [[ -z "${CLASSIFIED_DIR}" || ! -d "${CLASSIFIED_DIR}" ]]; then
  echo "ERROR: CLASSIFIED_DIR not set or missing: ${CLASSIFIED_DIR:-<empty>}" >&2
  exit 1
fi

if [[ -z "${WORK_ROOT}" ]]; then
  echo "ERROR: WORK_ROOT not set" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}/logs" "${ACCUMULATED_DIR}" "${YEARLY_MASKS_DIR}" "${TOTAL_MASKS_DIR}" "${FILTERED_DIR}"
cd "${REPO_ROOT}"

log "REPO_ROOT=${REPO_ROOT}"
log "LULC_STACK=${LULC_STACK} (band 1 = ${START_YEAR_BAND1})"
log "CLASSIFIED_DIR=${CLASSIFIED_DIR}"
log "WORK_ROOT=${WORK_ROOT}"
log "STEPS=${STEPS}"

if step_enabled "masks_accumulated"; then
  log "=== Step 1a: accumulated class masks ==="
  run_py filtering/create_accumulated_class_masks.py \
    --input-tif "${LULC_STACK}" \
    --output-dir "${ACCUMULATED_DIR}"
fi

if step_enabled "masks_yearly"; then
  log "=== Step 1b: yearly thematic masks ==="
  run_py filtering/create_yearly_masks.py \
    --input-tif "${LULC_STACK}" \
    --output-dir "${YEARLY_MASKS_DIR}" \
    --start-year-in-band-1 "${START_YEAR_BAND1}" \
    --from-year "${FROM_YEAR}" \
    --to-year "${TO_YEAR}" \
    --workers "${WORKERS}"
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

log "=== Pipeline finished ==="
log "Accumulated masks: ${ACCUMULATED_DIR}"
log "Yearly masks:      ${YEARLY_MASKS_DIR}"
log "Total masks:       ${TOTAL_MASKS_DIR}/mascara_total_<year>.tif"
log "Filtered rasters:  ${FILTERED_DIR}"
