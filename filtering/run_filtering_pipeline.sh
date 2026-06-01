#!/usr/bin/env bash
# =============================================================================
# MapBiomas Fire — filtrado por clases LULC (solo orquestación; scripts Python sin cambios)
#
# Leftraru (interactivo, sin sbatch):
#   cd ~/fire && git checkout feat/filtering_pipeline
#   bash filtering/run_filtering_pipeline.sh
#
# Opcional: filtering/cluster_paths.env sobreescribe las rutas por defecto.
# Cola SLURM: sbatch filtering/run_filtering_pipeline_slurm.sh
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

# --- Rutas por defecto (leftraru / flepin) — override con cluster_paths.env ---
PYTHON="${PYTHON:-/home/flepin/.conda/envs/mb_fuego/bin/python}"
LULC_STACK="${LULC_STACK:-/home/flepin/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-/home/flepin/classification_20260528}"
WORK_ROOT="${WORK_ROOT:-/home/flepin/filtering_work_local}"

MASCARAS_ROOT="${MASCARAS_ROOT:-${WORK_ROOT}/mascaras}"
ACCUMULATED_DIR="${ACCUMULATED_DIR:-${MASCARAS_ROOT}/acumuladas}"
YEARLY_MASKS_DIR="${YEARLY_MASKS_DIR:-${MASCARAS_ROOT}/by_year}"
TOTAL_MASKS_DIR="${TOTAL_MASKS_DIR:-${MASCARAS_ROOT}/totales}"
FILTERED_DIR="${FILTERED_DIR:-${WORK_ROOT}/classified_filtered}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2025}"
LULC_TO_YEAR="${LULC_TO_YEAR:-2024}"
START_YEAR_BAND1="${START_YEAR_BAND1:-2000}"
COPY_MASK_2025_FROM_2024="${COPY_MASK_2025_FROM_2024:-1}"
WORKERS="${WORKERS:-4}"
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

if [[ ! -e "${LULC_STACK}" ]]; then
  echo "ERROR: LULC_STACK not found: ${LULC_STACK}" >&2
  exit 1
fi

if [[ ! -d "${CLASSIFIED_DIR}" ]]; then
  echo "ERROR: CLASSIFIED_DIR not found: ${CLASSIFIED_DIR}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}/logs" "${ACCUMULATED_DIR}" "${YEARLY_MASKS_DIR}" "${TOTAL_MASKS_DIR}" "${FILTERED_DIR}"
cd "${REPO_ROOT}"

log "REPO_ROOT=${REPO_ROOT}"
if [[ "${LULC_STACK}" == *.vrt || "${LULC_STACK}" == *.VRT ]]; then
  echo "ERROR: LULC_STACK must be a single GeoTIFF (.tif), not VRT: ${LULC_STACK}" >&2
  exit 1
fi
log "LULC_STACK=${LULC_STACK} (band 1 = ${START_YEAR_BAND1})"
log "CLASSIFIED_DIR=${CLASSIFIED_DIR}"
log "WORK_ROOT=${WORK_ROOT}"
log "STEPS=${STEPS}"
log "LULC mask years: ${FROM_YEAR}-${LULC_TO_YEAR} | Filter/classified years: ${FROM_YEAR}-${TO_YEAR}"

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
    --to-year "${LULC_TO_YEAR}" \
    --workers "${WORKERS}"
fi

if [[ "${COPY_MASK_2025_FROM_2024}" == "1" ]] && step_enabled "masks_yearly"; then
  log "=== Copy 2024 yearly masks → 2025 (LULC sin banda 2025) ==="
  for stem in rio_lago infraestructura agricultura pastura; do
    src="${YEARLY_MASKS_DIR}/mascara_${stem}_2024.tif"
    dst="${YEARLY_MASKS_DIR}/mascara_${stem}_2025.tif"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: Missing ${src}" >&2
      exit 1
    fi
    cp -f "${src}" "${dst}"
    log "Copied: $(basename "${dst}")"
  done
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
