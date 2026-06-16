#!/usr/bin/env bash
# MapBiomas Fire — pipeline de filtrado post-clasificación
# Documentación: filtering/README.md
#
# Configuración (obligatoria): copiar y editar cluster_paths.env, luego:
#   source filtering/cluster_paths.env
#   bash filtering/run_filtering_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${SCRIPT_DIR}/cluster_paths.env"

_preserve_STEPS="${STEPS:-}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

[[ -n "${_preserve_STEPS}" ]] && STEPS="${_preserve_STEPS}"

# --- Rutas y runtime (sin defaults personales; definir en cluster_paths.env o export) ---
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  fi
fi

LULC_STACK="${LULC_STACK:-}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-}"
WORK_ROOT="${WORK_ROOT:-}"

MASCARAS_ROOT="${MASCARAS_ROOT:-${WORK_ROOT}/mascaras}"
ACCUMULATED_DIR="${ACCUMULATED_DIR:-${MASCARAS_ROOT}/acumuladas}"
YEARLY_MASKS_DIR="${YEARLY_MASKS_DIR:-${MASCARAS_ROOT}/by_year}"
TOTAL_MASKS_DIR="${TOTAL_MASKS_DIR:-${MASCARAS_ROOT}/totales}"

FILTER_OUTPUT_DIR="${FILTER_OUTPUT_DIR:-${WORK_ROOT}/classified_filtered}"
TEMPORAL_INTERMEDIATE_DIR="${TEMPORAL_INTERMEDIATE_DIR:-${WORK_ROOT}/classified_temporal}"
FILL_INTERMEDIATE_DIR="${FILL_INTERMEDIATE_DIR:-${WORK_ROOT}/classified_filled}"
KEEP_TEMPORAL_INTERMEDIATE="${KEEP_TEMPORAL_INTERMEDIATE:-0}"
KEEP_FILL_INTERMEDIATE="${KEEP_FILL_INTERMEDIATE:-0}"
LULC_INPUT_DIR="${LULC_INPUT_DIR:-}"

FILL_HOLES="${FILL_HOLES:-1}"
SKIP_FILL_HOLES="${SKIP_FILL_HOLES:-0}"
MAX_HOLE_AREA="${MAX_HOLE_AREA:-0}"
FILL_METHOD="${FILL_METHOD:-fill_holes}"

MIN_PATCH_SIEVE_PIXELS="${MIN_PATCH_SIEVE_PIXELS:-}"
MIN_PATCH_SIEVE_HA="${MIN_PATCH_SIEVE_HA:-}"
SKIP_MIN_PATCH_SIEVE="${SKIP_MIN_PATCH_SIEVE:-0}"
MIN_PATCH_CONNECTIVITY="${MIN_PATCH_CONNECTIVITY:-8}"
LULC_INTERMEDIATE_DIR="${LULC_INTERMEDIATE_DIR:-${WORK_ROOT}/classified_lulc}"

TEMPORAL_SUFFIX="${TEMPORAL_SUFFIX:-_first_burn_year}"
TEMPORAL_SPATIAL_MERGE="${TEMPORAL_SPATIAL_MERGE:-0}"
TEMPORAL_CONNECTIVITY="${TEMPORAL_CONNECTIVITY:-8}"
TEMPORAL_YEAR_TOKEN_INDEX="${TEMPORAL_YEAR_TOKEN_INDEX:-3}"
FILTER_NAME_CONTAINS="${FILTER_NAME_CONTAINS:-${TEMPORAL_NAME_CONTAINS:-}}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2025}"
LULC_TO_YEAR="${LULC_TO_YEAR:-2024}"
START_YEAR_BAND1="${START_YEAR_BAND1:-2000}"
COPY_MASK_2025_FROM_2024="${COPY_MASK_2025_FROM_2024:-1}"
LULC_STABILITY_WINDOW="${LULC_STABILITY_WINDOW:-4}"
LULC_AGRICULTURE_STABILITY_WINDOW="${LULC_AGRICULTURE_STABILITY_WINDOW:-}"
FILL_AGRICULTURAL_HOLES="${FILL_AGRICULTURAL_HOLES:-0}"
AGRICULTURE_MASKS_DIR="${AGRICULTURE_MASKS_DIR:-${YEARLY_MASKS_DIR}}"
WORKERS="${WORKERS:-4}"
FILL_VALUE="${FILL_VALUE:-0}"
STEPS="${STEPS:-all}"

DEFAULT_STEPS="masks_accumulated,masks_yearly,masks_total,filter"

CONFIG_HINT="Set variables in ${PATHS_FILE} (copy from cluster_paths.env.example) or export them before running."

log() { echo "[$(date -Iseconds)] $*"; }

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} is not set. ${CONFIG_HINT}" >&2
    exit 1
  fi
}

step_enabled() {
  local name="$1"
  if [[ "${STEPS}" == "all" ]]; then
    [[ ",${DEFAULT_STEPS}," == *",${name},"* ]]
    return
  fi
  [[ ",${STEPS}," == *",${name},"* ]]
}

steps_need_masks() {
  step_enabled "masks_accumulated" || step_enabled "masks_yearly" || step_enabled "masks_total"
}

steps_need_classified() {
  step_enabled "filter" || step_enabled "temporal_first_burn" || step_enabled "fill_holes" || step_enabled "lulc_filter" || step_enabled "min_patch_sieve" || step_enabled "fill_agricultural_holes"
}

steps_need_raw_classified_dir() {
  step_enabled "filter" || step_enabled "temporal_first_burn"
}

run_py() {
  log "RUN: $*"
  "${PYTHON}" "$@"
}

run_unified_filter() {
  local mode="${1:-full}"
  local classified_input="${CLASSIFIED_DIR}"
  local output_dir="${FILTER_OUTPUT_DIR}"
  local extra_args=()

  case "${mode}" in
    full)
      log "=== Filter: temporal → hole fill → LULC → optional ag fill (unified) ==="
      ;;
    temporal)
      log "=== Filter: temporal first-burn only ==="
      output_dir="${TEMPORAL_INTERMEDIATE_DIR}"
      extra_args+=(--temporal-only)
      ;;
    fill_holes)
      log "=== Filter: internal hole fill only ==="
      classified_input="${FILL_INPUT_DIR:-${TEMPORAL_INTERMEDIATE_DIR}}"
      output_dir="${FILL_INTERMEDIATE_DIR}"
      extra_args+=(--fill-only)
      ;;
    lulc)
      log "=== Filter: LULC masks only ==="
      classified_input="${LULC_INPUT_DIR:-${FILL_INTERMEDIATE_DIR}}"
      if [[ ! -d "${classified_input}" ]]; then
        classified_input="${TEMPORAL_INTERMEDIATE_DIR}"
      fi
      extra_args+=(--lulc-only)
      ;;
    fill_ag)
      log "=== Filter: agricultural hole fill only (post-LULC) ==="
      classified_input="${AG_FILL_INPUT_DIR:-${LULC_INTERMEDIATE_DIR}}"
      if [[ ! -d "${classified_input}" ]]; then
        classified_input="${FILTER_OUTPUT_DIR}"
      fi
      extra_args+=(--ag-fill-only)
      ;;
    min_patch)
      log "=== Filter: min-patch sieve only (after LULC) ==="
      classified_input="${MIN_PATCH_INPUT_DIR:-${LULC_INTERMEDIATE_DIR}}"
      if [[ ! -d "${classified_input}" ]]; then
        classified_input="${FILTER_OUTPUT_DIR}"
      fi
      extra_args+=(--min-patch-only)
      ;;
    *)
      echo "ERROR: unknown filter mode: ${mode}" >&2
      exit 1
      ;;
  esac

  if [[ "${mode}" == "fill_holes" && ! -d "${classified_input}" ]]; then
    echo "ERROR: Temporal input not found (run temporal step first): ${classified_input}" >&2
    exit 1
  fi

  if [[ "${mode}" == "lulc" && ! -d "${classified_input}" ]]; then
    echo "ERROR: Input for LULC step not found: ${classified_input}" >&2
    exit 1
  fi

  if [[ "${mode}" == "fill_ag" && ! -d "${classified_input}" ]]; then
    echo "ERROR: Input for agricultural hole fill not found (run LULC first): ${classified_input}" >&2
    exit 1
  fi

  if [[ "${mode}" == "min_patch" && ! -d "${classified_input}" ]]; then
    echo "ERROR: Input for min-patch sieve not found (run LULC first): ${classified_input}" >&2
    exit 1
  fi

  FILTER_ARGS=(
    filtering/run_classified_filters.py
    --classified-dir "${classified_input}"
    --masks-dir "${TOTAL_MASKS_DIR}"
    --output-dir "${output_dir}"
    --temporal-intermediate-dir "${TEMPORAL_INTERMEDIATE_DIR}"
    --fill-intermediate-dir "${FILL_INTERMEDIATE_DIR}"
    --from-year "${FROM_YEAR}"
    --to-year "${TO_YEAR}"
    --fill-value "${FILL_VALUE}"
    --workers "${WORKERS}"
    --temporal-suffix "${TEMPORAL_SUFFIX}"
    --year-token-index "${TEMPORAL_YEAR_TOKEN_INDEX}"
    --connectivity "${TEMPORAL_CONNECTIVITY}"
    --max-hole-area "${MAX_HOLE_AREA}"
    --fill-method "${FILL_METHOD}"
    --stats-json "${WORK_ROOT}/logs/filter_stats.json"
    --fill-stats-json "${WORK_ROOT}/logs/fill_stats.json"
    --lulc-intermediate-dir "${LULC_INTERMEDIATE_DIR}"
    --min-patch-stats-json "${WORK_ROOT}/logs/min_patch_stats.json"
    --ag-fill-stats-json "${WORK_ROOT}/logs/ag_fill_stats.json"
  )
  if [[ "${TEMPORAL_SPATIAL_MERGE}" == "1" ]]; then
    FILTER_ARGS+=(--spatial-merge)
  else
    FILTER_ARGS+=(--no-spatial-merge)
  fi
  if [[ "${KEEP_TEMPORAL_INTERMEDIATE}" == "1" ]]; then
    FILTER_ARGS+=(--keep-temporal-intermediate)
  fi
  if [[ "${KEEP_FILL_INTERMEDIATE}" == "1" ]]; then
    FILTER_ARGS+=(--keep-fill-intermediate)
  fi
  if [[ "${SKIP_FILL_HOLES}" == "1" || "${FILL_HOLES}" == "0" ]]; then
    FILTER_ARGS+=(--skip-fill)
  fi
  if [[ "${SKIP_MIN_PATCH_SIEVE}" == "1" ]]; then
    FILTER_ARGS+=(--skip-min-patch)
  elif [[ -n "${MIN_PATCH_SIEVE_PIXELS}" ]]; then
    FILTER_ARGS+=(--min-patch-min-pixels "${MIN_PATCH_SIEVE_PIXELS}")
  elif [[ -n "${MIN_PATCH_SIEVE_HA}" ]]; then
    FILTER_ARGS+=(--min-patch-min-ha "${MIN_PATCH_SIEVE_HA}")
  else
    FILTER_ARGS+=(--skip-min-patch)
  fi
  if [[ -n "${MIN_PATCH_CONNECTIVITY}" ]]; then
    FILTER_ARGS+=(--min-patch-connectivity "${MIN_PATCH_CONNECTIVITY}")
  fi
  if [[ "${FILL_AGRICULTURAL_HOLES}" == "1" ]]; then
    FILTER_ARGS+=(--fill-agricultural-holes)
    FILTER_ARGS+=(--agriculture-masks-dir "${AGRICULTURE_MASKS_DIR}")
  fi
  if [[ -n "${FILTER_NAME_CONTAINS}" ]]; then
    FILTER_ARGS+=(--name-contains "${FILTER_NAME_CONTAINS}")
    log "Name filter: ${FILTER_NAME_CONTAINS}"
  fi
  FILTER_ARGS+=("${extra_args[@]}")

  log "Classified input: ${classified_input}"
  log "Filter output:    ${output_dir}"
  if [[ "${mode}" == "full" ]]; then
    log "Temporal intermediate: ${TEMPORAL_INTERMEDIATE_DIR} (keep=${KEEP_TEMPORAL_INTERMEDIATE})"
    log "Fill intermediate:     ${FILL_INTERMEDIATE_DIR} (keep=${KEEP_FILL_INTERMEDIATE})"
    if [[ "${SKIP_FILL_HOLES}" == "1" || "${FILL_HOLES}" == "0" ]]; then
      log "Hole fill: disabled"
    else
      log "Hole fill: ${FILL_METHOD}, max_hole_area=${MAX_HOLE_AREA} (0=unlimited)"
    fi
    if [[ "${SKIP_MIN_PATCH_SIEVE}" == "1" ]]; then
      log "Min-patch sieve: disabled"
    elif [[ -n "${MIN_PATCH_SIEVE_PIXELS}" ]]; then
      log "Min-patch sieve: ${MIN_PATCH_SIEVE_PIXELS} px (after LULC)"
    elif [[ -n "${MIN_PATCH_SIEVE_HA}" ]]; then
      log "Min-patch sieve: ${MIN_PATCH_SIEVE_HA} ha (after LULC)"
    else
      log "Min-patch sieve: not configured (set MIN_PATCH_SIEVE_PIXELS or MIN_PATCH_SIEVE_HA)"
    fi
    if [[ "${FILL_AGRICULTURAL_HOLES}" == "1" ]]; then
      log "Agricultural hole fill: enabled (masks=${AGRICULTURE_MASKS_DIR})"
    else
      log "Agricultural hole fill: disabled"
    fi
  fi
  run_py "${FILTER_ARGS[@]}"
}

# --- Validar configuración según STEPS ---
require_var PYTHON
if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not found or not executable: ${PYTHON}" >&2
  exit 1
fi

require_var WORK_ROOT

if steps_need_masks; then
  require_var LULC_STACK
fi

if steps_need_raw_classified_dir; then
  require_var CLASSIFIED_DIR
fi

if { step_enabled "filter" || step_enabled "lulc_filter"; } && ! steps_need_masks; then
  if [[ ! -d "${TOTAL_MASKS_DIR}" ]]; then
    echo "ERROR: Masks dir not found (run mask steps first): ${TOTAL_MASKS_DIR}" >&2
    exit 1
  fi
fi

if steps_need_raw_classified_dir && [[ ! -d "${CLASSIFIED_DIR}" ]]; then
  echo "ERROR: CLASSIFIED_DIR not found: ${CLASSIFIED_DIR}" >&2
  exit 1
fi

if steps_need_masks; then
  if [[ ! -e "${LULC_STACK}" ]]; then
    echo "ERROR: LULC_STACK not found: ${LULC_STACK}" >&2
    exit 1
  fi
  if [[ "${LULC_STACK}" == *.vrt || "${LULC_STACK}" == *.VRT ]]; then
    echo "ERROR: LULC_STACK must be a single GeoTIFF (.tif), not VRT: ${LULC_STACK}" >&2
    exit 1
  fi
fi

mkdir -p "${WORK_ROOT}/logs" "${ACCUMULATED_DIR}" "${YEARLY_MASKS_DIR}" "${TOTAL_MASKS_DIR}" \
  "${FILTER_OUTPUT_DIR}" "${TEMPORAL_INTERMEDIATE_DIR}" "${FILL_INTERMEDIATE_DIR}"
cd "${REPO_ROOT}"

log "REPO_ROOT=${REPO_ROOT}"
log "PYTHON=${PYTHON}"
if steps_need_masks; then
  log "LULC_STACK=${LULC_STACK} (band 1 = ${START_YEAR_BAND1})"
fi
if [[ -n "${CLASSIFIED_DIR}" ]]; then
  log "CLASSIFIED_DIR=${CLASSIFIED_DIR}"
fi
log "WORK_ROOT=${WORK_ROOT}"
log "FILTER_OUTPUT_DIR=${FILTER_OUTPUT_DIR}"
log "STEPS=${STEPS}"
if steps_need_masks; then
  log "LULC mask years: ${FROM_YEAR}-${LULC_TO_YEAR} | Filter/classified years: ${FROM_YEAR}-${TO_YEAR}"
  log "LULC stability window (A2): ${LULC_STABILITY_WINDOW} years"
  if [[ -n "${LULC_AGRICULTURE_STABILITY_WINDOW}" ]]; then
    log "Agriculture stability window: ${LULC_AGRICULTURE_STABILITY_WINDOW} year(s)"
  fi
fi

if step_enabled "masks_accumulated"; then
  log "=== Step 1a: accumulated class masks ==="
  run_py filtering/create_accumulated_class_masks.py \
    --input-tif "${LULC_STACK}" \
    --output-dir "${ACCUMULATED_DIR}"
fi

if step_enabled "masks_yearly"; then
  log "=== Step 1b: yearly thematic masks ==="
  YEARLY_MASK_ARGS=(
    filtering/create_yearly_masks.py
    --input-tif "${LULC_STACK}"
    --output-dir "${YEARLY_MASKS_DIR}"
    --start-year-in-band-1 "${START_YEAR_BAND1}"
    --from-year "${FROM_YEAR}"
    --to-year "${LULC_TO_YEAR}"
    --stability-window "${LULC_STABILITY_WINDOW}"
    --workers "${WORKERS}"
  )
  if [[ -n "${LULC_AGRICULTURE_STABILITY_WINDOW}" ]]; then
    YEARLY_MASK_ARGS+=(--agriculture-stability-window "${LULC_AGRICULTURE_STABILITY_WINDOW}")
  fi
  run_py "${YEARLY_MASK_ARGS[@]}"
fi

if step_enabled "masks_yearly" && [[ "${TO_YEAR}" -gt "${LULC_TO_YEAR}" ]]; then
  log "=== Yearly masks for ${TO_YEAR} (filter year > LULC_TO_YEAR=${LULC_TO_YEAR}) ==="
  if run_py filtering/create_yearly_masks.py \
    --input-tif "${LULC_STACK}" \
    --output-dir "${YEARLY_MASKS_DIR}" \
    --start-year-in-band-1 "${START_YEAR_BAND1}" \
    --from-year "${TO_YEAR}" \
    --to-year "${TO_YEAR}" \
    --stability-window "${LULC_STABILITY_WINDOW}" \
    ${LULC_AGRICULTURE_STABILITY_WINDOW:+--agriculture-stability-window "${LULC_AGRICULTURE_STABILITY_WINDOW}"} \
    --workers 1; then
    log "Built stability masks for filter year ${TO_YEAR}"
  elif [[ "${COPY_MASK_2025_FROM_2024}" == "1" ]]; then
    log "WARN: Could not build ${TO_YEAR} masks from LULC stack; copying ${LULC_TO_YEAR} → ${TO_YEAR} (legacy fallback)"
    for stem in rio_lago infraestructura agricultura pastura; do
      src="${YEARLY_MASKS_DIR}/mascara_${stem}_${LULC_TO_YEAR}.tif"
      dst="${YEARLY_MASKS_DIR}/mascara_${stem}_${TO_YEAR}.tif"
      if [[ ! -f "${src}" ]]; then
        echo "ERROR: Missing ${src}" >&2
        exit 1
      fi
      cp -f "${src}" "${dst}"
      log "Copied: $(basename "${dst}")"
    done
  else
    echo "ERROR: No LULC band for filter year ${TO_YEAR} and COPY_MASK_2025_FROM_2024=0" >&2
    exit 1
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
  run_unified_filter full
fi

if step_enabled "temporal_first_burn"; then
  run_unified_filter temporal
fi

if step_enabled "fill_holes"; then
  run_unified_filter fill_holes
fi

if step_enabled "lulc_filter"; then
  run_unified_filter lulc
fi

if step_enabled "min_patch_sieve"; then
  run_unified_filter min_patch
fi

if step_enabled "fill_agricultural_holes"; then
  run_unified_filter fill_ag
fi

log "=== Pipeline finished ==="
log "Accumulated masks: ${ACCUMULATED_DIR}"
log "Yearly masks:      ${YEARLY_MASKS_DIR}"
log "Total masks:       ${TOTAL_MASKS_DIR}/mascara_total_<year>.tif"
if step_enabled "filter" || step_enabled "lulc_filter"; then
  log "Filtered output:   ${FILTER_OUTPUT_DIR}"
fi
if [[ "${KEEP_TEMPORAL_INTERMEDIATE}" == "1" ]] && { step_enabled "filter" || step_enabled "temporal_first_burn"; }; then
  log "Temporal intermediate: ${TEMPORAL_INTERMEDIATE_DIR}"
fi
if [[ "${KEEP_FILL_INTERMEDIATE}" == "1" ]] && { step_enabled "filter" || step_enabled "fill_holes"; }; then
  log "Fill intermediate:     ${FILL_INTERMEDIATE_DIR}"
fi
