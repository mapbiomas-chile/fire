#!/usr/bin/env bash
# MapBiomas Fire — filtro de área en polígonos (post-vectorize, §5)
# Documentación: filtering/README.md §5
#
# Flujo en evaluación (dos pasos):
#   1) pre_filter: corte fijo ≥ POLYGON_PRE_FILTER_HA (ej. 20 ha)
#   2) histograms → recommend (calcula p5, p10, p25, elbow) → filter (aplica UNA regla)
#
#   cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
#   source filtering/cluster_paths.env
#   bash filtering/run_polygon_area_pipeline.sh
#
# STEPS (comma-separated, or "all"):
#   pre_filter, histograms, recommend, filter
#
# Comparar reglas sin recalcular umbrales:
#   for r in p5 p10 p25 elbow; do
#     POLYGON_THRESHOLD_RULE=$r STEPS=filter bash filtering/run_polygon_area_pipeline.sh
#   done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${FILTERING_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"

_preserve_STEPS="${STEPS:-}"
_preserve_WORK_ROOT="${WORK_ROOT:-}"
_preserve_POLYGON_THRESHOLD_RULE="${POLYGON_THRESHOLD_RULE:-}"
_preserve_POLYGON_FILTERED_GPKG="${POLYGON_FILTERED_GPKG:-}"
_preserve_POLYGON_FILTERED_DIR="${POLYGON_FILTERED_DIR:-}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

[[ -n "${_preserve_STEPS}" ]] && STEPS="${_preserve_STEPS}"
[[ -n "${_preserve_WORK_ROOT}" ]] && WORK_ROOT="${_preserve_WORK_ROOT}"
[[ -n "${_preserve_POLYGON_THRESHOLD_RULE}" ]] && POLYGON_THRESHOLD_RULE="${_preserve_POLYGON_THRESHOLD_RULE}"
[[ -n "${_preserve_POLYGON_FILTERED_GPKG}" ]] && POLYGON_FILTERED_GPKG="${_preserve_POLYGON_FILTERED_GPKG}"
[[ -n "${_preserve_POLYGON_FILTERED_DIR}" ]] && POLYGON_FILTERED_DIR="${_preserve_POLYGON_FILTERED_DIR}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  fi
fi

WORK_ROOT="${WORK_ROOT:-}"
POLYGON_INPUT_DIR="${POLYGON_INPUT_DIR:-${WORK_ROOT}/polygons}"
POLYGON_PRE_FILTER_HA="${POLYGON_PRE_FILTER_HA:-}"
POLYGON_PRE_FILTER_DIR="${POLYGON_PRE_FILTER_DIR:-}"
POLYGON_HISTOGRAMS_DIR="${POLYGON_HISTOGRAMS_DIR:-${WORK_ROOT}/histogramas_area}"
POLYGON_THRESHOLDS_DIR="${POLYGON_THRESHOLDS_DIR:-${WORK_ROOT}/thresholds_area}"
POLYGON_FILTERED_GPKG="${POLYGON_FILTERED_GPKG:-}"
POLYGON_FILTERED_DIR="${POLYGON_FILTERED_DIR:-}"
POLYGON_THRESHOLD_RULE="${POLYGON_THRESHOLD_RULE:-p25}"
POLYGON_THRESHOLD_HA="${POLYGON_THRESHOLD_HA:-}"
POLYGON_PER_REGION_YEAR="${POLYGON_PER_REGION_YEAR:-1}"
POLYGON_INPUT_PATTERN="${POLYGON_INPUT_PATTERN:-*.gpkg}"
POLYGON_HISTOGRAM_BINS="${POLYGON_HISTOGRAM_BINS:-40}"

STEPS="${POLYGON_STEPS:-${STEPS:-all}}"

ha_tag=""
if [[ -n "${POLYGON_PRE_FILTER_HA}" ]]; then
  ha_tag="${POLYGON_PRE_FILTER_HA//./}"
fi

if [[ -z "${POLYGON_PRE_FILTER_DIR}" && -n "${POLYGON_PRE_FILTER_HA}" ]]; then
  POLYGON_PRE_FILTER_DIR="${WORK_ROOT}/polygons_min${ha_tag}ha"
fi

if [[ -n "${POLYGON_PRE_FILTER_HA}" ]]; then
  POLYGON_HISTOGRAMS_DIR="${POLYGON_HISTOGRAMS_DIR:-${WORK_ROOT}/histogramas_area_min${ha_tag}ha}"
  POLYGON_THRESHOLDS_DIR="${POLYGON_THRESHOLDS_DIR:-${WORK_ROOT}/thresholds_area_min${ha_tag}ha}"
  if [[ -z "${POLYGON_FILTERED_GPKG}" ]]; then
    POLYGON_FILTERED_GPKG="${WORK_ROOT}/polygons_filtered_min${ha_tag}ha_${POLYGON_THRESHOLD_RULE}.gpkg"
  fi
  if [[ -z "${POLYGON_FILTERED_DIR}" ]]; then
    POLYGON_FILTERED_DIR="${WORK_ROOT}/polygons_filtered_min${ha_tag}ha_${POLYGON_THRESHOLD_RULE}"
  fi
else
  POLYGON_FILTERED_GPKG="${POLYGON_FILTERED_GPKG:-${WORK_ROOT}/polygons_filtered.gpkg}"
  POLYGON_FILTERED_DIR="${POLYGON_FILTERED_DIR:-${WORK_ROOT}/polygons_filtered}"
fi

CONFIG_HINT="Set variables in ${PATHS_FILE} (copy from cluster_paths.20260619.env.leftraru) or export them."

log() { echo "[$(date -Iseconds)] $*"; }

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} is not set. ${CONFIG_HINT}" >&2
    exit 1
  fi
}

step_enabled() {
  local step="$1"
  if [[ "${STEPS}" == "all" ]]; then
    if [[ "${step}" == "pre_filter" ]]; then
      [[ -n "${POLYGON_PRE_FILTER_HA}" ]]
      return $?
    fi
    return 0
  fi
  [[ ",${STEPS}," == *",${step},"* ]]
}

analysis_input_dir() {
  if [[ -n "${POLYGON_PRE_FILTER_HA}" ]]; then
    if [[ -d "${POLYGON_PRE_FILTER_DIR}" ]]; then
      echo "${POLYGON_PRE_FILTER_DIR}"
      return
    fi
    if step_enabled "pre_filter"; then
      echo "${POLYGON_PRE_FILTER_DIR}"
      return
    fi
  fi
  echo "${POLYGON_INPUT_DIR}"
}

require_var PYTHON
if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not found: ${PYTHON}" >&2
  exit 1
fi

require_var WORK_ROOT
require_var POLYGON_INPUT_DIR

if [[ ! -d "${POLYGON_INPUT_DIR}" ]]; then
  echo "ERROR: POLYGON_INPUT_DIR not found: ${POLYGON_INPUT_DIR}" >&2
  echo "Run vectorize first (vectorize/run_vectorize_pipeline.sh)." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if step_enabled "pre_filter"; then
  if [[ -z "${POLYGON_PRE_FILTER_HA}" ]]; then
    echo "ERROR: pre_filter requires POLYGON_PRE_FILTER_HA (e.g. 20 for 20 ha)." >&2
    exit 1
  fi
  log "=== Polygon area: pre-filter >= ${POLYGON_PRE_FILTER_HA} ha ==="
  mkdir -p "${POLYGON_PRE_FILTER_DIR}"
  "${PYTHON}" filtering/filter_polygons_by_threshold.py \
    --input-dir "${POLYGON_INPUT_DIR}" \
    --output-dir "${POLYGON_PRE_FILTER_DIR}" \
    --output-gpkg "${POLYGON_PRE_FILTER_DIR}/polygons_pre_filter.gpkg" \
    --threshold-ha "${POLYGON_PRE_FILTER_HA}" \
    --pattern "${POLYGON_INPUT_PATTERN}"
  log "Pre-filtered polygons: ${POLYGON_PRE_FILTER_DIR}"
fi

ANALYSIS_DIR="$(analysis_input_dir)"

if step_enabled "histograms"; then
  log "=== Polygon area: histograms (input: ${ANALYSIS_DIR}) ==="
  mkdir -p "${POLYGON_HISTOGRAMS_DIR}"
  "${PYTHON}" filtering/summarize_histograms_by_region.py \
    --input-dir "${ANALYSIS_DIR}" \
    --output-dir "${POLYGON_HISTOGRAMS_DIR}" \
    --pattern "${POLYGON_INPUT_PATTERN}" \
    --bins "${POLYGON_HISTOGRAM_BINS}"
  log "Histograms: ${POLYGON_HISTOGRAMS_DIR}"
fi

if step_enabled "recommend"; then
  log "=== Polygon area: recommend thresholds p5/p10/p25/elbow (input: ${ANALYSIS_DIR}) ==="
  mkdir -p "${POLYGON_THRESHOLDS_DIR}"
  "${PYTHON}" filtering/recommend_polygon_area_thresholds.py \
    --input-dir "${ANALYSIS_DIR}" \
    --output-dir "${POLYGON_THRESHOLDS_DIR}" \
    --pattern "${POLYGON_INPUT_PATTERN}"
  log "Thresholds: ${POLYGON_THRESHOLDS_DIR}/threshold_summary.json"
  log "Review: ${POLYGON_THRESHOLDS_DIR}/thresholds_by_region_year.csv"
fi

if step_enabled "filter"; then
  log "=== Polygon area: percentile filter (input: ${ANALYSIS_DIR}) ==="
  FILTER_ARGS=(
    filtering/filter_polygons_by_threshold.py
    --input-dir "${ANALYSIS_DIR}"
    --pattern "${POLYGON_INPUT_PATTERN}"
  )

  if [[ -n "${POLYGON_THRESHOLD_HA}" ]]; then
    log "Manual threshold: ${POLYGON_THRESHOLD_HA} ha"
    FILTER_ARGS+=(--threshold-ha "${POLYGON_THRESHOLD_HA}")
  else
    local_json="${POLYGON_THRESHOLDS_DIR}/threshold_summary.json"
    if [[ ! -f "${local_json}" ]]; then
      echo "ERROR: ${local_json} not found. Run STEPS=recommend first." >&2
      exit 1
    fi
    log "Threshold rule: ${POLYGON_THRESHOLD_RULE} (per-region-year=${POLYGON_PER_REGION_YEAR})"
    FILTER_ARGS+=(
      --stats-summary-json "${local_json}"
      --threshold-rule "${POLYGON_THRESHOLD_RULE}"
    )
    if [[ "${POLYGON_PER_REGION_YEAR}" == "1" ]]; then
      FILTER_ARGS+=(--per-region-year)
    fi
  fi

  if [[ -n "${POLYGON_FILTERED_GPKG}" ]]; then
    mkdir -p "$(dirname "${POLYGON_FILTERED_GPKG}")"
    FILTER_ARGS+=(--output-gpkg "${POLYGON_FILTERED_GPKG}")
  fi
  if [[ -n "${POLYGON_FILTERED_DIR}" ]]; then
    mkdir -p "${POLYGON_FILTERED_DIR}"
    FILTER_ARGS+=(--output-dir "${POLYGON_FILTERED_DIR}")
  fi

  "${PYTHON}" "${FILTER_ARGS[@]}"
  log "Filtered output: ${POLYGON_FILTERED_GPKG:-${POLYGON_FILTERED_DIR}}"
fi

log "=== Polygon area pipeline finished ==="
