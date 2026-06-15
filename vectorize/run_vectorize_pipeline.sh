#!/usr/bin/env bash
# MapBiomas Fire — pipeline auxiliar de vectorización
# Documentación: vectorize/README.md
#
# Entrada: rasters ya filtrados (classified_filtered/)
# Salida: polígonos GeoPackage en VECTORIZE_OUTPUT_DIR
#
#   cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
#   nano vectorize/cluster_paths.env
#   source vectorize/cluster_paths.env
#   bash vectorize/run_vectorize_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${VECTORIZE_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"

_preserve_PYTHON="${PYTHON:-}"
_preserve_WORK_ROOT="${WORK_ROOT:-}"
_preserve_VECTORIZE_INPUT_DIR="${VECTORIZE_INPUT_DIR:-}"
_preserve_VECTORIZE_OUTPUT_DIR="${VECTORIZE_OUTPUT_DIR:-}"
_preserve_VECTORIZE_WORKERS="${VECTORIZE_WORKERS:-}"
_preserve_VECTORIZE_MASK_VALUE="${VECTORIZE_MASK_VALUE:-}"
_preserve_VECTORIZE_CONNECTIVITY="${VECTORIZE_CONNECTIVITY:-}"
_preserve_VECTORIZE_NAME_CONTAINS="${VECTORIZE_NAME_CONTAINS:-}"
_preserve_VECTORIZE_MERGED_GPKG="${VECTORIZE_MERGED_GPKG:-}"
_preserve_VECTORIZE_STATS_JSON="${VECTORIZE_STATS_JSON:-}"
_preserve_VECTORIZE_SIEVE_MIN_PIXELS="${VECTORIZE_SIEVE_MIN_PIXELS:-}"
_preserve_VECTORIZE_SKIP_SIEVE="${VECTORIZE_SKIP_SIEVE:-}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

[[ -n "${_preserve_PYTHON}" ]] && PYTHON="${_preserve_PYTHON}"
[[ -n "${_preserve_WORK_ROOT}" ]] && WORK_ROOT="${_preserve_WORK_ROOT}"
[[ -n "${_preserve_VECTORIZE_INPUT_DIR}" ]] && VECTORIZE_INPUT_DIR="${_preserve_VECTORIZE_INPUT_DIR}"
[[ -n "${_preserve_VECTORIZE_OUTPUT_DIR}" ]] && VECTORIZE_OUTPUT_DIR="${_preserve_VECTORIZE_OUTPUT_DIR}"
[[ -n "${_preserve_VECTORIZE_WORKERS}" ]] && VECTORIZE_WORKERS="${_preserve_VECTORIZE_WORKERS}"
[[ -n "${_preserve_VECTORIZE_MASK_VALUE}" ]] && VECTORIZE_MASK_VALUE="${_preserve_VECTORIZE_MASK_VALUE}"
[[ -n "${_preserve_VECTORIZE_CONNECTIVITY}" ]] && VECTORIZE_CONNECTIVITY="${_preserve_VECTORIZE_CONNECTIVITY}"
[[ -n "${_preserve_VECTORIZE_NAME_CONTAINS}" ]] && VECTORIZE_NAME_CONTAINS="${_preserve_VECTORIZE_NAME_CONTAINS}"
[[ -n "${_preserve_VECTORIZE_MERGED_GPKG}" ]] && VECTORIZE_MERGED_GPKG="${_preserve_VECTORIZE_MERGED_GPKG}"
[[ -n "${_preserve_VECTORIZE_STATS_JSON}" ]] && VECTORIZE_STATS_JSON="${_preserve_VECTORIZE_STATS_JSON}"
[[ -n "${_preserve_VECTORIZE_SIEVE_MIN_PIXELS}" ]] && VECTORIZE_SIEVE_MIN_PIXELS="${_preserve_VECTORIZE_SIEVE_MIN_PIXELS}"
[[ -n "${_preserve_VECTORIZE_SKIP_SIEVE}" ]] && VECTORIZE_SKIP_SIEVE="${_preserve_VECTORIZE_SKIP_SIEVE}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  fi
fi

WORK_ROOT="${WORK_ROOT:-}"
VECTORIZE_INPUT_DIR="${VECTORIZE_INPUT_DIR:-${WORK_ROOT}/classified_filtered}"
VECTORIZE_OUTPUT_DIR="${VECTORIZE_OUTPUT_DIR:-${WORK_ROOT}/polygons}"
VECTORIZE_WORKERS="${VECTORIZE_WORKERS:-4}"
VECTORIZE_MASK_VALUE="${VECTORIZE_MASK_VALUE:-1}"
VECTORIZE_CONNECTIVITY="${VECTORIZE_CONNECTIVITY:-8}"
VECTORIZE_NAME_CONTAINS="${VECTORIZE_NAME_CONTAINS:-}"
VECTORIZE_MERGED_GPKG="${VECTORIZE_MERGED_GPKG:-}"
VECTORIZE_STATS_JSON="${VECTORIZE_STATS_JSON:-${WORK_ROOT}/logs/vectorize_stats.json}"
VECTORIZE_SIEVE_MIN_PIXELS="${VECTORIZE_SIEVE_MIN_PIXELS:-112}"
VECTORIZE_SKIP_SIEVE="${VECTORIZE_SKIP_SIEVE:-0}"

CONFIG_HINT="Set variables in ${PATHS_FILE} (copy from cluster_paths.env.example) or export them."

log() { echo "[$(date -Iseconds)] $*"; }

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} is not set. ${CONFIG_HINT}" >&2
    exit 1
  fi
}

require_var PYTHON
if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not found or not executable: ${PYTHON}" >&2
  exit 1
fi

require_var VECTORIZE_INPUT_DIR
if [[ ! -d "${VECTORIZE_INPUT_DIR}" ]]; then
  echo "ERROR: VECTORIZE_INPUT_DIR not found: ${VECTORIZE_INPUT_DIR}" >&2
  echo "Run the filtering pipeline first (filtering/run_filtering_pipeline.sh)." >&2
  exit 1
fi

mkdir -p "${VECTORIZE_OUTPUT_DIR}" "$(dirname "${VECTORIZE_STATS_JSON}")"
cd "${REPO_ROOT}"

log "REPO_ROOT=${REPO_ROOT}"
log "PYTHON=${PYTHON}"
log "VECTORIZE_INPUT_DIR=${VECTORIZE_INPUT_DIR}"
log "VECTORIZE_OUTPUT_DIR=${VECTORIZE_OUTPUT_DIR}"
log "VECTORIZE_WORKERS=${VECTORIZE_WORKERS}"
log "SIEVE_MIN_PIXELS=${VECTORIZE_SIEVE_MIN_PIXELS:-off}"

ARGS=(
  lib/vectorize_filtered_classified.py
  --input-dir "${VECTORIZE_INPUT_DIR}"
  --output-dir "${VECTORIZE_OUTPUT_DIR}"
  --workers "${VECTORIZE_WORKERS}"
  --mask-value "${VECTORIZE_MASK_VALUE}"
  --connectivity "${VECTORIZE_CONNECTIVITY}"
  --stats-json "${VECTORIZE_STATS_JSON}"
)

if [[ "${VECTORIZE_SKIP_SIEVE}" == "1" ]]; then
  ARGS+=(--skip-sieve)
elif [[ -n "${VECTORIZE_SIEVE_MIN_PIXELS}" ]]; then
  ARGS+=(--sieve-min-pixels "${VECTORIZE_SIEVE_MIN_PIXELS}")
fi

if [[ -n "${VECTORIZE_NAME_CONTAINS}" ]]; then
  ARGS+=(--name-contains "${VECTORIZE_NAME_CONTAINS}")
  log "Name filter: ${VECTORIZE_NAME_CONTAINS}"
fi
if [[ -n "${VECTORIZE_MERGED_GPKG}" ]]; then
  ARGS+=(--merged-gpkg "${VECTORIZE_MERGED_GPKG}")
  log "Merged GPKG: ${VECTORIZE_MERGED_GPKG}"
fi

log "RUN: ${PYTHON} ${ARGS[*]}"
"${PYTHON}" "${ARGS[@]}"

log "=== Vectorize pipeline finished ==="
log "Polygons: ${VECTORIZE_OUTPUT_DIR}"
log "Stats:    ${VECTORIZE_STATS_JSON}"
