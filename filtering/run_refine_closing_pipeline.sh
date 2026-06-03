#!/usr/bin/env bash
# MapBiomas Fire — pilot pipeline: gentle morphological closing on burn masks
# Documentación: filtering/README.md (§ 3b)
#
# Input: rasters already filtered (LULC + temporal), e.g. classified_filtered/
# Output: classified_refined/ with *_closed.tif
#
#   cp filtering/cluster_paths.refine_closing.env.example filtering/cluster_paths.env
#   # or reuse filtering/cluster_paths.env and set REFINE_* variables
#   source filtering/cluster_paths.env
#   bash filtering/run_refine_closing_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${REFINE_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"

# Shell exports win over cluster_paths.env (so inline export ... && bash works).
_preserve_PYTHON="${PYTHON:-}"
_preserve_WORK_ROOT="${WORK_ROOT:-}"
_preserve_REFINE_INPUT_DIR="${REFINE_INPUT_DIR:-}"
_preserve_REFINE_OUTPUT_DIR="${REFINE_OUTPUT_DIR:-}"
_preserve_MAX_HOLE_AREA="${MAX_HOLE_AREA:-}"
_preserve_REFINE_METHOD="${REFINE_METHOD:-}"
_preserve_CLOSING_SIZE="${CLOSING_SIZE:-}"
_preserve_CLOSING_ITERATIONS="${CLOSING_ITERATIONS:-}"
_preserve_REFINE_OUTPUT_SUFFIX="${REFINE_OUTPUT_SUFFIX:-}"
_preserve_REFINE_NAME_CONTAINS="${REFINE_NAME_CONTAINS:-}"
_preserve_WORKERS="${WORKERS:-}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

[[ -n "${_preserve_PYTHON}" ]] && PYTHON="${_preserve_PYTHON}"
[[ -n "${_preserve_WORK_ROOT}" ]] && WORK_ROOT="${_preserve_WORK_ROOT}"
[[ -n "${_preserve_REFINE_INPUT_DIR}" ]] && REFINE_INPUT_DIR="${_preserve_REFINE_INPUT_DIR}"
[[ -n "${_preserve_REFINE_OUTPUT_DIR}" ]] && REFINE_OUTPUT_DIR="${_preserve_REFINE_OUTPUT_DIR}"
[[ -n "${_preserve_MAX_HOLE_AREA}" ]] && MAX_HOLE_AREA="${_preserve_MAX_HOLE_AREA}"
[[ -n "${_preserve_REFINE_METHOD}" ]] && REFINE_METHOD="${_preserve_REFINE_METHOD}"
[[ -n "${_preserve_CLOSING_SIZE}" ]] && CLOSING_SIZE="${_preserve_CLOSING_SIZE}"
[[ -n "${_preserve_CLOSING_ITERATIONS}" ]] && CLOSING_ITERATIONS="${_preserve_CLOSING_ITERATIONS}"
[[ -n "${_preserve_REFINE_OUTPUT_SUFFIX}" ]] && REFINE_OUTPUT_SUFFIX="${_preserve_REFINE_OUTPUT_SUFFIX}"
[[ -n "${_preserve_REFINE_NAME_CONTAINS}" ]] && REFINE_NAME_CONTAINS="${_preserve_REFINE_NAME_CONTAINS}"
[[ -n "${_preserve_WORKERS}" ]] && WORKERS="${_preserve_WORKERS}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  fi
fi

WORK_ROOT="${WORK_ROOT:-}"
REFINE_INPUT_DIR="${REFINE_INPUT_DIR:-}"
REFINE_OUTPUT_DIR="${REFINE_OUTPUT_DIR:-${WORK_ROOT}/classified_refined}"
REFINE_METHOD="${REFINE_METHOD:-fill_holes}"
MAX_HOLE_AREA="${MAX_HOLE_AREA:-16}"
CLOSING_SIZE="${CLOSING_SIZE:-2}"
CLOSING_ITERATIONS="${CLOSING_ITERATIONS:-1}"
REFINE_OUTPUT_SUFFIX="${REFINE_OUTPUT_SUFFIX:-_filled}"
REFINE_NAME_CONTAINS="${REFINE_NAME_CONTAINS:-}"
WORKERS="${WORKERS:-4}"
BURN_VALUE="${BURN_VALUE:-1}"
FILL_VALUE="${FILL_VALUE:-0}"

CONFIG_HINT="Set variables in ${PATHS_FILE} (see cluster_paths.refine_closing.env.example) or export them."

log() { echo "[$(date -Iseconds)] $*"; }

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} is not set. ${CONFIG_HINT}" >&2
    exit 1
  fi
}

run_py() {
  log "RUN: $*"
  "${PYTHON}" "$@"
}

require_var PYTHON
if [[ "${PYTHON}" == *"/path/to/"* ]]; then
  echo "ERROR: PYTHON is still the example placeholder in ${PATHS_FILE}." >&2
  echo "Edit that file or export PYTHON before running." >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not found: ${PYTHON}" >&2
  exit 1
fi

require_var WORK_ROOT
require_var REFINE_INPUT_DIR

if [[ ! -d "${REFINE_INPUT_DIR}" ]]; then
  echo "ERROR: REFINE_INPUT_DIR not found: ${REFINE_INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}/logs" "${REFINE_OUTPUT_DIR}"
cd "${REPO_ROOT}"

log "REPO_ROOT=${REPO_ROOT}"
log "PYTHON=${PYTHON}"
log "REFINE_INPUT_DIR=${REFINE_INPUT_DIR}"
log "REFINE_OUTPUT_DIR=${REFINE_OUTPUT_DIR}"
log "Method: ${REFINE_METHOD}"
if [[ "${REFINE_METHOD}" == "fill_holes" || "${REFINE_METHOD}" == "both" ]]; then
  if [[ "${MAX_HOLE_AREA}" == "0" ]]; then
    log "fill_holes: unlimited (all enclosed holes)"
  else
    log "fill_holes: max hole area = ${MAX_HOLE_AREA} px"
  fi
fi
if [[ "${REFINE_METHOD}" == "closing" || "${REFINE_METHOD}" == "both" ]]; then
  log "Closing: ${CLOSING_SIZE}x${CLOSING_SIZE}, iterations=${CLOSING_ITERATIONS}"
fi
if [[ -n "${REFINE_NAME_CONTAINS}" ]]; then
  log "Name filter: ${REFINE_NAME_CONTAINS}"
fi

REFINE_ARGS=(
  filtering/refine_burn_mask_closing.py
  --input-dir "${REFINE_INPUT_DIR}"
  --output-dir "${REFINE_OUTPUT_DIR}"
  --method "${REFINE_METHOD}"
  --max-hole-area "${MAX_HOLE_AREA}"
  --closing-size "${CLOSING_SIZE}"
  --iterations "${CLOSING_ITERATIONS}"
  --output-stem-suffix "${REFINE_OUTPUT_SUFFIX}"
  --burn-value "${BURN_VALUE}"
  --fill-value "${FILL_VALUE}"
  --workers "${WORKERS}"
  --stats-json "${WORK_ROOT}/logs/refine_stats.json"
)

if [[ -n "${REFINE_NAME_CONTAINS}" ]]; then
  REFINE_ARGS+=(--name-contains "${REFINE_NAME_CONTAINS}")
fi

log "=== Burn mask refine pipeline (pilot) ==="
run_py "${REFINE_ARGS[@]}"

log "=== Finished ==="
log "Output: ${REFINE_OUTPUT_DIR}"
