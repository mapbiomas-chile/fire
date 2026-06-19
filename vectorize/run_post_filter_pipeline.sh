#!/usr/bin/env bash
# MapBiomas Fire — post-filter: vectorize + filtro de área + vectorización nacional
# Documentación: classification/README.md (Production 20260619)
#
#   cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
#   cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
#   source vectorize/cluster_paths.env
#   bash vectorize/run_post_filter_pipeline.sh
#
# STEPS (comma-separated, or "all"):
#   vectorize, polygon_area, national

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VECTORIZE_PATHS="${VECTORIZE_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"
FILTERING_PATHS="${FILTERING_PATHS_FILE:-${REPO_ROOT}/filtering/cluster_paths.env}"

_preserve_STEPS="${STEPS:-}"

if [[ -f "${VECTORIZE_PATHS}" ]]; then
  # shellcheck source=/dev/null
  source "${VECTORIZE_PATHS}"
fi
if [[ -f "${FILTERING_PATHS}" ]]; then
  # shellcheck source=/dev/null
  source "${FILTERING_PATHS}"
fi

[[ -n "${_preserve_STEPS}" ]] && STEPS="${_preserve_STEPS}"

STEPS="${STEPS:-all}"

log() { echo "[$(date -Iseconds)] $*"; }

step_enabled() {
  local step="$1"
  [[ "${STEPS}" == "all" ]] && return 0
  [[ ",${STEPS}," == *",${step},"* ]]
}

export REPO_ROOT="${REPO_ROOT}"

log "POST-FILTER pipeline — STEPS=${STEPS}"
log "WORK_ROOT=${WORK_ROOT:-<unset>}"

if step_enabled "vectorize"; then
  log "=== Step: per-tile vectorize ==="
  bash "${SCRIPT_DIR}/run_vectorize_pipeline.sh"
fi

if step_enabled "polygon_area"; then
  log "=== Step: polygon area filter ==="
  export STEPS="${POLYGON_STEPS:-all}"
  bash "${REPO_ROOT}/filtering/run_polygon_area_pipeline.sh"
fi

if step_enabled "national"; then
  log "=== Step: national vectorize (merge + 112 px sieve + 200 m grouping) ==="
  bash "${SCRIPT_DIR}/run_vectorize_national_pipeline.sh"
fi

log "=== Post-filter pipeline finished ==="
