#!/usr/bin/env bash
# Grow burn scars from dNBR similarity (2019–2025 experiments)
#
#   cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
#   source auxiliares/cluster_paths.env
#   bash auxiliares/run_dnbr_grow_pipeline.sh
#
# Or one tile dry-run:
#   GROW_DRY_RUN=1 GROW_PATTERN='*r1_2019*' bash auxiliares/run_dnbr_grow_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

PYTHON="${PYTHON:-}"
MOSAIC_DIR="${MOSAIC_DIR:-/home/flepin/mosaics_cog}"
GROW_INPUT_DIR="${GROW_INPUT_DIR:-${OUTPUT_BY_TILE_FILLED:-${TO_GEE_ROOT:-/home/flepin/toGEE}/by_tile_filled}}"
GROW_OUTPUT_DIR="${GROW_OUTPUT_DIR:-${TO_GEE_ROOT:-/home/flepin/toGEE}/by_tile_dnbr_grown}"
GROW_FROM_YEAR="${GROW_FROM_YEAR:-2019}"
GROW_TO_YEAR="${GROW_TO_YEAR:-2025}"
GROW_WORKERS="${GROW_WORKERS:-4}"
GROW_PATTERN="${GROW_PATTERN:-*.tif}"
GROW_OUTPUT_SUFFIX="${GROW_OUTPUT_SUFFIX:-_dnbr_grown}"
GROW_MAX_RADIUS="${GROW_MAX_RADIUS:-8}"
GROW_MAD_K="${GROW_MAD_K:-2.5}"
GROW_MIN_DNBR="${GROW_MIN_DNBR:-0.05}"
GROW_MAX_GROWTH_RATIO="${GROW_MAX_GROWTH_RATIO:-2.0}"
GROW_SKIP_EXISTING="${GROW_SKIP_EXISTING:-0}"
GROW_DRY_RUN="${GROW_DRY_RUN:-0}"
GROW_DNBR_BAND="${GROW_DNBR_BAND:-}"
PROGRESS_HEARTBEAT_SEC="${PROGRESS_HEARTBEAT_SEC:-30}"

log() { echo "[$(date -Iseconds)] $*"; }

if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not set. Copy auxiliares/cluster_paths.to_gee.env.leftraru to cluster_paths.env" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${GROW_OUTPUT_DIR}" "${TO_GEE_ROOT:-/home/flepin/toGEE}/logs"

cmd=(
  "${PYTHON}" auxiliares/grow_burn_from_dnbr.py
  --input-dir "${GROW_INPUT_DIR}"
  --output-dir "${GROW_OUTPUT_DIR}"
  --mosaic-dir "${MOSAIC_DIR}"
  --from-year "${GROW_FROM_YEAR}"
  --to-year "${GROW_TO_YEAR}"
  --pattern "${GROW_PATTERN}"
  --output-suffix "${GROW_OUTPUT_SUFFIX}"
  --max-radius "${GROW_MAX_RADIUS}"
  --mad-k "${GROW_MAD_K}"
  --min-dnbr "${GROW_MIN_DNBR}"
  --max-growth-ratio "${GROW_MAX_GROWTH_RATIO}"
  --workers "${GROW_WORKERS}"
  --stats-json "${GROW_OUTPUT_DIR}/grow_stats.json"
)

if [[ -n "${GROW_DNBR_BAND}" ]]; then
  cmd+=(--dnbr-band "${GROW_DNBR_BAND}")
fi
if [[ "${GROW_SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi
if [[ "${GROW_DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

log "dNBR grow: ${GROW_INPUT_DIR} -> ${GROW_OUTPUT_DIR} (years ${GROW_FROM_YEAR}-${GROW_TO_YEAR})"
log "Tip: inspect band first with: ${PYTHON} auxiliares/inspect_mosaic_dnbr_band.py --mosaic-dir ${MOSAIC_DIR}"

"${cmd[@]}"
