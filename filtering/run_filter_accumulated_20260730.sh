#!/bin/bash
# Rebuild accumulated LULC masks (group A, incl. water 33 + infrastructure 24)
# and re-apply them to:
#   1) classification_20260730          (season)
#   2) classification_20260730_calendar (calendar)
#
# Does NOT apply agri (15) / pasture (18).
#
# Usage on leftraru:
#   cd ~/fire && git checkout feat/auxiliares-to-gee && git pull
#   conda activate mb_fuego
#   bash filtering/run_filter_accumulated_20260730.sh
#
# Optional env overrides:
#   LULC_STACK, ACCUMULATED_DIR, SEASON_DIR, CALENDAR_DIR,
#   SEASON_OUT, CALENDAR_OUT, REBUILD_MASKS=0|1, WORKERS, PYTHON

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

LULC_STACK="${LULC_STACK:-${HOME}/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif}"
# Prefer campaign mask tree if present; else a dedicated work dir for this re-run
ACCUMULATED_DIR="${ACCUMULATED_DIR:-${HOME}/classification_20260730_accA/mascaras/acumuladas}"

SEASON_DIR="${SEASON_DIR:-${HOME}/classification_20260730}"
CALENDAR_DIR="${CALENDAR_DIR:-${HOME}/classification_20260730_calendar}"

# New outputs (do not overwrite originals unless you set OUT == DIR)
SEASON_OUT="${SEASON_OUT:-${HOME}/classification_20260730_accA}"
CALENDAR_OUT="${CALENDAR_OUT:-${HOME}/classification_20260730_calendar_accA}"

REBUILD_MASKS="${REBUILD_MASKS:-1}"
WORKERS="${WORKERS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "============================================="
echo "Re-filter group A (accumulated LULC only)"
echo "  LULC_STACK:      ${LULC_STACK}"
echo "  ACCUMULATED_DIR: ${ACCUMULATED_DIR}"
echo "  Season in:       ${SEASON_DIR}"
echo "  Season out:      ${SEASON_OUT}"
echo "  Calendar in:     ${CALENDAR_DIR}"
echo "  Calendar out:    ${CALENDAR_OUT}"
echo "  REBUILD_MASKS:   ${REBUILD_MASKS}"
echo "  Classes: 29,23,61,34,25,33,24  (no 15/18)"
echo "============================================="

if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${FIRE_REPO}" ]]; then
  echo "ERROR: FIRE_REPO not found: ${FIRE_REPO}" >&2
  exit 1
fi
if [[ ! -d "${SEASON_DIR}" ]]; then
  echo "ERROR: season dir missing: ${SEASON_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CALENDAR_DIR}" ]]; then
  echo "ERROR: calendar dir missing: ${CALENDAR_DIR}" >&2
  exit 1
fi

mkdir -p "${ACCUMULATED_DIR}" "${SEASON_OUT}" "${CALENDAR_OUT}"

if [[ "${REBUILD_MASKS}" == "1" ]]; then
  if [[ ! -f "${LULC_STACK}" ]]; then
    echo "ERROR: LULC_STACK missing: ${LULC_STACK}" >&2
    exit 1
  fi
  echo "=== 1) Rebuild accumulated masks (incl. 33 water + 24 infrastructure) ==="
  "${PYTHON}" "${FIRE_REPO}/filtering/create_accumulated_class_masks.py" \
    --input-tif "${LULC_STACK}" \
    --output-dir "${ACCUMULATED_DIR}"
else
  echo "=== 1) Skip rebuild (REBUILD_MASKS=0); using ${ACCUMULATED_DIR} ==="
fi

echo "=== Accumulated files ==="
ls -lh "${ACCUMULATED_DIR}"/mascara_*_acumulado.tif

skip_flag=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_flag=(--skip-existing)
fi

echo "=== 2) Season: ${SEASON_DIR} -> ${SEASON_OUT} ==="
"${PYTHON}" "${FIRE_REPO}/filtering/filter_accumulated_lulc_only.py" \
  --input-dir "${SEASON_DIR}" \
  --accumulated-dir "${ACCUMULATED_DIR}" \
  --output-dir "${SEASON_OUT}" \
  --prefer-remap \
  --target-band 1 \
  --workers "${WORKERS}" \
  "${skip_flag[@]}"

echo "=== 3) Calendar: ${CALENDAR_DIR} -> ${CALENDAR_OUT} ==="
"${PYTHON}" "${FIRE_REPO}/filtering/filter_accumulated_lulc_only.py" \
  --input-dir "${CALENDAR_DIR}" \
  --accumulated-dir "${ACCUMULATED_DIR}" \
  --output-dir "${CALENDAR_OUT}" \
  --pattern "*.tif" \
  --target-band 1 \
  --workers "${WORKERS}" \
  "${skip_flag[@]}"

echo ""
echo "Done. Outputs:"
echo "  Season:   ${SEASON_OUT}"
echo "  Calendar: ${CALENDAR_OUT}"
echo "  Masks:    ${ACCUMULATED_DIR}"
echo "Per-file stats: *.json next to each output TIF (pixels_filtered_to_zero)."
