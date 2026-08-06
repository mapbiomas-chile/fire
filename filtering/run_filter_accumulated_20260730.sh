#!/bin/bash
# Rebuild accumulated LULC masks (group A, incl. water 33 + infrastructure 24),
# re-apply them to season + calendar products, then min-patch sieve:
#   drop connected burn scars with fewer than MIN_PATCH_PIXELS (default 223).
#
# Does NOT apply agri (15) / pasture (18).
#
# Usage on leftraru:
#   cd ~/fire && git checkout feat/auxiliares-to-gee && git pull
#   conda activate mb_fuego
#   bash filtering/run_filter_accumulated_20260730.sh
#
# Only surface sieve (if LULC accA already done):
#   RUN_LULC=0 bash filtering/run_filter_accumulated_20260730.sh
#
# Env:
#   LULC_STACK, ACCUMULATED_DIR, SEASON_DIR, CALENDAR_DIR,
#   SEASON_OUT, CALENDAR_OUT, SEASON_FINAL, CALENDAR_FINAL,
#   REBUILD_MASKS=0|1, RUN_LULC=0|1, RUN_SIEVE=0|1,
#   MIN_PATCH_PIXELS=223, MIN_PATCH_CONNECTIVITY=8, WORKERS, PYTHON

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

LULC_STACK="${LULC_STACK:-${HOME}/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif}"
ACCUMULATED_DIR="${ACCUMULATED_DIR:-${HOME}/classification_20260730_accA/mascaras/acumuladas}"

SEASON_DIR="${SEASON_DIR:-${HOME}/classification_20260730}"
CALENDAR_DIR="${CALENDAR_DIR:-${HOME}/classification_20260730_calendar}"

# Intermediate after LULC group A
SEASON_OUT="${SEASON_OUT:-${HOME}/classification_20260730_accA}"
CALENDAR_OUT="${CALENDAR_OUT:-${HOME}/classification_20260730_calendar_accA}"

# Final after min-patch sieve (>= 223 connected burn pixels)
SEASON_FINAL="${SEASON_FINAL:-${HOME}/classification_20260730_accA_min223}"
CALENDAR_FINAL="${CALENDAR_FINAL:-${HOME}/classification_20260730_calendar_accA_min223}"

REBUILD_MASKS="${REBUILD_MASKS:-1}"
RUN_LULC="${RUN_LULC:-1}"
RUN_SIEVE="${RUN_SIEVE:-1}"
MIN_PATCH_PIXELS="${MIN_PATCH_PIXELS:-223}"
MIN_PATCH_CONNECTIVITY="${MIN_PATCH_CONNECTIVITY:-8}"
WORKERS="${WORKERS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "============================================="
echo "20260730: accumulated LULC (A) + min-patch sieve"
echo "  LULC_STACK:       ${LULC_STACK}"
echo "  ACCUMULATED_DIR:  ${ACCUMULATED_DIR}"
echo "  Season in:        ${SEASON_DIR}"
echo "  Season LULC out:  ${SEASON_OUT}"
echo "  Season final:     ${SEASON_FINAL}"
echo "  Calendar in:      ${CALENDAR_DIR}"
echo "  Calendar LULC:    ${CALENDAR_OUT}"
echo "  Calendar final:   ${CALENDAR_FINAL}"
echo "  RUN_LULC:         ${RUN_LULC}  REBUILD_MASKS: ${REBUILD_MASKS}"
echo "  RUN_SIEVE:        ${RUN_SIEVE}  min_pixels>=${MIN_PATCH_PIXELS}  conn=${MIN_PATCH_CONNECTIVITY}"
echo "  Classes A:        29,23,61,34,25,33,24  (no 15/18)"
echo "============================================="

if [[ ! -f "${PYTHON}" && ! -x "${PYTHON}" ]]; then
  # allow python on PATH
  if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: PYTHON not found: ${PYTHON}" >&2
    exit 1
  fi
fi
if [[ ! -d "${FIRE_REPO}" ]]; then
  echo "ERROR: FIRE_REPO not found: ${FIRE_REPO}" >&2
  exit 1
fi

mkdir -p "${ACCUMULATED_DIR}"

skip_flag=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_flag=(--skip-existing)
fi

if [[ "${RUN_LULC}" == "1" ]]; then
  if [[ ! -d "${SEASON_DIR}" ]]; then
    echo "ERROR: season dir missing: ${SEASON_DIR}" >&2
    exit 1
  fi
  if [[ ! -d "${CALENDAR_DIR}" ]]; then
    echo "ERROR: calendar dir missing: ${CALENDAR_DIR}" >&2
    exit 1
  fi
  mkdir -p "${SEASON_OUT}" "${CALENDAR_OUT}"

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

  echo "=== 2) Season LULC A: ${SEASON_DIR} -> ${SEASON_OUT} ==="
  "${PYTHON}" "${FIRE_REPO}/filtering/filter_accumulated_lulc_only.py" \
    --input-dir "${SEASON_DIR}" \
    --accumulated-dir "${ACCUMULATED_DIR}" \
    --output-dir "${SEASON_OUT}" \
    --prefer-remap \
    --target-band 1 \
    --workers "${WORKERS}" \
    "${skip_flag[@]}"

  echo "=== 3) Calendar LULC A: ${CALENDAR_DIR} -> ${CALENDAR_OUT} ==="
  "${PYTHON}" "${FIRE_REPO}/filtering/filter_accumulated_lulc_only.py" \
    --input-dir "${CALENDAR_DIR}" \
    --accumulated-dir "${ACCUMULATED_DIR}" \
    --output-dir "${CALENDAR_OUT}" \
    --pattern "*.tif" \
    --target-band 1 \
    --workers "${WORKERS}" \
    "${skip_flag[@]}"
else
  echo "=== Skip LULC (RUN_LULC=0); sieve will use SEASON_OUT / CALENDAR_OUT ==="
fi

if [[ "${RUN_SIEVE}" == "1" ]]; then
  if [[ ! -d "${SEASON_OUT}" ]]; then
    echo "ERROR: season LULC dir missing for sieve: ${SEASON_OUT}" >&2
    exit 1
  fi
  if [[ ! -d "${CALENDAR_OUT}" ]]; then
    echo "ERROR: calendar LULC dir missing for sieve: ${CALENDAR_OUT}" >&2
    exit 1
  fi
  mkdir -p "${SEASON_FINAL}" "${CALENDAR_FINAL}"

  echo "=== 4) Season min-patch sieve: >= ${MIN_PATCH_PIXELS} px conn=${MIN_PATCH_CONNECTIVITY} ==="
  "${PYTHON}" "${FIRE_REPO}/filtering/sieve_min_patch_parallel.py" \
    --input-dir "${SEASON_OUT}" \
    --output-dir "${SEASON_FINAL}" \
    --pattern "*.tif" \
    --min-pixels "${MIN_PATCH_PIXELS}" \
    --connectivity "${MIN_PATCH_CONNECTIVITY}" \
    --mask-value 1 \
    --target-band 1 \
    --keep-names \
    --workers "${WORKERS}" \
    --stats-json "${SEASON_FINAL}/sieve_stats_min${MIN_PATCH_PIXELS}.json"

  echo "=== 5) Calendar min-patch sieve: >= ${MIN_PATCH_PIXELS} px ==="
  "${PYTHON}" "${FIRE_REPO}/filtering/sieve_min_patch_parallel.py" \
    --input-dir "${CALENDAR_OUT}" \
    --output-dir "${CALENDAR_FINAL}" \
    --pattern "*.tif" \
    --min-pixels "${MIN_PATCH_PIXELS}" \
    --connectivity "${MIN_PATCH_CONNECTIVITY}" \
    --mask-value 1 \
    --target-band 1 \
    --keep-names \
    --workers "${WORKERS}" \
    --stats-json "${CALENDAR_FINAL}/sieve_stats_min${MIN_PATCH_PIXELS}.json"
else
  echo "=== Skip sieve (RUN_SIEVE=0) ==="
fi

echo ""
echo "Done."
echo "  Season LULC A:   ${SEASON_OUT}"
echo "  Season final:    ${SEASON_FINAL}  (min ${MIN_PATCH_PIXELS} px connected)"
echo "  Calendar LULC A: ${CALENDAR_OUT}"
echo "  Calendar final:  ${CALENDAR_FINAL}"
echo "  Masks:           ${ACCUMULATED_DIR}"
