#!/usr/bin/env bash
# Quick status of toGEE outputs and missing filled tiles.
#
#   bash auxiliares/check_to_gee_status.sh

set -euo pipefail

TO_GEE_ROOT="${TO_GEE_ROOT:-${HOME}/toGEE}"
EXPECTED="${EXPECTED_TILE_COUNT:-52}"
EXPECTED_YEARS="${EXPECTED_YEAR_COUNT:-13}"

count_tifs() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -name '*.tif' 2>/dev/null | wc -l
  else
    echo 0
  fi
}

echo "=== toGEE status ==="
echo "Root: ${TO_GEE_ROOT}"
echo
printf "by_tile:              %s\n" "$(count_tifs "${TO_GEE_ROOT}/by_tile") / ${EXPECTED}"
printf "by_tile_filled:       %s\n" "$(count_tifs "${TO_GEE_ROOT}/by_tile_filled") / ${EXPECTED}"
printf "by_year_chile:        %s\n" "$(count_tifs "${TO_GEE_ROOT}/by_year_chile") / ${EXPECTED_YEARS}"
printf "by_year_chile_filled: %s\n" "$(count_tifs "${TO_GEE_ROOT}/by_year_chile_filled") / ${EXPECTED_YEARS}"
echo

if [[ -d "${TO_GEE_ROOT}/by_tile" ]]; then
  echo "Missing in by_tile_filled:"
  missing=0
  for f in "${TO_GEE_ROOT}"/by_tile/*_vector_masked.tif; do
    [[ -f "${f}" ]] || continue
    out="${TO_GEE_ROOT}/by_tile_filled/$(basename "${f}" .tif)_reference_filled.tif"
    if [[ ! -f "${out}" ]]; then
      echo "  $(basename "${out}")"
      missing=$((missing + 1))
    fi
  done
  [[ "${missing}" -eq 0 ]] && echo "  (none)"
fi
