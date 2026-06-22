#!/usr/bin/env bash
# Live view of toGEE pipeline output counts (run in a second terminal).
#
#   source auxiliares/cluster_paths.env   # optional
#   bash auxiliares/watch_to_gee_progress.sh
#
# Env: TO_GEE_ROOT, WATCH_INTERVAL_SEC (default 15)

set -euo pipefail

TO_GEE_ROOT="${TO_GEE_ROOT:-${HOME}/toGEE}"
INTERVAL="${WATCH_INTERVAL_SEC:-15}"

count_tifs() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -name '*.tif' 2>/dev/null | wc -l
  else
    echo 0
  fi
}

latest_file() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -name '*.tif' -printf '%T@ %f\n' 2>/dev/null \
      | sort -n | tail -1 | cut -d' ' -f2-
  fi
}

echo "Watching ${TO_GEE_ROOT} every ${INTERVAL}s (Ctrl+C to stop)"
echo "Expected totals: by_tile ~52 | by_tile_filled ~52 | by_year_chile ~13 | by_year_chile_filled ~13"
echo

while true; do
  n_tile=$(count_tifs "${TO_GEE_ROOT}/by_tile")
  n_tile_f=$(count_tifs "${TO_GEE_ROOT}/by_tile_filled")
  n_year=$(count_tifs "${TO_GEE_ROOT}/by_year_chile")
  n_year_f=$(count_tifs "${TO_GEE_ROOT}/by_year_chile_filled")
  last_tile=$(latest_file "${TO_GEE_ROOT}/by_tile" || true)
  last_fill=$(latest_file "${TO_GEE_ROOT}/by_tile_filled" || true)

  printf '\033[H\033[J'
  date -Iseconds
  echo
  printf "by_tile:              %3s / 52  %s\n" "${n_tile}" "${last_tile:+(last: ${last_tile})}"
  printf "by_tile_filled:       %3s / 52  %s\n" "${n_tile_f}" "${last_fill:+(last: ${last_fill})}"
  printf "by_year_chile:        %3s / 13\n" "${n_year}"
  printf "by_year_chile_filled: %3s / 13\n" "${n_year_f}"
  echo
  if [[ -f "${TO_GEE_ROOT}/logs/fill_reference_tiles_stats.json" ]]; then
    echo "fill stats: ${TO_GEE_ROOT}/logs/fill_reference_tiles_stats.json"
  fi
  if [[ -f "${TO_GEE_ROOT}/logs/mask_by_polygons_stats.json" ]]; then
    echo "mask stats: ${TO_GEE_ROOT}/logs/mask_by_polygons_stats.json"
  fi
  sleep "${INTERVAL}"
done
