# Shared reprocess policy for MapBiomas Fire bash pipelines.
#
# Default (in_place): delete outputs that will be regenerated before running.
# Opt-out: export REPROCESS_POLICY=skip_existing  (or SKIP_EXISTING=1)
# New run folder: user must say so explicitly; set OUTPUT_DIR / WORK_ROOT to a new path.
#
# shellcheck shell=bash

REPROCESS_POLICY="${REPROCESS_POLICY:-in_place}"

reprocess_skip_existing() {
  [[ "${REPROCESS_POLICY}" == "skip_existing" || "${SKIP_EXISTING:-0}" == "1" ]]
}

reprocess_remove_file() {
  local path="$1"
  [[ -e "${path}" ]] || return 0
  if reprocess_skip_existing; then
    return 1
  fi
  echo "[REPROCESS] rm ${path}"
  rm -f "${path}"
  return 0
}

reprocess_remove_glob() {
  local pattern="$1"
  if reprocess_skip_existing; then
    return 0
  fi
  shopt -s nullglob
  local -a matches=( ${pattern} )
  shopt -u nullglob
  local path
  for path in "${matches[@]}"; do
    echo "[REPROCESS] rm ${path}"
    rm -f "${path}"
  done
}
