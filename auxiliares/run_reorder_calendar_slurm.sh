#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Reorder season mosaics (classification_20260730) into calendar-year mosaics
# using the month band (band 2).
#
#   cd ~/fire && git pull
#   mkdir -p ~/logs
#   sbatch auxiliares/run_reorder_calendar_slurm.sh
#
# Prueba 1 año:
#   REORDER_FROM_YEAR=2015 REORDER_TO_YEAR=2015 \
#     sbatch auxiliares/run_reorder_calendar_slurm.sh
#
#SBATCH -J fire_reorder_calendar
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
AUX_DIR="${FIRE_REPO}/auxiliares"
SCRIPT="${AUX_DIR}/reorder_season_to_calendar.py"

INPUT_DIR="${SEASON_CLASS_DIR:-${HOME}/classification_20260730}"
OUTPUT_DIR="${CALENDAR_OUTPUT_DIR:-${HOME}/classification_20260730_calendar}"
FROM_YEAR="${REORDER_FROM_YEAR:-2013}"
TO_YEAR="${REORDER_TO_YEAR:-2025}"
SKIP_EXISTING="${REORDER_SKIP_EXISTING:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "Season -> calendar-year reorder — NLHPC"
echo "============================================="
echo "Repo:    ${FIRE_REPO}"
echo "Input:   ${INPUT_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Years:   ${FROM_YEAR}-${TO_YEAR}"
echo "Job id:  ${SLURM_JOB_ID:-local}"
echo "============================================="

mkdir -p "${HOME}/logs" "${OUTPUT_DIR}"

PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: missing dir: ${INPUT_DIR}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, pandas, rasterio; print('deps OK')"

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --from-year "${FROM_YEAR}"
  --to-year "${TO_YEAR}"
  --stats-csv "${OUTPUT_DIR}/season_to_calendar_stats.csv"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

cd "${FIRE_REPO}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done. Stats: ${OUTPUT_DIR}/season_to_calendar_stats.csv"
exit $?
