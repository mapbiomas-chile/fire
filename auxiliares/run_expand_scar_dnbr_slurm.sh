#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Expand scars 2019–2025 via connected dNBR growth (GEE-like).
# threshold = max(p10(dNBR|scar), MIN_DNBR=0.10); regions r1 r2 r4 r6.
#
#   cd ~/fire && git pull
#   mkdir -p ~/logs
#
#   # Limpiar corrida anterior (umbral fijo):
#   rm -rf ~/classification_20260713_dnbr_expanded
#   # o backup:
#   # mv ~/classification_20260713_dnbr_expanded ~/classification_20260713_dnbr_expanded_fixed010_bak
#
#   sbatch auxiliares/run_expand_scar_dnbr_slurm.sh
#
# Logs: ~/logs/fire_dnbr_expand_<JOBID>.out  y  .err
# Salidas: ~/classification_20260713_dnbr_expanded/
#
#SBATCH -J fire_dnbr_expand
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
AUX_DIR="${FIRE_REPO}/auxiliares"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${AUX_DIR}/cluster_paths.env}"
SCRIPT="${AUX_DIR}/expand_scar_from_dnbr.py"

CLASS_DIR="${EXPAND_CLASS_DIR:-${HOME}/classification_20260713}"
MOSAIC_DIR="${EXPAND_MOSAIC_DIR:-${HOME}/mosaics_cog}"
OUTPUT_DIR="${EXPAND_OUTPUT_DIR:-${HOME}/classification_20260713_dnbr_expanded}"
REGIONS="${EXPAND_REGIONS:-r1 r2 r4 r6}"
FROM_YEAR="${EXPAND_FROM_YEAR:-2019}"
TO_YEAR="${EXPAND_TO_YEAR:-2025}"
DNBR_BAND="${EXPAND_DNBR_BAND:-13}"
DNBR_PERCENTILE="${EXPAND_DNBR_PERCENTILE:-10}"
MIN_DNBR="${EXPAND_MIN_DNBR:-0.10}"
SKIP_EXISTING="${EXPAND_SKIP_EXISTING:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "dNBR EXPAND — NLHPC (GEE-like)"
echo "============================================="
echo "Repo:      ${FIRE_REPO}"
echo "Script:    ${SCRIPT}"
echo "Class dir: ${CLASS_DIR}"
echo "Mosaics:   ${MOSAIC_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Regions:   ${REGIONS}"
echo "Years:     ${FROM_YEAR}–${TO_YEAR}"
echo "dNBR band: ${DNBR_BAND} | p${DNBR_PERCENTILE} | min_dnbr=${MIN_DNBR}"
echo "Job id:    ${SLURM_JOB_ID:-local}"
echo "============================================="

mkdir -p "${HOME}/logs" "${OUTPUT_DIR}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${CLASS_DIR}" ]]; then
  echo "ERROR: Classification dir not found: ${CLASS_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MOSAIC_DIR}" ]]; then
  echo "ERROR: Mosaic dir not found: ${MOSAIC_DIR}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, pandas, rasterio, scipy; print('deps OK')"

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --class-dir "${CLASS_DIR}"
  --mosaic-dir "${MOSAIC_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --regions ${REGIONS}
  --from-year "${FROM_YEAR}"
  --to-year "${TO_YEAR}"
  --dnbr-band "${DNBR_BAND}"
  --dnbr-percentile "${DNBR_PERCENTILE}"
  --min-dnbr "${MIN_DNBR}"
  --stats-csv "${OUTPUT_DIR}/expand_stats.csv"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

cd "${FIRE_REPO}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done. Stats: ${OUTPUT_DIR}/expand_stats.csv"
exit $?
