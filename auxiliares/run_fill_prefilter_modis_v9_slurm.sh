#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Recover pre-filter burn ∩ buffered MODIS into classification_20260729
# national yearly mosaics (v9, burn = band 1). LULC A1+A2 applied AFTER add.
#
#   cd ~/fire && git pull
#   mkdir -p ~/logs
#   sbatch auxiliares/run_fill_prefilter_modis_v9_slurm.sh
#
# Prueba 1 año:
#   EXPAND_FROM_YEAR=2020 EXPAND_TO_YEAR=2020 \
#     sbatch auxiliares/run_fill_prefilter_modis_v9_slurm.sh
#
#SBATCH -J fire_prefilter_modis_v9
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
AUX_DIR="${FIRE_REPO}/auxiliares"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${AUX_DIR}/cluster_paths.env}"
SCRIPT="${AUX_DIR}/fill_from_prefilter_modis.py"

FINAL_DIR="${FINAL_CLASS_DIR:-${HOME}/classification_20260729}"
PREFILTER_DIR="${PREFILTER_CLASS_DIR:-${HOME}/classification_20260619}"
MODIS_DIR="${MODIS_DIR:-${HOME}/MODIS}"
OUTPUT_DIR="${PREFILTER_MODIS_OUTPUT_DIR:-${HOME}/classification_20260729_prefilter_modis}"
MASCARAS_ROOT="${MASCARAS_ROOT:-${HOME}/classification_20260619/filtering_work/mascaras}"
REGIONS="${EXPAND_REGIONS:-r1 r2 r4 r6}"
FROM_YEAR="${EXPAND_FROM_YEAR:-2019}"
TO_YEAR="${EXPAND_TO_YEAR:-2025}"
FINAL_BAND="${FINAL_BAND:-1}"
MODIS_BUFFER_PX="${MODIS_BUFFER_PX:-3}"
MIN_ADDED_PIXELS="${MIN_ADDED_PIXELS:-222}"
CLOSING_SIZE="${CLOSING_SIZE:-3}"
SKIP_EXISTING="${EXPAND_SKIP_EXISTING:-0}"
NO_LULC="${NO_LULC:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "PREFILTER ∩ buffered MODIS fill — v9 national"
echo "============================================="
echo "Repo:       ${FIRE_REPO}"
echo "Final:      ${FINAL_DIR} (band ${FINAL_BAND})"
echo "Prefilter:  ${PREFILTER_DIR}"
echo "MODIS:      ${MODIS_DIR}"
echo "Masks:      ${MASCARAS_ROOT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Regions:    ${REGIONS} (prefilter merge) | ${FROM_YEAR}-${TO_YEAR}"
echo "MODIS buf:  ${MODIS_BUFFER_PX} px | closing=${CLOSING_SIZE} | min_added=${MIN_ADDED_PIXELS}"
echo "Job id:     ${SLURM_JOB_ID:-local}"
echo "============================================="

mkdir -p "${HOME}/logs" "${OUTPUT_DIR}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

for d in "${FINAL_DIR}" "${PREFILTER_DIR}" "${MODIS_DIR}" "${MASCARAS_ROOT}"; do
  if [[ ! -d "${d}" ]]; then
    echo "ERROR: missing dir: ${d}" >&2
    exit 1
  fi
done

"${PYTHON}" -c "import numpy, pandas, rasterio, scipy; print('deps OK')"

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --layout national
  --final-dir "${FINAL_DIR}"
  --final-band "${FINAL_BAND}"
  --prefilter-dir "${PREFILTER_DIR}"
  --modis-dir "${MODIS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --mascaras-root "${MASCARAS_ROOT}"
  --regions ${REGIONS}
  --from-year "${FROM_YEAR}"
  --to-year "${TO_YEAR}"
  --modis-buffer-px "${MODIS_BUFFER_PX}"
  --closing-size "${CLOSING_SIZE}"
  --min-added-pixels "${MIN_ADDED_PIXELS}"
  --stats-csv "${OUTPUT_DIR}/prefilter_modis_stats.csv"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi
if [[ "${NO_LULC}" == "1" ]]; then
  cmd+=(--no-lulc)
fi

cd "${FIRE_REPO}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done. Stats: ${OUTPUT_DIR}/prefilter_modis_stats.csv"
exit $?
