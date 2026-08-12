#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# UNIDOS missing from preliminary → classification_20260806 (2013–2018).
# Then LULC A1 only on the union.
#
# Final / campaign folder:
#   ~/classification_20260806/burned_area_chile_temp_10_remap_{year}.tif
# Output:
#   ~/classification_20260806/prefilter_reference/
#
#   cd ~/fire && git pull
#   mkdir -p ~/logs
#   sbatch auxiliares/run_fill_prefilter_reference_20260806_slurm.sh
#
# Prueba 1 año:
#   EXPAND_FROM_YEAR=2017 EXPAND_TO_YEAR=2017 \
#     sbatch auxiliares/run_fill_prefilter_reference_20260806_slurm.sh
#
# Legacy gate:
#   FILL_MODE=touch-prefilter sbatch auxiliares/run_fill_prefilter_reference_20260806_slurm.sh
#
#SBATCH -J fire_fill_ref_0806
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
AUX_DIR="${FIRE_REPO}/auxiliares"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${AUX_DIR}/cluster_paths.env}"
SCRIPT="${AUX_DIR}/fill_from_prefilter_reference.py"

CAMPAIGN_DIR="${FINAL_CLASS_DIR:-${HOME}/classification_20260806}"
FINAL_DIR="${CAMPAIGN_DIR}"
PREFILTER_DIR="${PREFILTER_CLASS_DIR:-${HOME}/classification_20260619}"
REFERENCE_SHP="${REFERENCE_SCARS_SHP:-${HOME}/validation/UNIDOS_13_18.shp}"
OUTPUT_DIR="${PREFILTER_REF_OUTPUT_DIR:-${CAMPAIGN_DIR}/prefilter_reference}"
MASCARAS_ROOT="${MASCARAS_ROOT:-${HOME}/classification_20260619/filtering_work/mascaras}"
REGIONS="${EXPAND_REGIONS:-r1 r2 r4 r6}"
FROM_YEAR="${EXPAND_FROM_YEAR:-2013}"
TO_YEAR="${EXPAND_TO_YEAR:-2018}"
FINAL_BAND="${FINAL_BAND:-1}"
FINAL_PATTERN="${FINAL_PATTERN:-burned_area_chile_temp_10_remap_{year}.tif}"
FILL_MODE="${FILL_MODE:-missing-from-prefilter}"
SKIP_EXISTING="${EXPAND_SKIP_EXISTING:-0}"
NO_LULC="${NO_LULC:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "UNIDOS fill — classification_20260806"
echo "============================================="
echo "Repo:       ${FIRE_REPO}"
echo "Mode:       ${FILL_MODE}"
echo "Final:      ${FINAL_DIR} (band ${FINAL_BAND})"
echo "Pattern:    ${FINAL_PATTERN}"
echo "Prefilter:  ${PREFILTER_DIR}"
echo "Reference:  ${REFERENCE_SHP}"
echo "Masks:      ${MASCARAS_ROOT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Years:      ${FROM_YEAR}-${TO_YEAR} | LULC A1 after add"
echo "Job id:     ${SLURM_JOB_ID:-local}"
echo "============================================="

mkdir -p "${HOME}/logs" "${OUTPUT_DIR}"

if [[ -f "${PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${PATHS_FILE}"
fi

PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

for d in "${FINAL_DIR}" "${PREFILTER_DIR}" "${MASCARAS_ROOT}"; do
  if [[ ! -d "${d}" ]]; then
    echo "ERROR: missing dir: ${d}" >&2
    exit 1
  fi
done
if [[ ! -f "${REFERENCE_SHP}" ]]; then
  echo "ERROR: missing reference shapefile: ${REFERENCE_SHP}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, pandas, rasterio, geopandas; print('deps OK')"

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --layout national
  --mode "${FILL_MODE}"
  --final-dir "${FINAL_DIR}"
  --final-band "${FINAL_BAND}"
  --final-pattern "${FINAL_PATTERN}"
  --prefilter-dir "${PREFILTER_DIR}"
  --reference-shp "${REFERENCE_SHP}"
  --output-dir "${OUTPUT_DIR}"
  --mascaras-root "${MASCARAS_ROOT}"
  --regions ${REGIONS}
  --from-year "${FROM_YEAR}"
  --to-year "${TO_YEAR}"
  --stats-csv "${OUTPUT_DIR}/prefilter_reference_stats.csv"
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
echo "Done. Stats: ${OUTPUT_DIR}/prefilter_reference_stats.csv"
exit $?
