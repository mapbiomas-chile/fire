#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Editar #SBATCH --mail-user con tu correo antes del primer sbatch.
#SBATCH -J fire_to_gee_fill
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Relleno UNIDOS_13_18 (2013–2018) + passthrough 2019–2025 + mosaico Chile.
# Requiere: ~/fire/auxiliares/cluster_paths.env
#
# Corrida completa (52 tiles + 13 mosaicos) — por defecto:
#   cd ~/fire && git pull
#   cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
#   sbatch auxiliares/run_to_gee_fill_slurm.sh
#
# Solo tiles 2017 faltantes (reanudar):
#   TO_GEE_SLURM_MODE=missing_2017 sbatch auxiliares/run_to_gee_fill_slurm.sh
#
# Un tile: FILL_PATTERN='*r4_2017*vector_masked.tif' sbatch ...
#
# Logs: ~/logs/fire_to_gee_fill_<JOBID>.out  y  .err

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
AUX_DIR="${FIRE_REPO}/auxiliares"
PIPELINE_SCRIPT="${AUX_DIR}/run_to_gee_pipeline.sh"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${AUX_DIR}/cluster_paths.env}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

echo "============================================="
echo "toGEE — REFERENCE FILL — NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "Paths:    ${PATHS_FILE}"
echo "Job id:   ${SLURM_JOB_ID:-local}"
echo "Array:    ${SLURM_ARRAY_TASK_ID:-}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Missing ${PATHS_FILE}" >&2
  echo "  cp ${AUX_DIR}/cluster_paths.to_gee.env.leftraru ${PATHS_FILE}" >&2
  exit 1
fi

if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "ERROR: Pipeline not found: ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"

TO_GEE_SLURM_MODE="${TO_GEE_SLURM_MODE:-full}"
case "${TO_GEE_SLURM_MODE}" in
  full)
    export STEPS="${STEPS:-fill_tiles,fill_merge_years}"
    export FILL_PATTERN="${FILL_PATTERN:-*.tif}"
    export FILL_SKIP_EXISTING="${FILL_SKIP_EXISTING:-0}"
    export FILL_WORKERS="${FILL_WORKERS:-2}"
    ;;
  missing_2017)
    export STEPS="${STEPS:-fill_tiles}"
    export FILL_PATTERN="${FILL_PATTERN:-*2017*vector_masked.tif}"
    export FILL_SKIP_EXISTING="${FILL_SKIP_EXISTING:-1}"
    export FILL_WORKERS="${FILL_WORKERS:-1}"
    ;;
  *)
    echo "ERROR: Unknown TO_GEE_SLURM_MODE=${TO_GEE_SLURM_MODE} (use full or missing_2017)" >&2
    exit 1
    ;;
esac

export PROGRESS_HEARTBEAT_SEC="${PROGRESS_HEARTBEAT_SEC:-60}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  export STEPS="${STEPS:-fill_tiles}"
  export FILL_SKIP_EXISTING="${FILL_SKIP_EXISTING:-1}"
  export FILL_WORKERS="${FILL_WORKERS:-1}"
  case "${SLURM_ARRAY_TASK_ID}" in
    0) export FILL_PATTERN="*r2_2017*vector_masked.tif" ;;
    1) export FILL_PATTERN="*r4_2017*vector_masked.tif" ;;
    2) export FILL_PATTERN="*r6_2017*vector_masked.tif" ;;
    *)
      echo "ERROR: SLURM_ARRAY_TASK_ID must be 0, 1, or 2 (got ${SLURM_ARRAY_TASK_ID})" >&2
      exit 1
      ;;
  esac
  echo "Array task ${SLURM_ARRAY_TASK_ID} → pattern ${FILL_PATTERN}"
fi

echo "Mode:     ${TO_GEE_SLURM_MODE}"
echo "STEPS:    ${STEPS}"
echo "Pattern:  ${FILL_PATTERN}"
echo "Workers:  ${FILL_WORKERS}"
echo "Skip existing: ${FILL_SKIP_EXISTING}"

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: PYTHON not set in ${PATHS_FILE}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import geopandas, rasterio; print('geopandas/rasterio OK')"

mkdir -p ~/logs "${TO_GEE_ROOT:-$HOME/toGEE}/logs" \
  "${OUTPUT_BY_TILE_FILLED:-$HOME/toGEE/by_tile_filled}" \
  "${OUTPUT_BY_YEAR_FILLED:-$HOME/toGEE/by_year_chile_filled}"

cd "${FIRE_REPO}"
bash "${PIPELINE_SCRIPT}"
exit $?
