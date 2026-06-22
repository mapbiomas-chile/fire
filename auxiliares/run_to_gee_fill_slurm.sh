#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Editar #SBATCH --mail-user con tu correo antes del primer sbatch.
#SBATCH -J fire_to_gee_fill
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Relleno con cicatrices de referencia (UNIDOS_13_18) — tiles pesados en cómputo.
# Requiere auxiliares/cluster_paths.env (cp cluster_paths.to_gee.env.leftraru).
#
#   cd ~/fire
#   cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
#   sbatch auxiliares/run_to_gee_fill_slurm.sh
#
# Logs: ~/logs/fire_to_gee_fill_<JOBID>.out y .err

set -euo pipefail

# No usar BASH_SOURCE: bajo SLURM el script vive en /var/spool/slurmd/job<id>/
FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
AUX_DIR="${FIRE_REPO}/auxiliares"
PIPELINE_SCRIPT="${AUX_DIR}/run_to_gee_pipeline.sh"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${AUX_DIR}/cluster_paths.env}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

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
export STEPS="${STEPS:-fill_tiles}"
export FILL_SKIP_EXISTING="${FILL_SKIP_EXISTING:-1}"
export FILL_WORKERS="${FILL_WORKERS:-1}"
export FILL_PATTERN="${FILL_PATTERN:-*2017*vector_masked.tif}"
export PROGRESS_HEARTBEAT_SEC="${PROGRESS_HEARTBEAT_SEC:-60}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
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

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: PYTHON not set in ${PATHS_FILE}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import geopandas, rasterio; print('geopandas/rasterio OK')"

mkdir -p ~/logs "${TO_GEE_ROOT:-$HOME/toGEE}/logs"

cd "${FIRE_REPO}"
echo "STEPS=${STEPS} FILL_PATTERN=${FILL_PATTERN} FILL_WORKERS=${FILL_WORKERS}"
bash "${PIPELINE_SCRIPT}"
exit $?
