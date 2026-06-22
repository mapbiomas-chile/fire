#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Editar #SBATCH --mail-user con tu correo antes del primer sbatch.
#SBATCH -J fire_to_gee_fill
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Relleno con cicatrices de referencia (UNIDOS_13_18) — tiles pesados en cómputo.
# Requiere auxiliares/cluster_paths.env (cp cluster_paths.to_gee.env.leftraru).
#
# Solo los 3 tiles 2017 faltantes (skip-existing, r1 ya hecho):
#   sbatch auxiliares/run_to_gee_fill_slurm.sh
#
# Un tile concreto (ej. r4 2017):
#   FILL_PATTERN='*r4_2017*vector_masked.tif' sbatch auxiliares/run_to_gee_fill_slurm.sh
#
# Fill + mosaico nacional:
#   STEPS=fill_tiles,fill_merge_years sbatch auxiliares/run_to_gee_fill_slurm.sh
#
# Tres jobs en paralelo (uno por región faltante):
#   sbatch --array=0-2 auxiliares/run_to_gee_fill_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE_REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_to_gee_pipeline.sh"
PATHS_FILE="${AUXILIARES_PATHS_FILE:-${SCRIPT_DIR}/cluster_paths.env}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "toGEE — REFERENCE FILL — NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "Job id:   ${SLURM_JOB_ID:-local}"
echo "Array:    ${SLURM_ARRAY_TASK_ID:-}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Missing ${PATHS_FILE}" >&2
  echo "  cp ${SCRIPT_DIR}/cluster_paths.to_gee.env.leftraru ${PATHS_FILE}" >&2
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

# SLURM array: 0=r2, 1=r4, 2=r6 (2017 tiles that often fail on login node)
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
