#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J fire_class_filter
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Filtrado post-clasificación — SLURM NLHPC
# Documentación: filtering/README.md | filtering/CLUSTER.md

export OMP_NUM_THREADS=22

FIRE_REPO="/home/flepin/fire"
PIPELINE_SCRIPT="${FIRE_REPO}/filtering/run_filtering_pipeline.sh"
PATHS_FILE="${FIRE_REPO}/filtering/cluster_paths.env"

CLASSIFIED_DIR="${1:-}"
WORK_ROOT="${2:-}"
STEPS="${3:-}"

echo "============================================="
echo "FILTRADO POST-CLASIFICACIÓN — NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Create paths file from example:"
  echo "  cp ${FIRE_REPO}/filtering/cluster_paths.env.example ${PATHS_FILE}"
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"
export PYTHON="${PYTHON:-/home/flepin/.conda/envs/mb_fuego/bin/python}"

if [[ -n "${CLASSIFIED_DIR}" ]]; then
  export CLASSIFIED_DIR
fi
if [[ -n "${WORK_ROOT}" ]]; then
  export WORK_ROOT
fi
if [[ -n "${STEPS}" ]]; then
  export STEPS
fi

if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: Python not found: ${PYTHON}"
  exit 1
fi

"${PYTHON}" -c "import numpy, rasterio; print('numpy/rasterio OK')"

mkdir -p "${WORK_ROOT}" ~/logs

cd "${FIRE_REPO}"
bash "${PIPELINE_SCRIPT}"
exit $?
