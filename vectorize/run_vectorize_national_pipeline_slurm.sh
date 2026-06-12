#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J fire_vectorize_nat
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Vectorización nacional: merge por año + polygonize + agrupación 200 m
#
#   sbatch vectorize/run_vectorize_national_pipeline_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE_REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PATHS_FILE="${SCRIPT_DIR}/cluster_paths.env"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export VECTORIZE_NATIONAL_MERGE_WORKERS="${VECTORIZE_NATIONAL_MERGE_WORKERS:-4}"

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Missing ${PATHS_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"
export PYTHON="${PYTHON:?PYTHON not set in cluster_paths.env}"

"${PYTHON}" -c "import geopandas, rasterio; print('geopandas/rasterio OK')"
mkdir -p ~/logs

cd "${FIRE_REPO}"
bash "${SCRIPT_DIR}/run_vectorize_national_pipeline.sh"
