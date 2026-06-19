#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J fire_poly_area
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Filtro de área en polígonos (histogramas → umbrales → filter)
#
#   cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
#   sbatch filtering/run_polygon_area_pipeline_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE_REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_polygon_area_pipeline.sh"
PATHS_FILE="${SCRIPT_DIR}/cluster_paths.env"

STEPS_ARG="${1:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "FILTRO DE ÁREA EN POLÍGONOS — NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Missing paths file. Create it from the example:" >&2
  echo "  cp ${SCRIPT_DIR}/cluster_paths.20260619.env.leftraru ${PATHS_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"

if [[ -n "${STEPS_ARG}" ]]; then
  export STEPS="${STEPS_ARG}"
fi

export PYTHON="${PYTHON:?PYTHON not set in cluster_paths.env}"

"${PYTHON}" -c "import geopandas; print('geopandas OK')"
mkdir -p ~/logs

cd "${FIRE_REPO}"
bash "${PIPELINE_SCRIPT}"
exit $?
