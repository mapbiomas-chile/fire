#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Editar #SBATCH --mail-user con tu correo antes del primer sbatch.
#SBATCH -J fire_class_filter
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Filtrado post-clasificación — SLURM NLHPC
# Requiere filtering/cluster_paths.env (ver cluster_paths.env.example)
# Documentación: filtering/README.md | filtering/CLUSTER.md
#
#   sbatch filtering/run_filtering_pipeline_slurm.sh
#   sbatch filtering/run_filtering_pipeline_slurm.sh /path/to/classified /path/to/work filter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE_REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_filtering_pipeline.sh"
PATHS_FILE="${SCRIPT_DIR}/cluster_paths.env"

CLASSIFIED_DIR_ARG="${1:-}"
WORK_ROOT_ARG="${2:-}"
STEPS_ARG="${3:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"

echo "============================================="
echo "FILTRADO POST-CLASIFICACIÓN — NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Missing paths file. Create it from the example:" >&2
  echo "  cp ${SCRIPT_DIR}/cluster_paths.env.example ${PATHS_FILE}" >&2
  echo "  # edit PYTHON, LULC_STACK, CLASSIFIED_DIR, WORK_ROOT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"

if [[ -n "${CLASSIFIED_DIR_ARG}" ]]; then
  export CLASSIFIED_DIR="${CLASSIFIED_DIR_ARG}"
fi
if [[ -n "${WORK_ROOT_ARG}" ]]; then
  export WORK_ROOT="${WORK_ROOT_ARG}"
fi
if [[ -n "${STEPS_ARG}" ]]; then
  export STEPS="${STEPS_ARG}"
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: PYTHON not set in ${PATHS_FILE}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, rasterio; print('numpy/rasterio OK')"

mkdir -p ~/logs
if [[ -n "${WORK_ROOT:-}" ]]; then
  mkdir -p "${WORK_ROOT}"
fi

cd "${FIRE_REPO}"
bash "${PIPELINE_SCRIPT}"
exit $?
