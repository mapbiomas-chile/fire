#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Editar #SBATCH --mail-user con tu correo antes del primer sbatch.
#SBATCH -J fire_vectorize
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Pipeline auxiliar de vectorización — SLURM NLHPC
# Requiere vectorize/cluster_paths.env (ver cluster_paths.env.example)
# Documentación: vectorize/README.md | vectorize/CLUSTER.md
#
#   sbatch vectorize/run_vectorize_pipeline_slurm.sh
#   sbatch vectorize/run_vectorize_pipeline_slurm.sh /path/to/classified_filtered /path/to/polygons

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE_REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_vectorize_pipeline.sh"
PATHS_FILE="${SCRIPT_DIR}/cluster_paths.env"

INPUT_ARG="${1:-}"
OUTPUT_ARG="${2:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"

echo "============================================="
echo "VECTORIZACIÓN — PIPELINE AUXILIAR NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Missing paths file. Create it from the example:" >&2
  echo "  cp ${SCRIPT_DIR}/cluster_paths.env.example ${PATHS_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"

if [[ -n "${INPUT_ARG}" ]]; then
  export VECTORIZE_INPUT_DIR="${INPUT_ARG}"
fi
if [[ -n "${OUTPUT_ARG}" ]]; then
  export VECTORIZE_OUTPUT_DIR="${OUTPUT_ARG}"
fi

export VECTORIZE_WORKERS="${VECTORIZE_WORKERS:-22}"

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: PYTHON not set in ${PATHS_FILE}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import geopandas, rasterio; print('geopandas/rasterio OK')"

mkdir -p ~/logs

cd "${FIRE_REPO}"
bash "${PIPELINE_SCRIPT}"
exit $?
