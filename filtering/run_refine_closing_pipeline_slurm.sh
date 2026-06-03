#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# Edit #SBATCH --mail-user before first use.
#SBATCH -J fire_refine_close
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Pilot pipeline: gentle closing only (see run_refine_closing_pipeline.sh)
#
#   sbatch filtering/run_refine_closing_pipeline_slurm.sh
#   sbatch filtering/run_refine_closing_pipeline_slurm.sh /path/to/input /path/to/output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRE_REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_refine_closing_pipeline.sh"
PATHS_FILE="${SCRIPT_DIR}/cluster_paths.env"

REFINE_INPUT_ARG="${1:-}"
REFINE_OUTPUT_ARG="${2:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================="
echo "GENTLE CLOSING PILOT — NLHPC"
echo "============================================="
echo "Repo:     ${FIRE_REPO}"
echo "Pipeline: ${PIPELINE_SCRIPT}"
echo "============================================="

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: Create paths file:" >&2
  echo "  cp ${SCRIPT_DIR}/cluster_paths.refine_closing.env.example ${PATHS_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PATHS_FILE}"

export REPO_ROOT="${FIRE_REPO}"

if [[ -n "${REFINE_INPUT_ARG}" ]]; then
  export REFINE_INPUT_DIR="${REFINE_INPUT_ARG}"
fi
if [[ -n "${REFINE_OUTPUT_ARG}" ]]; then
  export REFINE_OUTPUT_DIR="${REFINE_OUTPUT_ARG}"
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: PYTHON not set in ${PATHS_FILE}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, rasterio, scipy; print('numpy/rasterio/scipy OK')"

mkdir -p ~/logs
cd "${FIRE_REPO}"
bash "${PIPELINE_SCRIPT}"
exit $?
