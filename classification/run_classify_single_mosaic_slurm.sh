#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_one
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-type=FAIL
#SBATCH -t 00:45:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Un mosaico + un modelo (pruebas puntuales).
#   sbatch --export=ALL classification/run_classify_single_mosaic_slurm.sh \
#     col1_chile_v1_r2_rnn_lstm_ckpt b14_chile_r2_2019_cog.tif

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"

if [[ -f "${CLASSIFICATION_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${CLASSIFICATION_DIR}/cluster_paths.env"
fi

MODEL_NAME="${1:?model checkpoint base name}"
MOSAIC_NAME="${2:?mosaic filename}"

PYTHON_ENV="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"
SCRIPT_PATH="${REPO_ROOT}/classification/classify_fire_model.py"
MODEL_DIR="${MODEL_DIR:-${HOME}/models_col1}"
MOSAIC_DIR="${MOSAIC_DIR:-${HOME}/mosaics_cog}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/classification_output}"
BLOCK_SIZE="${BLOCK_SIZE:-40000000}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-22}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

"${PYTHON_ENV}" "${SCRIPT_PATH}" \
  --model-path "${MODEL_DIR}/${MODEL_NAME}" \
  --mosaics "${MOSAIC_DIR}/${MOSAIC_NAME}" \
  --block-size "${BLOCK_SIZE}" \
  --output-dir "${OUTPUT_DIR}"
