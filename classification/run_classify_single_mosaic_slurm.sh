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

# Preserve caller-set paths before sourcing shared defaults (source overwrites OUTPUT_DIR).
_SAVED_OUTPUT_DIR="${OUTPUT_DIR:-}"
_SAVED_MODEL_DIR="${MODEL_DIR:-}"
_SAVED_MOSAIC_DIR="${MOSAIC_DIR:-}"
_SAVED_WRITE_PROBABILITY="${WRITE_PROBABILITY:-}"
_SAVED_BLOCK_SIZE="${BLOCK_SIZE:-}"
_SAVED_DECISION_THRESHOLD="${DECISION_THRESHOLD:-}"
_SAVED_PYTHON="${PYTHON:-}"

if [[ -f "${CLASSIFICATION_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${CLASSIFICATION_DIR}/cluster_paths.env"
fi

MODEL_NAME="${1:?model checkpoint base name}"
MOSAIC_NAME="${2:?mosaic filename}"

PYTHON_ENV="${_SAVED_PYTHON:-${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}}"
SCRIPT_PATH="${REPO_ROOT}/classification/classify_fire_model.py"
MODEL_DIR="${_SAVED_MODEL_DIR:-${MODEL_DIR:-${HOME}/models_col1}}"
MOSAIC_DIR="${_SAVED_MOSAIC_DIR:-${MOSAIC_DIR:-${HOME}/mosaics_cog}}"
OUTPUT_DIR="${_SAVED_OUTPUT_DIR:-${OUTPUT_DIR:-${HOME}/classification_output}}"
BLOCK_SIZE="${_SAVED_BLOCK_SIZE:-${BLOCK_SIZE:-40000000}}"
WRITE_PROBABILITY="${_SAVED_WRITE_PROBABILITY:-${WRITE_PROBABILITY:-0}}"
DECISION_THRESHOLD="${_SAVED_DECISION_THRESHOLD:-${DECISION_THRESHOLD:-}}"

echo "MODEL_DIR=${MODEL_DIR}"
echo "MOSAIC_DIR=${MOSAIC_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WRITE_PROBABILITY=${WRITE_PROBABILITY}"
echo "MODEL=${MODEL_NAME} MOSAIC=${MOSAIC_NAME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-22}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

# Fail early with clear messages
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: classify script missing: ${SCRIPT_PATH}" >&2
  echo "       git pull origin feat/auxiliares-to-gee ?" >&2
  exit 1
fi
if ! grep -q write-probability "${SCRIPT_PATH}"; then
  echo "ERROR: classify_fire_model.py has no --write-probability (old code)." >&2
  echo "       cd ~/fire && git pull origin feat/auxiliares-to-gee" >&2
  exit 1
fi
if [[ ! -f "${MOSAIC_DIR}/${MOSAIC_NAME}" ]]; then
  echo "ERROR: mosaic not found: ${MOSAIC_DIR}/${MOSAIC_NAME}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}" "${HOME}/logs"

cmd=(
  "${PYTHON_ENV}" "${SCRIPT_PATH}"
  --model-path "${MODEL_DIR}/${MODEL_NAME}"
  --mosaics "${MOSAIC_DIR}/${MOSAIC_NAME}"
  --block-size "${BLOCK_SIZE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${WRITE_PROBABILITY}" == "1" ]]; then
  cmd+=(--write-probability)
fi
if [[ -n "${DECISION_THRESHOLD}" ]]; then
  cmd+=(--decision-threshold "${DECISION_THRESHOLD}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done. Outputs:"
ls -lh "${OUTPUT_DIR}"/*2017* 2>/dev/null || ls -lh "${OUTPUT_DIR}" | tail -n 20
