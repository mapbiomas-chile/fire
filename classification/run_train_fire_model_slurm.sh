#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J train_fire_model
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=FAIL
#SBATCH -t 12:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Entrena modelo MLP o XGBoost usando classification/cluster_paths.env
#
#   cp classification/cluster_paths.model_modification.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   sbatch classification/run_train_fire_model_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/cluster_paths.env"
fi

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

COUNTRY="${COUNTRY:-chile}"
TRAIN_VERSION="${TRAIN_VERSION:-${MODEL_VERSION:-v1}}"
TRAIN_REGION="${TRAIN_REGION:-${REGION:-r2}}"
TRAINING_SAMPLES_DIR="${TRAINING_SAMPLES_DIR:-${HOME}/training_samples}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models_col1_model_mod}"
TRAIN_BACKEND="${TRAIN_BACKEND:-tensorflow}"
TRAIN_VALIDATION_SPLIT="${TRAIN_VALIDATION_SPLIT:-by_file}"
TRAIN_LOSS="${TRAIN_LOSS:-weighted}"
TRAIN_METRIC="${TRAIN_METRIC:-f1}"
TRAIN_SPATIAL_WINDOW_SIZE="${TRAIN_SPATIAL_WINDOW_SIZE:-0}"
TRAIN_N_ITER="${TRAIN_N_ITER:-7000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1000}"
TRAIN_SEED="${TRAIN_SEED:-42}"

mkdir -p "${MODELS_DIR}"

common_args=(
  --country "${COUNTRY}"
  --version "${TRAIN_VERSION}"
  --region "${TRAIN_REGION}"
  --training-samples-dir "${TRAINING_SAMPLES_DIR}"
  --models-dir "${MODELS_DIR}"
  --seed "${TRAIN_SEED}"
  --validation-split "${TRAIN_VALIDATION_SPLIT}"
  --metric "${TRAIN_METRIC}"
)

if [[ "${TRAIN_SPATIAL_WINDOW_SIZE}" -ge 3 ]]; then
  common_args+=(--spatial-window-size "${TRAIN_SPATIAL_WINDOW_SIZE}")
fi

if [[ "${TRAIN_BACKEND}" == "xgboost" ]]; then
  echo "[INFO] Training backend: XGBoost"
  "${PYTHON}" "${REPO_ROOT}/classification/train_fire_model_xgboost.py" \
    "${common_args[@]}"
else
  echo "[INFO] Training backend: TensorFlow MLP"
  train_args=(
    "${common_args[@]}"
    --loss "${TRAIN_LOSS}"
    --n-iter "${TRAIN_N_ITER}"
    --batch-size "${TRAIN_BATCH_SIZE}"
  )
  if [[ "${TRAIN_OVERSAMPLE_BURNED:-0}" == "1" ]]; then
    train_args+=(--oversample-burned)
  fi
  "${PYTHON}" "${REPO_ROOT}/classification/train_fire_model.py" \
    "${train_args[@]}"
fi
