#!/bin/bash
# Run one fire-model training job using environment variables.
# Called from run_train_fire_model_slurm.sh and run_train_chile_campaign_slurm.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

COUNTRY="${COUNTRY:-chile}"
TRAIN_VERSION="${TRAIN_VERSION:-${MODEL_VERSION:-v1}}"
TRAIN_REGION="${TRAIN_REGION:-${REGION:-r2}}"
SAMPLE_VERSION="${SAMPLE_VERSION:-v1}"
SAMPLE_START_YEAR="${SAMPLE_START_YEAR:-}"
SAMPLE_END_YEAR="${SAMPLE_END_YEAR:-}"
TRAINING_SAMPLES_DIR="${TRAINING_SAMPLES_DIR:-${HOME}/samples_col1}"
TRAINING_SAMPLE_LIST="${TRAINING_SAMPLE_LIST:-}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models_col1_20260619}"
TRAIN_BACKEND="${TRAIN_BACKEND:-tensorflow}"
TRAIN_VALIDATION_SPLIT="${TRAIN_VALIDATION_SPLIT:-by_file}"
TRAIN_LOSS="${TRAIN_LOSS:-weighted}"
TRAIN_METRIC="${TRAIN_METRIC:-f1}"
TRAIN_SPATIAL_WINDOW_SIZE="${TRAIN_SPATIAL_WINDOW_SIZE:-0}"
TRAIN_N_ITER="${TRAIN_N_ITER:-7000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1000}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_MAX_TRAINING_PIXELS="${TRAIN_MAX_TRAINING_PIXELS:-2000000}"
TRAIN_MAX_VALIDATION_PIXELS="${TRAIN_MAX_VALIDATION_PIXELS:-500000}"
TRAIN_INFERENCE_BLOCK_SIZE="${TRAIN_INFERENCE_BLOCK_SIZE:-500000}"
TRAIN_FIXED_DECISION_THRESHOLD="${TRAIN_FIXED_DECISION_THRESHOLD:-}"

if [[ -z "${TRAIN_REGION}" ]]; then
  echo "[ERROR] TRAIN_REGION is not set. For one model: export TRAIN_REGION=r1 TRAIN_VERSION=v1 ..."
  exit 1
fi
if [[ -z "${TRAIN_VERSION}" ]]; then
  echo "[ERROR] TRAIN_VERSION is not set."
  exit 1
fi

mkdir -p "${MODELS_DIR}"

echo "---------------------------------------------"
echo "[INFO] Training ${TRAIN_REGION} model ${TRAIN_VERSION}"
if [[ -n "${TRAINING_SAMPLE_LIST}" ]]; then
  echo "[INFO] Sample list: ${TRAINING_SAMPLE_LIST}"
else
  echo "[INFO] Sample files: ${SAMPLE_VERSION} in filename, years ${SAMPLE_START_YEAR:-*}-${SAMPLE_END_YEAR:-*}"
fi
if [[ -n "${TRAIN_FIXED_DECISION_THRESHOLD}" ]]; then
  echo "[INFO] Fixed threshold: ${TRAIN_FIXED_DECISION_THRESHOLD} (no per-region calibration)"
fi
echo "[INFO] Spatial window: ${TRAIN_SPATIAL_WINDOW_SIZE:-0}  oversample: ${TRAIN_OVERSAMPLE_BURNED:-0}  metric: ${TRAIN_METRIC}"
echo "[INFO] Output: ${MODELS_DIR}/col1_${COUNTRY}_${TRAIN_VERSION}_${TRAIN_REGION}_rnn_lstm_ckpt"
echo "---------------------------------------------"

common_args=(
  --country "${COUNTRY}"
  --version "${TRAIN_VERSION}"
  --region "${TRAIN_REGION}"
  --sample-version "${SAMPLE_VERSION}"
  --training-samples-dir "${TRAINING_SAMPLES_DIR}"
  --models-dir "${MODELS_DIR}"
  --seed "${TRAIN_SEED}"
  --validation-split "${TRAIN_VALIDATION_SPLIT}"
  --metric "${TRAIN_METRIC}"
)

if [[ -n "${SAMPLE_START_YEAR}" ]]; then
  common_args+=(--sample-start-year "${SAMPLE_START_YEAR}")
fi
if [[ -n "${SAMPLE_END_YEAR}" ]]; then
  common_args+=(--sample-end-year "${SAMPLE_END_YEAR}")
fi
if [[ -n "${TRAINING_SAMPLE_LIST}" ]]; then
  common_args+=(--sample-list-file "${TRAINING_SAMPLE_LIST}")
fi

if [[ "${TRAIN_SPATIAL_WINDOW_SIZE}" -ge 3 ]]; then
  common_args+=(--spatial-window-size "${TRAIN_SPATIAL_WINDOW_SIZE}")
fi

if [[ -n "${TRAIN_MAX_TRAINING_PIXELS}" ]]; then
  common_args+=(--max-training-pixels "${TRAIN_MAX_TRAINING_PIXELS}")
fi
if [[ -n "${TRAIN_MAX_VALIDATION_PIXELS}" ]]; then
  common_args+=(--max-validation-pixels "${TRAIN_MAX_VALIDATION_PIXELS}")
fi
common_args+=(--inference-block-size "${TRAIN_INFERENCE_BLOCK_SIZE}")

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
  if [[ -n "${TRAIN_FIXED_DECISION_THRESHOLD}" ]]; then
    train_args+=(--fixed-decision-threshold "${TRAIN_FIXED_DECISION_THRESHOLD}")
  fi
  "${PYTHON}" "${REPO_ROOT}/classification/train_fire_model.py" \
    "${train_args[@]}"
fi
