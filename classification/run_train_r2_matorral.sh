#!/bin/bash
# Train r2 v1 from the 8 matorral sample TIFFs (see training_samples_r2_matorral_v1.txt).
#
#   cp classification/cluster_paths.train_r2_matorral.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   bash classification/run_train_r2_matorral.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/cluster_paths.env"
fi

export TRAIN_REGION="${TRAIN_REGION:-r2}"
export TRAIN_VERSION="${TRAIN_VERSION:-v1}"
export TRAINING_SAMPLE_LIST="${TRAINING_SAMPLE_LIST:-${SCRIPT_DIR}/training_samples_r2_matorral_v1.txt}"

if [[ ! -f "${TRAINING_SAMPLE_LIST}" ]]; then
  echo "[ERROR] Sample list not found: ${TRAINING_SAMPLE_LIST}"
  exit 1
fi

echo "============================================="
echo "TRAIN r2 v1 (matorral only)"
echo "  Samples: ${TRAINING_SAMPLE_LIST}"
echo "  Output:  ${MODELS_DIR:-${HOME}/models_col1_r2_matorral}/col1_chile_v1_r2_rnn_lstm_ckpt"
echo "  Infer:   r2 mosaics 2013–2025 (classify separately)"
echo "============================================="

bash "${SCRIPT_DIR}/train_fire_model_once.sh"
