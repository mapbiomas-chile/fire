#!/bin/bash
# Submit all Chile training jobs (7 checkpoints) with year-filtered samples.
#
# Samples on disk: samples_fire_v1_b14_chile_rN_..._YYYY....tif
# Model v1/v2 is the checkpoint name; filename token stays v1.
#
#   cd ~/fire
#   cp classification/cluster_paths.model_modification.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   bash classification/run_train_chile_campaign.sh --dry-run
#   bash classification/run_train_chile_campaign.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/cluster_paths.env"
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

SAMPLE_VERSION="${SAMPLE_VERSION:-v1}"

declare -a JOBS=(
  "r1:v1:2013:2018"
  "r1:v2:2019:2025"
  "r4:v1:2013:2018"
  "r4:v2:2019:2025"
  "r6:v1:2013:2018"
  "r6:v2:2019:2025"
  "r2:v1:2013:2018"
)

echo "============================================="
echo "CHILE TRAINING CAMPAIGN"
echo "  Samples:     ${TRAINING_SAMPLES_DIR:-${HOME}/samples_col1}"
echo "  Output:      ${MODELS_DIR:-${HOME}/models_col1_20260618}"
echo "  Sample token:  ${SAMPLE_VERSION} (filename)"
echo "  Jobs:          ${#JOBS[@]}"
echo "============================================="

for job in "${JOBS[@]}"; do
  IFS=: read -r region model_version start_year end_year <<< "${job}"
  job_name="train_${region}_${model_version}"
  exports="ALL,TRAIN_REGION=${region},TRAIN_VERSION=${model_version},SAMPLE_VERSION=${SAMPLE_VERSION},SAMPLE_START_YEAR=${start_year},SAMPLE_END_YEAR=${end_year}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] ${job_name}  years=${start_year}-${end_year}  -> col1_chile_${model_version}_${region}_rnn_lstm_ckpt"
    continue
  fi

  job_id="$(sbatch -J "${job_name}" --export="${exports}" "${SCRIPT_DIR}/run_train_fire_model_slurm.sh" | awk '{print $4}')"
  echo "[SUBMIT] ${job_name}  job_id=${job_id}  years=${start_year}-${end_year}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "Preview sample lists:"
  python "${SCRIPT_DIR}/preview_training_campaign.py" "${TRAINING_SAMPLES_DIR:-${HOME}/samples_col1}"
fi
