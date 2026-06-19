#!/bin/bash
# Submit Chile training campaign as a single Slurm job (7 models, 1 hour).
#
#   cd ~/fire
#   cp classification/cluster_paths.train.env.leftraru classification/cluster_paths.env
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
echo "  Mode:        single Slurm job (1 h walltime)"
echo "  Samples:     ${TRAINING_SAMPLES_DIR:-${HOME}/samples_col1}"
echo "  Output:      ${MODELS_DIR:-${HOME}/models_col1_20260619}"
echo "  Models:      ${#JOBS[@]}"
echo "============================================="

for job in "${JOBS[@]}"; do
  IFS=: read -r region model_version start_year end_year <<< "${job}"
  echo "[PLAN] ${region} ${model_version}  years=${start_year}-${end_year}  -> col1_chile_${model_version}_${region}_rnn_lstm_ckpt"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  python "${SCRIPT_DIR}/preview_training_campaign.py" "${TRAINING_SAMPLES_DIR:-${HOME}/samples_col1}"
  exit 0
fi

job_id="$(sbatch --export=ALL "${SCRIPT_DIR}/run_train_chile_campaign_slurm.sh" | awk '{print $4}')"
echo ""
echo "[SUBMIT] train_chile_campaign  job_id=${job_id}  walltime=01:00:00"
echo "  tail -f ~/logs/train_chile_campaign_${job_id}.out"
