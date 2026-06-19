#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J train_chile_campaign
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Entrena los 7 checkpoints Chile en secuencia (1 job, ~6 min c/u).
#
#   source classification/cluster_paths.env
#   sbatch classification/run_train_chile_campaign_slurm.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"

if [[ -f "${CLASSIFICATION_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${CLASSIFICATION_DIR}/cluster_paths.env"
fi

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"
TRAIN_ONCE="${CLASSIFICATION_DIR}/train_fire_model_once.sh"

if [[ ! -f "${TRAIN_ONCE}" ]]; then
  echo "[ERROR] Not found: ${TRAIN_ONCE}"
  exit 1
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
echo "CHILE TRAINING CAMPAIGN (single job)"
echo "  Samples: ${TRAINING_SAMPLES_DIR:-${HOME}/samples_col1}"
echo "  Output:  ${MODELS_DIR:-${HOME}/models_col1_20260619}"
echo "  Models:  ${#JOBS[@]}"
echo "============================================="

failed=0
passed=0
campaign_start=$(date +%s)

for job in "${JOBS[@]}"; do
  IFS=: read -r region model_version start_year end_year <<< "${job}"
  echo ""
  echo "========== ${region} ${model_version} (${start_year}-${end_year}) =========="
  step_start=$(date +%s)

  export TRAIN_REGION="${region}"
  export TRAIN_VERSION="${model_version}"
  export SAMPLE_VERSION="${SAMPLE_VERSION}"
  export SAMPLE_START_YEAR="${start_year}"
  export SAMPLE_END_YEAR="${end_year}"

  if bash "${TRAIN_ONCE}"; then
    passed=$((passed + 1))
    step_secs=$(( $(date +%s) - step_start ))
    echo "[INFO] OK ${region} ${model_version} in ${step_secs}s"
  else
    failed=$((failed + 1))
    echo "[ERROR] FAILED ${region} ${model_version}"
  fi
done

total_secs=$(( $(date +%s) - campaign_start ))
echo ""
echo "============================================="
echo "CAMPAIGN SUMMARY"
echo "  Passed: ${passed}/${#JOBS[@]}"
echo "  Failed: ${failed}"
echo "  Total:  ${total_secs}s (~$(( total_secs / 60 )) min)"
echo "============================================="

if (( failed > 0 )); then
  exit 1
fi
