#!/bin/bash
# Lanza clasificación serie completa en un solo job Slurm (wrapper de v2).
#
#   cp classification/cluster_paths.classify_20260618.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   bash classification/run_classify_chile_campaign.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/cluster_paths.env"
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Un solo job Slurm: run_classify_fire_model_slurm_v2.sh"
  echo "  Modelos:  ${MODEL_DIR:-${HOME}/models_col1_20260618}"
  echo "  Salida:   ${OUTPUT_DIR:-${HOME}/classification_20260618}"
  echo "  Años:     ${START_YEAR:-2013}-${END_YEAR:-2025}"
  echo "  Regiones: ${REGIONS:-r1 r2 r4 r6}"
  exit 0
fi

job_id="$(sbatch --export=ALL "${SCRIPT_DIR}/run_classify_fire_model_slurm_v2.sh" | awk '{print $4}')"
echo "[SUBMIT] classi_fire_model  job_id=${job_id}"
echo "  tail -f ~/logs/classi_fire_model_${job_id}.out"
