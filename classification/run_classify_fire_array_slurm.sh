#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_fire_array
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=ernesto.cast.nav@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH -t 0:20:00
#SBATCH -o /home/%u/logs/%x_%A_%a.out
#SBATCH -e /home/%u/logs/%x_%A_%a.err

set -euo pipefail

CSV_FILE="${1:-}"
HAS_HEADER="${2:-1}"

if [[ -z "$CSV_FILE" ]]; then
  echo "Usage: sbatch --array=1-N run_classify_fire_array_slurm.sh <csv_file> [has_header]"
  exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must run as a Slurm array job."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFY_SCRIPT="${SCRIPT_DIR}/run_classify_fire_model_slurm.sh"

if [[ ! -f "$CLASSIFY_SCRIPT" ]]; then
  echo "Classify script not found: $CLASSIFY_SCRIPT"
  exit 1
fi

DATA_LINE=$((SLURM_ARRAY_TASK_ID + HAS_HEADER))
ROW="$(sed -n "${DATA_LINE}p" "$CSV_FILE" || true)"
ROW="${ROW%%$'\r'}"

if [[ -z "$ROW" ]]; then
  echo "Empty row for task ${SLURM_ARRAY_TASK_ID} (line ${DATA_LINE})."
  exit 1
fi

IFS=',' read -r MODEL_NAME MOSAIC_NAME _ <<< "$ROW"
MODEL_NAME="$(echo "$MODEL_NAME" | xargs)"
MOSAIC_NAME="$(echo "$MOSAIC_NAME" | xargs)"

if [[ -z "$MODEL_NAME" || -z "$MOSAIC_NAME" ]]; then
  echo "Invalid row at line ${DATA_LINE}: ${ROW}"
  echo "Expected format: model_name,mosaic_name"
  exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL_NAME} mosaic=${MOSAIC_NAME}"
bash "$CLASSIFY_SCRIPT" "$MODEL_NAME" "$MOSAIC_NAME"
