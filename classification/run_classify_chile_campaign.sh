#!/bin/bash
# Submit classification for r1, r2, r4, r6 (2013–2025) — one Slurm job per region.
#
#   cp classification/cluster_paths.classify_20260618.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   bash classification/run_classify_chile_campaign.sh --dry-run
#   bash classification/run_classify_chile_campaign.sh

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

REGIONS=(r1 r2 r4 r6)
WALLTIME="${CLASSIFY_WALLTIME:-03:30:00}"

echo "============================================="
echo "CHILE CLASSIFICATION CAMPAIGN"
echo "  Years:    ${START_YEAR:-2013}-${END_YEAR:-2025}"
echo "  Models:   ${MODEL_DIR:-${HOME}/models_col1_20260618}"
echo "  Output:   ${OUTPUT_DIR:-${HOME}/classification_20260618}"
echo "  Regions:  ${REGIONS[*]}"
echo "  Walltime: ${WALLTIME} per region job"
echo "============================================="

for region in "${REGIONS[@]}"; do
  echo "[PLAN] ${region}  -> ${OUTPUT_DIR:-${HOME}/classification_20260618}/b14_chile_${region}_<year>_cog_classified.tif"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

for region in "${REGIONS[@]}"; do
  job_id="$(
    sbatch \
      -J "classi_${region}" \
      -t "${WALLTIME}" \
      --export=ALL,REGION="${region}" \
      "${SCRIPT_DIR}/run_classify_region_slurm.sh" | awk '{print $4}'
  )"
  echo "[SUBMIT] classi_${region}  job_id=${job_id}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ~/logs/classi_region_<JOBID>.out"
