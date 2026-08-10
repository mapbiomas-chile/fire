#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
# UNIDOS ≥1000 ha / N=100 vs classification_20260806 — cola debug (30 min).
#
#   cd ~/fire
#   git checkout feat/auxiliares-to-gee && git pull
#   mkdir -p ~/logs
#   sbatch validation/run_unidos_validation_20260806_slurm.sh
#
# Un solo año (recomendado en debug si 6 años no entran en 30 min):
#   YEARS=2017 sbatch validation/run_unidos_validation_20260806_slurm.sh
#
#SBATCH -J fire_val_unidos
#SBATCH -p debug
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
RUNNER="${FIRE_REPO}/validation/run_unidos_validation_20260806.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VALIDATE_WORKERS="${VALIDATE_WORKERS:-2}"
export PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

# Defaults already in runner; allow sbatch env overrides
export REFERENCE_SHP="${REFERENCE_SHP:-${HOME}/validation/UNIDOS_13_18.shp}"
export CLASS_DIR="${CLASS_DIR:-${HOME}/classification_20260806}"
export VALIDATE_OUTPUT_ROOT="${VALIDATE_OUTPUT_ROOT:-${HOME}/validation/unidos_vs_20260806}"
export MIN_HA="${MIN_HA:-1000}"
export SAMPLE_N="${SAMPLE_N:-100}"
export SEED="${SEED:-42}"
export FROM_YEAR="${FROM_YEAR:-2013}"
export TO_YEAR="${TO_YEAR:-2018}"
# YEARS= empty → all years; set e.g. YEARS=2017 for smoke test on debug
export YEARS="${YEARS:-}"

echo "============================================="
echo "UNIDOS validation × classification_20260806"
echo "============================================="
echo "Job id:     ${SLURM_JOB_ID:-local}"
echo "Partition:  ${SLURM_JOB_PARTITION:-debug}"
echo "Repo:       ${FIRE_REPO}"
echo "Class dir:  ${CLASS_DIR}"
echo "Reference:  ${REFERENCE_SHP}"
echo "Output:     ${VALIDATE_OUTPUT_ROOT}"
echo "min_ha:     ${MIN_HA}  sample_n: ${SAMPLE_N}  seed: ${SEED}"
echo "years:      ${FROM_YEAR}-${TO_YEAR}  YEARS=${YEARS:-all}"
echo "Workers:    ${VALIDATE_WORKERS}"
echo "============================================="

mkdir -p "${HOME}/logs" "${VALIDATE_OUTPUT_ROOT}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: missing ${RUNNER} — git pull feat/auxiliares-to-gee" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" && ! -f "${PYTHON}" ]]; then
  echo "ERROR: PYTHON not found: ${PYTHON}" >&2
  exit 1
fi

# Prefer loaded modules on NLHPC if conda env binary needs them
if command -v module >/dev/null 2>&1; then
  module load Miniconda3 2>/dev/null || true
fi

cd "${FIRE_REPO}"
bash "${RUNNER}"
echo "Done. Output root: ${VALIDATE_OUTPUT_ROOT}"
