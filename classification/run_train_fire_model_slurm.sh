#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J train_fire_model
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH -t 00:30:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Entrena un solo modelo (región + versión) usando classification/cluster_paths.env
#
#   source classification/cluster_paths.env
#   sbatch classification/run_train_fire_model_slurm.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"
bash "${CLASSIFICATION_DIR}/train_fire_model_once.sh"
