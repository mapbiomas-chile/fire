#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J train_fire_model
#SBATCH -p v100
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-user=ernesto.cast.nav@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Load environment variables from .env file
source ~/.env

# Activate conda environment
cd "$MINICONDA_PATH"
source ./etc/profile.d/conda.sh
conda activate "$CONDA_ENV_PATH"

python /home/ecastillo/mapbiomas/fire/classification/train_fire_model.py \
  --country chile \
  --version v1 \
  --region r2 \
  --training-samples-dir /home/ecastillo/mapbiomas/training_samples/ \
  --models-dir /home/ecastillo/mapbiomas/models
