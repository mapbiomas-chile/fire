#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_fire_model
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-user=ernesto.cast.nav@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Load environment variables from .env file
source ~/.env

# Activate conda environment
cd "$MINICONDA_PATH"
source ./etc/profile.d/conda.sh
conda activate "$CONDA_ENV_PATH"

export OMP_NUM_THREADS=22
export TF_NUM_INTRAOP_THREADS=22
export TF_NUM_INTEROP_THREADS=2

# Runtime parameters (positional args):
# sbatch run_classify_fire_model_slurm.sh <model_name> <mosaic_name>
MODEL_NAME="${1}"
MOSAIC_NAME="${2}"
MODEL_PATH="/home/ecastillo/mapbiomas/models/${MODEL_NAME}"
MOSAIC_PATH="/home/ecastillo/mapbiomas/mosaics_cog/${MOSAIC_NAME}"

python /home/ecastillo/mapbiomas/fire/classification/classify_fire_model.py \
  --model-path "$MODEL_PATH" \
  --mosaics "$MOSAIC_PATH" \
  --block-size 40000000 \
  --output-dir /home/ecastillo/mapbiomas/output/classified