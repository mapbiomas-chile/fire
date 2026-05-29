#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_fire_model
#SBATCH -p debug
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err


module load Miniconda3
conda init bash
conda activate mb_fuego

export OMP_NUM_THREADS=22
export TF_NUM_INTRAOP_THREADS=22
export TF_NUM_INTEROP_THREADS=2

# Runtime parameters (positional args):
# sbatch run_classify_fire_model_slurm.sh <model_name> <mosaic_name>
MODEL_NAME="${1}"
MOSAIC_NAME="${2}"
MODEL_PATH="/home/flepin/models_col1/${MODEL_NAME}"
MOSAIC_PATH="/home/flepin/mosaics_cog/${MOSAIC_NAME}"

python /home/flepin/fire/classification/classify_fire_model.py \
  --model-path "$MODEL_PATH" \
  --mosaics "$MOSAIC_PATH" \
  --block-size 40000000 \
  --output-dir /home/flepin/classi_v2
