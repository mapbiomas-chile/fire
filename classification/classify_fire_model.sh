#!/bin/bash

export OMP_NUM_THREADS=22
export TF_NUM_INTRAOP_THREADS=22
export TF_NUM_INTEROP_THREADS=2

python /home/flepin/fire/classification/classify_fire_model.py \
  --model-path /home/flepin/models_col1/col1_chile_v2_r6_rnn_lstm_ckpt \
  --mosaics /home/flepin/mosaics_cog/b14_chile_r6_2019_cog.tif \
  --block-size 40000000 \
  --output-dir /home/flepin/prueba
