#!/bin/bash

export OMP_NUM_THREADS=22
export TF_NUM_INTRAOP_THREADS=22
export TF_NUM_INTEROP_THREADS=2

python /home/ecastillo/mapbiomas/fire/refactoring/classify_fire_model.py \
  --model-path /home/ecastillo/mapbiomas/models/col1_chile_v2_r6_rnn_lstm_ckpt \
  --mosaics /home/ecastillo/mapbiomas/mosaics_cog/b14_chile_r6_2025_cog.tif \
  --block-size 40000000 \
  --output-dir /home/ecastillo/mapbiomas/output/classified
