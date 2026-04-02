#!/bin/bash

python /home/ecastillo/mapbiomas/fire/refactoring/classify_fire_model.py \
  --model-path /home/ecastillo/mapbiomas/models/col1_chile_v2_r6_rnn_lstm_ckpt \
  --mosaics /home/ecastillo/mapbiomas/mosaics_cog/b14_chile_r6_2025_cog.tif\
  --output-dir /home/ecastillo/mapbiomas/output/classified
