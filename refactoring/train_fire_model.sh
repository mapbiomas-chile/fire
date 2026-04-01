#!/bin/bash

python /home/ecastillo/mapbiomas/fire/refactoring/train_fire_model.py \
  --country chile \
  --version v1 \
  --region r3 \
  --training-samples-dir /home/ecastillo/mapbiomas/training_samples/ \
  --models-dir /home/ecastillo/mapbiomas/models
