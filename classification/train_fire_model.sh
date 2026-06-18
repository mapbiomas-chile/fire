#!/bin/bash
# Local example — edit paths before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${SCRIPT_DIR}/train_fire_model.py" \
  --country chile \
  --version v1 \
  --region r2 \
  --training-samples-dir "${TRAINING_SAMPLES_DIR:-/path/to/samples}" \
  --models-dir "${MODELS_DIR:-/path/to/models}" \
  --validation-split by_file \
  --loss weighted \
  --oversample-burned \
  --metric f1 \
  --spatial-window-size 5
