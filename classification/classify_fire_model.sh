#!/bin/bash
# Local example — edit MODEL_DIR, MOSAIC_DIR, OUTPUT_DIR before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

MODEL_DIR="${MODEL_DIR:-/path/to/models}"
MOSAIC_DIR="${MOSAIC_DIR:-/path/to/mosaics_cog}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output}"

"${PYTHON}" "${SCRIPT_DIR}/classify_fire_model.py" \
  --model-path "${MODEL_DIR}/col1_chile_v1_r2_rnn_lstm_ckpt" \
  --mosaics "${MOSAIC_DIR}/b14_chile_r2_2019_cog.tif" \
  --block-size 40000000 \
  --output-dir "${OUTPUT_DIR}"
