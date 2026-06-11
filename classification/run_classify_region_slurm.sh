#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_region
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-type=FAIL
#SBATCH -t 01:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Clasifica mosaicos de una región y rango de años con un modelo MapBiomas Fire.
#
# Uso típico (con archivo de configuración):
#   cp classification/cluster_paths.env.example classification/cluster_paths.env
#   nano classification/cluster_paths.env
#   source classification/cluster_paths.env
#   sbatch classification/run_classify_region_slurm.sh
#
# Uso puntual sin archivo (variables de entorno):
#   export REGION=r2 MODEL_VERSION=v2 START_YEAR=2019 END_YEAR=2025
#   export OUTPUT_DIR="$HOME/classi_v2/r2_v2"
#   sbatch classification/run_classify_region_slurm.sh
#
# Un mosaico por job (alternativa):
#   sbatch classification/run_classify_fire_model_slurm.sh \
#     col1_chile_v2_r2_rnn_lstm_ckpt b14_chile_r2_2019_cog.tif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/cluster_paths.env"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-22}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

REGION="${REGION:-}"
MODEL_VERSION="${MODEL_VERSION:-v2}"
START_YEAR="${START_YEAR:-2019}"
END_YEAR="${END_YEAR:-2025}"
SATELLITE="${SATELLITE:-b14}"
COUNTRY="${COUNTRY:-chile}"
COLLECTION_NAME="${COLLECTION_NAME:-col1}"
BLOCK_SIZE="${BLOCK_SIZE:-40000000}"

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_ROOT}/classification/classify_fire_model.py}"
MOSAIC_DIR="${MOSAIC_DIR:-${HOME}/mosaics_cog}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models_col1}"

usage() {
  cat <<EOF
[ERROR] Falta configurar REGION (y revisar años / rutas).

Ejemplo:
  export REGION=r2 MODEL_VERSION=v2 START_YEAR=2019 END_YEAR=2025
  export OUTPUT_DIR="\${HOME}/classi_v2/r2_v2"
  sbatch classification/run_classify_region_slurm.sh

O copia y edita classification/cluster_paths.env
EOF
}

if [[ -z "${REGION}" ]]; then
  usage
  exit 1
fi

if [[ ! "${REGION}" =~ ^r[0-9]+$ ]]; then
  echo "[ERROR] REGION debe ser tipo r1, r2, r4, r6 (recibido: ${REGION})"
  exit 1
fi

if (( START_YEAR > END_YEAR )); then
  echo "[ERROR] START_YEAR (${START_YEAR}) > END_YEAR (${END_YEAR})"
  exit 1
fi

if [[ -n "${MODEL_NAME:-}" ]]; then
  MODEL_BASE="${MODEL_NAME}"
else
  MODEL_BASE="${COLLECTION_NAME}_${COUNTRY}_${MODEL_VERSION}_${REGION}_rnn_lstm_ckpt"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/classi_v2/${REGION}_${MODEL_VERSION}}"
MODEL_PATH="${MODEL_DIR}/${MODEL_BASE}"

echo "============================================="
echo "CLASIFICACIÓN POR REGIÓN"
echo "============================================="
echo "Región:        ${REGION}"
echo "Años:          ${START_YEAR}-${END_YEAR}"
echo "Modelo:        ${MODEL_BASE}"
echo "Mosaicos:      ${MOSAIC_DIR}/${SATELLITE}_${COUNTRY}_${REGION}_<year>_cog.tif"
echo "Salida:        ${OUTPUT_DIR}"
echo "Python:        ${PYTHON}"
echo "============================================="

for required in "${PYTHON}" "${SCRIPT_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] No existe: ${required}"
    exit 1
  fi
done

for required_dir in "${MOSAIC_DIR}" "${MODEL_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "[ERROR] No existe el directorio: ${required_dir}"
    exit 1
  fi
done

for suffix in .index .meta .data-00000-of-00001; do
  if [[ ! -e "${MODEL_PATH}${suffix}" ]]; then
    echo "[ERROR] Checkpoint incompleto (falta ${MODEL_PATH}${suffix})"
    exit 1
  fi
done

if [[ ! -e "${MODEL_PATH}_hyperparameters.json" ]]; then
  echo "[ERROR] No existe: ${MODEL_PATH}_hyperparameters.json"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -c "import numpy, scipy, tensorflow.compat.v1 as tf; print('deps OK')"

failed=0
processed=0

for (( YEAR=START_YEAR; YEAR<=END_YEAR; YEAR++ )); do
  MOSAIC_NAME="${SATELLITE}_${COUNTRY}_${REGION}_${YEAR}_cog.tif"
  MOSAIC_PATH="${MOSAIC_DIR}/${MOSAIC_NAME}"

  echo "---------------------------------------------"
  echo "Mosaico: ${MOSAIC_NAME}"
  echo "---------------------------------------------"

  if [[ ! -e "${MOSAIC_PATH}" ]]; then
    echo "[ERROR] No existe el mosaico: ${MOSAIC_PATH}"
    failed=$((failed + 1))
    continue
  fi

  "${PYTHON}" "${SCRIPT_PATH}" \
    --model-path "${MODEL_PATH}" \
    --mosaics "${MOSAIC_PATH}" \
    --block-size "${BLOCK_SIZE}" \
    --output-dir "${OUTPUT_DIR}"

  echo "[INFO] OK: ${MOSAIC_NAME}"
  processed=$((processed + 1))
done

echo "============================================="
echo "RESUMEN"
echo "  Procesados: ${processed}"
echo "  Fallidos:   ${failed}"
echo "  Salida:     ${OUTPUT_DIR}"
echo "============================================="

if (( failed > 0 )); then
  exit 1
fi
