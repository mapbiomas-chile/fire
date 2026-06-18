#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_region
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=128GB
#SBATCH --mail-type=FAIL
#SBATCH -t 01:30:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Clasifica mosaicos de una región y rango de años con un modelo MapBiomas Fire.
#
# Tiempo: regla práctica ≈ 10–15 min por año. Este default (~1.5 h) cubre ~6–8 años.
# Ajusta al pedir el job, p. ej. 13 años (2013–2025):
#   sbatch -t 03:30:00 classification/run_classify_region_slurm.sh
# O acota años en cluster_paths.env (START_YEAR / END_YEAR).
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
# Un mosaico por job (debug):
#   sbatch classification/run_classify_single_mosaic_slurm.sh \
#     col1_chile_v2_r2_rnn_lstm_ckpt b14_chile_r2_2019_cog.tif

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"

# Preserve REGION/MODEL_DIR from sbatch --export before loading shared defaults.
_SAVED_REGION="${REGION:-}"
_SAVED_MODEL_DIR="${MODEL_DIR:-}"
_SAVED_OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -f "${CLASSIFICATION_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${CLASSIFICATION_DIR}/cluster_paths.env"
fi

REGION="${_SAVED_REGION:-${REGION:-}}"
MODEL_DIR="${_SAVED_MODEL_DIR:-${MODEL_DIR:-}}"
OUTPUT_DIR="${_SAVED_OUTPUT_DIR:-${OUTPUT_DIR:-}}"
unset _SAVED_REGION _SAVED_MODEL_DIR _SAVED_OUTPUT_DIR

SCRIPT_DIR="${CLASSIFICATION_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-22}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

REGION="${REGION:-}"
MODEL_VERSION="${MODEL_VERSION:-}"
AUTO_MODEL_VERSION_BY_YEAR="${AUTO_MODEL_VERSION_BY_YEAR:-0}"
START_YEAR="${START_YEAR:-2019}"
END_YEAR="${END_YEAR:-2025}"
SATELLITE="${SATELLITE:-b14}"
COUNTRY="${COUNTRY:-chile}"
COLLECTION_NAME="${COLLECTION_NAME:-col1}"
BLOCK_SIZE="${BLOCK_SIZE:-5000000}"
DECISION_THRESHOLD="${DECISION_THRESHOLD:-}"
OPENING_FILTER_SIZE="${OPENING_FILTER_SIZE:-2}"
CLOSING_FILTER_SIZE="${CLOSING_FILTER_SIZE:-4}"

resolve_model_version() {
  local region="$1"
  local year="$2"

  if [[ -n "${MODEL_VERSION}" && "${AUTO_MODEL_VERSION_BY_YEAR}" != "1" ]]; then
    echo "${MODEL_VERSION}"
    return
  fi

  # r2: v1 for 2013–2025. r1/r4/r6: v1 through 2018, v2 from 2019.
  if [[ "${region}" == "r2" ]]; then
    echo "v1"
    return
  fi

  if (( year <= 2018 )); then
    echo "v1"
  else
    echo "v2"
  fi
}

model_base_for_version() {
  local version="$1"
  if [[ -n "${MODEL_NAME:-}" && "${AUTO_MODEL_VERSION_BY_YEAR}" != "1" ]]; then
    echo "${MODEL_NAME}"
    return
  fi
  echo "${COLLECTION_NAME}_${COUNTRY}_${version}_${REGION}_rnn_lstm_ckpt"
}

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

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/classi_v2/${REGION}}"

echo "============================================="
echo "CLASIFICACIÓN POR REGIÓN"
echo "============================================="
echo "Región:        ${REGION}"
echo "Años:          ${START_YEAR}-${END_YEAR}"
if [[ "${AUTO_MODEL_VERSION_BY_YEAR}" == "1" || -z "${MODEL_VERSION}" ]]; then
  echo "Modelo:        auto por año (r2→v1; r1/r4/r6→v1≤2018, v2≥2019)"
else
  echo "Modelo:        ${MODEL_VERSION} fijo"
fi
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

mkdir -p "${OUTPUT_DIR}"

verify_model_checkpoint() {
  local model_path="$1"
  for suffix in .index .meta .data-00000-of-00001; do
    if [[ ! -e "${model_path}${suffix}" ]]; then
      echo "[ERROR] Checkpoint incompleto (falta ${model_path}${suffix})"
      return 1
    fi
  done
  if [[ ! -e "${model_path}_hyperparameters.json" ]]; then
    echo "[ERROR] No existe: ${model_path}_hyperparameters.json"
    return 1
  fi
  return 0
}

"${PYTHON}" -c "import numpy, scipy, tensorflow.compat.v1 as tf; print('deps OK')"

failed=0
processed=0

for (( YEAR=START_YEAR; YEAR<=END_YEAR; YEAR++ )); do
  MOSAIC_NAME="${SATELLITE}_${COUNTRY}_${REGION}_${YEAR}_cog.tif"
  MOSAIC_PATH="${MOSAIC_DIR}/${MOSAIC_NAME}"
  YEAR_MODEL_VERSION="$(resolve_model_version "${REGION}" "${YEAR}")"
  MODEL_BASE="$(model_base_for_version "${YEAR_MODEL_VERSION}")"
  MODEL_PATH="${MODEL_DIR}/${MODEL_BASE}"
  OUTPUT_FILE="${OUTPUT_DIR}/${MOSAIC_NAME%.tif}_classified.tif"

  if [[ -f "${OUTPUT_FILE}" ]]; then
    echo "[SKIP] Ya existe: ${OUTPUT_FILE}"
    processed=$((processed + 1))
    continue
  fi

  echo "---------------------------------------------"
  echo "Mosaico: ${MOSAIC_NAME}"
  echo "Modelo:  ${MODEL_BASE}"
  echo "---------------------------------------------"

  if [[ ! -e "${MOSAIC_PATH}" ]]; then
    echo "[ERROR] No existe el mosaico: ${MOSAIC_PATH}"
    failed=$((failed + 1))
    continue
  fi

  if ! verify_model_checkpoint "${MODEL_PATH}"; then
    failed=$((failed + 1))
    continue
  fi

  classify_args=(
    --model-path "${MODEL_PATH}"
    --mosaics "${MOSAIC_PATH}"
    --block-size "${BLOCK_SIZE}"
    --output-dir "${OUTPUT_DIR}"
    --opening-filter-size "${OPENING_FILTER_SIZE}"
    --closing-filter-size "${CLOSING_FILTER_SIZE}"
  )
  if [[ -n "${DECISION_THRESHOLD}" ]]; then
    classify_args+=(--decision-threshold "${DECISION_THRESHOLD}")
  fi

  "${PYTHON}" "${SCRIPT_PATH}" "${classify_args[@]}"

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
