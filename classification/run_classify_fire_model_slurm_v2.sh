#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_fire_model
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-type=FAIL
#SBATCH -t 10:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Clasifica todos los mosaicos b14_chile_r*_YYYY_cog.tif en un solo job Slurm.
#
#   cp classification/cluster_paths.classify_20260618.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   sbatch classification/run_classify_fire_model_slurm_v2.sh
#
# Reglas modelo:
#   r2: v1 (2013–2025)
#   r1, r4, r6: v1 (2013–2018), v2 (2019–2025)

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/fire}"
CLASSIFICATION_DIR="${REPO_ROOT}/classification"

if [[ -f "${CLASSIFICATION_DIR}/cluster_paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${CLASSIFICATION_DIR}/cluster_paths.env"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-22}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

PYTHON_ENV="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_ROOT}/classification/classify_fire_model.py}"
MOSAIC_DIR="${MOSAIC_DIR:-${HOME}/mosaics_cog}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models_col1_20260618}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/classification_20260618}"
BLOCK_SIZE="${BLOCK_SIZE:-40000000}"
OPENING_FILTER_SIZE="${OPENING_FILTER_SIZE:-2}"
CLOSING_FILTER_SIZE="${CLOSING_FILTER_SIZE:-4}"
START_YEAR="${START_YEAR:-2013}"
END_YEAR="${END_YEAR:-2025}"
REGIONS="${REGIONS:-r1 r2 r4 r6}"

resolve_model_version() {
  local region="$1"
  local year="$2"

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

region_allowed() {
  local region="$1"
  for allowed in ${REGIONS}; do
    if [[ "${region}" == "${allowed}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "============================================="
echo "CLASIFICACIÓN MAPBIOMAS FUEGO (serie completa)"
echo "============================================="
echo "Python:   ${PYTHON_ENV}"
echo "Script:   ${SCRIPT_PATH}"
echo "Mosaicos: ${MOSAIC_DIR}"
echo "Modelos:  ${MODEL_DIR}"
echo "Salida:   ${OUTPUT_DIR}"
echo "Años:     ${START_YEAR}-${END_YEAR}"
echo "Regiones: ${REGIONS}"
echo "============================================="

for required in "${PYTHON_ENV}" "${SCRIPT_PATH}"; do
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

"${PYTHON_ENV}" -c "import numpy, scipy, tensorflow.compat.v1 as tf; print('deps OK')"

failed=0
processed=0
skipped=0

for MOSAIC_PATH in "${MOSAIC_DIR}"/b14_chile_r*_????_cog.tif; do
  [[ -e "${MOSAIC_PATH}" ]] || continue

  MOSAIC_NAME="$(basename "${MOSAIC_PATH}")"
  REGION="$(echo "${MOSAIC_NAME}" | grep -oE 'r[0-9]+' | head -n 1)"
  YEAR="$(echo "${MOSAIC_NAME}" | grep -oE '(201[3-9]|202[0-5])' | head -n 1)"

  if [[ -z "${REGION}" || -z "${YEAR}" ]]; then
    echo "[WARNING] No pude parsear región/año: ${MOSAIC_NAME}"
    skipped=$((skipped + 1))
    continue
  fi

  if (( YEAR < START_YEAR || YEAR > END_YEAR )); then
    continue
  fi

  if ! region_allowed "${REGION}"; then
    continue
  fi

  MODEL_VERSION="$(resolve_model_version "${REGION}" "${YEAR}")"
  MODEL_NAME="col1_chile_${MODEL_VERSION}_${REGION}_rnn_lstm_ckpt"
  MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

  echo "---------------------------------------------"
  echo "Mosaico: ${MOSAIC_NAME}"
  echo "Región:  ${REGION}  Año: ${YEAR}  Modelo: ${MODEL_NAME}"
  echo "---------------------------------------------"

  for suffix in .index .meta .data-00000-of-00001; do
    if [[ ! -e "${MODEL_PATH}${suffix}" ]]; then
      echo "[ERROR] Checkpoint incompleto: ${MODEL_PATH}${suffix}"
      failed=$((failed + 1))
      continue 2
    fi
  done

  if [[ ! -e "${MODEL_PATH}_hyperparameters.json" ]]; then
    echo "[ERROR] No existe: ${MODEL_PATH}_hyperparameters.json"
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

  if ! "${PYTHON_ENV}" "${SCRIPT_PATH}" "${classify_args[@]}"; then
    echo "[ERROR] Falló: ${MOSAIC_NAME}"
    failed=$((failed + 1))
    continue
  fi

  echo "[INFO] OK: ${MOSAIC_NAME}"
  processed=$((processed + 1))
done

echo "============================================="
echo "RESUMEN"
echo "  Procesados: ${processed}"
echo "  Fallidos:   ${failed}"
echo "  Omitidos:   ${skipped}"
echo "  Salida:     ${OUTPUT_DIR}"
echo "============================================="

if (( failed > 0 )); then
  exit 1
fi
