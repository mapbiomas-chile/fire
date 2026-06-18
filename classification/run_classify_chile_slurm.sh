#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_chile
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=128GB
#SBATCH --mail-type=FAIL
#SBATCH -t 10:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err
#
# Clasifica la serie Chile: r1, r2, r4, r6 × años 2013–2025 (un solo job).
#
#   cp classification/cluster_paths.classify.env.leftraru classification/cluster_paths.env
#   source classification/cluster_paths.env
#   sbatch --export=ALL classification/run_classify_chile_slurm.sh
#
# Reglas modelo:
#   r2: v1 (todos los años)
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
MODEL_DIR="${MODEL_DIR:-${HOME}/models_col1}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/classification_output}"
BLOCK_SIZE="${BLOCK_SIZE:-5000000}"
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
  if (( year <= 2018 )); then echo "v1"; else echo "v2"; fi
}

region_allowed() {
  local region="$1"
  for allowed in ${REGIONS}; do
    [[ "${region}" == "${allowed}" ]] && return 0
  done
  return 1
}

echo "============================================="
echo "CLASIFICACIÓN CHILE (serie completa)"
echo "============================================="
echo "Modelos:  ${MODEL_DIR}"
echo "Salida:   ${OUTPUT_DIR}"
echo "Años:     ${START_YEAR}-${END_YEAR}"
echo "Regiones: ${REGIONS}"
echo "============================================="

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_ENV}" -c "import numpy, scipy, tensorflow.compat.v1 as tf; print('deps OK')"

failed=0
processed=0

for MOSAIC_PATH in "${MOSAIC_DIR}"/b14_chile_r*_????_cog.tif; do
  [[ -e "${MOSAIC_PATH}" ]] || continue

  MOSAIC_NAME="$(basename "${MOSAIC_PATH}")"
  REGION="$(echo "${MOSAIC_NAME}" | grep -oE 'r[0-9]+' | head -n 1)"
  YEAR="$(echo "${MOSAIC_NAME}" | grep -oE '(201[3-9]|202[0-5])' | head -n 1)"

  [[ -z "${REGION}" || -z "${YEAR}" ]] && continue
  (( YEAR < START_YEAR || YEAR > END_YEAR )) && continue
  region_allowed "${REGION}" || continue

  MODEL_VERSION="$(resolve_model_version "${REGION}" "${YEAR}")"
  MODEL_PATH="${MODEL_DIR}/col1_chile_${MODEL_VERSION}_${REGION}_rnn_lstm_ckpt"
  OUTPUT_FILE="${OUTPUT_DIR}/${MOSAIC_NAME%.tif}_classified.tif"

  if [[ -f "${OUTPUT_FILE}" ]]; then
    echo "[SKIP] Ya existe: ${OUTPUT_FILE}"
    processed=$((processed + 1))
    continue
  fi

  echo "---------------------------------------------"
  echo "${MOSAIC_NAME}  modelo=${MODEL_VERSION}_${REGION}"
  echo "---------------------------------------------"

  for suffix in .index .meta .data-00000-of-00001; do
    [[ -e "${MODEL_PATH}${suffix}" ]] || { echo "[ERROR] Falta ${MODEL_PATH}${suffix}"; failed=$((failed + 1)); continue 2; }
  done
  [[ -e "${MODEL_PATH}_hyperparameters.json" ]] || { echo "[ERROR] Falta JSON"; failed=$((failed + 1)); continue; }

  if ! "${PYTHON_ENV}" "${SCRIPT_PATH}" \
    --model-path "${MODEL_PATH}" \
    --mosaics "${MOSAIC_PATH}" \
    --block-size "${BLOCK_SIZE}" \
    --output-dir "${OUTPUT_DIR}" \
    --opening-filter-size "${OPENING_FILTER_SIZE}" \
    --closing-filter-size "${CLOSING_FILTER_SIZE}"; then
    failed=$((failed + 1))
    continue
  fi
  processed=$((processed + 1))
done

echo "RESUMEN: ${processed} OK, ${failed} fallidos -> ${OUTPUT_DIR}"
if (( failed > 0 )); then
  exit 1
fi
exit 0
