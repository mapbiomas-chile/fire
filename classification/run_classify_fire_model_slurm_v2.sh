#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J classi_fire_model
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 01:30:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# =====================================================
# CONFIGURACIÓN DE THREADS
# =====================================================

export OMP_NUM_THREADS=22
export TF_NUM_INTRAOP_THREADS=22
export TF_NUM_INTEROP_THREADS=2

# =====================================================
# RUTAS
# =====================================================

PYTHON_ENV="/home/flepin/.conda/envs/mb_fuego/bin/python"

SCRIPT_PATH="/home/flepin/fire/classification/classify_fire_model.py"

MOSAIC_DIR="/home/flepin/mosaics_cog"
MODEL_DIR="/home/flepin/models_col1"
OUTPUT_DIR="/home/flepin/classi_v2"

# =====================================================
# VERIFICACIONES INICIALES
# =====================================================

echo "============================================="
echo "INICIO CLASIFICACIÓN MAPBIOMAS FUEGO"
echo "============================================="
echo "Python usado: $PYTHON_ENV"
echo "Script clasificación: $SCRIPT_PATH"
echo "Directorio mosaicos: $MOSAIC_DIR"
echo "Directorio modelos: $MODEL_DIR"
echo "Directorio salida: $OUTPUT_DIR"
echo "============================================="

if [ ! -e "$PYTHON_ENV" ]; then
  echo "ERROR: No existe el Python del ambiente:"
  echo "$PYTHON_ENV"
  exit 1
fi

if [ ! -e "$SCRIPT_PATH" ]; then
  echo "ERROR: No existe el script de clasificación:"
  echo "$SCRIPT_PATH"
  exit 1
fi

if [ ! -d "$MOSAIC_DIR" ]; then
  echo "ERROR: No existe el directorio de mosaicos:"
  echo "$MOSAIC_DIR"
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: No existe el directorio de modelos:"
  echo "$MODEL_DIR"
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "ERROR: No existe el directorio de salida:"
  echo "$OUTPUT_DIR"
  exit 1
fi

echo "Probando paquetes Python..."
$PYTHON_ENV -c "import numpy; print('numpy OK')"
$PYTHON_ENV -c "import scipy; print('scipy OK')"
$PYTHON_ENV -c "import tensorflow.compat.v1 as tf; print('tensorflow OK')"

echo "============================================="
echo "Iniciando loop de mosaicos"
echo "============================================="

# =====================================================
# LOOP PRINCIPAL
# =====================================================

for MOSAIC_PATH in ${MOSAIC_DIR}/b14_chile_r*_????_cog.tif; do

  MOSAIC_NAME=$(basename "$MOSAIC_PATH")

  REGION=$(echo "$MOSAIC_NAME" | grep -oE 'r[0-9]+' | head -n 1)
  YEAR=$(echo "$MOSAIC_NAME" | grep -oE '(201[3-9]|202[0-5])' | head -n 1)

  if [ -z "$REGION" ]; then
    echo "ERROR: No pude detectar la región en $MOSAIC_NAME"
    continue
  fi

  if [ -z "$YEAR" ]; then
    echo "ERROR: No pude detectar el año en $MOSAIC_NAME"
    continue
  fi

  if [ "$YEAR" -ge 2013 ] && [ "$YEAR" -le 2018 ]; then
    MODEL_VERSION="v1"
  elif [ "$YEAR" -ge 2019 ] && [ "$YEAR" -le 2025 ]; then
    MODEL_VERSION="v2"
  else
    echo "ERROR: Año fuera de rango: $YEAR en $MOSAIC_NAME"
    continue
  fi

  MODEL_NAME="col1_chile_${MODEL_VERSION}_${REGION}_rnn_lstm_ckpt"
  MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

  echo "---------------------------------------------"
  echo "Procesando mosaico: $MOSAIC_NAME"
  echo "Año detectado: $YEAR"
  echo "Región detectada: $REGION"
  echo "Versión modelo: $MODEL_VERSION"
  echo "Modelo base: $MODEL_PATH"
  echo "Salida: $OUTPUT_DIR"
  echo "---------------------------------------------"

  if [ ! -e "${MODEL_PATH}.index" ]; then
    echo "ERROR: No existe el archivo índice del modelo:"
    echo "${MODEL_PATH}.index"
    continue
  fi

  if [ ! -e "${MODEL_PATH}.meta" ]; then
    echo "ERROR: No existe el archivo meta del modelo:"
    echo "${MODEL_PATH}.meta"
    continue
  fi

  if [ ! -e "${MODEL_PATH}.data-00000-of-00001" ]; then
    echo "ERROR: No existe el archivo data del modelo:"
    echo "${MODEL_PATH}.data-00000-of-00001"
    continue
  fi

  $PYTHON_ENV "$SCRIPT_PATH" \
    --model-path "$MODEL_PATH" \
    --mosaics "$MOSAIC_PATH" \
    --block-size 40000000 \
    --output-dir "$OUTPUT_DIR"

  EXIT_CODE=$?

  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "ERROR: Falló la clasificación de $MOSAIC_NAME con código $EXIT_CODE"
    continue
  fi

  echo "Finalizado correctamente: $MOSAIC_NAME"

done

echo "============================================="
echo "PROCESO COMPLETO"
echo "============================================="
