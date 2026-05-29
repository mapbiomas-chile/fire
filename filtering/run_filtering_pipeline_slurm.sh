#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J fire_class_filter
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 22
#SBATCH --mem=64GB
#SBATCH --mail-user=felipe.lepin@ug.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00
#SBATCH -o /home/%u/logs/%x_%j.out
#SBATCH -e /home/%u/logs/%x_%j.err

# Post-classification filtering by land-cover class masks (all years / regions
# in CLASSIFIED_DIR). Stops before polygonize and area thresholding.
#
# Optional positional args:
#   sbatch run_filtering_pipeline_slurm.sh [CLASSIFIED_DIR] [WORK_ROOT] [STEPS]
#   STEPS: all | masks_accumulated,masks_yearly,masks_total,filter | filter

# =====================================================
# CONFIGURACIÓN DE THREADS
# =====================================================

export OMP_NUM_THREADS=22

# =====================================================
# RUTAS
# =====================================================

PYTHON_ENV="/home/flepin/.conda/envs/mb_fuego/bin/python"

FIRE_REPO="/home/flepin/fire"
PIPELINE_SCRIPT="${FIRE_REPO}/filtering/run_filtering_pipeline.sh"

LULC_STACK="${LULC_STACK:-/home/flepin/lulc/mapbiomas_chile_collection9.tif}"
CLASSIFIED_DIR="${1:-/home/flepin/classi_v2}"
WORK_ROOT="${2:-/home/flepin/filtering_work}"
STEPS="${3:-all}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2025}"
START_YEAR_BAND1="${START_YEAR_BAND1:-2000}"
WORKERS=22

# =====================================================
# VERIFICACIONES INICIALES
# =====================================================

echo "============================================="
echo "INICIO FILTRADO POR CLASES MAPBIOMAS FUEGO"
echo "============================================="
echo "Python usado:         $PYTHON_ENV"
echo "Script pipeline:      $PIPELINE_SCRIPT"
echo "Stack LULC:           $LULC_STACK"
echo "Clasificados entrada: $CLASSIFIED_DIR"
echo "Directorio trabajo:   $WORK_ROOT"
echo "Pasos (STEPS):        $STEPS"
echo "Años:                 ${FROM_YEAR}-${TO_YEAR}"
echo "Workers:              $WORKERS"
echo "============================================="

if [ ! -e "$PYTHON_ENV" ]; then
  echo "ERROR: No existe el Python del ambiente:"
  echo "$PYTHON_ENV"
  exit 1
fi

if [ ! -e "$PIPELINE_SCRIPT" ]; then
  echo "ERROR: No existe el script del pipeline:"
  echo "$PIPELINE_SCRIPT"
  exit 1
fi

if [ ! -d "$CLASSIFIED_DIR" ]; then
  echo "ERROR: No existe el directorio de clasificados:"
  echo "$CLASSIFIED_DIR"
  exit 1
fi

N_CLASSIFIED=$(find "$CLASSIFIED_DIR" -maxdepth 1 -name '*.tif' 2>/dev/null | wc -l)
if [ "$N_CLASSIFIED" -eq 0 ]; then
  echo "ERROR: No hay archivos .tif en:"
  echo "$CLASSIFIED_DIR"
  exit 1
fi
echo "GeoTIFF clasificados encontrados: $N_CLASSIFIED"

needs_lulc=0
if [ "$STEPS" = "all" ]; then
  needs_lulc=1
elif echo ",${STEPS}," | grep -qE ',masks_'; then
  needs_lulc=1
fi

if [ "$needs_lulc" -eq 1 ]; then
  if [ ! -e "$LULC_STACK" ]; then
    echo "ERROR: No existe el raster LULC para máscaras:"
    echo "$LULC_STACK"
    exit 1
  fi
fi

mkdir -p "$WORK_ROOT"

echo "Probando paquetes Python..."
$PYTHON_ENV -c "import numpy; print('numpy OK')"
$PYTHON_ENV -c "import rasterio; print('rasterio OK')"
$PYTHON_ENV -c "import geopandas; print('geopandas OK')"

echo "============================================="
echo "Iniciando pipeline de filtrado"
echo "============================================="

export REPO_ROOT="${FIRE_REPO}"
export PYTHON="${PYTHON_ENV}"
export LULC_STACK CLASSIFIED_DIR WORK_ROOT FROM_YEAR TO_YEAR START_YEAR_BAND1
export WORKERS STEPS

cd "${FIRE_REPO}"
bash "$PIPELINE_SCRIPT"

EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
  echo "ERROR: El pipeline de filtrado terminó con código $EXIT_CODE"
  exit "$EXIT_CODE"
fi

echo "============================================="
echo "PROCESO COMPLETO"
echo "Máscaras totales: ${WORK_ROOT}/mascaras/totales/"
echo "Rasters filtrados: ${WORK_ROOT}/classified_filtered/"
echo "============================================="
