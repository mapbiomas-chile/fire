#!/bin/bash
# UNIDOS 2013–2018 validation vs classification_20260806 (season remaps).
#
# Design:
#   * Reference: ~/validation/UNIDOS_13_18.shp  (Season)
#   * Classification: ~/classification_20260806/burned_area_chile_temp_10_remap_{YYYY}.tif
#   * Pool: scars ≥ 200 ha, seasons 2013–2018
#   * Sample: N=100, stratified by year, seed=42
#   * Per season: binary burn → Albers → clip → polygonize → intersect → Jaccard
#
#   cd ~/fire && git checkout feat/auxiliares-to-gee && git pull
#   conda activate mb_fuego
#   bash validation/run_unidos_validation_20260806.sh
#
# Overrides:
#   SAMPLE_N=100 MIN_HA=200 FROM_YEAR=2013 TO_YEAR=2018 SEED=42
#   CLASS_DIR=~/classification_20260806 REFERENCE_SHP=~/validation/UNIDOS_13_18.shp
#   VALIDATE_OUTPUT_ROOT=~/validation/unidos_vs_20260806
#   SKIP_SAMPLE=1   # use full UNIDOS ≥200 ha, no random sample
#   YEARS=2017      # only one year (default: all years in range)

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

REFERENCE_SHP="${REFERENCE_SHP:-${HOME}/validation/UNIDOS_13_18.shp}"
CLASS_DIR="${CLASS_DIR:-${HOME}/classification_20260806}"
OUTPUT_ROOT="${VALIDATE_OUTPUT_ROOT:-${HOME}/validation/unidos_vs_20260806}"
YEAR_COLUMN="${YEAR_COLUMN:-Season}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2018}"
MIN_HA="${MIN_HA:-200}"
SAMPLE_N="${SAMPLE_N:-100}"
SEED="${SEED:-42}"
WORKERS="${VALIDATE_WORKERS:-4}"
SKIP_EXISTING="${VALIDATE_SKIP_EXISTING:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
YEARS="${YEARS:-}"   # empty = loop FROM_YEAR..TO_YEAR

ALBERS_GPKG="${OUTPUT_ROOT}/ref/UNIDOS_13_18_albers.gpkg"
SAMPLE_GPKG="${OUTPUT_ROOT}/ref/unidos_ge${MIN_HA}ha_n${SAMPLE_N}_seed${SEED}.gpkg"
CATALOG_FOR_INTERSECT="${SAMPLE_GPKG}"

echo "============================================="
echo "UNIDOS 2013–2018 × classification_20260806"
echo "  Reference:  ${REFERENCE_SHP}"
echo "  Class dir:  ${CLASS_DIR}"
echo "  Output:     ${OUTPUT_ROOT}"
echo "  Years:      ${FROM_YEAR}-${TO_YEAR}  (or YEARS=${YEARS:-all})"
echo "  Sample:     N=${SAMPLE_N}  min_ha=${MIN_HA}  seed=${SEED}  skip_sample=${SKIP_SAMPLE}"
echo "============================================="

if [[ ! -d "${FIRE_REPO}" ]]; then
  echo "ERROR: missing repo ${FIRE_REPO}" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_SHP}" ]]; then
  echo "ERROR: missing reference ${REFERENCE_SHP}" >&2
  exit 1
fi
if [[ ! -d "${CLASS_DIR}" ]]; then
  echo "ERROR: missing classification ${CLASS_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/ref" "${OUTPUT_ROOT}/logs"
cd "${FIRE_REPO}"

"${PYTHON}" -c "import geopandas, rasterio, numpy, pandas; print('deps OK')"

# --- 1) Chile Albers reference ---
if [[ ! -f "${ALBERS_GPKG}" ]]; then
  echo "=== Reproject UNIDOS → Chile Albers ==="
  "${PYTHON}" validation/reproject_vector_to_equal_area.py \
    --input "${REFERENCE_SHP}" \
    --output "${ALBERS_GPKG}" \
    --preset chile_albers
else
  echo "=== Reuse Albers reference: ${ALBERS_GPKG} ==="
fi

# --- 2) Sample ≥200 ha, N=100, 2013–2018 ---
if [[ "${SKIP_SAMPLE}" == "1" ]]; then
  CATALOG_FOR_INTERSECT="${ALBERS_GPKG}"
  echo "=== SKIP_SAMPLE=1 → use full Albers catalog (≥200 applied only if you add filter later) ==="
else
  if [[ ! -f "${SAMPLE_GPKG}" ]]; then
    echo "=== Sample scars ≥${MIN_HA} ha, N=${SAMPLE_N}, seed=${SEED}, stratify by year ==="
    "${PYTHON}" validation/sample_reference_scars.py \
      --catalog "${ALBERS_GPKG}" \
      --year-column "${YEAR_COLUMN}" \
      --from-year "${FROM_YEAR}" \
      --to-year "${TO_YEAR}" \
      --min-ha "${MIN_HA}" \
      --sample-n "${SAMPLE_N}" \
      --seed "${SEED}" \
      --stratify-by-year \
      --output "${SAMPLE_GPKG}" \
      --manifest-json "${SAMPLE_GPKG%.gpkg}.manifest.json"
  else
    echo "=== Reuse sample: ${SAMPLE_GPKG} ==="
  fi
  CATALOG_FOR_INTERSECT="${SAMPLE_GPKG}"
fi

# --- 3) Year list ---
if [[ -n "${YEARS}" ]]; then
  YEAR_LIST="${YEARS}"
else
  YEAR_LIST=$(seq "${FROM_YEAR}" "${TO_YEAR}" | tr '\n' ' ')
fi

skip_flag=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_flag=(--skip-existing)
fi

# --- 4) Per-season validation ---
for YEAR in ${YEAR_LIST}; do
  season_tif="${CLASS_DIR}/burned_area_chile_temp_10_remap_${YEAR}.tif"
  if [[ ! -f "${season_tif}" ]]; then
    # fallbacks checked later by Python; warn early
    echo "WARN: preferred season file missing: ${season_tif}"
  fi

  echo ""
  echo "########## Season ${YEAR} ##########"
  log_out="${OUTPUT_ROOT}/logs/season_${YEAR}.log"

  set +e
  "${PYTHON}" validation/run_unidos_classification_validation.py \
    --year "${YEAR}" \
    --reference-shp "${CATALOG_FOR_INTERSECT}" \
    --classification-dir "${CLASS_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --year-column "${YEAR_COLUMN}" \
    --workers "${WORKERS}" \
    --python "${PYTHON}" \
    "${skip_flag[@]}" \
    2>&1 | tee "${log_out}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: season ${YEAR} failed (exit ${rc}). See ${log_out}" >&2
    # continue other years
    continue
  fi

  echo "OK season ${YEAR} → ${OUTPUT_ROOT}/season_${YEAR}/"
done

echo ""
echo "=== Summary of Jaccard CSVs ==="
find "${OUTPUT_ROOT}" -path '*/05_jaccard/*.csv' -print 2>/dev/null || true
echo "Sample catalog: ${CATALOG_FOR_INTERSECT}"
echo "Done. Root: ${OUTPUT_ROOT}"
