#!/bin/bash
# UNIDOS 2013–2018 validation vs classification_20260806 (season remaps).
#
# Design (ONLY ≥1000 ha sample):
#   * Reference: ~/validation/UNIDOS_13_18.shp  (Season)
#   * Classification: ~/classification_20260806/burned_area_chile_temp_10_remap_{YYYY}.tif
#   * Pool: scars ≥ 1000 ha, seasons 2013–2018
#   * Sample: N=100, stratified by year, seed=42
#   * Per season: binary burn → Albers → clip → polygonize → intersect → Jaccard
#   * Combined CSV: jaccard_all_ge1000ha_n100.csv
#
# Clean re-run (borra salida y regenera solo esta campaña):
#   CLEAN_OUTPUT=1 CLEAN_LEGACY=1 bash validation/run_unidos_validation_20260806.sh
#
# Overrides:
#   SAMPLE_N=100 MIN_HA=1000 FROM_YEAR=2013 TO_YEAR=2018 SEED=42
#   VALIDATE_OUTPUT_ROOT=~/validation/unidos_vs_20260806_ge1000ha
#   CLEAN_OUTPUT=1 CLEAN_LEGACY=1 YEARS=2017

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"

REFERENCE_SHP="${REFERENCE_SHP:-${HOME}/validation/UNIDOS_13_18.shp}"
CLASS_DIR="${CLASS_DIR:-${HOME}/classification_20260806}"
# Dedicated folder so ≥1000 ha campaign is not mixed with older 200/250 ha runs
OUTPUT_ROOT="${VALIDATE_OUTPUT_ROOT:-${HOME}/validation/unidos_vs_20260806_ge1000ha}"
YEAR_COLUMN="${YEAR_COLUMN:-Season}"

FROM_YEAR="${FROM_YEAR:-2013}"
TO_YEAR="${TO_YEAR:-2018}"
MIN_HA="${MIN_HA:-1000}"
SAMPLE_N="${SAMPLE_N:-100}"
SEED="${SEED:-42}"
WORKERS="${VALIDATE_WORKERS:-4}"
SKIP_EXISTING="${VALIDATE_SKIP_EXISTING:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
CLEAN_LEGACY="${CLEAN_LEGACY:-0}"
YEARS="${YEARS:-}"

ALBERS_GPKG="${OUTPUT_ROOT}/ref/UNIDOS_13_18_albers.gpkg"
SAMPLE_GPKG="${OUTPUT_ROOT}/ref/unidos_ge${MIN_HA}ha_n${SAMPLE_N}_seed${SEED}.gpkg"
COMBINED_CSV="${OUTPUT_ROOT}/jaccard_all_ge${MIN_HA}ha_n${SAMPLE_N}.csv"
CATALOG_FOR_INTERSECT="${SAMPLE_GPKG}"

echo "============================================="
echo "UNIDOS 2013–2018 × classification_20260806"
echo "  Reference:  ${REFERENCE_SHP}"
echo "  Class dir:  ${CLASS_DIR}"
echo "  Output:     ${OUTPUT_ROOT}"
echo "  Years:      ${FROM_YEAR}-${TO_YEAR}  (or YEARS=${YEARS:-all})"
echo "  Sample:     N=${SAMPLE_N}  min_ha=${MIN_HA}  seed=${SEED}  skip_sample=${SKIP_SAMPLE}"
echo "  Clean:      CLEAN_OUTPUT=${CLEAN_OUTPUT} CLEAN_LEGACY=${CLEAN_LEGACY}"
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

if [[ "${CLEAN_OUTPUT}" == "1" && -d "${OUTPUT_ROOT}" ]]; then
  echo "=== CLEAN_OUTPUT=1: removing ${OUTPUT_ROOT} ==="
  rm -rf "${OUTPUT_ROOT}"
fi

if [[ "${CLEAN_LEGACY}" == "1" ]]; then
  for legacy in \
    "${HOME}/validation/unidos_vs_20260806" \
    "${HOME}/validation/unidos_vs_20260730_season"
  do
    if [[ -d "${legacy}" ]]; then
      echo "=== CLEAN_LEGACY: removing ${legacy} ==="
      rm -rf "${legacy}"
    fi
  done
fi

mkdir -p "${OUTPUT_ROOT}/ref" "${OUTPUT_ROOT}/logs"
cd "${FIRE_REPO}"

"${PYTHON}" -c "import geopandas, rasterio, numpy, pandas; print('deps OK')"

if [[ ! -f "${ALBERS_GPKG}" ]]; then
  echo "=== Reproject UNIDOS → Chile Albers ==="
  "${PYTHON}" validation/reproject_vector_to_equal_area.py \
    --input "${REFERENCE_SHP}" \
    --output "${ALBERS_GPKG}" \
    --preset chile_albers
else
  echo "=== Reuse Albers reference: ${ALBERS_GPKG} ==="
fi

if [[ "${SKIP_SAMPLE}" == "1" ]]; then
  CATALOG_FOR_INTERSECT="${ALBERS_GPKG}"
  echo "=== SKIP_SAMPLE=1 → use full Albers catalog (no random sample) ==="
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

if [[ -n "${YEARS}" ]]; then
  YEAR_LIST="${YEARS}"
else
  YEAR_LIST=$(seq "${FROM_YEAR}" "${TO_YEAR}" | tr '\n' ' ')
fi

skip_flag=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_flag=(--skip-existing)
fi

for YEAR in ${YEAR_LIST}; do
  season_tif="${CLASS_DIR}/burned_area_chile_temp_10_remap_${YEAR}.tif"
  if [[ ! -f "${season_tif}" ]]; then
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
    continue
  fi

  echo "OK season ${YEAR} → ${OUTPUT_ROOT}/season_${YEAR}/"
done

echo ""
echo "=== Combine Jaccard CSVs → ${COMBINED_CSV} ==="
"${PYTHON}" - <<PY
from pathlib import Path
import pandas as pd

root = Path(r"${OUTPUT_ROOT}")
paths = sorted(root.glob("season_*/05_jaccard/*_jaccard.csv"))
if not paths:
    print("[WARN] No Jaccard CSVs found under season_*/05_jaccard/")
    raise SystemExit(0)
frames = []
for p in paths:
    df = pd.read_csv(p)
    df.insert(0, "source_csv", p.name)
    frames.append(df)
out = pd.concat(frames, ignore_index=True)
out_path = Path(r"${COMBINED_CSV}")
out.to_csv(out_path, index=False)
print(f"[INFO] Wrote {len(out)} rows from {len(paths)} files → {out_path}")
PY

echo ""
echo "=== Done (≥${MIN_HA} ha only) ==="
echo "  Root:           ${OUTPUT_ROOT}"
echo "  Sample:         ${CATALOG_FOR_INTERSECT}"
echo "  Combined CSV:   ${COMBINED_CSV}"
find "${OUTPUT_ROOT}" -path '*/05_jaccard/*.csv' -print 2>/dev/null || true
