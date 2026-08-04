#!/bin/bash
# Validate UNIDOS_13_18 vs classification_20260730 **by fire season**.
# Default smoke test: season 2017.
#
# Matching rule (season-to-season):
#   UNIDOS.Season == YYYY  ↔  ~/classification_20260730/YYYY.tif
#                           or YYYY_remap.tif
# (band 1 = burn). Calendar reordering is NOT used.
#
#   cd ~/fire
#   git checkout feat/auxiliares-to-gee && git pull
#   conda activate mb_fuego
#   bash validation/run_unidos_validation_year.sh
#
# Other season:
#   VALIDATE_YEAR=2016 bash validation/run_unidos_validation_year.sh

set -euo pipefail

FIRE_REPO="${REPO_ROOT:-${HOME}/fire}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/mb_fuego/bin/python}"
SCRIPT="${FIRE_REPO}/validation/run_unidos_classification_validation.py"

YEAR="${VALIDATE_YEAR:-2017}"
REFERENCE_SHP="${REFERENCE_SHP:-${HOME}/validation/UNIDOS_13_18.shp}"
CLASS_DIR="${CLASS_DIR:-${HOME}/classification_20260730}"
OUTPUT_ROOT="${VALIDATE_OUTPUT_ROOT:-${HOME}/validation/unidos_vs_20260730_season}"
YEAR_COLUMN="${YEAR_COLUMN:-Season}"
WORKERS="${VALIDATE_WORKERS:-4}"
SKIP_EXISTING="${VALIDATE_SKIP_EXISTING:-0}"

echo "============================================="
echo "UNIDOS vs classification — SEASON mode"
echo "  Fire season:  ${YEAR}"
echo "  Reference:    ${REFERENCE_SHP}  (column ${YEAR_COLUMN})"
echo "  Class dir:    ${CLASS_DIR}"
echo "  Match:        Season=${YEAR} ↔ ${YEAR}.tif | ${YEAR}_remap.tif"
echo "  Output:       ${OUTPUT_ROOT}"
echo "============================================="

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing ${SCRIPT} — pull feat/auxiliares-to-gee" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_SHP}" ]]; then
  echo "ERROR: missing reference: ${REFERENCE_SHP}" >&2
  exit 1
fi
if [[ ! -d "${CLASS_DIR}" ]]; then
  echo "ERROR: missing classification dir: ${CLASS_DIR}" >&2
  exit 1
fi

# Preflight: season mosaic + year column sample
if [[ ! -f "${CLASS_DIR}/${YEAR}_remap.tif" && ! -f "${CLASS_DIR}/${YEAR}.tif" ]]; then
  echo "ERROR: no season raster ${CLASS_DIR}/${YEAR}.tif or ${YEAR}_remap.tif" >&2
  ls -la "${CLASS_DIR}" | head -40 >&2 || true
  exit 1
fi

"${PYTHON}" - <<PY
import geopandas as gpd
from pathlib import Path
shp = Path(r"${REFERENCE_SHP}")
gdf = gpd.read_file(shp, rows=5)
print("deps OK; reference columns:", list(gdf.columns))
print("reference CRS:", gdf.crs)
col = "${YEAR_COLUMN}"
if col not in gdf.columns:
    lower = {c.lower(): c for c in gdf.columns}
    if col.lower() in lower:
        print(f"WARN: year column case mismatch: use {lower[col.lower()]!r}")
    else:
        print(f"WARN: {col!r} not in first columns sample — full check in Python runner")
else:
    full = gpd.read_file(shp)
    print(f"Season column {col!r} value counts (head):")
    print(full[col].value_counts(dropna=False).head(15))
    print(f"n scars with {col}==${YEAR}:", int((full[col].astype(str).str[:4] == "${YEAR}").sum()))
PY

cmd=(
  "${PYTHON}" "${SCRIPT}"
  --year "${YEAR}"
  --reference-shp "${REFERENCE_SHP}"
  --classification-dir "${CLASS_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --year-column "${YEAR_COLUMN}"
  --workers "${WORKERS}"
  --python "${PYTHON}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

cd "${FIRE_REPO}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo ""
echo "Results (season ${YEAR}):"
ls -lh "${OUTPUT_ROOT}/season_${YEAR}/05_jaccard/" 2>/dev/null || true
ls -lh "${OUTPUT_ROOT}/season_${YEAR}/04_hits/" 2>/dev/null || true
echo "Manifest: ${OUTPUT_ROOT}/season_${YEAR}/run_manifest.json"
