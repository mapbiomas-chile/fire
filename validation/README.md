# Validation

Scripts to validate and prepare reference layers used to evaluate the burned-area pipeline (see [../classification/README.md](../classification/README.md) and [../filtering/README.md](../filtering/README.md)). Most tools live in this **`validation/`** directory. **Polygonizing** classified rasters (mask pixels → GeoPackages) uses **`filtering/polygonize_mask_parallel.py`** — it is **not** under `validation/`; see [filtering/README.md](../filtering/README.md). **`reproject_raster_to_equal_area.py`** warps GeoTIFFs to the same equal-area CRS as vectors; **`merge_reprojected_tiles_by_year.py`** optionally mosaics regional tiles into one raster per year before polygonize or QA. After **`intersect_top_n_scars_with_classified.py`**, run **`calculate_jaccard_index.py`** on each hits GeoPackage to derive **B = unary_union(bᵢ)** and per-scar Jaccard metrics.

## Contents

| File | Purpose |
| --- | --- |
| `reproject_vector_to_equal_area.py` | Reprojects a vector layer to an equal-area CRS suitable for Chile at national level and annotates each feature with its area. |
| `reproject_raster_to_equal_area.py` | Warps GeoTIFFs to the same equal-area presets (parallel folder in / folder out). |
| `merge_reprojected_tiles_by_year.py` | Mosaics regional equal-area GeoTIFFs into one raster per calendar year (optional step before polygonize). |
| `split_vector_by_year.py` | Splits a vector layer into one GeoPackage per calendar year (default year column: `Season`). |
| `plot_area_distribution.py` | Plots the polygon-area distribution (in hectares): log x-axis histogram plus a linear-scale ruler for the same range. |
| `intersect_top_n_scars_with_classified.py` | Crosses a scar catalog with **polygonized** classified GeoPackages (produced with **`../filtering/polygonize_mask_parallel.py`**). Optional top N / `--by-year`; output: `scar` + `classified_hits` layers. |
| `calculate_jaccard_index.py` | From a hits GeoPackage (`--hits-gpkg`), one row per scar: **B = unary_union(bᵢ)**, **J = area(A∩B)/area(A∪B)**. Legacy mode: single intersection layer + reference/classified totals. |
| `spatial_validation_metrics.py` | Singh et al. (2015) closeness **D** per reference–segment pair; scar-level **TP/FP/FN**, commission/omission, Jaccard, Dice. See `requirements-spatial-validation.txt`. |
| `extract_top_fire_events.py` | Build a top-N fire-events layer from yearly national `chile_YYYY_events.gpkg` files (area range, optional year exclusions). |
| `sample_reference_scars.py` | Filter scars ≥ min ha (default 1000) and draw a random sample across the multi-year series (optional stratify-by-year). |
| `build_mixed_reference_catalog.py` | Merge UNIDOS (2013–2018) + GABAM (to 2022, skip 2019–2020) into one Albers catalog. |
| `run_unidos_classification_validation.py` | One fire-season year: UNIDOS_13_18 vs `classification_20260730` (season-to-season). |
| `run_unidos_validation_year.sh` | Cluster wrapper (default season 2017). |

## Sample design (≥ 1000 ha, random in the series)

### Option A — UNIDOS only (2013–2018)

```bash
# 1) Catalog in Chile Albers (areas in ha)
python validation/reproject_vector_to_equal_area.py \
  --input ~/validation/UNIDOS_13_18.shp \
  --output ~/validation/UNIDOS_13_18_albers.gpkg \
  --preset chile_albers

# 2) Eligible pool: Season 2013–2018, area >= 1000 ha → random sample
python validation/sample_reference_scars.py \
  --catalog ~/validation/UNIDOS_13_18_albers.gpkg \
  --year-column Season \
  --from-year 2013 --to-year 2018 \
  --min-ha 1000 \
  --sample-n 100 \
  --seed 42 \
  --stratify-by-year \
  --output ~/validation/samples/unidos_ge1000ha_n100_seed42.gpkg
```

### Option B — UNIDOS 2013–2018 + GABAM to 2022 (omit 2019–2020)

```bash
# 1) Merge references (Albers + area_ha + Season + source)
python validation/build_mixed_reference_catalog.py \
  --unidos ~/validation/UNIDOS_13_18.shp \
  --gabam ~/validation/GABAM_chile.shp \
  --unidos-year-column Season \
  --gabam-year-column year \
  --gabam-exclude-years 2019,2020 \
  --output ~/validation/mixed_unidos_gabam_albers.gpkg

# 2) Sample N=100 scars ≥1000 ha, stratified by year (2013–2018, 2021–2022)
python validation/sample_reference_scars.py \
  --catalog ~/validation/mixed_unidos_gabam_albers.gpkg \
  --year-column Season \
  --from-year 2013 --to-year 2022 \
  --min-ha 1000 \
  --sample-n 100 \
  --seed 42 \
  --stratify-by-year \
  --output ~/validation/samples/mixed_ge1000ha_n100_seed42.gpkg
```

Years in the pool: **2013–2018 (UNIDOS)** + **2021–2022 (GABAM)**. 2019 and 2020 are dropped from GABAM by default.

- **`--stratify-by-year`**: reparte N lo más pareja posible entre temporadas/años (recomendado).
- **Sin estratificar**: un sorteo global en toda la serie.
- El GPKG de muestra se usa después como `--catalog` del `intersect_…`.
- También en un solo paso:  
  `intersect_top_n_scars_with_classified.py --min-area-ha 1000 --sample-n 100 --seed 42 --stratify-by-year`

## UNIDOS 2013–2018 vs classification_20260806 (cluster)

Season product: `~/classification_20260806/burned_area_chile_temp_10_remap_{year}.tif`  
Reference: `~/validation/UNIDOS_13_18.shp` → sample ≥**1000 ha**, **N=100**, years **2013–2018**, seed 42.

```bash
cd ~/fire
git checkout feat/auxiliares-to-gee && git pull
conda activate mb_fuego
bash validation/run_unidos_validation_20260806.sh
```

Outputs under `~/validation/unidos_vs_20260806/`:
- `ref/unidos_ge1000ha_n100_seed42.gpkg` (muestra)
- `season_YYYY/{04_hits,05_jaccard,run_manifest.json}`
- `logs/season_YYYY.log`

One year only: `YEARS=2017 bash validation/run_unidos_validation_20260806.sh`

### SLURM (debug, 30 min)

```bash
cd ~/fire && git checkout feat/auxiliares-to-gee && git pull
mkdir -p ~/logs
sbatch validation/run_unidos_validation_20260806_slurm.sh

# smoke test 1 year on debug
YEARS=2017 sbatch validation/run_unidos_validation_20260806_slurm.sh
```

Logs: `~/logs/fire_val_unidos_<jobid>.out` / `.err`

## UNIDOS vs classification_20260730 (season-to-season)

Reference scars (`~/validation/UNIDOS_13_18.shp`) are tagged by **fire season** (`Season`).  
National MapBiomas season products are `~/classification_20260730/{Season}.tif` or `{Season}_remap.tif` (band 1 = burn).  
Compare **the same season id**, not calendar-reordered rasters.

```bash
cd ~/fire
git checkout feat/auxiliares-to-gee && git pull
conda activate mb_fuego

# Smoke test: season 2017
bash validation/run_unidos_validation_year.sh

# Another season
VALIDATE_YEAR=2016 bash validation/run_unidos_validation_year.sh
```

Outputs under `~/validation/unidos_vs_20260730_season/season_2017/`:
hits GPKG, Jaccard CSV, spatial metrics summary, `run_manifest.json`.

Notes:
- Polygonize output is `mapbiomas_chile_nat_{year}_albers_mask1.gpkg` (suffix required by the polygonizer).
- National mosaics are **clipped** to UNIDOS scars for that season (+5 km buffer) before polygonize — full-Chile polygonize OOMs.
- If a previous failed run left partial outputs, remove `~/validation/unidos_vs_20260730_season/season_2017/` and re-run, or use without `VALIDATE_SKIP_EXISTING=1`.

## Extract top fire events

### `extract_top_fire_events.py`

Select the largest events from national vector outputs (`vectorize/` → `polygons_chile/chile_YYYY_events.gpkg`).

```bash
python validation/extract_top_fire_events.py \
  --input-dir /path/to/polygons_chile \
  --output-gpkg /path/to/top50_incendios.gpkg \
  --from-year 2014 --to-year 2025 \
  --min-ha 200 --max-ha 5000 --top-n 50 \
  --exclude-years 2019
```

Output columns: `id`, `event_id`, `anio`, `area_m2`, `area_ha`, `fragment_count`, `max_gap_m`, `geometry`.

## Spatial validation (Singh et al. 2015)

### Environment

```bash
python -m pip install -r validation/requirements-spatial-validation.txt
```

Requires hits GeoPackages from `intersect_top_n_scars_with_classified.py` (layers `scar` + `classified_hits`).

### `spatial_validation_metrics.py`

**Pairwise** (each reference polygon × each intersecting classified segment): Over/Under-segmentation, **D**, **D_norm** (= 1 − D/√2).

**Scar summary** (union of all segments per scar, optional `--by-region`): TP, FP, FN, commission/omission, Jaccard, Dice, areas (ha), detection percentages; best/mean **D_norm** across pairs.

```bash
python validation/spatial_validation_metrics.py \
    --hits-dir "D:/MAPBIOMAS/FUEGO/hits_classified_20260512_by_year" \
    --hits-pattern "cicatrices_hits_classified_20260512_*.gpkg" \
    --output-dir "D:/MAPBIOMAS/FUEGO/spatial_metrics_classified_20260512" \
    --by-region \
    --aggregate-summary-csv "D:/MAPBIOMAS/FUEGO/spatial_metrics_classified_20260512/summary_by_year_region.csv"
```

## Equal-area reprojection (vectors)

### `reproject_vector_to_equal_area.py`
Reprojects a vector layer (shapefile, GPKG, GeoJSON, ...) to an equal-area CRS so that area calculations and intersections in square meters / hectares are accurate for the whole country. Adds `area_m2` and `area_ha` columns to the output.

Presets:

- `chile_albers` (default): Albers Conic Equal Area custom-tuned for Chile (`+proj=aea +lat_1=-18 +lat_2=-55 +lat_0=-37 +lon_0=-71 +datum=WGS84 +units=m`).
- `south_america_albers`: `ESRI:102033`, the standard South America Albers Equal Area Conic.

Any arbitrary CRS understood by `pyproj` (EPSG code, proj string, WKT) can be provided via `--target-crs` to override the preset.

```bash
python validation/reproject_vector_to_equal_area.py \
    --input /mnt/e/mapbiomas/fire/Cicatrices/cicatrices.shp \
    --output /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --preset chile_albers
```

Override the preset with a custom CRS:

```bash
python validation/reproject_vector_to_equal_area.py \
    --input input.shp \
    --output output.gpkg \
    --target-crs EPSG:32719
```

## Equal-area reprojection (rasters)

### `reproject_raster_to_equal_area.py`
Reads each GeoTIFF under `--input-dir` matching `--pattern` (default `*.tif`), warps all bands to the same **Chile Albers** (or `--preset` / `--target-crs`) as `reproject_vector_to_equal_area.py`, and writes `{stem}{suffix}.tif` into `--output-dir` (default suffix `_albers`). Uses **parallel** workers (`--workers`).

Default resampling is **nearest** (recommended for class masks / binary products). Use `--resampling bilinear` or `cubic` for continuous rasters.

Requires `rasterio`.

```bash
python validation/reproject_raster_to_equal_area.py \
    --input-dir /path/to/classified_filtered \
    --output-dir /path/to/classified_filtered_albers \
    --preset chile_albers \
    --resampling nearest \
    --workers 4
```

### `merge_reprojected_tiles_by_year.py`
After regional tiles share the same projected CRS (e.g. output of `reproject_raster_to_equal_area.py`), groups them by **calendar year** read from the filename stem (`--year-token-index`, default `3` for `b14_chile_rX_YYYY_...`) and writes one merged GeoTIFF per year to `--output-dir` as `{--output-stem}_{year}.tif` using `rasterio.merge`. All tiles in a year must share CRS and band count. Overlaps: `--method` `first` (default), `last`, `min`, or `max`. Parallelizes over **years** with `--workers` (one process per year).

Requires `rasterio`.

```bash
python validation/merge_reprojected_tiles_by_year.py \
    --input-dir /mnt/e/mapbiomas/fire/filtered_reprojected_20260512 \
    --output-dir /mnt/e/mapbiomas/fire/filtered_reprojected_20260512_by_year \
    --pattern "*.tif" \
    --output-stem filtered_20260512_albers \
    --method first \
    --workers 8
```

## Split by year

### `split_vector_by_year.py`
Reads a vector layer and writes **one GeoPackage per year**. The default year attribute is `Season` (typical for CONAF-style seasons); use `--year-column` for another field. Output files are `{prefix}_{year}.gpkg` (prefix defaults to the input filename stem). Each file contains a single layer whose name matches that stem. Rows with missing or unparseable years are skipped with a warning.

Typical flow: reproject to equal area (so `area_ha` exists), then split, then run `plot_area_distribution.py` per file (pass `--layer` equal to the file stem, since each yearly GPKG names its layer that way).

```bash
python validation/split_vector_by_year.py \
    --input /path/to/cicatrices_albers.gpkg \
    --output-dir /path/to/cicatrices_by_year
```

Optional arguments: `--year-column`, `--layer` (when reading a multi-layer input), `--prefix` (output filename prefix).

Plot area distributions for every yearly GPKG in a directory:

```bash
out_dir=/path/to/cicatrices_by_year/plots
mkdir -p "$out_dir"
for f in /path/to/cicatrices_by_year/*.gpkg; do
  stem=$(basename "$f" .gpkg)
  python validation/plot_area_distribution.py \
      --input "$f" \
      --output "$out_dir/${stem}_area_distribution.png" \
      --layer "$stem"
done
```

## Area distribution plot

### `plot_area_distribution.py`
Reads a vector layer (typically the output of `reproject_vector_to_equal_area.py`) and produces a histogram of polygon areas in hectares with a **logarithmic** x-axis and a **linear-scale ruler** below the same numeric range. Expects a column with areas in hectares (default `area_ha`). For GeoPackages produced by `split_vector_by_year.py`, set `--layer` to the filename without `.gpkg`.

```bash
python validation/plot_area_distribution.py \
    --input /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --output /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_area_distribution.png
```

## Scar catalog intersections

### `intersect_top_n_scars_with_classified.py`

#### Role

Reads one **scar catalog** (`--catalog`) and all classified polygon GeoPackages under `--classified-dir`. For each scar (optionally restricted by `--top-n` and `--area-column`), it keeps classified polygons whose geometry **intersects** the scar and whose **year parsed from the filename** matches the scar’s calendar year. Full classified geometries are preserved (not clipped). The script does **not** build `B = unary_union(bᵢ)`, nor `A ∩ B`, `A ∪ B`, nor spatial indices — use **`calculate_jaccard_index.py --hits-gpkg`** as the next step. Parallel execution via `--workers`.

**Run by year:** use **`--by-year`** with **`--output-dir`** (and optional **`--output-stem`**) to write **one GeoPackage per calendar year** present in the catalog (`{stem}_{year}.gpkg`). Each yearly run processes only scars with that `scar_year`. With **`--top-n`**, the *N* largest scars are chosen **within each year** (not globally). Alternatively, **`--year YYYY`** with a single **`--output`** restricts one run to that calendar year only; **`--top-n`** then applies within that year.

**`--top-n` without `--by-year`:** the *N* largest scars are chosen **globally** across the catalog (after optional `--year`), then intersected in one output file.

**Scar calendar year:** use `--year-column` to name the attribute that holds the year. If omitted, the script uses the first available among `year`, `Season` (typical CONAF-style), or `IgnDate` (ISO date string parsed for the year).

#### Organizing scars and classified data by year and region

A practical layout is:

- **Scars:** split by calendar year (e.g. with `split_vector_by_year.py`) so each `--catalog` holds only events for one year. That keeps runs small and matches season-based workflows. Alternatively, keep one national catalog and use **`--by-year`** so the script still writes **separate GeoPackages per year** from the same catalog.
- **Polygonized classified data:** produced with **`filtering/polygonize_mask_parallel.py`** (in the **`filtering/`** directory of this repo). Typical outputs: **one GeoPackage per MapBiomas tile** (region + year), or **one per yearly mosaic** after `merge_reprojected_tiles_by_year.py` and polygonize. Put every yearly GeoPackage the run needs in the same `--classified-dir`.

The script loads **every** `*.gpkg` in `--classified-dir`, groups them by **year** (fourth underscore-separated token in the stem), and for each scar opens only the files for `scar_year`. Within that year it uses every classified file listed for that year (multiple regions or a single mosaic).

You can also use a **single national catalog** (all years in one file) and a `--classified-dir` that contains several years — matching is always **per scar year**, so scars and tiles from different years are not mixed.

#### Layers in each output GeoPackage

Each result file — either a single `--output` path or `{--output-stem}_{year}.gpkg` when using `--by-year` — contains two layers:
- **`scar`:** one row per processed scar. All catalog columns are kept, plus `scar_id` (from `FireID` when present), `scar_year`, `scar_area_m2`, `scar_area_ha` (from geometry in the projected CRS). Geometry = **A**.
- **`classified_hits`:** one row per classified polygon **bᵢ** that intersects some A. Original polygon attributes are kept, plus `scar_id` (join key to `scar`), `scar_year`, `region`, `classified_file`. Geometry = full **bᵢ**. A scar with no hits has no rows here (it still appears in `scar`).

#### Classified filename convention

The stem must split on `_` into at least four segments. **Region** = third token, **year** = fourth (e.g. `b14_chile_r1_2013_...` keeps region `r1` and year `2013`). For **yearly mosaic** polygon outputs such as `filtered_20260512_albers_2013_mask1.gpkg`, the fourth token is still the calendar year (`2013`); the third token is a stem label (e.g. `albers`), not a MapBiomas `rX` code. Suffixes such as `_mask1` do not change token positions.

#### CRS

The catalog must use a **projected** CRS (the script rejects geographic CRSs so areas stay meaningful). Scars and classified polygons should share that CRS (e.g. Chile Albers: `reproject_vector_to_equal_area.py` on scars, `reproject_raster_to_equal_area.py` on classified rasters, then polygonize with **`filtering/polygonize_mask_parallel.py`**).

#### From rasters to polygon GeoPackages (script in `filtering/`)

**Polygonization is implemented in the `filtering/` package**, not in `validation/`: run [`../filtering/polygonize_mask_parallel.py`](../filtering/polygonize_mask_parallel.py) (documented in [filtering/README.md](../filtering/README.md)). It writes **one GeoPackage per input GeoTIFF**; year and region tokens in the stem stay aligned with the filename convention above when you use MapBiomas-style names (or yearly mosaic stems such as `filtered_*_albers_2013.tif`).

From the repository root:

```bash
python filtering/polygonize_mask_parallel.py \
    --input-dir /path/to/classified_filtered_agriculture_albers \
    --output-dir /path/to/classified_filtered_agriculture_albers_polygons \
    --pattern "b14_chile_r*_2017_*_albers.tif" \
    --mask-value 1 \
    --workers 4
```

Repeat per year or use a broader `--pattern` if the folder is already one year only.

#### Examples

**One national catalog, one output file per calendar year** (e.g. top 10 scars per year, 4 workers):

```bash
python validation/intersect_top_n_scars_with_classified.py \
    --catalog /mnt/e/mapbiomas/fire/validation/cicatrices/incendios_agrupados_albers.gpkg \
    --layer incendios_agrupados_albers \
    --by-year \
    --output-dir /mnt/e/mapbiomas/fire/validation/cicatrices/hits_20260512_by_year \
    --output-stem incendios_hits_20260512 \
    --top-n 10 \
    --classified-dir /mnt/e/mapbiomas/fire/filtered_reprojected_20260512_by_year_polygons \
    --workers 4
```

**Single calendar year, one file** (use `--output`, not `--by-year`):

```bash
python validation/intersect_top_n_scars_with_classified.py \
    --catalog /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --year 2017 \
    --top-n 50 \
    --classified-dir /path/to/all_years_polygons \
    --output /mnt/e/mapbiomas/fire/Cicatrices/hits_2017_top50.gpkg \
    --workers 15
```

**Yearly scar catalog** (e.g. output of `split_vector_by_year.py`) + polygon folder for that same year (all regions):

```bash
python validation/intersect_top_n_scars_with_classified.py \
    --catalog /mnt/e/mapbiomas/fire/Cicatrices/by_year/cicatrices_albers_2017.gpkg \
    --layer cicatrices_albers_2017 \
    --classified-dir /path/to/classified_2017_albers_polygons \
    --top-n 50 \
    --output /mnt/e/mapbiomas/fire/Cicatrices/hits_2017_top50.gpkg \
    --workers 15
```

Use `--layer` only when the GeoPackage names its layer differently from the filename stem (as with `split_vector_by_year.py`). If the default layer works, omit `--layer`.

**Single national catalog** and a classified directory with multiple years:

```bash
python validation/intersect_top_n_scars_with_classified.py \
    --catalog /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --top-n 50 \
    --classified-dir /path/to/all_years_polygons \
    --output /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_top50_hits.gpkg \
    --workers 15
```

Only tiles whose parsed year matches each scar’s `scar_year` are read for that scar.

## Jaccard index (per scar)

### `calculate_jaccard_index.py`

#### Role

Consumes a **hits GeoPackage** written by `intersect_top_n_scars_with_classified.py` (`--hits-gpkg`), reads layers **`scar`** (polygon **A**) and **`classified_hits`** (polygons **bᵢ**). For each `scar_id`, builds **B = unary_union(bᵢ)** over hits with that id, then:

- **intersection_area** = area(**A** ∩ **B**)
- **union_area** = area(**A**) + area(**B**) − intersection_area (= area(**A** ∪ **B**))
- **Jaccard** = intersection_area / union_area

If a scar has **no** rows in `classified_hits`, **B** is empty: **J = 0** and union_area equals area(**A**). Output is one CSV row per scar (`scar_id`, optional `scar_year`, areas in m², `jaccard_index`, `jaccard_percent`).

**Legacy mode** (mutually exclusive with `--hits-gpkg`): pass `--intersection` plus `--reference` / `--classified` (or `--reference-area-m2` / `--classified-area-m2`) for a **single** global overlap metric, as when **A** and **B** are each one polygon or pre-aggregated layers.

#### Example (one hits file)

```bash
python validation/calculate_jaccard_index.py \
    --hits-gpkg /mnt/e/mapbiomas/fire/validation/cicatrices/hits_20260512_by_year/incendios_hits_20260512_2017.gpkg \
    --output-csv /mnt/e/mapbiomas/fire/validation/cicatrices/jaccard_2017.csv
```

#### Example (loop over yearly hits from `--by-year`)

```bash
out_dir=/mnt/e/mapbiomas/fire/validation/cicatrices/jaccard_20260512
mkdir -p "$out_dir"
for f in /mnt/e/mapbiomas/fire/validation/cicatrices/hits_20260512_by_year/*.gpkg; do
  stem=$(basename "$f" .gpkg)
  python validation/calculate_jaccard_index.py \
      --hits-gpkg "$f" \
      --output-csv "$out_dir/${stem}_jaccard.csv"
done
```

Optional: `--scar-layer` / `--classified-hits-layer` if layer names differ from the defaults `scar` and `classified_hits`.
