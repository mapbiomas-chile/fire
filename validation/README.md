# Validation

Scripts to validate and prepare reference layers used to evaluate the burned-area pipeline (see [../classification/README.md](../classification/README.md) and [../filtering/README.md](../filtering/README.md)). Most tools work on **vector** reference data (e.g. fire-scar polygons). **`reproject_raster_to_equal_area.py`** warps classified **GeoTIFFs** to the same equal-area CRS as vectors; **`merge_reprojected_tiles_by_year.py`** optionally mosaics regional tiles into one raster per year before polygonize or QA.

## Contents

| File | Purpose |
| --- | --- |
| `reproject_vector_to_equal_area.py` | Reprojects a vector layer to an equal-area CRS suitable for Chile at national level and annotates each feature with its area. |
| `reproject_raster_to_equal_area.py` | Warps GeoTIFFs to the same equal-area presets (parallel folder in / folder out). |
| `merge_reprojected_tiles_by_year.py` | Mosaics regional equal-area GeoTIFFs into one raster per calendar year (optional step before polygonize). |
| `split_vector_by_year.py` | Splits a vector layer into one GeoPackage per calendar year (default year column: `Season`). |
| `plot_area_distribution.py` | Plots the polygon-area distribution (in hectares): log x-axis histogram plus a linear-scale ruler for the same range. |
| `intersect_top_n_scars_with_classified.py` | Crosses a scar vector catalog with polygonized classified tiles: for each scar (optionally top N by area), finds intersecting classified polygons by year and writes one GeoPackage (`scar` + `classified_hits`) for downstream indices. |

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

Reads one **scar catalog** (`--catalog`) and all classified polygon GeoPackages under `--classified-dir`. For each scar (optionally restricted by `--top-n` and `--area-column`), it keeps classified polygons whose geometry **intersects** the scar and whose **year parsed from the filename** matches the scar’s calendar year. Full classified geometries are preserved (not clipped). The script does **not** build `B = unary_union(bᵢ)`, nor `A ∩ B`, `A ∪ B`, nor spatial indices — those belong in a downstream step (e.g. Jaccard). Parallel execution via `--workers`.

**Run by year:** use **`--by-year`** with **`--output-dir`** (and optional **`--output-stem`**) to write **one GeoPackage per calendar year** present in the catalog (`{stem}_{year}.gpkg`). Each yearly run processes only scars with that `scar_year`. With **`--top-n`**, the *N* largest scars are chosen **within each year** (not globally). Alternatively, **`--year YYYY`** with a single **`--output`** restricts one run to that calendar year only; **`--top-n`** then applies within that year.

**`--top-n` without `--by-year`:** the *N* largest scars are chosen **globally** across the catalog (after optional `--year`), then intersected in one output file.

**Scar calendar year:** use `--year-column` to name the attribute that holds the year. If omitted, the script uses the first available among `year`, `Season` (typical CONAF-style), or `IgnDate` (ISO date string parsed for the year).

#### Organizing scars and classified data by year and region

A practical layout is:

- **Cicatrices:** split by calendar year (e.g. with `split_vector_by_year.py`) so each `--catalog` holds only events for one year. That keeps runs small and matches how you study seasons. Alternatively, keep one national catalog and use **`--by-year`** so the script still writes **separate GeoPackages per year** from the same catalog.
- **Clasificado polygonizado:** typically **one GeoPackage per MapBiomas tile** (region + year), or **one per yearly mosaic** after `merge_reprojected_tiles_by_year.py` + [`filtering/polygonize_mask_parallel.py`](../filtering/polygonize_mask_parallel.py). Put every yearly GPKG the run needs in the same `--classified-dir`.

The script loads **every** `*.gpkg` in `--classified-dir`, groups them by **year** (fourth underscore-separated token in the stem), and for each scar opens only the files for `scar_year`. Within that year it uses every classified file listed for that year (multiple regions or a single mosaic).

You can also use a **single national catalog** (all years in one file) and a `--classified-dir` that contains several years — matching is always **per scar year**, so scars and tiles from different years are not mixed.

#### Layers in each output GeoPackage

Each result file — either a single `--output` path or `{--output-stem}_{year}.gpkg` when using `--by-year` — contains two layers:
- **`scar`:** one row per processed scar. All catalog columns are kept, plus `scar_id` (from `FireID` when present), `scar_year`, `scar_area_m2`, `scar_area_ha` (from geometry in the projected CRS). Geometry = **A**.
- **`classified_hits`:** one row per classified polygon **bᵢ** that intersects some A. Original polygon attributes are kept, plus `scar_id` (join key to `scar`), `scar_year`, `region`, `classified_file`. Geometry = full **bᵢ**. A scar with no hits has no rows here (it still appears in `scar`).

#### Classified filename convention

The stem must split on `_` into at least four segments. **Region** = third token, **year** = fourth (e.g. `b14_chile_r1_2013_...` keeps region `r1` and year `2013`). For **yearly mosaic** polygon outputs such as `filtered_20260512_albers_2013_mask1.gpkg`, the fourth token is still the calendar year (`2013`); the third token is a stem label (e.g. `albers`), not a MapBiomas `rX` code. Suffixes such as `_mask1` do not change token positions.

#### CRS

The catalog must use a **projected** CRS (the script rejects geographic CRSs so areas stay meaningful). Scars and classified polygons should share that CRS (e.g. Chile Albers: `reproject_vector_to_equal_area.py` on scars, `reproject_raster_to_equal_area.py` before polygonize on classified rasters).

#### From rasters to polygon GeoPackages

Use [`filtering/polygonize_mask_parallel.py`](../filtering/polygonize_mask_parallel.py) (see [filtering/README.md](../filtering/README.md)). One output GeoPackage per warped GeoTIFF; year and region tokens stay aligned with the convention above.

Polygonize one year and all regions (adjust `--pattern` to your tile naming):

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
