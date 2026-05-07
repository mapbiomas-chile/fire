# Validation

Scripts to validate and prepare reference layers used to evaluate the burned-area pipeline (see [../classification/README.md](../classification/README.md) and [../filtering/README.md](../filtering/README.md)). These tools operate on vector reference data (e.g. fire-scar polygons) independently of any single pipeline stage.

## Contents

| File | Purpose |
| --- | --- |
| `reproject_to_equal_area.py` | Reprojects a vector layer to an equal-area CRS suitable for Chile at national level and annotates each feature with its area. |
| `split_vector_by_year.py` | Splits a vector layer into one GeoPackage per calendar year (default year column: `Season`). |
| `plot_area_distribution.py` | Plots the polygon-area distribution (in hectares): log x-axis histogram plus a linear-scale ruler for the same range. |
| `filter_large_polygons.py` | Keeps only polygons whose area exceeds a minimum threshold in hectares. |
| `export_large_scars_individual.py` | Exports one GeoPackage per large scar, named by scar ID, area and year. |
| `intersect_large_scars_with_classified.py` | Intersects each large scar with classified polygons from the same year and writes one GeoPackage per scar. |

## Equal-area reprojection

### `reproject_to_equal_area.py`
Reprojects a vector layer (shapefile, GPKG, GeoJSON, ...) to an equal-area CRS so that area calculations and intersections in square meters / hectares are accurate for the whole country. Adds `area_m2` and `area_ha` columns to the output.

Presets:

- `chile_albers` (default): Albers Conic Equal Area custom-tuned for Chile (`+proj=aea +lat_1=-18 +lat_2=-55 +lat_0=-37 +lon_0=-71 +datum=WGS84 +units=m`).
- `south_america_albers`: `ESRI:102033`, the standard South America Albers Equal Area Conic.

Any arbitrary CRS understood by `pyproj` (EPSG code, proj string, WKT) can be provided via `--target-crs` to override the preset.

```bash
python validation/reproject_to_equal_area.py \
    --input /mnt/e/mapbiomas/fire/Cicatrices/cicatrices.shp \
    --output /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --preset chile_albers
```

Override the preset with a custom CRS:

```bash
python validation/reproject_to_equal_area.py \
    --input input.shp \
    --output output.gpkg \
    --target-crs EPSG:32719
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
Reads a vector layer (typically the output of `reproject_to_equal_area.py`) and produces a histogram of polygon areas in hectares with a **logarithmic** x-axis and a **linear-scale ruler** below the same numeric range. Expects a column with areas in hectares (default `area_ha`). For GeoPackages produced by `split_vector_by_year.py`, set `--layer` to the filename without `.gpkg`.

```bash
python validation/plot_area_distribution.py \
    --input /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --output /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_area_distribution.png
```

## Large-polygon filter

### `filter_large_polygons.py`
Reads a vector layer with an area column in hectares (by default `area_ha`) and writes only the polygons whose area is greater than a minimum threshold. The default threshold is `5000 ha`.

```bash
python validation/filter_large_polygons.py \
    --input /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_albers.gpkg \
    --output /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_gt_5000ha.gpkg
```

## Individual large scars

### `export_large_scars_individual.py`
Reads a GeoPackage of large scars and exports one GeoPackage per scar. Output filenames are based on `FireID`, rounded `area_ha`, and the year extracted from `IgnDate`.

```bash
python validation/export_large_scars_individual.py \
    --input /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_gt_5000ha.gpkg \
    --output-dir /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_grandes
```

## Large-scar intersections

### `intersect_large_scars_with_classified.py`
For each large scar GeoPackage, finds the classified polygon GeoPackages from the same year, intersects them, adds the `region` parsed from the classified filename, and writes one output GeoPackage per scar. Scars are processed in parallel with `--workers`.

```bash
python validation/intersect_large_scars_with_classified.py \
    --scars-dir /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_grandes \
    --classified-dir /mnt/e/mapbiomas/fire/classified_filtered_reprojected_polygons \
    --output-dir /mnt/e/mapbiomas/fire/Cicatrices/cicatrices_grandes_intersections \
    --workers 15
```
