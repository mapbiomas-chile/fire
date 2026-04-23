# Validation

Scripts to validate and prepare reference layers used to evaluate the burned-area pipeline (see [../classification/README.md](../classification/README.md) and [../filtering/README.md](../filtering/README.md)). These tools operate on vector reference data (e.g. fire-scar polygons) independently of any single pipeline stage.

## Contents

| File | Purpose |
| --- | --- |
| `reproject_to_equal_area.py` | Reprojects a vector layer to an equal-area CRS suitable for Chile at national level and annotates each feature with its area. |
| `plot_area_distribution.py` | Plots the polygon-area distribution (in hectares) of a vector layer with linear and log x-axis panels. |
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

## Area distribution plot

### `plot_area_distribution.py`
Reads a vector layer (typically the output of `reproject_to_equal_area.py`) and produces a two-panel figure of the polygon-area distribution. The left panel uses a linear x-axis and the right panel a log-10 x-axis with tick labels kept in their linear-equivalent hectare values. The log panel is useful when areas span several orders of magnitude (typical for fire-scar sizes).

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
