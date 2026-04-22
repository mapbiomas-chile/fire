# Filtering

Utilities that run after the burned-area classifier (see [../classification/README.md](../classification/README.md)). The goal of this stage is to turn raw classified rasters into clean, analysis-ready products: per-year masks of non-burnable classes, filtered rasters, polygonized fire scars and summary statistics used to pick filtering thresholds.

## Suggested pipeline

A typical post-classification run chains these steps:

1. **Build non-burnable masks** — use `create_accumulated_class_masks.py`, `create_yearly_water_infra_masks.py` and `create_total_masks_by_year.py` to combine yearly land-cover layers into one binary mask per year (`1` = remove, `0` = keep).
2. **Filter classified rasters** — run `filter_classified_parallel.py` to apply the year-matched mask to every classified tile.
3. **Polygonize** — turn the filtered rasters into one GeoPackage per tile using `polygonize_mask_parallel.py`.
4. **Summarize** — render region-grouped polygon-area histograms with `summarize_histograms_by_region.py` to inspect the distribution and pick a minimum-area threshold.
5. **Apply thresholds** — drop small polygons with `filter_polygons_by_threshold.py` using the chosen minimum area.

Auxiliary scripts (GEE downloads, tile listing, mosaicking and metadata inspection) support the steps above and can be used standalone.

## Mask building

### `create_accumulated_class_masks.py`
For a selected land-cover class, produces one accumulated mask by OR-ing the class across all bands (years) of the input raster. The hard-coded `CLASS_SPECS` list covers rocky outcrop (`29`), sand/beach/dune (`23`), salt flat (`61`), ice/snow (`34`) and other non-vegetated areas (`25`), each written to a fixed `mascara_<name>_acumulado.tif` filename.

### `create_yearly_water_infra_masks.py`
Writes one binary mask per year for the river/lake (`33`) and infrastructure (`24`) classes. These are time-varying and therefore kept as yearly rasters instead of accumulated ones.

### `create_total_masks_by_year.py`
Combines the accumulated masks produced by `create_accumulated_class_masks.py` with the yearly water/infrastructure masks into one per-year total mask. The output of this step is the mask consumed by `filter_classified_parallel.py`.

## Raster filtering

### `filter_classified_parallel.py`
Applies a year-specific binary mask (`1` = remove, `0` = keep) to every classified GeoTIFF in a directory. The script extracts the year from each tile's name with the regex `20\d{2}`, selects the matching yearly mask, reprojects if needed and writes filtered tiles plus a per-file JSON report. Parallelized across cores with `multiprocessing.Pool`.

### `polygonize_mask_parallel.py`
Converts the burned-area pixels (`mask value = 1` by default) of each filtered raster into polygons via `rasterio.features.shapes`, writing one GeoPackage per input. Connected mask pixels become connected polygons. Runs in parallel using `ProcessPoolExecutor`.

## Statistics

### `summarize_histograms_by_region.py`
Groups polygon files by region code extracted from the filename (`rX`) and writes one histogram per file under `region_<X>/histogramas/`. Only regions `1`, `2`, `4` and `6` are rendered.

## Thresholding

### `filter_polygons_by_threshold.py`
Reads polygon GeoPackages and keeps only polygons larger than the selected minimum area (in hectares). Produces a single merged GeoPackage with the surviving polygons.
