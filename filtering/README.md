# Filtering

Utilities that run after the burned-area classifier (see [../classification/README.md](../classification/README.md)). The goal of this stage is to turn raw classified rasters into clean, analysis-ready products: per-year masks of non-burnable classes, filtered rasters, polygonized fire scars and summary statistics used to pick filtering thresholds.

## Cluster execution (NLHPC, class masks + raster filter only)

Same pattern as [`../classification/run_classify_fire_model_slurm_v2.sh`](../classification/run_classify_fire_model_slurm_v2.sh): NLHPC `main`, Python absoluto `mb_fuego`, verificaciones de rutas/paquetes, logs en `/home/%u/logs/`.

```bash
sbatch /home/flepin/fire/filtering/run_filtering_pipeline_slurm.sh
```

Optional args: `CLASSIFIED_DIR` `WORK_ROOT` `STEPS` (defaults: `/home/flepin/classi_v2`, `/home/flepin/filtering_work`, `all`). Edit `LULC_STACK` at the top of the SLURM script if needed.

`run_filtering_pipeline.sh` holds the pipeline logic; **submit** `run_filtering_pipeline_slurm.sh` on NLHPC. Does not polygonize or apply area thresholds.

## Suggested pipeline

A typical post-classification run chains these steps:

1. **Build non-burnable masks** — use `create_accumulated_class_masks.py`, `create_yearly_masks.py` and `create_total_masks_by_year.py` to combine yearly land-cover layers into one binary mask per year (`1` = remove, `0` = keep).
2. **Filter classified rasters** — run `filter_classified_parallel.py` to apply the year-matched mask to every classified GeoTIFF.
3. **Polygonize** — run `polygonize_mask_parallel.py` on filtered GeoTIFFs (one GeoPackage per input raster). Use the same CRS as your tiles (often geographic from the classifier until you reproject elsewhere for area-based work).
4. **Summarize** — render region-grouped polygon-area histograms with `summarize_histograms_by_region.py` to inspect the distribution and pick a minimum-area threshold.
5. **Apply thresholds** — drop small polygons with `filter_polygons_by_threshold.py` using the chosen minimum area.

For optional **equal-area reprojection** of classified tiles or **yearly mosaics** before polygonize / validation against reference vectors, see [../validation/README.md](../validation/README.md) (`reproject_raster_to_equal_area.py`, `merge_reprojected_tiles_by_year.py`).

Auxiliary scripts (GEE downloads, tile listing, and metadata inspection) support the steps above and can be used standalone.

## Mask building

### `create_accumulated_class_masks.py`
For a selected land-cover class, produces one accumulated mask by OR-ing the class across all bands (years) of the input raster. The hard-coded `CLASS_SPECS` list covers rocky outcrop (`29`), sand/beach/dune (`23`), salt flat (`61`), ice/snow (`34`) and other non-vegetated areas (`25`), each written to a fixed `mascara_<name>_acumulado.tif` filename.

### `create_yearly_masks.py`
Writes one binary mask per year for each of these MapBiomas-style classes (read from the corresponding band of the input multi-year stack): river/lake **rio_lago** (`33`), **infraestructura** (`24`), **agricultura** (`15`) and **pastura** (`18`). Outputs use the filenames expected by `create_total_masks_by_year.py` (`mascara_<class>_<year>.tif`). Parallelizes by calendar year with `--workers` (one process per year by default: `cpu_count - 1`).

### `create_total_masks_by_year.py`
Combines all accumulated masks from `create_accumulated_class_masks.py` with the yearly thematic masks from `create_yearly_masks.py` (río/lago, infraestructura, agricultura, pastura) into one `mascara_total_<year>.tif` per year. The output is consumed by `filter_classified_parallel.py`. For a split directory tree, use `--mascaras-root` (`acumuladas/`, `by_year/`, `totales/`) and `--workers` to parallelize by year.

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
