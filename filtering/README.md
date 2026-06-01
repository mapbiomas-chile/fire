# Filtering

**Leftraru interactivo (sin sbatch):** [LOCAL.md](LOCAL.md) — `cluster_paths.leftraru.env.example` → `cluster_paths.env`, luego `bash filtering/run_filtering_pipeline.sh`.

**Cola SLURM (NLHPC):** [CLUSTER.md](CLUSTER.md) — `sbatch filtering/run_filtering_pipeline_slurm.sh`.

Utilities that run after the burned-area classifier (see [../classification/README.md](../classification/README.md)). The goal of this stage is to turn raw classified rasters into clean, analysis-ready products: per-year masks of non-burnable classes, filtered rasters, polygonized fire scars and summary statistics used to pick filtering thresholds.

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

### `filter_temporal_first_burn_year.py`
Removes **persistence** across years: if a pixel is burned in 2017, 2018 and 2019, only **2017** keeps the burn (same spatial tile, years processed from `--from-year`). Grouping is by filename with the year token replaced. Run **after** LULC / class filtering.

```bash
python filtering/filter_temporal_first_burn_year.py \
  --input-dir /home/flepin/filtering_work_local/classified_filtered \
  --output-dir /home/flepin/filtering_work_local/classified_first_burn \
  --from-year 2013 --to-year 2025 --workers 4
```

### `polygonize_mask_parallel.py`
Converts the burned-area pixels (`mask value = 1` by default) of each filtered raster into polygons via `rasterio.features.shapes`, writing one GeoPackage per input. Connected mask pixels become connected polygons. Runs in parallel using `ProcessPoolExecutor`.

## Statistics

### `summarize_histograms_by_region.py`
Groups polygon files by region code extracted from the filename (`rX`) and writes one histogram per file under `region_<X>/histogramas/`. Only regions `1`, `2`, `4` and `6` are rendered.

## Thresholding

### `filter_polygons_by_threshold.py`
Reads polygon GeoPackages and keeps only polygons larger than the selected minimum area (in hectares). Produces a single merged GeoPackage with the surviving polygons.
