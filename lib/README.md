# Auxiliary library (`lib/`)

Reusable Python helpers for MapBiomas Fire Chile. **Pipelines** live in their own folders (`classification/`, `filtering/`, `vectorize/`); this folder holds importable code they share.

## Contents

| Module | Purpose |
|--------|---------|
| `lulc_stability.py` | 4-year LULC stability windows for yearly non-burnable masks (A2) |
| `tile_metadata.py` | Parse calendar year, region (`r1`, `r2`, …) and tile id from MapBiomas filenames |
| `vectorize.py` | Polygonize binary burn masks (raster → GeoDataFrame / GeoPackage) |
| `vectorize_filtered_classified.py` | Per-tile vectorization CLI |
| `vectorize_national_by_year.py` | Merge tiles by year → polygonize Chile → group events |
| `raster_by_year.py` | Merge regional rasters into one GeoTIFF per calendar year |
| `sieve_burn_mask.py` | Remove small connected burn components from binary masks (pre-vectorize sieve) |
| `group_fire_events.py` | Group nearby polygons into multipolygon fire events; pixel/ha fragment filters |

## Vectorization pipeline

To vectorize **after** classification + filtering, use the dedicated auxiliary pipeline:

```bash
cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
source vectorize/cluster_paths.env
bash vectorize/run_vectorize_pipeline.sh
```

On NLHPC: `sbatch vectorize/run_vectorize_pipeline_slurm.sh`

Full docs: [vectorize/README.md](../vectorize/README.md).

**National (Chile-wide by year):** [vectorize/run_vectorize_national_pipeline.sh](../vectorize/run_vectorize_national_pipeline.sh) — merge → sieve (112 px) → polygonize → fragment filter (112 px) → group scars within 200 m.

## Dependencies

`numpy`, `rasterio`, `geopandas`, `shapely`, `scipy` (sieve / morphological helpers)

## Use from Python

```python
from pathlib import Path
from lib.vectorize import polygonize_raster_file

polygonize_raster_file(Path("tile.tif"), Path("tile_burn.gpkg"))
```
