# Auxiliary library (`lib/`)

Reusable Python helpers for MapBiomas Fire Chile. **Pipelines** live in their own folders (`classification/`, `filtering/`, `vectorize/`); this folder holds importable code they share.

## Contents

| Module | Purpose |
|--------|---------|
| `tile_metadata.py` | Parse calendar year, region (`r1`, `r2`, …) and tile id from MapBiomas filenames |
| `vectorize.py` | Polygonize binary burn masks (raster → GeoDataFrame / GeoPackage) |
| `vectorize_filtered_classified.py` | Low-level CLI (called by `vectorize/run_vectorize_pipeline.sh`) |

## Vectorization pipeline

To vectorize **after** classification + filtering, use the dedicated auxiliary pipeline:

```bash
cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
source vectorize/cluster_paths.env
bash vectorize/run_vectorize_pipeline.sh
```

On NLHPC: `sbatch vectorize/run_vectorize_pipeline_slurm.sh`

Full docs: [vectorize/README.md](../vectorize/README.md).

## Dependencies

`numpy`, `rasterio`, `geopandas`, `shapely`

## Use from Python

```python
from pathlib import Path
from lib.vectorize import polygonize_raster_file

polygonize_raster_file(Path("tile.tif"), Path("tile_burn.gpkg"))
```
