# Shared library (`lib/`)

[Español](README.es.md) | **English**

Importable Python helpers shared by `filtering/`, `vectorize/`, and related tools. Pipelines and path orchestration live in those package directories—**not** here.

---

## Modules

| Module | Purpose |
|--------|---------|
| `lulc_stability.py` | Multi-year LULC stability windows for non-burnable masks |
| `tile_metadata.py` | Parse year / region tokens from MapBiomas filenames |
| `vectorize.py` | Core polygonize of binary burn masks |
| `vectorize_filtered_classified.py` | Per-tile vectorization CLI |
| `vectorize_national_by_year.py` | National yearly merge, sieve, event grouping |
| `raster_by_year.py` | Merge regional rasters into yearly mosaics |
| `sieve_burn_mask.py` | Remove small connected burn components |
| `group_fire_events.py` | Aggregate nearby polygons into multipolygon events |

---

## Usage

```python
from pathlib import Path
from lib.vectorize import polygonize_raster_file

polygonize_raster_file(Path("tile.tif"), Path("tile_burn.gpkg"))
```

Pipeline entry points:

```bash
source vectorize/cluster_paths.env
bash vectorize/run_vectorize_pipeline.sh
bash vectorize/run_vectorize_national_pipeline.sh
```

See [vectorize/README.md](../vectorize/README.md).

---

## Dependencies

`numpy`, `rasterio`, `geopandas`, `shapely`, `scipy` (for sieve / morphology helpers).
