# Utilities

Auxiliary scripts that support the burned-area pipeline (see [../classification/README.md](../classification/README.md) and [../filtering/README.md](../filtering/README.md)). These tools are independent of any single stage: they download Google Earth Engine (GEE) assets, list and mosaic raster tiles, and inspect GeoTIFF metadata.

## Contents

| Script | Description |
| --- | --- |
| `download_regiones_fuego_asset.py` | Export the `regiones_fuego_chile_v1` FeatureCollection from GEE to Google Drive. |
| `fire_regions_bbox_geojson.py` | **Single entry point** for fire-region geometry: convex hull with one region excluded → axis-aligned bbox envelope → optional GeoJSON export; same helpers used by tile listing and mosaicking. |
| `list_intersecting_tiles.py` | List `.tif` tiles whose bounding box intersects the convex hull of fire regions (excluding the configured region). |
| `mosaic_subset_clip_bbox.py` | Merge TIFF tiles from a subset folder and clip the mosaic to that hull’s bounding box envelope. |
| `print_tif_metadata.py` | Print key GeoTIFF metadata for quick compatibility checks. |

## Google Earth Engine downloads

### `download_regiones_fuego_asset.py`

Exports the `regiones_fuego_chile_v1` FeatureCollection to Google Drive as GeoJSON or GPKG. Configuration is hard-coded at the top of the file (asset ID, project ID, Drive folder, output filename and format). File names keep the `regiones_fuego_*` prefix to match the Earth Engine asset id.

## Fire regions bbox (`fire_regions_bbox_geojson.py`)

One script implements the full pipeline:

1. Read the fire-regions vector layer.
2. Drop features whose `--exclude-region` matches (default region `5`).
3. Build the **convex hull** of the remaining polygons.
4. Take its **axis-aligned bounding box** (rectangle = hull envelope).
5. When run as CLI: write that polygon to **`--output`** GeoJSON and store numeric bounds (`minx`, `miny`, `maxx`, `maxy`) on the feature.

Importable helpers (used by other utilities without running the CLI):

| Function | Role |
| --- | --- |
| `convex_hull_excluding_region` | Convex hull geometry and CRS — `list_intersecting_tiles.py` uses this for tile intersection tests. |
| `bbox_envelope_excluding_region` | Bbox rectangle + CRS — `mosaic_subset_clip_bbox.py` uses this as merge bounds. |

### CLI example

```bash
python3 utilities/fire_regions_bbox_geojson.py \
  --geojson path/to/regiones_fuego.geojson \
  --output path/to/fire_regions_bbox.geojson
```

## Tile listing and mosaicking

### `list_intersecting_tiles.py`

Calls `fire_regions_bbox_geojson.convex_hull_excluding_region`, then lists every `.tif` under `--tiles-dir` whose bounding box intersects that hull.

### `mosaic_subset_clip_bbox.py`

Merges TIFFs under `--subset-dir` and clips using rasterio merge bounds. Supply either **`--geojson`** (fire regions → bbox envelope computed as above) or **`--bbox-geojson`** with the polygon written by `fire_regions_bbox_geojson.py`.

## Raster inspection

### `print_tif_metadata.py`

Prints CRS, transform, size, dtype, and band count as a quick compatibility check when combining rasters from different sources.

```bash
python3 print_tif_metadata.py <path/to/raster.tif>
```
