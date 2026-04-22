# Utilities

Auxiliary scripts that support the burned-area pipeline (see [../classification/README.md](../classification/README.md) and [../filtering/README.md](../filtering/README.md)). These tools are independent of any single stage: they download Earth Engine assets, list and mosaic raster tiles, and inspect GeoTIFF metadata.

## Contents

| File | Purpose |
| --- | --- |
| `download_regiones_fuego_asset.py` | Exports the `regiones_fuego_chile_v1` FeatureCollection to Google Drive. |
| `list_intersecting_tiles.py` | Lists `.tif` tiles whose bounding box intersects a GeoJSON convex hull. |
| `mosaic_subset_clip_bbox.py` | Merges a subset of TIFF tiles and clips the mosaic to a bounding box. |
| `print_tif_metadata.py` | Prints key GeoTIFF metadata for quick compatibility checks. |

## GEE downloads

### `download_regiones_fuego_asset.py`
Exports the `regiones_fuego_chile_v1` FeatureCollection to Google Drive as GeoJSON or GPKG. Configuration is hard-coded at the top of the file (asset ID, project ID, Drive folder, output filename and format).

## Tile selection and mosaicking

### `list_intersecting_tiles.py`
Builds the convex hull of a GeoJSON polygon layer and lists every `.tif` in a tiles directory whose bounding box intersects that hull. Used to pick the subset of tiles that needs processing for a given set of fire regions.

### `mosaic_subset_clip_bbox.py`
Merges the TIFFs listed by `list_intersecting_tiles.py` and clips the resulting mosaic to the bounding box of the fire-regions convex hull (region 5 excluded), producing a single mosaic ready for inspection or further processing.

## Raster inspection

### `print_tif_metadata.py`
Prints key GeoTIFF metadata (CRS, transform, size, dtype, band count) as a quick compatibility check when stitching rasters from different sources.

```bash
python print_tif_metadata.py <path/to/raster.tif>
```
