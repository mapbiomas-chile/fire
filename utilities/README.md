# Utilities

[Español](README.es.md) | **English**

Supporting scripts that are **stage-independent**: Google Earth Engine export helpers, fire-region geometry, tile listing, mosaicking of GeoTIFF subsets, and quick GeoTIFF metadata inspection.

They do not replace the main classification → filtering → vectorize chain ([root README](../README.md)).

---

## Inventory

| Script | Role |
|--------|------|
| `download_regiones_fuego_asset.py` | Export `regiones_fuego_chile_v1` FeatureCollection from GEE to Drive |
| `fire_regions_bbox_geojson.py` | Convex hull (excluding a region, default 5) → bbox envelope → optional GeoJSON |
| `list_intersecting_tiles.py` | List `.tif` tiles intersecting the fire-region hull |
| `mosaic_subset_clip_bbox.py` | Merge subset tiles clipped to that bbox |
| `print_tif_metadata.py` | Print CRS, transform, size, dtype, band count |

Importable helpers from `fire_regions_bbox_geojson.py`: `convex_hull_excluding_region`, `bbox_envelope_excluding_region`.

---

## Examples

```bash
python utilities/fire_regions_bbox_geojson.py \
  --geojson path/to/regiones_fuego.geojson \
  --output path/to/fire_regions_bbox.geojson

python utilities/print_tif_metadata.py path/to/raster.tif
```

`download_regiones_fuego_asset.py` uses hard-coded GEE/project settings at the top of the file; edit those constants before running with Earth Engine credentials.
