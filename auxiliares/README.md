# auxiliares/

Scripts outside the core classify → filter → vectorize pipeline. Used to prepare products for external workflows (e.g. Google Earth Engine).

## toGEE — vector mask → raster → Chile yearly mosaic

Applies the **final polygon layer** (e.g. `polygons_filtered_min20ha_p25/`) as a mask on `classified_filtered/` rasters. Produces aligned GeoTIFFs for GEE upload.

```text
classified_filtered/*.tif
    + polygons_filtered_min20ha_p25/*_burn.gpkg
        → mask (burn pixels inside polygons only)
        → /home/flepin/toGEE/by_tile/          (52 tiles, region × year)
        → /home/flepin/toGEE/by_year_chile/    (13 mosaics, one per year)
```

### leftraru

```bash
cd ~/fire
cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
source auxiliares/cluster_paths.env
bash auxiliares/run_to_gee_pipeline.sh
```

### Steps

| `STEPS` | Script | Output |
|---------|--------|--------|
| `mask_tiles` | `mask_classified_by_polygons.py` | `toGEE/by_tile/*_vector_masked.tif` |
| `merge_years` | `validation/merge_reprojected_tiles_by_year.py` | `toGEE/by_year_chile/chile_burn_p25_<year>.tif` |

Partial rerun:

```bash
STEPS=mask_tiles bash auxiliares/run_to_gee_pipeline.sh
STEPS=merge_years bash auxiliares/run_to_gee_pipeline.sh
```

### Pairing rule

For each raster `foo.tif`, the polygon file must be `foo_burn.gpkg` (suffix configurable via `--polygon-suffix`).

### Manual

```bash
python auxiliares/mask_classified_by_polygons.py \
  --input-dir "${WORK_ROOT}/classified_filtered" \
  --polygon-dir "${WORK_ROOT}/polygons_filtered_min20ha_p25" \
  --output-dir "/home/flepin/toGEE/by_tile"

python validation/merge_reprojected_tiles_by_year.py \
  --input-dir "/home/flepin/toGEE/by_tile" \
  --output-dir "/home/flepin/toGEE/by_year_chile" \
  --output-stem chile_burn_p25
```

The vector GPKG layer in `polygons_filtered_min20ha_p25/` is the **authoritative** burn extent; the output raster keeps the original grid and only retains burn pixels inside those polygons.
