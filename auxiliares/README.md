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
        → reference fill (UNIDOS_13_18.shp, Season)
        → /home/flepin/toGEE/by_tile_filled/
        → /home/flepin/toGEE/by_year_chile_filled/
```

### leftraru

```bash
cd ~/fire
cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
source auxiliares/cluster_paths.env
bash auxiliares/run_to_gee_pipeline.sh
```

Fill only (after mask + merge already done):

```bash
STEPS=fill_tiles,fill_merge_years bash auxiliares/run_to_gee_pipeline.sh
```

Full chain including reference fill:

```bash
STEPS=all_with_fill bash auxiliares/run_to_gee_pipeline.sh
```

### Steps

| `STEPS` | Script | Output |
|---------|--------|--------|
| `mask_tiles` | `mask_classified_by_polygons.py` | `toGEE/by_tile/*_vector_masked.tif` |
| `merge_years` | `validation/merge_reprojected_tiles_by_year.py` | `toGEE/by_year_chile/chile_burn_p25_<year>.tif` |
| `fill_tiles` | `fill_raster_from_reference_scars.py` | `toGEE/by_tile_filled/*_reference_filled.tif` |
| `fill_merge_years` | `validation/merge_reprojected_tiles_by_year.py` | `toGEE/by_year_chile_filled/chile_burn_p25_filled_<year>.tif` |

Partial rerun:

```bash
STEPS=mask_tiles bash auxiliares/run_to_gee_pipeline.sh
STEPS=merge_years bash auxiliares/run_to_gee_pipeline.sh
STEPS=fill_tiles bash auxiliares/run_to_gee_pipeline.sh
STEPS=fill_merge_years bash auxiliares/run_to_gee_pipeline.sh
```

### Reference fill

Uses `/home/flepin/validation/UNIDOS_13_18.shp` (column `Season` = year). **Only years 2013–2018** are filled from reference; tiles for 2019–2025 are copied unchanged to `by_tile_filled/` so the national merge still has all years. For each tile/year, rasterizes reference scars on the TIF grid and sets burn pixels where the reference mask is 1 but the raster is still 0. By default (`FILL_REQUIRE_OVERLAP=1`) only polygons that already overlap existing burn pixels are used—interior gaps are filled, not whole missed scars.

```bash
python auxiliares/fill_raster_from_reference_scars.py \
  --input-dir "${OUTPUT_BY_TILE}" \
  --output-dir "${OUTPUT_BY_TILE_FILLED}" \
  --reference-shp "${REFERENCE_SCARS_SHP}" \
  --year-column Season \
  --dry-run
```

### Pairing rule

For each raster `foo.tif`, the script looks for `foo_burn.gpkg`, then `foo_mask1.gpkg`, then any GPKG with the same **region × year** (`POLYGON_SUFFIX=auto`, default). Use `--dry-run` to verify pairs before writing.

```bash
python auxiliares/mask_classified_by_polygons.py \
  --input-dir "${WORK_ROOT}/classified_filtered" \
  --polygon-dir "${WORK_ROOT}/polygons_filtered_min20ha_p25" \
  --output-dir "/home/flepin/toGEE/by_tile" \
  --dry-run
```

### Manual merge

```bash
python validation/merge_reprojected_tiles_by_year.py \
  --input-dir "/home/flepin/toGEE/by_tile" \
  --output-dir "/home/flepin/toGEE/by_year_chile" \
  --output-stem chile_burn_p25
```

The vector GPKG layer in `polygons_filtered_min20ha_p25/` is the **authoritative** burn extent; the output raster keeps the original grid and only retains burn pixels inside those polygons.

### Progress / “¿está corriendo o colgado?”

While the pipeline runs, Python scripts print `[PROGRESS]` lines every ~30 s (bar, count, ETA, last file). Tune with `PROGRESS_HEARTBEAT_SEC` in `cluster_paths.env`.

In a **second terminal** on leftraru:

```bash
source auxiliares/cluster_paths.env
bash auxiliares/watch_to_gee_progress.sh
```

Refreshes every 15 s with TIF counts per folder and the latest file written.

### SLURM (tiles pesados, ej. 2017)

Los tiles 2017 pueden agotar RAM en el nodo login. Usar cómputo con `auxiliares/run_to_gee_fill_slurm.sh` (128 GB, 1 worker).

```bash
cd ~/fire
cp auxiliares/cluster_paths.to_gee.env.leftraru auxiliares/cluster_paths.env
# Editar mail-user en run_to_gee_fill_slurm.sh si hace falta

# Opción A — un job secuencial (3 tiles 2017 faltantes, ~2–4 h)
sbatch auxiliares/run_to_gee_fill_slurm.sh

# Opción B — 3 jobs en paralelo (uno por región r2/r4/r6)
sbatch --array=0-2 auxiliares/run_to_gee_fill_slurm.sh

# Cuando by_tile_filled tenga 52 tifs (login node):
source auxiliares/cluster_paths.env
STEPS=fill_merge_years bash auxiliares/run_to_gee_pipeline.sh
```

Seguir el job: `tail -f ~/logs/fire_to_gee_fill_<jobid>.out`
