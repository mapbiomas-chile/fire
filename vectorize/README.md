# Vectorize pipeline (auxiliary)

**Post-filtering** step: convert binary burn rasters to polygons (GeoPackage).

This pipeline runs **after** classification and filtering. It does not replace those steps.

```text
classification/          →  raw *_classified.tif
        │
        ▼
filtering/               →  classified_filtered/  (temporal + fill + LULC)
        │
        ├─ vectorize/ (per tile)     →  polygons/  (*.gpkg per tile)
        │
        └─ vectorize/ (national)     →  national_vector/
                                         mosaics_by_year/chile_<year>.tif
                                         polygons_chile/chile_<year>_events.gpkg
```

Core Python functions live in [`../lib/`](../lib/README.md). This folder is only orchestration and cluster config.

---

## Per-tile vectorization (default)

## Quick start (leftraru — nodo login)

```bash
cd ~/fire

cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env

conda activate mb_fuego
bash vectorize/run_vectorize_pipeline.sh
```

Guía interactiva: [LOCAL.md](LOCAL.md).  
SLURM (opcional): [CLUSTER.md](CLUSTER.md).

---

## National vectorization (Chile by year)

Merges all regional tiles into **one raster per year**, removes isolated patches below **112 connected pixels** (sieve), polygonizes, applies the **same pixel rule** to fragments before grouping nearby scars into **multipolygon fire events** within **200 m**.

```text
merge → sieve (≥112 px) → polygonize → fragment filter (≥112 px) → group ≤200 m
```

```bash
source vectorize/cluster_paths.env
bash vectorize/run_vectorize_national_pipeline.sh
```

SLURM (opcional): `sbatch vectorize/run_vectorize_national_pipeline_slurm.sh`

### Output layout

| Path | Content |
|------|---------|
| `national_vector/mosaics_by_year/chile_2019.tif` | Merged burn mask for Chile, year 2019 |
| `national_vector/mosaics_by_year_sieved/chile_2019.tif` | After removing small isolated patches |
| `national_vector/polygons_chile/chile_2019_events.gpkg` | Grouped events (MultiPolygon) |
| `national_vector/polygons_raw/` | Optional raw fragments (`VECTORIZE_NATIONAL_KEEP_RAW=1`) |

### Event layer columns

| Column | Description |
|--------|-------------|
| `event_id` | Unique id per year (`chile_2019_000001`, …) |
| `year` | Calendar year |
| `fragment_count` | Polygon fragments merged into this event |
| `area_ha` / `area_m2` | Event area |
| `max_gap_m` | Grouping distance used (default 200) |
| `geometry` | Polygon or MultiPolygon |

### Config (in `cluster_paths.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTORIZE_NATIONAL_WORK_ROOT` | `$WORK_ROOT/national_vector` | Output root |
| `VECTORIZE_NATIONAL_GROUP_DISTANCE_M` | `200` | Max gap between scars to merge |
| `VECTORIZE_NATIONAL_FROM_YEAR` / `TO_YEAR` | `2013` / `2025` | Year range |
| `VECTORIZE_NATIONAL_MERGE_WORKERS` | `2` | Parallel yearly mosaic jobs |
| `VECTORIZE_NATIONAL_KEEP_RAW` | `0` | Set `1` to keep ungrouped polygons |
| `VECTORIZE_NATIONAL_SIEVE_MIN_PIXELS` | `112` | Min connected burn component (pixels) on raster |
| `VECTORIZE_NATIONAL_SIEVE_MIN_HA` | — | Optional ha rule instead of pixels |
| `VECTORIZE_NATIONAL_SKIP_SIEVE` | `0` | Set `1` to disable raster sieve |
| `VECTORIZE_NATIONAL_FRAGMENT_MIN_PIXELS` | `112` | Min fragment size (px) before 200 m grouping |
| `VECTORIZE_NATIONAL_FRAGMENT_MIN_HA` | — | Legacy ha rule for fragment filter |
| `VECTORIZE_NATIONAL_SKIP_FRAGMENT_FILTER` | `0` | Set `1` to allow small fragments into grouping |

Implementation: [`lib/vectorize_national_by_year.py`](../lib/vectorize_national_by_year.py).

---

## Per-tile configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYTHON` | Yes | — | Interpreter with `geopandas`, `rasterio` |
| `WORK_ROOT` | Yes* | — | Filtering work root |
| `VECTORIZE_INPUT_DIR` | No | `$WORK_ROOT/classified_filtered` | Post-filter rasters |
| `VECTORIZE_OUTPUT_DIR` | No | `$WORK_ROOT/polygons` | Output GeoPackages |
| `VECTORIZE_WORKERS` | No | `4` | Parallel workers |
| `VECTORIZE_SIEVE_MIN_PIXELS` | No | `112` | Min connected burn component before polygonize |
| `VECTORIZE_SKIP_SIEVE` | No | `0` | Set `1` to polygonize without sieve |
| `VECTORIZE_MERGED_GPKG` | No | — | Optional single merged layer |
| `VECTORIZE_STATS_JSON` | No | `$WORK_ROOT/logs/vectorize_stats.json` | Run summary |

\*Or set `VECTORIZE_INPUT_DIR` explicitly without `WORK_ROOT`.

---

## Output

One `.gpkg` per input raster in `VECTORIZE_OUTPUT_DIR`, with attributes:

| Column | Description |
|--------|-------------|
| `year` | Calendar year from filename |
| `region` | Region id (`1`, `2`, `4`, `6`, …) |
| `source_file` | Source GeoTIFF name |
| `mask_value` | Polygonized pixel value (default `1`) |

---

## Files

| File | Role |
|------|------|
| `run_vectorize_pipeline.sh` | Per-tile vectorization (interactive) |
| `run_vectorize_pipeline_slurm.sh` | SLURM wrapper (per tile) |
| `run_vectorize_national_pipeline.sh` | Chile-wide: merge by year + group events |
| `run_vectorize_national_pipeline_slurm.sh` | SLURM wrapper (national) |
| `cluster_paths.env.example` | Path template → copy to `cluster_paths.env` |

Implementation: [`lib/vectorize_filtered_classified.py`](../lib/vectorize_filtered_classified.py).

Legacy wrapper (same backend): `filtering/polygonize_mask_parallel.py`.

---

## Next steps after vectorization

- Area histograms / threshold: [filtering/README.md](../filtering/README.md) § 5
- Validation vs reference scars: [validation/README.md](../validation/README.md)
