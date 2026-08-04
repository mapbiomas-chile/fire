# Vectorization pipeline

[Español](README.es.md) | **English**

Converts **filtered** binary burn rasters into polygonal fire events (GeoPackage). This stage does **not** replace classification or LULC filtering.

Upstream: [filtering](../filtering/README.md). Shared algorithms: [lib](../lib/README.md). Interactive notes: [LOCAL.md](LOCAL.md). SLURM: [CLUSTER.md](CLUSTER.md).

---

## 1. Two operational modes

```text
classified_filtered/*.tif
        │
        ├─► Per-tile polygons  →  WORK_ROOT/polygons/*.gpkg
        │         │
        │         └─► Optional area filter (filtering/run_polygon_area_pipeline.sh)
        │
        └─► National by year   →  national_vector/
                 merge → sieve (≥112 px) → polygonize
                 → fragment filter (≥112 px) → group events ≤ 200 m
```

| Mode | Script | Primary product |
|------|--------|-----------------|
| Per tile | `run_vectorize_pipeline.sh` | One GPKG per filtered tile |
| National | `run_vectorize_national_pipeline.sh` | `chile_<year>_events.gpkg` |
| Full post-filter chain | `run_post_filter_pipeline.sh` | Tile + area + (optional) national |

---

## 2. Methods (parameters for replication)

### Per-tile

1. Read binary burn mask (`mask_value = 1` by default).  
2. Optional pre-polygonize sieve (`VECTORIZE_SIEVE_MIN_PIXELS`, default often **112** ≈ 1 ha at 30 m). Production area evaluation may set `VECTORIZE_SKIP_SIEVE=1` and enforce minimum size later in vector hectares.  
3. Polygonize connected components; attach `year`, `region`, `source_file`.

### National

1. Merge all regional tiles for calendar year \(Y\) (`lib/raster_by_year.py`).  
2. Raster sieve: keep components with ≥ **112** pixels (≈ 1 ha).  
3. Polygonize; drop fragments smaller than **112** px before grouping (unless skipped).  
4. Group components within **200 m** into multipolygon **events** (`max_gap_m`).

Implementation: `lib/vectorize_national_by_year.py`.

---

## 3. Replication (NLHPC login node — recommended)

Heavy jobs may use SLURM wrappers; default documentation assumes **login-node** bash with conda `mb_fuego`.

```bash
cd ~/fire
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env
conda activate mb_fuego

# 1) Tile polygons
bash vectorize/run_vectorize_pipeline.sh

# 2) ≥20 ha pre-filter + percentile rule
bash filtering/run_polygon_area_pipeline.sh

# 3) National products (optional)
bash vectorize/run_vectorize_national_pipeline.sh

# Equivalent sequential helper:
# bash vectorize/run_post_filter_pipeline.sh
```

---

## 4. Configuration variables

### Per-tile (`cluster_paths.env`)

| Variable | Default idea | Role |
|----------|--------------|------|
| `PYTHON` | Conda `mb_fuego` | Interpreter |
| `WORK_ROOT` | filtering work root | Shared paths |
| `VECTORIZE_INPUT_DIR` | `$WORK_ROOT/classified_filtered` | Input rasters |
| `VECTORIZE_OUTPUT_DIR` | `$WORK_ROOT/polygons` | Output GPKGs |
| `VECTORIZE_WORKERS` | 4 (lower to 2 if login is busy) | Parallelism |
| `VECTORIZE_SIEVE_MIN_PIXELS` | 112 | Pre-polygonize sieve |
| `VECTORIZE_SKIP_SIEVE` | 0 / 1 | Disable sieve |

### National

| Variable | Default | Role |
|----------|---------|------|
| `VECTORIZE_NATIONAL_GROUP_DISTANCE_M` | 200 | Event grouping distance |
| `VECTORIZE_NATIONAL_SIEVE_MIN_PIXELS` | 112 | Raster sieve |
| `VECTORIZE_NATIONAL_FRAGMENT_MIN_PIXELS` | 112 | Fragment filter |
| `VECTORIZE_NATIONAL_FROM_YEAR` / `TO_YEAR` | 2013 / 2025 | Years |

---

## 5. Output attributes

**Tile polygons:** `year`, `region`, `source_file`, `mask_value`, geometry.

**National events:** `event_id`, `year`, `fragment_count`, `area_ha`, `area_m2`, `max_gap_m`, geometry (Polygon / MultiPolygon).

---

## 6. Dependencies

`geopandas`, `rasterio`, `shapely`, `numpy` (plus modules under `lib/`).
