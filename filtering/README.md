# Filtering (post-classification)

[Español](README.es.md) | **English**

Raster post-processing applied **after** neural classification and **before** (or in coordination with) vectorization. The module removes persistent multi-year burns, fills internal holes, suppresses non-burnable MapBiomas land-cover classes, and optionally sieves small patches.

Upstream: [classification](../classification/README.md). Downstream: [vectorize](../vectorize/README.md), [validation](../validation/README.md).

---

## 1. Processing sequence

```text
Multi-band MapBiomas LULC stack
        │
        ▼
[1] Non-burnable masks (accumulated + yearly + total)
        │
Raw *_classified.tif (classifier)
        │
        ▼
[2] Temporal first-burn year (pixel priority)
        │
        ▼
[3] Internal hole fill (fill_holes)
        │
        ▼
[4] LULC filter (mascara_total_Y)
        │
        ▼
[4c] Agricultural enclosed-void fill (optional)
        │
        ▼
[4b] Min-patch sieve on raster (optional; often skipped in production)
        │
        ▼
classified_filtered/  →  vectorize / polygon area analysis
```

Orchestration: `run_filtering_pipeline.sh` (steps 1–4). Linked chain: `run_classified_filters.py` (steps 2–4±4b).

---

## 2. Stage methods (production defaults)

### 2.1 LULC masks

**Goal.** For each filter year \(Y\), build `mascara_total_Y.tif` where **1** = drop burn, **0** = keep.

| Substep | Script | Content |
|---------|--------|---------|
| Accumulated OR over all LULC bands | `create_accumulated_class_masks.py` | Rock 29, sand 23, salt 61, ice 34, non-veg 25 |
| Yearly stability for dynamic classes | `create_yearly_masks.py` | Water 33, infrastructure 24, agriculture 15, pasture 18 |
| Total | `create_total_masks_by_year.py` | Union for year \(Y\) |

**Stability window** (default `LULC_STABILITY_WINDOW=4`): a pixel is tagged only if the class persists for four consecutive LULC years around \(Y\) (forward in time when possible; backward near the end of the stack). Use window `1` for single-year masks. Band 1 year of the stack is set by `START_YEAR_BAND1` (often 2000). If 2025 LULC is missing, optional fallback `COPY_MASK_2025_FROM_2024=1`.

Input must be a multi-band **GeoTIFF**, not a VRT.

### 2.2 Temporal first-burn year

**Goal.** Eliminate multi-year persistence on the same pixel: the **earliest** burn year is retained (2013 ≻ 2014 ≻ … ≻ 2025).

Script: `filter_temporal_first_burn_year.py`. Year is parsed from filename token index 3 (`b14_chile_r1_2013_...`). Optional 8-neighbour merge (`TEMPORAL_SPATIAL_MERGE=1`) attributes new year-\(Y\) pixels connected to an older scar to the origin year.

### 2.3 Hole fill

**Goal.** Fill **enclosed** unburned voids inside scars without moving the external perimeter.

Script: `refine_burn_mask_closing.py` with `--method fill_holes`. Production `MAX_HOLE_AREA=0` fills all fully enclosed holes (area in pixels; 0 disables the size cap).

### 2.4 LULC application

Script: `filter_classified_parallel.py`. For each tile of year \(Y\), zeros burn where `mascara_total_Y = 1` (reproject mask to the tile grid as needed).

### 2.5 Agricultural enclosed voids (optional)

After LULC removes cropland, `fill_agricultural_holes_in_scars.py` may restore burns only in **enclosed** agricultural holes **inside** scars. Prefer strict agriculture masks (`LULC_AGRICULTURE_STABILITY_WINDOW=1`). Enabled with `FILL_AGRICULTURAL_HOLES=1`.

### 2.6 Min-patch sieve (optional)

`sieve_min_patch_parallel.py` drops small 8-connected burn components. Production often sets `SKIP_MIN_PATCH_SIEVE=1` and defers size control to the **vector** area pipeline (≥ 20 ha, then percentiles).

---

## 3. Polygon-area filtering (vector domain)

After per-tile vectorization (`vectorize/`), the optional area pipeline (`run_polygon_area_pipeline.sh`):

1. **Hard pre-filter** `POLYGON_PRE_FILTER_HA=20` (≥ 20 ha).  
2. Histograms of remaining polygon areas.  
3. Recommend **p5, p10, p25, elbow** thresholds per region × year.  
4. Apply **one** rule (production default **p25**) via `filter_polygons_by_threshold.py`.

This does not replace raster filters; it removes small **events** in attribute space (hectares).

---

## 4. Replication on NLHPC (production 20260619)

Interactive detail: [LOCAL.md](LOCAL.md). SLURM: [CLUSTER.md](CLUSTER.md).

```bash
cd ~/fire
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
# Ensure CLASSIFIED_DIR points to classifier outputs, LULC_STACK is set

bash filtering/run_filtering_pipeline.sh
# or: sbatch filtering/run_filtering_pipeline_slurm.sh
```

Expected layout under `$WORK_ROOT`:

```text
$WORK_ROOT/
├── mascaras/{acumuladas,by_year,totales}/
├── classified_filtered/     # primary raster product
└── logs/{filter_stats,fill_stats}.json
```

Key production variables (see `cluster_paths.20260619.env.leftraru`):

| Variable | Typical value | Meaning |
|----------|---------------|---------|
| `FROM_YEAR` / `TO_YEAR` | 2013 / 2025 | Burn series |
| `LULC_TO_YEAR` | 2024 | Last real LULC year band |
| `LULC_STABILITY_WINDOW` | 4 | Dynamic class stability |
| `MAX_HOLE_AREA` | 0 | Fill all enclosed holes |
| `FILL_AGRICULTURAL_HOLES` | 1 | Strict ag void fill |
| `SKIP_MIN_PATCH_SIEVE` | 1 | No raster min-patch sieve |
| `WORKERS` | 4 | Parallelism |

Partial reruns: `export STEPS=filter` (or `masks_*`, `temporal_first_burn`, `fill_holes`, `lulc_filter`, `min_patch_sieve`).

---

## 5. Filename convention

```text
b14_chile_r1_2013_cog_classified.tif
              ^^^^ token #3 = calendar year (0-based underscore split as documented in code)
```

---

## 6. Dependencies

| Stages | Packages |
|--------|----------|
| Masks + raster filters | `numpy`, `rasterio` |
| Hole fill / morphology | + `scipy` |
| Polygon area tools | `geopandas`, `matplotlib` |

Environment commonly used on leftraru: Conda **`mb_fuego`**.

---

## 7. File index

| File | Role |
|------|------|
| `create_accumulated_class_masks.py` | §2.1 accumulated |
| `create_yearly_masks.py` | §2.1 yearly |
| `create_total_masks_by_year.py` | §2.1 total |
| `filter_temporal_first_burn_year.py` | §2.2 |
| `refine_burn_mask_closing.py` | §2.3 |
| `filter_classified_parallel.py` | §2.4 |
| `fill_agricultural_holes_in_scars.py` | §2.5 |
| `sieve_min_patch_parallel.py` | §2.6 |
| `run_classified_filters.py` | Steps 2–4 chain |
| `run_filtering_pipeline.sh` | Full mask + filter orchestration |
| `run_polygon_area_pipeline.sh` | Vector area filter §3 |
| `recommend_polygon_area_thresholds.py` | Percentile / elbow rules |
| `filter_polygons_by_threshold.py` | Apply one rule or fixed ha |
| `cluster_paths.20260619.env.leftraru` | Production paths |
| `cluster_paths.env.example` | Generic template |
