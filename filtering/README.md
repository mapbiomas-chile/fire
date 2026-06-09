# Filtering

**Post-classification** utilities for MapBiomas Fire: clean classifier rasters before polygonizing or validating against reference fire scars.

- **Input:** classifier GeoTIFFs → [`../classification/`](../classification/README.md)
- **Filtered output:** binary 0/1 rasters ready for polygonize or validation → [`../validation/`](../validation/README.md)

---

## Workflow (logical order)

Filters are applied in this sequence. Each step has one or more associated scripts.

```text
Multi-band LULC stack
        │
        ▼
┌───────────────────────────────────────┐
│ 1. LULC masks (non-burnable classes)  │  create_accumulated_class_masks.py
│                                       │  create_yearly_masks.py
│                                       │  create_total_masks_by_year.py
└───────────────────────────────────────┘
        │
        ▼  mascara_total_<year>.tif
┌───────────────────────────────────────┐
│ 2. Temporal filter (first burn year)  │  filter_temporal_first_burn_year.py
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 3. Internal hole fill                 │  refine_burn_mask_closing.py
│    (fill_holes, all enclosed gaps)    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4. LULC filter on classified tiles  │  filter_classified_parallel.py
└───────────────────────────────────────┘
        │
        ▼  classified_filtered/
   (optional, for validation)
┌───────────────────────────────────────┐
│ 5. Polygonize → histograms → threshold│  polygonize_mask_parallel.py
│                                       │  summarize_histograms_by_region.py
│                                       │  filter_polygons_by_threshold.py
└───────────────────────────────────────┘
```

Steps **1–4** (mask build + filter) can run together via `run_classified_filters.py` (steps 2–4) or the bash pipeline (full steps 1–4). See [Automated pipeline](#automated-pipeline) at the end.

---

## 1. LULC masks (non-burnable classes)

**Purpose:** build, for each year, a binary mask where `1` = pixel to **remove** from the classified raster (non-burnable MapBiomas class) and `0` = keep.

**Input:** one multi-band MapBiomas GeoTIFF (one band = one year). Do not use `.vrt`.

**Output of this stage:** `mascara_total_<year>.tif` (one file per year).

### Scripts (in order)

| Step | Script | Output |
|------|--------|--------|
| 1a | `create_accumulated_class_masks.py` | `mascara_<class>_acumulado.tif` |
| 1b | `create_yearly_masks.py` | `mascara_<class>_<year>.tif` |
| 1c | `create_total_masks_by_year.py` | `mascara_total_<year>.tif` |

**1a — Accumulated** (`create_accumulated_class_masks.py`): OR of **time-invariant** classes across all bands of the stack:

| Class | MapBiomas code | File |
|-------|----------------|------|
| Rocky outcrop | 29 | `mascara_roca_acumulado.tif` |
| Sand / beach / dune | 23 | `mascara_arena_acumulado.tif` |
| Salt flat | 61 | `mascara_salar_acumulado.tif` |
| Ice / snow | 34 | `mascara_hielo_acumulado.tif` |
| Non-vegetated | 25 | `mascara_sin_vegetacion_acumulado.tif` |

**1b — Yearly** (`create_yearly_masks.py`): one mask per year for **variable** classes:

| Class | Code | Output pattern |
|-------|------|----------------|
| River / lake | 33 | `mascara_rio_lago_<year>.tif` |
| Infrastructure | 24 | `mascara_infraestructura_<year>.tif` |
| Agriculture | 15 | `mascara_agricultura_<year>.tif` |
| Pasture | 18 | `mascara_pastura_<year>.tif` |

Parallelizes by calendar year (`--workers`). If the LULC stack has no 2025 band, the pipeline can copy 2024 masks to 2025 (`COPY_MASK_2025_FROM_2024=1`).

**1c — Total per year** (`create_total_masks_by_year.py`): combines accumulated + yearly masks into one raster per year. This is the input for the LULC filter on classified tiles.

```bash
python filtering/create_accumulated_class_masks.py \
  --input-tif /path/to/lulc_stack.tif \
  --output-dir /path/to/mascaras/acumuladas

python filtering/create_yearly_masks.py \
  --input-tif /path/to/lulc_stack.tif \
  --output-dir /path/to/mascaras/by_year \
  --start-year-in-band-1 2000 --from-year 2013 --to-year 2024

python filtering/create_total_masks_by_year.py \
  --mascaras-root /path/to/mascaras \
  --from-year 2013 --to-year 2025
```

---

## 2. Temporal filter (first burn year)

**Purpose:** remove **multi-year persistence** at the same pixel. If a pixel is burned in 2013 and again in 2014, it is kept only in **2013** (2013 > 2014 > 2015 > … > 2025).

**Script:** `filter_temporal_first_burn_year.py`

- Input: **raw** classified tiles (one raster per tile × year).
- Groups all years of the same tile (year at token index 3: `b14_chile_r1_**2013**_...`).
- Processes years in chronological order; the first burn year wins the pixel.
- Output: `uint8` rasters with values `0` and `1`, suffix `_first_burn_year`.

Optional (`TEMPORAL_SPATIAL_MERGE=1`): pixels newly burned in year Y but **8-connected** to a scar from an earlier year are attributed to that origin year (e.g. Dec 2017 / Jan 2018 split).

```bash
python filtering/filter_temporal_first_burn_year.py \
  --input-dir /path/to/classi_v2 \
  --output-dir /path/to/classified_temporal \
  --from-year 2013 --to-year 2025 \
  --no-spatial-merge
```

---

## 3. Internal hole fill

**Purpose:** fill **all enclosed** 0-pixels inside burn scars without moving the outer boundary. Runs **after** temporal dedup and **before** LULC so non-burnable classes are removed last.

**Script:** `refine_burn_mask_closing.py` (pipeline uses `--method fill_holes` only).

| `--method` | What it does |
|------------|--------------|
| **`fill_holes`** (pipeline default) | Fills enclosed gaps inside scars; **outer edge unchanged** |
| `closing` | Morphological closing; also smooths / expands the **border** |
| `both` | `fill_holes` then `closing` |

Pipeline default: **`--max-hole-area 0`** (all enclosed holes). For a gentler standalone run, use e.g. `9` or `16`.

```bash
python filtering/refine_burn_mask_closing.py \
  --input-dir /path/to/classified_temporal \
  --output-dir /path/to/classified_filled \
  --method fill_holes --max-hole-area 0 --output-stem-suffix ""
```

---

## 4. LULC filter on classified rasters

**Purpose:** for each classified tile of year Y, apply `mascara_total_Y.tif`. Where the mask is `1`, burned pixels are set to `0`.

**Script:** `filter_classified_parallel.py`

- Extracts the year from the filename (`20xx`).
- Reprojects the mask to the tile grid when needed.
- Writes `*_filtered_<timestamp>.tif` plus a per-tile JSON summary.

```bash
python filtering/filter_classified_parallel.py \
  --input-dir /path/to/classified_filled \
  --masks-dir /path/to/mascaras/totales \
  --output-dir /path/to/classified_filtered \
  --workers 4
```

---

## Steps 2 + 3 + 4 in one command

**Script:** `run_classified_filters.py`

Chains `filter_temporal_first_burn_year.py` → `refine_burn_mask_closing.py` → `filter_classified_parallel.py`. This is what the bash pipeline step `filter` runs.

```bash
python filtering/run_classified_filters.py \
  --classified-dir /path/to/classi_v2 \
  --masks-dir /path/to/mascaras/totales \
  --output-dir /path/to/classified_filtered \
  --from-year 2013 --to-year 2025 \
  --max-hole-area 0 \
  --stats-json /path/to/logs/filter_stats.json \
  --fill-stats-json /path/to/logs/fill_stats.json
```

| Flag | Use |
|------|-----|
| `--temporal-only` | Temporal filter only (step 2) |
| `--fill-only` | Hole fill only (step 3); input = temporal output |
| `--lulc-only` | LULC filter only (step 4) |
| `--skip-fill` | Temporal → LULC without hole fill |
| `--no-spatial-merge` | Same-pixel dedup only (default) |
| `--name-contains 141228` | Limit temporal/fill steps to matching filenames |

---

## Standalone refine pilot (legacy)

`run_refine_closing_pipeline.sh` remains for re-running hole fill / closing on an **existing** folder without the full pipeline. Production flow uses step 3 above.

| File | Role |
|------|------|
| `run_refine_closing_pipeline.sh` | Standalone orchestration |
| `run_refine_closing_pipeline_slurm.sh` | SLURM wrapper |
| `cluster_paths.refine_closing.env.example` | Path template |

**Note:** `fill_holes` does not fill narrow gaps that connect to the background (not fully enclosed). Use `--method both` with a small closing kernel if needed.

---

## 5. Vectorize, histograms, and threshold (optional)

**After** `classified_filtered/` is ready. Vectorization has its own auxiliary pipeline in [`../vectorize/`](../vectorize/README.md).

| Step | Script / pipeline | Purpose |
|------|-------------------|---------|
| Polygonize | [`vectorize/run_vectorize_pipeline.sh`](../vectorize/README.md) | Pixels = 1 → polygons; one GPKG per raster |
| Histograms | `summarize_histograms_by_region.py` | Area distribution by region (`r1`, `r2`, …) to pick a threshold |
| Threshold | `filter_polygons_by_threshold.py` | Keep polygons ≥ N hectares |

```bash
cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
# edit PYTHON, WORK_ROOT, then:
bash vectorize/run_vectorize_pipeline.sh

python filtering/summarize_histograms_by_region.py \
  --input-dir /path/to/polygons --output-dir /path/to/histograms

python filtering/filter_polygons_by_threshold.py \
  --input-dir /path/to/polygons \
  --output-gpkg /path/to/polygons_filtered.gpkg \
  --threshold-ha 10
```

For validation against reference scars: reproject to an equal-area CRS ([`../validation/`](../validation/README.md)) and run `intersect_top_n_scars_with_classified.py`.

---

## Automated pipeline

When masks and classified tiles are already on disk, you can run **the full flow 1–4** with bash instead of calling each script separately.

| File | Role |
|------|------|
| `run_filtering_pipeline.sh` | Orchestration: masks + filtering |
| `run_filtering_pipeline_slurm.sh` | SLURM wrapper for NLHPC |
| `cluster_paths.env.example` | Path template → copy to `cluster_paths.env` |

### Configuration (required, portable across users)

The pipeline **does not embed personal paths** in code. Each user defines paths in a local file (not committed):

```bash
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
# Edit: PYTHON, LULC_STACK, CLASSIFIED_DIR, WORK_ROOT
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh
```

| Variable | Required when | Description |
|----------|---------------|-------------|
| `PYTHON` | Always | Interpreter with `numpy` + `rasterio` |
| `WORK_ROOT` | Always | Output root (masks + filtered rasters) |
| `LULC_STACK` | Mask steps (`masks_*`) | Multi-band MapBiomas GeoTIFF |
| `CLASSIFIED_DIR` | Step `filter` or `temporal_first_burn` | Raw classified tiles |
| `STEPS` | Optional (default `all`) | Steps to run |

Other settings (`FROM_YEAR`, `WORKERS`, `MAX_HOLE_AREA`, `FILTER_OUTPUT_DIR`, etc.) have sensible defaults in `cluster_paths.env.example`.

### Pipeline steps (`STEPS`)

| `STEPS` | Equivalent to | Script |
|---------|---------------|--------|
| `masks_accumulated` | § 1a | `create_accumulated_class_masks.py` |
| `masks_yearly` | § 1b | `create_yearly_masks.py` |
| `masks_total` | § 1c | `create_total_masks_by_year.py` |
| **`filter`** | **§ 2 + § 3 + § 4** | **`run_classified_filters.py`** |

Partial reruns: `temporal_first_burn` (§ 2), `fill_holes` (§ 3), `lulc_filter` (§ 4).

```bash
cd /path/to/fire
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
# edit paths, then:
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh
```

- **Interactive (no sbatch):** [LOCAL.md](LOCAL.md)
- **SLURM queue:** [CLUSTER.md](CLUSTER.md)

### Other variables (optional)

| Variable | Description |
|----------|-------------|
| `FILTER_OUTPUT_DIR` | Final output (default: `$WORK_ROOT/classified_filtered`) |
| `MAX_HOLE_AREA` | Hole fill limit in px; `0` = all enclosed holes (default) |
| `SKIP_FILL_HOLES` | `1` = temporal → LULC without fill |
| `KEEP_TEMPORAL_INTERMEDIATE` | `1` = keep § 2 intermediate output |
| `KEEP_FILL_INTERMEDIATE` | `1` = keep § 3 intermediate output |
| `FROM_YEAR` / `TO_YEAR` | Year range (default 2013–2025) |
| `LULC_TO_YEAR` | Last year with a real LULC band (default 2024) |
| `FILTER_NAME_CONTAINS` | Limit § 2–3 to certain filenames |
| `TEMPORAL_SPATIAL_MERGE` | `1` = enable spatial merge (§ 2) |

### Output layout

```text
${WORK_ROOT}/
├── mascaras/
│   ├── acumuladas/          ← § 1a
│   ├── by_year/             ← § 1b
│   └── totales/             ← § 1c  (mascara_total_<year>.tif)
├── classified_filtered/     ← § 4 final output (*_filtered_<timestamp>.tif)
├── classified_temporal/     ← § 2 only (if KEEP_TEMPORAL_INTERMEDIATE=1)
├── classified_filled/       ← § 3 only (if KEEP_FILL_INTERMEDIATE=1)
└── logs/
    ├── filter_stats.json
    └── fill_stats.json
```

---

## Filename convention

Expected MapBiomas tile names:

```text
b14_chile_r1_2013_cog_classified.tif
              ^^^^
              token index 3 = calendar year
```

After § 2: `..._first_burn_year.tif`  
After § 4: `..._first_burn_year_filtered_<timestamp>.tif`

---

## Dependencies

| Stages | Packages |
|--------|----------|
| § 1–3 (masks + filtering) | `numpy`, `rasterio` |
| § 3b closing | `numpy`, `rasterio`, `scipy` |
| § 4 polygonize + threshold | `geopandas` |
| § 4 histograms | `geopandas`, `matplotlib` |

Typical environment: Conda `mb_fuego` (or another env; set path in `PYTHON`).

---

## File index

| File | Section |
|------|---------|
| `create_accumulated_class_masks.py` | § 1a |
| `create_yearly_masks.py` | § 1b |
| `create_total_masks_by_year.py` | § 1c |
| `filter_classified_parallel.py` | § 2 |
| `filter_temporal_first_burn_year.py` | § 3 |
| `refine_burn_mask_closing.py` | § 3b |
| `run_classified_filters.py` | § 2 + § 3 |
| `polygonize_mask_parallel.py` | § 4 |
| `summarize_histograms_by_region.py` | § 4 |
| `filter_polygons_by_threshold.py` | § 4 |
| `run_filtering_pipeline.sh` | Pipeline |
| `run_filtering_pipeline_slurm.sh` | Pipeline (SLURM) |
| `run_refine_closing_pipeline.sh` | Closing pilot pipeline |
| `run_refine_closing_pipeline_slurm.sh` | Closing pilot (SLURM) |
| `cluster_paths.env.example` | LULC+temporal config |
| `cluster_paths.refine_closing.env.example` | Closing pilot config |
