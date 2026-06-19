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
│    (fill_holes; max-hole-area config) │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4. LULC filter on classified tiles  │  filter_classified_parallel.py
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4c. Ag hole fill (optional)           │  fill_agricultural_holes_in_scars.py
│     enclosed cropland voids in scars  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4b. Min-patch sieve (optional)        │  sieve_min_patch_parallel.py
│     remove small connected patches    │
└───────────────────────────────────────┘
        │
        ▼  classified_filtered/
   (optional, post-vector QA)
┌───────────────────────────────────────┐
│ 5. Polygonize → histograms → threshold│  polygonize / recommend thresholds
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

**1b — Yearly** (`create_yearly_masks.py`): one mask per **filter year** Y for **variable** classes:

| Class | Code | Output pattern |
|-------|------|----------------|
| River / lake | 33 | `mascara_rio_lago_<year>.tif` |
| Infrastructure | 24 | `mascara_infraestructura_<year>.tif` |
| Agriculture | 15 | `mascara_agricultura_<year>.tif` |
| Pasture | 18 | `mascara_pastura_<year>.tif` |

**Stability rule (default `LULC_STABILITY_WINDOW=4`):** for filter year **Y**, a pixel is marked only if it belongs to the class in **four consecutive LULC years**:

- Usually **forward** from Y: `Y, Y+1, Y+2, Y+3` (e.g. filter 2017 → LULC 2017–2020).
- Near the stack end: **backward** ending at Y (e.g. filter 2025 → LULC 2022–2025).

The mask `mascara_*_Y.tif` is applied **only** to burn rasters of year **Y** (not to the other years in the window). Set `LULC_STABILITY_WINDOW=1` for legacy single-year masks.

If the LULC stack has no band for filter year 2025, the pipeline tries to build 2025 masks from the stack; on failure it can copy 2024 → 2025 (`COPY_MASK_2025_FROM_2024=1`, legacy fallback).

**1c — Total per year** (`create_total_masks_by_year.py`): combines accumulated + yearly masks into one raster per year. This is the input for the LULC filter on classified tiles.

```bash
python filtering/create_accumulated_class_masks.py \
  --input-tif /path/to/lulc_stack.tif \
  --output-dir /path/to/mascaras/acumuladas

python filtering/create_yearly_masks.py \
  --input-tif /path/to/lulc_stack.tif \
  --output-dir /path/to/mascaras/by_year \
  --start-year-in-band-1 2000 --from-year 2013 --to-year 2024 \
  --stability-window 4

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

**Purpose:** fill enclosed 0-pixels inside burn scars without moving the outer boundary. Runs **after** temporal dedup and **before** LULC so non-burnable classes are removed last.

**Script:** `refine_burn_mask_closing.py` (pipeline uses `--method fill_holes` only).

| `--method` | What it does |
|------------|--------------|
| **`fill_holes`** (pipeline default) | Fills enclosed gaps inside scars; **outer edge unchanged** |
| `closing` | Morphological closing; also smooths / expands the **border** |
| `both` | `fill_holes` then `closing` |

**`--max-hole-area`** limits which enclosed holes are filled (in pixels²):

| Context | Default | Meaning |
|---------|---------|---------|
| **Main pipeline** (`run_filtering_pipeline.sh`) | `0` | Fill **all** enclosed holes (`MAX_HOLE_AREA=0`) |

Override via `MAX_HOLE_AREA` in `filtering/cluster_paths.env`.

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

## 4c. Agricultural hole fill (optional, after LULC)

**Purpose:** after LULC removes cropland pixels, **re-burn** fully enclosed agricultural voids **inside** existing burn scars (MapBiomas often marks fields inside fire perimeters).

Requires **strict agriculture masks** (`LULC_AGRICULTURE_STABILITY_WINDOW=1` or `--agriculture-stability-window 1` in `create_yearly_masks.py`).

**Script:** `fill_agricultural_holes_in_scars.py` (wired into `run_classified_filters.py` with `FILL_AGRICULTURAL_HOLES=1`).

Compare with baseline (`filtering_work`, stability window 4, no ag fill) using `cluster_paths.ag_strict.env.example`.

---

## 4b. Min-patch sieve (after LULC)

**Purpose:** remove **small connected burn patches** left after LULC (edge fragments, salt-and-pepper noise). Runs on **raster**, before polygonize.

**Why after LULC:** LULC is the step that most fragments scars along class boundaries; filtering earlier would not catch those artifacts.

**Script:** `sieve_min_patch_parallel.py` (also wired into `run_classified_filters.py` and `run_filtering_pipeline.sh`).

Set **either** `--min-pixels` or `--min-area-ha` (ha is converted from each tile's geotransform):

```bash
python filtering/sieve_min_patch_parallel.py \
  --input-dir /path/to/classified_lulc \
  --output-dir /path/to/classified_filtered \
  --min-area-ha 1 \
  --workers 4
```

**Calibrating the threshold:** polygonize LULC output without sieve → histograms → `recommend_polygon_area_thresholds.py` → set `MIN_PATCH_SIEVE_HA` or convert ha to pixels (~11 px ≈ 1 ha at 30 m).

| Env variable | Role |
|--------------|------|
| `MIN_PATCH_SIEVE_HA` | Min connected patch area (ha) |
| `MIN_PATCH_SIEVE_PIXELS` | Overrides ha rule |
| `SKIP_MIN_PATCH_SIEVE` | Set `1` to disable |
| `MIN_PATCH_CONNECTIVITY` | `8` (default) or `4` |

Partial rerun after LULC only:

```bash
STEPS=min_patch_sieve MIN_PATCH_SIEVE_HA=1 bash filtering/run_filtering_pipeline.sh
```

---

## Steps 2 + 3 + 4 (+ 4b) in one command

**Script:** `run_classified_filters.py`

Chains `filter_temporal_first_burn_year.py` → `refine_burn_mask_closing.py` → `filter_classified_parallel.py` → optional `sieve_min_patch_parallel.py`. This is what the bash pipeline step `filter` runs.

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
| `--min-patch-only` | Min-patch sieve only (step 4b; input = LULC output) |
| `--skip-min-patch` | Temporal → fill → LULC without min-patch sieve |
| `--skip-fill` | Temporal → LULC without hole fill |
| `--no-spatial-merge` | Same-pixel dedup only (default) |
| `--name-contains 141228` | Limit temporal/fill steps to matching filenames |

---

**Note:** `fill_holes` does not fill narrow gaps that connect to the background (not fully enclosed). Use `--method both` with a small closing kernel if needed.

---

## 5. Vectorize, histograms, and area threshold (optional)

**After** `classified_filtered/` is ready. This is **separate** from the national 112-pixel sieve (special case in `vectorize/`).

| Step | Script | Purpose |
|------|--------|---------|
| Polygonize | [`vectorize/run_vectorize_pipeline.sh`](../vectorize/README.md) | One GPKG per tile |
| Histograms | `summarize_histograms_by_region.py` | **Visual** area distributions (PNG per region/year) |
| Recommend threshold | `recommend_polygon_area_thresholds.py` | **Numeric** thresholds from the same polygons (JSON + CSV) |
| Filter | `filter_polygons_by_threshold.py` | Keep polygons ≥ threshold (ha) |
| **Orchestration** | `run_polygon_area_pipeline.sh` | Runs histograms → recommend → filter from `cluster_paths.env` |
| **Full post-filter** | [`vectorize/run_post_filter_pipeline.sh`](../vectorize/run_post_filter_pipeline.sh) | vectorize + polygon area + national (STEPS) |

### How to choose a minimum area

**Evaluación actual (20260619):** dos pasos en cascada:

1. **Corte fijo ≥ 20 ha** — elimina eventos pequeños (`POLYGON_PRE_FILTER_HA=20`).
2. Sobre lo que queda: **histogramas** → **recommend** (calcula p5, p10, p25, elbow) → **filter** (aplica una regla).

```text
polygons/  →  >= 20 ha  →  polygons_min20ha/
                              →  histogramas + umbrales (p5/p10/p25/elbow)
                              →  polygons_filtered_min20ha_<regla>.gpkg
```

`recommend_polygon_area_thresholds.py` **siempre calcula las cuatro reglas** en el JSON; `filter_polygons_by_threshold.py` aplica **solo una** por corrida (`--threshold-rule p5|p10|p25|elbow`).

Flujo sin corte previo (solo percentil sobre todos los polígonos): dejar `POLYGON_PRE_FILTER_HA` vacío.

### Reglas percentiles (paso 2)

| Rule | Meaning | Typical use |
|------|---------|-------------|
| `p5` | 5th percentile of polygon areas | Aggressive: drops the smallest 5% by count |
| `p10` | 10th percentile | **Default starting point** |
| `p25` | 25th percentile | More conservative (keeps more small polygons) |
| `bottom5_mean` | Mean area of the smallest 5% | Smooth variant of the lower tail |
| `elbow` | Knee on the cumulative distribution | When the histogram has a clear break |

**Important:** this filter runs on **vectors** (hectares), after polygonize. It does not replace raster filters (temporal, LULC, hole fill).

```bash
# Env producción 20260619 (vectorize + polygon area)
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env

# Opción A: pipeline orquestado
sbatch vectorize/run_vectorize_pipeline_slurm.sh
bash filtering/run_polygon_area_pipeline.sh
sbatch vectorize/run_vectorize_national_pipeline_slurm.sh

# Opción B: todo en secuencia (login)
bash vectorize/run_post_filter_pipeline.sh

# Opción C: pasos manuales (mismo criterio)
export VECTORIZE_SKIP_SIEVE=1
bash vectorize/run_vectorize_pipeline.sh

# 2) Histograms (PNG for inspection)
python filtering/summarize_histograms_by_region.py \
  --input-dir "${WORK_ROOT}/polygons" \
  --output-dir "${WORK_ROOT}/histogramas_area"

# 3) Recommended thresholds (JSON + CSV per region and region×year)
python filtering/recommend_polygon_area_thresholds.py \
  --input-dir "${WORK_ROOT}/polygons" \
  --output-dir "${WORK_ROOT}/thresholds_area"

# 4) Filter (per-region×year P25 example; fallback: region → global)
python filtering/filter_polygons_by_threshold.py \
  --input-dir "${WORK_ROOT}/polygons" \
  --output-gpkg "${WORK_ROOT}/polygons_filtered.gpkg" \
  --stats-summary-json "${WORK_ROOT}/thresholds_area/threshold_summary.json" \
  --threshold-rule p25 \
  --per-region-year
```

Or set one manual cutoff for all tiles:

```bash
python filtering/filter_polygons_by_threshold.py \
  --input-dir "${WORK_ROOT}/polygons" \
  --output-gpkg "${WORK_ROOT}/polygons_filtered.gpkg" \
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
| `filter` | **§ 2 + § 3 + § 4 (+ § 4b if configured)** | **`run_classified_filters.py`** |
| `min_patch_sieve` | § 4b only (input = LULC output) | `run_classified_filters.py --min-patch-only` |

Partial reruns: `temporal_first_burn` (§ 2), `fill_holes` (§ 3), `lulc_filter` (§ 4), `min_patch_sieve` (§ 4b).

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
| `MIN_PATCH_SIEVE_HA` | Min connected patch (ha) for § 4b; unset = skip sieve |
| `MIN_PATCH_SIEVE_PIXELS` | Overrides ha rule for § 4b |
| `SKIP_MIN_PATCH_SIEVE` | `1` = disable § 4b even if ha/px set |
| `MIN_PATCH_CONNECTIVITY` | `8` (default) or `4` |
| `LULC_INTERMEDIATE_DIR` | LULC output when rerunning only § 4b |
| `KEEP_TEMPORAL_INTERMEDIATE` | `1` = keep § 2 intermediate output |
| `KEEP_FILL_INTERMEDIATE` | `1` = keep § 3 intermediate output |
| `FROM_YEAR` / `TO_YEAR` | Year range (default 2013–2025) |
| `LULC_TO_YEAR` | Last year with a real LULC band (default 2024) |
| `LULC_STABILITY_WINDOW` | Consecutive LULC years for A2 classes (default `4`; `1` = legacy) |
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
| `filter_temporal_first_burn_year.py` | § 2 |
| `refine_burn_mask_closing.py` | § 3 |
| `filter_classified_parallel.py` | § 4 |
| `sieve_min_patch_parallel.py` | § 4b |
| `run_classified_filters.py` | § 2 + § 3 + § 4 (+ § 4b) |
| `polygonize_mask_parallel.py` | § 4 |
| `summarize_histograms_by_region.py` | § 5 |
| `recommend_polygon_area_thresholds.py` | § 5 |
| `filter_polygons_by_threshold.py` | § 5 |
| `run_filtering_pipeline.sh` | Pipeline |
| `run_filtering_pipeline_slurm.sh` | Pipeline (SLURM) |
| `run_polygon_area_pipeline.sh` | Polygon area filter (§ 5) |
| `run_polygon_area_pipeline_slurm.sh` | Polygon area filter (SLURM) |
| `cluster_paths.env.example` | LULC+temporal config template |
| `cluster_paths.20260619.env.leftraru` | Production paths — leftraru |
