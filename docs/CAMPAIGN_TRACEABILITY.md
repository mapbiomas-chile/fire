# Campaign data management and traceability

[Español](CAMPAIGN_TRACEABILITY.es.md) | **English**

Operational guidance for **archiving products**, **filename conventions**, and **citing the software configuration** after a MapBiomas Fire Chile campaign run. Complements module READMEs with material suitable for technical reports and peer review.

**Reference production campaign id:** `20260619` (conservative MLP recipe). Replace folder names when a new campaign is promoted.

---

## 1. Campaign artifact checklist

Retain the following for each campaign you may need to re-run, audit, or publish. Paths illustrate the `20260619` layout on NLHPC; adapt username and campaign stamp.

### 1.1 Software provenance (always)

| Item | What to store | Why |
|------|----------------|-----|
| Git commit | SHA from `git rev-parse HEAD` (full 40 chars preferred) | Exact pipeline code |
| Git branch / tag | e.g. `main`, optional annotated tag `campaign-20260619` | Interpret history |
| Active envs | Snapshot of each used `cluster_paths.env` (see §3) | Absolute paths & flags |
| Conda / Python | Env name + `python -V`; optional `conda list --export` | Library stack |
| Run date | ISO date of train / classify / filter / vectorize | Temporal provenance |
| Operator | Username or lab role | Contactability |

Suggested side-car folder (does **not** need to live in the git repo):

```text
~/campaigns/chile_fire_20260619/
├── PROVENANCE.md                 # filled §3 template
├── git_sha.txt
├── envs/
│   ├── classification.cluster_paths.env
│   ├── filtering.cluster_paths.env
│   └── vectorize.cluster_paths.env
├── logs/                         # optional copies of SLURM .out/.err
└── manifests/                    # optional file lists (find / ls)
```

### 1.2 Models (training stage)

| Artifact | Pattern / path (example) | Archive? |
|----------|--------------------------|----------|
| TensorFlow checkpoint set | `~/models_col1_20260619/col1_chile_<v>_<region>_rnn_lstm_ckpt*` | **Yes** (full set) |
| Hyperparameters JSON | `..._rnn_lstm_ckpt_hyperparameters.json` | **Yes** (one per model) |
| Training job logs | `~/logs/train_chile_campaign_*.out` | Recommended |
| Sample inventory | output of `preview_training_campaign.py` or sample list used | Recommended |

Production expects **seven** Chile models (r1·v1, r1·v2, r2·v1, r4·v1, r4·v2, r6·v1, r6·v2). Confirm:

```bash
ls ~/models_col1_20260619/*_hyperparameters.json | wc -l   # expect 7
```

### 1.3 Classification (inference stage)

| Artifact | Pattern / path (example) | Archive? |
|----------|--------------------------|----------|
| Classified tiles | `~/classification_20260619/b14_chile_r*_????_*_classified.tif` | **Yes** |
| Count check | ~**52** tiles (4 regions × 13 years 2013–2025) | Verification |
| Classify logs | `~/logs/classi_chile_*.out` | Recommended |

### 1.4 Filtering stage

| Artifact | Pattern / path (example) | Archive? |
|----------|--------------------------|----------|
| LULC masks | `$WORK_ROOT/mascaras/{acumuladas,by_year,totales}/` | **Yes** (rebuildable but costly) |
| Filtered rasters | `$WORK_ROOT/classified_filtered/*.tif` | **Yes** (canonical raster product pre-vector) |
| Stats JSON | `$WORK_ROOT/logs/filter_stats.json`, `fill_stats.json` | **Yes** |
| Intermediate temporal/filled | only if `KEEP_*_INTERMEDIATE=1` | Optional |

With `WORK_ROOT=~/classification_20260619/filtering_work`:

```bash
ls $WORK_ROOT/classified_filtered/*.tif | wc -l   # expect ~52
ls $WORK_ROOT/mascaras/totales/mascara_total_*.tif | wc -l
```

Also record **identity of `LULC_STACK`** (path + checksum if available), `START_YEAR_BAND1`, stability windows, `MAX_HOLE_AREA`, `FILL_AGRICULTURAL_HOLES`, `SKIP_MIN_PATCH_SIEVE`.

### 1.5 Vectorization and area rules

| Artifact | Pattern / path (example) | Archive? |
|----------|--------------------------|----------|
| Per-tile polygons | `$WORK_ROOT/polygons/*.gpkg` | **Yes** if used for QA |
| Min-area stage | `polygons_min20ha/`, histograms, `thresholds_area_min20ha/` | **Yes** if published |
| Filtered by rule | `polygons_filtered_min20ha_<rule>.gpkg` (e.g. `p25`) | **Yes** for map product |
| National mosaics | `national_vector/mosaics_by_year/` | Optional |
| National events | `national_vector/polygons_chile/chile_<year>_events.gpkg` | **Yes** for national product |
| Vectorize stats | `$WORK_ROOT/logs/vectorize_stats.json` | Recommended |

Record **sieve min pixels** (often 112), **grouping distance** (often 200 m), **pre-filter ha** (often 20), **threshold rule** (`p25`, …).

### 1.6 Inputs external to the repo (reference, do not re-upload lightly)

| Input | Typical location concept | Note in PROVENANCE |
|-------|--------------------------|--------------------|
| Training samples | e.g. `~/samples_col1` | Count of TIFFs; year filters |
| Feature mosaics | e.g. `~/mosaics_cog` | Naming `b14_chile_rX_YYYY_cog.tif` |
| LULC stack | multi-band MapBiomas GeoTIFF | Band-1 year; path |
| Reference scars | validation GPKG/SHP (e.g. UNIDOS 2013–2018) | CRS; year field |

### 1.7 Minimum set for “can we rebuild the map?”

1. Git SHA + three `cluster_paths.env` snapshots.  
2. Seven model directories + seven `*_hyperparameters.json`.  
3. `classified_filtered/` **or** classified tiles + ability to re-filter with archived masks env.  
4. Published vector product(s) used in figures/tables (`polygons_filtered_*` and/or national events).  
5. Parameters listed in §3.2 of the provenance template.

---

## 2. Product naming conventions

### 2.1 Tokens

| Token | Meaning | Examples |
|-------|---------|----------|
| `b14` | Satellite / feature collection id used in mosaic naming | fixed in Chile naming |
| `chile` | Country | — |
| `r1`, `r2`, `r4`, `r6` | Processing regions (fire regions) | r5 excluded from mosaic workflow |
| `YYYY` | Calendar year in **file stem** for mosaics / classified / filtered | `2017` |
| `v1`, `v2` | **Model period** version (not sample filename token alone) | v1=2013–2018 for r1/r4/r6; v2=2019–2025 |
| `col1` | Collection token in model names | Collection 1 |

Filename years for classified products refer to the **season mosaic year** used at inference (season convention of the mosaic library). Document any later calendar remapping outside this pipeline in the campaign PROVENANCE note.

### 2.2 Standard product patterns

| Stage | Pattern | Example |
|-------|---------|---------|
| Feature mosaic (input) | `b14_chile_<region>_<year>_cog.tif` | `b14_chile_r4_2017_cog.tif` |
| Classified raster | `<mosaic_stem>_classified.tif` | `b14_chile_r4_2017_cog_classified.tif` |
| After temporal step | `..._first_burn_year.tif` | — |
| After LULC filter | `..._filtered_<timestamp>.tif` | under `classified_filtered/` |
| TF checkpoint (base) | `col1_chile_<version>_<region>_rnn_lstm_ckpt` | `col1_chile_v1_r4_rnn_lstm_ckpt` |
| Hyperparameters | `<checkpoint_base>_hyperparameters.json` | — |
| Tile polygon GPKG | often derived from filtered raster stem | via `vectorize/` |
| National yearly events | `chile_<year>_events.gpkg` | `chile_2017_events.gpkg` |
| National yearly raster | `chile_<year>.tif` (post-merge / sieved variants in subfolders) | under `national_vector/` |
| Area pre-filter outputs | `polygons_min20ha/` | fixed ≥ 20 ha |
| Area rule product | `polygons_filtered_min20ha_<rule>.gpkg` | `_p25` for production default |
| Campaign directories | `models_col1_<stamp>`, `classification_<stamp>` | `_20260619` |

### 2.3 Model version × region × years

| Region | Model `v1` years | Model `v2` years |
|--------|------------------|------------------|
| r2 | 2013–2018 **and** 2019–2025 (single v1) | — |
| r1, r4, r6 | 2013–2018 | 2019–2025 |

### 2.4 Parsing year and region from MapBiomas-style stems

Underscore-separated stem, common for tiles:

```text
b14_chile_r1_2013_cog_classified
 0    1    2   3   ...
           │   └── calendar year (token index 3)
           └────── region code
```

Yearly national products use a shorter stem (`chile_2017_...`) where the year appears after the country token. Always cross-check against `lib/tile_metadata.py` when scripting.

### 2.5 Campaign stamp

Use a single date stamp `YYYYMMDD` (or agreed label) for **both** model and classification trees when promoting a recipe:

```text
models_col1_20260619/
classification_20260619/
classification_20260619/filtering_work/
```

Avoid mixing stamps across models and rasters without documenting a deliberate hybrid.

---

## 3. How to cite commit + environment (technical report)

### 3.1 Capture (run once at campaign freeze)

```bash
cd ~/fire
git rev-parse HEAD > ~/campaigns/chile_fire_20260619/git_sha.txt
git status -sb >> ~/campaigns/chile_fire_20260619/git_sha.txt
date -Iseconds > ~/campaigns/chile_fire_20260619/frozen_at.txt

mkdir -p ~/campaigns/chile_fire_20260619/envs
# If local envs were copies of leftraru templates, copy the *used* files (after edits):
cp classification/cluster_paths.env ~/campaigns/chile_fire_20260619/envs/classification.cluster_paths.env 2>/dev/null || true
cp filtering/cluster_paths.env      ~/campaigns/chile_fire_20260619/envs/filtering.cluster_paths.env 2>/dev/null || true
cp vectorize/cluster_paths.env      ~/campaigns/chile_fire_20260619/envs/vectorize.cluster_paths.env 2>/dev/null || true
```

Do **not** commit secrets; envs usually contain only paths and numeric flags. Prefer **private archive** of env snapshots.

### 3.2 Provenance template (`PROVENANCE.md`)

Copy-paste and fill:

```markdown
# Campaign provenance — MapBiomas Fire Chile

## Identification
- Campaign id: 20260619
- Product intent: production burned-area classification (conservative MLP)
- Freeze date (UTC or local): YYYY-MM-DD
- Operator:

## Software
- Repository: https://github.com/mapbiomas-chile/fire
- Branch: main
- Commit SHA:
- Working tree clean? (yes/no; if no, list uncommitted files)

## Runtime environment
- Host: NLHPC leftraru (or other)
- Conda env: mb_fuego
- Python version:
- TensorFlow version (if known):

## Path configuration snapshots
- classification/cluster_paths.env → envs/classification.cluster_paths.env
- filtering/cluster_paths.env → envs/filtering.cluster_paths.env
- vectorize/cluster_paths.env → envs/vectorize.cluster_paths.env
- Template origin: cluster_paths.20260619.env.leftraru (train/classify/filter/vectorize as applicable)

## Training hyperparameters (if models trained in this campaign)
- Architecture: MLP 7-14-7-14-7 ReLU, 2-class softmax
- Optimizer: Adam; learning rate: 0.001
- Batch size: 1000; iterations: 7000
- Loss: weighted cross-entropy (inverse class frequency)
- Validation: by_file, train fraction 0.7; seed: 42
- Caps: 2e6 train / 5e5 val pixels
- Decision threshold: 0.55 (fixed)
- Spatial window: 0; oversample burned: off; metric: IoU

## Inputs
- Training samples directory:
- Mosaics directory:
- LULC stack path + band-1 start year:
- Reference scars (if any):

## Outputs (absolute paths)
- Models:
- Classification:
- Filtering work root:
- Published vector product(s):

## Post-processing parameters
- LULC stability window:
- MAX_HOLE_AREA / FILL_AGRICULTURAL_HOLES / SKIP_MIN_PATCH_SIEVE:
- VECTORIZE_SKIP_SIEVE / sieve min pixels:
- POLYGON_PRE_FILTER_HA / POLYGON_THRESHOLD_RULE:
- National group distance m / national sieve px:

## Notes
- Deviations from published docs or templates:
```

### 3.3 Short paragraph for a methods / tech-report section

**English (fill brackets):**

> Burned-area classification for Chile was produced with the open MapBiomas Fire Chile repository ([mapbiomas-chile/fire](https://github.com/mapbiomas-chile/fire)), branch `main`, commit `[FULL_SHA]`, frozen on `[DATE]`. Training and inference used a five-layer multilayer perceptron (hidden widths 7–14–7–14–7), Adam optimizer (learning rate 0.001), batch size 1000, 7000 iterations, inverse-frequency weighted cross-entropy, spatial holdout by sample scene (70%/30%, seed 42), and a fixed burned-class probability threshold of 0.55. Absolute paths and post-processing flags are archived in campaign envelope `[PATH_TO_CAMPAIGN/envs]`, derived from the `20260619` path templates. Intermediate and final rasters and vectors are stored under `models_col1_[STAMP]`, `classification_[STAMP]`, and the associated `filtering_work` tree.

**Spanish (completar corchetes):**

> La clasificación de área quemada para Chile se generó con el repositorio abierto MapBiomas Fire Chile ([mapbiomas-chile/fire](https://github.com/mapbiomas-chile/fire)), rama `main`, *commit* `[FULL_SHA]`, congelado el `[FECHA]`. El entrenamiento e inferencia emplearon un *multilayer perceptron* de cinco capas ocultas (anchuras 7–14–7–14–7), optimizador Adam (tasa 0,001), lote 1000, 7000 iteraciones, entropía cruzada ponderada por frecuencia inversa de clase, partición espacial por escena de muestreo (70%/30%, semilla 42) y umbral fijo de probabilidad de quemado 0,55. Las rutas absolutas y los indicadores de postproceso se archivan en el sobre de campaña `[RUTA/envs]`, derivados de las plantillas `20260619`. Los rasters y vectores intermedios y finales se almacenan en `models_col1_[STAMP]`, `classification_[STAMP]` y el árbol `filtering_work` asociado.

### 3.4 BibTeX-like software note (optional)

```bibtex
@software{mapbiomas_fire_chile,
  title        = {MapBiomas Fire Chile burned-area pipeline},
  author       = {{MapBiomas Chile}},
  year         = {2026},
  version      = {commit FULL_SHA},
  url          = {https://github.com/mapbiomas-chile/fire},
  note         = {Branch main; campaign 20260619; see campaign PROVENANCE.md}
}
```

---

## 4. Related documentation

| Topic | Document |
|-------|----------|
| End-to-end pipeline | [../README.md](../README.md) |
| Training / inference parameters | [../classification/README.md](../classification/README.md) |
| LULC and temporal filters | [../filtering/README.md](../filtering/README.md) |
| Polygons and national events | [../vectorize/README.md](../vectorize/README.md) |
