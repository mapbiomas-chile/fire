<p align="center">
  <img src="docs/assets/logo_mapbiomas_Fire.png" alt="MapBiomas Fire" width="360"/>
</p>

# MapBiomas Fire — Chile (`chile-fire`)

[Español](README.es.md) | **English**

Burned-area mapping tools for **Chile**, developed within the MapBiomas Fire collection. The repository implements an end-to-end, file-based workflow on local storage or HPC (NLHPC leftraru): sample-based model training, mosaic inference, land-cover post-filtering, vectorization, and spatial validation against reference fire scars.

**Production reference campaign:** paths and parameters labelled `20260619` (conservative multilayer perceptron recipe). Update paths when promoting a new campaign directory.

---

## 1. Scientific workflow (overview)

```text
Training samples (labelled GeoTIFF)
        │
        ▼
[1] classification/   Train MLP per region × period → checkpoint
        │
        ▼
Seasonal / yearly mosaics (feature GeoTIFF)
        │
        ▼
[1] classification/   Inference → binary classified rasters
        │
        ▼
[2] filtering/        Temporal consistency, hole fill, LULC masks
        │
        ▼
[3] vectorize/        Raster → polygons (tile and/or national)
        │
        ▼
[2|3] area filtering  Minimum event size / percentile rules (vector domain)
        │
        ▼
[4] validation/       Equal-area metrics vs reference scar catalogue
```

| Stage | Module | Role |
|-------|--------|------|
| 1 | [`classification/`](classification/README.md) | Supervised burned/unburned classification (TensorFlow MLP) |
| 2 | [`filtering/`](filtering/README.md) | Post-classification raster cleaning using MapBiomas LULC |
| 3 | [`vectorize/`](vectorize/README.md) | Polygonization and national event grouping |
| 4 | [`validation/`](validation/README.md) | Reference alignment, intersection, Jaccard / Singh metrics |
| — | [`lib/`](lib/README.md) | Shared Python helpers |
| — | [`utilities/`](utilities/README.md) | GEE export helpers, tile listing, mosaicking |
| — | [`experiments/`](experiments/README.md) | Archived A/B env files (not production) |
| — | `collection_010/` | Legacy Collection 0.1.0 materials |

---

## 2. Prerequisites for replication

| Requirement | Notes |
|-------------|--------|
| Python 3.11+ | Conda env with `numpy`, `rasterio`, `scipy`, `tensorflow` (1.x compat mode), `geopandas`, `shapely`, `pandas`, `matplotlib` |
| GeoTIFF samples | Training stacks with a `landcover` label band |
| Yearly mosaics | Same feature bands as training samples |
| MapBiomas LULC stack | Multi-band GeoTIFF (not VRT) for mask generation |
| HPC (optional) | NLHPC SLURM; interactive use of login node for many post-steps |

**Path configuration** is never hard-coded in production scripts. Copy `cluster_paths.*.env.example` or the provided `*.leftraru` templates to a **local** `cluster_paths.env` (gitignored), edit absolute paths, then `source` it before running.

---

## 3. Recommended end-to-end sequence (NLHPC)

Adjust user names and campaign folders to your account.

```bash
cd ~/fire
git fetch origin && git checkout main && git pull

# Unified production paths (train + classify + filter)
cp classification/cluster_paths.20260619.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env

# (A) Train seven Chile checkpoints (optional if models already exist)
bash classification/run_train_chile_campaign.sh

# (B) Classify full series r1/r2/r4/r6 × 2013–2025
sbatch --export=ALL classification/run_classify_chile_slurm.sh

# (C) LULC + temporal + hole-fill filtering
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh

# (D) Vectorize tiles → area filter → optional national products
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
source vectorize/cluster_paths.env
bash vectorize/run_post_filter_pipeline.sh
```

Module-level documentation expands each step with parameters suitable for peer review and independent reimplementation.

---

## 4. Chile model versioning convention

Checkpoints are named  
`col1_chile_<version>_<region>_rnn_lstm_ckpt`.

| Region | 2013–2018 | 2019–2025 |
|--------|-----------|-----------|
| r2 | v1 | v1 |
| r1, r4, r6 | v1 | v2 |

Training sample files typically use the token `samples_fire_v1_*` (`SAMPLE_VERSION=v1`). The **checkpoint version** is selected by the **year range** of the training subsample, not by that token alone.

---

## 5. Reproducibility checklist

- [ ] Record git commit SHA (`git rev-parse HEAD`) with every campaign run  
- [ ] Archive `*_hyperparameters.json` next to each TensorFlow checkpoint  
- [ ] Preserve the exact `cluster_paths.env` used (not only the template)  
- [ ] Note decision threshold (production default **0.55**), loss, seed, and train/val split mode  
- [ ] Document LULC stack identity, year of band 1, stability windows, and filter step flags  
- [ ] Store vectorization sieve / grouping distances when publishing national events  

**Extended operational guidance** (artifact checklists, product naming tables, and report citation templates for commit + environment):

- [docs/CAMPAIGN_TRACEABILITY.md](docs/CAMPAIGN_TRACEABILITY.md) (English)  
- [docs/CAMPAIGN_TRACEABILITY.es.md](docs/CAMPAIGN_TRACEABILITY.es.md) (Spanish)

---

## 6. Language

| Document | Language |
|----------|----------|
| [`README.md`](README.md) | English (this file) |
| [`README.es.md`](README.es.md) | Spanish |
| Module `README.md` / `README.es.md` | English / Spanish in each package |
| [`docs/CAMPAIGN_TRACEABILITY.md`](docs/CAMPAIGN_TRACEABILITY.md) | Campaign archive, naming, citation (EN) |
| [`docs/CAMPAIGN_TRACEABILITY.es.md`](docs/CAMPAIGN_TRACEABILITY.es.md) | Same content (ES) |

When both languages exist, treat English and Spanish as peer versions; keep parameters and command sequences identical.
