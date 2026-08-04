# Classification pipeline

[Español](README.es.md) | **English**

Supervised training and mosaic inference for the MapBiomas Chile burned-area product. Processing is **disk-based** (GeoTIFF samples and mosaics); Earth Engine is not required at training or inference time.

Peer documentation: [repository root](../README.md), [filtering](../filtering/README.md), [vectorize](../vectorize/README.md).

---

## 1. Scope

| Task | Entry points |
|------|----------------|
| Train one region × version | `train_fire_model.py`, `run_train_fire_model_slurm.sh` |
| Train full Chile campaign (7 models) | `run_train_chile_campaign.sh` → SLURM |
| Classify mosaics | `classify_fire_model.py`, region / full-series SLURM scripts |
| Campaign inventory | `preview_training_campaign.py` |

---

## 2. Model architecture (production TensorFlow MLP)

Architecture is fixed in `train_fire_model.py` / `fire_model_common.py` (`create_model_graph`).

| Component | Specification |
|-----------|----------------|
| Type | Fully connected **multilayer perceptron** (MLP) |
| Input standardization | Per-feature mean and standard deviation from the **training** set |
| Hidden layers | **5** dense ReLU layers: **7 → 14 → 7 → 14 → 7** units |
| Output | **2** logits (unburned / burned) with softmax |
| Weight init | Truncated normal, \(\sigma = 1/\sqrt{n_{\mathrm{in}}}\) |
| Optimizer | **Adam**, learning rate **0.001** |
| Batch size | **1000** |
| Iterations | **7000** |
| Loss | **Weighted softmax cross-entropy** (inverse class frequency weights) |
| Validation split | **70 % / 30 %** of sample **scenes (files)** (`--validation-split by_file`) |
| Random seed | **42** |
| Pixel caps | ≤ **2×10⁶** training, ≤ **5×10⁵** validation pixels (random subsample) |
| Decision threshold | Production fixed **0.55** (\(P(\text{burned}) \ge 0.55\)); stored in `*_hyperparameters.json` |
| Spatial context window | Production **0** (disabled) |
| Burned oversampling | Production **disabled** |
| Selection metric | **IoU** on validation (when threshold is not fixed) |

An XGBoost baseline (`train_fire_model_xgboost.py`) shares the feature construction path; production Chile uses the TensorFlow MLP.

---

## 3. Software layout

### Python modules

| File | Role |
|------|------|
| `fire_model_common.py` | Dataset schema, standardization, graph, metrics, spatial features |
| `train_fire_model.py` | TensorFlow training |
| `train_fire_model_xgboost.py` | XGBoost baseline |
| `classify_fire_model.py` | Inference + morphological opening/closing |
| `preview_training_campaign.py` | Sample coverage per of the seven Chile models |

### Orchestration

| Script | Use case |
|--------|----------|
| `run_train_chile_campaign_slurm.sh` | All 7 models (one SLURM job) |
| `run_train_fire_model_slurm.sh` | Single model |
| `run_classify_chile_slurm.sh` | Full series |
| `run_classify_region_slurm.sh` | One region / year window |
| `run_classify_single_mosaic_slurm.sh` | One mosaic (debug / figures) |
| `train_fire_model_once.sh` | Shared training body for SLURM |
| `cluster_paths.20260619.env.leftraru` | Production path template |
| `cluster_paths.train.env.leftraru` | Training shortcut |
| `cluster_paths.classify.env.leftraru` | Classification shortcut |
| `cluster_paths.env.example` | Generic template |

Copy a template to **`classification/cluster_paths.env`** (gitignored) before jobs.

---

## 4. Chile model × year mapping

Checkpoint naming: `col1_chile_<version>_<region>_rnn_lstm_ckpt`.

| Region | Years 2013–2018 | Years 2019–2025 |
|--------|-----------------|-----------------|
| r2 | v1 | v1 |
| r1, r4, r6 | v1 | v2 |

Training campaign plan (`run_train_chile_campaign.sh`):

| Region | Version | Sample years |
|--------|---------|--------------|
| r1 | v1 | 2013–2018 |
| r1 | v2 | 2019–2025 |
| r2 | v1 | 2013–2018 |
| r4 | v1 | 2013–2018 |
| r4 | v2 | 2019–2025 |
| r6 | v1 | 2013–2018 |
| r6 | v2 | 2019–2025 |

---

## 5. Training procedure (replication)

### 5.1 Inputs

- Directory of sample GeoTIFFs (predictive bands + label band description **`landcover`**).
- Year and region encoded in filenames (filtering via `--sample-start-year` / `--sample-end-year`).

### 5.2 Spatial holdout

Scenes (files) are split into train and validation with seed **42** and fraction **0.7**. All pixels from a given sample file belong to a single subset, reducing spatial leakage among adjacent pixels from the same image.

### 5.3 NLHPC — full campaign

```bash
cd ~/fire && git checkout main && git pull
cp classification/cluster_paths.train.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env

bash classification/run_train_chile_campaign.sh --dry-run   # inventory
bash classification/run_train_chile_campaign.sh              # sbatch
# tail -f ~/logs/train_chile_campaign_<JOBID>.out
```

Typical outputs:

```text
$MODELS_DIR/col1_chile_<v>_<region>_rnn_lstm_ckpt*
$MODELS_DIR/col1_chile_<v>_<region>_rnn_lstm_ckpt_hyperparameters.json
```

Production directory on leftraru: `/home/flepin/models_col1_20260619`.

### 5.4 Local single-model example

```bash
python classification/train_fire_model.py \
  --country chile --version v1 --region r2 \
  --training-samples-dir /path/to/samples_col1 \
  --models-dir /path/to/models \
  --validation-split by_file \
  --loss weighted \
  --batch-size 1000 --n-iter 7000 --learning-rate 0.001 \
  --seed 42 \
  --max-training-pixels 2000000 --max-validation-pixels 500000 \
  --metric iou \
  --fixed-decision-threshold 0.55
```

---

## 6. Inference procedure (replication)

### 6.1 Algorithm

1. Build per-pixel feature vectors matching the training `DATASET_SCHEMA`.
2. Softmax → burned-class probability \(P(\text{burned})\).
3. Threshold with `DECISION_THRESHOLD` (or `--decision-threshold`).
4. Morphological opening (default size 2) and closing (default size 4).
5. Write `uint8` GeoTIFF `<mosaic_stem>_classified.tif`.

### 6.2 Full Chile series (SLURM)

```bash
cp classification/cluster_paths.classify.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_chile_slurm.sh
# ls $OUTPUT_DIR/*_classified.tif | wc -l   # expect ~52
```

Production output: `/home/flepin/classification_20260619`.

### 6.3 Single mosaic

```bash
export MODEL_DIR=~/models_col1_20260619
export MOSAIC_DIR=~/mosaics_cog
export OUTPUT_DIR=~/classification_output
sbatch --export=ALL classification/run_classify_single_mosaic_slurm.sh \
  col1_chile_v1_r4_rnn_lstm_ckpt \
  b14_chile_r4_2017_cog.tif
```

If the env file is sourced **inside** the job, export `OUTPUT_DIR` **before** submit and use a script version that preserves overrides, or pass absolute paths carefully (see package notes when updating from `feat/auxiliares-to-gee`).

---

## 7. Link to post-processing

Classified rasters feed `filtering/` (`CLASSIFIED_DIR`). Production filter env:  
`filtering/cluster_paths.20260619.env.leftraru`.  
Then vectorize with `vectorize/cluster_paths.20260619.env.leftraru`.

---

## 8. Naming conventions

| Artifact | Pattern |
|----------|---------|
| TF checkpoint base | `col1_<country>_<version>_<region>_rnn_lstm_ckpt` |
| Hyperparameters | `<checkpoint>_hyperparameters.json` |
| Classified mosaic | `<mosaic_stem>_classified.tif` |
| Mosaic example | `b14_chile_r4_2017_cog.tif` |

---

## 9. Experimental configurations

A/B “conservative” env files used before promoting **20260619** live under [`../experiments/`](../experiments/README.md). They are not required for production replication once models exist under `models_col1_20260619`.
