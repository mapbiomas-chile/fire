# Classification pipeline

Training and inference for the MapBiomas Chile burned-area model. Reads rasters from disk, writes checkpoints and classified GeoTIFFs. No Earth Engine at runtime.

## Layout

### Python

| File | Purpose |
| --- | --- |
| `fire_model_common.py` | Shared metrics, spatial features, model graph, data loading. |
| `train_fire_model.py` | TensorFlow MLP training from local sample TIFFs. |
| `train_fire_model_xgboost.py` | XGBoost baseline on the same feature pipeline. |
| `classify_fire_model.py` | Inference on mosaic TIFFs + morphological opening/closing. |
| `preview_training_campaign.py` | Preview which samples map to each of the 7 Chile checkpoints. |

### Slurm (NLHPC)

| Script | Use case |
| --- | --- |
| `run_train_chile_campaign_slurm.sh` | Train all 7 Chile models in one job (~1 h). |
| `run_train_fire_model_slurm.sh` | Train one model (`TRAIN_REGION`, `TRAIN_VERSION`, year range). |
| `run_classify_chile_slurm.sh` | Classify full series r1/r2/r4/r6 × 2013–2025 in one job (~10 h). |
| `run_classify_region_slurm.sh` | Classify one region + year range (partial reruns, A/B tests). |
| `run_classify_single_mosaic_slurm.sh` | One mosaic + one checkpoint (debug). |

### Launchers & config

| File | Purpose |
| --- | --- |
| `run_train_chile_campaign.sh` | `sbatch` wrapper for the 7-model training campaign. |
| `train_fire_model_once.sh` | Shared training logic used by Slurm scripts. |
| `train_fire_model.sh` / `classify_fire_model.sh` | Local examples (edit paths before use). |
| `cluster_paths.env.example` | Generic path template. |
| `cluster_paths.train.env.leftraru` | leftraru training paths (committed). |
| `cluster_paths.classify.env.leftraru` | leftraru classification paths (committed). |

Copy a `*.env.leftraru` to `cluster_paths.env` (gitignored) before submitting jobs.

## Model version rules (Chile)

| Region | 2013–2018 | 2019–2025 |
| --- | --- | --- |
| r2 | v1 | v1 |
| r1, r4, r6 | v1 | v2 |

Sample filenames use `samples_fire_v1_*` (`SAMPLE_VERSION=v1` selects which TIFFs to read). Checkpoint v1/v2 is chosen by **year range**, not by the `v1` token in the sample name.

## Training

### Key options

- `--validation-split by_file` — spatial holdout by sample scene (default).
- `--loss weighted` — inverse-frequency class weights (or `focal`, `cross_entropy`).
- `--oversample-burned` — balanced batches when possible.
- `--metric f1` — validation metric for checkpoint/threshold selection.
- `--spatial-window-size 5` — local mean/std on dNBR-like bands.
- `DECISION_THRESHOLD` in hyperparameters JSON — tuned on validation; legacy checkpoints fall back to argmax.

### Local example

```bash
cd classification
python train_fire_model.py \
  --country chile --version v1 --region r2 \
  --training-samples-dir /path/to/samples \
  --models-dir /path/to/models \
  --validation-split by_file --loss weighted --oversample-burned \
  --metric f1 --spatial-window-size 5
```

### Classification inputs/outputs

- Input: `--model-path`, `--mosaics`, `--output-dir`, optional `--decision-threshold`.
- Output: `<mosaic_stem>_classified.tif` (uint8 mask after opening/closing).

---

## NLHPC leftraru

```bash
cd ~/fire
git fetch origin && git checkout feature/model_modification && git pull
```

### Training campaign (7 models, 1 job)

```bash
cp classification/cluster_paths.train.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env

bash classification/run_train_chile_campaign.sh --dry-run
bash classification/run_train_chile_campaign.sh
# tail -f ~/logs/train_chile_campaign_<JOBID>.out
```

Output: `/home/flepin/models_col1_20260618/col1_chile_<version>_<region>_rnn_lstm_ckpt*`

### Train one model

```bash
export TRAIN_REGION=r1 TRAIN_VERSION=v2 SAMPLE_START_YEAR=2019 SAMPLE_END_YEAR=2025
sbatch --export=ALL classification/run_train_fire_model_slurm.sh
```

### Classify full Chile series

```bash
cp classification/cluster_paths.classify.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_chile_slurm.sh
# tail -f ~/logs/classi_chile_<JOBID>.out
# ls /home/flepin/classification_20260618/*_classified.tif | wc -l   # expect ~52
```

### Classify one region (partial rerun or A/B)

```bash
cp classification/cluster_paths.classify.env.leftraru classification/cluster_paths.env
export REGION=r2 START_YEAR=2019 END_YEAR=2019
export AUTO_MODEL_VERSION_BY_YEAR=1
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_region_slurm.sh
```

For legacy production models, set `MODEL_DIR=/home/flepin/models_col1` and `OUTPUT_DIR=/home/flepin/classification_20260616` in `cluster_paths.env`.

### r2 isolated — matorral-only training (8 samples)

Train `col1_chile_v1_r2_rnn_lstm_ckpt` from an explicit sample list, then classify **r2 only** (2013–2025):

```bash
cp classification/cluster_paths.train_r2_matorral.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
bash classification/run_train_r2_matorral.sh

cp classification/cluster_paths.classify_r2_matorral.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_chile_slurm.sh

cp filtering/cluster_paths.r2_matorral.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh
```

Sample list: `classification/training_samples_r2_matorral_v1.txt` (years 2013, 2016–2018, matorral tiles only).  
Checkpoint: `/home/flepin/models_col1_r2_matorral/` (does not overwrite the 7-model campaign until you copy it).

### Conservative retrain (A/B vs 20260618 overestimation)

Address inflated burned area: no spatial window, no oversample, **fixed threshold 0.55**, IoU for checkpoint selection.

```bash
cp classification/cluster_paths.train_conservative.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
bash classification/run_train_chile_campaign.sh
# output: /home/flepin/models_col1_conservative/

cp classification/cluster_paths.classify_conservative.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_chile_slurm.sh
# output: /home/flepin/classification_conservative/

cp filtering/cluster_paths.conservative.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh
```

Compare classified/filtered layers against `classification_20260618` in QGIS before replacing production files.

### Compute notes

| Task | Walltime (script default) | Partition |
| --- | --- | --- |
| Training campaign | 1 h, 64 GB | `main` (CPU) |
| Training one model | 30 min | `main` |
| Classify full series | 10 h, 128 GB | `main` |
| Classify one region | 1.5 h, 128 GB | `main` |

If a job dies with `oom_kill` / `Killed` in `.err`, lower `BLOCK_SIZE` (default `5000000`) or request more memory. Existing `*_classified.tif` files are skipped on rerun (`[SKIP]`).

Cancel test jobs: `scancel <JOBID>`. Override walltime: `sbatch -t HH:MM:SS ...`.

## Conventions

- TensorFlow checkpoints: `col1_<country>_<version>_<region>_rnn_lstm_ckpt`
- XGBoost: `col1_<country>_<version>_<region>_xgboost.json`
- Classified output: `<mosaic_stem>_classified.tif`
- Training TIFFs must include a `landcover` band description (label).
