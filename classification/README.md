# Classification pipeline

Training and inference scripts for the MapBiomas Chile burned-area neural network model. The pipeline is local-only and HPC-friendly: it reads rasters from disk, writes checkpoints and classified GeoTIFFs, and has no runtime dependency on Earth Engine.

## Contents

| File | Purpose |
| --- | --- |
| `fire_model_common.py` | Shared metrics, spatial features, model graph, and data loading helpers. |
| `train_fire_model.py` | Trains the TensorFlow MLP from local training-sample TIFFs. |
| `train_fire_model_xgboost.py` | Trains an XGBoost baseline on the same feature pipeline. |
| `train_fire_model.sh` | Minimal local launcher for `train_fire_model.py`. |
| `run_train_fire_model_slurm.sh` | Slurm job that runs training on the NLHPC `v100` partition. |
| `classify_fire_model.py` | Runs inference on one or more mosaic TIFFs and applies morphological opening/closing. |
| `classify_fire_model.sh` | Minimal local launcher for `classify_fire_model.py`. |
| `run_classify_fire_model_slurm.sh` | Slurm job that classifies a single `(model, mosaic)` pair. |
| `run_classify_region_slurm.sh` | Slurm job that classifies a **region + year range** with a given model version (reprocessing). |
| `cluster_paths.env.example` | Local path/config template for `run_classify_region_slurm.sh`. |
| `run_classify_tiles.py` | Helper that submits `run_classify_fire_model_slurm.sh` for all configured regions/years. |

## Model overview

The default backend is a fully-connected network with 5 hidden ReLU layers (`7 → 14 → 7 → 14 → 7`) and a 2-class output (burned / not burned). Input features are inferred from the training sample band descriptions: the band named `landcover` is used as the label and every other band is used as an input feature. Inputs are standardized using the training-set mean and standard deviation, which are persisted in the hyperparameters JSON so inference produces consistent results.

On branch `feature/model_modification`, training and inference also support:

- **Fire-oriented validation metrics**: IoU, F1, precision, recall on the burned class.
- **Spatial validation split**: hold out entire sample scenes (`--validation-split by_file`, default).
- **Class imbalance handling**: weighted loss (default), focal loss, or balanced batch oversampling.
- **Calibrated threshold**: `DECISION_THRESHOLD` in the hyperparameters JSON, tuned on validation data.
- **Local spatial context**: optional window mean/std on dNBR/rNBR/NBR bands (`--spatial-window-size 5`).
- **XGBoost backend**: `train_fire_model_xgboost.py` + `MODEL_BACKEND=xgboost` in hyperparameters.

Legacy checkpoints without `DECISION_THRESHOLD` still work: inference falls back to argmax.

## Training

### Inputs

- `--training-samples-dir`: directory of GeoTIFF training samples. File names must match the pattern `*_<version>_*_<region>_*.tif` (for example `samples_v1_foo_r2_bar.tif`).
- Each sample TIFF must contain a band whose description is `landcover`; this is used as the label. All remaining bands are used as input features.
- `--country`, `--version`, `--region`: string tokens that control which files are selected and how the output model is named.
- `--models-dir`: output directory for checkpoints and the hyperparameters JSON.
- `--seed`: random seed (default `42`).
- `--validation-split by_file`: spatial split by sample file (recommended).
- `--loss weighted`: inverse-frequency class weights (default). Alternatives: `cross_entropy`, `focal`.
- `--oversample-burned`: balanced batches when possible.
- `--metric f1`: validation metric used to pick the best checkpoint/threshold (`iou`, `recall`, `precision`).
- `--spatial-window-size 5`: add local mean/std features for dNBR-like bands.

### Outputs

Given `--country chile --version v1 --region r2`, TensorFlow training writes:

- `col1_chile_v1_r2_rnn_lstm_ckpt.*` — TensorFlow v1 checkpoint files.
- `col1_chile_v1_r2_rnn_lstm_ckpt_hyperparameters.json` — mean/std, layer sizes, `DECISION_THRESHOLD`, `VALIDATION_METRICS`, dataset schema, and optional `SPATIAL_FEATURE_CONFIG`.

XGBoost training writes:

- `col1_chile_v1_r2_xgboost.json`
- `col1_chile_v1_r2_xgboost_hyperparameters.json`

### Recommended training command

```bash
cd classification
python train_fire_model.py \
  --country chile \
  --version v1 \
  --region r2 \
  --training-samples-dir /path/to/training_samples \
  --models-dir /path/to/models \
  --validation-split by_file \
  --loss weighted \
  --oversample-burned \
  --metric f1 \
  --spatial-window-size 5
```

XGBoost benchmark:

```bash
python train_fire_model_xgboost.py \
  --country chile \
  --version v1 \
  --region r2 \
  --training-samples-dir /path/to/training_samples \
  --models-dir /path/to/models \
  --validation-split by_file \
  --spatial-window-size 5
```

Install XGBoost when needed: `pip install xgboost`

### Run locally

```bash
./train_fire_model.sh
```

### Run on Slurm (NLHPC)

```bash
sbatch run_train_fire_model_slurm.sh
```

## Classification

### Inputs

- `--model-path`: TensorFlow checkpoint base path (without extension) or XGBoost `.json` model.
- `--hyperparameters-path`: optional, defaults to `<model-stem>_hyperparameters.json`.
- `--mosaics`: one or more mosaic GeoTIFFs. Band order must match the input bands the model was trained on.
- `--output-dir`: directory where classified rasters are written.
- `--block-size`: number of pixels processed per inference block (default `40_000_000`).
- `--decision-threshold`: override `DECISION_THRESHOLD` from hyperparameters JSON.
- `--opening-filter-size` / `--closing-filter-size`: morphological structuring-element sizes (default `2` and `4`; pass `0` to disable).

Inference loads the TensorFlow model **once per mosaic** (not once per block).

### Outputs

For each input mosaic `foo.tif`, the script writes `foo_classified.tif` into `--output-dir`. The output is a single-band `uint8` GeoTIFF (`deflate` compression, predictor 2, tiled, nodata `0`) containing the burned-area mask after opening and closing.

### Run locally

```bash
./classify_fire_model.sh
```

### Run on Slurm (NLHPC)

**One mosaic per job:**

```bash
sbatch run_classify_fire_model_slurm.sh <model_name> <mosaic_name>
```

**Region + year range (reprocessing):**

```bash
cp classification/cluster_paths.env.example classification/cluster_paths.env
source classification/cluster_paths.env
sbatch classification/run_classify_region_slurm.sh
```

### Bulk submission helper

```bash
python run_classify_tiles.py
```

## Conventions

- TensorFlow checkpoint names follow `col1_<country>_<version>_<region>_rnn_lstm_ckpt`.
- XGBoost model names follow `col1_<country>_<version>_<region>_xgboost.json`.
- Classified mosaic names follow `<mosaic_stem>_classified.tif`.
- Training files must be TIFFs with band descriptions (one band must be named `landcover`).
