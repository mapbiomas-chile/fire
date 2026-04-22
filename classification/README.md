# Classification pipeline

Training and inference scripts for the MapBiomas Chile burned-area neural network model. The pipeline is local-only and HPC-friendly: it reads rasters from disk, writes checkpoints and classified GeoTIFFs, and has no runtime dependency on Earth Engine.

## Contents

| File | Purpose |
| --- | --- |
| `train_fire_model.py` | Trains the feed-forward neural network from local training-sample TIFFs. |
| `train_fire_model.sh` | Minimal local launcher for `train_fire_model.py`. |
| `run_train_fire_model_slurm.sh` | Slurm job that runs training on the NLHPC `v100` partition. |
| `classify_fire_model.py` | Runs inference on one or more mosaic TIFFs and applies morphological opening/closing. |
| `classify_fire_model.sh` | Minimal local launcher for `classify_fire_model.py`. |
| `run_classify_fire_model_slurm.sh` | Slurm job that classifies a single `(model, mosaic)` pair. |
| `run_classify_fire_array_slurm.sh` | Slurm array wrapper that reads `(model, mosaic)` rows from a CSV and dispatches one classification job per row. |
| `run_classify_tiles.py` | Helper that submits `run_classify_fire_model_slurm.sh` for all configured regions/years. |

## Model overview

The model is a fully-connected network with 5 hidden ReLU layers (`7 → 14 → 7 → 14 → 7`) and a 2-class output (burned / not burned). Input features are inferred from the training sample band descriptions: the band named `landcover` is used as the label and every other band is used as an input feature. Inputs are standardized using the training-set mean and standard deviation, which are persisted in the hyperparameters JSON so inference produces consistent results.

## Training

### Inputs

- `--training-samples-dir`: directory of GeoTIFF training samples. File names must match the pattern `*_<version>_*_<region>_*.tif` (for example `samples_v1_foo_r2_bar.tif`).
- Each sample TIFF must contain a band whose description is `landcover`; this is used as the label. All remaining bands are used as input features.
- `--country`, `--version`, `--region`: string tokens that control which files are selected and how the output model is named.
- `--models-dir`: output directory for checkpoints and the hyperparameters JSON.
- `--seed`: random seed for shuffling and batch sampling (default `42`).

### Outputs

Given `--country chile --version v1 --region r2`, training writes:

- `col1_chile_v1_r2_rnn_lstm_ckpt.*` — TensorFlow v1 checkpoint files.
- `col1_chile_v1_r2_rnn_lstm_ckpt_hyperparameters.json` — training-set mean/std, layer sizes, learning rate and the inferred dataset schema (band indices and names).

### Run locally

```bash
./train_fire_model.sh
```

The launcher calls `train_fire_model.py` with `chile / v1 / r3` against `training_samples/` and writes to `models/`.

### Run on Slurm (NLHPC)

```bash
sbatch run_train_fire_model_slurm.sh
```

The job uses the `v100` partition and expects `~/.env` to define `MINICONDA_PATH` and `CONDA_ENV_PATH`.

## Classification

### Inputs

- `--model-path`: checkpoint base path (without extension). The script also accepts `<model-path>.meta` as evidence the checkpoint exists.
- `--hyperparameters-path`: optional, defaults to `<model-path>_hyperparameters.json`. The file must contain a `DATASET_SCHEMA` entry; models trained with the old pipeline are rejected.
- `--mosaics`: one or more mosaic GeoTIFFs. Band order must match the input bands the model was trained on.
- `--output-dir`: directory where classified rasters are written.
- `--block-size`: number of pixels processed per inference block (default `40_000_000`).
- `--opening-filter-size` / `--closing-filter-size`: morphological structuring-element sizes (default `2` and `4`; pass `0` to disable the corresponding filter).

### Outputs

For each input mosaic `foo.tif`, the script writes `foo_classified.tif` into `--output-dir`. The output is a single-band `uint8` GeoTIFF (`deflate` compression, predictor 2, tiled, nodata `0`) containing the burned-area mask after opening and closing.

### Run locally

```bash
./classify_fire_model.sh
```

Edit the model path and mosaic path inside the script before running.

### Run on Slurm (NLHPC)

Single job:

```bash
sbatch run_classify_fire_model_slurm.sh <model_name> <mosaic_name>
```

The launcher resolves the model from `~/mapbiomas/models/<model_name>` and the mosaic from `~/mapbiomas/mosaics_cog/<mosaic_name>` and writes to `~/mapbiomas/output/classified`.

Array job (CSV-driven):

```bash
sbatch --array=1-N run_classify_fire_array_slurm.sh <csv_file> [has_header]
```

Each row of `<csv_file>` must be `model_name,mosaic_name`. `has_header` is `1` when the first line is a header (default) and `0` otherwise. Every array task then shells out to `run_classify_fire_model_slurm.sh`.

### Bulk submission helper

`run_classify_tiles.py` holds a `region → model` dictionary and a year range, and submits one `sbatch run_classify_fire_model_slurm.sh` per `(region, year)` pair using the mosaic name template `b14_chile_<region>_<year>_cog.tif`:

```bash
python run_classify_tiles.py
```

Edit the `models` dictionary and the year range at the top of the file to match the campaign.

## Conventions

- Checkpoint names follow `col1_<country>_<version>_<region>_rnn_lstm_ckpt`.
- Classified mosaic names follow `<mosaic_stem>_classified.tif`.
- Training files must be TIFFs with band descriptions (one band must be named `landcover`).
