#!/usr/bin/env python3
"""
Train an XGBoost burned-area classifier on the same feature pipeline as train_fire_model.py.

Useful short-term benchmark against the legacy TensorFlow MLP without changing mosaics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fire_model_common import (
  BURNED_CLASS_INDEX,
  build_spatial_feature_config,
  compute_fire_metrics,
  compute_standardization,
  find_optimal_threshold,
  infer_dataset_schema,
  load_scene_matrices,
  save_hyperparameters,
  resolve_training_files,
  split_files_by_scene,
)


def train_xgboost_model(
  training_features,
  training_labels,
  validation_features,
  validation_labels,
  hyperparameters,
  model_path,
  json_path,
  metric_name,
):
  try:
    import xgboost as xgb
  except ImportError as exc:
    raise RuntimeError("xgboost is required. Install with: pip install xgboost") from exc

  scale_pos_weight = float(np.sum(training_labels != BURNED_CLASS_INDEX)) / max(
    np.sum(training_labels == BURNED_CLASS_INDEX), 1
  )

  dtrain = xgb.DMatrix(training_features, label=training_labels)
  dvalid = xgb.DMatrix(validation_features, label=validation_labels)

  params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": hyperparameters["TRAINING_CONFIG"]["learning_rate"],
    "max_depth": hyperparameters["TRAINING_CONFIG"]["max_depth"],
    "subsample": hyperparameters["TRAINING_CONFIG"]["subsample"],
    "colsample_bytree": hyperparameters["TRAINING_CONFIG"]["colsample_bytree"],
    "scale_pos_weight": scale_pos_weight,
    "seed": hyperparameters["TRAINING_CONFIG"]["seed"],
  }

  booster = xgb.train(
    params,
    dtrain,
    num_boost_round=hyperparameters["TRAINING_CONFIG"]["n_estimators"],
    evals=[(dvalid, "validation")],
    verbose_eval=hyperparameters["TRAINING_CONFIG"]["log_every"],
  )

  burned_probs = booster.predict(dvalid)
  threshold, metrics = find_optimal_threshold(
    validation_labels,
    burned_probs,
    metric=metric_name,
    burned_class_index=BURNED_CLASS_INDEX,
  )

  print(
    f"[INFO] Validation metrics at calibrated threshold: "
    f"f1={metrics['f1']:.4f} iou={metrics['iou']:.4f} "
    f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
  )

  booster.save_model(str(model_path))
  hyperparameters["DECISION_THRESHOLD"] = float(threshold)
  hyperparameters["BURNED_CLASS_INDEX"] = BURNED_CLASS_INDEX
  hyperparameters["VALIDATION_METRICS"] = {
    "selection_metric": metric_name,
    "thresholded": metrics,
    "argmax_at_best_checkpoint": compute_fire_metrics(
      validation_labels,
      (burned_probs >= 0.5).astype(np.int64),
      BURNED_CLASS_INDEX,
    ),
  }
  save_hyperparameters(json_path, hyperparameters)

  print(f"[INFO] XGBoost model saved at: {model_path}")
  print(f"[INFO] Hyperparameters saved at: {json_path}")


def main():
  parser = argparse.ArgumentParser(description="Train XGBoost fire model.")
  parser.add_argument("--country", default="chile")
  parser.add_argument("--version", required=True, help='Model checkpoint version token, e.g. "v1" or "v2".')
  parser.add_argument("--region", required=True)
  parser.add_argument("--sample-version", default="v1", help="Filename token in samples_fire_v1_* TIFFs.")
  parser.add_argument("--training-samples-dir", required=True)
  parser.add_argument("--models-dir", required=True)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--train-fraction", type=float, default=0.7)
  parser.add_argument(
    "--validation-split",
    choices=["by_file", "random_pixel"],
    default="by_file",
  )
  parser.add_argument("--metric", choices=["f1", "iou", "recall", "precision"], default="f1")
  parser.add_argument("--n-estimators", type=int, default=300)
  parser.add_argument("--max-depth", type=int, default=6)
  parser.add_argument("--learning-rate", type=float, default=0.1)
  parser.add_argument("--subsample", type=float, default=0.8)
  parser.add_argument("--colsample-bytree", type=float, default=0.8)
  parser.add_argument("--log-every", type=int, default=50)
  parser.add_argument("--spatial-window-size", type=int, default=0)
  parser.add_argument("--spatial-feature-bands", nargs="*", default=None)
  parser.add_argument("--sample-start-year", type=int, default=None)
  parser.add_argument("--sample-end-year", type=int, default=None)
  parser.add_argument("--sample-files", nargs="*", default=None)
  parser.add_argument("--sample-list-file", default=None)
  parser.add_argument("--sample-name-contains", default=None)
  args = parser.parse_args()

  training_samples_dir = Path(args.training_samples_dir)
  models_dir = Path(args.models_dir)
  models_dir.mkdir(parents=True, exist_ok=True)

  selected_files = resolve_training_files(
    training_samples_dir,
    args.region,
    sample_version=args.sample_version,
    sample_start_year=args.sample_start_year,
    sample_end_year=args.sample_end_year,
    sample_files=args.sample_files,
    sample_list_file=Path(args.sample_list_file) if args.sample_list_file else None,
    sample_name_contains=args.sample_name_contains,
  )
  if not selected_files:
    raise RuntimeError(
      f"No training files matched model={args.version} region={args.region} "
      f"years={args.sample_start_year}-{args.sample_end_year}"
    )

  dataset_schema = infer_dataset_schema(selected_files[0])
  spatial_feature_config = build_spatial_feature_config(
    dataset_schema["INPUT_BAND_NAMES"],
    args.spatial_window_size if args.spatial_window_size >= 3 else None,
    args.spatial_feature_bands,
  )

  bi = dataset_schema["INPUT_BAND_INDICES"]
  li = dataset_schema["LABEL_BAND_INDEX"]

  if args.validation_split == "by_file":
    train_files, val_files = split_files_by_scene(selected_files, args.train_fraction, args.seed)
    training_features, training_labels = load_scene_matrices(train_files, bi, li, spatial_feature_config)
    validation_features, validation_labels = load_scene_matrices(val_files, bi, li, spatial_feature_config)
  else:
    features, labels = load_scene_matrices(selected_files, bi, li, spatial_feature_config)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(features.shape[0])
    training_size = max(1, min(features.shape[0] - 1, int(features.shape[0] * args.train_fraction)))
    training_features = features[indices[:training_size]]
    validation_features = features[indices[training_size:]]
    training_labels = labels[indices[:training_size]]
    validation_labels = labels[indices[training_size:]]

  data_mean, data_std = compute_standardization(training_features)
  extra_features = 0
  if spatial_feature_config:
    extra_features = (
      len(spatial_feature_config["BAND_INDICES_IN_INPUT"]) * spatial_feature_config["EXTRA_FEATURES_PER_BAND"]
    )

  hyperparameters = {
    "MODEL_BACKEND": "xgboost",
    "data_mean": data_mean,
    "data_std": data_std,
    "NUM_INPUT": len(bi) + extra_features,
    "DATASET_SCHEMA": {
      "INPUT_BAND_INDICES": bi,
      "INPUT_BAND_NAMES": dataset_schema["INPUT_BAND_NAMES"],
      "LABEL_BAND_INDEX": li,
      "LABEL_NAME": "landcover",
    },
    "SPATIAL_FEATURE_CONFIG": spatial_feature_config,
    "TRAINING_CONFIG": {
      "n_estimators": args.n_estimators,
      "max_depth": args.max_depth,
      "learning_rate": args.learning_rate,
      "subsample": args.subsample,
      "colsample_bytree": args.colsample_bytree,
      "validation_split": args.validation_split,
      "metric_name": args.metric,
      "spatial_window_size": args.spatial_window_size,
      "spatial_feature_bands": args.spatial_feature_bands,
      "seed": args.seed,
      "train_fraction": args.train_fraction,
      "log_every": args.log_every,
    },
  }

  model_base = f"col1_{args.country}_{args.version}_{args.region}_xgboost.json"
  model_path = models_dir / model_base
  json_path = models_dir / f"{model_base.replace('.json', '')}_hyperparameters.json"

  print(f"[INFO] Training set size: {training_features.shape[0]:,}")
  print(f"[INFO] Validation set size: {validation_features.shape[0]:,}")

  train_xgboost_model(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    hyperparameters,
    model_path,
    json_path,
    metric_name=args.metric,
  )


if __name__ == "__main__":
  main()
