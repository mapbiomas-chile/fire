#!/usr/bin/env python3
"""
Local-only training pipeline for burned area model (HPC-friendly).

Enhancements over the legacy trainer:
- spatial validation split by sample scene/file
- fire-oriented metrics (IoU, F1, precision/recall)
- class imbalance handling (weights, oversampling, focal loss)
- calibrated burned-class probability threshold
- optional local spatial context features (window mean/std)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from fire_model_common import (
  BURNED_CLASS_INDEX,
  build_spatial_feature_config,
  compute_class_weights,
  compute_fire_metrics,
  compute_standardization,
  create_model_graph,
  find_optimal_threshold,
  infer_dataset_schema,
  load_scene_matrices,
  maybe_subsample_pixels,
  sample_training_batch,
  save_hyperparameters,
  resolve_training_files,
  split_files_by_scene,
)

tf.disable_v2_behavior()


def predict_burned_probabilities(sess, tensors, features: np.ndarray, block_size: int = 500_000) -> np.ndarray:
  if features.shape[0] == 0:
    return np.array([], dtype=np.float32)

  blocks = []
  for start in range(0, features.shape[0], block_size):
    end = min(start + block_size, features.shape[0])
    probs = sess.run(
      tensors["burned_probabilities"],
      feed_dict={tensors["x_input"]: features[start:end]},
    )[:, BURNED_CLASS_INDEX]
    blocks.append(probs)
  return np.concatenate(blocks, axis=0)


def train_model(
  training_features,
  training_labels,
  validation_features,
  validation_labels,
  hyperparameters,
  model_path,
  json_path,
  oversample_burned,
  metric_name,
  log_every,
  seed,
  inference_block_size=500_000,
  fixed_decision_threshold=None,
):
  batch_size = int(hyperparameters["TRAINING_CONFIG"]["batch_size"])
  n_iter = int(hyperparameters["TRAINING_CONFIG"]["n_iter"])
  train_size = training_features.shape[0]

  if train_size < 2:
    raise ValueError("Training set has fewer than 2 rows after filtering.")
  if batch_size > train_size:
    batch_size = train_size

  graph, tensors, saver = create_model_graph(hyperparameters, training=True)
  rng = np.random.default_rng(seed)

  validation_dict = {
    tensors["x_input"]: validation_features,
    tensors["y_input"]: validation_labels,
  }

  best_state = {
    "metric": -1.0,
    "iteration": 0,
    "metrics": None,
    "threshold": hyperparameters.get("DECISION_THRESHOLD", 0.5),
  }

  start_time = time.time()
  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for iteration in range(n_iter + 1):
      batch_x, batch_y = sample_training_batch(
        training_features,
        training_labels,
        batch_size,
        oversample_burned=oversample_burned,
        burned_class_index=BURNED_CLASS_INDEX,
        rng=rng,
      )
      sess.run(
        tensors["optimizer"],
        feed_dict={tensors["x_input"]: batch_x, tensors["y_input"]: batch_y},
      )

      if iteration % log_every == 0 or iteration == n_iter:
        burned_probs = predict_burned_probabilities(
          sess,
          tensors,
          validation_features,
          block_size=inference_block_size,
        )
        if fixed_decision_threshold is not None:
          threshold = float(fixed_decision_threshold)
          y_pred = (burned_probs >= threshold).astype(np.int64)
          metrics = compute_fire_metrics(validation_labels, y_pred, BURNED_CLASS_INDEX)
          metrics["threshold"] = threshold
        else:
          threshold, metrics = find_optimal_threshold(
            validation_labels,
            burned_probs,
            metric=metric_name,
            burned_class_index=BURNED_CLASS_INDEX,
          )
          y_pred = (burned_probs >= threshold).astype(np.int64)
        argmax_metrics = compute_fire_metrics(validation_labels, y_pred, BURNED_CLASS_INDEX)

        print(
          f"[PROGRESS] Iteration {iteration}/{n_iter} "
          f"val_f1={metrics['f1']:.4f} val_iou={metrics['iou']:.4f} "
          f"val_precision={metrics['precision']:.4f} val_recall={metrics['recall']:.4f} "
          f"threshold={threshold:.2f}"
        )

        score = metrics[metric_name]
        if score >= best_state["metric"]:
          best_state = {
            "metric": score,
            "iteration": iteration,
            "metrics": metrics,
            "threshold": threshold,
            "argmax_metrics": argmax_metrics,
          }
          saver.save(sess, str(model_path))

  duration = time.time() - start_time
  print(f"[INFO] Training completed in: {time.strftime('%H:%M:%S', time.gmtime(duration))}")
  print(
    f"[INFO] Best validation {metric_name}={best_state['metric']:.4f} "
    f"at iteration {best_state['iteration']}"
  )

  if fixed_decision_threshold is not None:
    hyperparameters["DECISION_THRESHOLD"] = float(fixed_decision_threshold)
    threshold_mode = "fixed"
  else:
    hyperparameters["DECISION_THRESHOLD"] = float(best_state["threshold"])
    threshold_mode = "calibrated"
  hyperparameters["BURNED_CLASS_INDEX"] = BURNED_CLASS_INDEX
  hyperparameters["VALIDATION_METRICS"] = {
    "selection_metric": metric_name,
    "threshold_mode": threshold_mode,
    "best_iteration": best_state["iteration"],
    "thresholded": best_state["metrics"],
    "argmax_at_best_checkpoint": best_state.get("argmax_metrics"),
  }
  save_hyperparameters(json_path, hyperparameters)

  print(f"[INFO] Final model saved at: {model_path}")
  print(f"[INFO] Hyperparameters saved at: {json_path}")
  if fixed_decision_threshold is not None:
    print(f"[INFO] Fixed decision threshold: {hyperparameters['DECISION_THRESHOLD']:.3f}")
  else:
    print(f"[INFO] Calibrated decision threshold: {hyperparameters['DECISION_THRESHOLD']:.3f}")


def build_hyperparameters(
  dataset_schema,
  spatial_feature_config,
  num_input_features,
  data_mean,
  data_std,
  class_weights,
  args,
):
  return {
    "MODEL_BACKEND": "tensorflow_mlp",
    "data_mean": data_mean,
    "data_std": data_std,
    "lr": args.learning_rate,
    "NUM_N_L1": 7,
    "NUM_N_L2": 14,
    "NUM_N_L3": 7,
    "NUM_N_L4": 14,
    "NUM_N_L5": 7,
    "NUM_CLASSES": 2,
    "NUM_INPUT": num_input_features,
    "CLASS_WEIGHTS": class_weights,
    "DATASET_SCHEMA": {
      "INPUT_BAND_INDICES": dataset_schema["INPUT_BAND_INDICES"],
      "INPUT_BAND_NAMES": dataset_schema["INPUT_BAND_NAMES"],
      "LABEL_BAND_INDEX": dataset_schema["LABEL_BAND_INDEX"],
      "LABEL_NAME": "landcover",
    },
    "SPATIAL_FEATURE_CONFIG": spatial_feature_config,
    "TRAINING_CONFIG": {
      "loss": args.loss,
      "focal_gamma": args.focal_gamma,
      "batch_size": args.batch_size,
      "n_iter": args.n_iter,
      "validation_split": args.validation_split,
      "oversample_burned": args.oversample_burned,
      "metric_name": args.metric,
      "spatial_window_size": args.spatial_window_size,
      "spatial_feature_bands": args.spatial_feature_bands,
      "sample_version": args.sample_version,
      "sample_start_year": args.sample_start_year,
      "sample_end_year": args.sample_end_year,
      "max_training_pixels": args.max_training_pixels,
      "max_validation_pixels": args.max_validation_pixels,
      "seed": args.seed,
      "train_fraction": args.train_fraction,
      "fixed_decision_threshold": args.fixed_decision_threshold,
    },
  }


def main():
  parser = argparse.ArgumentParser(description="Train fire model (local-only, HPC-friendly).")
  parser.add_argument("--country", default="chile", help="Country token for model naming.")
  parser.add_argument("--version", required=True, help='Model checkpoint version token, e.g. "v1" or "v2".')
  parser.add_argument("--region", required=True, help='Region token, e.g. "r2".')
  parser.add_argument(
    "--sample-version",
    default="v1",
    help="Version token in sample filenames (Chile TIFFs are samples_fire_v1_*).",
  )
  parser.add_argument("--training-samples-dir", required=True, help="Local folder with training sample TIFFs.")
  parser.add_argument("--models-dir", required=True, help="Local output folder for checkpoints and JSON.")
  parser.add_argument("--seed", type=int, default=42, help="Random seed.")
  parser.add_argument("--train-fraction", type=float, default=0.7, help="Fraction of sample scenes used for training.")
  parser.add_argument(
    "--validation-split",
    choices=["by_file", "random_pixel"],
    default="by_file",
    help="Validation strategy. by_file avoids spatial leakage.",
  )
  parser.add_argument(
    "--loss",
    choices=["cross_entropy", "weighted", "focal"],
    default="weighted",
    help="Training loss. weighted uses inverse-frequency class weights.",
  )
  parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss.")
  parser.add_argument(
    "--oversample-burned",
    action="store_true",
    help="Use balanced batches (half burned / half not burned) when possible.",
  )
  parser.add_argument(
    "--metric",
    choices=["f1", "iou", "recall", "precision"],
    default="f1",
    help="Validation metric used to pick the best checkpoint and threshold.",
  )
  parser.add_argument("--batch-size", type=int, default=1000)
  parser.add_argument("--n-iter", type=int, default=7000)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--log-every", type=int, default=100)
  parser.add_argument(
    "--spatial-window-size",
    type=int,
    default=0,
    help="Odd window size for local mean/std features (0 disables spatial context).",
  )
  parser.add_argument(
    "--spatial-feature-bands",
    nargs="*",
    default=None,
    help="Input band name tokens for spatial context (default: auto-detect dNBR/rNBR/NBR).",
  )
  parser.add_argument(
    "--sample-start-year",
    type=int,
    default=None,
    help="Keep training samples whose filename year is >= this value.",
  )
  parser.add_argument(
    "--sample-end-year",
    type=int,
    default=None,
    help="Keep training samples whose filename year is <= this value.",
  )
  parser.add_argument(
    "--sample-files",
    nargs="*",
    default=None,
    help="Explicit training sample basenames or paths (overrides glob/year filter).",
  )
  parser.add_argument(
    "--sample-list-file",
    default=None,
    help="Text file with one training sample basename per line (# comments allowed).",
  )
  parser.add_argument(
    "--sample-name-contains",
    default=None,
    help="Keep only samples whose filename contains this substring.",
  )
  parser.add_argument(
    "--max-training-pixels",
    type=int,
    default=None,
    help="Randomly cap training pixels to limit RAM (validation unchanged unless --max-validation-pixels).",
  )
  parser.add_argument(
    "--max-validation-pixels",
    type=int,
    default=None,
    help="Randomly cap validation pixels used for metrics during training.",
  )
  parser.add_argument(
    "--inference-block-size",
    type=int,
    default=500_000,
    help="Pixels per block when scoring validation during training.",
  )
  parser.add_argument(
    "--fixed-decision-threshold",
    type=float,
    default=None,
    help="Use this burned-class probability cutoff for validation and inference (skip per-region calibration).",
  )
  args = parser.parse_args()

  training_samples_dir = Path(args.training_samples_dir)
  models_dir = Path(args.models_dir)
  models_dir.mkdir(parents=True, exist_ok=True)

  if not training_samples_dir.exists():
    raise FileNotFoundError(f"Training samples dir does not exist: {training_samples_dir}")

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
    available = [p.name for p in sorted(training_samples_dir.glob("*.tif"))]
    raise RuntimeError(
      "No training files matched the selected region/year window.\n"
      f"model_version: {args.version}\n"
      f"sample_version (filename): {args.sample_version}\n"
      f"region: {args.region}\n"
      f"years: {args.sample_start_year}-{args.sample_end_year}\n"
      f"dir: {training_samples_dir}\n"
      f"available_tifs: {available}"
    )

  print(f"[INFO] Selected files for training: {[p.name for p in selected_files]}")
  dataset_schema = infer_dataset_schema(selected_files[0], label_name="landcover")
  print(f"[INFO] Inferred dataset schema: {dataset_schema}")

  spatial_window_size = args.spatial_window_size if args.spatial_window_size >= 3 else None
  spatial_feature_config = build_spatial_feature_config(
    dataset_schema["INPUT_BAND_NAMES"],
    spatial_window_size,
    args.spatial_feature_bands,
  )
  if spatial_feature_config:
    print(f"[INFO] Spatial context enabled: {spatial_feature_config}")

  bi = dataset_schema["INPUT_BAND_INDICES"]
  li = dataset_schema["LABEL_BAND_INDEX"]

  if args.validation_split == "by_file":
    train_files, val_files = split_files_by_scene(selected_files, args.train_fraction, args.seed)
    print(f"[INFO] Spatial split train files: {[p.name for p in train_files]}")
    print(f"[INFO] Spatial split validation files: {[p.name for p in val_files]}")
    training_features, training_labels = load_scene_matrices(
      train_files, bi, li, spatial_feature_config,
      max_pixels=args.max_training_pixels, seed=args.seed,
    )
    validation_features, validation_labels = load_scene_matrices(
      val_files, bi, li, spatial_feature_config,
      max_pixels=args.max_validation_pixels, seed=args.seed + 1,
    )
  else:
    features, labels = load_scene_matrices(
      selected_files, bi, li, spatial_feature_config,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(features.shape[0])
    training_size = max(1, min(features.shape[0] - 1, int(features.shape[0] * args.train_fraction)))
    training_features = features[indices[:training_size]]
    validation_features = features[indices[training_size:]]
    training_labels = labels[indices[:training_size]]
    validation_labels = labels[indices[training_size:]]
    del features, labels
    training_features, training_labels = maybe_subsample_pixels(
      training_features, training_labels, args.max_training_pixels, args.seed, "training",
    )
    validation_features, validation_labels = maybe_subsample_pixels(
      validation_features, validation_labels, args.max_validation_pixels, args.seed + 1, "validation",
    )

  data_mean, data_std = compute_standardization(training_features)
  class_weights = compute_class_weights(training_labels, BURNED_CLASS_INDEX)

  extra_features = 0
  if spatial_feature_config:
    extra_features = (
      len(spatial_feature_config["BAND_INDICES_IN_INPUT"]) * spatial_feature_config["EXTRA_FEATURES_PER_BAND"]
    )
  num_input_features = len(bi) + extra_features

  hyperparameters = build_hyperparameters(
    dataset_schema,
    spatial_feature_config,
    num_input_features,
    data_mean,
    data_std,
    class_weights,
    args,
  )
  hyperparameters["TRAINING_SAMPLE_FILES"] = [path.name for path in selected_files]

  model_base = f"col1_{args.country}_{args.version}_{args.region}_rnn_lstm_ckpt"
  model_path = models_dir / model_base
  json_path = models_dir / f"{model_base}_hyperparameters.json"

  print(f"[INFO] Training set size: {training_features.shape[0]:,}")
  print(f"[INFO] Validation set size: {validation_features.shape[0]:,}")
  print(f"[INFO] Burned prevalence train={training_labels.mean():.4f} val={validation_labels.mean():.4f}")
  print(f"[INFO] Class weights: {class_weights}")
  print(f"[INFO] Input features: {num_input_features}")

  if args.fixed_decision_threshold is not None:
    print(f"[INFO] Fixed decision threshold (no calibration): {args.fixed_decision_threshold:.3f}")

  np.random.seed(args.seed)
  tf.set_random_seed(args.seed)
  train_model(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    hyperparameters,
    model_path,
    json_path,
    oversample_burned=args.oversample_burned,
    metric_name=args.metric,
    log_every=args.log_every,
    seed=args.seed,
    inference_block_size=args.inference_block_size,
    fixed_decision_threshold=args.fixed_decision_threshold,
  )


if __name__ == "__main__":
  main()
