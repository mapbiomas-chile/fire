#!/usr/bin/env python3
"""Shared helpers for fire model training and inference."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage

BURNED_CLASS_INDEX = 1
DEFAULT_LABEL_NAME = "landcover"
SPATIAL_BAND_PATTERNS = ("rdnbr", "dnbr", "nbr")


def fully_connected_layer(input_tensor, n_neurons, activation=None):
  import tensorflow.compat.v1 as tf

  input_size = input_tensor.get_shape().as_list()[1]
  weights = tf.Variable(
    tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))),
    name="W",
  )
  bias = tf.Variable(tf.zeros([n_neurons]), name="b")
  layer = tf.matmul(input_tensor, weights) + bias
  if activation == "relu":
    layer = tf.nn.relu(layer)
  return layer


def infer_dataset_schema(sample_path: Path, label_name: str = DEFAULT_LABEL_NAME) -> dict[str, Any]:
  with rasterio.open(sample_path) as src:
    band_count = src.count
    band_descriptions = list(src.descriptions)

  band_names = [name if name is not None else f"band_{i}" for i, name in enumerate(band_descriptions)]
  if label_name not in band_names:
    raise ValueError(
      f"Label band '{label_name}' not found in sample bands: {band_names}. "
      "Make sure training samples include this label band description."
    )

  label_band_index = band_names.index(label_name)
  input_band_indices = [i for i in range(band_count) if i != label_band_index]
  input_band_names = [band_names[i] for i in input_band_indices]

  return {
    "NUM_INPUT": len(input_band_indices),
    "INPUT_BAND_INDICES": input_band_indices,
    "INPUT_BAND_NAMES": input_band_names,
    "LABEL_BAND_INDEX": label_band_index,
    "ALL_BAND_NAMES": band_names,
  }


def resolve_spatial_feature_band_indices(
  input_band_names: list[str],
  spatial_feature_bands: list[str] | None = None,
) -> list[int]:
  if spatial_feature_bands:
    indices = []
    for token in spatial_feature_bands:
      token_lower = token.lower()
      matches = [i for i, name in enumerate(input_band_names) if token_lower in name.lower()]
      if not matches:
        raise ValueError(f"Spatial feature band '{token}' not found in input bands: {input_band_names}")
      indices.append(matches[0])
    return indices

  for pattern in SPATIAL_BAND_PATTERNS:
    matches = [i for i, name in enumerate(input_band_names) if pattern in name.lower()]
    if matches:
      return [matches[0]]
  return []


def add_window_statistics(features_hw: np.ndarray, band_indices: list[int], window_size: int) -> np.ndarray:
  if window_size < 3 or window_size % 2 == 0:
    raise ValueError("spatial window size must be an odd integer >= 3")

  extras = []
  for band_idx in band_indices:
    band = np.nan_to_num(features_hw[:, :, band_idx], nan=0.0).astype(np.float64)
    mean = ndimage.uniform_filter(band, size=window_size, mode="nearest")
    mean_sq = ndimage.uniform_filter(band * band, size=window_size, mode="nearest")
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    extras.extend([mean, std])

  if not extras:
    return features_hw

  return np.concatenate([features_hw] + [arr[..., np.newaxis] for arr in extras], axis=2)


def build_spatial_feature_config(
  input_band_names: list[str],
  window_size: int | None,
  spatial_feature_bands: list[str] | None = None,
) -> dict[str, Any] | None:
  if not window_size:
    return None

  band_indices = resolve_spatial_feature_band_indices(input_band_names, spatial_feature_bands)
  if not band_indices:
    raise ValueError(
      "Spatial window requested but no dNBR/rNBR/NBR band was found. "
      "Pass --spatial-feature-bands explicitly."
    )

  return {
    "WINDOW_SIZE": window_size,
    "BAND_INDICES_IN_INPUT": band_indices,
    "BAND_NAMES": [input_band_names[i] for i in band_indices],
    "EXTRA_FEATURES_PER_BAND": 2,
  }


def augment_features_with_spatial_context(
  features_hw: np.ndarray,
  spatial_feature_config: dict[str, Any] | None,
) -> np.ndarray:
  if spatial_feature_config is None:
    return features_hw
  return add_window_statistics(
    features_hw,
    spatial_feature_config["BAND_INDICES_IN_INPUT"],
    spatial_feature_config["WINDOW_SIZE"],
  )


def image_to_feature_matrix(
  image_path: Path,
  input_band_indices: list[int],
  label_band_index: int,
  spatial_feature_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  with rasterio.open(image_path) as dataset:
    data = dataset.read()

  data = np.transpose(data, (1, 2, 0))
  features_hw = data[:, :, input_band_indices].astype(np.float32)
  labels_hw = data[:, :, label_band_index]

  features_hw = augment_features_with_spatial_context(features_hw, spatial_feature_config)
  valid_mask = np.all(~np.isnan(features_hw), axis=2) & ~np.isnan(labels_hw)

  features = features_hw[valid_mask]
  labels = labels_hw[valid_mask].astype(np.int64)
  return features, labels


def split_files_by_scene(files: list[Path], train_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
  if len(files) < 2:
    raise ValueError("Need at least 2 sample files for spatial validation split.")

  rng = np.random.default_rng(seed)
  shuffled = files.copy()
  rng.shuffle(shuffled)

  train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
  train_files = shuffled[:train_count]
  val_files = shuffled[train_count:]
  return train_files, val_files


def load_scene_matrices(
  files: list[Path],
  input_band_indices: list[int],
  label_band_index: int,
  spatial_feature_config: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
  feature_blocks = []
  label_blocks = []
  for image_path in files:
    features, labels = image_to_feature_matrix(
      image_path,
      input_band_indices,
      label_band_index,
      spatial_feature_config,
    )
    if features.shape[0] == 0:
      print(f"[WARNING] 0 valid pixels in: {image_path}")
      continue
    print(f"[INFO] {image_path.name}: {features.shape[0]:,} valid pixels")
    feature_blocks.append(features)
    label_blocks.append(labels)

  if not feature_blocks:
    raise RuntimeError("No valid pixels found across selected sample files.")

  return np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0)


def compute_standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  data_mean = features.mean(axis=0)
  data_std = features.std(axis=0)
  data_std = np.where(data_std == 0, 1.0, data_std)
  return data_mean, data_std


def compute_class_weights(labels: np.ndarray, burned_class_index: int = BURNED_CLASS_INDEX) -> np.ndarray:
  class_counts = np.bincount(labels.astype(np.int64), minlength=2)
  class_counts = np.where(class_counts == 0, 1, class_counts)
  total = class_counts.sum()
  weights = total / (len(class_counts) * class_counts.astype(np.float64))
  if burned_class_index != 1:
    weights = weights.copy()
    weights[[0, burned_class_index]] = weights[[burned_class_index, 0]]
  return weights.astype(np.float32)


def compute_fire_metrics(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  burned_class_index: int = BURNED_CLASS_INDEX,
) -> dict[str, float]:
  y_true = y_true.astype(np.int64)
  y_pred = y_pred.astype(np.int64)

  tp = int(np.sum((y_true == burned_class_index) & (y_pred == burned_class_index)))
  fp = int(np.sum((y_true != burned_class_index) & (y_pred == burned_class_index)))
  fn = int(np.sum((y_true == burned_class_index) & (y_pred != burned_class_index)))
  tn = int(np.sum((y_true != burned_class_index) & (y_pred != burned_class_index)))

  precision = tp / (tp + fp) if (tp + fp) else 0.0
  recall = tp / (tp + fn) if (tp + fn) else 0.0
  f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
  iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
  accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

  return {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "iou": float(iou),
    "tp": float(tp),
    "fp": float(fp),
    "fn": float(fn),
    "tn": float(tn),
    "burned_prevalence": float(np.mean(y_true == burned_class_index)),
  }


def find_optimal_threshold(
  y_true: np.ndarray,
  burned_probs: np.ndarray,
  metric: str = "f1",
  burned_class_index: int = BURNED_CLASS_INDEX,
) -> tuple[float, dict[str, float]]:
  thresholds = np.linspace(0.05, 0.95, 19)
  best_threshold = 0.5
  best_metrics = compute_fire_metrics(y_true, (burned_probs >= 0.5).astype(np.int64), burned_class_index)

  for threshold in thresholds:
    y_pred = (burned_probs >= threshold).astype(np.int64)
    metrics = compute_fire_metrics(y_true, y_pred, burned_class_index)
    score = metrics[metric]
    best_score = best_metrics[metric]
    if score > best_score:
      best_threshold = float(threshold)
      best_metrics = metrics

  best_metrics["threshold"] = best_threshold
  return best_threshold, best_metrics


def build_loss_tensor(logits, labels, loss_name: str, class_weights: np.ndarray | None, focal_gamma: float):
  import tensorflow.compat.v1 as tf

  if loss_name == "focal":
    one_hot = tf.one_hot(labels, depth=int(logits.shape[-1]))
    probs = tf.nn.softmax(logits)
    pt = tf.reduce_sum(one_hot * probs, axis=1)
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=logits)
    modulating = tf.pow(1.0 - pt, focal_gamma)
    return tf.reduce_mean(modulating * ce)

  per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  if loss_name == "weighted" and class_weights is not None:
    weights = tf.gather(tf.constant(class_weights, dtype=tf.float32), labels)
    return tf.reduce_mean(per_example * weights)
  return tf.reduce_mean(per_example)


def create_model_graph(hyperparameters: dict[str, Any], training: bool = False):
  import tensorflow.compat.v1 as tf

  graph = tf.Graph()
  with graph.as_default():
    x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters["NUM_INPUT"]], name="x_input")
    y_input = tf.placeholder(tf.int64, shape=[None], name="y_input")

    normalized = (x_input - hyperparameters["data_mean"]) / hyperparameters["data_std"]
    hidden1 = fully_connected_layer(normalized, n_neurons=hyperparameters["NUM_N_L1"], activation="relu")
    hidden2 = fully_connected_layer(hidden1, n_neurons=hyperparameters["NUM_N_L2"], activation="relu")
    hidden3 = fully_connected_layer(hidden2, n_neurons=hyperparameters["NUM_N_L3"], activation="relu")
    hidden4 = fully_connected_layer(hidden3, n_neurons=hyperparameters["NUM_N_L4"], activation="relu")
    hidden5 = fully_connected_layer(hidden4, n_neurons=hyperparameters["NUM_N_L5"], activation="relu")
    logits = fully_connected_layer(hidden5, n_neurons=hyperparameters["NUM_CLASSES"])

  tensors: dict[str, Any] = {"x_input": x_input, "logits": logits}
  with graph.as_default():
    probs = tf.nn.softmax(logits, name="burned_probabilities")
    predicted_class = tf.argmax(logits, 1, name="predicted_class")
    tensors["burned_probabilities"] = probs
    tensors["predicted_class"] = predicted_class

    if training:
      training_config = hyperparameters.get("TRAINING_CONFIG", {})
      loss_name = training_config.get("loss", "cross_entropy")
      class_weights = hyperparameters.get("CLASS_WEIGHTS")
      focal_gamma = float(training_config.get("focal_gamma", 2.0))
      loss = build_loss_tensor(logits, y_input, loss_name, class_weights, focal_gamma)
      optimizer = tf.train.AdamOptimizer(hyperparameters["lr"]).minimize(loss)
      tensors["y_input"] = y_input
      tensors["loss"] = loss
      tensors["optimizer"] = optimizer

    saver = tf.train.Saver()

  return graph, tensors, saver


def load_hyperparameters(hyperparameters_path: Path) -> dict[str, Any]:
  with hyperparameters_path.open("r", encoding="utf-8") as json_file:
    hyperparameters = json.load(json_file)

  dataset_schema = hyperparameters.get("DATASET_SCHEMA")
  if dataset_schema is None:
    raise RuntimeError(
      "DATASET_SCHEMA not found in model hyperparameters. "
      "This model was likely trained with an older pipeline."
    )

  hyperparameters["data_mean"] = np.array(hyperparameters["data_mean"], dtype=np.float32)
  hyperparameters["data_std"] = np.array(hyperparameters["data_std"], dtype=np.float32)
  hyperparameters["data_std"] = np.where(hyperparameters["data_std"] == 0, 1.0, hyperparameters["data_std"])
  if "CLASS_WEIGHTS" in hyperparameters:
    hyperparameters["CLASS_WEIGHTS"] = np.array(hyperparameters["CLASS_WEIGHTS"], dtype=np.float32)
  return hyperparameters


def save_hyperparameters(path: Path, hyperparameters: dict[str, Any]) -> None:
  serializable = json.loads(json.dumps(hyperparameters, default=_json_default))
  with path.open("w", encoding="utf-8") as json_file:
    json.dump(serializable, json_file, indent=2)


def _json_default(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, Path):
    return str(value)
  raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def select_training_files(training_samples_dir: Path, version: str, region: str) -> list[Path]:
  pattern = re.compile(rf".*_({version})_.*_{region}_.*\.tif$")
  return [p for p in sorted(training_samples_dir.glob("*.tif")) if pattern.search(p.name)]


def prepare_mosaic_feature_matrix(
  mosaic_path: Path,
  hyperparameters: dict[str, Any],
) -> tuple[np.ndarray, tuple[int, int]]:
  with rasterio.open(mosaic_path) as src:
    data = src.read()

  data = np.transpose(data, (1, 2, 0))
  input_band_indices = hyperparameters["DATASET_SCHEMA"]["INPUT_BAND_INDICES"]
  features_hw = data[:, :, input_band_indices].astype(np.float32)
  spatial_feature_config = hyperparameters.get("SPATIAL_FEATURE_CONFIG")
  features_hw = augment_features_with_spatial_context(features_hw, spatial_feature_config)

  height, width = features_hw.shape[:2]
  feature_matrix = features_hw.reshape(height * width, features_hw.shape[2])
  feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
  return feature_matrix, (height, width)


def sample_training_batch(
  features: np.ndarray,
  labels: np.ndarray,
  batch_size: int,
  oversample_burned: bool,
  burned_class_index: int,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  if not oversample_burned:
    idx = rng.choice(features.shape[0], batch_size, replace=False)
    return features[idx], labels[idx]

  burned_idx = np.flatnonzero(labels == burned_class_index)
  not_burned_idx = np.flatnonzero(labels != burned_class_index)
  if burned_idx.size == 0 or not_burned_idx.size == 0:
    idx = rng.choice(features.shape[0], batch_size, replace=False)
    return features[idx], labels[idx]

  half = batch_size // 2
  burned_take = min(half, burned_idx.size)
  not_burned_take = batch_size - burned_take
  chosen = np.concatenate(
    [
      rng.choice(burned_idx, burned_take, replace=burned_take > burned_idx.size),
      rng.choice(not_burned_idx, not_burned_take, replace=not_burned_take > not_burned_idx.size),
    ]
  )
  rng.shuffle(chosen)
  return features[chosen], labels[chosen]
