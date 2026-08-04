#!/usr/bin/env python3
"""
Local-only burned area classification pipeline (HPC-friendly).
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import rasterio
import tensorflow.compat.v1 as tf
from scipy import ndimage

from fire_model_common import (
  BURNED_CLASS_INDEX,
  create_model_graph,
  load_hyperparameters,
  prepare_mosaic_feature_matrix,
)

tf.disable_v2_behavior()


def classify_pixels(
  data_vector,
  model_path,
  hyperparameters,
  block_size,
  decision_threshold=None,
  return_probabilities: bool = False,
):
  threshold = decision_threshold
  if threshold is None:
    threshold = hyperparameters.get("DECISION_THRESHOLD")
  burned_class_index = int(hyperparameters.get("BURNED_CLASS_INDEX", BURNED_CLASS_INDEX))

  num_pixels = data_vector.shape[0]
  num_blocks = max(1, (num_pixels + block_size - 1) // block_size)
  output = np.empty(num_pixels, dtype=np.int64)
  burned_probs = np.empty(num_pixels, dtype=np.float32) if return_probabilities else None

  tf.compat.v1.reset_default_graph()
  graph, tensors, saver = create_model_graph(hyperparameters, training=False)

  with tf.Session(graph=graph) as sess:
    saver.restore(sess, str(model_path))

    for block_idx in range(num_blocks):
      start_idx = block_idx * block_size
      end_idx = min((block_idx + 1) * block_size, num_pixels)
      print(f"[INFO] Processing block {block_idx + 1}/{num_blocks} (pixels {start_idx} to {end_idx})")
      data_block = data_vector[start_idx:end_idx]

      if threshold is None and not return_probabilities:
        output[start_idx:end_idx] = sess.run(
          tensors["predicted_class"],
          feed_dict={tensors["x_input"]: data_block},
        )
      else:
        probs = sess.run(
          tensors["burned_probabilities"],
          feed_dict={tensors["x_input"]: data_block},
        )
        p_burn = probs[:, burned_class_index].astype(np.float32)
        if return_probabilities:
          burned_probs[start_idx:end_idx] = p_burn
        if threshold is None:
          output[start_idx:end_idx] = np.argmax(probs, axis=1)
        else:
          output[start_idx:end_idx] = (p_burn >= float(threshold)).astype(np.int64)

  if return_probabilities:
    return output, burned_probs
  return output


def classify_pixels_xgboost(
  data_vector,
  model_path,
  hyperparameters,
  decision_threshold=None,
  return_probabilities: bool = False,
):
  try:
    import xgboost as xgb
  except ImportError as exc:
    raise RuntimeError("xgboost is required for MODEL_BACKEND=xgboost. Install with: pip install xgboost") from exc

  booster = xgb.Booster()
  booster.load_model(str(model_path))

  threshold = decision_threshold
  if threshold is None:
    threshold = hyperparameters.get("DECISION_THRESHOLD", 0.5)
  burned_class_index = int(hyperparameters.get("BURNED_CLASS_INDEX", BURNED_CLASS_INDEX))

  dmatrix = xgb.DMatrix(data_vector)
  probs = booster.predict(dmatrix, output_margin=False)
  if probs.ndim == 1:
    burned_probs = probs.astype(np.float32)
  else:
    burned_probs = probs[:, burned_class_index].astype(np.float32)

  if threshold is None:
    labels = (burned_probs >= 0.5).astype(np.int64)
  else:
    labels = (burned_probs >= float(threshold)).astype(np.int64)

  if return_probabilities:
    return labels, burned_probs
  return labels


def apply_spatial_filter(output_image_data, opening_filter_size=None, closing_filter_size=None):
  binary_image = output_image_data > 0

  if opening_filter_size is False:
    open_image = binary_image
  else:
    m = int(opening_filter_size) if opening_filter_size is not None else 2
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((m, m)))

  if closing_filter_size is False:
    close_image = open_image
  else:
    n = int(closing_filter_size) if closing_filter_size is not None else 4
    close_image = ndimage.binary_closing(open_image, structure=np.ones((n, n)))

  return close_image.astype("uint8")


def write_geotiff(path: Path, array: np.ndarray, profile_src: dict, *, dtype, nodata) -> None:
  profile = profile_src.copy()
  profile.update(
    dtype=dtype,
    count=1,
    compress="deflate",
    predictor=2 if np.dtype(dtype).kind in ("i", "u") else 3,
    tiled=True,
    nodata=nodata,
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  with rasterio.open(path, "w", **profile) as dst:
    dst.write(array.astype(dtype), 1)


def classify_single_mosaic(
  mosaic_path,
  output_path,
  model_path,
  hyperparameters,
  block_size=40000000,
  opening_filter_size=None,
  closing_filter_size=None,
  decision_threshold=None,
  write_probability: bool = False,
  probability_path: Path | None = None,
):
  data_classify_vector, (height, width) = prepare_mosaic_feature_matrix(mosaic_path, hyperparameters)

  if data_classify_vector.shape[1] != hyperparameters["NUM_INPUT"]:
    raise RuntimeError(
      f"Band mismatch: model expects {hyperparameters['NUM_INPUT']} features, "
      f"but classification data has {data_classify_vector.shape[1]}"
    )

  backend = hyperparameters.get("MODEL_BACKEND", "tensorflow_mlp")
  if backend == "xgboost":
    result = classify_pixels_xgboost(
      data_classify_vector,
      model_path,
      hyperparameters,
      decision_threshold=decision_threshold,
      return_probabilities=write_probability,
    )
  else:
    result = classify_pixels(
      data_classify_vector,
      model_path,
      hyperparameters,
      block_size=block_size,
      decision_threshold=decision_threshold,
      return_probabilities=write_probability,
    )

  if write_probability:
    output_data_classified, burned_probs = result
  else:
    output_data_classified = result
    burned_probs = None

  del data_classify_vector
  gc.collect()

  with rasterio.open(mosaic_path) as src:
    profile = src.profile.copy()

  if write_probability and burned_probs is not None:
    prob_out = probability_path or output_path.with_name(
      output_path.name.replace("_classified.tif", "_probability.tif")
    )
    if prob_out == output_path:
      prob_out = output_path.with_name(f"{output_path.stem}_probability.tif")
    write_geotiff(
      Path(prob_out),
      burned_probs.reshape(height, width),
      profile,
      dtype=rasterio.float32,
      nodata=-1.0,
    )
    print(f"[INFO] Saved burned probability (pre-morphology): {prob_out}")
    del burned_probs

  output_image_data = output_data_classified.reshape(height, width)
  del output_data_classified
  filtered = apply_spatial_filter(output_image_data, opening_filter_size, closing_filter_size)

  write_geotiff(Path(output_path), filtered, profile, dtype=rasterio.uint8, nodata=0)
  print(f"[INFO] Saved classified raster: {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Classify burned area using local model and local mosaics.")
  parser.add_argument(
    "--model-path",
    required=True,
    help="Local checkpoint base path (TensorFlow) or .json model (XGBoost).",
  )
  parser.add_argument(
    "--hyperparameters-path",
    default=None,
    help="Optional hyperparameters JSON path. Defaults to <model-path>_hyperparameters.json",
  )
  parser.add_argument("--mosaics", nargs="+", required=True, help="Local mosaic tif path(s) to classify.")
  parser.add_argument("--output-dir", required=True, help="Local output directory for classified rasters.")
  parser.add_argument(
    "--block-size",
    type=int,
    default=5_000_000,
    help="Pixels per inference block (TensorFlow only). Lower if OOM on large mosaics.",
  )
  parser.add_argument("--opening-filter-size", type=int, default=2, help="Opening filter size. Use 0 to disable.")
  parser.add_argument("--closing-filter-size", type=int, default=4, help="Closing filter size. Use 0 to disable.")
  parser.add_argument(
    "--decision-threshold",
    type=float,
    default=None,
    help="Override burned-class probability threshold from hyperparameters JSON.",
  )
  parser.add_argument(
    "--write-probability",
    action="store_true",
    help=(
      "Also write a float32 GeoTIFF of burned-class softmax probability "
      "(*_probability.tif), before morphological opening/closing."
    ),
  )
  args = parser.parse_args()

  model_path = Path(args.model_path)
  if args.hyperparameters_path:
    hyperparameters_path = Path(args.hyperparameters_path)
  else:
    model_stem = model_path.with_suffix("") if model_path.suffix.lower() == ".json" else model_path
    hyperparameters_path = Path(f"{model_stem}_hyperparameters.json")
  output_dir = Path(args.output_dir)

  backend = None
  if hyperparameters_path.exists():
    hyperparameters = load_hyperparameters(hyperparameters_path)
    backend = hyperparameters.get("MODEL_BACKEND", "tensorflow_mlp")
  else:
    hyperparameters = None

  if backend == "xgboost":
    if not model_path.exists():
      raise FileNotFoundError(f"XGBoost model not found: {model_path}")
  elif not model_path.exists() and not Path(f"{args.model_path}.meta").exists():
    raise FileNotFoundError(f"Model checkpoint not found: {model_path} (or {args.model_path}.meta)")

  if hyperparameters is None:
    raise FileNotFoundError(f"Hyperparameters file not found: {hyperparameters_path}")

  opening_filter = False if args.opening_filter_size == 0 else args.opening_filter_size
  closing_filter = False if args.closing_filter_size == 0 else args.closing_filter_size

  threshold = args.decision_threshold
  if threshold is None:
    threshold = hyperparameters.get("DECISION_THRESHOLD")
  if threshold is not None:
    print(f"[INFO] Using burned-class decision threshold: {threshold:.3f}")
  if args.write_probability:
    print("[INFO] Will write burned-class probability GeoTIFF")

  for mosaic in args.mosaics:
    mosaic_path = Path(mosaic)
    if not mosaic_path.exists():
      raise FileNotFoundError(f"Mosaic not found: {mosaic_path}")

    output_name = f"{mosaic_path.stem}_classified.tif"
    output_path = output_dir / output_name
    print(f"[INFO] Classifying mosaic: {mosaic_path}")
    classify_single_mosaic(
      mosaic_path=mosaic_path,
      output_path=output_path,
      model_path=model_path,
      hyperparameters=hyperparameters,
      block_size=args.block_size,
      opening_filter_size=opening_filter,
      closing_filter_size=closing_filter,
      decision_threshold=threshold,
      write_probability=args.write_probability,
    )

  print("[INFO] Classification script finished.")


if __name__ == "__main__":
  main()
