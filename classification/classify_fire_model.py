#!/usr/bin/env python3
"""
Local-only burned area classification pipeline (HPC-friendly).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import tensorflow.compat.v1 as tf
from scipy import ndimage

tf.disable_v2_behavior()


def fully_connected_layer(input_tensor, n_neurons, activation=None):
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


def create_model_graph(hyperparameters):
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

        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name="cross_entropy_loss",
        )
        tf.train.AdamOptimizer(hyperparameters["lr"]).minimize(cross_entropy)
        tf.argmax(logits, 1, name="predicted_class")
        saver = tf.train.Saver()
    return graph, {"x_input": x_input}, saver


def classify_pixels(data_vector, model_path, hyperparameters, block_size):
    num_pixels = data_vector.shape[0]
    num_blocks = (num_pixels + block_size - 1) // block_size
    output_blocks = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)
        print(f"[INFO] Processing block {i + 1}/{num_blocks} (pixels {start_idx} to {end_idx})")
        data_block = data_vector[start_idx:end_idx]

        tf.compat.v1.reset_default_graph()
        graph, placeholders, saver = create_model_graph(hyperparameters)
        with tf.Session(graph=graph) as sess:
            saver.restore(sess, str(model_path))
            output_block = sess.run(
                graph.get_tensor_by_name("predicted_class:0"),
                feed_dict={placeholders["x_input"]: data_block},
            )
            output_blocks.append(output_block)

    return np.concatenate(output_blocks, axis=0)


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


def load_hyperparameters(hyperparameters_path):
    with hyperparameters_path.open("r", encoding="utf-8") as json_file:
        hyperparameters = json.load(json_file)

    dataset_schema = hyperparameters.get("DATASET_SCHEMA")
    if dataset_schema is None:
        raise RuntimeError(
            "DATASET_SCHEMA not found in model hyperparameters. "
            "This model was likely trained with an older pipeline."
        )

    hyperparameters["data_mean"] = np.array(hyperparameters["data_mean"])
    hyperparameters["data_std"] = np.array(hyperparameters["data_std"])
    hyperparameters["data_std"] = np.where(hyperparameters["data_std"] == 0, 1.0, hyperparameters["data_std"])
    return hyperparameters


def classify_single_mosaic(
    mosaic_path,
    output_path,
    model_path,
    hyperparameters,
    block_size=40000000,
    opening_filter_size=None,
    closing_filter_size=None,
):
    with rasterio.open(mosaic_path) as src:
        data = src.read()
        profile = src.profile.copy()

    data = np.transpose(data, (1, 2, 0))
    input_band_indices = hyperparameters["DATASET_SCHEMA"]["INPUT_BAND_INDICES"]
    data_classify = data[:, :, input_band_indices]
    data_classify_vector = data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])
    data_classify_vector = np.nan_to_num(data_classify_vector, nan=0.0)

    if data_classify_vector.shape[1] != hyperparameters["NUM_INPUT"]:
        raise RuntimeError(
            f"Band mismatch: model expects {hyperparameters['NUM_INPUT']} bands, "
            f"but classification data has {data_classify_vector.shape[1]}"
        )

    output_data_classified = classify_pixels(data_classify_vector, model_path, hyperparameters, block_size=block_size)
    output_image_data = output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])
    filtered = apply_spatial_filter(output_image_data, opening_filter_size, closing_filter_size)

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="deflate",
        predictor=2,
        tiled=True,
        nodata=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    print(f"[INFO] Saved classified raster: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify burned area using local model and local mosaics.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local checkpoint base path (without extension), e.g. /data/models/col1_chile_v1_r2_rnn_lstm_ckpt",
    )
    parser.add_argument(
        "--hyperparameters-path",
        default=None,
        help="Optional hyperparameters JSON path. Defaults to <model-path>_hyperparameters.json",
    )
    parser.add_argument(
        "--mosaics",
        nargs="+",
        required=True,
        help="Local mosaic tif path(s) to classify.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local output directory for classified rasters.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=40000000,
        help="Number of pixels per block during inference.",
    )
    parser.add_argument(
        "--opening-filter-size",
        type=int,
        default=2,
        help="Opening filter size. Use 0 to disable opening filter.",
    )
    parser.add_argument(
        "--closing-filter-size",
        type=int,
        default=4,
        help="Closing filter size. Use 0 to disable closing filter.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    hyperparameters_path = (
        Path(args.hyperparameters_path)
        if args.hyperparameters_path
        else Path(f"{args.model_path}_hyperparameters.json")
    )
    output_dir = Path(args.output_dir)

    if not model_path.exists() and not Path(f"{args.model_path}.meta").exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path} (or {args.model_path}.meta)"
        )
    if not hyperparameters_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparameters_path}")

    hyperparameters = load_hyperparameters(hyperparameters_path)
    opening_filter = False if args.opening_filter_size == 0 else args.opening_filter_size
    closing_filter = False if args.closing_filter_size == 0 else args.closing_filter_size

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
        )

    print("[INFO] Classification script finished.")


if __name__ == "__main__":
    main()
