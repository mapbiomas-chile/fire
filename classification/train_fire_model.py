#!/usr/bin/env python3
"""
Local-only training pipeline for burned area model (HPC-friendly).

This script keeps the training architecture and artifact naming compatible
with the original notebook pipeline, but removes all external interactions.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import rasterio
import tensorflow.compat.v1 as tf

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


def infer_dataset_schema(sample_path, label_name="landcover"):
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


def process_image(image_path):
    with rasterio.open(image_path) as dataset:
        data = dataset.read()
    data = np.transpose(data, (1, 2, 0))
    vector = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    # Keep rows where at least one band is valid.
    cleaned = vector[~np.isnan(vector).all(axis=1)]
    return cleaned


def filter_valid_data_and_shuffle(data, seed=42):
    mask = np.all(~np.isnan(data), axis=1)
    valid_data = data[mask]
    rng = np.random.default_rng(seed)
    rng.shuffle(valid_data)
    return valid_data


def train_model(training_data, validation_data, bi, li, model_path, json_path, data_mean, data_std):
    lr = 0.001
    batch_size = 1000
    n_iter = 7000
    num_input = len(bi)
    num_classes = 2

    num_n_l1 = 7
    num_n_l2 = 14
    num_n_l3 = 7
    num_n_l4 = 14
    num_n_l5 = 7

    graph = tf.Graph()
    with graph.as_default():
        x_input = tf.placeholder(tf.float32, shape=[None, num_input], name="x_input")
        y_input = tf.placeholder(tf.int64, shape=[None], name="y_input")

        normalized = (x_input - data_mean) / data_std
        hidden1 = fully_connected_layer(normalized, n_neurons=num_n_l1, activation="relu")
        hidden2 = fully_connected_layer(hidden1, n_neurons=num_n_l2, activation="relu")
        hidden3 = fully_connected_layer(hidden2, n_neurons=num_n_l3, activation="relu")
        hidden4 = fully_connected_layer(hidden3, n_neurons=num_n_l4, activation="relu")
        hidden5 = fully_connected_layer(hidden4, n_neurons=num_n_l5, activation="relu")
        logits = fully_connected_layer(hidden5, n_neurons=num_classes)

        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
        )
        optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        outputs = tf.argmax(logits, 1)
        correct_prediction = tf.equal(outputs, y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    train_size = training_data.shape[0]
    if train_size < batch_size:
        batch_size = train_size

    if train_size < 2:
        raise ValueError("Training set has fewer than 2 rows after filtering.")

    start_time = time.time()
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        validation_dict = {x_input: validation_data[:, bi], y_input: validation_data[:, li]}

        for i in range(n_iter + 1):
            idx = np.random.choice(train_size, batch_size, replace=False)
            batch = training_data[idx, :]
            feed_dict = {x_input: batch[:, bi], y_input: batch[:, li]}
            sess.run(optimizer, feed_dict=feed_dict)

            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict=validation_dict) * 100
                saver.save(sess, str(model_path))
                print(f"[PROGRESS] Iteration {i}/{n_iter} - Validation Accuracy: {acc:.2f}%")

    duration = time.time() - start_time
    print(f"[INFO] Training completed in: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

    hyperparameters = {
        "data_mean": data_mean.tolist(),
        "data_std": data_std.tolist(),
        "lr": lr,
        "NUM_N_L1": num_n_l1,
        "NUM_N_L2": num_n_l2,
        "NUM_N_L3": num_n_l3,
        "NUM_N_L4": num_n_l4,
        "NUM_N_L5": num_n_l5,
        "NUM_CLASSES": num_classes,
        "NUM_INPUT": num_input,
        "DATASET_SCHEMA": {
            "INPUT_BAND_INDICES": bi,
            "LABEL_BAND_INDEX": li,
            "LABEL_NAME": "landcover",
        },
    }
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(hyperparameters, json_file)

    print(f"[INFO] Final model saved at: {model_path}")
    print(f"[INFO] Hyperparameters saved at: {json_path}")


def select_training_files(training_samples_dir, version, region):
    pattern = re.compile(rf".*_({version})_.*_{region}_.*\.tif$")
    matches = [
        p for p in sorted(training_samples_dir.glob("*.tif")) if pattern.search(p.name)
    ]
    return matches


def main():
    parser = argparse.ArgumentParser(description="Train fire model (local-only, HPC-friendly).")
    parser.add_argument("--country", default="chile", help="Country token for model naming.")
    parser.add_argument("--version", required=True, help='Version token, e.g. "v1".')
    parser.add_argument("--region", required=True, help='Region token, e.g. "r2".')
    parser.add_argument(
        "--training-samples-dir",
        required=True,
        help="Local folder with training sample TIFFs.",
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Local output folder for model checkpoints and hyperparameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle and batch sampling.",
    )
    args = parser.parse_args()

    training_samples_dir = Path(args.training_samples_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not training_samples_dir.exists():
        raise FileNotFoundError(f"Training samples dir does not exist: {training_samples_dir}")

    selected_files = select_training_files(training_samples_dir, args.version, args.region)
    if not selected_files:
        available = [p.name for p in sorted(training_samples_dir.glob("*.tif"))]
        raise RuntimeError(
            "No training files matched the selected version/region.\n"
            f"selector: trainings_{args.version}_{args.region}\n"
            f"dir: {training_samples_dir}\n"
            f"available_tifs: {available}"
        )

    print(f"[INFO] Selected files for training: {[p.name for p in selected_files]}")
    first_sample_path = selected_files[0]
    dataset_schema = infer_dataset_schema(first_sample_path, label_name="landcover")
    print(f"[INFO] Inferred dataset schema: {dataset_schema}")

    vectors = []
    for image_path in selected_files:
        print(f"[INFO] Processing image: {image_path}")
        cleaned = process_image(image_path)
        if cleaned.shape[0] > 0:
            print(f"[INFO] Valid pixels: {cleaned.shape[0]}")
            vectors.append(cleaned)
        else:
            print(f"[WARNING] 0 valid pixels in: {image_path}")

    if not vectors:
        raise RuntimeError("No valid data found across selected samples.")

    data = np.concatenate(vectors)
    valid_data = filter_valid_data_and_shuffle(data, seed=args.seed)
    if valid_data.shape[0] == 0:
        raise RuntimeError("No valid rows after NaN filtering.")

    train_fraction = 0.7
    training_size = int(valid_data.shape[0] * train_fraction)
    if training_size <= 0 or training_size >= valid_data.shape[0]:
        raise RuntimeError(
            f"Invalid split sizes. total={valid_data.shape[0]}, training_size={training_size}"
        )

    training_data = valid_data[:training_size, :]
    validation_data = valid_data[training_size:, :]
    bi = dataset_schema["INPUT_BAND_INDICES"]
    li = dataset_schema["LABEL_BAND_INDEX"]

    data_mean = training_data[:, bi].mean(axis=0)
    data_std = training_data[:, bi].std(axis=0)
    data_std = np.where(data_std == 0, 1.0, data_std)

    model_base = f"col1_{args.country}_{args.version}_{args.region}_rnn_lstm_ckpt"
    model_path = models_dir / model_base
    json_path = models_dir / f"{model_base}_hyperparameters.json"

    print(f"[INFO] Training set size: {training_data.shape[0]}")
    print(f"[INFO] Validation set size: {validation_data.shape[0]}")
    print(f"[INFO] Mean of training bands: {data_mean}")
    print(f"[INFO] Standard deviation of training bands: {data_std}")

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    train_model(training_data, validation_data, bi, li, model_path, json_path, data_mean, data_std)


if __name__ == "__main__":
    main()
