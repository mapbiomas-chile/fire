# A_4_1_tensorflow_feature_maps_extraction.py
# last update: '2026/01/27'
# MapBiomas Fire Classification Algorithms Step A_4_1
# Functions for TensorFlow Embedding / Feature Maps Extraction

# ====================================
# 📦 IMPORT LIBRARIES
# ====================================

import os
import time
import math
import json
import shutil
import subprocess
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee
from tqdm import tqdm
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
import pyproj
from scipy import ndimage

# ====================================
# 🌍 SAFE GLOBALS / DEFAULTS
# ====================================

if 'bucket_name' not in globals():
    bucket_name = 'mapbiomas-fire'

if 'country' not in globals():
    country = 'chile'

if 'collection_name' not in globals():
    collection_name = 'col1'  # logical model collection

if 'collection_folder' not in globals():
    collection_folder = 'collection1'  # GCS folder

if 'models_folder' not in globals():
    models_folder = f'models_{collection_name}'

if 'mosaics_folder' not in globals():
    mosaics_folder = 'mosaics_cog'

if 'ee_project' not in globals():
    ee_project = f'mapbiomas-{country}'

if 'ee_collection_folder' not in globals():
    ee_collection_folder = 'COLLECTION1'

if 'BASE_DATASET_PATH' not in globals():
    raise RuntimeError("[ERROR] BASE_DATASET_PATH is not defined. Run A_0_1 first.")

if 'fs' not in globals():
    import gcsfs
    fs = gcsfs.GCSFileSystem(project=ee_project)

# Use real logger if it exists; fallback only if absent
if 'log_message' not in globals():
    def log_message(msg):
        print(f"[LOG] {msg}")

LOCAL_BASE_FOLDER = f"/content/{BASE_DATASET_PATH}"

# ====================================
# 🧰 SUPPORT FUNCTIONS
# ====================================

def fully_connected_layer(input_tensor, n_neurons, activation=None, name=None):
    input_size = input_tensor.get_shape().as_list()[1]

    W = tf.Variable(
        tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))),
        name=f'W_{name}' if name else 'W'
    )
    b = tf.Variable(
        tf.zeros([n_neurons]),
        name=f'b_{name}' if name else 'b'
    )

    layer = tf.matmul(input_tensor, W) + b

    if activation == 'relu':
        layer = tf.nn.relu(layer)

    if name:
        layer = tf.identity(layer, name=name)

    return layer

def load_image(image_path):
    log_message(f"[INFO] Loading image from path: {image_path}")
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

def convert_to_array(dataset):
    log_message("[INFO] Converting dataset to NumPy array")
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return stacked_data

def reshape_single_vector(data_classify):
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

def reproject_geometry(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)

def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

def clip_image_by_grid(geom, image, output, buffer_distance_meters=100, max_attempts=5, retry_delay=5):
    attempt = 0
    while attempt < max_attempts:
        try:
            log_message(f"[INFO] Attempt {attempt+1}/{max_attempts} to clip image: {image}")
            with rasterio.open(image) as src:
                image_crs = src.crs
                geom_shape = shape(geom)
                geom_proj = reproject_geometry(geom_shape, 'EPSG:4326', image_crs)
                expanded_geom = geom_proj.buffer(buffer_distance_meters)
                expanded_geom_geojson = mapping(expanded_geom)

                if has_significant_intersection(expanded_geom_geojson, src.bounds):
                    out_image, out_transform = mask(src, [expanded_geom_geojson], crop=True, nodata=np.nan, filled=True)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "crs": src.crs
                    })

                    with rasterio.open(output, 'w', **out_meta) as dest:
                        dest.write(out_image)

                    log_message(f"[INFO] Image clipped successfully: {output}")
                    return True
                else:
                    log_message(f"[INFO] Insufficient overlap for clipping: {image}")
                    return False
        except Exception as e:
            log_message(f"[ERROR] Error during clipping: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1

    log_message(f"[ERROR] Failed to clip image after {max_attempts} attempts: {image}")
    return False

def convert_to_multiband_raster(dataset_classify, image_data_scene_hwc, output_image_name):
    log_message(f"[INFO] Converting embedding array to multi-band GeoTIFF: {output_image_name}")

    rows, cols, bands = image_data_scene_hwc.shape
    driver = gdal.GetDriverByName('GTiff')

    options = [
        'COMPRESS=DEFLATE',
        'PREDICTOR=2',
        'TILED=YES',
        'BIGTIFF=YES'
    ]

    out_ds = driver.Create(output_image_name, cols, rows, bands, gdal.GDT_Float32, options=options)

    for b in range(bands):
        out_ds.GetRasterBand(b + 1).WriteArray(image_data_scene_hwc[:, :, b].astype(np.float32))

    out_ds.SetGeoTransform(dataset_classify.GetGeoTransform())
    out_ds.SetProjection(dataset_classify.GetProjection())
    out_ds.FlushCache()
    out_ds = None

    log_message(f"[INFO] Multi-band raster saved: {output_image_name}")

def build_vrt(vrt_path, input_tif_list):
    if isinstance(input_tif_list, str):
        input_tif_list = input_tif_list.split()

    missing_files = [f for f in input_tif_list if not os.path.exists(f)]
    if missing_files:
        raise RuntimeError(f"The following input files do not exist: {missing_files}")

    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    vrt = gdal.BuildVRT(vrt_path, input_tif_list)
    if vrt is None:
        raise RuntimeError(f"Failed to create VRT at {vrt_path}")
    vrt = None

def translate_to_tiff(vrt_path, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)

    options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "COPY_SRC_OVERVIEWS=YES",
            "BIGTIFF=YES"
        ]
    )
    result = gdal.Translate(output_path, vrt_path, options=options)
    if result is None:
        raise RuntimeError(f"Failed to translate VRT to TIFF: {output_path}")
    result = None

def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_list, suffix=""):
    try:
        name_out_vrt_suffixed = name_out_vrt.replace(".tif", f"{suffix}.vrt") if suffix else name_out_vrt.replace(".tif", ".vrt")
        name_out_tif_suffixed = name_out_tif.replace(".tif", f"{suffix}.tif") if suffix else name_out_tif

        build_vrt(name_out_vrt_suffixed, files_tif_list)
        translate_to_tiff(name_out_vrt_suffixed, name_out_tif_suffixed)
    except Exception as e:
        log_message(f"[ERROR] Failed to generate optimized image. {e}")
        return False

    if not os.path.exists(name_out_tif_suffixed):
        log_message(f"[ERROR] Output image not found locally after generation: {name_out_tif_suffixed}")
        return False

    return True

def clean_directories(directories_to_clean):
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        log_message(f"[INFO] Directory ready: {directory}")

def check_or_create_collection(collection, ee_project):
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        print(f'[INFO] Creating new GEE collection: {collection}')
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        os.system(create_command)
    else:
        print(f'[INFO] Collection already exists: {collection}')

def upload_to_gee(gcs_path, asset_id, satellite, region, year, version, embedding_layer):
    timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.now().strftime('%Y-%m-%d')

    try:
        ee.data.getAsset(asset_id)
        log_message(f"[INFO] Asset already exists. Deleting: {asset_id}")
        ee.data.deleteAsset(asset_id)
        time.sleep(2)
    except ee.EEException:
        log_message(f"[INFO] Asset does not exist yet. Proceeding with upload: {asset_id}")

    upload_command = (
        f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
        f'--pyramiding_policy=sample '
        f'--property satellite={satellite} '
        f'--property region={region} '
        f'--property year={year} '
        f'--property version={version} '
        f'--property embedding_layer={embedding_layer} '
        f'--property source=IPAM '
        f'--property type=annual_embedding_feature_map '
        f'--property time_start={timestamp_start} '
        f'--property time_end={timestamp_end} '
        f'--property create_date={creation_date} '
        f'{gcs_path}'
    )

    log_message(f"[INFO] Starting upload to GEE: {asset_id}")
    status = os.system(upload_command)

    if status == 0:
        log_message(f"[INFO] Upload completed successfully: {asset_id}")
    else:
        log_message(f"[ERROR] Upload failed for GEE asset: {asset_id}")
        log_message(f"[ERROR] Command status code: {status}")

def remove_temporary_files(files_to_remove):
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                log_message(f"[INFO] Temporary file removed: {file}")
            except Exception as e:
                log_message(f"[ERROR] Failed to remove file: {file}. Details: {str(e)}")

# ====================================
# 🧠 MODEL / EMBEDDING FUNCTIONS
# ====================================

def create_embedding_graph(hyperparameters, embedding_layer='h5'):
    graph = tf.Graph()

    with graph.as_default():
        x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

        data_mean = np.array(hyperparameters['data_mean'], dtype=np.float32)
        data_std = np.array(hyperparameters['data_std'], dtype=np.float32)
        data_std = np.where(data_std == 0, 1e-6, data_std)

        normalized = (x_input - data_mean) / data_std

        h1 = fully_connected_layer(normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu', name='h1')
        h2 = fully_connected_layer(h1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu', name='h2')
        h3 = fully_connected_layer(h2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu', name='h3')
        h4 = fully_connected_layer(h3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu', name='h4')
        h5 = fully_connected_layer(h4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu', name='h5')

        logits = fully_connected_layer(h5, n_neurons=hyperparameters['NUM_CLASSES'], name='logits')
        tf.argmax(logits, 1, name='predicted_class')
        tf.global_variables_initializer()
        saver = tf.train.Saver()

    layer_tensor_name = f"{embedding_layer}:0"
    return graph, {'x_input': x_input, 'y_input': y_input}, saver, layer_tensor_name

def extract_embeddings(data_classify_vector, model_path, hyperparameters, embedding_layer='h5', block_size=40000000):
    log_message(f"[INFO] Starting embedding extraction with model at path: {model_path}")
    log_message(f"[INFO] Selected embedding layer: {embedding_layer}")

    num_pixels = data_classify_vector.shape[0]
    num_blocks = (num_pixels + block_size - 1) // block_size
    output_blocks = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)
        log_message(f"[INFO] Processing block {i+1}/{num_blocks} (pixels {start_idx} to {end_idx})")

        data_block = data_classify_vector[start_idx:end_idx]

        tf.compat.v1.reset_default_graph()
        graph, placeholders, saver, layer_tensor_name = create_embedding_graph(
            hyperparameters,
            embedding_layer=embedding_layer
        )

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)

            layer_tensor = graph.get_tensor_by_name(layer_tensor_name)
            embedding_block = sess.run(
                layer_tensor,
                feed_dict={placeholders['x_input']: data_block}
            )
            output_blocks.append(embedding_block)

    output_embeddings = np.concatenate(output_blocks, axis=0)
    log_message(f"[INFO] Embedding extraction completed. Output shape: {output_embeddings.shape}")
    return output_embeddings

def process_single_image_embedding(dataset_classify, version, region, folder_temp, embedding_layer='h5'):
    gcs_model_file = (
        f'gs://{BASE_DATASET_PATH}/{models_folder}/'
        f'{collection_name}_{country}_{version}_{region}_rnn_lstm_ckpt*'
    )

    model_file_local_temp = (
        f'{folder_temp}/{collection_name}_{country}_{version}_{region}_rnn_lstm_ckpt'
    )

    try:
        subprocess.run(f'gsutil cp "{gcs_model_file}" "{folder_temp}"', shell=True, check=True)
        time.sleep(2)
        fs.invalidate_cache()
        log_message(f"[INFO] Model downloaded successfully.")
    except subprocess.CalledProcessError as e:
        log_message(f"[ERROR] Failed to download model from GCS: {e}")
        return None

    json_path = (
        f'{folder_temp}/{collection_name}_{country}_{version}_{region}_rnn_lstm_ckpt_hyperparameters.json'
    )

    with open(json_path, 'r') as json_file:
        hyperparameters = json.load(json_file)

    dataset_schema = hyperparameters.get("DATASET_SCHEMA")
    if dataset_schema is None:
        raise RuntimeError(
            "[ERROR] DATASET_SCHEMA not found in model hyperparameters. "
            "This model was likely trained with an older pipeline."
        )

    input_band_indices = dataset_schema["INPUT_BAND_INDICES"]

    data_classify = convert_to_array(dataset_classify)
    data_classify = data_classify[:, :, input_band_indices]
    data_classify_vector = reshape_single_vector(data_classify)

    if data_classify_vector.shape[1] != hyperparameters["NUM_INPUT"]:
        raise RuntimeError(
            f"[ERROR] Band mismatch: model expects {hyperparameters['NUM_INPUT']} bands, "
            f"but embedding data has {data_classify_vector.shape[1]}"
        )

    output_embeddings = extract_embeddings(
        data_classify_vector,
        model_file_local_temp,
        hyperparameters,
        embedding_layer=embedding_layer
    )

    H, W = data_classify.shape[:2]
    C = output_embeddings.shape[-1]
    output_image_data_hwc = output_embeddings.reshape([H, W, C])

    return output_image_data_hwc

# ====================================
# 🚀 MAIN EMBEDDING WORKFLOW
# ====================================

def process_year_by_satellite_embedding(
    satellite_years,
    bucket_name,
    folder_mosaic,
    folder_temp,
    suffix,
    ee_project,
    country,
    version,
    region,
    simulate_test=False,
    embedding_layer='h5'
):
    try:
        grid = ee.FeatureCollection(
            f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}'
        )
        grid_landsat = grid.getInfo()['features']
    except Exception as e:
        log_message(f"[ERROR] Failed to load GEE grid: {e}")
        return

    start_time = time.time()

    gee_embedding_collection = (
        f'projects/{ee_project}/assets/FIRE/{ee_collection_folder}/CLASSIFICATION_EMBEDDINGS/'
        f'embedding_field_{country}_{version}'
    )
    check_or_create_collection(gee_embedding_collection, ee_project)

    for satellite_year in satellite_years[:1 if simulate_test else None]:
        satellite = satellite_year['satellite']
        years = satellite_year['years'][:1 if simulate_test else None]

        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                test_tag = "_test" if simulate_test else ""
                image_name = (
                    f"embedding_{embedding_layer}_{country}_{satellite}_{version}_"
                    f"region{region[1:]}_{year}{suffix}{test_tag}"
                )

                gcs_filename = f'gs://{BASE_DATASET_PATH}/result_embeddings/{image_name}.tif'
                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{BASE_DATASET_PATH}/{mosaics_folder}/{satellite}_{country}_{region}_{year}_cog.tif'

                if not os.path.exists(local_cog_path):
                    status_dl = os.system(f'gsutil cp "{gcs_cog_path}" "{local_cog_path}"')
                    time.sleep(2)
                    fs.invalidate_cache()
                    if status_dl != 0:
                        log_message(f"[ERROR] Failed to download COG: {gcs_cog_path}")
                        pbar_years.update(1)
                        continue

                input_scenes = []
                grids_to_process = [grid_landsat[0]] if simulate_test else grid_landsat

                with tqdm(total=len(grids_to_process), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid_feature in grids_to_process:
                        orbit = grid_feature['properties']['ORBITA']
                        point = grid_feature['properties']['PONTO']
                        geometry_scene = grid_feature['geometry']

                        output_image_name = (
                            f'{folder_temp}/embedding_{collection_name}_{embedding_layer}_'
                            f'{country}_{region}_{version}_{orbit}_{point}_{year}.tif'
                        )
                        clipped_mosaic = (
                            f'{folder_temp}/embedding_mosaic_{collection_name}_{embedding_layer}_'
                            f'{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        )

                        if os.path.isfile(output_image_name):
                            pbar_scenes.update(1)
                            continue

                        clipping_success = clip_image_by_grid(geometry_scene, local_cog_path, clipped_mosaic)

                        if clipping_success:
                            dataset_classify = load_image(clipped_mosaic)
                            image_data = process_single_image_embedding(
                                dataset_classify,
                                version,
                                region,
                                folder_temp,
                                embedding_layer=embedding_layer
                            )

                            if image_data is not None:
                                convert_to_multiband_raster(dataset_classify, image_data, output_image_name)
                                input_scenes.append(output_image_name)

                            remove_temporary_files([clipped_mosaic])
                        else:
                            log_message(f"[WARNING] Clipping failed for scene {orbit}/{point}.")

                        pbar_scenes.update(1)

                if input_scenes:
                    merge_output_temp = f"{folder_temp}/merged_embedding_temp_{year}.tif"
                    output_image = f"{folder_temp}/{image_name}.tif"

                    generate_optimized_image(merge_output_temp, output_image, input_scenes)

                    wait_time = 0
                    while not os.path.exists(output_image) and wait_time < 10:
                        time.sleep(1)
                        wait_time += 1

                    if not os.path.exists(output_image):
                        log_message(f"[ERROR] Output image not found locally after wait. Skipping upload: {output_image}")
                        pbar_years.update(1)
                        continue

                    size_mb = os.path.getsize(output_image) / (1024 * 1024)
                    if size_mb < 0.01:
                        log_message(f"[ERROR] Output image too small ({size_mb:.2f} MB). Likely failed.")
                        pbar_years.update(1)
                        continue

                    status_upload = os.system(f'gsutil cp "{output_image}" "{gcs_filename}"')
                    time.sleep(2)
                    fs.invalidate_cache()

                    if status_upload == 0:
                        log_message(f"[INFO] Upload to GCS succeeded: {gcs_filename}")
                        if os.system(f'gsutil ls "{gcs_filename}"') == 0:
                            upload_to_gee(
                                gcs_filename,
                                f'{gee_embedding_collection}/{image_name}',
                                satellite,
                                region,
                                year,
                                version,
                                embedding_layer
                            )
                        else:
                            log_message(f"[ERROR] File not found on GCS after upload.")
                    else:
                        log_message(f"[ERROR] Upload to GCS failed with code {status_upload}")

                clean_directories([folder_temp])
                elapsed = time.time() - start_time
                log_message(f"[INFO] Year {year} embedding processing completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                pbar_years.update(1)

def render_embedding_models(models_to_process, simulate_test=False):
    log_message(f"[INFO] [render_embedding_models] STARTING PROCESSINGS FOR EMBEDDING MODELS {models_to_process}")

    for model_info in models_to_process:
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info.get("simulation", False)
        embedding_layer = model_info.get("embedding_layer", "h5")

        log_message(f"[INFO] Processing model: {model_name}")
        log_message(f"[INFO] Selected mosaics: {mosaics}")
        log_message(f"[INFO] Selected embedding layer: {embedding_layer}")
        log_message(f"[INFO] Simulation mode: {simulation}")

        parts = model_name.split('_')
        if len(parts) < 4:
            log_message(f"[ERROR] Unexpected model name format: {model_name}")
            continue

        model_country = parts[1]
        version = parts[2]
        region = parts[3]

        folder_temp = f'{LOCAL_BASE_FOLDER}/tmp_embedding'
        folder_mosaic = f'{LOCAL_BASE_FOLDER}/{mosaics_folder}'

        for directory in [folder_temp, folder_mosaic]:
            os.makedirs(directory, exist_ok=True)

        clean_directories([folder_temp, folder_mosaic])

        satellite_years = []
        for mosaic in mosaics:
            mosaic_parts = mosaic.split('_')
            if len(mosaic_parts) < 4:
                log_message(f"[WARNING] Unexpected mosaic name format, skipping: {mosaic}")
                continue

            satellite = mosaic_parts[0]
            year = int(mosaic_parts[3])
            satellite_years.append({
                "satellite": satellite,
                "years": [year]
            })

        if simulation:
            log_message(f"[SIMULATION] Would process embedding model: {model_name} with mosaics: {mosaics} and layer: {embedding_layer}")
        else:
            process_year_by_satellite_embedding(
                satellite_years=satellite_years,
                bucket_name=bucket_name,
                folder_mosaic=folder_mosaic,
                folder_temp=folder_temp,
                suffix='',
                ee_project=f'mapbiomas-{model_country}',
                country=model_country,
                version=version,
                region=region,
                simulate_test=simulate_test,
                embedding_layer=embedding_layer
            )

    log_message(f"[INFO] [render_embedding_models] FINISH PROCESSINGS FOR EMBEDDING MODELS {models_to_process}")
