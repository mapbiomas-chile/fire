# last_update: '2026/01/27', github:'mapbiomas/chile-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_3_1_tensorflow_classification_burned_area.py
### Step A_3_1 - Functions for TensorFlow classification of burned areas

# ====================================
# 📦 INSTALL AND IMPORT LIBRARIES
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

from scipy import ndimage
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee
from tqdm import tqdm
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
import pyproj

# ====================================
# 🌍 SAFE GLOBALS / DEFAULTS
# ====================================

if 'bucket_name' not in globals():
    bucket_name = 'mapbiomas-fire'

if 'country' not in globals():
    country = 'chile'

if 'collection_name' not in globals():
    collection_name = 'col1'  # nombre lógico del modelo

if 'collection_folder' not in globals():
    collection_folder = 'collection1'  # carpeta GCS

if 'models_folder' not in globals():
    models_folder = f'models_{collection_name}'

if 'mosaics_folder' not in globals():
    mosaics_folder = 'mosaics_cog'

if 'ee_collection_folder' not in globals():
    ee_collection_folder = 'COLLECTION1'

if 'ee_project' not in globals():
    ee_project = f'mapbiomas-{country}'

if 'BASE_DATASET_PATH' not in globals():
    raise RuntimeError("[ERROR] BASE_DATASET_PATH is not defined. Run A_0_1 first.")

LOCAL_BASE_FOLDER = f"/content/{BASE_DATASET_PATH}"

# ====================================
# 🧰 SUPPORT FUNCTIONS (utils)
# ====================================

def load_image(image_path):
    log_message(f"[INFO] Loading image from path: {image_path}")
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

def convert_to_array(dataset):
    log_message(f"[INFO] Converting dataset to NumPy array")
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return stacked_data

def reshape_image_output(output_data_classified, data_classify):
    log_message(f"[INFO] Reshaping classified data back to image format")
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

def reshape_single_vector(data_classify):
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

def filter_spatial(output_image_data):
    try:
        cfs = closing_filter_size
    except NameError:
        cfs = None

    try:
        ofs = opening_filter_size
    except NameError:
        ofs = None

    log_message("[INFO] Applying spatial filtering on classified image")

    binary_image = output_image_data > 0

    if ofs is False:
        log_message("[INFO] Skipping opening filter step as requested.")
        open_image = binary_image
    else:
        try:
            m = int(ofs) if ofs is not None else 2
        except (ValueError, TypeError):
            log_message(f"[WARNING] Invalid opening filter size '{ofs}'; defaulting to 2x2.")
            m = 2

        log_message(f"[INFO] Applying opening filter with {m}x{m} structuring element.")
        open_image = ndimage.binary_opening(binary_image, structure=np.ones((m, m)))

    if cfs is False:
        log_message("[INFO] Skipping closing filter step as requested.")
        close_image = open_image
    else:
        try:
            n = int(cfs) if cfs is not None else 4
        except (ValueError, TypeError):
            log_message(f"[WARNING] Invalid closing filter size '{cfs}'; defaulting to 4x4.")
            n = 4

        log_message(f"[INFO] Applying closing filter with {n}x{n} structuring element.")
        close_image = ndimage.binary_closing(open_image, structure=np.ones((n, n)))

    return close_image.astype('uint8')

def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    log_message(f"[INFO] Converting array to GeoTIFF raster: {output_image_name}")
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')

    options = [
        'COMPRESS=DEFLATE',
        'PREDICTOR=2',
        'TILED=YES',
        'BIGTIFF=YES'
    ]
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Byte, options=options)

    image_data_scene_uint8 = image_data_scene.astype('uint8')
    outDs.GetRasterBand(1).WriteArray(image_data_scene_uint8)
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None
    log_message(f"[INFO] Raster conversion completed and saved as: {output_image_name}")

def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    log_message(f"[INFO] Checking for significant intersection with minimum area of {min_intersection_area}")
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

def reproject_geometry(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)

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

def build_vrt(vrt_path, input_tif_list):
    if isinstance(input_tif_list, str):
        input_tif_list = input_tif_list.split()

    missing_files = [f for f in input_tif_list if not os.path.exists(f)]
    if missing_files:
        raise RuntimeError(f"The following input files do not exist: {missing_files}")

    if os.path.exists(vrt_path):
        log_message(f"[INFO] VRT already exists. Removing: {vrt_path}")
        os.remove(vrt_path)

    vrt = gdal.BuildVRT(vrt_path, input_tif_list)
    if vrt is None:
        raise RuntimeError(f"Failed to create VRT at {vrt_path}")
    vrt = None

def translate_to_tiff(vrt_path, output_path):
    if os.path.exists(output_path):
        log_message(f"[INFO] TIFF already exists. Removing: {output_path}")
        os.remove(output_path)

    options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "COPY_SRC_OVERVIEWS=YES",
            "BIGTIFF=YES"
        ],
        noData=0
    )
    result = gdal.Translate(output_path, vrt_path, options=options)
    if result is None:
        raise RuntimeError(f"Failed to translate VRT to TIFF: {output_path}")
    result = None

def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_list, suffix=""):
    try:
        name_out_vrt_suffixed = name_out_vrt.replace(".tif", f"{suffix}.vrt") if suffix else name_out_vrt.replace(".tif", ".vrt")
        name_out_tif_suffixed = name_out_tif.replace(".tif", f"{suffix}.tif") if suffix else name_out_tif

        log_message(f"[INFO] Building VRT from: {files_tif_list}")
        build_vrt(name_out_vrt_suffixed, files_tif_list)
        log_message(f"[INFO] VRT created: {name_out_vrt_suffixed}")

        log_message(f"[INFO] Translating VRT to optimized TIFF: {name_out_tif_suffixed}")
        translate_to_tiff(name_out_vrt_suffixed, name_out_tif_suffixed)
        log_message(f"[INFO] Optimized TIFF saved: {name_out_tif_suffixed}")

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
            os.makedirs(directory)
            log_message(f"[INFO] Cleaned and recreated directory: {directory}")
        else:
            os.makedirs(directory)
            log_message(f"[INFO] Created directory: {directory}")

def check_or_create_collection(collection, ee_project):
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        print(f'[INFO] Criando nova coleção no GEE: {collection}')
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        os.system(create_command)
    else:
        print(f'[INFO] Coleção já existe: {collection}')

def upload_to_gee(gcs_path, asset_id, satellite, region, year, version):
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
        f'--pyramiding_policy=mode '
        f'--property satellite={satellite} '
        f'--property region={region} '
        f'--property year={year} '
        f'--property version={version} '
        f'--property source=IPAM '
        f'--property type=annual_burned_area '
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

def fully_connected_layer(input, n_neurons, activation=None):
    input_size = input.get_shape().as_list()[1]

    W = tf.Variable(
        tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))),
        name='W'
    )
    b = tf.Variable(tf.zeros([n_neurons]), name='b')

    layer = tf.matmul(input, W) + b

    if activation == 'relu':
        layer = tf.nn.relu(layer)

    return layer

def create_model_graph(hyperparameters):
    graph = tf.Graph()

    with graph.as_default():
        x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

        data_mean = np.array(hyperparameters['data_mean'], dtype=np.float32)
        data_std = np.array(hyperparameters['data_std'], dtype=np.float32)
        data_std = np.where(data_std == 0, 1e-6, data_std)

        normalized = (x_input - data_mean) / data_std

        hidden1 = fully_connected_layer(normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu')

        logits = fully_connected_layer(hidden5, n_neurons=hyperparameters['NUM_CLASSES'])

        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name='cross_entropy_loss'
        )

        tf.train.AdamOptimizer(hyperparameters['lr']).minimize(cross_entropy)
        tf.argmax(logits, 1, name='predicted_class')

        tf.global_variables_initializer()
        saver = tf.train.Saver()

    return graph, {'x_input': x_input, 'y_input': y_input}, saver

def classify(data_classify_vector, model_path, hyperparameters, block_size=40000000):
    log_message(f"[INFO] Starting classification with model at path: {model_path}")

    num_pixels = data_classify_vector.shape[0]
    num_blocks = (num_pixels + block_size - 1) // block_size
    output_blocks = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)
        log_message(f"[INFO] Processing block {i+1}/{num_blocks} (pixels {start_idx} to {end_idx})")

        data_block = data_classify_vector[start_idx:end_idx]

        tf.compat.v1.reset_default_graph()

        graph, placeholders, saver = create_model_graph(hyperparameters)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)

            output_block = sess.run(
                graph.get_tensor_by_name('predicted_class:0'),
                feed_dict={placeholders['x_input']: data_block}
            )

            output_blocks.append(output_block)

    output_data_classify = np.concatenate(output_blocks, axis=0)
    log_message(f"[INFO] Classification completed")

    return output_data_classify

def process_single_image(dataset_classify, version, region, folder_temp):
    gcs_model_file = (
        f'gs://{BASE_DATASET_PATH}/{models_folder}/'
        f'{collection_name}_{country}_{version}_{region}_rnn_lstm_ckpt*'
    )

    model_file_local_temp = (
        f'{folder_temp}/{collection_name}_{country}_{version}_{region}_rnn_lstm_ckpt'
    )

    log_message(f"[INFO] Downloading TensorFlow model from GCS {gcs_model_file} to {folder_temp}.")

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

    log_message(f"[INFO] Converting GDAL dataset to NumPy array.")
    data_classify = convert_to_array(dataset_classify)
    data_classify = data_classify[:, :, input_band_indices]

    log_message(f"[INFO] Reshaping data into a single pixel vector.")
    data_classify_vector = reshape_single_vector(data_classify)

    log_message(
        f"[INFO] Using input bands indices from model: {input_band_indices} "
        f"(total={len(input_band_indices)})"
    )

    if data_classify_vector.shape[1] != hyperparameters["NUM_INPUT"]:
        raise RuntimeError(
            f"[ERROR] Band mismatch: model expects {hyperparameters['NUM_INPUT']} bands, "
            f"but classification data has {data_classify_vector.shape[1]}"
        )

    log_message(f"[INFO] Running classification using the model.")
    output_data_classified = classify(data_classify_vector, model_file_local_temp, hyperparameters)

    log_message(f"[INFO] Reshaping classified data back into image format.")
    output_image_data = reshape_image_output(output_data_classified, data_classify)

    log_message(f"[INFO] Applying spatial filtering and completing the processing of this scene.")
    return filter_spatial(output_image_data)

def process_year_by_satellite(
    satellite_years,
    bucket_name,
    folder_mosaic,
    folder_temp,
    suffix,
    ee_project,
    country,
    version,
    region,
    simulate_test=False
):
    log_message(f"[INFO] Processing year by satellite for country: {country}, version: {version}, region: {region}")

    grid = ee.FeatureCollection(
        f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}'
    )
    grid_landsat = grid.getInfo()['features']
    start_time = time.time()

    gee_collection_path = (
        f'projects/{ee_project}/assets/FIRE/{ee_collection_folder}/CLASSIFICATION/'
        f'burned_area_{country}_{version}'
    )
    check_or_create_collection(gee_collection_path, ee_project)

    for satellite_year in satellite_years[:1 if simulate_test else None]:
        satellite = satellite_year['satellite']
        years = satellite_year['years'][:1 if simulate_test else None]

        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                test_tag = "_test" if simulate_test else ""
                image_name = f"burned_area_{country}_{satellite}_{version}_region{region[1:]}_{year}{suffix}{test_tag}"
                gcs_filename = f'gs://{BASE_DATASET_PATH}/result_classified/{image_name}.tif'

                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{BASE_DATASET_PATH}/{mosaics_folder}/{satellite}_{country}_{region}_{year}_cog.tif'

                if not os.path.exists(local_cog_path):
                    log_message(f"[INFO] Downloading COG from GCS: {gcs_cog_path}")
                    os.system(f'gsutil cp "{gcs_cog_path}" "{local_cog_path}"')
                    time.sleep(2)
                    fs.invalidate_cache()

                input_scenes = []
                grids_to_process = [grid_landsat[0]] if simulate_test else grid_landsat

                with tqdm(total=len(grids_to_process), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid_feature in grids_to_process:
                        orbit = grid_feature['properties']['ORBITA']
                        point = grid_feature['properties']['PONTO']
                        geometry_scene = grid_feature['geometry']

                        output_image_name = (
                            f'{folder_temp}/image_{collection_name}_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'
                        )
                        nbr_clipped = (
                            f'{folder_temp}/image_mosaic_{collection_name}_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        )

                        if os.path.isfile(output_image_name):
                            log_message(f"[INFO] Scene {orbit}/{point} already processed. Skipping.")
                            pbar_scenes.update(1)
                            continue

                        clipping_success = clip_image_by_grid(geometry_scene, local_cog_path, nbr_clipped)

                        if clipping_success:
                            dataset_classify = load_image(nbr_clipped)
                            image_data = process_single_image(dataset_classify, version, region, folder_temp)

                            if image_data is not None:
                                convert_to_raster(dataset_classify, image_data, output_image_name)
                                input_scenes.append(output_image_name)

                            remove_temporary_files([nbr_clipped])
                        else:
                            log_message(f"[WARNING] Clipping failed for scene {orbit}/{point}.")

                        pbar_scenes.update(1)

                if input_scenes:
                    merge_output_temp = f"{folder_temp}/merged_temp_{year}.tif"
                    output_image = f"{folder_temp}/{image_name}.tif"

                    generate_optimized_image(merge_output_temp, output_image, input_scenes)

                    wait_time = 0
                    while not os.path.exists(output_image) and wait_time < 10:
                        time.sleep(1)
                        wait_time += 1

                    if not os.path.exists(output_image):
                        log_message(f"[ERROR] Output image not found locally after wait. Skipping upload: {output_image}")
                        continue

                    size_mb = os.path.getsize(output_image) / (1024 * 1024)
                    if size_mb < 0.01:
                        log_message(f"[ERROR] Output image too small ({size_mb:.2f} MB). Likely failed.")
                        continue

                    log_message(f"[INFO] Output image verified. Size: {size_mb:.2f} MB")

                    status_upload = os.system(f'gsutil cp "{output_image}" "{gcs_filename}"')
                    time.sleep(2)
                    fs.invalidate_cache()

                    if status_upload == 0:
                        log_message(f"[INFO] Upload to GCS succeeded: {gcs_filename}")
                        if os.system(f'gsutil ls "{gcs_filename}"') == 0:
                            upload_to_gee(
                                gcs_filename,
                                f'{gee_collection_path}/{image_name}',
                                satellite,
                                region,
                                year,
                                version
                            )
                        else:
                            log_message(f"[ERROR] File not found on GCS after upload.")
                    else:
                        log_message(f"[ERROR] Upload to GCS failed with code {status_upload}")

                clean_directories([folder_temp])
                elapsed = time.time() - start_time
                log_message(f"[INFO] Year {year} processing completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                pbar_years.update(1)

def render_classify_models(models_to_classify, simulate_test=False):
    log_message(f"[INFO] [render_classify_models] STARTING PROCESSINGS FOR CLASSIFY MODELS {models_to_classify}")

    for model_info in models_to_classify:
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info["simulation"]

        log_message(f"[INFO] Processing model: {model_name}")
        log_message(f"[INFO] Selected mosaics: {mosaics}")
        log_message(f"[INFO] Simulation mode: {simulation}")

        parts = model_name.split('_')
        if len(parts) < 4:
            log_message(f"[ERROR] Unexpected model name format: {model_name}")
            continue

        model_country = parts[1]
        version = parts[2]
        region = parts[3]

        folder_temp = f'{LOCAL_BASE_FOLDER}/tmp1'
        folder_mosaic = f'{LOCAL_BASE_FOLDER}/{mosaics_folder}'

        log_message(f"[INFO] Starting the classification process for country: {model_country}.")

        for directory in [folder_temp, folder_mosaic]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                log_message(f"[INFO] Directory already exists: {directory}")

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
            log_message(f"[SIMULATION] Would process model: {model_name} with mosaics: {mosaics}")
        else:
            process_year_by_satellite(
                satellite_years=satellite_years,
                bucket_name=bucket_name,
                folder_mosaic=folder_mosaic,
                folder_temp=folder_temp,
                suffix='',
                ee_project=f'mapbiomas-{model_country}',
                country=model_country,
                version=version,
                region=region,
                simulate_test=simulate_test
            )

    log_message(f"[INFO] [render_classify_models] FINISH PROCESSINGS FOR CLASSIFY MODELS {models_to_classify}")
