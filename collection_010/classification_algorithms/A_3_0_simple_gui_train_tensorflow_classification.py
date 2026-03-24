# last_update: '2025/06/02', github:'mapbiomas/chile-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_3_0_simple_gui_train_tensorflow_classification.py
### Step A_3_0 - Simple graphic user interface for selecting years for burned area classification

# ====================================
# 📦 INSTALL AND IMPORT LIBRARIES
# ====================================

import subprocess
import sys
import importlib
import os
import time
import math
import shutil

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        clear_console()

def clear_console():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

install_and_import('rasterio')
install_and_import('gcsfs')
install_and_import('ipywidgets')

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import VBox, HBox

import numpy as np
from scipy import ndimage
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee
from tqdm import tqdm
from shapely.geometry import shape, box, mapping
import gcsfs

# ====================================
# 🌍 SAFE GLOBALS / DEFAULTS
# ====================================

if 'bucket_name' not in globals():
    bucket_name = 'mapbiomas-fire'

if 'country' not in globals():
    country = 'chile'

if 'collection_name' not in globals():
    collection_name = 'col1'

if 'models_folder' not in globals():
    models_folder = f'models_{collection_name}'

if 'mosaics_folder' not in globals():
    mosaics_folder = 'mosaics_cog'

if 'ee_project' not in globals():
    ee_project = f'mapbiomas-{country}'

if 'fs' not in globals():
    fs = gcsfs.GCSFileSystem(project=ee_project)

if 'BASE_DATASET_PATH' not in globals():
    raise RuntimeError("[ERROR] BASE_DATASET_PATH is not defined. Run A_0_1 first.")

# ====================================
# 🧠 CORE CLASSES
# ====================================

class ModelRepository:
    def __init__(self, bucket_name, country):
        self.bucket = bucket_name
        self.country = country
        self.base_folder = BASE_DATASET_PATH
        self.fs = fs

    def list_models(self):
        training_folder = f"{self.base_folder}/{models_folder}/"
        try:
            files = self.fs.ls(training_folder)
            return [file.split('/')[-1] for file in files if file.endswith('.meta')], len(files)
        except FileNotFoundError:
            return [], 0

    def list_mosaics(self, region):
        mosaics_path = f"{self.base_folder}/{mosaics_folder}/"
        try:
            files = self.fs.ls(mosaics_path)
            return [file.split('/')[-1] for file in files if f"_{region}_" in file], len(files)
        except FileNotFoundError:
            return [], 0

    def list_classified(self):
        classified_folder = f"{self.base_folder}/result_classified/"
        try:
            files = self.fs.ls(classified_folder)
            return [file.split('/')[-1] for file in files], len(files)
        except FileNotFoundError:
            return [], 0

    def is_classified(self, mosaic_file):
        classified_files, _ = self.list_classified()

        parts = mosaic_file.split('_')
        if len(parts) < 4:
            return False

        sat = parts[0]
        region = parts[2]
        year = parts[3]

        for classified_name in classified_files:
            if (
                sat in classified_name and
                f"region{region[1:]}" in classified_name and
                year in classified_name
            ):
                return True
        return False


# ====================================
# 🧰 SUPPORT FUNCTIONS
# ====================================

mosaic_checkboxes_dict = {}
mosaic_checkbox_states = {}

def display_selected_mosaics(model, selected_country, region):
    repo = ModelRepository(bucket_name=bucket_name, country=selected_country)
    mosaic_files, mosaic_count = repo.list_mosaics(region)
    classified_files, classified_count = repo.list_classified()

    mosaics_panel = widgets.Output(
        layout={'border': '1px solid black', 'height': '200px', 'overflow_y': 'scroll'}
    )

    checkboxes_mosaics = []
    saved_states = mosaic_checkbox_states.get(model, None)

    with mosaics_panel:
        if mosaic_files:
            for idx, file in enumerate(mosaic_files):
                classified = repo.is_classified(file)
                checkbox_mosaic = widgets.Checkbox(
                    value=False,
                    description=file + (" ⚠️" if classified else "")
                )

                if saved_states and idx < len(saved_states):
                    checkbox_mosaic.value = saved_states[idx]

                checkboxes_mosaics.append(checkbox_mosaic)
                display(checkbox_mosaic)
        else:
            log_message(f"No mosaics found for region {region}")

    mosaic_checkboxes_dict[model] = checkboxes_mosaics

    def toggle_select_all(change):
        for checkbox in checkboxes_mosaics:
            checkbox.value = change['new']

    select_all_checkbox = widgets.Checkbox(value=False, description="Select All")
    select_all_checkbox.observe(toggle_select_all, names='value')

    legend_panel = widgets.Output(
        layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'}
    )
    with legend_panel:
        print("⚠️ Files already classified. They will overwrite previous classifications if the checkbox remains checked.")

    return widgets.VBox([select_all_checkbox, mosaics_panel, legend_panel])


def update_interface():
    clear_output(wait=True)
    display(
        VBox(
            checkboxes,
            layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0', width='700px')
        )
    )

    mosaic_panels_widgets = [panel[2] for panel in mosaic_panels]
    display(
        HBox(
            mosaic_panels_widgets,
            layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')
        )
    )


def collect_selected_models():
    return [checkbox.description for checkbox in checkboxes if checkbox.value]


def simulate_processing_click(b):
    selected_models = collect_selected_models()
    models_to_simulate = []

    if selected_models:
        for model in selected_models:
            log_message(f'Simulando o processamento do modelo: {model}')
            model_key = f"{model}.meta" if not model.endswith('.meta') else model

            if model_key in mosaic_checkboxes_dict:
                mosaic_checkboxes = mosaic_checkboxes_dict[model_key]
                selected_mosaics = [cb.description.replace(" ⚠️", "").strip() for cb in mosaic_checkboxes if cb.value]
                log_message(f'Mosaicos selecionados para simulação: {selected_mosaics}')

                if not selected_mosaics:
                    log_message(f"Nenhum mosaico selecionado para o modelo: {model}")
                    continue

                model_obj = {
                    "model": model,
                    "mosaics": selected_mosaics,
                    "simulation": True
                }
                models_to_simulate.append(model_obj)
            else:
                log_message(f'Nenhum mosaico encontrado para o modelo: {model_key}')

        if models_to_simulate:
            log_message(f'Chamando render_classify_models para simulação com: {models_to_simulate}')
            render_classify_models(models_to_simulate)
        else:
            log_message("Nenhum mosaico foi selecionado para nenhum modelo.")
    else:
        log_message("Nenhum modelo selecionado.")


def classify_burned_area_click(b):
    selected_models = collect_selected_models()
    models_to_classify = []

    log_message('selected_models')
    log_message(selected_models)
    log_message('mosaic_checkboxes_dict')
    log_message(mosaic_checkboxes_dict)

    if selected_models:
        for model in selected_models:
            log_message(f'Processing model: {model}')
            model_key = f"{model}.meta" if not model.endswith('.meta') else model
            log_message(f'Checking for model_key: {model_key}')

            parts = model.split('_')
            if len(parts) < 4:
                log_message(f"[ERROR] Unexpected model name format: {model}")
                continue

            model_country = parts[1]
            version = parts[2]
            region = parts[3].split('.')[0]

            log_message(f'Extracted country: {model_country}, version: {version}, region: {region}')

            if model_key in mosaic_checkboxes_dict:
                log_message(f'Found mosaic checkboxes for model: {model_key}')
                mosaic_checkboxes = mosaic_checkboxes_dict[model_key]
                log_message('mosaic_checkboxes')
                log_message(mosaic_checkboxes)

                selected_mosaics = [cb.description.replace(" ⚠️", "").strip() for cb in mosaic_checkboxes if cb.value]
                log_message(f"Selected mosaics for model {model_key}: {selected_mosaics}")

                if not selected_mosaics:
                    log_message(f"No mosaics selected for model: {model_key}")
                    continue

                model_obj = {
                    "model": model,
                    "mosaics": selected_mosaics,
                    "simulation": False,
                    "country": model_country,
                    "version": version,
                    "region": region
                }
                models_to_classify.append(model_obj)
            else:
                log_message(f"No mosaics found for model: {model_key}")

        if models_to_classify:
            log_message(f"Calling render_classify_models with: {models_to_classify}")
            render_classify_models(models_to_classify)
        else:
            log_message("No mosaics were selected for any models.")
    else:
        log_message("No models selected.")


def on_select_country(country_name):
    global selected_country
    selected_country = country_name

    repo = ModelRepository(bucket_name=bucket_name, country=country_name)
    training_files, file_count = repo.list_models()

    if training_files:
        global checkboxes, mosaic_panels
        checkboxes = []
        mosaic_panels = []

        for file in training_files:
            if file.endswith('.meta'):
                parts = file.split('_')
                if len(parts) < 4:
                    continue

                region = parts[3].split('.')[0]

                checkbox = widgets.Checkbox(
                    value=False,
                    description=file.split('.')[0],
                    layout=widgets.Layout(width='700px')
                )
                checkbox.observe(lambda change, f=file, reg=region: update_panels(change, f, reg), names='value')
                checkboxes.append(checkbox)

        update_interface()
    else:
        message = widgets.HTML(
            value=f"<b style='color: red;'>No files found in the 'models' folder (Total: {file_count}).</b>"
        )
        clear_output(wait=True)
        display(message)


def update_panels(change, file, region):
    global mosaic_panels, selected_country

    if change['new']:
        panel = display_selected_mosaics(file, selected_country, region)
        mosaic_panels.append((file, region, panel))
    else:
        if file in mosaic_checkboxes_dict:
            checkbox_list = mosaic_checkboxes_dict[file]
            mosaic_checkbox_states[file] = [cb.value for cb in checkbox_list]

        mosaic_panels = [p for p in mosaic_panels if p[0] != file or p[1] != region]

    update_interface()


# ====================================
# 🚀 INITIALIZATION
# ====================================

on_select_country(country)

def execute_burned_area_classification(mode=None):
    models_to_classify = []

    for model, mosaic_checkboxes in mosaic_checkboxes_dict.items():
        selected_mosaics = [
            cb.description.replace(" ⚠️", "").strip()
            for cb in mosaic_checkboxes
            if cb.value
        ]

        if not selected_mosaics:
            print(f"[INFO] No mosaics selected for model: {model}")
            continue

        model_obj = {
            "model": model,
            "mosaics": selected_mosaics,
            "simulation": False
        }
        models_to_classify.append(model_obj)

    simulate_test = mode == 'test'

    if models_to_classify:
        print(f"[INFO] Starting classification for selected models.")
        render_classify_models(models_to_classify, simulate_test=simulate_test)
    else:
        print("[INFO] No models or mosaics selected. Classification skipped.")

update_interface()
