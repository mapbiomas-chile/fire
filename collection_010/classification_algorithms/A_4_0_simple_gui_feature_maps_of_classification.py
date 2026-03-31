# A_4_0_simple_gui_feature_maps_of_classification.py
# last update: '2026/01/27'
# MapBiomas Fire Classification Algorithms Step A_4_0
# Simple graphic user interface for feature maps extraction

# ====================================
# 📦 IMPORT LIBRARIES
# ====================================

import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox
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

if 'collection_folder' not in globals():
    collection_folder = 'collection1'

if 'models_folder' not in globals():
    models_folder = f'models_{collection_name}'

if 'mosaics_folder' not in globals():
    mosaics_folder = 'mosaics_cog'

if 'ee_project' not in globals():
    ee_project = f'mapbiomas-{country}'

if 'BASE_DATASET_PATH' not in globals():
    raise RuntimeError("[ERROR] BASE_DATASET_PATH is not defined. Run A_0_1 first.")

if 'fs' not in globals():
    fs = gcsfs.GCSFileSystem(project=ee_project)

# Si no existe un logger real, usa fallback
if 'log_message' not in globals():
    def log_message(msg):
        print(f"[LOG] {msg}")

# ====================================
# 🧪 MOCK TEMPORAL
# ====================================

# Reemplazar por import real cuando A_4_1 esté listo
def render_embedding_models(models_to_process, simulate_test=False):
    print("[MOCK] Chamada para render_embedding_models. Processamento simulado.")
    for model in models_to_process:
        print(f"[MOCK] Processando: {model['model']} com camada {model['embedding_layer']}")

# ====================================
# 🧠 GLOBAL GUI STATE
# ====================================

EMB_selected_country = ''
EMB_checkboxes = []
EMB_mosaic_panels = []
EMB_mosaic_checkboxes_dict = {}
EMB_mosaic_checkbox_states = {}

EMB_selected_embedding_layer = 'h5'
EMB_embedding_layer_selector = None

# ====================================
# 🧠 CORE CLASS
# ====================================

class ModelRepository:
    """Gerencia a listagem de modelos, mosaicos e embeddings existentes no GCS."""
    def __init__(self, bucket_name, country):
        self.bucket = bucket_name
        self.country = country
        self.base_folder = BASE_DATASET_PATH
        self.fs = fs

    def list_models(self):
        training_folder = f"{self.base_folder}/{models_folder}/"
        try:
            files = self.fs.ls(training_folder)
            return [file.split('/')[-1].replace('.meta', '') for file in files if file.endswith('.meta')], len(files)
        except Exception as e:
            log_message(f"[WARNING] Could not list models: {str(e)}")
            return [], 0

    def list_mosaics(self, region):
        mosaics_path = f"{self.base_folder}/{mosaics_folder}/"
        region_filter = f"_{region}_"
        try:
            files = self.fs.ls(mosaics_path)
            return [file.split('/')[-1] for file in files if region_filter in file], len(files)
        except Exception as e:
            log_message(f"[WARNING] Could not list mosaics: {str(e)}")
            return [], 0

    def list_embeddings(self):
        embeddings_folder = f"{self.base_folder}/result_embeddings/"
        try:
            files = self.fs.ls(embeddings_folder)
            return [file.split('/')[-1] for file in files if file.endswith('.tif')], len(files)
        except Exception:
            return [], 0

    def is_embedding_generated(self, mosaic_file, embedding_layer):
        embeddings_files, _ = self.list_embeddings()

        mosaic_base = mosaic_file.replace('.tif', '').replace('_cog', '')
        layer_tag = embedding_layer.lower()

        for emb in embeddings_files:
            if mosaic_base in emb and layer_tag in emb:
                return True

        return False

# ====================================
# 🧰 SUPPORT FUNCTIONS
# ====================================

def create_layer_selector_panel():
    global EMB_selected_embedding_layer, EMB_embedding_layer_selector

    layer_options = {
        'L1 (7 Bandas) - Features de Baixo Nível (Espectral)': 'h1',
        'L2 (14 Bandas) - Features Intermediárias (Contexto Temporal Simples)': 'h2',
        'L3 (7 Bandas) - Features Consolidadas (Síntese Compacta)': 'h3',
        'L4 (14 Bandas) - Features de Alto Nível (Padrões Temporais Complexos)': 'h4',
        'L5 (7 Bandas - Padrão) - Representação Latente Final (Discriminação)': 'h5'
    }

    if EMB_embedding_layer_selector is None:
        EMB_embedding_layer_selector = widgets.RadioButtons(
            options=layer_options,
            value=EMB_selected_embedding_layer,
            description='Camada p/ Embedding:',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )

        def update_selected_layer(change):
            global EMB_selected_embedding_layer
            EMB_selected_embedding_layer = change['new']

        EMB_embedding_layer_selector.observe(update_selected_layer, names='value')

    EMB_selected_embedding_layer = EMB_embedding_layer_selector.value

    return widgets.VBox(
        [
            widgets.HTML("<b>Escolha a camada para extração de embedding:</b>"),
            EMB_embedding_layer_selector
        ],
        layout=widgets.Layout(border='1px solid blue', padding='10px', margin='10px 0')
    )

def display_selected_mosaics_embedding(model, selected_country, region):
    repo = ModelRepository(bucket_name=bucket_name, country=selected_country)
    mosaic_files, mosaic_count = repo.list_mosaics(region)

    mosaics_panel = widgets.Output(
        layout={'border': '1px solid black', 'height': '200px', 'overflow_y': 'scroll'}
    )
    checkboxes_mosaics = []
    saved_states = EMB_mosaic_checkbox_states.get(model, None)

    with mosaics_panel:
        if mosaic_files:
            for idx, file in enumerate(mosaic_files):
                embedding_generated = repo.is_embedding_generated(file, EMB_selected_embedding_layer)

                checkbox_mosaic = widgets.Checkbox(
                    value=False,
                    description=file + (" 🌟 (Embedding OK)" if embedding_generated else "")
                )

                if saved_states and idx < len(saved_states):
                    checkbox_mosaic.value = saved_states[idx]

                checkboxes_mosaics.append(checkbox_mosaic)
                display(checkbox_mosaic)
        else:
            log_message(f"No mosaics found for region {region}")

    EMB_mosaic_checkboxes_dict[model] = checkboxes_mosaics

    def toggle_select_all(change):
        for checkbox in EMB_mosaic_checkboxes_dict.get(model, []):
            checkbox.value = change['new']

    select_all_checkbox = widgets.Checkbox(value=False, description="Select All")
    select_all_checkbox.observe(toggle_select_all, names='value')

    legend_panel = widgets.Output(
        layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'}
    )
    with legend_panel:
        print("🌟 Embeddings já gerados para este mosaico/camada. Eles serão sobrescritos se selecionados.")

    return widgets.VBox([select_all_checkbox, mosaics_panel, legend_panel])

def update_interface():
    global EMB_checkboxes, EMB_mosaic_panels

    clear_output(wait=True)

    layer_panel = create_layer_selector_panel()
    display(layer_panel)

    display(
        VBox(
            EMB_checkboxes,
            layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0', width='700px')
        )
    )

    EMB_mosaic_panels_widgets = [panel[2] for panel in EMB_mosaic_panels]
    display(
        HBox(
            EMB_mosaic_panels_widgets,
            layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')
        )
    )

    display(widgets.HTML(
        "<b>Ação:</b> Após selecionar os modelos e mosaicos, execute a célula separada com "
        "<code>execute_embedding_generation_click(None)</code>"
    ))

def collect_selected_models():
    return [checkbox.description for checkbox in EMB_checkboxes if checkbox.value]

def execute_embedding_generation_click(b):
    global EMB_selected_embedding_layer

    selected_models = collect_selected_models()
    models_to_process = []

    current_layer_choice = EMB_selected_embedding_layer
    if not current_layer_choice:
        log_message("Por favor, selecione a camada de embedding para extração.")
        return

    if selected_models:
        for model in selected_models:
            if model in EMB_mosaic_checkboxes_dict:
                mosaic_checkboxes = EMB_mosaic_checkboxes_dict[model]
                selected_mosaics = [
                    cb.description.replace(" 🌟 (Embedding OK)", "").strip()
                    for cb in mosaic_checkboxes if cb.value
                ]

                if not selected_mosaics:
                    log_message(f"Nenhum mosaico selecionado para o modelo: {model}")
                    continue

                model_obj = {
                    "model": model,
                    "mosaics": selected_mosaics,
                    "simulation": False,
                    "embedding_layer": current_layer_choice
                }
                models_to_process.append(model_obj)
            else:
                log_message(f"Nenhum mosaico encontrado para o modelo: {model}")

        if models_to_process:
            log_message(f"[INFO] Chamando o extrator de embeddings para: {models_to_process}")
            render_embedding_models(models_to_process, simulate_test=False)
        else:
            log_message("Nenhum mosaico foi selecionado para nenhum modelo.")
    else:
        log_message("Nenhum modelo selecionado.")

def on_select_country(country_name):
    global EMB_selected_country, EMB_checkboxes, EMB_mosaic_panels

    EMB_selected_country = country_name

    repo = ModelRepository(bucket_name=bucket_name, country=EMB_selected_country)
    training_files, file_count = repo.list_models()

    if training_files:
        EMB_checkboxes = []
        EMB_mosaic_panels = []

        for file in training_files:
            try:
                parts = file.split('_')
                region_part = parts[3]

                checkbox = widgets.Checkbox(
                    value=False,
                    description=file,
                    layout=widgets.Layout(width='700px')
                )

                checkbox.observe(
                    lambda change, f=file, reg=region_part: update_panels(change, f, reg),
                    names='value'
                )
                EMB_checkboxes.append(checkbox)

            except Exception:
                log_message(f"[WARNING] Arquivo de modelo com nome inesperado: {file}")

        update_interface()
    else:
        log_message("Nenhum arquivo de modelo encontrado.")
        clear_output(wait=True)
        layer_panel = create_layer_selector_panel()
        display(layer_panel)
        display(widgets.HTML("<b style='color: red;'>Nenhum modelo encontrado para este país.</b>"))

def update_panels(change, file, region):
    global EMB_mosaic_panels, EMB_selected_country, EMB_mosaic_checkboxes_dict, EMB_mosaic_checkbox_states

    if change['new']:
        panel = display_selected_mosaics_embedding(file, EMB_selected_country, region)
        EMB_mosaic_panels.append((file, region, panel))
    else:
        if file in EMB_mosaic_checkboxes_dict:
            checkbox_list = EMB_mosaic_checkboxes_dict[file]
            EMB_mosaic_checkbox_states[file] = [cb.value for cb in checkbox_list]

        EMB_mosaic_panels = [p for p in EMB_mosaic_panels if p[0] != file or p[1] != region]

    update_interface()

# ====================================
# 🚀 INITIALIZATION
# ====================================

on_select_country(country)
