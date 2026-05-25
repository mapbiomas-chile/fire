# last_update: '2026/01/26', github:'mapbiomas/chile-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_2_0 - Simple Graphic User Interface for Training Models
# Modified: manual sample selection + manual output model name

# ====================================
# 📦 IMPORT LIBRARIES
# ====================================

import os
import re
import time
import gcsfs
import ipywidgets as widgets
import sys
from IPython.display import display, clear_output

# TensorFlow in compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ====================================
# 🌍 GLOBAL SETTINGS AND FILESYSTEM
# ====================================

if 'bucket_name' not in globals():
    bucket_name = 'mapbiomas-fire'

if 'ee_project' not in globals():
    ee_project = 'mapbiomas-chile'

if 'collection_name' not in globals():
    collection_name = 'col1'

if 'models_folder' not in globals():
    models_folder = f'models_{collection_name}'

if 'base_subfolder' not in globals():
    base_subfolder = 'b24'

if 'fs' not in globals():
    fs = gcsfs.GCSFileSystem(project=ee_project)


# ====================================
# 🎛️ INTERFACE CLASS
# ====================================

class TrainingInterface:
    """
    Interface for manually selecting training sample files and triggering model training.
    """

    def __init__(self, country, preparation_function, log_func):
        self.country = country
        self.preparation_function = preparation_function
        self.log = log_func

        self.training_files = []
        self.sample_selector = None

        self.version_widget = None
        self.region_widget = None
        self.output_name_widget = None

        self.selected_version = None
        self.selected_region = None
        self.selected_output_model_name = None

        self.render_interface()

    def list_training_samples_folder(self):
        """
        List files in 'training_samples' folder.
        """
        path = f"{BASE_DATASET_PATH}/training_samples/"

        try:
            fs.invalidate_cache()
            files = fs.ls(path)

            sample_files = [
                file.split('/')[-1]
                for file in files
                if file.split('/')[-1].lower().endswith((".tif", ".tiff"))
            ]

            return sorted(sample_files)

        except FileNotFoundError:
            self.log(f"[ERROR] Folder not found: gs://{path}")
            return []

        except Exception as e:
            self.log(f"[ERROR] Could not list training samples from gs://{path}: {str(e)}")
            return []

    def get_active_checkbox(self):
        """
        Backward-compatible method used by A_2_1.

        A_2_1 may still expect something like:
        trainings_v1_r1

        Here we generate that label from the manually defined version and region.
        """
        if self.selected_version is None or self.selected_region is None:
            return None

        return f"trainings_{self.selected_version}_{self.selected_region}"

    def get_model_metadata(self):
        """
        Returns manually defined model metadata for A_2_1.

        This allows A_2_1 to use the exact output model name selected by the user.
        """

        if self.selected_version is None or self.selected_region is None:
            return None

        if self.selected_output_model_name is None or self.selected_output_model_name == "":
            output_model_name = (
                f"{collection_name}_{self.country}_"
                f"{self.selected_version}_{self.selected_region}_rnn_lstm_ckpt"
            )
        else:
            output_model_name = self.selected_output_model_name

        output_model_name = output_model_name.replace(" ", "_")

        return {
            "version": self.selected_version,
            "region": self.selected_region,
            "output_model_name": output_model_name
        }

    def list_existing_models(self):
        """
        Return a set of model checkpoint base names.
        """
        prefix_path = f"{BASE_DATASET_PATH}/{models_folder}/"

        try:
            fs.invalidate_cache()
            files = fs.ls(prefix_path)

            model_files = [
                os.path.basename(f).split('.')[0]
                for f in files
                if 'ckpt' in f and 'hyperparameters' not in f
            ]

            return set(model_files)

        except Exception as e:
            self.log(f"[WARNING] Could not list existing models: {str(e)}")
            return set()

    def build_default_output_model_name(self, version, region):
        """
        Build default output model name using collection, country, version and region.
        """
        version = version.strip().replace(" ", "_")
        region = region.strip().replace(" ", "_")

        if version == "" or region == "":
            return ""

        return f"{collection_name}_{self.country}_{version}_{region}_rnn_lstm_ckpt"

    def update_output_placeholder(self, change=None):
        """
        Updates placeholder/help value when version or region changes.
        Does not overwrite user-entered output name.
        """
        if self.output_name_widget is None:
            return

        version = self.version_widget.value.strip() if self.version_widget is not None else ""
        region = self.region_widget.value.strip() if self.region_widget is not None else ""

        default_name = self.build_default_output_model_name(version, region)

        self.output_name_widget.placeholder = (
            f"Optional. Default: {default_name}"
            if default_name != ""
            else "Optional. Example: col1_chile_v1_r6_rnn_lstm_ckpt"
        )

    def train_models_click(self, b):
        """
        Train using only the manually selected sample files.
        """
        selected_files = list(self.sample_selector.value)

        if len(selected_files) == 0:
            self.log("[ERROR] No sample files selected.")
            return

        version = self.version_widget.value.strip()
        region = self.region_widget.value.strip()
        output_model_name = self.output_name_widget.value.strip()

        if version == "":
            self.log("[ERROR] Version is empty. Example: v1")
            return

        if region == "":
            self.log("[ERROR] Region is empty. Example: r1")
            return

        # Clean names to avoid spaces in model outputs
        version = version.replace(" ", "_")
        region = region.replace(" ", "_")
        output_model_name = output_model_name.replace(" ", "_")

        if output_model_name == "":
            output_model_name = self.build_default_output_model_name(version, region)

        # Optional safety check
        if not output_model_name.endswith("_rnn_lstm_ckpt"):
            self.log("[WARNING] Output model name does not end with '_rnn_lstm_ckpt'.")
            self.log("[WARNING] The model will still be saved using the name you provided.")

        self.selected_version = version
        self.selected_region = region
        self.selected_output_model_name = output_model_name

        self.log("========================================")
        self.log("[INFO] Manual sample selection mode")
        self.log(f"[INFO] Selected model version: {self.selected_version}")
        self.log(f"[INFO] Selected model region: {self.selected_region}")
        self.log(f"[INFO] Output model name: {self.selected_output_model_name}")
        self.log(f"[INFO] Number of selected samples: {len(selected_files)}")
        self.log("========================================")

        for file in selected_files:
            self.log(f"[SAMPLE] {file}")

        self.preparation_function(selected_files)

    def create_scrollable_text_panel(self, title, items, border_color='black', height='150px'):
        """
        Create a scrollable text panel.
        """
        title_widget = widgets.HTML(value=f"<b>{title}</b>")

        output = widgets.Output(
            layout=widgets.Layout(
                border=f'1px solid {border_color}',
                height=height,
                overflow_y='auto',
                margin='5px 0 10px 0',
                padding='6px'
            )
        )

        with output:
            if items:
                for item in items:
                    print(f" - {item}")
            else:
                print("No items found.")

        return widgets.VBox([title_widget, output])

    def display_existing_models(self):
        """
        Display a scrollable list of existing models from the GCS bucket.
        """
        existing = sorted(self.list_existing_models())

        panel = self.create_scrollable_text_panel(
            title=f"Existing trained models ({len(existing)}):",
            items=existing,
            border_color='green',
            height='150px'
        )

        display(panel)

    def render_interface(self):
        """
        Render the full manual selection interface.
        """
        clear_output(wait=True)

        self.training_files = self.list_training_samples_folder()
        num_files = len(self.training_files)

        header = widgets.HTML(
            value=(
                f"<b>Selected country:</b> {self.country} ({num_files} files found)"
                f"<br><b>Base subfolder:</b> <code>{base_subfolder or '(root)'}</code>"
                f"<br><b>Mode:</b> Manual sample selection"
            ),
            layout=widgets.Layout(margin='0 0 10px 0')
        )

        display(header)

        files_panel = self.create_scrollable_text_panel(
            title="Available sample files:",
            items=self.training_files,
            border_color='black',
            height='180px'
        )

        display(files_panel)

        if num_files == 0:
            display(
                widgets.HTML(
                    "<b style='color: red;'>No files found in training_samples.</b>"
                )
            )
            return

        self.display_existing_models()

        self.version_widget = widgets.Text(
            value="v1",
            description="Version:",
            placeholder="Example: v1, v2, v3",
            layout=widgets.Layout(width="420px")
        )

        self.region_widget = widgets.Text(
            value="r1",
            description="Region:",
            placeholder="Example: r1, r4, r6, r1_mix",
            layout=widgets.Layout(width="420px")
        )

        self.output_name_widget = widgets.Text(
            value="",
            description="Output:",
            placeholder="Optional. Example: col1_chile_v1_r6_rnn_lstm_ckpt",
            layout=widgets.Layout(width="760px")
        )

        self.version_widget.observe(self.update_output_placeholder, names="value")
        self.region_widget.observe(self.update_output_placeholder, names="value")
        self.update_output_placeholder()

        self.sample_selector = widgets.SelectMultiple(
            options=self.training_files,
            description="Samples:",
            rows=20,
            layout=widgets.Layout(width="95%", height="420px")
        )

        selector_title = widgets.HTML(
            value="<b>Select the sample files to use for training:</b>",
            layout=widgets.Layout(margin='10px 0 5px 0')
        )

        help_text = widgets.HTML(
            value=(
                "<p style='color: gray;'>"
                "Use Ctrl + click to select multiple samples. "
                "Only the selected files will be used for training. "
                "The version and region fields define the default model name. "
                "If Output is empty, the model name will be created automatically."
                "</p>"
            )
        )

        output_help = widgets.HTML(
            value=(
                "<p style='color: gray;'>"
                "Default output format: "
                "<code>col1_chile_VERSION_REGION_rnn_lstm_ckpt</code>. "
                "Example: <code>col1_chile_v1_r6_rnn_lstm_ckpt</code>."
                "</p>"
            )
        )

        display(selector_title)
        display(help_text)
        display(self.version_widget)
        display(self.region_widget)
        display(self.output_name_widget)
        display(output_help)
        display(self.sample_selector)

        train_button = widgets.Button(
            description="Train selected samples",
            button_style='success',
            layout=widgets.Layout(width='240px')
        )

        train_button.on_click(self.train_models_click)

        display(
            widgets.HBox(
                [train_button],
                layout=widgets.Layout(
                    justify_content='flex-start',
                    margin='20px 0'
                )
            )
        )

        footer = widgets.HTML(
            value=(
                "<b style='color: orange;'>"
                "⚠️ Existing models with the same output name will be overwritten."
                "</b>"
            )
        )

        display(footer)
