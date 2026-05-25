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

        self.selected_version = None
        self.selected_region = None

        self.render_interface()

    def list_training_samples_folder(self):
        """
        List files in 'training_samples' folder for the selected country.
        """
        path = f"{BASE_DATASET_PATH}/training_samples/"

        try:
            files = fs.ls(path)
            return sorted([
                file.split('/')[-1]
                for file in files
                if file.split('/')[-1].lower().endswith((".tif", ".tiff"))
            ])
        except FileNotFoundError:
            return []
        except Exception as e:
            self.log(f"[ERROR] Could not list training samples: {str(e)}")
            return []

    def get_active_checkbox(self):
        """
        Backward-compatible method.
        Returns a synthetic label using the manually selected version and region.
        This keeps A_2_1 working without major changes.
        """
        if self.selected_version is None or self.selected_region is None:
            return None

        return f"trainings_{self.selected_version}_{self.selected_region}"

    def list_existing_models(self):
        """
        Return a set of model checkpoint base names.
        """
        prefix_path = f"{BASE_DATASET_PATH}/{models_folder}/"

        try:
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

        if version == "":
            self.log("[ERROR] Version is empty. Example: v1")
            return

        if region == "":
            self.log("[ERROR] Region is empty. Example: r1")
            return

        self.selected_version = version
        self.selected_region = region

        self.log("========================================")
        self.log("[INFO] Manual sample selection mode")
        self.log(f"[INFO] Selected model version: {self.selected_version}")
        self.log(f"[INFO] Selected model region: {self.selected_region}")
        self.log(f"[INFO] Number of selected samples: {len(selected_files)}")
        self.log("========================================")

        for file in selected_files:
            self.log(f"[SAMPLE] {file}")

        self.preparation_function(selected_files)

    def create_scrollable_text_panel(self, title, items, border_color='black', height='150px'):
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

        return VBox([title_widget, output])

    def display_existing_models(self):
        """
        Display a scrollable list of existing models from the GCS bucket.
        """
        fs.invalidate_cache()

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
        Renders the full interface.
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
            display(widgets.HTML("<b style='color: red;'>No files found in training_samples.</b>"))
            return

        self.display_existing_models()

        self.version_widget = widgets.Text(
            value="v1",
            description="Version:",
            placeholder="Example: v1",
            layout=widgets.Layout(width="300px")
        )

        self.region_widget = widgets.Text(
            value="r1",
            description="Region:",
            placeholder="Example: r1",
            layout=widgets.Layout(width="300px")
        )

        self.sample_selector = widgets.SelectMultiple(
            options=self.training_files,
            description="Samples:",
            rows=18,
            layout=widgets.Layout(width="95%", height="380px")
        )

        selector_title = widgets.HTML(
            value="<b>Select the sample files to use for training:</b>",
            layout=widgets.Layout(margin='10px 0 5px 0')
        )

        display(selector_title)
        display(self.version_widget)
        display(self.region_widget)
        display(self.sample_selector)

        train_button = widgets.Button(
            description="Train selected samples",
            button_style='success',
            layout=widgets.Layout(width='240px')
        )

        train_button.on_click(self.train_models_click)

        display(
            HBox(
                [train_button],
                layout=widgets.Layout(
                    justify_content='flex-start',
                    margin='20px 0'
                )
            )
        )

        footer = widgets.HTML(
            "<b style='color: orange;'>⚠️ Existing models with the same version and region will be overwritten.</b>"
        )

        display(footer)
