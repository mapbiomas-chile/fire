# NLHPC leftraru — rama feature/model_modification

## Conectar la rama en el cluster

```bash
cd ~/fire
git fetch origin
git checkout feature/model_modification
git pull origin feature/model_modification
```

## Reglas de versión de modelo (producción actual)

| Región | Años 2013–2018 | Años 2019–2025 |
|--------|----------------|----------------|
| r2     | v1             | v1             |
| r1, r4, r6 | v1         | v2             |

`run_classify_region_slurm.sh` aplica estas reglas cuando `AUTO_MODEL_VERSION_BY_YEAR=1` (o `MODEL_VERSION` está vacío).

## Muestras de entrenamiento

Ruta en leftraru: `/home/flepin/samples_col1`

Verificar antes de entrenar:

```bash
cd /home/flepin/samples_col1
ls *_v1_*_r2_*.tif    # ejemplo r2 v1
python - <<'PY'
import rasterio
from pathlib import Path
files = sorted(Path("/home/flepin/samples_col1").glob("*.tif"))[:3]
for p in files:
    with rasterio.open(p) as src:
        print(p.name, "->", src.descriptions)
PY
```

Cada TIFF debe tener banda `landcover` y nombres tipo:

```text
samples_fire_v1_b14_chile_r1_chile_matorral_20130000000000-0000000000.tif
```

El **v1 en el nombre del archivo** no es el modelo final: define qué TIFFs leer (`SAMPLE_VERSION=v1`).  
El **v1/v2 del checkpoint** (`col1_chile_v1_r1_...` vs `col1_chile_v2_r1_...`) se elige por **rango de años** en el nombre del sample.

| Checkpoint | Región | Años en muestras |
|------------|--------|------------------|
| v1 | r1, r4, r6 | 2013–2018 |
| v2 | r1, r4, r6 | 2019–2025 |
| v1 | r2 | 2013–2018 |

Preview (sin entrenar):

```bash
cd ~/fire/classification
python preview_training_campaign.py /home/flepin/samples_col1
```

## Entrenamiento — campaña completa (7 modelos)

```bash
cd ~/fire
cp classification/cluster_paths.model_modification.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env

bash classification/run_train_chile_campaign.sh --dry-run
bash classification/run_train_chile_campaign.sh
```

Genera en `/home/flepin/models_col1_20260618/`:

```text
col1_chile_v1_r1_rnn_lstm_ckpt*
col1_chile_v2_r1_rnn_lstm_ckpt*
col1_chile_v1_r4_rnn_lstm_ckpt*
col1_chile_v2_r4_rnn_lstm_ckpt*
col1_chile_v1_r6_rnn_lstm_ckpt*
col1_chile_v2_r6_rnn_lstm_ckpt*
col1_chile_v1_r2_rnn_lstm_ckpt*
```

## Entrenamiento — un solo modelo

```bash
export TRAIN_REGION=r1
export TRAIN_VERSION=v2
export SAMPLE_START_YEAR=2019
export SAMPLE_END_YEAR=2025
sbatch classification/run_train_fire_model_slurm.sh
```

Seguimiento:

```bash
squeue -u flepin
tail -f ~/logs/train_fire_model_<JOBID>.out
```

Salida: `/home/flepin/models_col1_20260618/col1_chile_<version>_<region>_rnn_lstm_ckpt*`

El job usa partición **main** (CPU; el MLP no necesita GPU).

Benchmark XGBoost:

```bash
export TRAIN_BACKEND=xgboost
sbatch classification/run_train_fire_model_slurm.sh
```

## Clasificar con modelo nuevo (prueba A/B)

Después del entrenamiento, clasificar un año de prueba contra el modelo experimental:

```bash
export MODEL_DIR="/home/flepin/models_col1_20260618"
export MODEL_NAME="col1_chile_v1_r2_rnn_lstm_ckpt"
export OUTPUT_DIR="/home/flepin/classification_model_mod_test"
export REGION=r2
export START_YEAR=2019
export END_YEAR=2019
export AUTO_MODEL_VERSION_BY_YEAR=0
export MODEL_VERSION=v1

source classification/cluster_paths.env   # PYTHON, MOSAIC_DIR, etc.
sbatch classification/run_classify_region_slurm.sh
```

Comparar en QGIS:
- baseline: `/home/flepin/classification_20260616/`
- experimental: `/home/flepin/classification_model_mod_test/`

## Clasificación (modelos existentes en producción)

```bash
cp classification/cluster_paths.leftraru classification/cluster_paths.env
export REGION=r2
source classification/cluster_paths.env
sbatch classification/run_classify_region_slurm.sh
```

Salida por defecto: `/home/flepin/classification_20260616`  
Modelos: `/home/flepin/models_col1`

## Notas

- Los archivos `*.env.leftraru` se commitean en esta rama; `cluster_paths.env` local sigue en `.gitignore`.
- Checkpoints legacy siguen funcionando sin `DECISION_THRESHOLD`.
- Reentrenar es **opcional**; producción puede seguir con `models_col1`.

## Uso responsable del cómputo (NLHPC)

El walltime que pides en `sbatch` es el **máximo** que Slurm reserva; aunque el job termine antes, cuenta contra tu cuota.

| Tarea | Default en script | Regla práctica |
|-------|-------------------|----------------|
| Entrenamiento MLP/XGBoost | **30 min** | Suele bastar; si TIMEOUT: `sbatch -t 01:00:00 ...` |
| Clasificación por región | **1 h 30 min** | ~10–15 min × número de años |
| Un solo año | `run_classify_fire_model_slurm*.sh` ~30–40 min | Acotar `START_YEAR=END_YEAR` |

Recomendaciones:

1. **Acota años** en `cluster_paths.env` (`START_YEAR` / `END_YEAR`) en lugar de correr 2013–2025 de una vez si no hace falta.
2. **Pide solo el tiempo necesario:** `sbatch -t HH:MM:SS script.sh`
3. **Cancela jobs de prueba:** `scancel <JOBID>`
4. **Entrenamiento en `main`**, no en `v100` (este MLP no usa GPU).
5. Modelos experimentales en `models_col1_20260618`; no reentrenar si ya tienes checkpoint en `models_col1`.
