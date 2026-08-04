# Pipeline de clasificación

**Español** | [English](README.md)

Entrenamiento supervisado e inferencia sobre mosaicos para el producto de área quemada MapBiomas Chile. El procesamiento es **basado en disco** (muestras y mosaicos GeoTIFF); no se requiere Earth Engine en tiempo de entrenamiento ni de inferencia.

Documentación asociada: [raíz del repositorio](../README.es.md), [filtrado](../filtering/README.es.md), [vectorización](../vectorize/README.es.md).

---

## 1. Alcance

| Tarea | Puntos de entrada |
|-------|-------------------|
| Entrenar una región × versión | `train_fire_model.py`, `run_train_fire_model_slurm.sh` |
| Campaña completa Chile (7 modelos) | `run_train_chile_campaign.sh` → SLURM |
| Clasificar mosaicos | `classify_fire_model.py`, *scripts* SLURM por región / serie |
| Inventario de campaña | `preview_training_campaign.py` |

---

## 2. Arquitectura del modelo (MLP TensorFlow de producción)

Definida en `train_fire_model.py` / `fire_model_common.py` (`create_model_graph`).

| Componente | Especificación |
|------------|----------------|
| Tipo | *Multilayer perceptron* (MLP) totalmente conectado |
| Estandarización de entrada | Media y desviación estándar por característica del conjunto de **entrenamiento** |
| Capas ocultas | **5** densas ReLU: **7 → 14 → 7 → 14 → 7** neuronas |
| Salida | **2** logits (no quemado / quemado) + *softmax* |
| Inicialización de pesos | Normal truncada, \(\sigma = 1/\sqrt{n_{\mathrm{in}}}\) |
| Optimizador | **Adam**, tasa de aprendizaje **0,001** |
| Tamaño de lote | **1000** |
| Iteraciones | **7000** |
| Función de pérdida | Entropía cruzada *softmax* **ponderada** (pesos inversos a la frecuencia de clase) |
| Partición validación | **70 % / 30 %** de **escenas (archivos)** (`--validation-split by_file`) |
| Semilla aleatoria | **42** |
| Límites de muestreo | ≤ **2×10⁶** píxeles train, ≤ **5×10⁵** val (submuestreo aleatorio) |
| Umbral de decisión | Producción fijo **0,55** (\(P(\mathrm{quemado}) \ge 0{,}55\)); en `*_hyperparameters.json` |
| Ventana espacial local | Producción **0** (desactivada) |
| *Oversampling* de quemado | Producción **desactivado** |
| Métrica de selección | **IoU** en validación (si el umbral no es fijo) |

Existe una línea base XGBoost (`train_fire_model_xgboost.py`); la producción Chile usa la MLP TensorFlow.

---

## 3. Organización del software

### Módulos Python

| Archivo | Función |
|---------|---------|
| `fire_model_common.py` | Esquema de datos, estandarización, grafo, métricas, *features* espaciales |
| `train_fire_model.py` | Entrenamiento TensorFlow |
| `train_fire_model_xgboost.py` | Línea base XGBoost |
| `classify_fire_model.py` | Inferencia + *opening* / *closing* morfológico |
| `preview_training_campaign.py` | Cobertura de muestras por modelo |

### Orquestación

| *Script* | Uso |
|----------|-----|
| `run_train_chile_campaign_slurm.sh` | Los 7 modelos (un *job* SLURM) |
| `run_train_fire_model_slurm.sh` | Un solo modelo |
| `run_classify_chile_slurm.sh` | Serie completa |
| `run_classify_region_slurm.sh` | Una región / ventana de años |
| `run_classify_single_mosaic_slurm.sh` | Un mosaico (depuración / figuras) |
| `train_fire_model_once.sh` | Cuerpo de entrenamiento compartido para SLURM |
| `cluster_paths.20260619.env.leftraru` | Plantilla de rutas de producción |
| `cluster_paths.train.env.leftraru` | Atajo de entrenamiento |
| `cluster_paths.classify.env.leftraru` | Atajo de clasificación |
| `cluster_paths.env.example` | Plantilla genérica |

Copie una plantilla a **`classification/cluster_paths.env`** (ignorado por git) antes de los *jobs*.

---

## 4. Correspondencia región × periodo (Chile)

Nombre de *checkpoint*: `col1_chile_<version>_<region>_rnn_lstm_ckpt`.

| Región | Años 2013–2018 | Años 2019–2025 |
|--------|----------------|----------------|
| r2 | v1 | v1 |
| r1, r4, r6 | v1 | v2 |

Plan de la campaña de entrenamiento (`run_train_chile_campaign.sh`):

| Región | Versión | Años de muestra |
|--------|---------|-----------------|
| r1 | v1 | 2013–2018 |
| r1 | v2 | 2019–2025 |
| r2 | v1 | 2013–2018 |
| r4 | v1 | 2013–2018 |
| r4 | v2 | 2019–2025 |
| r6 | v1 | 2013–2018 |
| r6 | v2 | 2019–2025 |

---

## 5. Procedimiento de entrenamiento (replicación)

### 5.1 Entradas

- Directorio de GeoTIFF de muestra (bandas predictoras + banda de etiqueta con descripción **`landcover`**).
- Año y región codificados en el nombre (filtrado por `--sample-start-year` / `--sample-end-year`).

### 5.2 Retención espacial (*holdout*)

Las escenas (archivos) se reparten en train y validación con semilla **42** y fracción **0,7**. Todos los píxeles de un archivo pertenecen a un solo subconjunto, lo que reduce la fuga de información entre píxeles vecinos de la misma imagen.

### 5.3 NLHPC — campaña completa

```bash
cd ~/fire && git checkout main && git pull
cp classification/cluster_paths.train.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env

bash classification/run_train_chile_campaign.sh --dry-run   # inventario
bash classification/run_train_chile_campaign.sh              # sbatch
# tail -f ~/logs/train_chile_campaign_<JOBID>.out
```

Salidas típicas:

```text
$MODELS_DIR/col1_chile_<v>_<region>_rnn_lstm_ckpt*
$MODELS_DIR/col1_chile_<v>_<region>_rnn_lstm_ckpt_hyperparameters.json
```

Directorio de producción en leftraru: `/home/flepin/models_col1_20260619`.

### 5.4 Ejemplo local de un modelo

```bash
python classification/train_fire_model.py \
  --country chile --version v1 --region r2 \
  --training-samples-dir /path/to/samples_col1 \
  --models-dir /path/to/models \
  --validation-split by_file \
  --loss weighted \
  --batch-size 1000 --n-iter 7000 --learning-rate 0.001 \
  --seed 42 \
  --max-training-pixels 2000000 --max-validation-pixels 500000 \
  --metric iou \
  --fixed-decision-threshold 0.55
```

---

## 6. Procedimiento de inferencia (replicación)

### 6.1 Algoritmo

1. Construir vectores de características por píxel según `DATASET_SCHEMA`.
2. *Softmax* → probabilidad de clase quemada \(P(\mathrm{quemado})\).
3. Umbral con `DECISION_THRESHOLD` (o `--decision-threshold`).
4. *Opening* morfológico (por defecto tamaño 2) y *closing* (por defecto 4).
5. Escribir GeoTIFF `uint8` `<mosaic_stem>_classified.tif`.

### 6.2 Serie completa Chile (SLURM)

```bash
cp classification/cluster_paths.classify.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_chile_slurm.sh
# ls $OUTPUT_DIR/*_classified.tif | wc -l   # ~52
```

Salida de producción: `/home/flepin/classification_20260619`.

### 6.3 Un solo mosaico

```bash
export MODEL_DIR=~/models_col1_20260619
export MOSAIC_DIR=~/mosaics_cog
export OUTPUT_DIR=~/classification_output
sbatch --export=ALL classification/run_classify_single_mosaic_slurm.sh \
  col1_chile_v1_r4_rnn_lstm_ckpt \
  b14_chile_r4_2017_cog.tif
```

---

## 7. Enlace al postproceso

Los clasificados alimentan `filtering/` (`CLASSIFIED_DIR`). Env de filtro de producción:  
`filtering/cluster_paths.20260619.env.leftraru`.  
Vectorización: `vectorize/cluster_paths.20260619.env.leftraru`.

---

## 8. Convenciones de nombres

| Artefacto | Patrón |
|-----------|--------|
| Base del *checkpoint* TF | `col1_<country>_<version>_<region>_rnn_lstm_ckpt` |
| Hiperparámetros | `<checkpoint>_hyperparameters.json` |
| Clasificado | `<mosaic_stem>_classified.tif` |
| Ejemplo de mosaico | `b14_chile_r4_2017_cog.tif` |

---

## 9. Configuraciones experimentales

Los env “conservadores” A/B previos a la promoción **20260619** están en [`../experiments/`](../experiments/README.es.md). No son necesarios para replicar producción si los modelos ya existen en `models_col1_20260619`.
