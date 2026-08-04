# Gestión de datos de campaña y trazabilidad

**Español** | [English](CAMPAIGN_TRACEABILITY.md)

Guía operativa para **archivar productos**, **convención de nombres** y **citar la configuración de software** tras una corrida de MapBiomas Fire Chile. Complementa los README de módulo con material apto para informes técnicos y revisión por pares.

**Identificador de campaña de producción de referencia:** `20260619` (receta MLP conservadora). Sustituya los nombres de carpeta al promover una nueva campaña.

---

## 1. Lista de control de artefactos de campaña

Conserve lo siguiente en cada campaña que deba reejecutar, auditar o publicar. Las rutas ilustran el layout `20260619` en NLHPC; adapte usuario y sello de campaña.

### 1.1 Procedencia del software (siempre)

| Elemento | Qué guardar | Motivo |
|----------|-------------|--------|
| Commit de Git | SHA de `git rev-parse HEAD` (40 caracteres preferible) | Código exacto del *pipeline* |
| Rama / etiqueta | p. ej. `main`, etiqueta opcional `campaign-20260619` | Interpretar el historial |
| Envs activos | Instantánea de cada `cluster_paths.env` usado (véase §3) | Rutas absolutas y *flags* |
| Conda / Python | Nombre del env + `python -V`; opcional `conda list --export` | Stack de librerías |
| Fecha de corrida | Fecha ISO de train / classify / filter / vectorize | Procedencia temporal |
| Operador | Usuario o rol | Contactabilidad |

Carpeta de apoyo sugerida (**no** requiere vivir en el repositorio git):

```text
~/campaigns/chile_fire_20260619/
├── PROVENANCE.md                 # plantilla §3 completada
├── git_sha.txt
├── envs/
│   ├── classification.cluster_paths.env
│   ├── filtering.cluster_paths.env
│   └── vectorize.cluster_paths.env
├── logs/                         # copias opcionales de .out/.err de SLURM
└── manifests/                    # listados opcionales de archivos
```

### 1.2 Modelos (entrenamiento)

| Artefacto | Patrón / ruta (ejemplo) | ¿Archivar? |
|-----------|-------------------------|------------|
| *Checkpoints* TensorFlow | `~/models_col1_20260619/col1_chile_<v>_<region>_rnn_lstm_ckpt*` | **Sí** (juego completo) |
| JSON de hiperparámetros | `..._rnn_lstm_ckpt_hyperparameters.json` | **Sí** (uno por modelo) |
| Logs de entrenamiento | `~/logs/train_chile_campaign_*.out` | Recomendado |
| Inventario de muestras | salida de `preview_training_campaign.py` o lista usada | Recomendado |

La producción espera **siete** modelos de Chile. Compruebe:

```bash
ls ~/models_col1_20260619/*_hyperparameters.json | wc -l   # esperado: 7
```

### 1.3 Clasificación (inferencia)

| Artefacto | Patrón / ruta (ejemplo) | ¿Archivar? |
|-----------|-------------------------|------------|
| Teselas clasificadas | `~/classification_20260619/b14_chile_r*_????_*_classified.tif` | **Sí** |
| Control de conteo | ~**52** teselas (4 regiones × 13 años 2013–2025) | Verificación |
| Logs de clasificación | `~/logs/classi_chile_*.out` | Recomendado |

### 1.4 Filtrado

| Artefacto | Patrón / ruta (ejemplo) | ¿Archivar? |
|-----------|-------------------------|------------|
| Máscaras LULC | `$WORK_ROOT/mascaras/{acumuladas,by_year,totales}/` | **Sí** (reconstruibles pero costosas) |
| Rasters filtrados | `$WORK_ROOT/classified_filtered/*.tif` | **Sí** (producto raster canónico pre-vector) |
| JSON de estadísticas | `$WORK_ROOT/logs/filter_stats.json`, `fill_stats.json` | **Sí** |
| Intermedios temporal/relleno | solo si `KEEP_*_INTERMEDIATE=1` | Opcional |

Con `WORK_ROOT=~/classification_20260619/filtering_work`:

```bash
ls $WORK_ROOT/classified_filtered/*.tif | wc -l   # esperado ~52
ls $WORK_ROOT/mascaras/totales/mascara_total_*.tif | wc -l
```

Documente también la **identidad de `LULC_STACK`**, `START_YEAR_BAND1`, ventanas de estabilidad, `MAX_HOLE_AREA`, `FILL_AGRICULTURAL_HOLES`, `SKIP_MIN_PATCH_SIEVE`.

### 1.5 Vectorización y reglas de área

| Artefacto | Patrón / ruta (ejemplo) | ¿Archivar? |
|-----------|-------------------------|------------|
| Polígonos por tesela | `$WORK_ROOT/polygons/*.gpkg` | **Sí** si se usan en control de calidad |
| Etapa de área mínima | `polygons_min20ha/`, histogramas, `thresholds_area_min20ha/` | **Sí** si se publican |
| Filtrado por regla | `polygons_filtered_min20ha_<regla>.gpkg` (p. ej. `p25`) | **Sí** para el mapa |
| Mosaicos nacionales | `national_vector/mosaics_by_year/` | Opcional |
| Eventos nacionales | `national_vector/polygons_chile/chile_<año>_events.gpkg` | **Sí** para producto nacional |
| Estadísticas de vectorización | `$WORK_ROOT/logs/vectorize_stats.json` | Recomendado |

Registre **mínimo de píxeles del *sieve*** (a menudo 112), **distancia de agrupación** (a menudo 200 m), **ha de prefiltro** (a menudo 20), **regla de umbral** (`p25`, …).

### 1.6 Entradas externas al repositorio

| Entrada | Concepto de ubicación | Anotar en PROVENANCE |
|---------|------------------------|----------------------|
| Muestras de entrenamiento | p. ej. `~/samples_col1` | Conteo de TIFF; filtros de año |
| Mosaicos de predictores | p. ej. `~/mosaics_cog` | Nombres `b14_chile_rX_YYYY_cog.tif` |
| Stack LULC | GeoTIFF multibanda MapBiomas | Año de la banda 1; ruta |
| Cicatrices de referencia | GPKG/SHP de validación | CRS; campo de año |

### 1.7 Conjunto mínimo para “¿podemos reconstruir el mapa?”

1. SHA de Git + tres instantáneas de `cluster_paths.env`.  
2. Siete *checkpoints* + siete `*_hyperparameters.json`.  
3. `classified_filtered/` **o** clasificados + capacidad de refiltrar con el env archivado.  
4. Producto(s) vectorial(es) publicados para figuras/tablas.  
5. Parámetros del §3.2 de la plantilla de procedencia.

---

## 2. Convención de nombres de productos

### 2.1 Tokens

| Token | Significado | Ejemplos |
|-------|-------------|----------|
| `b14` | Id de colección de satélite / predictores en el nombre del mosaico | fijo en el naming Chile |
| `chile` | País | — |
| `r1`, `r2`, `r4`, `r6` | Regiones de procesamiento | r5 excluida del flujo de mosaicos |
| `YYYY` | Año calendario en el *stem* de mosaicos / clasificados / filtrados | `2017` |
| `v1`, `v2` | Versión de **periodo de modelo** | v1=2013–2018 en r1/r4/r6; v2=2019–2025 |
| `col1` | Token de colección en nombres de modelo | Collection 1 |

Los años en los clasificados se refieren al **año del mosaico de temporada** usado en la inferencia. Cualquier reorden a año calendario **fuera** de este *pipeline* debe anotarse en el PROVENANCE de la campaña.

### 2.2 Patrones estándar

| Etapa | Patrón | Ejemplo |
|-------|--------|---------|
| Mosaico de predictores (entrada) | `b14_chile_<region>_<año>_cog.tif` | `b14_chile_r4_2017_cog.tif` |
| Raster clasificado | `<stem_mosaico>_classified.tif` | `b14_chile_r4_2017_cog_classified.tif` |
| Tras paso temporal | `..._first_burn_year.tif` | — |
| Tras filtro LULC | `..._filtered_<timestamp>.tif` | en `classified_filtered/` |
| *Checkpoint* TF (base) | `col1_chile_<version>_<region>_rnn_lstm_ckpt` | `col1_chile_v1_r4_rnn_lstm_ckpt` |
| Hiperparámetros | `<base_checkpoint>_hyperparameters.json` | — |
| GPKG por tesela | derivado del *stem* del filtrado | vía `vectorize/` |
| Eventos nacionales anuales | `chile_<año>_events.gpkg` | `chile_2017_events.gpkg` |
| Raster nacional anual | `chile_<año>.tif` (variantes post-*merge* / *sieve*) | en `national_vector/` |
| Salidas prefiltro de área | `polygons_min20ha/` | ≥ 20 ha fijas |
| Producto por regla de área | `polygons_filtered_min20ha_<regla>.gpkg` | `_p25` por defecto |
| Directorios de campaña | `models_col1_<sello>`, `classification_<sello>` | `_20260619` |

### 2.3 Versión de modelo × región × años

| Región | Años del modelo `v1` | Años del modelo `v2` |
|--------|----------------------|----------------------|
| r2 | 2013–2018 **y** 2019–2025 (solo v1) | — |
| r1, r4, r6 | 2013–2018 | 2019–2025 |

### 2.4 Parsing de año y región

*Stem* típico de tesela, separado por `_`:

```text
b14_chile_r1_2013_cog_classified
 0    1    2   3   ...
           │   └── año (índice de token 3)
           └────── código de región
```

Productos nacionales anuales usan un *stem* más corto (`chile_2017_...`). Verifique siempre con `lib/tile_metadata.py` al automatizar.

### 2.5 Sello de campaña

Use un único sello `YYYYMMDD` (o etiqueta acordada) para **ambos** árboles de modelo y clasificación al promover una receta:

```text
models_col1_20260619/
classification_20260619/
classification_20260619/filtering_work/
```

Evite mezclar sellos entre modelos y rasters sin documentar un híbrido deliberado.

---

## 3. Cómo citar *commit* + entorno (informe técnico)

### 3.1 Captura (una vez al congelar la campaña)

```bash
cd ~/fire
git rev-parse HEAD > ~/campaigns/chile_fire_20260619/git_sha.txt
git status -sb >> ~/campaigns/chile_fire_20260619/git_sha.txt
date -Iseconds > ~/campaigns/chile_fire_20260619/frozen_at.txt

mkdir -p ~/campaigns/chile_fire_20260619/envs
cp classification/cluster_paths.env ~/campaigns/chile_fire_20260619/envs/classification.cluster_paths.env 2>/dev/null || true
cp filtering/cluster_paths.env      ~/campaigns/chile_fire_20260619/envs/filtering.cluster_paths.env 2>/dev/null || true
cp vectorize/cluster_paths.env      ~/campaigns/chile_fire_20260619/envs/vectorize.cluster_paths.env 2>/dev/null || true
```

No suba secretos al repositorio público; archive los env en un sobre **privado** de campaña.

### 3.2 Plantilla de procedencia (`PROVENANCE.md`)

Copie y complete:

```markdown
# Procedencia de campaña — MapBiomas Fire Chile

## Identificación
- Id de campaña: 20260619
- Intención del producto: clasificación de área quemada de producción (MLP conservador)
- Fecha de congelamiento:
- Operador:

## Software
- Repositorio: https://github.com/mapbiomas-chile/fire
- Rama: main
- Commit SHA:
- ¿Árbol de trabajo limpio? (sí/no; si no, listar archivos)

## Entorno de ejecución
- Host: NLHPC leftraru (u otro)
- Conda env: mb_fuego
- Versión de Python:
- Versión de TensorFlow (si se conoce):

## Instantáneas de configuración de rutas
- classification/cluster_paths.env → envs/classification.cluster_paths.env
- filtering/cluster_paths.env → envs/filtering.cluster_paths.env
- vectorize/cluster_paths.env → envs/vectorize.cluster_paths.env
- Origen de plantilla: cluster_paths.20260619.env.leftraru (según corresponda)

## Hiperparámetros de entrenamiento (si se entrenó en esta campaña)
- Arquitectura: MLP 7-14-7-14-7 ReLU, softmax 2 clases
- Optimizador: Adam; tasa: 0,001
- Lote: 1000; iteraciones: 7000
- Pérdida: entropía cruzada ponderada (frecuencia inversa de clase)
- Validación: by_file, fracción train 0,7; semilla: 42
- Topes: 2e6 train / 5e5 val
- Umbral de decisión: 0,55 (fijo)
- Ventana espacial: 0; oversample quemado: off; métrica: IoU

## Entradas
- Directorio de muestras:
- Directorio de mosaicos:
- Stack LULC + año de la banda 1:
- Cicatrices de referencia (si aplica):

## Salidas (rutas absolutas)
- Modelos:
- Clasificación:
- Raíz de trabajo de filtrado:
- Producto(s) vectorial(es) publicados:

## Parámetros de postproceso
- Ventana de estabilidad LULC:
- MAX_HOLE_AREA / FILL_AGRICULTURAL_HOLES / SKIP_MIN_PATCH_SIEVE:
- VECTORIZE_SKIP_SIEVE / píxeles mínimos de sieve:
- POLYGON_PRE_FILTER_HA / POLYGON_THRESHOLD_RULE:
- Distancia de agrupación nacional m / sieve nacional px:

## Notas
- Desvíos respecto a la documentación o plantillas:
```

### 3.3 Párrafo corto para métodos / informe técnico

**Español (completar corchetes):**

> La clasificación de área quemada para Chile se generó con el repositorio abierto MapBiomas Fire Chile ([mapbiomas-chile/fire](https://github.com/mapbiomas-chile/fire)), rama `main`, *commit* `[FULL_SHA]`, congelado el `[FECHA]`. El entrenamiento e inferencia emplearon un *multilayer perceptron* de cinco capas ocultas (anchuras 7–14–7–14–7), optimizador Adam (tasa 0,001), lote 1000, 7000 iteraciones, entropía cruzada ponderada por frecuencia inversa de clase, partición espacial por escena de muestreo (70%/30%, semilla 42) y umbral fijo de probabilidad de quemado 0,55. Las rutas absolutas y los indicadores de postproceso se archivan en el sobre de campaña `[RUTA/envs]`, derivados de las plantillas `20260619`. Los rasters y vectores intermedios y finales se almacenan en `models_col1_[STAMP]`, `classification_[STAMP]` y el árbol `filtering_work` asociado.

**English (fill brackets):**

> Burned-area classification for Chile was produced with the open MapBiomas Fire Chile repository ([mapbiomas-chile/fire](https://github.com/mapbiomas-chile/fire)), branch `main`, commit `[FULL_SHA]`, frozen on `[DATE]`. Training and inference used a five-layer multilayer perceptron (hidden widths 7–14–7–14–7), Adam optimizer (learning rate 0.001), batch size 1000, 7000 iterations, inverse-frequency weighted cross-entropy, spatial holdout by sample scene (70%/30%, seed 42), and a fixed burned-class probability threshold of 0.55. Absolute paths and post-processing flags are archived in campaign envelope `[PATH_TO_CAMPAIGN/envs]`, derived from the `20260619` path templates. Intermediate and final rasters and vectors are stored under `models_col1_[STAMP]`, `classification_[STAMP]`, and the associated `filtering_work` tree.

### 3.4 Nota de software estilo BibTeX (opcional)

```bibtex
@software{mapbiomas_fire_chile,
  title        = {MapBiomas Fire Chile burned-area pipeline},
  author       = {{MapBiomas Chile}},
  year         = {2026},
  version      = {commit FULL_SHA},
  url          = {https://github.com/mapbiomas-chile/fire},
  note         = {Branch main; campaign 20260619; see campaign PROVENANCE.md}
}
```

---

## 4. Documentación relacionada

| Tema | Documento |
|------|-----------|
| Pipeline de extremo a extremo | [../README.es.md](../README.es.md) |
| Parámetros de entrenamiento / inferencia | [../classification/README.es.md](../classification/README.es.md) |
| Filtros LULC y temporales | [../filtering/README.es.md](../filtering/README.es.md) |
| Polígonos y eventos nacionales | [../vectorize/README.es.md](../vectorize/README.es.md) |
