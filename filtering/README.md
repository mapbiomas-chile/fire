# Filtering

Utilidades de **post-clasificación** MapBiomas Fire: limpian los rasters del clasificador antes de polygonizar o validar contra cicatrices de referencia.

- **Entrada:** GeoTIFFs del clasificador → [`../classification/`](../classification/README.md)
- **Salida filtrada:** rasters binarios 0/1 listos para polygonize o validación → [`../validation/`](../validation/README.md)

---

## Flujo de trabajo (orden lógico)

Los filtros se aplican en esta secuencia. Cada bloque tiene uno o más scripts asociados.

```text
Stack LULC multibanda
        │
        ▼
┌───────────────────────────────────────┐
│ 1. Máscaras LULC (clases no quemables)│  create_accumulated_class_masks.py
│                                       │  create_yearly_masks.py
│                                       │  create_total_masks_by_year.py
└───────────────────────────────────────┘
        │
        ▼  mascara_total_<year>.tif
┌───────────────────────────────────────┐
│ 2. Filtro LULC sobre clasificados    │  filter_classified_parallel.py
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 3. Filtro temporal (first burn year)  │  filter_temporal_first_burn_year.py
└───────────────────────────────────────┘
        │
        ▼  classified_filtered/
   (opcional, para validación)
┌───────────────────────────────────────┐
│ 4. Polygonize → histogramas → umbral  │  polygonize_mask_parallel.py
│                                       │  summarize_histograms_by_region.py
│                                       │  filter_polygons_by_threshold.py
└───────────────────────────────────────┘
```

Los pasos **1–3** pueden ejecutarse juntos con `run_classified_filters.py` (pasos 2+3) o con el pipeline bash (pasos 1–3 completos). Ver [Pipeline automatizado](#pipeline-automatizado) al final.

---

## 1. Máscaras LULC (clases no quemables)

**Qué hace:** construye, para cada año, una máscara binaria donde `1` = píxel a **eliminar** del clasificado (clase MapBiomas no quemable) y `0` = mantener.

**Entrada:** un GeoTIFF multibanda MapBiomas (una banda = un año). No usar `.vrt`.

**Salida final de esta etapa:** `mascara_total_<year>.tif` (una por año).

### Scripts (en orden)

| Orden | Script | Salida |
|-------|--------|--------|
| 1a | `create_accumulated_class_masks.py` | `mascara_<clase>_acumulado.tif` |
| 1b | `create_yearly_masks.py` | `mascara_<clase>_<year>.tif` |
| 1c | `create_total_masks_by_year.py` | `mascara_total_<year>.tif` |

**1a — Acumuladas** (`create_accumulated_class_masks.py`): OR de clases **fijas en el tiempo** sobre todas las bandas del stack:

| Clase | Código MapBiomas | Archivo |
|-------|------------------|---------|
| Roca | 29 | `mascara_roca_acumulado.tif` |
| Arena / playa / duna | 23 | `mascara_arena_acumulado.tif` |
| Salar | 61 | `mascara_salar_acumulado.tif` |
| Hielo / nieve | 34 | `mascara_hielo_acumulado.tif` |
| Sin vegetación | 25 | `mascara_sin_vegetacion_acumulado.tif` |

**1b — Anuales** (`create_yearly_masks.py`): una máscara por año y por clase **variable**:

| Clase | Código | Patrón de salida |
|-------|--------|------------------|
| Río / lago | 33 | `mascara_rio_lago_<year>.tif` |
| Infraestructura | 24 | `mascara_infraestructura_<year>.tif` |
| Agricultura | 15 | `mascara_agricultura_<year>.tif` |
| Pastura | 18 | `mascara_pastura_<year>.tif` |

Paraleliza por año (`--workers`). Si el LULC no tiene banda 2025, el pipeline puede copiar máscaras 2024→2025 (`COPY_MASK_2025_FROM_2024=1`).

**1c — Total por año** (`create_total_masks_by_year.py`): combina acumuladas + anuales en un solo raster por año. Este es el insumo del filtro LULC sobre clasificados.

```bash
python filtering/create_accumulated_class_masks.py \
  --input-tif /path/to/lulc_stack.tif \
  --output-dir /path/to/mascaras/acumuladas

python filtering/create_yearly_masks.py \
  --input-tif /path/to/lulc_stack.tif \
  --output-dir /path/to/mascaras/by_year \
  --start-year-in-band-1 2000 --from-year 2013 --to-year 2024

python filtering/create_total_masks_by_year.py \
  --mascaras-root /path/to/mascaras \
  --from-year 2013 --to-year 2025
```

---

## 2. Filtro LULC sobre clasificados

**Qué hace:** para cada tile clasificado del año Y, aplica `mascara_total_Y.tif`. Donde la máscara es `1`, el píxel quemado pasa a `0`.

**Script:** `filter_classified_parallel.py`

- Extrae el año del nombre del archivo (`20xx`).
- Reproyecta la máscara al grid del tile si hace falta.
- Escribe `*_filtered_<timestamp>.tif` + un JSON de resumen por tile.

```bash
python filtering/filter_classified_parallel.py \
  --input-dir /path/to/classi_v2 \
  --masks-dir /path/to/mascaras/totales \
  --output-dir /path/to/classified_lulc_only \
  --workers 4
```

---

## 3. Filtro temporal (first burn year)

**Qué hace:** elimina **persistencia multi-año** en el mismo píxel. Si un píxel está quemado en 2013 y otra vez en 2014, solo queda en **2013** (2013 > 2014 > 2015 > … > 2025).

**Script:** `filter_temporal_first_burn_year.py`

- Agrupa todos los años de un mismo tile (año en token índice 3: `b14_chile_r1_**2013**_...`).
- Recorre años en orden cronológico; el primer año con quema gana el píxel.
- Salida: rasters `uint8` con valores `0` y `1`, sufijo `_first_burn_year`.

Opcional (`TEMPORAL_SPATIAL_MERGE=1`): píxeles nuevos en año Y pero **8-conectados** a una cicatriz de un año anterior se atribuyen a ese año origen (caso dic 2017 / ene 2018).

```bash
python filtering/filter_temporal_first_burn_year.py \
  --input-dir /path/to/classified_lulc_only \
  --output-dir /path/to/classified_filtered \
  --from-year 2013 --to-year 2025 \
  --no-spatial-merge
```

---

## Pasos 2 + 3 en un solo comando

**Script:** `run_classified_filters.py`

Encadena `filter_classified_parallel.py` → `filter_temporal_first_burn_year.py`. Es lo que ejecuta el paso `filter` del pipeline bash.

```bash
python filtering/run_classified_filters.py \
  --classified-dir /path/to/classi_v2 \
  --masks-dir /path/to/mascaras/totales \
  --output-dir /path/to/classified_filtered \
  --from-year 2013 --to-year 2025 \
  --stats-json /path/to/logs/filter_stats.json
```

| Flag | Uso |
|------|-----|
| `--lulc-only` | Solo filtro LULC (paso 2) |
| `--temporal-only` | Solo filtro temporal (paso 3); entrada = salida del paso 2 |
| `--no-spatial-merge` | Solo dedup por mismo píxel (default) |
| `--name-contains 141228` | Limitar temporal a ciertos tiles |

---

## 4. Polygonize, histogramas y umbral (opcional)

Pasos **posteriores** al filtrado raster. No están en el pipeline bash; se corren cuando necesitas polígonos o validación vectorial.

| Paso | Script | Qué hace |
|------|--------|----------|
| Polygonize | `polygonize_mask_parallel.py` | Píxeles = 1 → polígonos; un GPKG por raster |
| Histogramas | `summarize_histograms_by_region.py` | Distribución de áreas por región (`r1`, `r2`, …) para elegir umbral |
| Umbral | `filter_polygons_by_threshold.py` | Conserva polígonos ≥ N hectáreas |

```bash
python filtering/polygonize_mask_parallel.py \
  --input-dir /path/to/classified_filtered \
  --output-dir /path/to/polygons --mask-value 1 --workers 4

python filtering/summarize_histograms_by_region.py \
  --input-dir /path/to/polygons --output-dir /path/to/histograms

python filtering/filter_polygons_by_threshold.py \
  --input-dir /path/to/polygons \
  --output-gpkg /path/to/polygons_filtered.gpkg \
  --threshold-ha 10
```

Para validación con cicatrices: reproyectar a CRS de área igual ([`../validation/`](../validation/README.md)) y usar `intersect_top_n_scars_with_classified.py`.

---

## Pipeline automatizado

Cuando las máscaras y los clasificados ya están en disco, puedes ejecutar **todo el flujo 1–3** con bash en lugar de correr script por script.

| Archivo | Rol |
|---------|-----|
| `run_filtering_pipeline.sh` | Orquestación: máscaras + filtrado |
| `run_filtering_pipeline_slurm.sh` | Wrapper SLURM para NLHPC |
| `cluster_paths.env.example` | Plantilla de rutas → copiar a `cluster_paths.env` |

### Pasos del pipeline (`STEPS`)

| `STEPS` | Equivale a | Script |
|---------|------------|--------|
| `masks_accumulated` | § 1a | `create_accumulated_class_masks.py` |
| `masks_yearly` | § 1b | `create_yearly_masks.py` |
| `masks_total` | § 1c | `create_total_masks_by_year.py` |
| **`filter`** | **§ 2 + § 3** | **`run_classified_filters.py`** |

Legacy (re-ejecución parcial): `lulc_filter` (solo § 2), `temporal_first_burn` (solo § 3).

```bash
cd ~/fire
cp filtering/cluster_paths.env.example filtering/cluster_paths.env   # opcional
bash filtering/run_filtering_pipeline.sh                             # STEPS=all
```

- **Leftraru interactivo (sin sbatch):** [LOCAL.md](LOCAL.md)
- **Cola SLURM:** [CLUSTER.md](CLUSTER.md)

### Variables de entorno

| Variable | Descripción |
|----------|-------------|
| `LULC_STACK` | GeoTIFF multibanda MapBiomas |
| `CLASSIFIED_DIR` | Tiles clasificados de entrada |
| `WORK_ROOT` | Raíz de salidas |
| `FILTER_OUTPUT_DIR` | Salida final (§ 2 + § 3) → `classified_filtered/` |
| `FROM_YEAR` / `TO_YEAR` | Rango de años (default 2013–2025) |
| `LULC_TO_YEAR` | Último año con banda LULC real (default 2024) |
| `KEEP_LULC_INTERMEDIATE` | 1 = conservar salida intermedia del § 2 |
| `FILTER_NAME_CONTAINS` | Limitar § 3 a ciertos filenames |
| `TEMPORAL_SPATIAL_MERGE` | 1 = activar fusión espacial (§ 3) |

### Estructura de salida

```text
${WORK_ROOT}/
├── mascaras/
│   ├── acumuladas/          ← § 1a
│   ├── by_year/             ← § 1b
│   └── totales/             ← § 1c  (mascara_total_<year>.tif)
├── classified_filtered/     ← § 2 + § 3 (salida final)
├── classified_lulc_only/    ← § 2 solo (si KEEP_LULC_INTERMEDIATE=1)
└── logs/filter_stats.json
```

---

## Convención de nombres

Tiles MapBiomas esperados:

```text
b14_chile_r1_2013_cog_classified.tif
              ^^^^
              token índice 3 = año calendario
```

Tras § 2: `..._filtered_<timestamp>.tif`  
Tras § 3: `..._filtered_<timestamp>_first_burn_year.tif`

---

## Dependencias

| Etapas | Paquetes |
|--------|----------|
| § 1–3 (máscaras + filtrado) | `numpy`, `rasterio` |
| § 4 polygonize + umbral | `geopandas` |
| § 4 histogramas | `geopandas`, `matplotlib` |

Ambiente cluster: Conda `mb_fuego`.

---

## Índice de archivos

| Archivo | Sección |
|---------|---------|
| `create_accumulated_class_masks.py` | § 1a |
| `create_yearly_masks.py` | § 1b |
| `create_total_masks_by_year.py` | § 1c |
| `filter_classified_parallel.py` | § 2 |
| `filter_temporal_first_burn_year.py` | § 3 |
| `run_classified_filters.py` | § 2 + § 3 |
| `polygonize_mask_parallel.py` | § 4 |
| `summarize_histograms_by_region.py` | § 4 |
| `filter_polygons_by_threshold.py` | § 4 |
| `run_filtering_pipeline.sh` | Pipeline |
| `run_filtering_pipeline_slurm.sh` | Pipeline (SLURM) |
| `cluster_paths.env.example` | Pipeline (config) |
