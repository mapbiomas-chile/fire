# Filtrado (post-clasificación)

**Español** | [English](README.md)

Postproceso sobre **raster** aplicado **después** de la clasificación neuronal y **antes** de (o en coordinación con) la vectorización. El módulo elimina quemas multi-anuales persistentes, rellena huecos internos, suprime clases MapBiomas no quemables y, opcionalmente, aplica un *sieve* de parches pequeños.

Upstream: [clasificación](../classification/README.es.md). Downstream: [vectorización](../vectorize/README.es.md), [validación](../validation/README.es.md).

---

## 1. Secuencia de procesamiento

```text
Stack LULC MapBiomas multibanda
        │
        ▼
[1] Máscaras no quemables (acumuladas + anuales + total)
        │
Clasificados brutos *_classified.tif
        │
        ▼
[2] Filtro temporal: primer año de quema en el píxel
        │
        ▼
[3] Relleno de huecos internos (fill_holes)
        │
        ▼
[4] Filtro LULC (mascara_total_Y)
        │
        ▼
[4c] Relleno de huecos agrícolas cerrados (opcional)
        │
        ▼
[4b] Sieve de parches mínimos en raster (opcional; a menudo omitido en producción)
        │
        ▼
classified_filtered/  →  vectorize / análisis de área poligonal
```

Orquestación: `run_filtering_pipeline.sh` (pasos 1–4). Cadena ligada: `run_classified_filters.py` (pasos 2–4±4b).

---

## 2. Métodos por etapa (valores de producción)

### 2.1 Máscaras LULC

**Objetivo.** Para cada año de filtro \(Y\), generar `mascara_total_Y.tif` donde **1** = eliminar quema y **0** = conservar.

| Subpaso | Script | Contenido |
|---------|--------|-----------|
| OR acumulado en todas las bandas LULC | `create_accumulated_class_masks.py` | Roca 29, arena 23, salar 61, hielo 34, sin vegetación 25 |
| Estabilidad anual de clases dinámicas | `create_yearly_masks.py` | Agua 33, infraestructura 24, agricultura 15, pastura 18 |
| Total | `create_total_masks_by_year.py` | Unión para el año \(Y\) |

**Ventana de estabilidad** (por defecto `LULC_STABILITY_WINDOW=4`): el píxel se marca solo si la clase persiste durante cuatro años LULC consecutivos en torno a \(Y\). Use ventana `1` para máscaras de un solo año. El año de la banda 1 del stack se define con `START_YEAR_BAND1` (a menudo 2000). Si falta LULC 2025: opcional `COPY_MASK_2025_FROM_2024=1`.

La entrada debe ser un **GeoTIFF** multibanda, no un VRT.

### 2.2 Primer año de quema (temporal)

**Objetivo.** Eliminar la persistencia multi-anual en el mismo píxel: se conserva el año de quema **más temprano** (2013 ≻ 2014 ≻ … ≻ 2025).

*Script:* `filter_temporal_first_burn_year.py`. El año se lee del token de nombre en la posición documentada (`b14_chile_r1_2013_...`). Fusión opcional por 8-vecinos (`TEMPORAL_SPATIAL_MERGE=1`).

### 2.3 Relleno de huecos

**Objetivo.** Rellenar vacíos no quemados **totalmente cerrados** dentro de cicatrices, sin mover el perímetro exterior.

*Script:* `refine_burn_mask_closing.py` con `--method fill_holes`. En producción `MAX_HOLE_AREA=0` rellena todos los huecos cerrados (el límite se mide en píxeles; 0 desactiva el tope de tamaño).

### 2.4 Aplicación LULC

*Script:* `filter_classified_parallel.py`. Por tesela del año \(Y\), anula la quema donde `mascara_total_Y = 1`.

### 2.5 Huecos agrícolas cerrados (opcional)

Tras el LULC, `fill_agricultural_holes_in_scars.py` puede restaurar quema solo en huecos agrícolas **cerrados** **dentro** de cicatrices. Se recomienda máscara agrícola estricta (`LULC_AGRICULTURE_STABILITY_WINDOW=1`). Activo con `FILL_AGRICULTURAL_HOLES=1`.

### 2.6 Sieve de parches (opcional)

`sieve_min_patch_parallel.py` elimina componentes conectados pequeños. En producción suele usarse `SKIP_MIN_PATCH_SIEVE=1` y el control de tamaño pasa al pipeline **vectorial** (≥ 20 ha y luego percentiles).

---

## 3. Filtro de área en polígonos (dominio vectorial)

Tras la vectorización por tesela (`vectorize/`), el pipeline opcional (`run_polygon_area_pipeline.sh`):

1. **Prefiltro fijo** `POLYGON_PRE_FILTER_HA=20` (≥ 20 ha).  
2. Histogramas de áreas restantes.  
3. Recomendación de umbrales **p5, p10, p25, elbow** por región × año.  
4. Aplicación de **una** regla (producción por defecto **p25**).

No sustituye los filtros raster; elimina eventos pequeños en espacio de atributos (hectáreas).

---

## 4. Replicación en NLHPC (producción 20260619)

Detalle interactivo: [LOCAL.md](LOCAL.md). SLURM: [CLUSTER.md](CLUSTER.md).

```bash
cd ~/fire
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env

bash filtering/run_filtering_pipeline.sh
# o: sbatch filtering/run_filtering_pipeline_slurm.sh
```

Estructura esperada bajo `$WORK_ROOT`:

```text
$WORK_ROOT/
├── mascaras/{acumuladas,by_year,totales}/
├── classified_filtered/     # producto raster principal
└── logs/{filter_stats,fill_stats}.json
```

Variables clave de producción (véase `cluster_paths.20260619.env.leftraru`):

| Variable | Valor típico | Significado |
|----------|--------------|-------------|
| `FROM_YEAR` / `TO_YEAR` | 2013 / 2025 | Serie de quema |
| `LULC_TO_YEAR` | 2024 | Último año LULC real |
| `LULC_STABILITY_WINDOW` | 4 | Estabilidad de clases dinámicas |
| `MAX_HOLE_AREA` | 0 | Rellenar todos los huecos cerrados |
| `FILL_AGRICULTURAL_HOLES` | 1 | Huecos agrícolas cerrados |
| `SKIP_MIN_PATCH_SIEVE` | 1 | Sin sieve raster mínimo |
| `WORKERS` | 4 | Paralelismo |

Reejecuciones parciales: `export STEPS=filter` (u otras etapas listadas en el *script*).

---

## 5. Convención de nombres

```text
b14_chile_r1_2013_cog_classified.tif
              ^^^^ año calendario en el nombre del archivo
```

---

## 6. Dependencias

| Etapas | Paquetes |
|--------|----------|
| Máscaras + filtros raster | `numpy`, `rasterio` |
| Huecos / morfología | + `scipy` |
| Herramientas de área poligonal | `geopandas`, `matplotlib` |

Entorno habitual en leftraru: Conda **`mb_fuego`**.

---

## 7. Índice de archivos

| Archivo | Función |
|---------|---------|
| `create_accumulated_class_masks.py` | §2.1 acumuladas |
| `create_yearly_masks.py` | §2.1 anuales |
| `create_total_masks_by_year.py` | §2.1 total |
| `filter_temporal_first_burn_year.py` | §2.2 |
| `refine_burn_mask_closing.py` | §2.3 |
| `filter_classified_parallel.py` | §2.4 |
| `fill_agricultural_holes_in_scars.py` | §2.5 |
| `sieve_min_patch_parallel.py` | §2.6 |
| `run_classified_filters.py` | Cadena 2–4 |
| `run_filtering_pipeline.sh` | Orquestación completa |
| `run_polygon_area_pipeline.sh` | Filtro de área §3 |
| `cluster_paths.20260619.env.leftraru` | Rutas de producción |
| `cluster_paths.env.example` | Plantilla genérica |
