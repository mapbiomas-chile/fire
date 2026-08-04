# Pipeline de vectorización

**Español** | [English](README.md)

Convierte rasters binarios de quema **ya filtrados** en eventos poligonales (GeoPackage). Esta etapa **no** sustituye la clasificación ni el filtro LULC.

Upstream: [filtrado](../filtering/README.es.md). Algoritmos compartidos: [lib](../lib/README.es.md). Notas interactivas: [LOCAL.md](LOCAL.md). SLURM: [CLUSTER.md](CLUSTER.md).

---

## 1. Dos modos operativos

```text
classified_filtered/*.tif
        │
        ├─► Polígonos por tesela  →  WORK_ROOT/polygons/*.gpkg
        │         │
        │         └─► Filtro de área opcional (filtering/run_polygon_area_pipeline.sh)
        │
        └─► Nacional por año   →  national_vector/
                 merge → sieve (≥112 px) → poligonizar
                 → filtro de fragmentos (≥112 px) → agrupar ≤ 200 m
```

| Modo | Script | Producto principal |
|------|--------|--------------------|
| Por tesela | `run_vectorize_pipeline.sh` | Un GPKG por raster filtrado |
| Nacional | `run_vectorize_national_pipeline.sh` | `chile_<año>_events.gpkg` |
| Cadena completa post-filtro | `run_post_filter_pipeline.sh` | Tesela + área + (opcional) nacional |

---

## 2. Métodos (parámetros para replicación)

### Por tesela

1. Leer máscara binaria de quema (`mask_value = 1` por defecto).  
2. *Sieve* opcional pre-poligonización (`VECTORIZE_SIEVE_MIN_PIXELS`, a menudo **112** ≈ 1 ha a 30 m). En evaluación de área de producción puede usarse `VECTORIZE_SKIP_SIEVE=1` y controlar el tamaño en hectáreas más adelante.  
3. Poligonizar; atributos `year`, `region`, `source_file`.

### Nacional

1. Fusionar teselas regionales del año calendario \(Y\).  
2. *Sieve* en raster: ≥ **112** píxeles conectados.  
3. Poligonizar; filtrar fragmentos < **112** px antes de agrupar.  
4. Agrupar componentes a ≤ **200 m** en **eventos** multipolígono.

Implementación: `lib/vectorize_national_by_year.py`.

---

## 3. Replicación (nodo login NLHPC — recomendado)

```bash
cd ~/fire
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env
conda activate mb_fuego

bash vectorize/run_vectorize_pipeline.sh
bash filtering/run_polygon_area_pipeline.sh
bash vectorize/run_vectorize_national_pipeline.sh   # opcional
```

---

## 4. Variables de configuración

### Por tesela

| Variable | Idea por defecto | Rol |
|----------|------------------|-----|
| `PYTHON` | Conda `mb_fuego` | Intérprete |
| `WORK_ROOT` | raíz del trabajo de filtrado | Rutas compartidas |
| `VECTORIZE_INPUT_DIR` | `$WORK_ROOT/classified_filtered` | Entrada |
| `VECTORIZE_OUTPUT_DIR` | `$WORK_ROOT/polygons` | Salida |
| `VECTORIZE_WORKERS` | 4 (bajar a 2 si el login satura) | Paralelismo |
| `VECTORIZE_SIEVE_MIN_PIXELS` | 112 | *Sieve* pre-poligonizar |
| `VECTORIZE_SKIP_SIEVE` | 0 / 1 | Desactivar *sieve* |

### Nacional

| Variable | Default | Rol |
|----------|---------|-----|
| `VECTORIZE_NATIONAL_GROUP_DISTANCE_M` | 200 | Distancia de agrupación |
| `VECTORIZE_NATIONAL_SIEVE_MIN_PIXELS` | 112 | *Sieve* raster |
| `VECTORIZE_NATIONAL_FRAGMENT_MIN_PIXELS` | 112 | Filtro de fragmentos |
| `VECTORIZE_NATIONAL_FROM_YEAR` / `TO_YEAR` | 2013 / 2025 | Años |

---

## 5. Atributos de salida

**Tesela:** `year`, `region`, `source_file`, `mask_value`, geometría.

**Eventos nacionales:** `event_id`, `year`, `fragment_count`, `area_ha`, `area_m2`, `max_gap_m`, geometría.

---

## 6. Dependencias

`geopandas`, `rasterio`, `shapely`, `numpy` (y módulos en `lib/`).
