# Validación

**Español** | [English](README.md)

Herramientas para preparar **capas de cicatrices de referencia** y cuantificar la concordancia con las quemas mapeadas tras la clasificación / filtrado / vectorización. Las métricas de área requieren proyección de **área igual**.

Relacionado: [clasificación](../classification/README.es.md), [filtrado](../filtering/README.es.md), [vectorización](../vectorize/README.es.md).

---

## 1. Cadena típica de validación

```text
Cicatrices de referencia (atributo Season / año)
        │
        ▼
Reproyección a área igual (preset Chile Albers)
        │
        ▼
Partición opcional por año calendario
        │
        ▼
Producto clasificado poligonizado (filtering/polygonize o salidas de vectorize)
        │  (mismo CRS)
        ▼
Intersección cicatriz × polígonos clasificados → GeoPackage de *hits*
        │
        ├─► Índice de Jaccard por cicatriz
        └─► Métricas de cercanía Singh et al. (2015)
```

La poligonización de clasificados se implementa en **`filtering/polygonize_mask_parallel.py`**, no en esta carpeta.

---

## 2. Inventario de módulos

| Script | Función |
|--------|---------|
| `reproject_vector_to_equal_area.py` | Vector → CRS de área igual; `area_m2`, `area_ha` |
| `reproject_raster_to_equal_area.py` | Warp de raster al mismo CRS |
| `merge_reprojected_tiles_by_year.py` | Mosaico anual de teselas regionales |
| `split_vector_by_year.py` | Un GPKG por año (columna `Season` por defecto) |
| `plot_area_distribution.py` | Histograma de áreas |
| `intersect_top_n_scars_with_classified.py` | Intersección (`scar`, `classified_hits`) |
| `calculate_jaccard_index.py` | Jaccard por cicatriz |
| `spatial_validation_metrics.py` | Over/under-segmentación, \(D\), \(D_{\mathrm{norm}}\) |
| `extract_top_fire_events.py` | Top-\(N\) eventos desde `chile_YYYY_events.gpkg` |

Dependencias de métricas espaciales: `pip install -r validation/requirements-spatial-validation.txt`.

---

## 3. Presets de CRS de área igual

| Preset | Definición |
|--------|------------|
| `chile_albers` (por defecto) | Albers ajustado a Chile |
| `south_america_albers` | ESRI:102033 |

Puede sobrescribirse con cualquier `--target-crs` válido para `pyproj`.

---

## 4. Intersecciones y métricas

**Intersecciones.** El catálogo debe estar proyectado. El emparejamiento es por **año calendario**. Capas de salida: `scar` (\(A\)) y `classified_hits` (polígonos \(b_i\) completos).

**Jaccard.** \(B = \bigcup_i b_i\), \(J = |A∩B|/|A∪B|\). Sin *hits*: \(J = 0\).

**Singh et al. (2015).** Over- y under-segmentación por pares;  
\(D = \sqrt{OS^2 + US^2}\), \(D_{\mathrm{norm}} = 1 - D/\sqrt{2}\).

---

## 5. Ejemplos mínimos reproducibles

```bash
python validation/reproject_vector_to_equal_area.py \
  --input cicatrices.shp --output cicatrices_albers.gpkg --preset chile_albers

python validation/intersect_top_n_scars_with_classified.py \
  --catalog cicatrices_albers.gpkg --year 2017 --top-n 50 \
  --classified-dir /path/to/polygons_2017 \
  --output hits_2017.gpkg --workers 8

python validation/calculate_jaccard_index.py \
  --hits-gpkg hits_2017.gpkg --output-csv jaccard_2017.csv

python validation/spatial_validation_metrics.py \
  --hits-dir /path/to/hits_by_year \
  --hits-pattern "hits_*.gpkg" \
  --output-dir /path/to/spatial_metrics --by-region
```

---

## 6. Top de eventos nacionales

```bash
python validation/extract_top_fire_events.py \
  --input-dir /path/to/polygons_chile \
  --output-gpkg top50.gpkg \
  --from-year 2014 --to-year 2025 \
  --min-ha 200 --max-ha 5000 --top-n 50
```
