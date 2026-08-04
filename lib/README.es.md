# Biblioteca compartida (`lib/`)

**Español** | [English](README.md)

Utilidades Python importables compartidas por `filtering/`, `vectorize/` y herramientas afines. Los *pipelines* y la orquestación de rutas viven en esos directorios, **no** aquí.

---

## Módulos

| Módulo | Propósito |
|--------|-----------|
| `lulc_stability.py` | Ventanas de estabilidad multi-anual LULC |
| `tile_metadata.py` | Parsing de año / región en nombres MapBiomas |
| `vectorize.py` | Poligonización de máscaras binarias de quema |
| `vectorize_filtered_classified.py` | CLI de vectorización por tesela |
| `vectorize_national_by_year.py` | Merge nacional, *sieve*, eventos |
| `raster_by_year.py` | Fusión de rasters regionales a mosaicos anuales |
| `sieve_burn_mask.py` | Eliminación de componentes conectados pequeños |
| `group_fire_events.py` | Agrupación de polígonos cercanos en eventos |

---

## Uso

```python
from pathlib import Path
from lib.vectorize import polygonize_raster_file

polygonize_raster_file(Path("tile.tif"), Path("tile_burn.gpkg"))
```

Puntos de entrada del pipeline: [vectorize/README.es.md](../vectorize/README.es.md).

---

## Dependencias

`numpy`, `rasterio`, `geopandas`, `shapely`, `scipy`.
