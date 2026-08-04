# Utilidades

**Español** | [English](README.md)

*Scripts* de soporte **independientes de etapa**: exportación desde Google Earth Engine, geometría de regiones de fuego, listado de tiles, mosaico de subconjuntos GeoTIFF e inspección rápida de metadatos.

No sustituyen la cadena principal clasificación → filtrado → vectorización ([README raíz](../README.es.md)).

---

## Inventario

| Script | Función |
|--------|---------|
| `download_regiones_fuego_asset.py` | Exportar `regiones_fuego_chile_v1` desde GEE a Drive |
| `fire_regions_bbox_geojson.py` | Casco convexo (excluye región, por defecto 5) → *bbox* → GeoJSON opcional |
| `list_intersecting_tiles.py` | Listar tiles `.tif` que intersectan el casco |
| `mosaic_subset_clip_bbox.py` | Fusionar subconjunto recortado al *bbox* |
| `print_tif_metadata.py` | CRS, *transform*, tamaño, *dtype*, bandas |

Helpers importables: `convex_hull_excluding_region`, `bbox_envelope_excluding_region`.

---

## Ejemplos

```bash
python utilities/fire_regions_bbox_geojson.py \
  --geojson path/to/regiones_fuego.geojson \
  --output path/to/fire_regions_bbox.geojson

python utilities/print_tif_metadata.py path/to/raster.tif
```

`download_regiones_fuego_asset.py` lleva constantes GEE/proyecto al inicio del archivo; edítelas antes de ejecutar con credenciales de Earth Engine.
