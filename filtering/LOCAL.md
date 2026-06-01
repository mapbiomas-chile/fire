# Filtrado local (Windows)

## 1. Rutas de datos

| Qué | Ruta (ajústala si guardas los datos en otro disco) |
|-----|---------------------------------------------------|
| LULC multibanda | `D:/flepin/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif` |
| Clasificados | `D:/flepin/classification_20260528/*.tif` |
| Salidas | `D:/flepin/filtering_work_local/` |

El stack LULC llega hasta **2024**. Para filtrar tiles **2025** (clasificación copiada desde 2024), el pipeline duplica las máscaras 2024 → 2025 (`COPY_MASK_2025_FROM_2024=1`).

## 2. Configuración

```bash
cd /c/Users/pipel/OneDrive/Escritorio/MAPBIOMAS/CURSOR/fire
git checkout feat/filtering_pipeline

cp filtering/cluster_paths.local.env.example filtering/cluster_paths.env
```

Edita `cluster_paths.env`:

1. **`LULC_STACK`** y **`CLASSIFIED_DIR`** si no están en `D:/flepin/`.
2. **`START_YEAR_BAND1`**: año de la banda 1 del GeoTIFF. Comprueba con:

```bash
python -c "import rasterio; p=r'D:/flepin/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif'; s=rasterio.open(p); print('bands', s.count); s.close()"
```

Si la banda 1 es 2000 y la 25 es 2024, usa `START_YEAR_BAND1=2000`.

## 3. Ejecutar

**Git Bash:**

```bash
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh
```

**PowerShell** (mismas rutas; ejecuta los 4 scripts Python como en el README principal).

## 4. Salidas

```
D:/flepin/filtering_work_local/
├── mascaras/acumuladas/
├── mascaras/by_year/
├── mascaras/totales/     ← mascara_total_2025.tif (copiada desde 2024 si aplica)
└── classified_filtered/
```
