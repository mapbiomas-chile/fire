# Cluster — filtrado por clases LULC

Pipeline en bash (`run_filtering_pipeline.sh` + `run_filtering_pipeline_slurm.sh`). Los scripts Python en `filtering/` **no se modifican**; solo se configuran rutas en `cluster_paths.env`.

## 1. Repositorio Git

| Qué | Ruta / rama |
|-----|-------------|
| Repo clonado | `/home/flepin/fire` |
| Rama | `feat/filtering_pipeline` |

```bash
cd ~/fire
git fetch origin
git checkout feat/filtering_pipeline
git pull
```

## 2. Entorno Conda

| Qué | Ruta |
|-----|------|
| Ambiente | `mb_fuego` |
| Python | `/home/flepin/.conda/envs/mb_fuego/bin/python` |

Paquetes mínimos: `numpy`, `rasterio` (y dependencias GDAL del sistema).

```bash
conda activate mb_fuego
python -c "import numpy, rasterio; print('OK')"
```

## 3. Datos de entrada (fuera del repo)

### 3.1 Stack LULC multibanda (`LULC_STACK`)

Un solo raster con **una banda por año** (MapBiomas collection02).

| Parámetro | Valor típico |
|-----------|----------------|
| Archivo | `/home/flepin/lulc_collection02/vrt/lulc_chile_collection02_1999_2025.vrt` |
| Año banda 1 | `1999` (`START_YEAR_BAND1`) |
| Años usados en máscaras | `2013`–`2025` (`FROM_YEAR` / `TO_YEAR`) |

**Importante:** si aparece `Writing through VRTSourcedRasterBand is not supported`, materializa el stack a GeoTIFF (una vez):

```bash
gdal_translate -co COMPRESS=DEFLATE -co TILED=YES \
  /home/flepin/lulc_collection02/vrt/lulc_chile_collection02_1999_2025.vrt \
  /home/flepin/lulc_collection02/lulc_chile_collection02_1999_2025.tif
```

Luego en `cluster_paths.env` apunta `LULC_STACK` al `.tif`.

*(Teselas GEE sueltas en `lulc_collection02/` no se usan en esta rama sin un script de mosaico; hace falta el stack multibanda o un GeoTIFF equivalente.)*

### 3.2 Clasificados de fuego (`CLASSIFIED_DIR`)

| Qué | Ruta típica |
|-----|-------------|
| Tiles clasificados | `/home/flepin/classi_v2/*.tif` |

Cada nombre debe contener el año (`20xx`) para emparejar con `mascara_total_<year>.tif`.

## 4. Salidas (`WORK_ROOT`)

Por defecto: `/home/flepin/filtering_work`

```
filtering_work/
├── mascaras/
│   ├── acumuladas/          # mascara_*_acumulado.tif
│   ├── by_year/             # mascara_rio_lago_2013.tif, ...
│   └── totales/             # mascara_total_2013.tif, ...
└── classified_filtered/     # tiles filtrados + JSON por tile
```

## 5. Configuración de rutas

```bash
cd ~/fire
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
# editar cluster_paths.env si hace falta
```

## 6. Ejecución

### Prueba en login (opcional)

```bash
source ~/fire/filtering/cluster_paths.env
cd ~/fire
bash filtering/run_filtering_pipeline.sh
```

### SLURM

```bash
cd ~/fire
sbatch filtering/run_filtering_pipeline_slurm.sh
# o solo filtrado si las máscaras ya existen:
sbatch filtering/run_filtering_pipeline_slurm.sh "" "" filter
```

Logs: `~/logs/fire_class_filter_<JOBID>.out` / `.err`

## 7. Pasos del pipeline

| Paso | Script Python | Qué hace |
|------|---------------|----------|
| `masks_accumulated` | `create_accumulated_class_masks.py` | Máscaras OR años fijos (roca, arena, salar, hielo, sin vegetación) |
| `masks_yearly` | `create_yearly_masks.py` | Máscaras por año (río/lago, infra, agricultura, pastura) |
| `masks_total` | `create_total_masks_by_year.py` | `mascara_total_<year>.tif` |
| `filter` | `filter_classified_parallel.py` | Aplica máscara a clasificados |

`STEPS=all` ejecuta los cuatro. No incluye polygonize ni umbral de área (scripts aparte en `filtering/README.md`).

## 8. Checklist rápido

- [ ] `~/fire` en rama `feat/filtering_pipeline`
- [ ] `filtering/cluster_paths.env` creado desde `.example`
- [ ] `LULC_STACK` existe (preferible `.tif`, no solo VRT)
- [ ] `CLASSIFIED_DIR` con GeoTIFFs
- [ ] `mb_fuego` con rasterio
- [ ] `~/logs` existe o SLURM lo crea
- [ ] `sbatch` desde nodo de envío
