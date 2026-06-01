# Filtrado interactivo en leftraru (sin SLURM)

Correr el pipeline **en la máquina leftraru** (SSH), **sin** `sbatch`: no entra a la cola del cluster, pero usa los mismos datos y scripts que en un job.

Para enviar a la cola → [CLUSTER.md](CLUSTER.md) y `sbatch filtering/run_filtering_pipeline_slurm.sh`.

## 1. Datos en leftraru

| Qué | Ruta |
|-----|------|
| Repo | `/home/flepin/fire` (rama `feat/filtering_pipeline`) |
| LULC multibanda | `/home/flepin/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif` |
| Clasificados | `/home/flepin/classification_20260528/*.tif` |
| Salidas | `/home/flepin/filtering_work_local/` |

LULC hasta **2024**; para **2025** el pipeline copia máscaras 2024→2025 (`COPY_MASK_2025_FROM_2024=1`).

## 2. Configuración

Las rutas ya están en `run_filtering_pipeline.sh`. Solo hace falta clonar/actualizar el repo:

```bash
cd ~/fire
git fetch origin
git checkout feat/filtering_pipeline
git pull
```

Opcional: `cp filtering/cluster_paths.leftraru.env.example filtering/cluster_paths.env` para cambiar rutas sin editar el `.sh`.

Comprueba bandas del LULC y ajusta `START_YEAR_BAND1` si hace falta (por defecto `2000`):

```bash
$PYTHON -c "
import rasterio
p='/home/flepin/lulc_collection02/lulc_2025_subset_mosaic_bbox_without_region5.tif'
with rasterio.open(p) as s:
    print('bands', s.count)
"
```

## 3. Ejecutar (sin sbatch)

```bash
cd ~/fire
conda activate mb_fuego
bash filtering/run_filtering_pipeline.sh
```

Recomendado en sesión larga:

```bash
screen -S filter
# ... comandos de arriba ...
# Ctrl+A, D para detach
```

**No uses** `sbatch` si quieres solo interactivo.

## 4. Salidas

```
/home/flepin/filtering_work_local/
├── mascaras/acumuladas/
├── mascaras/by_year/
├── mascaras/totales/
└── classified_filtered/
```

## 5. Pasos parciales

En `cluster_paths.env`:

```bash
export STEPS="masks_accumulated,masks_yearly,masks_total"   # solo máscaras
# export STEPS="filter"                                     # solo filtrar
```

## 6. Si se queda sin memoria en login

- Baja `WORKERS` (p. ej. `2` o `1`) en `cluster_paths.env`.
- O usa SLURM con más RAM: `sbatch filtering/run_filtering_pipeline_slurm.sh`.
