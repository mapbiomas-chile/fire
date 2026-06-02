# NLHPC — ejecución con SLURM

Guía para enviar el pipeline a la cola.  
Índice de scripts y flujo completo: [README.md](README.md).

## Checklist

- [ ] Repo en `~/fire`, rama `feat/filtering_pipeline`
- [ ] `filtering/cluster_paths.env` desde `cluster_paths.env.example`
- [ ] `LULC_STACK` es GeoTIFF (`.tif`), no `.vrt`
- [ ] `CLASSIFIED_DIR` con tiles clasificados
- [ ] Conda `mb_fuego` con `numpy`, `rasterio`
- [ ] `~/logs` existe

## Configuración

```bash
cd ~/fire
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
# editar rutas si hace falta
```

## Ejecutar

```bash
cd ~/fire
sbatch filtering/run_filtering_pipeline_slurm.sh

# Solo filtrado (máscaras ya generadas):
sbatch filtering/run_filtering_pipeline_slurm.sh "" "" filter
```

Logs: `~/logs/fire_class_filter_<JOBID>.out` / `.err`

## Pasos del pipeline

| `STEPS` | Script |
|---------|--------|
| `masks_accumulated` | `create_accumulated_class_masks.py` |
| `masks_yearly` | `create_yearly_masks.py` |
| `masks_total` | `create_total_masks_by_year.py` |
| `filter` | `run_classified_filters.py` (LULC + temporal) |

`STEPS=all` ejecuta los cuatro. Polygonize y umbral de área son pasos manuales aparte (ver [README.md](README.md)).
