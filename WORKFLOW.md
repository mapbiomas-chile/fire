# Workflow — reprocesar en la misma carpeta

Salida canónica Chile (modelos junio 2026):

| Etapa | Ruta |
| --- | --- |
| Clasificados | `/home/flepin/classification_20260618/` |
| Filtrado | `/home/flepin/classification_20260618/filtering_work/` |
| Modelos | `/home/flepin/models_col1_20260618/` |

## Regla por defecto

**No crear carpeta nueva** salvo que lo pidas explícitamente. Al volver a correr un paso:

1. Borrar en esa carpeta solo los archivos que se van a regenerar (región/año/paso).
2. Ejecutar el pipeline.

Los scripts Slurm/bash usan `REPROCESS_POLICY=in_place` (default): eliminan el output previo y lo vuelven a generar.

Para **no** borrar y solo completar faltantes:

```bash
export REPROCESS_POLICY=skip_existing
```

## Ejemplos manuales (leftraru)

### Reclasificar solo r2 en `classification_20260618`

```bash
BASE=/home/flepin/classification_20260618
rm -f ${BASE}/b14_chile_r2_*_classified.tif
rm -f ${BASE}/filtering_work/classified_filtered/b14_chile_r2_*

export REGIONS=r2
cp classification/cluster_paths.classify.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
sbatch --export=ALL classification/run_classify_chile_slurm.sh
```

### Refiltrar solo r2 (clasificados ya en `BASE`)

```bash
BASE=/home/flepin/classification_20260618
rm -f ${BASE}/filtering_work/classified_filtered/b14_chile_r2_*
# opcional: intermedios
rm -f ${BASE}/filtering_work/classified_temporal/b14_chile_r2_* 2>/dev/null
rm -f ${BASE}/filtering_work/classified_filled/b14_chile_r2_* 2>/dev/null
rm -f ${BASE}/filtering_work/classified_lulc/b14_chile_r2_* 2>/dev/null

cp filtering/cluster_paths.20260618.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
export FILTER_NAME_CONTAINS=r2
export STEPS=filter
bash filtering/run_filtering_pipeline.sh
```

### Carpeta nueva (solo si lo pides)

```bash
export OUTPUT_DIR=/home/flepin/classification_20260620_experiment
export WORK_ROOT=/home/flepin/classification_20260620_experiment/filtering_work
```
