# NLHPC — vectorización con SLURM

Pipeline auxiliar que corre **después** de clasificación y filtrado. Detalle: [README.md](README.md).

## Checklist

- [ ] Clasificación y filtrado ya ejecutados (`classified_filtered/` existe)
- [ ] Repo clonado en `~/fire`
- [ ] `vectorize/cluster_paths.env` creado desde `cluster_paths.env.example`
- [ ] `PYTHON` y `WORK_ROOT` (o `VECTORIZE_INPUT_DIR`) editados
- [ ] `geopandas` instalado en el env (`conda install -c conda-forge geopandas`)
- [ ] `~/logs` existe

## Configuración

```bash
cd ~/fire
cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
nano vectorize/cluster_paths.env
```

## Ejecutar

```bash
cd ~/fire
sbatch vectorize/run_vectorize_pipeline_slurm.sh

# Override rutas:
sbatch vectorize/run_vectorize_pipeline_slurm.sh \
  /home/USER/classification_YYYYMMDD/filtering_work/classified_filtered \
  /home/USER/classification_YYYYMMDD/filtering_work/polygons
```

Logs: `~/logs/fire_vectorize_<JOBID>.out` / `.err`

## Orden del flujo completo

```text
1. classification/   →  clasificados en bruto
2. filtering/        →  bash filtering/run_filtering_pipeline.sh
3. vectorize/        →  sbatch vectorize/run_vectorize_pipeline_slurm.sh
```
