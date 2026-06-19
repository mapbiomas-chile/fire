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

**NLHPC leftraru — producción (classification_20260619):**

```bash
cd ~/fire
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env
```

**Otro entorno** — plantilla genérica:

```bash
cd ~/fire
cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
nano vectorize/cluster_paths.env
```

## Ejecutar

```bash
cd ~/fire

# 1) Polygonize por tesela
sbatch vectorize/run_vectorize_pipeline_slurm.sh

# 2) Filtro de área: >= 20 ha → histogramas → umbrales → filter (p25 default)
bash filtering/run_polygon_area_pipeline.sh

# 3) Vectorización nacional: merge anual + sieve 112 px + agrupación 200 m
sbatch vectorize/run_vectorize_national_pipeline_slurm.sh

# O los tres en secuencia (nodo login):
bash vectorize/run_post_filter_pipeline.sh
```

Logs: `~/logs/fire_vectorize_<JOBID>.out` / `.err`

## Flujo completo

```text
1. classification/   →  sbatch run_classify_chile_slurm.sh
2. filtering/        →  bash run_filtering_pipeline.sh  (o sbatch)
3. vectorize/        →  sbatch run_vectorize_pipeline_slurm.sh
4. filtering §5      →  bash run_polygon_area_pipeline.sh
5. vectorize/        →  sbatch run_vectorize_national_pipeline_slurm.sh
```
