# NLHPC — vectorization / vectorización

Post-classification / post-filtering auxiliary pipeline. Methods: [README.md](README.md) · [README.es.md](README.es.md).

**Recommended:** login node — see [LOCAL.md](LOCAL.md). SLURM wrappers are optional for heavier resource needs.


## Checklist

- [ ] Clasificación y filtrado ya ejecutados (`classified_filtered/` existe)
- [ ] Repo clonado en `~/fire`
- [ ] `vectorize/cluster_paths.env` creado desde `cluster_paths.20260619.env.leftraru`
- [ ] `filtering/cluster_paths.env` creado desde `cluster_paths.20260619.env.leftraru`
- [ ] `geopandas` instalado en el env (`conda install -c conda-forge geopandas`)

## Configuración

```bash
cd ~/fire
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env
```

## Ejecutar (nodo login)

```bash
cd ~/fire
conda activate mb_fuego
source vectorize/cluster_paths.env

# 1) Polygonize por tesela
bash vectorize/run_vectorize_pipeline.sh

# 2) Filtro de área: >= 20 ha → histogramas → umbrales → filter (p25 default)
bash filtering/run_polygon_area_pipeline.sh

# 3) Vectorización nacional (opcional)
bash vectorize/run_vectorize_national_pipeline.sh

# O los tres en secuencia:
bash vectorize/run_post_filter_pipeline.sh
```

## Opcional: SLURM

Si el login queda saturado o quieres más CPUs:

```bash
mkdir -p ~/logs
sbatch vectorize/run_vectorize_pipeline_slurm.sh
sbatch vectorize/run_vectorize_national_pipeline_slurm.sh
```

Logs: `~/logs/fire_vectorize_<JOBID>.out` / `.err`

## Flujo completo

```text
1. classification/   →  sbatch run_classify_chile_slurm.sh
2. filtering/        →  bash run_filtering_pipeline.sh  (login)
3. vectorize/        →  bash run_vectorize_pipeline.sh  (login)
4. filtering §5      →  bash run_polygon_area_pipeline.sh (login)
5. vectorize/        →  bash run_vectorize_national_pipeline.sh (login, opcional)
```
