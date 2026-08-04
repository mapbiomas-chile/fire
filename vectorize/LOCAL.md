# Interactive execution (login node, no SLURM) / Ejecución interactiva (nodo login)

[Español + English commands] · Full methods: [README.md](README.md) · [README.es.md](README.es.md)

Vectorization and polygon-area filtering run on the **login node** by default (same as interactive raster filtering). SLURM wrappers (`*_slurm.sh`) remain available if more CPUs/RAM are required.

La vectorización y el filtro de área se ejecutan por defecto en el **nodo login**.

## 1. Configurar rutas (una vez)

```bash
cd ~/fire
cp vectorize/cluster_paths.20260619.env.leftraru vectorize/cluster_paths.env
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source vectorize/cluster_paths.env
```

## 2. Vectorizar por tesela

```bash
conda activate mb_fuego
bash vectorize/run_vectorize_pipeline.sh
```

`VECTORIZE_WORKERS=4` por defecto en el env 20260619. Si el login se pone lento, baja a `2`:

```bash
export VECTORIZE_WORKERS=2
bash vectorize/run_vectorize_pipeline.sh
```

## 3. Filtro de área en polígonos

```bash
bash filtering/run_polygon_area_pipeline.sh
```

## 4. Vectorización nacional (opcional)

```bash
bash vectorize/run_vectorize_national_pipeline.sh
```

## 5. Todo en secuencia

```bash
bash vectorize/run_post_filter_pipeline.sh
```

## Verificación

```bash
export WORK_ROOT="/home/flepin/classification_20260619/filtering_work"
ls ${WORK_ROOT}/polygons/*.gpkg | wc -l
ls ${WORK_ROOT}/polygons_min20ha/*.gpkg 2>/dev/null | wc -l
```
