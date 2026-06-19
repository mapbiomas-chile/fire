# Ejecución interactiva (nodo login, sin SLURM)

Guía mínima por SSH. Detalle del flujo: [README.md](README.md).

Vectorización y filtro de área se ejecutan en el **nodo login** (como el filtrado raster). Los scripts `*_slurm.sh` quedan como opción si necesitas más CPUs/RAM.

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
