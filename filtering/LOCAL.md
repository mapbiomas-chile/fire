# Interactive execution (no SLURM) / Ejecución interactiva (sin SLURM)

[Español + English commands] · Full methods: [README.md](README.md) · [README.es.md](README.es.md)

## 1. Configurar rutas (una vez por máquina/usuario)

```bash
cd /path/to/fire
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
nano filtering/cluster_paths.env
```

Editar al menos:

| Variable | Ejemplo |
|----------|---------|
| `PYTHON` | `/home/USER/.conda/envs/mb_fuego/bin/python` |
| `LULC_STACK` | GeoTIFF multibanda MapBiomas |
| `CLASSIFIED_DIR` | Carpeta con clasificados en bruto |
| `WORK_ROOT` | Carpeta donde quieres las salidas |

Ejemplo (leftraru — producción 20260619):

```bash
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
# o exportar manualmente:
export CLASSIFIED_DIR="/home/flepin/classification_20260619"
export WORK_ROOT="/home/flepin/classification_20260619/filtering_work"
export STEPS="all"
```

## 2. Ejecutar

```bash
conda activate mb_fuego   # o el env que definiste en PYTHON
source filtering/cluster_paths.env
bash filtering/run_filtering_pipeline.sh
```

También puedes exportar variables en la misma línea sin archivo:

```bash
export PYTHON=... CLASSIFIED_DIR=... WORK_ROOT=... LULC_STACK=... STEPS=filter
bash filtering/run_filtering_pipeline.sh
```

## 3. Pasos parciales

```bash
export STEPS="masks_accumulated,masks_yearly,masks_total"   # solo máscaras
export STEPS="filter"                                       # temporal + fill + LULC
```

## 4. Memoria

En nodo login: `export WORKERS=2` en `cluster_paths.env`.  
Más RAM: [CLUSTER.md](CLUSTER.md) (`sbatch`).
