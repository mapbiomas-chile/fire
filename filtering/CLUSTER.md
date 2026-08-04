# NLHPC — SLURM execution / ejecución con SLURM

[Español + English commands] · Full methods: [README.md](README.md) · [README.es.md](README.es.md)

## Checklist

- [ ] Repo clonado en tu `$HOME` (o ruta en `REPO_ROOT`)
- [ ] `filtering/cluster_paths.env` creado desde `cluster_paths.env.example`
- [ ] `PYTHON`, `LULC_STACK`, `CLASSIFIED_DIR`, `WORK_ROOT` editados con **tus** rutas
- [ ] `LULC_STACK` es GeoTIFF (`.tif`), no `.vrt`
- [ ] Correo en `#SBATCH --mail-user` de `run_filtering_pipeline_slurm.sh` (opcional)
- [ ] `~/logs` existe

## Configuración

**NLHPC leftraru — producción (classification_20260619):**

```bash
cd ~/fire
cp filtering/cluster_paths.20260619.env.leftraru filtering/cluster_paths.env
source filtering/cluster_paths.env
```

**Otro entorno** — plantilla genérica:

```bash
cd ~/fire
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
nano filtering/cluster_paths.env
```

## Ejecutar

```bash
cd ~/fire
sbatch filtering/run_filtering_pipeline_slurm.sh

# Solo filtrado (máscaras ya generadas):
sbatch filtering/run_filtering_pipeline_slurm.sh \
  /home/flepin/classification_20260619 \
  /home/flepin/classification_20260619/filtering_work filter
```

Logs: `~/logs/fire_class_filter_<JOBID>.out` / `.err`

## Pasos del pipeline

| `STEPS` | Qué hace |
|---------|----------|
| `masks_accumulated` | Máscaras acumuladas |
| `masks_yearly` | Máscaras anuales |
| `masks_total` | `mascara_total_<year>.tif` |
| `filter` | Temporal + hole fill + LULC (+ optional steps) |

`STEPS=all` runs all configured mask and filter stages.

**Next step (vectorize + area filter):** [`../vectorize/CLUSTER.md`](../vectorize/CLUSTER.md).
