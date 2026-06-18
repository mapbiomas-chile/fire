# NLHPC — ejecución con SLURM

Guía para la cola. Detalle del flujo: [README.md](README.md).

## Checklist

- [ ] Repo clonado en tu `$HOME` (o ruta en `REPO_ROOT`)
- [ ] `filtering/cluster_paths.env` creado desde `cluster_paths.env.example`
- [ ] `PYTHON`, `LULC_STACK`, `CLASSIFIED_DIR`, `WORK_ROOT` editados con **tus** rutas
- [ ] `LULC_STACK` es GeoTIFF (`.tif`), no `.vrt`
- [ ] Correo en `#SBATCH --mail-user` de `run_filtering_pipeline_slurm.sh` (opcional)
- [ ] `~/logs` existe

## Configuración

**NLHPC leftraru** (rutas del cluster, en la rama de trabajo):

```bash
cd ~/fire
cp filtering/cluster_paths.20260618.env.leftraru filtering/cluster_paths.env
# baseline sin ag-fill: filtering/cluster_paths.env.leftraru (ajustar rutas a 20260618)
# comparación anterior: filtering/cluster_paths.ag_strict.env.leftraru → classification_20260609
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
sbatch -t 02:00:00 filtering/run_filtering_pipeline_slurm.sh

# Solo filtrado (máscaras ya generadas); opcional: override rutas en la línea de comando:
sbatch filtering/run_filtering_pipeline_slurm.sh /home/flepin/classification_20260618 /home/flepin/classification_20260618/filtering_work filter
```

Logs: `~/logs/fire_class_filter_<JOBID>.out` / `.err`

## Pasos del pipeline

| `STEPS` | Qué hace |
|---------|----------|
| `masks_accumulated` | Máscaras acumuladas |
| `masks_yearly` | Máscaras anuales |
| `masks_total` | `mascara_total_<year>.tif` |
| `filter` | LULC + temporal |

`STEPS=all` ejecuta los cuatro.

**Siguiente paso (vectorización):** pipeline auxiliar en [`../vectorize/CLUSTER.md`](../vectorize/CLUSTER.md).  
Histogramas y umbral de polígonos: [README.md](README.md) § 5.
