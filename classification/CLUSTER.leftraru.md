# NLHPC leftraru — rama feature/model_modification

## Conectar la rama en el cluster

```bash
cd ~/fire
git fetch origin
git checkout feature/model_modification
git pull origin feature/model_modification
```

## Reglas de versión de modelo (producción actual)

| Región | Años 2013–2018 | Años 2019–2025 |
|--------|----------------|----------------|
| r2     | v1             | v1             |
| r1, r4, r6 | v1         | v2             |

`run_classify_region_slurm.sh` aplica estas reglas cuando `AUTO_MODEL_VERSION_BY_YEAR=1` (o `MODEL_VERSION` está vacío).

## Clasificación (modelos existentes)

```bash
cp classification/cluster_paths.leftraru classification/cluster_paths.env
# editar REGION si hace falta
source classification/cluster_paths.env
sbatch classification/run_classify_region_slurm.sh
```

Salida por defecto: `/home/flepin/classification_20260616`

## Entrenamiento (modelos mejorados)

```bash
cp classification/cluster_paths.model_modification.env.leftraru classification/cluster_paths.env
source classification/cluster_paths.env
# ajustar TRAIN_REGION, TRAIN_VERSION, MODELS_DIR
sbatch classification/run_train_fire_model_slurm.sh
```

Modelos nuevos se guardan en `/home/flepin/models_col1_model_mod` (separado de producción).

Benchmark XGBoost:

```bash
export TRAIN_BACKEND=xgboost
sbatch classification/run_train_fire_model_slurm.sh
```

## Notas

- Los archivos `*.env.leftraru` se commitean en esta rama; `cluster_paths.env` local sigue en `.gitignore`.
- Checkpoints legacy siguen funcionando sin `DECISION_THRESHOLD`.
- Para forzar una versión fija: `export MODEL_VERSION=v1` y `export AUTO_MODEL_VERSION_BY_YEAR=0`.
