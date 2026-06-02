# Leftraru — ejecución interactiva (sin SLURM)

Guía mínima para correr el pipeline por SSH **sin** `sbatch`.  
Índice de scripts y flujo completo: [README.md](README.md).

## 1. Preparar

```bash
cd ~/fire
git fetch origin && git checkout feat/filtering_pipeline && git pull
conda activate mb_fuego
```

Opcional (sobreescribir rutas del `.sh`):

```bash
cp filtering/cluster_paths.env.example filtering/cluster_paths.env
# editar cluster_paths.env
source filtering/cluster_paths.env
```

## 2. Ejecutar

```bash
bash filtering/run_filtering_pipeline.sh
```

Sesión larga: `screen -S filter` antes de lanzar.

## 3. Pasos parciales

```bash
export STEPS="masks_accumulated,masks_yearly,masks_total"   # solo máscaras
export STEPS="filter"                                       # LULC + temporal
export STEPS="lulc_filter"                                  # solo LULC
export STEPS="temporal_first_burn"                          # solo temporal
```

## 4. Memoria en login node

Baja `WORKERS=2` o `1` en `cluster_paths.env`.  
Para más RAM: [CLUSTER.md](CLUSTER.md) (`sbatch`).
