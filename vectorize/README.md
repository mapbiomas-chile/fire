# Vectorize pipeline (auxiliary)

**Post-filtering** step: convert binary burn rasters to polygons (GeoPackage).

This pipeline runs **after** classification and filtering. It does not replace those steps.

```text
classification/          →  raw *_classified.tif
        │
        ▼
filtering/               →  classified_filtered/  (temporal + fill + LULC)
        │
        ▼
vectorize/  (this)       →  polygons/  (*.gpkg per tile)
```

Core Python functions live in [`../lib/`](../lib/README.md). This folder is only orchestration and cluster config.

---

## Quick start (leftraru)

```bash
cd ~/fire

cp vectorize/cluster_paths.env.example vectorize/cluster_paths.env
nano vectorize/cluster_paths.env    # PYTHON, WORK_ROOT

source vectorize/cluster_paths.env
bash vectorize/run_vectorize_pipeline.sh
```

With SLURM:

```bash
cd ~/fire
mkdir -p ~/logs
sbatch vectorize/run_vectorize_pipeline_slurm.sh
```

See [CLUSTER.md](CLUSTER.md) for NLHPC details.

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYTHON` | Yes | — | Interpreter with `geopandas`, `rasterio` |
| `WORK_ROOT` | Yes* | — | Filtering work root |
| `VECTORIZE_INPUT_DIR` | No | `$WORK_ROOT/classified_filtered` | Post-filter rasters |
| `VECTORIZE_OUTPUT_DIR` | No | `$WORK_ROOT/polygons` | Output GeoPackages |
| `VECTORIZE_WORKERS` | No | `4` | Parallel workers |
| `VECTORIZE_MERGED_GPKG` | No | — | Optional single merged layer |
| `VECTORIZE_STATS_JSON` | No | `$WORK_ROOT/logs/vectorize_stats.json` | Run summary |

\*Or set `VECTORIZE_INPUT_DIR` explicitly without `WORK_ROOT`.

---

## Output

One `.gpkg` per input raster in `VECTORIZE_OUTPUT_DIR`, with attributes:

| Column | Description |
|--------|-------------|
| `year` | Calendar year from filename |
| `region` | Region id (`1`, `2`, `4`, `6`, …) |
| `source_file` | Source GeoTIFF name |
| `mask_value` | Polygonized pixel value (default `1`) |

---

## Files

| File | Role |
|------|------|
| `run_vectorize_pipeline.sh` | Main pipeline (interactive) |
| `run_vectorize_pipeline_slurm.sh` | SLURM wrapper for NLHPC |
| `cluster_paths.env.example` | Path template → copy to `cluster_paths.env` |

Implementation: [`lib/vectorize_filtered_classified.py`](../lib/vectorize_filtered_classified.py).

Legacy wrapper (same backend): `filtering/polygonize_mask_parallel.py`.

---

## Next steps after vectorization

- Area histograms / threshold: [filtering/README.md](../filtering/README.md) § 5
- Validation vs reference scars: [validation/README.md](../validation/README.md)
