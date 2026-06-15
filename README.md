# chile-fire

This repository contains the scripts for mapping burned areas in Chile as part of the MapBiomas Fire Collection and Project.

## Repository layout

- `classification/` — Training and inference pipeline for the burned-area neural network model, including local and Slurm launchers. See [classification/README.md](classification/README.md).
- `filtering/` — Post-classification utilities: mask building and spatial filtering. See [filtering/README.md](filtering/README.md).
- `vectorize/` — Auxiliary pipeline: per-tile and **national** raster → polygon after filtering. See [vectorize/README.md](vectorize/README.md).
- `validation/` — Reference fire-scar layers: equal-area reprojection (vector and raster), yearly split/dissolve, area plots, polygon filters, and intersections with classified polygons. See [validation/README.md](validation/README.md).
- `test/` — Optional probes (e.g. Google Cloud Storage permissions). Not part of the core pipeline.
- `lib/` — Reusable Python helpers (tile metadata, raster → vector). See [lib/README.md](lib/README.md).
- `utilities/` — Auxiliary tooling: GEE downloads, tile listing, mosaicking and metadata inspection. See [utilities/README.md](utilities/README.md).
- `collection_010/` — Legacy assets and notebooks from Collection 0.1.0.

## Documentation

- [Classification pipeline](classification/README.md): how to train the burned-area model and how to run inference on yearly mosaics.
- [Filtering](filtering/README.md): how to clean classified rasters (`filtering/cluster_paths.env`, then `bash filtering/run_filtering_pipeline.sh`).
- [Vectorize](vectorize/README.md): polygonize filtered rasters per tile or Chile-wide by year (`vectorize/cluster_paths.env`). National flow: merge → sieve (min connected pixels) → polygonize → group events within 200 m.
- [Validation](validation/README.md): preparing reference scars, aligning CRS with classified products, and intersection exports.
- [Lib](lib/README.md): reusable Python helpers (used by `vectorize/` and other pipelines).
- [Utilities](utilities/README.md): shared helpers used across the pipeline.
- [Test utilities](test/README.md): optional checks (e.g. cloud storage access).