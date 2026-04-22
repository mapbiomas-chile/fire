# chile-fire

This repository contains the scripts for mapping burned areas in Chile as part of the MapBiomas Fire Collection and Project.

## Repository layout

- `classification/` — Training and inference pipeline for the burned-area neural network model, including local and Slurm launchers. See [classification/README.md](classification/README.md).
- `filtering/` — Post-classification utilities: mask building, spatial filtering, polygonization and statistical summaries. See [filtering/README.md](filtering/README.md).
- `utilities/` — Auxiliary tooling: GEE downloads, tile listing, mosaicking and metadata inspection. See [utilities/README.md](utilities/README.md).
- `collection_010/` — Legacy assets and notebooks from Collection 0.1.0.

## Documentation

- [Classification pipeline](classification/README.md): how to train the burned-area model and how to run inference on yearly mosaics.
- [Filtering](filtering/README.md): how to clean classified rasters, polygonize them and compute statistics.
- [Utilities](utilities/README.md): shared helpers used across the pipeline.
