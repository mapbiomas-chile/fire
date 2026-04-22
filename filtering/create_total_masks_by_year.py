#!/usr/bin/env python3
"""Create yearly total masks combining accumulated and yearly masks."""

import argparse
from pathlib import Path

import numpy as np
import rasterio


ACCUMULATED_MASK_NAMES = [
    "mascara_alfloramiento_rocoso_acumulado.tif",
    "mascara_arena_playa_duna_acumulado.tif",
    "mascara_salar_acumulado.tif",
    "mascara_hielo_nieve_acumulado.tif",
    "mascara_otra_area_sin_vegetacion_acumulado.tif",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mascara_total_<year>.tif as OR of accumulated masks "
            "and yearly rio_lago/infraestructura masks."
        )
    )
    parser.add_argument(
        "--masks-dir",
        default="/mnt/e/mapbiomas/fire/lulc_2025/mascaras_acumuladas",
        help="Directory containing accumulated and yearly mask TIFFs.",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2013,
        help="First year to process (default: 2013).",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2024,
        help="Last year to process, inclusive (default: 2024).",
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    with rasterio.open(path) as src:
        data = src.read(1)
    return data


def main() -> int:
    args = parse_args()
    masks_dir = Path(args.masks_dir)

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    accumulated_arrays = []
    base_profile = None
    for name in ACCUMULATED_MASK_NAMES:
        path = masks_dir / name
        with rasterio.open(path) as src:
            data = src.read(1)
            if base_profile is None:
                base_profile = src.profile.copy()
            accumulated_arrays.append(data > 0)

    accumulated_union = np.logical_or.reduce(accumulated_arrays)

    if base_profile is None:
        raise RuntimeError("Could not read accumulated masks profile.")

    base_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
    )

    for year in range(args.from_year, args.to_year + 1):
        rio_path = masks_dir / f"mascara_rio_lago_{year}.tif"
        infra_path = masks_dir / f"mascara_infraestructura_{year}.tif"

        rio_mask = read_mask(rio_path) > 0
        infra_mask = read_mask(infra_path) > 0

        total_mask = np.logical_or.reduce([accumulated_union, rio_mask, infra_mask]).astype(
            np.uint8
        )

        output_path = masks_dir / f"mascara_total_{year}.tif"
        with rasterio.open(output_path, "w", **base_profile) as dst:
            dst.write(total_mask, 1)

        print(f"[INFO] Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
