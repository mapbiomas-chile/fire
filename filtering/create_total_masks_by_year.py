#!/usr/bin/env python3
"""Create yearly total masks combining accumulated and yearly masks."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Salida de create_agriculture_intersection_mask.py (constante en el tiempo; se une a cada año).
AGR_INTERSECTION_MASK_NAME = "mascara_agricultura_interseccion_todos_anos.tif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mascara_total_<year>.tif as OR of accumulated masks, optional "
            "agriculture-intersection mask, and yearly rio_lago/infraestructura masks."
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
    parser.add_argument(
        "--agriculture-intersection-mask",
        type=Path,
        default=None,
        help=(
            "Optional 0/1 mask (same grid as other masks) OR'ed into every yearly total. "
            f"If omitted, uses <masks-dir>/{AGR_INTERSECTION_MASK_NAME} when that file exists."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel workers (one year per task). Default: min(year count, CPU cores). "
            "Uses threads so the accumulated union is not copied per worker."
        ),
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    with rasterio.open(path) as src:
        data = src.read(1)
    return data


def write_total_for_year(
    year: int,
    masks_dir: Path,
    accumulated_union: np.ndarray,
    base_profile: dict,
) -> Path:
    """Compute OR(accumulated_union, rio_year, infra_year) and write mascara_total_<year>.tif."""
    rio_path = masks_dir / f"mascara_rio_lago_{year}.tif"
    infra_path = masks_dir / f"mascara_infraestructura_{year}.tif"

    rio_mask = read_mask(rio_path) > 0
    infra_mask = read_mask(infra_path) > 0

    total_mask = np.logical_or.reduce([accumulated_union, rio_mask, infra_mask]).astype(
        np.uint8
    )

    output_path = masks_dir / f"mascara_total_{year}.tif"
    profile = dict(base_profile)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(total_mask, 1)

    return output_path


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

    agr_path = (
        args.agriculture_intersection_mask
        if args.agriculture_intersection_mask is not None
        else masks_dir / AGR_INTERSECTION_MASK_NAME
    )
    if agr_path.exists():
        with rasterio.open(agr_path) as src:
            accumulated_arrays.append(src.read(1) > 0)
        print(f"[INFO] Including in union: {agr_path}")
    elif args.agriculture_intersection_mask is not None:
        raise FileNotFoundError(f"Agriculture intersection mask not found: {agr_path}")
    else:
        print(
            f"[WARN] Agriculture intersection mask not found, skipping: {agr_path}"
        )

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

    years = list(range(args.from_year, args.to_year + 1))
    n_years = len(years)
    cpus = os.cpu_count() or 1
    if args.workers is not None:
        if args.workers < 1:
            raise ValueError("--workers must be >= 1")
        n_workers = args.workers
    else:
        n_workers = min(n_years, cpus)

    if n_workers <= 1:
        for year in years:
            out = write_total_for_year(year, masks_dir, accumulated_union, base_profile)
            print(f"[INFO] Saved: {out}")
    else:
        print(f"[INFO] Parallel years with {n_workers} worker thread(s) ({n_years} year(s), {cpus} CPU(s))")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    write_total_for_year,
                    year,
                    masks_dir,
                    accumulated_union,
                    base_profile,
                ): year
                for year in years
            }
            for fut in as_completed(futures):
                year = futures[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed processing year {year}") from e
                print(f"[INFO] Saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
