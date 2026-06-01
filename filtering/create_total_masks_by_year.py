#!/usr/bin/env python3
"""Create yearly total masks by OR-ing accumulated masks with per-year thematic masks."""

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mascara_total_<year>.tif as OR of accumulated masks and yearly "
            "rio_lago, infraestructura, agricultura, and pastura masks."
        )
    )
    parser.add_argument(
        "--masks-dir",
        default="/mnt/e/mapbiomas/fire/lulc_2025/mascaras_acumuladas",
        help=(
            "Legacy single directory: accumulated files, yearly masks, and totals live "
            "here (used when --accumulated-dir is not set)."
        ),
    )
    parser.add_argument(
        "--mascaras-root",
        type=Path,
        default=None,
        help=(
            "Layout with subfolders: read accumulated masks from <root>/acumuladas, "
            "yearly masks from <root>/by_year, write mascara_total_<year>.tif to "
            "<root>/totales (unless paths are overridden)."
        ),
    )
    parser.add_argument(
        "--accumulated-dir",
        type=Path,
        default=None,
        help=(
            "Directory with mascara_*_acumulado.tif files. "
            "Required for split layout if --mascaras-root is omitted."
        ),
    )
    parser.add_argument(
        "--yearly-dir",
        type=Path,
        default=None,
        help="Directory with mascara_rio_lago_<year>.tif etc. (split layout).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where mascara_total_<year>.tif are written (split layout).",
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
    yearly_dir: Path,
    output_dir: Path,
    accumulated_union: np.ndarray,
    base_profile: dict,
) -> Path:
    """Compute OR of accumulated union and yearly thematic masks; write mascara_total_<year>.tif."""
    rio_path = yearly_dir / f"mascara_rio_lago_{year}.tif"
    infra_path = yearly_dir / f"mascara_infraestructura_{year}.tif"
    agr_path = yearly_dir / f"mascara_agricultura_{year}.tif"
    past_path = yearly_dir / f"mascara_pastura_{year}.tif"

    rio_mask = read_mask(rio_path) > 0
    infra_mask = read_mask(infra_path) > 0
    agr_mask = read_mask(agr_path) > 0
    past_mask = read_mask(past_path) > 0

    total_mask = np.logical_or.reduce(
        [accumulated_union, rio_mask, infra_mask, agr_mask, past_mask]
    ).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mascara_total_{year}.tif"
    profile = dict(base_profile)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(total_mask, 1)

    return output_path


def _resolve_directories(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Return (accumulated_dir, yearly_dir, output_dir)."""
    if args.mascaras_root is not None:
        root = Path(args.mascaras_root)
        acc = Path(args.accumulated_dir) if args.accumulated_dir else root / "acumuladas"
        yrl = Path(args.yearly_dir) if args.yearly_dir else root / "by_year"
        out = Path(args.output_dir) if args.output_dir else root / "totales"
        return acc, yrl, out

    if (
        args.accumulated_dir is not None
        or args.yearly_dir is not None
        or args.output_dir is not None
    ):
        missing = []
        if args.accumulated_dir is None:
            missing.append("--accumulated-dir")
        if args.yearly_dir is None:
            missing.append("--yearly-dir")
        if args.output_dir is None:
            missing.append("--output-dir")
        if missing:
            raise ValueError(
                "For a split layout, pass --mascaras-root or all of "
                f"{', '.join(['--accumulated-dir', '--yearly-dir', '--output-dir'])}. "
                f"Missing: {', '.join(missing)}"
            )
        return Path(args.accumulated_dir), Path(args.yearly_dir), Path(args.output_dir)

    single = Path(args.masks_dir)
    return single, single, single


def main() -> int:
    args = parse_args()
    accumulated_dir, yearly_dir, output_dir = _resolve_directories(args)

    if not accumulated_dir.is_dir():
        raise FileNotFoundError(
            f"Accumulated masks directory not found: {accumulated_dir}"
        )
    if not yearly_dir.is_dir():
        raise FileNotFoundError(f"Yearly masks directory not found: {yearly_dir}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    print(f"[INFO] accumulated-dir: {accumulated_dir}")
    print(f"[INFO] yearly-dir:     {yearly_dir}")
    print(f"[INFO] output-dir:     {output_dir}")

    accumulated_arrays = []
    base_profile = None
    for name in ACCUMULATED_MASK_NAMES:
        path = accumulated_dir / name
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
            out = write_total_for_year(
                year, yearly_dir, output_dir, accumulated_union, base_profile
            )
            print(f"[INFO] Saved: {out}")
    else:
        print(f"[INFO] Parallel years with {n_workers} worker thread(s) ({n_years} year(s), {cpus} CPU(s))")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    write_total_for_year,
                    year,
                    yearly_dir,
                    output_dir,
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
