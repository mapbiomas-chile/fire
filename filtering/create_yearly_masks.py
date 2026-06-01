#!/usr/bin/env python3
"""Create yearly binary masks for selected land-cover classes (time-varying)."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

_FILTERING_DIR = Path(__file__).resolve().parent
if str(_FILTERING_DIR) not in sys.path:
    sys.path.insert(0, str(_FILTERING_DIR))

from gtiff_io import open_mask_writer
from lulc_year_from_name import year_from_lulc_path

# (output stem, class id) — filenames: mascara_<stem>_<year>.tif
TARGET_CLASSES = [
    ("rio_lago", 33),
    ("infraestructura", 24),
    ("agricultura", 15),
    ("pastura", 18),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate yearly 0/1 masks: rio_lago (33), infraestructura (24), "
            "agricultura (15), pastura (18). Input: multi-band stack (--input-tif) "
            "or yearly mosaics (--yearly-dir)."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-tif", help="Multi-band LULC stack.")
    src.add_argument(
        "--yearly-dir",
        help="Directory with lulc_<year>.tif (from mosaic_gee_lulc_tiles.py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for yearly mask TIFFs.",
    )
    parser.add_argument(
        "--yearly-pattern",
        default="lulc_*.tif",
        help="Glob under --yearly-dir (default: lulc_*.tif).",
    )
    parser.add_argument(
        "--start-year-in-band-1",
        type=int,
        default=2000,
        help="With --input-tif: year of band 1 (default: 2000).",
    )
    parser.add_argument("--from-year", type=int, default=2013)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (one year per process).",
    )
    return parser.parse_args()


def _write_masks_for_year(
    data: np.ndarray,
    profile: dict,
    output_dir: Path,
    year: int,
) -> int:
    n = 0
    for class_name, class_value in TARGET_CLASSES:
        mask = (data == class_value).astype(np.uint8)
        output_path = output_dir / f"mascara_{class_name}_{year}.tif"
        with open_mask_writer(output_path, profile) as dst:
            dst.write(mask, 1)
        n += 1
    return n


def _process_one_year_stack(
    input_path_str: str,
    output_dir_str: str,
    year: int,
    start_year_in_band_1: int,
) -> tuple[int, int]:
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        band = year - start_year_in_band_1 + 1
        if band < 1 or band > src.count:
            raise ValueError(
                f"Year {year} maps to band {band}, outside raster range 1..{src.count}"
            )
        profile = src.profile.copy()
        data = src.read(band)

    n = _write_masks_for_year(data, profile, output_dir, year)
    return year, n


def _process_one_year_file(
    lulc_path_str: str,
    output_dir_str: str,
    year: int,
) -> tuple[int, int]:
    lulc_path = Path(lulc_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(lulc_path) as src:
        profile = src.profile.copy()
        data = src.read(1)

    n = _write_masks_for_year(data, profile, output_dir, year)
    return year, n


def _years_from_yearly_dir(
    yearly_dir: Path,
    pattern: str,
    from_year: int,
    to_year: int,
) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in sorted(yearly_dir.glob(pattern)):
        year = year_from_lulc_path(path)
        if year is None:
            print(f"[WARN] Skip (no year): {path.name}")
            continue
        if from_year <= year <= to_year:
            out.append((year, path))
    if not out:
        raise RuntimeError(
            f"No files in {yearly_dir} matching {pattern!r} for {from_year}-{to_year}"
        )
    return out


def _run_parallel_yearly_file(tasks: list[tuple[str, str, int]], workers: int) -> None:
    if workers <= 1:
        for path_str, out_str, year in tasks:
            y, n = _process_one_year_file(path_str, out_str, year)
            print(f"[INFO] Year {y}: wrote {n} mask TIFFs")
        return
    print(f"[INFO] Parallel yearly-dir with {workers} worker(s), {len(tasks)} year(s)")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one_year_file, p, o, y): y for p, o, y in tasks
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                y, n = fut.result()
                print(f"[INFO] Year {y}: wrote {n} mask TIFFs")
            except Exception as e:
                raise RuntimeError(f"Failed year {year}") from e


def _run_parallel_stack(tasks: list[tuple[str, str, int, int]], workers: int) -> None:
    if workers <= 1:
        for in_str, out_str, year, start in tasks:
            y, n = _process_one_year_stack(in_str, out_str, year, start)
            print(f"[INFO] Year {y}: wrote {n} mask TIFFs")
        return
    print(f"[INFO] Parallel stack with {workers} worker(s), {len(tasks)} year(s)")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one_year_stack, i, o, y, s): y
            for i, o, y, s in tasks
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                y, n = fut.result()
                print(f"[INFO] Year {y}: wrote {n} mask TIFFs")
            except Exception as e:
                raise RuntimeError(f"Failed year {year}") from e


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.yearly_dir:
        yearly_dir = Path(args.yearly_dir)
        if not yearly_dir.is_dir():
            raise FileNotFoundError(f"Yearly LULC directory not found: {yearly_dir}")
        year_files = _years_from_yearly_dir(
            yearly_dir, args.yearly_pattern, args.from_year, args.to_year
        )
        tasks = [(str(p), str(output_dir.resolve()), y) for y, p in year_files]
        _run_parallel_yearly_file(tasks, args.workers)
        return 0

    input_path = Path(args.input_tif)
    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    years = list(range(args.from_year, args.to_year + 1))
    in_str = str(input_path.resolve())
    out_str = str(output_dir.resolve())
    tasks = [(in_str, out_str, y, args.start_year_in_band_1) for y in years]
    _run_parallel_stack(tasks, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
