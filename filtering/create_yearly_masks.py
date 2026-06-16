#!/usr/bin/env python3
"""
Create yearly binary masks for selected land-cover classes (time-varying).

For each filter year Y, a pixel is marked only if the class is stable across a
4-year LULC window (default): [Y, Y+1, Y+2, Y+3], or [Y-3..Y] near the stack end.
The mask mascara_<class>_Y.tif applies only when filtering burn rasters of year Y.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.lulc_stability import stability_window_years, year_to_band  # noqa: E402

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
            "Generate yearly 0/1 masks for rio_lago (33), infraestructura (24), "
            "agricultura (15), pastura (18). "
            "By default a pixel is marked only if the class is stable across a "
            "forward 4-year LULC window anchored at the filter year."
        )
    )
    parser.add_argument("--input-tif", required=True, help="Input multi-band raster.")
    parser.add_argument(
        "--output-dir",
        default="/mnt/e/mapbiomas/fire/lulc_2025/mascaras_acumuladas",
        help="Output directory for yearly mask TIFFs.",
    )
    parser.add_argument(
        "--start-year-in-band-1",
        type=int,
        default=2000,
        help="Year represented by band 1 (default: 2000).",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2013,
        help="First filter year to export (default: 2013).",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2024,
        help="Last filter year to export, inclusive (default: 2024).",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=4,
        help=(
            "Number of consecutive LULC years that must match the class "
            "(default: 4). Use 1 for legacy single-year masks."
        ),
    )
    parser.add_argument(
        "--agriculture-stability-window",
        type=int,
        default=None,
        help=(
            "Override stability window for agricultura (15) only. "
            "Use 1 for strict single-year agriculture masking."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (one filter year per process).",
    )
    return parser.parse_args()


def _read_window_stack(
    src: rasterio.DatasetReader,
    window_years: list[int],
    start_year_in_band_1: int,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for calendar_year in window_years:
        band = year_to_band(calendar_year, start_year_in_band_1)
        if band < 1 or band > src.count:
            raise ValueError(
                f"Year {calendar_year} maps to band {band}, outside raster range 1..{src.count}"
            )
        arrays.append(src.read(band))
    return arrays


def _stable_class_mask(window_arrays: list[np.ndarray], class_value: int) -> np.ndarray:
    stable = np.ones(window_arrays[0].shape, dtype=bool)
    for data in window_arrays:
        stable &= data == class_value
    return stable.astype(np.uint8)


def _window_years(
    filter_year: int,
    stability_window: int,
    *,
    lulc_min_year: int,
    lulc_max_year: int,
) -> list[int]:
    if stability_window == 1:
        return [filter_year]
    return stability_window_years(
        filter_year,
        lulc_min_year=lulc_min_year,
        lulc_max_year=lulc_max_year,
        window_size=stability_window,
    )


def _process_one_year(
    input_path_str: str,
    output_dir_str: str,
    filter_year: int,
    start_year_in_band_1: int,
    lulc_min_year: int,
    lulc_max_year: int,
    stability_window: int,
    agriculture_stability_window: int | None,
) -> tuple[int, int, dict[str, list[int]]]:
    """Write four mask GeoTIFFs for one filter year. Returns (year, files_written, windows)."""
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_by_class: dict[str, list[int]] = {}

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        n = 0
        for class_name, class_value in TARGET_CLASSES:
            class_window = stability_window
            if class_name == "agricultura" and agriculture_stability_window is not None:
                class_window = agriculture_stability_window
            window_years = _window_years(
                filter_year,
                class_window,
                lulc_min_year=lulc_min_year,
                lulc_max_year=lulc_max_year,
            )
            windows_by_class[class_name] = window_years
            window_arrays = _read_window_stack(src, window_years, start_year_in_band_1)
            mask = _stable_class_mask(window_arrays, class_value)
            output_path = output_dir / f"mascara_{class_name}_{filter_year}.tif"
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(mask, 1)
            n += 1

    return filter_year, n, windows_by_class


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_tif)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.stability_window < 1:
        raise ValueError("--stability-window must be >= 1")
    if args.agriculture_stability_window is not None and args.agriculture_stability_window < 1:
        raise ValueError("--agriculture-stability-window must be >= 1")

    with rasterio.open(input_path) as src:
        lulc_min_year = args.start_year_in_band_1
        lulc_max_year = args.start_year_in_band_1 + src.count - 1

    output_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.from_year, args.to_year + 1))
    in_str = str(input_path.resolve())
    out_str = str(output_dir.resolve())

    ag_note = (
        f"agriculture_stability_window={args.agriculture_stability_window}"
        if args.agriculture_stability_window is not None
        else "agriculture uses --stability-window"
    )
    print(
        f"[INFO] LULC stack years: {lulc_min_year}-{lulc_max_year} | "
        f"filter years: {args.from_year}-{args.to_year} | "
        f"stability_window={args.stability_window} | {ag_note}",
        flush=True,
    )

    def _report(year: int, n: int, windows: dict[str, list[int]]) -> None:
        ag_w = windows.get("agricultura", [])
        print(
            f"[INFO] Filter year {year}: wrote {n} mask TIFFs "
            f"(ag window {ag_w[0]}-{ag_w[-1] if ag_w else 'n/a'})",
            flush=True,
        )

    task_kwargs = (
        lulc_min_year,
        lulc_max_year,
        args.stability_window,
        args.agriculture_stability_window,
    )

    if args.workers <= 1:
        for year in years:
            y, n, windows = _process_one_year(
                in_str,
                out_str,
                year,
                args.start_year_in_band_1,
                *task_kwargs,
            )
            _report(y, n, windows)
        return 0

    print(
        f"[INFO] Parallel filter years with {args.workers} worker process(es), "
        f"{len(years)} year(s)",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _process_one_year,
                in_str,
                out_str,
                year,
                args.start_year_in_band_1,
                *task_kwargs,
            ): year
            for year in years
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                y, n, windows = fut.result()
            except Exception as e:
                raise RuntimeError(f"Failed filter year {year}") from e
            _report(y, n, windows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
