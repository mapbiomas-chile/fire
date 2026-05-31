#!/usr/bin/env python3
"""Create accumulated binary masks (OR across all bands or yearly mosaics)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

_FILTERING_DIR = Path(__file__).resolve().parent
if str(_FILTERING_DIR) not in sys.path:
    sys.path.insert(0, str(_FILTERING_DIR))

from lulc_year_from_name import year_from_lulc_path

CLASS_SPECS = [
    (29, "mascara_alfloramiento_rocoso_acumulado.tif"),
    (23, "mascara_arena_playa_duna_acumulado.tif"),
    (61, "mascara_salar_acumulado.tif"),
    (34, "mascara_hielo_nieve_acumulado.tif"),
    (25, "mascara_otra_area_sin_vegetacion_acumulado.tif"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one binary mask per class using OR across years. "
            "Input is either a multi-band stack (--input-tif) or yearly mosaics "
            "(--yearly-dir, e.g. lulc_2013.tif from mosaic_gee_lulc_tiles.py)."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-tif", help="Multi-band LULC stack (bands = years).")
    src.add_argument(
        "--yearly-dir",
        help="Directory with one LULC GeoTIFF per year (lulc_<year>.tif).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output mask TIFFs will be written.",
    )
    parser.add_argument(
        "--yearly-pattern",
        default="lulc_*.tif",
        help="Glob under --yearly-dir (default: lulc_*.tif).",
    )
    parser.add_argument("--from-year", type=int, default=2013)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Processing chunk size in pixels (default: 2048).",
    )
    return parser.parse_args()


def iter_windows(height: int, width: int, chunk_size: int):
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _write_profile(src_profile: dict) -> dict:
    profile = src_profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
    )
    return profile


def from_multiband_stack(input_path: Path, output_dir: Path, chunk_size: int) -> None:
    with rasterio.open(input_path) as src:
        if src.count < 1:
            raise ValueError("Input raster has no bands.")

        profile = _write_profile(src.profile)
        outputs = {}
        try:
            for _, filename in CLASS_SPECS:
                outputs[filename] = rasterio.open(output_dir / filename, "w", **profile)

            for window in iter_windows(src.height, src.width, chunk_size):
                block = src.read(window=window)
                for class_value, filename in CLASS_SPECS:
                    mask = np.any(block == class_value, axis=0).astype(np.uint8)
                    outputs[filename].write(mask, 1, window=window)
        finally:
            for dst in outputs.values():
                dst.close()


def _collect_yearly_paths(
    yearly_dir: Path,
    pattern: str,
    from_year: int,
    to_year: int,
) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for path in sorted(yearly_dir.glob(pattern)):
        year = year_from_lulc_path(path)
        if year is None:
            print(f"[WARN] Skip (no year): {path.name}")
            continue
        if from_year <= year <= to_year:
            paths.append((year, path))
    if not paths:
        raise RuntimeError(
            f"No yearly LULC files in {yearly_dir} matching {pattern!r} "
            f"for years {from_year}-{to_year}"
        )
    return paths


def from_yearly_dir(
    yearly_dir: Path,
    pattern: str,
    from_year: int,
    to_year: int,
    output_dir: Path,
    chunk_size: int,
) -> None:
    year_paths = _collect_yearly_paths(yearly_dir, pattern, from_year, to_year)
    years = sorted({y for y, _ in year_paths})
    print(f"[INFO] Accumulating over years: {years}")

    with rasterio.open(year_paths[0][1]) as ref:
        profile = _write_profile(ref.profile)
        height, width = ref.height, ref.width

    outputs = {}
    try:
        for _, filename in CLASS_SPECS:
            outputs[filename] = rasterio.open(output_dir / filename, "w", **profile)

        for window in iter_windows(height, width, chunk_size):
            class_masks = {
                filename: np.zeros((window.height, window.width), dtype=bool)
                for _, filename in CLASS_SPECS
            }
            for _, path in year_paths:
                with rasterio.open(path) as src:
                    if src.height != height or src.width != width:
                        raise ValueError(
                            f"Grid mismatch: {path.name} vs {year_paths[0][1].name}"
                        )
                    data = src.read(1, window=window)
                for class_value, filename in CLASS_SPECS:
                    class_masks[filename] |= data == class_value

            for _, filename in CLASS_SPECS:
                outputs[filename].write(class_masks[filename].astype(np.uint8), 1, window=window)
    finally:
        for dst in outputs.values():
            dst.close()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_tif:
        input_path = Path(args.input_tif)
        if not input_path.exists():
            raise FileNotFoundError(f"Input raster not found: {input_path}")
        from_multiband_stack(input_path, output_dir, args.chunk_size)
    else:
        yearly_dir = Path(args.yearly_dir)
        if not yearly_dir.is_dir():
            raise FileNotFoundError(f"Yearly LULC directory not found: {yearly_dir}")
        from_yearly_dir(
            yearly_dir,
            args.yearly_pattern,
            args.from_year,
            args.to_year,
            output_dir,
            args.chunk_size,
        )

    for _, filename in CLASS_SPECS:
        print(f"[INFO] Saved: {output_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
