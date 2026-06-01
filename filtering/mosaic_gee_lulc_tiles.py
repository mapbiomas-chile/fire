#!/usr/bin/env python3
"""Mosaic GEE-exported LULC tiles into one GeoTIFF per calendar year.

Typical GEE layout (MapBiomas Chile)::

    lulc_chile_collection02_20130000000000-0000000000.tif
    lulc_chile_collection02_20130000065536-0000000000.tif
    lulc_chile_collection02_20130000131072-0000000000.tif

All tiles sharing the same four-digit year after ``collection02_`` are merged
with ``rasterio.merge`` into ``{output_dir}/lulc_{year}.tif``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.merge import merge

_FILTERING_DIR = Path(__file__).resolve().parent
if str(_FILTERING_DIR) not in sys.path:
    sys.path.insert(0, str(_FILTERING_DIR))

from gtiff_io import mask_gtiff_profile

YEAR_RE = re.compile(r"collection02[_-](\d{4})|_(\d{4})\d{8}-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge GEE LULC tiles into yearly mosaics.")
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory with per-tile LULC GeoTIFFs from GEE.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for lulc_<year>.tif mosaics.",
    )
    p.add_argument(
        "--pattern",
        default="lulc_chile_collection02_*.tif",
        help="Glob under input-dir (default: lulc_chile_collection02_*.tif).",
    )
    p.add_argument("--from-year", type=int, default=2013)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers (one year per process). Default 1: each worker loads "
            "full tiles into RAM; >2 often OOMs on 64GB nodes."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing lulc_<year>.tif outputs.",
    )
    return p.parse_args()


def extract_year(path: Path) -> int | None:
    m = YEAR_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def mosaic_one_year(
    year: int,
    tile_paths: list[str],
    output_dir_str: str,
    overwrite: bool,
) -> tuple[int, str]:
    output_dir = Path(output_dir_str)
    out_path = output_dir / f"lulc_{year}.tif"
    if out_path.exists() and not overwrite:
        return year, f"skip (exists): {out_path.name}"

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        ref_profile = datasets[0].profile.copy()
        ref_profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            dtype=ref_profile.get("dtype", mosaic.dtype),
        )
        profile = mask_gtiff_profile(ref_profile)
        profile["dtype"] = ref_profile["dtype"]
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic[0], 1)
    finally:
        for ds in datasets:
            ds.close()

    return year, f"ok ({len(tile_paths)} tile(s)): {out_path.name}"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    by_year: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob(args.pattern)):
        year = extract_year(path)
        if year is None:
            print(f"[WARN] Skip (no year in name): {path.name}")
            continue
        if args.from_year <= year <= args.to_year:
            by_year[year].append(path)

    if not by_year:
        raise RuntimeError(
            f"No tiles matching {args.pattern!r} with years "
            f"{args.from_year}-{args.to_year} in {input_dir}"
        )

    years = sorted(by_year.keys())
    print(f"[INFO] Years to mosaic: {years}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Workers: {args.workers}")

    tasks = [
        (y, [str(p) for p in by_year[y]], str(output_dir), args.overwrite) for y in years
    ]

    if args.workers <= 1:
        for year, paths, out_str, ow in tasks:
            y, status = mosaic_one_year(year, paths, out_str, ow)
            print(f"[INFO] Year {y}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            futs = {
                ex.submit(mosaic_one_year, y, paths, out_str, ow): y
                for y, paths, out_str, ow in tasks
            }
            for fut in as_completed(futs):
                year = futs[fut]
                try:
                    y, status = fut.result()
                    print(f"[INFO] Year {y}: {status}")
                except Exception as e:
                    raise RuntimeError(f"Mosaic failed for year {year}") from e

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
