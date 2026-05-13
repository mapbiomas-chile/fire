#!/usr/bin/env python3
"""Mosaic reprojected regional tiles into one GeoTIFF per calendar year.

Expects MapBiomas-style stems such as
``b14_chile_r1_2013_cog_classified_filtered_..._albers.tif`` where the year is
the N-th segment when splitting the stem on ``_`` (default N=3: ``2013``).

All tiles in a group must share CRS, band count and dtypes. Overlap pixels are
combined with rasterio's ``merge`` (default method ``first``; optional ``min`` /
``max`` / ``last`` for edge cases).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.merge import merge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Group regional GeoTIFFs by year (from filename tokens) and write one merged "
            "mosaic per year into --output-dir."
        )
    )
    p.add_argument("--input-dir", required=True, help="Directory with per-region rasters.")
    p.add_argument("--output-dir", required=True, help="Directory for yearly mosaics.")
    p.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob under input-dir (default: *.tif).",
    )
    p.add_argument(
        "--year-token-index",
        type=int,
        default=3,
        metavar="N",
        help=(
            "0-based index of the calendar-year token when splitting the filename stem on "
            "'_' (default: 3 for b14_chile_r1_2013_...)."
        ),
    )
    p.add_argument(
        "--output-stem",
        default="merged",
        help="Output files are {output-stem}_{year}.tif (default: merged).",
    )
    p.add_argument(
        "--method",
        choices=("first", "last", "min", "max"),
        default="first",
        help="rasterio.merge overlap rule (default: first).",
    )
    p.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Override output nodata (default: use first source's nodata if any).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (one yearly mosaic per task, default: cpu_count - 1).",
    )
    return p.parse_args()


def year_from_path(path: Path, token_index: int) -> int:
    parts = path.stem.split("_")
    if token_index < 0 or token_index >= len(parts):
        raise ValueError(f"Cannot read year from stem (index {token_index}): {path.name}")
    raw = parts[token_index]
    if not raw.isdigit() or len(raw) != 4:
        raise ValueError(
            f"Year token at index {token_index} is not a 4-digit year in: {path.name}"
        )
    y = int(raw)
    if not (1900 <= y <= 2100):
        raise ValueError(f"Unreasonable year {y} parsed from: {path.name}")
    return y


def group_paths_by_year(paths: list[Path], token_index: int) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = defaultdict(list)
    for p in paths:
        y = year_from_path(p, token_index)
        groups[y].append(p)
    return dict(groups)


def _merge_one_year(
    year: int,
    paths: list[Path],
    output_dir: Path,
    output_stem: str,
    method: str,
    nodata_override: float | None,
) -> tuple[int, str, int]:
    paths = sorted(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_stem}_{year}.tif"

    srcs = [rasterio.open(p) for p in paths]
    try:
        first = srcs[0]
        for i, s in enumerate(srcs[1:], start=1):
            if s.crs != first.crs:
                raise ValueError(
                    f"CRS mismatch for year {year}: {paths[0].name} vs {paths[i].name}"
                )
            if s.count != first.count:
                raise ValueError(
                    f"Band count mismatch for year {year}: {paths[0].name} vs {paths[i].name}"
                )

        nodata = nodata_override
        if nodata is None and first.nodata is not None:
            nodata = first.nodata

        mosaic, out_transform = merge(
            srcs,
            nodata=nodata,
            method=method,
        )

        profile = first.profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            crs=first.crs,
            nodata=nodata,
            compress=profile.get("compress", "deflate"),
            predictor=profile.get("predictor", 2),
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()

    return year, out_path.name, len(paths)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise RuntimeError(f"No rasters matching {args.pattern!r} in {input_dir}")

    groups = group_paths_by_year(paths, args.year_token_index)
    years_sorted = sorted(groups.keys())
    print(f"[INFO] Found {len(paths)} rasters in {len(years_sorted)} year groups: {years_sorted}")
    print(f"[INFO] Merge method: {args.method}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Output stem: {args.output_stem}_<year>.tif")

    workers = max(1, min(args.workers, len(years_sorted)))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _merge_one_year,
                year,
                groups[year],
                output_dir,
                args.output_stem,
                args.method,
                args.nodata,
            )
            for year in years_sorted
        ]
        for fut in as_completed(futs):
            y, name, n_tiles = fut.result()
            print(f"[INFO] Year {y}: merged {n_tiles} tiles -> {name}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
