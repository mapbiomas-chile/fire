#!/usr/bin/env python3
"""Reproject single- or multi-band GeoTIFFs to an equal-area CRS (Chile / SA presets).

Matches the vector presets in ``reproject_vector_to_equal_area.py`` so classified masks and
reference scars can share the same projected CRS for intersection and area work.

Default resampling is *nearest* (suitable for class labels / binary masks). Use
``--resampling bilinear`` (or cubic) for continuous rasters.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

CHILE_ALBERS_PROJ = (
    "+proj=aea +lat_1=-18 +lat_2=-55 +lat_0=-37 +lon_0=-71 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

PRESETS = {
    "chile_albers": CHILE_ALBERS_PROJ,
    "south_america_albers": "ESRI:102033",
}

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warp GeoTIFFs to an equal-area CRS (default: Chile Albers)."
    )
    p.add_argument("--input-dir", required=True, help="Directory with input .tif files.")
    p.add_argument("--output-dir", required=True, help="Directory for warped outputs.")
    p.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern under input-dir (default: *.tif).",
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="chile_albers",
        help="Equal-area CRS preset (default: chile_albers).",
    )
    p.add_argument(
        "--target-crs",
        default=None,
        help="Override preset: EPSG code, proj string, or WKT.",
    )
    p.add_argument(
        "--resampling",
        choices=sorted(RESAMPLING.keys()),
        default="nearest",
        help="Warp resampling (default: nearest, for categorical rasters).",
    )
    p.add_argument(
        "--suffix",
        default="_albers",
        help="Append to output stem before .tif (default: _albers).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (default: cpu_count - 1).",
    )
    return p.parse_args()


def _warp_one(
    src_path: Path,
    output_dir: Path,
    dst_crs: str,
    resampling: Resampling,
    suffix: str,
) -> tuple[str, str]:
    """Return (input_name, status)."""
    out_name = f"{src_path.stem}{suffix}.tif"
    dst_path = output_dir / out_name

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Source has no CRS: {src_path}")

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress=profile.get("compress", "deflate"),
            predictor=profile.get("predictor", 2),
            tiled=True,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.transform,
                    dst_crs=dst.crs,
                    resampling=resampling,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )

    return src_path.name, f"ok -> {dst_path.name}"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    dst_crs = args.target_crs if args.target_crs else PRESETS[args.preset]
    resampling = RESAMPLING[args.resampling]
    suffix = args.suffix

    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise RuntimeError(f"No files matching {args.pattern!r} in {input_dir}")

    print(f"[INFO] Files: {len(paths)}")
    print(f"[INFO] Target CRS: {dst_crs}")
    print(f"[INFO] Resampling: {args.resampling}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Workers: {args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(_warp_one, p, output_dir, dst_crs, resampling, suffix)
            for p in paths
        ]
        for fut in as_completed(futs):
            name, status = fut.result()
            print(f"[INFO] {name}: {status}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
