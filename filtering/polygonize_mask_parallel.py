#!/usr/bin/env python3
"""
Polygonize mask pixels from filtered rasters in parallel.

For each input raster:
- read one band
- select pixels equal to mask value (default: 1)
- convert connected mask pixels to polygons using raster grid geometry
- write one GeoPackage (.gpkg)

Implementation lives in ``lib.vectorize``; this script is a thin CLI wrapper.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.vectorize import polygonize_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polygonize mask pixels from a directory of rasters in parallel."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input rasters (e.g. classified_filtered).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write GPKG outputs (one per raster).",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern for input rasters (default: *.tif).",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Band index (1-based) to polygonize (default: 1).",
    )
    parser.add_argument(
        "--mask-value",
        type=float,
        default=1,
        help="Pixel value to polygonize (default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel workers (default: cpu_count-1).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Pixel connectivity for polygonization (4 or 8, default: 8).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    suffix = f"_mask{int(args.mask_value)}"
    summaries = polygonize_directory(
        input_dir,
        output_dir,
        pattern=args.pattern,
        band=args.band,
        mask_value=args.mask_value,
        connectivity=args.connectivity,
        workers=args.workers,
        output_suffix=suffix,
    )

    total_polygons = sum(row["polygon_count"] for row in summaries)
    for row in summaries:
        print(f"[INFO] Wrote {row['output_file']} (polygons={row['polygon_count']})")

    print(f"[INFO] Finished. Total polygons: {total_polygons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
