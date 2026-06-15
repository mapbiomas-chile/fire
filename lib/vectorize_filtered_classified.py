#!/usr/bin/env python3
"""
Vectorize post-filter classified burn masks (GeoTIFF → GeoPackage).

Typical input: ``classified_filtered/`` after ``run_filtering_pipeline.sh``
(temporal → hole fill → LULC). Writes one ``.gpkg`` per raster with columns
``year``, ``region``, ``source_file``, ``mask_value``.

Run from the repository root::

    python lib/vectorize_filtered_classified.py \\
      --input-dir /path/to/classified_filtered \\
      --output-dir /path/to/polygons
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.vectorize import merge_polygon_outputs, polygonize_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Polygonize filtered classified burn rasters (uint8 0/1) to GeoPackage. "
            "One output file per input tile; optional merged layer."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder with post-filter rasters (e.g. classified_filtered/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder for per-tile GeoPackages.",
    )
    parser.add_argument("--pattern", default="*.tif", help="Input glob (default: *.tif).")
    parser.add_argument(
        "--name-contains",
        default=None,
        help="Only process filenames containing this substring.",
    )
    parser.add_argument("--band", type=int, default=1, help="Raster band (1-based, default: 1).")
    parser.add_argument(
        "--mask-value",
        type=float,
        default=1,
        help="Pixel value to polygonize (default: 1).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Pixel connectivity (default: 8).",
    )
    parser.add_argument(
        "--sieve-min-pixels",
        type=int,
        default=None,
        help="Remove connected burn components smaller than N pixels before polygonize.",
    )
    parser.add_argument(
        "--skip-sieve",
        action="store_true",
        help="Polygonize without removing small connected components.",
    )
    parser.add_argument(
        "--sieve-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Connectivity for the pre-vectorize sieve (default: 8).",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers.")
    parser.add_argument(
        "--output-suffix",
        default="_burn",
        help="Suffix before .gpkg on each output (default: _burn).",
    )
    parser.add_argument(
        "--merged-gpkg",
        default=None,
        help="Optional path to write all polygons into one GeoPackage.",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional JSON summary of polygon counts per tile.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    summaries = polygonize_directory(
        input_dir,
        output_dir,
        pattern=args.pattern,
        name_contains=args.name_contains,
        band=args.band,
        mask_value=args.mask_value,
        connectivity=args.connectivity,
        sieve_min_pixels=None if args.skip_sieve else args.sieve_min_pixels,
        sieve_connectivity=args.sieve_connectivity,
        workers=args.workers,
        output_suffix=args.output_suffix,
    )

    total_polygons = sum(row["polygon_count"] for row in summaries)
    for row in summaries:
        print(
            f"[INFO] {row['output_file']} "
            f"(polygons={row['polygon_count']}, year={row['year']}, region={row['region']})",
            flush=True,
        )

    merged_summary = None
    if args.merged_gpkg:
        merged_summary = merge_polygon_outputs(summaries, Path(args.merged_gpkg))
        print(
            f"[INFO] Merged {merged_summary['polygon_count']} polygons from "
            f"{merged_summary['source_tiles']} tiles → {merged_summary['merged_file']}",
            flush=True,
        )

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "tile_count": len(summaries),
            "total_polygons": total_polygons,
            "tiles": summaries,
            "merged": merged_summary,
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote stats: {stats_path}", flush=True)

    print(f"[INFO] Finished. Tiles={len(summaries)} total_polygons={total_polygons}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
