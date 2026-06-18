#!/usr/bin/env python3
"""Remove small connected burn patches from classified rasters (parallel)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.sieve_burn_mask import sieve_raster_file  # noqa: E402


def _sieve_one_file(args: tuple) -> dict:
    (
        tif_path,
        output_dir,
        min_pixels,
        min_area_ha,
        mask_value,
        connectivity,
        run_timestamp,
    ) = args
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_name = f"{tif_path.stem}_minpatch_{run_timestamp}.tif"
    output_path = output_dir / output_name
    stats = sieve_raster_file(
        tif_path,
        min_pixels=min_pixels,
        min_area_ha=min_area_ha,
        mask_value=mask_value,
        connectivity=connectivity,
        output_path=output_path,
    )
    stats["input_file"] = str(tif_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop connected burn components below a minimum size in each raster. "
            "Intended after the LULC filter step."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--name-contains", default=None)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Minimum connected component size in pixels.",
    )
    parser.add_argument(
        "--min-area-ha",
        type=float,
        default=None,
        help="Minimum component area in ha (from raster geotransform).",
    )
    parser.add_argument("--mask-value", type=float, default=1)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--workers", type=int, default=max(1, (cpu_count() or 1) - 1))
    parser.add_argument("--stats-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.min_pixels is None and args.min_area_ha is None:
        raise ValueError("Provide --min-pixels or --min-area-ha.")
    if args.min_pixels is not None and args.min_pixels < 1:
        raise ValueError("--min-pixels must be >= 1.")

    paths: list[Path] = []
    for path in sorted(input_dir.glob(args.pattern)):
        if not path.is_file():
            continue
        if args.name_contains and args.name_contains not in path.name:
            continue
        paths.append(path)
    if not paths:
        raise RuntimeError(f"No rasters found in {input_dir} with pattern {args.pattern!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    worker_count = max(1, min(args.workers, len(paths)))
    tasks = [
        (
            path,
            output_dir,
            args.min_pixels,
            args.min_area_ha,
            args.mask_value,
            args.connectivity,
            run_timestamp,
        )
        for path in paths
    ]

    if worker_count == 1:
        summaries = [_sieve_one_file(task) for task in tasks]
    else:
        with Pool(processes=worker_count) as pool:
            summaries = pool.map(_sieve_one_file, tasks)

    total_removed = sum(row.get("pixels_removed", 0) for row in summaries)
    print(
        f"[INFO] Sieved {len(summaries)} rasters; total pixels removed={total_removed}",
        flush=True,
    )
    for row in summaries:
        print(
            f"[INFO] {Path(row['input_file']).name}: "
            f"{row.get('components_before', '?')} -> {row.get('components_after', '?')} components, "
            f"removed {row.get('pixels_removed', 0)} px -> {Path(row['output_file']).name}",
            flush=True,
        )

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "min_pixels": args.min_pixels,
            "min_area_ha": args.min_area_ha,
            "mask_value": args.mask_value,
            "connectivity": args.connectivity,
            "tiles": summaries,
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote stats: {stats_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
