#!/usr/bin/env python3
"""
Unified classified filtering: LULC mask per year, then temporal first-burn dedup.

Runs filter_classified_parallel.py → filter_temporal_first_burn_year.py in sequence.
Use --lulc-only or --temporal-only to run a single stage (same as legacy pipeline steps).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print(f"[INFO] RUN: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply LULC non-burnable masks and temporal first-burn dedup to classified tiles."
        )
    )
    parser.add_argument("--classified-dir", required=True, help="Raw classified GeoTIFFs.")
    parser.add_argument("--masks-dir", required=True, help="Directory with mascara_total_<year>.tif.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Final filtered rasters (after LULC + temporal).",
    )
    parser.add_argument(
        "--lulc-intermediate-dir",
        default=None,
        help="LULC-only output (default: <output-dir>/_lulc_intermediate).",
    )
    parser.add_argument(
        "--keep-lulc-intermediate",
        action="store_true",
        help="Keep LULC intermediate rasters after temporal step.",
    )
    parser.add_argument("--from-year", type=int, default=2013)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--fill-value", type=float, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-band", type=int, default=1)
    parser.add_argument("--temporal-suffix", default="_first_burn_year")
    parser.add_argument(
        "--year-token-index",
        type=int,
        default=3,
        help="Calendar year token index in MapBiomas filenames (default: 3).",
    )
    parser.add_argument(
        "--spatial-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--name-contains",
        default=None,
        help="Only apply temporal step to filenames containing this substring.",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="JSON path for temporal step statistics.",
    )
    parser.add_argument(
        "--lulc-only",
        action="store_true",
        help="Run only the LULC mask step (writes to --output-dir).",
    )
    parser.add_argument(
        "--temporal-only",
        action="store_true",
        help="Run only temporal dedup (--classified-dir = LULC-filtered input).",
    )
    args = parser.parse_args()

    if args.lulc_only and args.temporal_only:
        raise ValueError("Use at most one of --lulc-only and --temporal-only.")

    classified_dir = Path(args.classified_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)

    if not classified_dir.is_dir():
        raise FileNotFoundError(f"Classified dir not found: {classified_dir}")
    if not args.temporal_only and not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    py = sys.executable
    lulc_script = SCRIPT_DIR / "filter_classified_parallel.py"
    temporal_script = SCRIPT_DIR / "filter_temporal_first_burn_year.py"

    run_lulc = not args.temporal_only
    run_temporal = not args.lulc_only

    if run_lulc and run_temporal:
        lulc_dir = Path(args.lulc_intermediate_dir) if args.lulc_intermediate_dir else output_dir / "_lulc_intermediate"
    elif run_lulc:
        lulc_dir = output_dir
    else:
        lulc_dir = classified_dir

    if run_lulc:
        lulc_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                py,
                str(lulc_script),
                "--input-dir",
                str(classified_dir),
                "--masks-dir",
                str(masks_dir),
                "--output-dir",
                str(lulc_dir),
                "--workers",
                str(args.workers),
                "--target-band",
                str(args.target_band),
                "--fill-value",
                str(args.fill_value),
            ]
        )

    if run_temporal:
        temporal_input = lulc_dir if run_lulc else classified_dir
        if not temporal_input.is_dir():
            raise FileNotFoundError(f"Temporal input dir not found: {temporal_input}")
        output_dir.mkdir(parents=True, exist_ok=True)

        temporal_cmd = [
            py,
            str(temporal_script),
            "--input-dir",
            str(temporal_input),
            "--output-dir",
            str(output_dir),
            "--from-year",
            str(args.from_year),
            "--to-year",
            str(args.to_year),
            "--fill-value",
            str(args.fill_value),
            "--workers",
            str(args.workers),
            "--target-band",
            str(args.target_band),
            "--suffix",
            args.temporal_suffix,
            "--year-token-index",
            str(args.year_token_index),
            "--connectivity",
            str(args.connectivity),
        ]
        if args.spatial_merge:
            temporal_cmd.append("--spatial-merge")
        else:
            temporal_cmd.append("--no-spatial-merge")
        if args.name_contains:
            temporal_cmd.extend(["--name-contains", args.name_contains])
        if args.stats_json:
            temporal_cmd.extend(["--stats-json", args.stats_json])
        _run(temporal_cmd)

        if not args.keep_lulc_intermediate and run_lulc:
            shutil.rmtree(lulc_dir)
            print(f"[INFO] Removed LULC intermediate: {lulc_dir}", flush=True)

    print("[INFO] Classified filtering finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
