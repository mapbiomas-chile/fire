#!/usr/bin/env python3
"""
Unified classified filtering: temporal first-burn dedup → internal hole fill → LULC.

Runs filter_temporal_first_burn_year.py → refine_burn_mask_closing.py →
filter_classified_parallel.py in sequence.

Use --temporal-only, --fill-only, or --lulc-only to run a single stage.
Use --skip-fill to run temporal → LULC without hole fill.
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
            "Apply temporal first-burn dedup, optional internal hole fill, "
            "then LULC non-burnable masks to classified tiles."
        )
    )
    parser.add_argument("--classified-dir", required=True, help="Input GeoTIFF folder for the first stage.")
    parser.add_argument("--masks-dir", default=None, help="Directory with mascara_total_<year>.tif (required for LULC step).")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Final filtered rasters (after temporal + fill + LULC when running full pipeline).",
    )
    parser.add_argument(
        "--temporal-intermediate-dir",
        default=None,
        help="Temporal step output (default: <output-dir>/_temporal_intermediate).",
    )
    parser.add_argument(
        "--fill-intermediate-dir",
        default=None,
        help="Hole-fill step output (default: <output-dir>/_fill_intermediate).",
    )
    parser.add_argument(
        "--keep-temporal-intermediate",
        action="store_true",
        help="Keep temporal intermediate rasters after the full pipeline.",
    )
    parser.add_argument(
        "--keep-fill-intermediate",
        action="store_true",
        help="Keep hole-fill intermediate rasters after the full pipeline.",
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
        help="Only apply temporal/fill steps to filenames containing this substring.",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="JSON path for temporal step statistics.",
    )
    parser.add_argument(
        "--fill-stats-json",
        default=None,
        help="JSON path for hole-fill step statistics.",
    )
    parser.add_argument(
        "--skip-fill",
        action="store_true",
        help="Skip internal hole fill (temporal → LULC only).",
    )
    parser.add_argument(
        "--fill-method",
        choices=("fill_holes", "closing", "both"),
        default="fill_holes",
        help="Refine method for the fill step (default: fill_holes).",
    )
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=0,
        help="Max enclosed hole size in pixels to fill; 0 = unlimited (default).",
    )
    parser.add_argument(
        "--closing-size",
        type=int,
        default=2,
        help="Closing kernel side when fill-method is closing/both (default: 2).",
    )
    parser.add_argument(
        "--fill-iterations",
        type=int,
        default=1,
        help="Closing passes when fill-method is closing/both (default: 1).",
    )
    parser.add_argument(
        "--temporal-only",
        action="store_true",
        help="Run only temporal dedup (writes to --output-dir).",
    )
    parser.add_argument(
        "--fill-only",
        action="store_true",
        help="Run only hole fill (--classified-dir = temporal output).",
    )
    parser.add_argument(
        "--lulc-only",
        action="store_true",
        help="Run only the LULC mask step (writes to --output-dir).",
    )
    args = parser.parse_args()

    single_stage_flags = sum(
        int(flag) for flag in (args.temporal_only, args.fill_only, args.lulc_only)
    )
    if single_stage_flags > 1:
        raise ValueError("Use at most one of --temporal-only, --fill-only, and --lulc-only.")

    classified_dir = Path(args.classified_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    output_dir = Path(args.output_dir)

    if not classified_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {classified_dir}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    run_temporal = args.temporal_only or not (args.fill_only or args.lulc_only)
    run_fill = (
        not args.skip_fill
        and not args.temporal_only
        and not args.lulc_only
    ) or args.fill_only
    run_lulc = args.lulc_only or not (args.temporal_only or args.fill_only)

    if run_lulc:
        if masks_dir is None:
            raise ValueError("--masks-dir is required when running the LULC step.")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Masks dir not found: {masks_dir}")

    py = sys.executable
    temporal_script = SCRIPT_DIR / "filter_temporal_first_burn_year.py"
    refine_script = SCRIPT_DIR / "refine_burn_mask_closing.py"
    lulc_script = SCRIPT_DIR / "filter_classified_parallel.py"

    full_pipeline = run_temporal and run_lulc and not args.temporal_only and not args.lulc_only

    if run_temporal and not args.temporal_only:
        temporal_dir = (
            Path(args.temporal_intermediate_dir)
            if args.temporal_intermediate_dir
            else output_dir / "_temporal_intermediate"
        )
    elif args.temporal_only:
        temporal_dir = output_dir
    else:
        temporal_dir = classified_dir

    if run_fill and full_pipeline:
        fill_dir = (
            Path(args.fill_intermediate_dir)
            if args.fill_intermediate_dir
            else output_dir / "_fill_intermediate"
        )
    elif args.fill_only:
        fill_dir = output_dir
    else:
        fill_dir = temporal_dir

    current_input = classified_dir

    if run_temporal:
        temporal_dir.mkdir(parents=True, exist_ok=True)
        temporal_cmd = [
            py,
            str(temporal_script),
            "--input-dir",
            str(current_input),
            "--output-dir",
            str(temporal_dir),
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
        current_input = temporal_dir

    if run_fill:
        fill_dir.mkdir(parents=True, exist_ok=True)
        refine_cmd = [
            py,
            str(refine_script),
            "--input-dir",
            str(current_input),
            "--output-dir",
            str(fill_dir),
            "--band",
            str(args.target_band),
            "--burn-value",
            "1",
            "--fill-value",
            str(int(args.fill_value)),
            "--method",
            args.fill_method,
            "--max-hole-area",
            str(args.max_hole_area),
            "--closing-size",
            str(args.closing_size),
            "--iterations",
            str(args.fill_iterations),
            "--output-stem-suffix",
            "",
            "--workers",
            str(args.workers),
        ]
        if args.name_contains:
            refine_cmd.extend(["--name-contains", args.name_contains])
        if args.fill_stats_json:
            refine_cmd.extend(["--stats-json", args.fill_stats_json])
        _run(refine_cmd)
        current_input = fill_dir

    if run_lulc:
        output_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                py,
                str(lulc_script),
                "--input-dir",
                str(current_input),
                "--masks-dir",
                str(masks_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                str(args.workers),
                "--target-band",
                str(args.target_band),
                "--fill-value",
                str(args.fill_value),
            ]
        )

    if full_pipeline:
        if run_temporal and not args.keep_temporal_intermediate:
            shutil.rmtree(temporal_dir)
            print(f"[INFO] Removed temporal intermediate: {temporal_dir}", flush=True)
        if run_fill and not args.keep_fill_intermediate:
            shutil.rmtree(fill_dir)
            print(f"[INFO] Removed fill intermediate: {fill_dir}", flush=True)

    print("[INFO] Classified filtering finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
