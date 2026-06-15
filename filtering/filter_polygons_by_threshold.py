#!/usr/bin/env python3
"""
Filter polygon GPKG files by a minimum area threshold and export one GPKG.

Threshold can be set manually (--threshold-ha) or taken from
recommend_polygon_area_thresholds.py (--stats-summary-json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.tile_metadata import parse_calendar_year, parse_region  # noqa: E402

ALLOWED_REGIONS = {"1", "2", "4", "6"}

RULE_KEY_MAP = {
    "p5": "rule_p5_threshold_ha",
    "p10": "rule_p10_threshold_ha",
    "p25": "rule_p25_threshold_ha",
    "bottom5_mean": "rule_bottom5_mean_threshold_ha",
    "elbow": "rule_elbow_threshold_ha",
    # Legacy aliases
    "area_cap": "rule_area_cap_threshold_ha",
    "score": "rule_score_threshold_ha",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter polygons by minimum area threshold and write one output GPKG."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output-gpkg",
        default=None,
        help="Optional merged output GPKG (all tiles combined).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write one filtered GPKG per input tile (same filenames). For histograms.",
    )
    parser.add_argument(
        "--stats-summary-json",
        default=None,
        help="JSON from recommend_polygon_area_thresholds.py.",
    )
    parser.add_argument(
        "--threshold-ha",
        type=float,
        default=None,
        help="Manual minimum area (ha). Overrides JSON rules.",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=list(RULE_KEY_MAP.keys()),
        default="p10",
        help="Rule from summary JSON (default: p10).",
    )
    parser.add_argument(
        "--per-region",
        action="store_true",
        help="Use by_region thresholds (series pooled per region).",
    )
    parser.add_argument(
        "--per-region-year",
        action="store_true",
        help=(
            "Use by_region_year thresholds per tile/year "
            "(fallback: region, then global)."
        ),
    )
    parser.add_argument("--target-crs", default="EPSG:32719")
    parser.add_argument("--pattern", default="*.gpkg")
    return parser.parse_args()


def _threshold_from_block(block: dict, rule_key: str) -> float | None:
    recs = block.get("threshold_recommendations", {})
    value = recs.get(rule_key)
    if value is None:
        return None
    return float(value)


def lookup_threshold(
    summary: dict,
    rule: str,
    *,
    region: str | None,
    year: int | None,
    per_region_year: bool,
    per_region: bool,
) -> float:
    rule_key = RULE_KEY_MAP[rule]

    if per_region_year and region is not None and year is not None:
        year_block = summary.get("by_region_year", {}).get(region, {}).get(str(year), {})
        threshold = _threshold_from_block(year_block, rule_key)
        if threshold is not None:
            return threshold

    if (per_region_year or per_region) and region is not None:
        region_block = summary.get("by_region", {}).get(region, {})
        threshold = _threshold_from_block(region_block, rule_key)
        if threshold is not None:
            return threshold

    global_block = summary.get("global", summary)
    threshold = _threshold_from_block(global_block, rule_key)
    if threshold is None:
        raise ValueError(
            f"Threshold for rule '{rule}' ({rule_key}) not found in summary JSON."
        )
    return threshold


def resolve_threshold_map(args: argparse.Namespace, summary: dict) -> dict[str | None, float]:
    if args.per_region_year:
        return {}

    if args.per_region:
        thresholds: dict[str | None, float] = {}
        for region in ALLOWED_REGIONS:
            thresholds[region] = lookup_threshold(
                summary,
                args.threshold_rule,
                region=region,
                year=None,
                per_region_year=False,
                per_region=True,
            )
        return thresholds

    return {
        None: lookup_threshold(
            summary,
            args.threshold_rule,
            region=None,
            year=None,
            per_region_year=False,
            per_region=False,
        )
    }


def main() -> int:
    args = parse_args()
    if args.per_region and args.per_region_year:
        raise ValueError("Use only one of --per-region or --per-region-year.")

    input_dir = Path(args.input_dir)
    output_gpkg = Path(args.output_gpkg) if args.output_gpkg else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_gpkg is None and output_dir is None:
        raise ValueError("Provide --output-gpkg and/or --output-dir.")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    gpkg_files = sorted(input_dir.glob(args.pattern))
    if not gpkg_files:
        raise RuntimeError(f"No files found in {input_dir} with pattern {args.pattern}")

    summary: dict | None = None
    threshold_map: dict[str | None, float] = {}

    if args.threshold_ha is not None:
        if args.threshold_ha < 0:
            raise ValueError("--threshold-ha must be >= 0")
        threshold_map = {None: float(args.threshold_ha)}
    else:
        if not args.stats_summary_json:
            raise ValueError("Provide --threshold-ha or --stats-summary-json.")

        summary_path = Path(args.stats_summary_json)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        threshold_map = resolve_threshold_map(args, summary)

    if None in threshold_map:
        print(f"[INFO] Using global minimum area threshold: {threshold_map[None]} ha")
    elif args.per_region:
        print(f"[INFO] Using per-region thresholds ({args.threshold_rule}):")
        for region in sorted(threshold_map):
            print(f"       r{region}: {threshold_map[region]} ha")
    elif args.per_region_year:
        print(
            f"[INFO] Using per-region×year thresholds ({args.threshold_rule}); "
            "fallback: region → global"
        )

    filtered_frames: list[gpd.GeoDataFrame] = []
    total_before = 0
    total_after = 0

    for gpkg_path in gpkg_files:
        gdf = gpd.read_file(gpkg_path)
        total_before += len(gdf)
        if gdf.empty:
            continue

        region = parse_region(gpkg_path)
        year = parse_calendar_year(gpkg_path)

        if args.threshold_ha is not None:
            threshold_ha = threshold_map[None]
        elif args.per_region_year:
            if summary is None:
                raise RuntimeError("Summary JSON required for --per-region-year.")
            threshold_ha = lookup_threshold(
                summary,
                args.threshold_rule,
                region=region,
                year=year,
                per_region_year=True,
                per_region=False,
            )
        else:
            threshold_ha = threshold_map.get(region) if args.per_region else threshold_map[None]
            if threshold_ha is None:
                threshold_ha = threshold_map.get(None)
        if threshold_ha is None:
            print(f"[WARNING] {gpkg_path.name}: no region / threshold; skipping")
            continue

        gdf_proj = gdf.to_crs(args.target_crs)
        area_m2 = gdf_proj.geometry.area.astype(float)
        area_ha = area_m2 / 10000.0
        keep = area_ha >= threshold_ha

        kept = gdf.loc[keep].copy()
        if kept.empty:
            print(
                f"[INFO] {gpkg_path.name}: kept 0 / {len(gdf)} "
                f"(thr={threshold_ha} ha, r{region}, year={year})"
            )
            continue

        kept["source_file"] = gpkg_path.name
        kept["region"] = region
        kept["year"] = year
        kept["area_m2"] = area_m2.loc[keep].values
        kept["area_ha"] = area_ha.loc[keep].values
        kept["threshold_ha_used"] = float(threshold_ha)
        if output_dir is not None:
            tile_out = output_dir / gpkg_path.name
            kept.to_file(tile_out, driver="GPKG")
        filtered_frames.append(kept)
        total_after += len(kept)
        print(
            f"[INFO] {gpkg_path.name}: kept {len(kept)} / {len(gdf)} "
            f"(thr={threshold_ha} ha, r{region}, year={year})"
        )

    if output_gpkg is not None:
        if filtered_frames:
            out_gdf = gpd.GeoDataFrame(
                pd.concat(filtered_frames, ignore_index=True), crs=filtered_frames[0].crs
            )
        else:
            out_gdf = gpd.GeoDataFrame(
                {
                    "source_file": [],
                    "region": [],
                    "year": [],
                    "area_m2": [],
                    "area_ha": [],
                    "threshold_ha_used": [],
                },
                geometry=[],
                crs="EPSG:4326",
            )

        output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        out_gdf.to_file(output_gpkg, driver="GPKG")
        print(f"[INFO] Wrote filtered GPKG: {output_gpkg}")

    if output_dir is not None:
        n_tiles = len(list(output_dir.glob(args.pattern)))
        print(f"[INFO] Wrote {n_tiles} per-tile GPKG(s) under: {output_dir}")
    print(f"[INFO] Total polygons kept: {total_after} / {total_before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
