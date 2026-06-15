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

from lib.tile_metadata import parse_region  # noqa: E402

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
    parser.add_argument("--output-gpkg", required=True)
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
        help="Use by_region thresholds from summary JSON (requires region in filename).",
    )
    parser.add_argument("--target-crs", default="EPSG:32719")
    parser.add_argument("--pattern", default="*.gpkg")
    return parser.parse_args()


def _lookup_threshold(summary: dict, rule: str, *, region: str | None) -> float:
    rule_key = RULE_KEY_MAP[rule]
    if region is not None and "by_region" in summary:
        block = summary["by_region"].get(region, {})
        recs = block.get("threshold_recommendations", {})
        if rule_key in recs:
            return float(recs[rule_key])
    block = summary.get("global", summary)
    recs = block.get("threshold_recommendations", {})
    threshold = recs.get(rule_key)
    if threshold is None:
        raise ValueError(
            f"Threshold for rule '{rule}' ({rule_key}) not found in summary JSON."
        )
    return float(threshold)


def resolve_threshold_map(args: argparse.Namespace) -> dict[str | None, float]:
    if args.threshold_ha is not None:
        if args.threshold_ha < 0:
            raise ValueError("--threshold-ha must be >= 0")
        return {None: float(args.threshold_ha)}

    if not args.stats_summary_json:
        raise ValueError("Provide --threshold-ha or --stats-summary-json.")

    summary_path = Path(args.stats_summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    if args.per_region:
        thresholds: dict[str | None, float] = {}
        for region in ALLOWED_REGIONS:
            try:
                thresholds[region] = _lookup_threshold(
                    summary, args.threshold_rule, region=region
                )
            except ValueError:
                thresholds[region] = _lookup_threshold(
                    summary, args.threshold_rule, region=None
                )
        return thresholds

    return {None: _lookup_threshold(summary, args.threshold_rule, region=None)}


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_gpkg = Path(args.output_gpkg)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    gpkg_files = sorted(input_dir.glob(args.pattern))
    if not gpkg_files:
        raise RuntimeError(f"No files found in {input_dir} with pattern {args.pattern}")

    threshold_map = resolve_threshold_map(args)
    if None in threshold_map:
        print(f"[INFO] Using global minimum area threshold: {threshold_map[None]} ha")
    else:
        print(f"[INFO] Using per-region thresholds ({args.threshold_rule}):")
        for region in sorted(threshold_map):
            print(f"       r{region}: {threshold_map[region]} ha")

    filtered_frames: list[gpd.GeoDataFrame] = []
    total_before = 0
    total_after = 0

    for gpkg_path in gpkg_files:
        gdf = gpd.read_file(gpkg_path)
        total_before += len(gdf)
        if gdf.empty:
            continue

        region = parse_region(gpkg_path)
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
            print(f"[INFO] {gpkg_path.name}: kept 0 / {len(gdf)} (thr={threshold_ha} ha)")
            continue

        kept["source_file"] = gpkg_path.name
        kept["region"] = region
        kept["area_m2"] = area_m2.loc[keep].values
        kept["area_ha"] = area_ha.loc[keep].values
        kept["threshold_ha_used"] = float(threshold_ha)
        filtered_frames.append(kept)
        total_after += len(kept)
        print(f"[INFO] {gpkg_path.name}: kept {len(kept)} / {len(gdf)} (thr={threshold_ha} ha)")

    if filtered_frames:
        out_gdf = gpd.GeoDataFrame(
            pd.concat(filtered_frames, ignore_index=True), crs=filtered_frames[0].crs
        )
    else:
        out_gdf = gpd.GeoDataFrame(
            {
                "source_file": [],
                "region": [],
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
    print(f"[INFO] Total polygons kept: {total_after} / {total_before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
