#!/usr/bin/env python3
"""
Filter polygon GPKG files by a selected minimum-area threshold and export one GPKG.

The threshold is provided explicitly in hectares via --threshold-ha.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


RULE_KEY_MAP = {
    "area_cap": "rule_area_cap_threshold_ha",
    "score": "rule_score_threshold_ha",
    "elbow": "rule_elbow_threshold_ha",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter polygons by minimum area threshold and write one output GPKG."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with input polygon GPKG files.",
    )
    parser.add_argument(
        "--output-gpkg",
        required=True,
        help="Path to output filtered GPKG.",
    )
    parser.add_argument(
        "--stats-summary-json",
        default=None,
        help="Optional summary.json from summarize_polygon_stats_parallel.py.",
    )
    parser.add_argument(
        "--threshold-ha",
        type=float,
        default=None,
        help="Manual minimum area threshold in hectares. Overrides summary rule if provided.",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=["area_cap", "score", "elbow"],
        default="score",
        help="Threshold rule from summary.json (default: score).",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:32719",
        help="Projected CRS used to calculate area in meters (default: EPSG:32719).",
    )
    parser.add_argument(
        "--pattern",
        default="*.gpkg",
        help="Input file pattern (default: *.gpkg).",
    )
    return parser.parse_args()


def resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold_ha is not None:
        if args.threshold_ha < 0:
            raise ValueError("--threshold-ha must be >= 0")
        return float(args.threshold_ha)

    if not args.stats_summary_json:
        raise ValueError("Provide --threshold-ha or --stats-summary-json.")

    summary_path = Path(args.stats_summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    recs = summary.get("threshold_recommendations", {})
    rule_key = RULE_KEY_MAP[args.threshold_rule]
    threshold = recs.get(rule_key)
    if threshold is None:
        raise ValueError(
            f"Threshold for rule '{args.threshold_rule}' not found in {summary_path}. "
            "Use --threshold-ha to provide a manual value."
        )
    return float(threshold)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_gpkg = Path(args.output_gpkg)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    gpkg_files = sorted(input_dir.glob(args.pattern))
    if not gpkg_files:
        raise RuntimeError(f"No files found in {input_dir} with pattern {args.pattern}")

    threshold_ha = resolve_threshold(args)
    print(f"[INFO] Using minimum area threshold: {threshold_ha} ha")
    print(f"[INFO] Files to process: {len(gpkg_files)}")

    filtered_frames: list[gpd.GeoDataFrame] = []
    total_before = 0
    total_after = 0

    for gpkg_path in gpkg_files:
        gdf = gpd.read_file(gpkg_path)
        total_before += len(gdf)
        if gdf.empty:
            continue

        gdf_proj = gdf.to_crs(args.target_crs)
        area_m2 = gdf_proj.geometry.area.astype(float)
        area_ha = area_m2 / 10000.0
        keep = area_ha >= threshold_ha

        kept = gdf.loc[keep].copy()
        if kept.empty:
            print(f"[INFO] {gpkg_path.name}: kept 0 / {len(gdf)}")
            continue

        kept["source_file"] = gpkg_path.name
        kept["area_m2"] = area_m2.loc[keep].values
        kept["area_ha"] = area_ha.loc[keep].values
        kept["threshold_ha_used"] = float(threshold_ha)
        filtered_frames.append(kept)
        total_after += len(kept)
        print(f"[INFO] {gpkg_path.name}: kept {len(kept)} / {len(gdf)}")

    if filtered_frames:
        out_gdf = gpd.GeoDataFrame(pd.concat(filtered_frames, ignore_index=True), crs=filtered_frames[0].crs)
    else:
        out_gdf = gpd.GeoDataFrame(
            {"source_file": [], "area_m2": [], "area_ha": [], "threshold_ha_used": []},
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
