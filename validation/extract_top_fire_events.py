#!/usr/bin/env python3
"""Extract the largest fire events in an area range from yearly Chile event GPKGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a layer with the N largest fire events in a ha range."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with chile_YYYY_events.gpkg files.",
    )
    parser.add_argument(
        "--output-gpkg",
        required=True,
        help="Output GeoPackage path.",
    )
    parser.add_argument("--from-year", type=int, default=2014)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--min-ha", type=float, default=200.0)
    parser.add_argument("--max-ha", type=float, default=5000.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--exclude-years",
        default="",
        help="Comma-separated years to skip (e.g. 2019,2020).",
    )
    parser.add_argument("--pattern", default="chile_{year}_events.gpkg")
    return parser.parse_args()


def _parse_exclude_years(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(token.strip()) for token in raw.split(",") if token.strip()}


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.min_ha > args.max_ha:
        raise ValueError("--min-ha must be <= --max-ha")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")

    exclude_years = _parse_exclude_years(args.exclude_years)
    frames: list[gpd.GeoDataFrame] = []
    for year in range(args.from_year, args.to_year + 1):
        if year in exclude_years:
            continue
        path = input_dir / args.pattern.format(year=year)
        if not path.is_file():
            print(f"[WARN] Missing: {path.name}", flush=True)
            continue
        gdf = gpd.read_file(path)
        if "year" not in gdf.columns:
            gdf = gdf.assign(year=year)
        frames.append(gdf)

    if not frames:
        raise RuntimeError(f"No event GPKG files found in {input_dir}")

    all_events = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=frames[0].crs,
    )
    in_range = all_events[
        (all_events["area_ha"] >= args.min_ha) & (all_events["area_ha"] <= args.max_ha)
    ].copy()
    if in_range.empty:
        raise RuntimeError(
            f"No events in {args.min_ha}-{args.max_ha} ha for {args.from_year}-{args.to_year}"
        )

    ranked = in_range.sort_values(
        ["area_ha", "year", "event_id"],
        ascending=[False, True, True],
    ).head(args.top_n)

    out = ranked.reset_index(drop=True)
    out.insert(0, "id", range(1, len(out) + 1))
    out = out.rename(columns={"year": "anio"})

    columns = [
        "id",
        "event_id",
        "anio",
        "area_m2",
        "area_ha",
        "fragment_count",
        "max_gap_m",
        "geometry",
    ]
    columns = [col for col in columns if col in out.columns]
    out = out[columns]

    output_path = Path(args.output_gpkg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(output_path, driver="GPKG")

    print(f"[INFO] Events in range: {len(in_range)}", flush=True)
    print(f"[INFO] Wrote top {len(out)} -> {output_path}", flush=True)
    print(
        out[["id", "event_id", "anio", "area_ha", "area_m2"]].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
