#!/usr/bin/env python3
"""Create one polygon feature per year from a date field."""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

DRIVER_BY_EXT = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group polygons by year extracted from a date column and write one "
            "feature per year."
        )
    )
    parser.add_argument("--input", required=True, help="Input vector file path.")
    parser.add_argument("--output", required=True, help="Output vector file path.")
    parser.add_argument(
        "--date-column",
        default="IgnDate",
        help="Date column used to extract year (default: IgnDate).",
    )
    parser.add_argument(
        "--year-field",
        default="year",
        help="Name of year field in output (default: year).",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Keep only years >= min-year.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Keep only years <= max-year.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated explicit year list to keep, e.g. 2013,2014,2018.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Input layer name for multi-layer formats such as GPKG.",
    )
    parser.add_argument(
        "--method",
        choices=("multipart", "dissolve"),
        default="multipart",
        help=(
            "Geometry merge mode. 'multipart' is faster and creates one "
            "MultiPolygon per year; 'dissolve' performs topological union."
        ),
    )
    return parser.parse_args()


def parse_year_set(years_arg: str | None) -> set[int] | None:
    if years_arg is None:
        return None
    values = [item.strip() for item in years_arg.split(",") if item.strip()]
    if not values:
        raise ValueError("Parameter --years is empty.")
    return {int(item) for item in values}


def to_multipolygon(geometries) -> MultiPolygon | None:
    polygons = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(list(geom.geoms))
    if not polygons:
        return None
    return MultiPolygon(polygons)


def build_output(
    gdf: gpd.GeoDataFrame,
    year_field: str,
    method: str,
) -> gpd.GeoDataFrame:
    rows = []
    for year, group in gdf.groupby("_year", sort=True):
        if method == "dissolve":
            geom = unary_union(group.geometry.values)
        else:
            geom = to_multipolygon(group.geometry.values)
        if geom is None or geom.is_empty:
            continue
        rows.append({year_field: int(year), "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    driver = DRIVER_BY_EXT.get(output_path.suffix.lower())
    if driver is None:
        raise ValueError(
            f"Unsupported output extension '{output_path.suffix}'. "
            f"Supported: {sorted(DRIVER_BY_EXT)}"
        )

    read_kwargs = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(input_path, **read_kwargs)

    if args.date_column not in gdf.columns:
        raise ValueError(
            f"Column '{args.date_column}' not found. Available: {list(gdf.columns)}"
        )

    years = pd.to_datetime(gdf[args.date_column], errors="coerce").dt.year
    keep = years.notna()
    allowed_years = parse_year_set(args.years)
    if args.min_year is not None:
        keep &= years >= args.min_year
    if args.max_year is not None:
        keep &= years <= args.max_year
    if allowed_years is not None:
        keep &= years.isin(allowed_years)

    working = gdf.loc[keep, ["geometry"]].copy()
    working["_year"] = years[keep].astype(int).values

    result = build_output(working, args.year_field, args.method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_path, driver=driver)

    years_out = sorted(result[args.year_field].astype(int).tolist()) if len(result) else []
    print(f"[INFO] Input features: {len(gdf)}")
    print(f"[INFO] Kept features: {len(working)}")
    print(f"[INFO] Output records: {len(result)}")
    print(f"[INFO] Output years: {years_out}")
    print(f"[INFO] Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
