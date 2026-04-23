#!/usr/bin/env python3
"""Filter a vector layer to keep only features from a given year."""

import argparse
from pathlib import Path

import geopandas as gpd


DRIVER_BY_EXT = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only features whose date column matches a target year."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input vector file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output vector file.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2017,
        help="Year to keep (default: 2017).",
    )
    parser.add_argument(
        "--date-column",
        default="IgnDate",
        help="Date column used to extract year (default: IgnDate).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name for multi-layer formats such as GPKG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    read_kwargs = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(input_path, **read_kwargs)

    if args.date_column not in gdf.columns:
        raise ValueError(
            f"Column '{args.date_column}' not found. "
            f"Available: {list(gdf.columns)}"
        )

    gdf["_year"] = gdf[args.date_column].astype(str).str[:4].astype(int)
    keep = gdf["_year"] == args.year
    kept = gdf.loc[keep].drop(columns="_year").copy()

    print(f"[INFO] Input features: {len(gdf)}")
    print(f"[INFO] Year filter: {args.year}")
    print(f"[INFO] Features kept: {len(kept)}")

    driver = DRIVER_BY_EXT.get(output_path.suffix.lower())
    if driver is None:
        raise ValueError(
            f"Unsupported output extension '{output_path.suffix}'. "
            f"Supported: {sorted(DRIVER_BY_EXT)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_file(output_path, driver=driver)

    print(f"[INFO] Wrote filtered layer: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
