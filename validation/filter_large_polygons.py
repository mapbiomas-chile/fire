#!/usr/bin/env python3
"""Keep polygons whose area is greater than a minimum threshold in hectares."""

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
        description="Filter a vector layer to keep polygons larger than a minimum area."
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
        "--min-area-ha",
        type=float,
        default=5000.0,
        help="Minimum area in hectares to keep (default: 5000).",
    )
    parser.add_argument(
        "--area-column",
        default="area_ha",
        help="Column containing polygon area in hectares (default: area_ha).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name for multi-layer formats such as GPKG.",
    )
    return parser.parse_args()


def resolve_driver(output_path: Path) -> str:
    driver = DRIVER_BY_EXT.get(output_path.suffix.lower())
    if driver is None:
        raise ValueError(
            f"Unsupported output extension '{output_path.suffix}'. "
            f"Supported: {sorted(DRIVER_BY_EXT)}"
        )
    return driver


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input vector not found: {input_path}")

    read_kwargs = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(input_path, **read_kwargs)

    if args.area_column not in gdf.columns:
        raise ValueError(
            f"Column '{args.area_column}' not found in {input_path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    keep = gdf[args.area_column].astype(float) > float(args.min_area_ha)
    kept = gdf.loc[keep].copy()

    print(f"[INFO] Input features: {len(gdf)}")
    print(f"[INFO] Threshold: {args.min_area_ha} ha")
    print(f"[INFO] Features kept: {len(kept)}")

    driver = resolve_driver(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_file(output_path, driver=driver)

    print(f"[INFO] Wrote filtered layer: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
