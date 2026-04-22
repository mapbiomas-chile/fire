#!/usr/bin/env python3
"""List TIFF tiles whose bounding boxes intersect a vector convex hull."""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a convex hull from a GeoJSON layer and list .tif tiles "
            "whose bounding boxes intersect it."
        )
    )
    parser.add_argument("--geojson", required=True, help="Path to input GeoJSON polygons.")
    parser.add_argument("--tiles-dir", required=True, help="Directory with .tif tiles.")
    parser.add_argument(
        "--region-field",
        default="region",
        help="Field name used to identify regions (default: region).",
    )
    parser.add_argument(
        "--exclude-region",
        default="5",
        help="Region value to exclude before convex hull (default: 5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional text output path (one intersecting tile path per line).",
    )
    return parser.parse_args()


def build_convex_hull(vector_path: Path, region_field: str, exclude_region: str):
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"Input vector is empty: {vector_path}")
    if gdf.crs is None:
        raise ValueError("Input vector has no CRS. Define CRS before running.")
    if region_field not in gdf.columns:
        raise ValueError(
            f"Field '{region_field}' not found. Available fields: {list(gdf.columns)}"
        )

    filtered = gdf[gdf[region_field].astype(str) != str(exclude_region)]
    if filtered.empty:
        raise ValueError(
            f"Filtering removed all features. Check --region-field and --exclude-region."
        )

    return filtered.union_all().convex_hull, filtered.crs


def find_intersecting_tiles(tiles_dir: Path, hull_geom, hull_crs) -> list[Path]:
    tif_paths = sorted(tiles_dir.rglob("*.tif"))
    intersects: list[Path] = []

    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            tile_bbox = box(*src.bounds)

            if src.crs is None:
                continue

            if src.crs != hull_crs:
                hull_series = gpd.GeoSeries([hull_geom], crs=hull_crs).to_crs(src.crs)
                hull_in_tile_crs = hull_series.iloc[0]
            else:
                hull_in_tile_crs = hull_geom

            if tile_bbox.intersects(hull_in_tile_crs):
                intersects.append(tif_path)

    return intersects


def main() -> int:
    args = parse_args()
    vector_path = Path(args.geojson)
    tiles_dir = Path(args.tiles_dir)

    if not vector_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {vector_path}")
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    hull_geom, hull_crs = build_convex_hull(
        vector_path, args.region_field, args.exclude_region
    )
    intersecting_tiles = find_intersecting_tiles(tiles_dir, hull_geom, hull_crs)

    for tile in intersecting_tiles:
        print(tile)

    print(f"\nTotal intersecting tiles: {len(intersecting_tiles)}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(str(tile) for tile in intersecting_tiles) + "\n",
            encoding="utf-8",
        )
        print(f"Saved list to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
