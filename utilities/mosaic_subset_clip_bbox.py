#!/usr/bin/env python3
"""Mosaic subset TIFF tiles and clip to convex-hull bounding box."""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge TIFF tiles from a subset folder and clip to the bounding box "
            "of the convex hull of fire regions (excluding region 5)."
        )
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to regiones_fuego GeoJSON.",
    )
    parser.add_argument(
        "--subset-dir",
        required=True,
        help="Directory containing subset TIFF tiles.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GeoTIFF path.",
    )
    parser.add_argument(
        "--region-field",
        default="region",
        help="Region field name (default: region).",
    )
    parser.add_argument(
        "--exclude-region",
        default="5",
        help="Region value to exclude (default: 5).",
    )
    return parser.parse_args()


def get_convex_hull_bbox(geojson_path: Path, region_field: str, exclude_region: str):
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"Input vector is empty: {geojson_path}")
    if gdf.crs is None:
        raise ValueError("Input vector has no CRS.")
    if region_field not in gdf.columns:
        raise ValueError(
            f"Field '{region_field}' not found. Available: {list(gdf.columns)}"
        )

    filtered = gdf[gdf[region_field].astype(str) != str(exclude_region)]
    if filtered.empty:
        raise ValueError("No features left after filtering excluded region.")

    hull = filtered.union_all().convex_hull
    return hull.envelope, filtered.crs


def main() -> int:
    args = parse_args()
    geojson_path = Path(args.geojson)
    subset_dir = Path(args.subset_dir)
    output_path = Path(args.output)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    tif_paths = sorted(subset_dir.rglob("*.tif"))
    if not tif_paths:
        raise ValueError(f"No .tif files found in: {subset_dir}")

    bbox_geom, bbox_crs = get_convex_hull_bbox(
        geojson_path, args.region_field, args.exclude_region
    )

    srcs = [rasterio.open(p) for p in tif_paths]
    try:
        target_crs = srcs[0].crs
        if target_crs is None:
            raise ValueError(f"First tile has no CRS: {tif_paths[0]}")

        bbox_series = gpd.GeoSeries([bbox_geom], crs=bbox_crs).to_crs(target_crs)
        minx, miny, maxx, maxy = bbox_series.iloc[0].bounds

        mosaic_arr, mosaic_transform = merge(
            srcs,
            bounds=(minx, miny, maxx, maxy),
        )

        profile = srcs[0].profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": mosaic_transform,
                "count": mosaic_arr.shape[0],
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_arr)
    finally:
        for src in srcs:
            src.close()

    print(f"[INFO] Mosaic clipped to convex-hull bbox saved at: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
