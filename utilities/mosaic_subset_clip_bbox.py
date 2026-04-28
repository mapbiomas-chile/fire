#!/usr/bin/env python3
"""Mosaic subset TIFF tiles and clip to convex-hull bounding box."""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge

from fire_regions_bbox_geojson import bbox_envelope_excluding_region


def bbox_geom_from_geojson_file(path: Path):
    """Load a polygon footprint (e.g. exported bbox) and return (geometry, crs)."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Bbox GeoJSON has no features: {path}")
    if gdf.crs is None:
        raise ValueError(f"Bbox GeoJSON has no CRS: {path}")
    geom = gdf.union_all()
    return geom, gdf.crs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge TIFF tiles from a subset folder and clip to merge bounds. "
            "Either pass fire regions (--geojson) to derive the bbox envelope, "
            "or pass a precomputed bbox polygon (--bbox-geojson), e.g. from "
            "fire_regions_bbox_geojson.py."
        )
    )
    region_or_bbox = parser.add_mutually_exclusive_group(required=True)
    region_or_bbox.add_argument(
        "--geojson",
        type=Path,
        help="Fire-regions vector; bbox envelope is computed inside (exclude region 5 by default).",
    )
    region_or_bbox.add_argument(
        "--bbox-geojson",
        type=Path,
        help="Ready-made bbox polygon GeoJSON (same CRS as tiles, typically EPSG:4326).",
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


def main() -> int:
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output_path = Path(args.output)

    if args.bbox_geojson is not None:
        bbox_path = Path(args.bbox_geojson)
        if not bbox_path.exists():
            raise FileNotFoundError(f"Bbox GeoJSON not found: {bbox_path}")
        bbox_geom, bbox_crs = bbox_geom_from_geojson_file(bbox_path)
        print(f"[INFO] Using bbox polygon from: {bbox_path}")
    else:
        geojson_path = Path(args.geojson)
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
        bbox_geom, bbox_crs = bbox_envelope_excluding_region(
            geojson_path,
            region_field=args.region_field,
            exclude_region=args.exclude_region,
        )

    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    tif_paths = sorted(subset_dir.rglob("*.tif"))
    if not tif_paths:
        raise ValueError(f"No .tif files found in: {subset_dir}")

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

    print(f"[INFO] Mosaic clipped to bbox bounds saved at: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
