#!/usr/bin/env python3
"""Fire regions → convex hull (excluding one region) → axis-aligned bbox envelope → GeoJSON.

Also exposes ``convex_hull_excluding_region`` and ``bbox_envelope_excluding_region`` for
``list_intersecting_tiles.py`` and ``mosaic_subset_clip_bbox.py``.

CLI (writes one polygon FeatureCollection):

    python3 utilities/fire_regions_bbox_geojson.py --geojson INPUT.geojson --output bbox.geojson
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def _filter_regions_gdf(
    vector_path: Path,
    region_field: str,
    exclude_region: str,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"Input vector is empty: {vector_path}")
    if gdf.crs is None:
        raise ValueError("Input vector has no CRS.")
    if region_field not in gdf.columns:
        raise ValueError(
            f"Field '{region_field}' not found. Available: {list(gdf.columns)}"
        )
    filtered = gdf[gdf[region_field].astype(str) != str(exclude_region)]
    if filtered.empty:
        raise ValueError(
            "No features left after excluding region "
            f"'{exclude_region}' (field '{region_field}')."
        )
    return filtered


def convex_hull_excluding_region(
    vector_path: Path | str,
    *,
    region_field: str = "region",
    exclude_region: str = "5",
) -> Tuple[BaseGeometry, object]:
    """Convex hull of geometries whose ``region_field`` value is not ``exclude_region``."""
    path = Path(vector_path)
    filtered = _filter_regions_gdf(path, region_field, exclude_region)
    hull = filtered.union_all().convex_hull
    return hull, filtered.crs


def bbox_envelope_excluding_region(
    vector_path: Path | str,
    *,
    region_field: str = "region",
    exclude_region: str = "5",
) -> Tuple[BaseGeometry, object]:
    """Axis-aligned bounding-box polygon (envelope of the convex hull above)."""
    hull, crs = convex_hull_excluding_region(
        vector_path, region_field=region_field, exclude_region=exclude_region
    )
    return hull.envelope, crs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load fire-region polygons, exclude one region by attribute, compute the convex hull, "
            "take its axis-aligned bounding box (envelope), and save that rectangle as GeoJSON."
        )
    )
    parser.add_argument(
        "--geojson",
        required=True,
        type=Path,
        help="Input vector layer for fire regions (GeoJSON or other formats GeoPandas supports).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output GeoJSON path for the bbox polygon.",
    )
    parser.add_argument(
        "--region-field",
        default="region",
        help="Region attribute name (default: region).",
    )
    parser.add_argument(
        "--exclude-region",
        default="5",
        help="Region value to exclude before hull/bbox (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.geojson)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input vector not found: {input_path}")

    bbox_geom, crs = bbox_envelope_excluding_region(
        input_path,
        region_field=args.region_field,
        exclude_region=args.exclude_region,
    )

    minx, miny, maxx, maxy = bbox_geom.bounds

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [bbox_geom],
            "label": ["fire_regions_hull_bbox_envelope"],
            "exclude_region": [args.exclude_region],
            "source": [str(input_path.resolve())],
            "minx": [minx],
            "miny": [miny],
            "maxx": [maxx],
            "maxy": [maxy],
        },
        crs=crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")

    print(f"[INFO] Bbox envelope GeoJSON saved at: {output_path}")
    print(f"[INFO] Bounds minx,miny,maxx,maxy: {minx}, {miny}, {maxx}, {maxy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
