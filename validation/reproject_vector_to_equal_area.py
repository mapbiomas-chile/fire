#!/usr/bin/env python3
"""Reproject a vector layer to an equal-area CRS suitable for Chile (national level).

Presets:
  - chile_albers (default): Albers Conic Equal Area custom-tuned for Chile.
  - south_america_albers: ESRI:102033 (South America Albers Equal Area Conic).

Any arbitrary CRS understood by pyproj (EPSG code, proj string, WKT) can be
provided via --target-crs to override the preset.
"""

import argparse
from pathlib import Path

import geopandas as gpd


CHILE_ALBERS_PROJ = (
    "+proj=aea +lat_1=-18 +lat_2=-55 +lat_0=-37 +lon_0=-71 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

PRESETS = {
    "chile_albers": CHILE_ALBERS_PROJ,
    "south_america_albers": "ESRI:102033",
}

DRIVER_BY_EXT = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject a vector layer to an equal-area CRS appropriate for "
            "Chile at national level, and annotate each feature with its area."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input vector file (.shp, .gpkg, .geojson, ...).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output vector file. Driver is inferred from extension.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="chile_albers",
        help="Equal-area CRS preset (default: chile_albers).",
    )
    parser.add_argument(
        "--target-crs",
        default=None,
        help=(
            "Override the preset with any CRS understood by pyproj "
            "(EPSG:xxxx, proj string, or WKT)."
        ),
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


def compute_area_ha(gdf: gpd.GeoDataFrame, projected_crs: str) -> gpd.GeoSeries:
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS declared.")
    source = gdf if gdf.crs.is_projected else gdf.to_crs(projected_crs)
    return source.geometry.area.astype(float) / 10000.0


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input vector not found: {input_path}")

    target_crs = args.target_crs if args.target_crs else PRESETS[args.preset]
    driver = resolve_driver(output_path)

    read_kwargs = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(input_path, **read_kwargs)
    if gdf.crs is None:
        raise ValueError(
            f"Input layer has no CRS declared: {input_path}. "
            "Assign a CRS before reprojecting."
        )

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Features: {len(gdf)}")
    print(f"[INFO] Source CRS: {gdf.crs.to_string()}")
    print(f"[INFO] Target CRS: {target_crs}")

    area_before_ha = compute_area_ha(gdf, target_crs)

    gdf_proj = gdf.to_crs(target_crs)
    area_after_m2 = gdf_proj.geometry.area.astype(float)
    area_after_ha = area_after_m2 / 10000.0

    gdf_proj = gdf_proj.copy()
    gdf_proj["area_m2"] = area_after_m2.values
    gdf_proj["area_ha"] = area_after_ha.values

    total_before = float(area_before_ha.sum())
    total_after = float(area_after_ha.sum())
    delta_pct = (
        (total_after - total_before) / total_before * 100.0
        if total_before > 0
        else 0.0
    )

    print(f"[INFO] Total area before (ha): {total_before:,.2f}")
    print(f"[INFO] Total area after  (ha): {total_after:,.2f}")
    print(f"[INFO] Relative delta: {delta_pct:+.4f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_proj.to_file(output_path, driver=driver)

    print(f"[INFO] Wrote reprojected layer: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
