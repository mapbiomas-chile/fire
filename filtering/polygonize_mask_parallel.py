#!/usr/bin/env python3
"""
Polygonize mask pixels from filtered rasters in parallel.

For each input raster:
- read one band
- select pixels equal to mask value (default: 1)
- convert connected mask pixels to polygons using raster grid geometry
- write one GeoPackage (.gpkg)
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def polygonize_one_file(
    tif_path: Path,
    output_dir: Path,
    band: int,
    mask_value: float,
    connectivity: int,
) -> tuple[str, int]:
    with rasterio.open(tif_path) as src:
        data = src.read(band)
        transform = src.transform
        crs = src.crs

    mask = data == mask_value
    if not np.any(mask):
        gdf_empty = gpd.GeoDataFrame(
            {"source_file": [], "mask_value": []},
            geometry=[],
            crs=crs,
        )
        out_path = output_dir / f"{tif_path.stem}_mask{int(mask_value)}.gpkg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_empty.to_file(out_path, driver="GPKG")
        return str(out_path), 0

    raster_for_shapes = np.where(mask, 1, 0).astype(np.uint8)

    geoms = []
    values = []
    for geom, val in shapes(
        raster_for_shapes,
        mask=mask,
        transform=transform,
        connectivity=connectivity,
    ):
        if int(val) == 1:
            geoms.append(shape(geom))
            values.append(1)

    gdf = gpd.GeoDataFrame(
        {
            "source_file": [tif_path.name] * len(geoms),
            "mask_value": values,
        },
        geometry=geoms,
        crs=crs,
    )

    out_path = output_dir / f"{tif_path.stem}_mask{int(mask_value)}.gpkg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    return str(out_path), len(geoms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polygonize mask pixels from a directory of rasters in parallel."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input rasters (e.g. classified_filtered).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write GPKG outputs (one per raster).",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern for input rasters (default: *.tif).",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Band index (1-based) to polygonize (default: 1).",
    )
    parser.add_argument(
        "--mask-value",
        type=float,
        default=1,
        help="Pixel value to polygonize (default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel workers (default: cpu_count-1).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Pixel connectivity for polygonization (4 or 8, default: 8).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    tif_files = sorted(input_dir.glob(args.pattern))
    if not tif_files:
        raise RuntimeError(f"No rasters found in {input_dir} with pattern {args.pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(tif_files)))

    print(f"[INFO] Found {len(tif_files)} raster files.")
    print(f"[INFO] Using {workers} parallel workers.")
    print(f"[INFO] Polygonizing mask value = {args.mask_value}.")

    total_polygons = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                polygonize_one_file,
                tif_path,
                output_dir,
                args.band,
                args.mask_value,
                args.connectivity,
            )
            for tif_path in tif_files
        ]

        for future in as_completed(futures):
            out_path, n_polygons = future.result()
            total_polygons += n_polygons
            print(f"[INFO] Wrote {out_path} (polygons={n_polygons})")

    print(f"[INFO] Finished. Total polygons: {total_polygons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
