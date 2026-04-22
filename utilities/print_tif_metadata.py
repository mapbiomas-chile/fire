#!/usr/bin/env python3
"""
Print key GeoTIFF metadata for quick grid compatibility checks.
"""

import argparse
from pathlib import Path

import rasterio


def main():
    parser = argparse.ArgumentParser(description="Print key metadata from a GeoTIFF.")
    parser.add_argument("tif_path", help="Path to input .tif file")
    args = parser.parse_args()

    tif_path = Path(args.tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"File not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        print(f"path: {tif_path}")
        print(f"driver: {src.driver}")
        print(f"crs: {src.crs}")
        print(f"width: {src.width}")
        print(f"height: {src.height}")
        print(f"count: {src.count}")
        print(f"dtypes: {src.dtypes}")
        print(f"nodata: {src.nodata}")
        print(f"transform: {src.transform}")
        print(f"res: {src.res}")
        print(f"bounds: {src.bounds}")


if __name__ == "__main__":
    main()
