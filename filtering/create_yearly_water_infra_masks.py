#!/usr/bin/env python3
"""Create yearly binary masks for rio_lago and infraestructura classes."""

import argparse
from pathlib import Path

import numpy as np
import rasterio


TARGET_CLASSES = [
    ("rio_lago", 33),
    ("infraestructura", 24),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate yearly 0/1 masks for classes rio_lago (33) and "
            "infraestructura (24)."
        )
    )
    parser.add_argument("--input-tif", required=True, help="Input multi-band raster.")
    parser.add_argument(
        "--output-dir",
        default="/mnt/e/mapbiomas/fire/lulc_2025/mascaras_acumuladas",
        help="Output directory for yearly mask TIFFs.",
    )
    parser.add_argument(
        "--start-year-in-band-1",
        type=int,
        default=2000,
        help="Year represented by band 1 (default: 2000).",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2013,
        help="First year to export (default: 2013).",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2024,
        help="Last year to export, inclusive (default: 2024).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_tif)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        for year in range(args.from_year, args.to_year + 1):
            band = year - args.start_year_in_band_1 + 1
            if band < 1 or band > src.count:
                raise ValueError(
                    f"Year {year} maps to band {band}, outside raster range 1..{src.count}"
                )

            data = src.read(band)
            for class_name, class_value in TARGET_CLASSES:
                mask = (data == class_value).astype(np.uint8)
                output_path = output_dir / f"mascara_{class_name}_{year}.tif"
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(mask, 1)
                print(
                    f"[INFO] Saved {output_path} (year={year}, band={band}, class={class_value})"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
