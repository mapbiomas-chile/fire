#!/usr/bin/env python3
"""
Inspect dNBR/rNBR/NBR band index in MapBiomas mosaic COGs.

Example:
  python auxiliares/inspect_mosaic_dnbr_band.py \\
    --mosaic-dir /home/flepin/mosaics_cog \\
    --region 1 --year 2019
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxiliares.mosaic_dnbr import (  # noqa: E402
    band_descriptions,
    find_dnbr_band_index,
    mosaic_path_for_tile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List mosaic bands and detect dNBR index.")
    parser.add_argument("--mosaic-dir", required=True, help="Directory with b14_chile_r*_YYYY_cog.tif")
    parser.add_argument("--region", default="1", help="Region id (default: 1)")
    parser.add_argument("--year", type=int, default=2019, help="Calendar year (default: 2019)")
    parser.add_argument("--satellite", default="b14")
    parser.add_argument("--country", default="chile")
    parser.add_argument("--dnbr-band", type=int, default=None, help="Override detected band (1-based)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mosaic_path = mosaic_path_for_tile(
        mosaic_dir=Path(args.mosaic_dir),
        region=str(args.region),
        year=args.year,
        satellite=args.satellite,
        country=args.country,
    )
    if not mosaic_path.is_file():
        print(f"ERROR: mosaic not found: {mosaic_path}", file=sys.stderr)
        return 1

    with rasterio.open(mosaic_path) as src:
        descriptions = band_descriptions(src)
        band_idx = find_dnbr_band_index(descriptions, explicit_band=args.dnbr_band)

    print(f"Mosaic: {mosaic_path}")
    print(f"Detected dNBR band: {band_idx} ({descriptions[band_idx - 1]})")
    print("All bands:")
    for i, name in enumerate(descriptions, start=1):
        marker = " <-- dNBR" if i == band_idx else ""
        print(f"  {i:2d}: {name}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
