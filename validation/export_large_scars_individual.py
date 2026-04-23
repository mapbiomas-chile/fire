#!/usr/bin/env python3
"""Export one GeoPackage per fire scar from a vector layer."""

import argparse
from datetime import datetime
from pathlib import Path

import geopandas as gpd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one GeoPackage per fire scar, named by ID, area, and year."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input GeoPackage with large fire scars.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where one GeoPackage per scar will be written.",
    )
    parser.add_argument(
        "--id-column",
        default="FireID",
        help="Scar ID column (default: FireID).",
    )
    parser.add_argument(
        "--date-column",
        default="IgnDate",
        help="Date column used to extract year (default: IgnDate).",
    )
    parser.add_argument(
        "--area-column",
        default="area_ha",
        help="Area column in hectares (default: area_ha).",
    )
    return parser.parse_args()


def extract_year(value: object) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty date value.")
    try:
        return datetime.fromisoformat(text[:10]).year
    except ValueError:
        return int(text[:4])


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    gdf = gpd.read_file(input_path)
    required = [args.id_column, args.date_column, args.area_column]
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(gdf)
    print(f"[INFO] Input scars: {total}")

    for idx, row in gdf.iterrows():
        scar_id = str(row[args.id_column]).strip()
        year = extract_year(row[args.date_column])
        area_ha = float(row[args.area_column])
        area_ha_rounded = int(round(area_ha))

        out_name = f"{scar_id}_{area_ha_rounded}ha_{year}.gpkg"
        out_path = output_dir / out_name

        single = gdf.iloc[[idx]].copy()
        single["year"] = year
        single.to_file(out_path, driver="GPKG")
        print(f"[INFO] Wrote ({idx + 1}/{total}): {out_name}")

    print(f"[INFO] Finished writing individual scars to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
