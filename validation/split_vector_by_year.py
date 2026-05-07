#!/usr/bin/env python3
"""Split a vector layer into one GeoPackage file per calendar year.

The year is taken from a column (default: Season, as in CONAF-style seasons).
Each output file is written with driver GPKG; the layer name matches the file
stem (without extension).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a vector file into yearly GeoPackages.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input vector file (.gpkg, .shp, .geojson, ...).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where yearly .gpkg files will be written (created if missing).",
    )
    parser.add_argument(
        "--year-column",
        default="Season",
        help="Attribute column with year (int, float, or datetime-like). Default: Season.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name when reading multi-layer formats such as GPKG.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Output filename prefix. Default: stem of --input "
            "(e.g. foo.gpkg -> foo_2013.gpkg).",
        ),
    )
    return parser.parse_args()


def year_key(series: pd.Series) -> pd.Series:
    """Return an integer year series; invalid values become NA."""
    s = series
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year.astype("Int64")

    num = pd.to_numeric(s, errors="coerce")
    plausible = num.notna() & (num >= 1800) & (num <= 2200)
    if plausible.all():
        return num.astype("Int64")

    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.year.astype("Int64")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    read_kw = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(input_path, **read_kw)

    col = args.year_column
    if col not in gdf.columns:
        raise ValueError(
            f"Year column {col!r} not in layer. Columns: {list(gdf.columns)}"
        )

    y = year_key(gdf[col])
    n_bad = int(y.isna().sum())
    if n_bad:
        print(
            f"[WARN] {n_bad} feature(s) have missing or unparseable {col!r}; "
            "they are omitted from outputs."
        )

    gdf = gdf.loc[y.notna()].copy()
    gdf["_split_year"] = y.loc[gdf.index].astype(int)

    prefix = args.prefix if args.prefix else input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(gdf["_split_year"].unique())
    print(f"[INFO] Input: {input_path} ({len(gdf)} features after year filter)")
    if not years:
        print("[WARN] No features with a valid year; nothing written.")
        return 0
    print(f"[INFO] Years: {years[0]}..{years[-1]} ({len(years)} files)")

    for yr in years:
        sub = gdf.loc[gdf["_split_year"] == yr].drop(columns=["_split_year"])
        out_path = output_dir / f"{prefix}_{yr}.gpkg"
        layer = out_path.stem
        sub.to_file(out_path, driver="GPKG", layer=layer)
        print(f"[INFO] Wrote {out_path} ({len(sub)} features, layer={layer!r})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
