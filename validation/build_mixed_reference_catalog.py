#!/usr/bin/env python3
"""Build a mixed reference catalog: UNIDOS seasons + GABAM years.

Default design for MapBiomas Fire Chile validation:

  * **UNIDOS** (Miranda et al. style): years / seasons **2013–2018**
  * **GABAM**: years up to **2022**, **excluding 2019 and 2020**
    (high error in that collection for those years)

Outputs one Chile-Albers GeoPackage ready for
``sample_reference_scars.py`` (min 200 ha + random sample).

Example::

  python validation/build_mixed_reference_catalog.py \\
    --unidos ~/validation/UNIDOS_13_18.shp \\
    --gabam ~/validation/GABAM_chile.shp \\
    --unidos-year-column Season \\
    --gabam-year-column year \\
    --output ~/validation/mixed_unidos_gabam_albers.gpkg
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

CHILE_ALBERS = (
    "+proj=aea +lat_1=-18 +lat_2=-55 +lat_0=-37 +lon_0=-71 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge UNIDOS + GABAM reference scars into one projected catalog."
    )
    p.add_argument("--unidos", type=Path, required=True, help="UNIDOS vector (e.g. 2013–2018).")
    p.add_argument("--gabam", type=Path, required=True, help="GABAM vector.")
    p.add_argument("--unidos-layer", default=None)
    p.add_argument("--gabam-layer", default=None)
    p.add_argument("--unidos-year-column", default="Season")
    p.add_argument(
        "--gabam-year-column",
        default=None,
        help="GABAM year column (default: first of year, Season, Year, IgnDate).",
    )
    p.add_argument("--unidos-from-year", type=int, default=2013)
    p.add_argument("--unidos-to-year", type=int, default=2018)
    p.add_argument("--gabam-from-year", type=int, default=2013)
    p.add_argument("--gabam-to-year", type=int, default=2022)
    p.add_argument(
        "--gabam-exclude-years",
        default="2019,2020",
        help="Comma-separated GABAM years to drop (default: 2019,2020).",
    )
    p.add_argument(
        "--target-crs",
        default=CHILE_ALBERS,
        help="Projected CRS for areas (default: Chile Albers).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GeoPackage (mixed catalog, Chile Albers).",
    )
    p.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional JSON summary of counts by source/year.",
    )
    return p.parse_args()


def extract_year(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        y = int(value)
        return y if 1900 <= y <= 2100 else None
    if isinstance(value, (float, np.floating)):
        y = int(value)
        return y if 1900 <= y <= 2100 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).year
    except ValueError:
        pass
    digits = "".join(ch if ch.isdigit() else " " for ch in text[:10]).split()
    if not digits:
        return None
    y = int(digits[0][:4])
    return y if 1900 <= y <= 2100 else None


def resolve_year_column(columns: list[str], preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in columns:
            raise ValueError(f"Year column {preferred!r} not found. Columns: {columns}")
        return preferred
    for candidate in ("year", "Year", "Season", "season", "IgnDate", "YEAR"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Cannot find year column among: {columns}")


def read_source(
    path: Path,
    *,
    layer: str | None,
    year_column: str | None,
    source: str,
    from_year: int,
    to_year: int,
    exclude_years: set[int],
    target_crs: str,
) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {source} file: {path}")
    kw: dict = {}
    if layer:
        kw["layer"] = layer
    gdf = gpd.read_file(path, **kw)
    if gdf.empty:
        raise RuntimeError(f"{source} catalog is empty: {path}")
    if gdf.crs is None:
        raise ValueError(f"{source} has no CRS: {path}")

    ycol = resolve_year_column(list(gdf.columns), year_column)
    years = gdf[ycol].map(extract_year)
    gdf = gdf.assign(scar_year=years, source=source, year_column_src=ycol)
    gdf = gdf.loc[gdf["scar_year"].notna()].copy()
    gdf["scar_year"] = gdf["scar_year"].astype(int)
    gdf = gdf.loc[
        (gdf["scar_year"] >= from_year)
        & (gdf["scar_year"] <= to_year)
        & (~gdf["scar_year"].isin(exclude_years))
    ].copy()
    if gdf.empty:
        raise RuntimeError(
            f"{source}: no features in {from_year}-{to_year} "
            f"(excluding {sorted(exclude_years)}) after year filter."
        )

    if gdf.crs.to_string() != target_crs and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf["area_m2"] = gdf.geometry.area.astype(float)
    gdf["area_ha"] = gdf["area_m2"] / 10_000.0
    # Harmonized year field for sample_reference_scars / intersect
    gdf["Season"] = gdf["scar_year"]
    return gdf


def parse_exclude(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(t.strip()) for t in raw.split(",") if t.strip()}


def main() -> int:
    args = parse_args()
    exclude = parse_exclude(args.gabam_exclude_years)

    unidos = read_source(
        args.unidos,
        layer=args.unidos_layer,
        year_column=args.unidos_year_column,
        source="UNIDOS",
        from_year=args.unidos_from_year,
        to_year=args.unidos_to_year,
        exclude_years=set(),
        target_crs=args.target_crs,
    )
    gabam = read_source(
        args.gabam,
        layer=args.gabam_layer,
        year_column=args.gabam_year_column,
        source="GABAM",
        from_year=args.gabam_from_year,
        to_year=args.gabam_to_year,
        exclude_years=exclude,
        target_crs=args.target_crs,
    )

    # Keep a stable slim schema + original attrs when possible
    keep_base = ["source", "scar_year", "Season", "area_m2", "area_ha", "geometry"]
    frames = []
    for gdf, label in ((unidos, "UNIDOS"), (gabam, "GABAM")):
        # Preserve FireID if present
        extra = [c for c in ("FireID", "Id", "ID", "id") if c in gdf.columns]
        cols = [c for c in keep_base + extra if c in gdf.columns]
        part = gdf[cols].copy()
        part["ref_uid"] = [f"{label}_{i}" for i in range(1, len(part) + 1)]
        frames.append(part)

    mixed = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=args.target_crs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    mixed.to_file(args.output, driver="GPKG", layer="reference_scars")

    by_src_year = (
        mixed.groupby(["source", "scar_year"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    print("[INFO] Counts by source × year:")
    print(by_src_year.to_string(index=False))
    print(
        f"[INFO] Total: {len(mixed)} scars | "
        f"years {sorted(mixed['scar_year'].unique())} | "
        f"GABAM excluded years: {sorted(exclude)}"
    )
    print(f"[INFO] Wrote: {args.output}")

    manifest = {
        "unidos": str(args.unidos.resolve()),
        "gabam": str(args.gabam.resolve()),
        "output": str(args.output.resolve()),
        "unidos_years": f"{args.unidos_from_year}-{args.unidos_to_year}",
        "gabam_years": f"{args.gabam_from_year}-{args.gabam_to_year}",
        "gabam_exclude_years": sorted(exclude),
        "n_total": len(mixed),
        "n_unidos": int((mixed["source"] == "UNIDOS").sum()),
        "n_gabam": int((mixed["source"] == "GABAM").sum()),
        "counts": by_src_year.to_dict(orient="records"),
        "crs": args.target_crs,
    }
    man_path = args.manifest_json or args.output.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Manifest: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
