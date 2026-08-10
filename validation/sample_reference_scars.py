#!/usr/bin/env python3
"""Sample reference fire scars for validation.

Typical MapBiomas Fire Chile design:
  1. Keep scars with area >= ``--min-ha`` (default 200 ha).
  2. Draw a random sample across the multi-year series
     (optionally stratified by year so every season is represented).

Input should already be in a **projected / equal-area CRS** (e.g. Chile Albers from
``reproject_vector_to_equal_area.py``) so areas are meaningful. If missing,
areas are computed from geometry (still requires projected CRS).

Example (leftraru)::

  python validation/sample_reference_scars.py \\
    --catalog ~/validation/UNIDOS_13_18_albers.gpkg \\
    --year-column Season \\
    --from-year 2013 --to-year 2018 \\
    --min-ha 200 \\
    --sample-n 60 \\
    --seed 42 \\
    --stratify-by-year \\
    --output ~/validation/samples/unidos_ge200ha_n60_seed42.gpkg
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter reference scars by minimum area and draw a random sample "
            "across the multi-year series for validation."
        )
    )
    p.add_argument("--catalog", required=True, type=Path, help="Reference scars vector.")
    p.add_argument("--layer", default=None, help="Optional GPKG layer name.")
    p.add_argument(
        "--year-column",
        default=None,
        help="Year/season column (default: first of year, Season, IgnDate).",
    )
    p.add_argument("--from-year", type=int, default=None)
    p.add_argument("--to-year", type=int, default=None)
    p.add_argument(
        "--min-ha",
        type=float,
        default=200.0,
        help="Minimum scar area in hectares (default: 200).",
    )
    p.add_argument(
        "--max-ha",
        type=float,
        default=None,
        help="Optional maximum scar area in hectares.",
    )
    p.add_argument(
        "--area-column",
        default=None,
        help=(
            "Column with area in ha (default: use area_ha if present, else "
            "compute from geometry)."
        ),
    )
    p.add_argument(
        "--sample-n",
        type=int,
        required=True,
        metavar="N",
        help="Number of scars to sample (after area/year filters).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (default: 42).",
    )
    p.add_argument(
        "--stratify-by-year",
        action="store_true",
        help=(
            "Split N as evenly as possible across years that still have "
            "eligible scars (recommended for multi-season validation)."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output GeoPackage of sampled scars.",
    )
    p.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional JSON with sampling parameters and counts.",
    )
    return p.parse_args()


def extract_year(value: object) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty date value.")
    try:
        return datetime.fromisoformat(text[:10]).year
    except ValueError:
        return int(text[:4])


def resolve_year_column(columns: list[str], preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in columns:
            raise ValueError(f"Year column {preferred!r} not in catalog: {columns}")
        return preferred
    for candidate in ("year", "Season", "IgnDate"):
        if candidate in columns:
            return candidate
    raise ValueError(
        "Cannot resolve year column: pass --year-column or add year/Season/IgnDate."
    )


def row_year(value: object, year_column: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        return int(value)
    return extract_year(value)


def ensure_area_ha(gdf: gpd.GeoDataFrame, area_column: str | None) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if area_column is not None:
        if area_column not in out.columns:
            raise ValueError(f"Area column {area_column!r} not found.")
        out["sample_area_ha"] = pd.to_numeric(out[area_column], errors="coerce")
        return out
    if "area_ha" in out.columns:
        out["sample_area_ha"] = pd.to_numeric(out["area_ha"], errors="coerce")
        return out
    if out.crs is None:
        raise ValueError("Catalog has no CRS; reproject before computing areas.")
    if out.crs.is_geographic:
        raise ValueError(
            "Catalog CRS is geographic. Reproject with "
            "validation/reproject_vector_to_equal_area.py first."
        )
    out["sample_area_ha"] = out.geometry.area / 10_000.0
    return out


def sample_stratified(
    gdf: gpd.GeoDataFrame,
    n: int,
    *,
    year_col: str,
    seed: int,
) -> gpd.GeoDataFrame:
    """Sample up to n rows, allocating as evenly as possible across years."""
    years = sorted(gdf[year_col].unique())
    if not years:
        return gdf.iloc[0:0].copy()

    # Initial equal split
    base = n // len(years)
    rem = n % len(years)
    allot: dict[int, int] = {
        int(y): base + (1 if i < rem else 0) for i, y in enumerate(years)
    }

    rng = np.random.default_rng(seed)
    picked: list[gpd.GeoDataFrame] = []
    leftover_slots = 0

    for y in years:
        sub = gdf.loc[gdf[year_col] == y]
        want = allot[int(y)]
        if want <= 0:
            continue
        if len(sub) <= want:
            picked.append(sub)
            leftover_slots += want - len(sub)
        else:
            idx = rng.choice(sub.index.to_numpy(), size=want, replace=False)
            picked.append(sub.loc[idx])

    # Distribute unfilled slots to years that still have eligible scars
    if leftover_slots > 0 and picked:
        already = pd.concat(picked) if picked else gdf.iloc[0:0]
        remaining = gdf.drop(index=already.index, errors="ignore")
        if not remaining.empty:
            take = min(leftover_slots, len(remaining))
            idx = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
            picked.append(remaining.loc[idx])

    if not picked:
        return gdf.iloc[0:0].copy()
    return (
        gpd.GeoDataFrame(pd.concat(picked), crs=gdf.crs)
        .sample(frac=1.0, random_state=seed)  # shuffle order
        .reset_index(drop=True)
    )


def sample_global(
    gdf: gpd.GeoDataFrame,
    n: int,
    *,
    seed: int,
) -> gpd.GeoDataFrame:
    if len(gdf) <= n:
        return gdf.copy().reset_index(drop=True)
    return gdf.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    if args.sample_n < 1:
        raise SystemExit("--sample-n must be >= 1")
    if args.min_ha < 0:
        raise SystemExit("--min-ha must be >= 0")
    if args.max_ha is not None and args.max_ha < args.min_ha:
        raise SystemExit("--max-ha must be >= --min-ha")

    read_kw: dict = {}
    if args.layer:
        read_kw["layer"] = args.layer
    gdf = gpd.read_file(args.catalog, **read_kw)
    if gdf.empty:
        raise RuntimeError(f"Empty catalog: {args.catalog}")
    if gdf.crs is None:
        raise ValueError(f"Catalog has no CRS: {args.catalog}")

    year_col_src = resolve_year_column(list(gdf.columns), args.year_column)
    gdf = ensure_area_ha(gdf, args.area_column)
    gdf = gdf.assign(
        scar_year=[row_year(v, year_col_src) for v in gdf[year_col_src]],
        sample_area_ha=gdf["sample_area_ha"].astype(float),
    )

    pool = gdf.copy()
    n0 = len(pool)
    if args.from_year is not None:
        pool = pool.loc[pool["scar_year"] >= args.from_year]
    if args.to_year is not None:
        pool = pool.loc[pool["scar_year"] <= args.to_year]
    n_year = len(pool)
    pool = pool.loc[pool["sample_area_ha"] >= args.min_ha]
    if args.max_ha is not None:
        pool = pool.loc[pool["sample_area_ha"] <= args.max_ha]
    n_area = len(pool)

    if pool.empty:
        raise RuntimeError(
            f"No scars left after filters "
            f"(year range + area >= {args.min_ha} ha). "
            f"Start={n0}, after year={n_year}, after area={n_area}."
        )

    if args.stratify_by_year:
        sample = sample_stratified(
            pool, args.sample_n, year_col="scar_year", seed=args.seed
        )
        mode = "stratified_by_year"
    else:
        sample = sample_global(pool, args.sample_n, seed=args.seed)
        mode = "global_random"

    if sample.empty:
        raise RuntimeError("Sampling produced zero scars.")

    sample = sample.copy()
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    sample["sample_seed"] = args.seed
    sample["sample_mode"] = mode
    sample["sample_min_ha"] = args.min_ha

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    sample.to_file(args.output, driver="GPKG", layer="scar_sample")

    counts = (
        sample.groupby("scar_year").size().sort_index().to_dict()
        if "scar_year" in sample.columns
        else {}
    )
    manifest = {
        "catalog": str(args.catalog.resolve()),
        "output": str(args.output.resolve()),
        "year_column": year_col_src,
        "from_year": args.from_year,
        "to_year": args.to_year,
        "min_ha": args.min_ha,
        "max_ha": args.max_ha,
        "sample_n_requested": args.sample_n,
        "sample_n_drawn": len(sample),
        "pool_after_filters": n_area,
        "seed": args.seed,
        "mode": mode,
        "counts_by_year": {str(k): int(v) for k, v in counts.items()},
        "area_ha_min": float(sample["sample_area_ha"].min()),
        "area_ha_max": float(sample["sample_area_ha"].max()),
        "area_ha_mean": float(sample["sample_area_ha"].mean()),
    }
    manifest_path = args.manifest_json or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[INFO] Pool after filters: {n_area} scars (>= {args.min_ha} ha)")
    print(f"[INFO] Mode: {mode} | seed={args.seed} | drawn={len(sample)}")
    print(f"[INFO] Counts by year: {counts}")
    print(f"[INFO] Wrote sample: {args.output}")
    print(f"[INFO] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
