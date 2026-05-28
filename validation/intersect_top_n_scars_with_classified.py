#!/usr/bin/env python3
"""Identify classified polygons that intersect each scar; write A and hits as geometries.

This script is intentionally generic: it does NOT compute B = unary_union(b_i),
A intersect B, A union B, nor any spatial index. Downstream ``calculate_jaccard_index.py``
consumes the two output layers and derives B and per-scar Jaccard metrics.

Use ``--by-year`` to run one output GeoPackage per calendar year (recommended when
the classified folder holds many years). With ``--top-n``, ranks are **within each
year** when ``--by-year`` is set, otherwise **global** across the catalog.
"""

from __future__ import annotations

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "From a fire-scar catalog, optionally pick the N largest scars and find every "
            "classified polygon that intersects each scar (same year). Output is a GeoPackage "
            "with layers 'scar' and 'classified_hits', or one such file per year with "
            "--by-year. Use validation/calculate_jaccard_index.py --hits-gpkg on that output "
            "to build B = unary_union(b_i) and per-scar Jaccard."
        )
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Vector file with all scars (e.g. GeoPackage). Must be in a projected CRS.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name when reading a multi-layer catalog (optional).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set: with --by-year, the N largest scars **per year**; otherwise the N largest "
            "**globally**. If omitted, every scar considered is processed."
        ),
    )
    parser.add_argument(
        "--area-column",
        default="area_ha",
        help="Column used to rank scars when --top-n is set (default: area_ha).",
    )
    parser.add_argument(
        "--year-column",
        default=None,
        help=(
            "Catalog attribute holding the calendar year for joining to classified filenames. "
            "If omitted, uses the first available among: year, Season, IgnDate (CONAF-style)."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        metavar="YYYY",
        help=(
            "Process only scars for this calendar year (single run). Incompatible with "
            "--by-year. Implies one output file (--output)."
        ),
    )
    parser.add_argument(
        "--by-year",
        action="store_true",
        help=(
            "Write one GeoPackage per distinct scar year: {--output-stem}_{year}.gpkg "
            "under --output-dir. With --top-n, selection is per year. Incompatible with "
            "--year and with --output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Required with --by-year: directory for yearly outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default="scar_classified_hits",
        help="With --by-year: output files are {stem}_{year}.gpkg (default: scar_classified_hits).",
    )
    parser.add_argument(
        "--classified-dir",
        required=True,
        help="Directory with yearly classified polygon GeoPackages.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GeoPackage path (required unless --by-year).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Number of parallel workers (default: 15).",
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


def row_year(row: pd.Series, year_column: str | None) -> int:
    if year_column is not None:
        if year_column not in row.index:
            raise ValueError(f"Year column {year_column!r} not found in catalog.")
        val = row[year_column]
        if pd.isna(val):
            raise ValueError(f"Missing value in year column {year_column!r}.")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            y = int(val)
            if 1900 <= y <= 2100:
                return y
        return extract_year(val)

    if "year" in row.index and pd.notna(row["year"]):
        return int(row["year"])
    if "Season" in row.index and pd.notna(row["Season"]):
        return int(row["Season"])
    if "IgnDate" in row.index and pd.notna(row["IgnDate"]):
        return extract_year(row["IgnDate"])
    raise ValueError(
        "Cannot derive scar year: use --year-column or add 'year', 'Season', or 'IgnDate'."
    )


def extract_classified_year(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    return int(parts[3])


def extract_region(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse region from filename: {path.name}")
    return parts[2]


def process_one_scar(
    idx: int,
    scar_id: str,
    scar_year: int,
    geometry_wkt_str: str,
    crs_wkt: str,
    classified_by_year: dict[int, list[str]],
    temp_dir_str: str,
):
    temp_dir = Path(temp_dir_str)
    scar_geom = wkt.loads(geometry_wkt_str)
    bounds = scar_geom.bounds

    frames: list[gpd.GeoDataFrame] = []
    for classified_path_str in classified_by_year.get(scar_year, []):
        classified_path = Path(classified_path_str)
        region = extract_region(classified_path)

        candidates = gpd.read_file(classified_path, bbox=bounds)
        if candidates.empty:
            continue

        candidates = candidates.loc[candidates.geometry.intersects(scar_geom)].copy()
        if candidates.empty:
            continue

        candidates["region"] = region
        candidates["classified_file"] = classified_path.name
        candidates["scar_id"] = scar_id
        candidates["scar_year"] = scar_year
        frames.append(candidates)

    if not frames:
        return scar_id, scar_year, 0, None

    hits = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs_wkt)
    temp_path = temp_dir / f"hits_{idx:08d}.gpkg"
    hits.to_file(temp_path, driver="GPKG", layer="classified_hits")
    return scar_id, scar_year, len(hits), str(temp_path)


def build_scar_layer(gdf: gpd.GeoDataFrame, year_column: str | None) -> gpd.GeoDataFrame:
    out = gdf.copy().reset_index(drop=True)
    scar_ids: list[str] = []
    scar_years: list[int] = []
    for idx, (_, row) in enumerate(out.iterrows(), start=1):
        if "FireID" in row.index and pd.notna(row["FireID"]):
            sid = str(row["FireID"]).strip() or f"row_{idx}"
        else:
            sid = f"row_{idx}"
        scar_ids.append(sid)
        scar_years.append(row_year(row, year_column))
    out["scar_id"] = scar_ids
    out["scar_year"] = scar_years
    out["scar_area_m2"] = out.geometry.area.astype(float)
    out["scar_area_ha"] = out["scar_area_m2"] / 10000.0
    return out


def _rank_column(scar_gdf: gpd.GeoDataFrame, area_column: str) -> str:
    if area_column in scar_gdf.columns:
        return area_column
    return "scar_area_ha"


def apply_top_n(
    scar_gdf: gpd.GeoDataFrame,
    top_n: int | None,
    area_column: str,
) -> gpd.GeoDataFrame:
    if top_n is None:
        return scar_gdf
    if top_n < 1:
        raise ValueError("--top-n must be >= 1 when set.")
    col = _rank_column(scar_gdf, area_column)
    return (
        scar_gdf.sort_values(col, ascending=False, na_position="last")
        .head(top_n)
        .reset_index(drop=True)
    )


def intersect_and_write(
    scar_gdf: gpd.GeoDataFrame,
    classified_by_year: dict[int, list[str]],
    output_path: Path,
    workers: int,
) -> None:
    if scar_gdf.empty:
        return

    crs_wkt = scar_gdf.crs.to_wkt()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    print(
        f"[INFO] Scars this batch: {len(scar_gdf)}; "
        f"workers: {workers}; output: {output_path}"
    )

    with tempfile.TemporaryDirectory(prefix="scar_hits_") as temp_dir:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, (_, row) in enumerate(scar_gdf.iterrows(), start=1):
                fut = executor.submit(
                    process_one_scar,
                    idx,
                    str(row["scar_id"]),
                    int(row["scar_year"]),
                    row.geometry.wkt,
                    crs_wkt,
                    classified_by_year,
                    temp_dir,
                )
                futures[fut] = (idx, row["scar_id"])

            temp_paths: list[str] = []
            total = len(futures)
            for i, fut in enumerate(as_completed(futures), start=1):
                scar_id, scar_year, n_hits, temp_path = fut.result()
                if temp_path:
                    temp_paths.append(temp_path)
                print(
                    f"[INFO] Completed ({i}/{total}): "
                    f"scar_id={scar_id} year={scar_year} hits={n_hits}"
                )

        if temp_paths:
            hits_frames = [gpd.read_file(p, layer="classified_hits") for p in temp_paths]
            all_hits = gpd.GeoDataFrame(
                pd.concat(hits_frames, ignore_index=True),
                crs=scar_gdf.crs,
            )
        else:
            all_hits = gpd.GeoDataFrame(
                {
                    "scar_id": [],
                    "scar_year": [],
                    "region": [],
                    "classified_file": [],
                },
                geometry=[],
                crs=scar_gdf.crs,
            )

    scar_gdf.to_file(output_path, driver="GPKG", layer="scar")
    all_hits.to_file(output_path, driver="GPKG", layer="classified_hits", mode="a")

    print(
        f"[INFO] Wrote layers 'scar' ({len(scar_gdf)} features) and "
        f"'classified_hits' ({len(all_hits)} features) to: {output_path}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.by_year:
        if args.output:
            raise SystemExit("With --by-year use --output-dir and --output-stem, not --output.")
        if not args.output_dir:
            raise SystemExit("--by-year requires --output-dir.")
        if args.year is not None:
            raise SystemExit("Use either --year (single year) or --by-year, not both.")
    else:
        if not args.output:
            raise SystemExit("--output is required unless --by-year is set.")
        if args.output_dir:
            raise SystemExit("--output-dir is only used with --by-year.")


def main() -> int:
    args = parse_args()
    _validate_args(args)

    catalog_path = Path(args.catalog)
    classified_dir = Path(args.classified_dir)

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not classified_dir.exists():
        raise FileNotFoundError(f"Classified directory not found: {classified_dir}")

    read_kw: dict = {}
    if args.layer:
        read_kw["layer"] = args.layer

    gdf = gpd.read_file(catalog_path, **read_kw)
    if gdf.crs is None:
        raise ValueError(f"Catalog has no CRS: {catalog_path}")
    if gdf.crs.is_geographic:
        raise ValueError(
            "Catalog CRS must be projected (equal-area) for correct areas; "
            f"got geographic CRS: {gdf.crs}"
        )
    if gdf.empty:
        raise RuntimeError(f"Catalog is empty: {catalog_path}")

    classified_paths = sorted(classified_dir.glob("*.gpkg"))
    if not classified_paths:
        raise RuntimeError(f"No GeoPackages found in: {classified_dir}")

    classified_by_year: dict[int, list[str]] = {}
    for path in classified_paths:
        year = extract_classified_year(path)
        classified_by_year.setdefault(year, []).append(str(path))

    scar_gdf = build_scar_layer(gdf, args.year_column)

    rank_key = _rank_column(scar_gdf, args.area_column)

    print(f"[INFO] Classified GPKG files: {len(classified_paths)} (years: {sorted(classified_by_year.keys())})")
    print(f"[INFO] Workers: {args.workers}")

    if args.by_year:
        out_dir = Path(args.output_dir)
        years = sorted(scar_gdf["scar_year"].unique())
        for y in years:
            sub = scar_gdf.loc[scar_gdf["scar_year"] == y].copy()
            sub = apply_top_n(sub, args.top_n, args.area_column)
            if sub.empty:
                print(f"[INFO] Year {y}: no scars after --top-n filter, skip.")
                continue
            if y not in classified_by_year:
                print(
                    f"[WARN] Year {y}: no classified GeoPackage in --classified-dir; "
                    f"writing scars with empty hits."
                )
            out_path = out_dir / f"{args.output_stem}_{y}.gpkg"
            print(f"[INFO] --- Year {y}: {len(sub)} scar(s) -> {out_path.name} ---")
            intersect_and_write(sub, classified_by_year, out_path, args.workers)
        print("[INFO] Done (by-year).")
        return 0

    if args.year is not None:
        scar_gdf = scar_gdf.loc[scar_gdf["scar_year"] == args.year].copy()
        if scar_gdf.empty:
            raise RuntimeError(f"No scars in catalog for year {args.year}.")

    scar_gdf = apply_top_n(scar_gdf, args.top_n, args.area_column)
    if scar_gdf.empty:
        raise RuntimeError("No scars left after filters.")

    out = Path(args.output)
    if args.top_n:
        print(f"[INFO] Scars to process: {len(scar_gdf)} (global top {args.top_n} by {rank_key})")
    else:
        print(f"[INFO] Scars to process: {len(scar_gdf)}")

    intersect_and_write(scar_gdf, classified_by_year, out, args.workers)
    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
