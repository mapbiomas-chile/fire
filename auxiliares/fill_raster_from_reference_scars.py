#!/usr/bin/env python3
"""
Fill gaps in masked burn rasters using a reference scar shapefile.

For each input GeoTIFF, selects reference polygons for the tile year (``Season``
or another column), rasterizes them on the raster grid, and sets burn pixels
where the reference mask is 1 but the raster is still 0.

By default only reference polygons that overlap existing burn pixels are used
(``--require-overlap``), so entire missed scars are not added—only holes inside
or next to scars we already partially detected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.tile_metadata import parse_calendar_year, parse_region  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing burn pixels using a reference scar vector layer."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with input rasters.")
    parser.add_argument("--output-dir", required=True, help="Directory for filled rasters.")
    parser.add_argument(
        "--reference-shp",
        required=True,
        help="Reference scar vector file (.shp, .gpkg, ...).",
    )
    parser.add_argument(
        "--year-column",
        default="Season",
        help="Attribute with calendar year (default: Season).",
    )
    parser.add_argument("--pattern", default="*.tif", help="Input glob (default: *.tif).")
    parser.add_argument(
        "--burn-value",
        type=float,
        default=1,
        help="Burn pixel value (default: 1).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_reference_filled",
        help="Suffix appended to output filename stem (default: _reference_filled).",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2013,
        help="First calendar year to apply reference fill (default: 2013).",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2018,
        help="Last calendar year to apply reference fill (default: 2018).",
    )
    parser.add_argument(
        "--passthrough-outside-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy input rasters unchanged for years outside from-year..to-year "
            "so downstream merges still see all tiles (default: true)."
        ),
    )
    parser.add_argument(
        "--require-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only fill inside reference polygons that overlap existing burn pixels "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (default: cpu_count - 1).",
    )
    parser.add_argument("--stats-json", default=None, help="Optional JSON summary path.")
    parser.add_argument("--dry-run", action="store_true", help="Report tasks only.")
    return parser.parse_args()


def year_key(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year.astype("Int64")
    num = pd.to_numeric(series, errors="coerce")
    plausible = num.notna() & (num >= 1800) & (num <= 2200)
    if plausible.all():
        return num.astype("Int64")
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.year.astype("Int64")


def load_reference_by_year(
    reference_path: Path,
    *,
    year_column: str,
) -> dict[int, gpd.GeoDataFrame]:
    gdf = gpd.read_file(reference_path)
    if year_column not in gdf.columns:
        raise ValueError(
            f"Year column {year_column!r} not in reference layer. "
            f"Columns: {list(gdf.columns)}"
        )
    years = year_key(gdf[year_column])
    gdf = gdf.loc[years.notna()].copy()
    gdf["_ref_year"] = years.loc[gdf.index].astype(int)

    by_year: dict[int, gpd.GeoDataFrame] = {}
    for year, group in gdf.groupby("_ref_year"):
        sub = group.drop(columns=["_ref_year"])
        by_year[int(year)] = sub.reset_index(drop=True)
    return by_year


def clip_to_raster_bounds(
    gdf: gpd.GeoDataFrame,
    *,
    height: int,
    width: int,
    transform,
    crs,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError("Reference layer has no CRS.")
    if crs is None:
        raise ValueError("Raster has no CRS.")

    projected = gdf.to_crs(crs)
    rbounds = array_bounds(height, width, transform)
    bbox = box(rbounds[0], rbounds[1], rbounds[2], rbounds[3])
    clip_box = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
    clipped = gpd.clip(projected, clip_box)
    return clipped.loc[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()


def rasterize_shapes(
    shapes: list[tuple],
    *,
    out_shape: tuple[int, int],
    transform,
) -> np.ndarray:
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def select_reference_shapes(
    gdf: gpd.GeoDataFrame,
    *,
    burn_mask: np.ndarray,
    out_shape: tuple[int, int],
    transform,
    crs,
    require_overlap: bool,
) -> list[tuple]:
    clipped = clip_to_raster_bounds(
        gdf,
        height=out_shape[0],
        width=out_shape[1],
        transform=transform,
        crs=crs,
    )
    if clipped.empty:
        return []

    shapes: list[tuple] = []
    for geom in clipped.geometry:
        if geom is None or geom.is_empty:
            continue
        if require_overlap:
            feat_mask = rasterize_shapes(
                [(geom, 1)],
                out_shape=out_shape,
                transform=transform,
            )
            if not (feat_mask & burn_mask).any():
                continue
        shapes.append((geom, 1))
    return shapes


def fill_one_raster(
    tif_path: Path,
    output_dir: Path,
    year_gdf: gpd.GeoDataFrame,
    *,
    burn_value: float,
    output_suffix: str,
    require_overlap: bool,
) -> dict:
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    burn_mask = data == burn_value
    shapes = select_reference_shapes(
        year_gdf,
        burn_mask=burn_mask,
        out_shape=(height, width),
        transform=transform,
        crs=crs,
        require_overlap=require_overlap,
    )
    ref_mask = rasterize_shapes(
        shapes,
        out_shape=(height, width),
        transform=transform,
    )

    fill_mask = (ref_mask == 1) & ~burn_mask
    out_data = np.where(fill_mask, burn_value, data).astype(data.dtype)

    output_name = f"{tif_path.stem}{output_suffix}.tif"
    output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    profile.update(
        count=1,
        dtype=out_data.dtype,
        compress="deflate",
        predictor=2,
        tiled=True,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_data, 1)

    pixels_before = int(burn_mask.sum())
    pixels_after = int((out_data == burn_value).sum())
    return {
        "input_raster": str(tif_path),
        "output_raster": str(output_path),
        "region": parse_region(tif_path),
        "year": parse_calendar_year(tif_path),
        "reference_polygons_used": len(shapes),
        "burn_pixels_before": pixels_before,
        "burn_pixels_after": pixels_after,
        "pixels_filled": int(fill_mask.sum()),
    }


def passthrough_one_raster(
    tif_path: Path,
    output_dir: Path,
    *,
    output_suffix: str,
) -> dict:
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_path = output_dir / f"{tif_path.stem}{output_suffix}.tif"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tif_path, output_path)
    return {
        "input_raster": str(tif_path),
        "output_raster": str(output_path),
        "region": parse_region(tif_path),
        "year": parse_calendar_year(tif_path),
        "action": "passthrough",
        "pixels_filled": 0,
    }


def _passthrough_task(task: tuple) -> dict:
    tif_path, output_dir, output_suffix = task
    return passthrough_one_raster(tif_path, output_dir, output_suffix=output_suffix)


def _fill_one_raster_task(task: tuple) -> dict:
    tif_path, output_dir, year_gdf, burn_value, output_suffix, require_overlap = task
    return fill_one_raster(
        tif_path,
        output_dir,
        year_gdf,
        burn_value=burn_value,
        output_suffix=output_suffix,
        require_overlap=require_overlap,
    )


def year_in_fill_range(year: int | None, *, from_year: int, to_year: int) -> bool:
    return year is not None and from_year <= year <= to_year


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reference_path = Path(args.reference_shp)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference vector not found: {reference_path}")

    tif_paths = sorted(input_dir.glob(args.pattern))
    if not tif_paths:
        raise RuntimeError(f"No rasters found in {input_dir} with pattern {args.pattern!r}")

    if args.from_year > args.to_year:
        raise ValueError(f"--from-year ({args.from_year}) must be <= --to-year ({args.to_year})")

    by_year = load_reference_by_year(reference_path, year_column=args.year_column)
    by_year = {
        year: gdf
        for year, gdf in by_year.items()
        if args.from_year <= year <= args.to_year
    }
    if not by_year:
        raise RuntimeError(
            f"No reference features in years {args.from_year}..{args.to_year}. "
            "Check --from-year / --to-year and the reference layer."
        )
    print(
        f"[INFO] Reference: {reference_path} | fill years {args.from_year}..{args.to_year} "
        f"({len(by_year)} year groups in reference)",
        flush=True,
    )
    print(
        f"[INFO] Require overlap with existing burns: {args.require_overlap}",
        flush=True,
    )

    fill_tasks = []
    passthrough_tasks = []
    skipped = []
    for tif_path in tif_paths:
        year = parse_calendar_year(tif_path)
        if year is None:
            skipped.append({"raster": str(tif_path), "reason": "no_year"})
            continue

        if not year_in_fill_range(year, from_year=args.from_year, to_year=args.to_year):
            if args.passthrough_outside_range:
                passthrough_tasks.append((tif_path, output_dir, args.output_suffix))
            else:
                skipped.append({"raster": str(tif_path), "reason": f"outside_range_{year}"})
            continue

        year_gdf = by_year.get(year)
        if year_gdf is None or year_gdf.empty:
            skipped.append({"raster": str(tif_path), "reason": f"no_reference_year_{year}"})
            continue
        fill_tasks.append(
            (
                tif_path,
                output_dir,
                year_gdf,
                args.burn_value,
                args.output_suffix,
                args.require_overlap,
            )
        )

    print(
        f"[INFO] Input rasters: {len(tif_paths)} | Fill: {len(fill_tasks)} | "
        f"Passthrough: {len(passthrough_tasks)} | Skipped: {len(skipped)}",
        flush=True,
    )
    if not fill_tasks and not passthrough_tasks:
        raise RuntimeError("No raster tasks resolved. Check filenames and year range.")

    if args.dry_run:
        for task in fill_tasks[:5]:
            print(
                f"[DRY-RUN] FILL {task[0].name} (year={parse_calendar_year(task[0])}, "
                f"region=r{parse_region(task[0])})",
                flush=True,
            )
        for task in passthrough_tasks[:3]:
            print(f"[DRY-RUN] COPY {task[0].name}", flush=True)
        print(
            f"[DRY-RUN] Would fill {len(fill_tasks)} raster(s) and "
            f"passthrough {len(passthrough_tasks)}.",
            flush=True,
        )
        return 0

    summaries: list[dict] = []
    workers = max(1, args.workers)
    all_tasks: list[tuple[str, tuple]] = [
        ("fill", t) for t in fill_tasks
    ] + [("passthrough", t) for t in passthrough_tasks]

    if workers == 1:
        for kind, task in all_tasks:
            if kind == "fill":
                summary = _fill_one_raster_task(task)
                summary["action"] = "fill"
            else:
                summary = _passthrough_task(task)
            summaries.append(summary)
            if kind == "fill":
                print(
                    f"[INFO] {Path(summary['input_raster']).name}: "
                    f"+{summary['pixels_filled']} px "
                    f"({summary['burn_pixels_before']} → {summary['burn_pixels_after']})",
                    flush=True,
                )
            else:
                print(f"[INFO] {Path(summary['input_raster']).name}: passthrough", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for kind, task in all_tasks:
                if kind == "fill":
                    fut = pool.submit(_fill_one_raster_task, task)
                else:
                    fut = pool.submit(_passthrough_task, task)
                futures[fut] = (kind, task[0])

            for fut in as_completed(futures):
                kind, tif_path = futures[fut]
                summary = fut.result()
                summary["action"] = kind
                summaries.append(summary)
                if kind == "fill":
                    print(
                        f"[INFO] {tif_path.name}: "
                        f"+{summary['pixels_filled']} px "
                        f"({summary['burn_pixels_before']} → {summary['burn_pixels_after']})",
                        flush=True,
                    )
                else:
                    print(f"[INFO] {tif_path.name}: passthrough", flush=True)

    payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "reference_shp": str(reference_path),
        "year_column": args.year_column,
        "from_year": args.from_year,
        "to_year": args.to_year,
        "require_overlap": args.require_overlap,
        "tiles_filled": sum(1 for s in summaries if s.get("action") == "fill"),
        "tiles_passthrough": sum(1 for s in summaries if s.get("action") == "passthrough"),
        "tiles_skipped": len(skipped),
        "skipped": skipped,
        "summaries": summaries,
    }
    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Stats: {stats_path}")

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} raster(s).", flush=True)
    print(
        f"[INFO] Wrote {len(summaries)} raster(s) to {output_dir} "
        f"({payload['tiles_filled']} filled, {payload['tiles_passthrough']} passthrough)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
