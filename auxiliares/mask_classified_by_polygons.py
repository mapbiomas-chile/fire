#!/usr/bin/env python3
"""
Mask classified GeoTIFFs using final polygon layers (vector → raster).

For each input raster in ``classified_filtered/``, finds the matching
``{stem}_burn.gpkg`` in the polygon directory, rasterizes polygons to the
raster grid, and writes burn pixels only where the vector mask is 1.

Typical use: apply ``polygons_filtered_min20ha_p25/`` to
``classified_filtered/`` for GEE export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.tile_metadata import parse_calendar_year, parse_region  # noqa: E402

AUTO_POLYGON_SUFFIXES = ("_burn", "_mask1", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask classified rasters using polygon GeoPackages (one pair per tile)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with classified rasters (e.g. classified_filtered/).",
    )
    parser.add_argument(
        "--polygon-dir",
        required=True,
        help="Directory with per-tile polygon GPKG files (e.g. polygons_filtered_min20ha_p25/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for masked output rasters.",
    )
    parser.add_argument("--pattern", default="*.tif", help="Input raster glob (default: *.tif).")
    parser.add_argument(
        "--polygon-suffix",
        default="auto",
        help=(
            "Polygon stem suffix before .gpkg, or 'auto' to try _burn, _mask1, then "
            "region×year fallback (default: auto)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report raster/polygon pairs; do not write outputs.",
    )
    parser.add_argument(
        "--burn-value",
        type=float,
        default=1,
        help="Burn pixel value to keep inside polygons (default: 1).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_vector_masked",
        help="Suffix appended to output filename stem (default: _vector_masked).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (default: cpu_count - 1).",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def normalize_polygon_suffix(suffix: str) -> str:
    if suffix == "":
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def build_polygon_region_year_index(polygon_dir: Path) -> dict[tuple[str, int], list[Path]]:
    index: dict[tuple[str, int], list[Path]] = {}
    for gpkg_path in polygon_dir.glob("*.gpkg"):
        region = parse_region(gpkg_path)
        year = parse_calendar_year(gpkg_path)
        if region is None or year is None:
            continue
        index.setdefault((region, year), []).append(gpkg_path)
    for key in index:
        index[key] = sorted(index[key], key=lambda p: p.name)
    return index


def resolve_polygon_path(
    tif_path: Path,
    polygon_dir: Path,
    *,
    polygon_suffix: str,
    polygon_index: dict[tuple[str, int], list[Path]],
) -> tuple[Path | None, str]:
    if polygon_suffix == "auto":
        suffixes = AUTO_POLYGON_SUFFIXES
    else:
        suffixes = (normalize_polygon_suffix(polygon_suffix),)

    for suffix in suffixes:
        candidate = polygon_dir / f"{tif_path.stem}{suffix}.gpkg"
        if candidate.exists():
            return candidate, f"stem{suffix or ''}"

    region = parse_region(tif_path)
    year = parse_calendar_year(tif_path)
    if region is None or year is None:
        return None, "missing_region_or_year"

    candidates = polygon_index.get((region, year), [])
    if len(candidates) == 1:
        return candidates[0], "region_year"
    if len(candidates) > 1:
        stem = tif_path.stem
        best = max(
            candidates,
            key=lambda p: len(os.path.commonprefix([stem, p.stem])),
        )
        return best, "region_year_best_stem"

    return None, "not_found"


def rasterize_polygon_mask(
    gdf: gpd.GeoDataFrame,
    *,
    out_shape: tuple[int, int],
    transform,
    crs,
) -> np.ndarray:
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    if gdf.crs is None:
        raise ValueError("Polygon layer has no CRS.")
    if crs is None:
        raise ValueError("Raster has no CRS.")

    projected = gdf.to_crs(crs)
    shapes = [(geom, 1) for geom in projected.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def mask_one_tile(
    tif_path: Path,
    polygon_path: Path,
    output_dir: Path,
    *,
    burn_value: float,
    output_suffix: str,
) -> dict:
    tif_path = Path(tif_path)
    polygon_path = Path(polygon_path)
    output_dir = Path(output_dir)

    if not polygon_path.exists():
        raise FileNotFoundError(f"Polygon file not found for {tif_path.name}: {polygon_path}")

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    gdf = gpd.read_file(polygon_path)
    vector_mask = rasterize_polygon_mask(
        gdf,
        out_shape=(height, width),
        transform=transform,
        crs=crs,
    )

    burn_mask = data == burn_value
    inside_vector = vector_mask == 1
    kept = burn_mask & inside_vector
    out_data = np.where(kept, data, 0).astype(data.dtype)

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
    pixels_after = int(kept.sum())
    return {
        "input_raster": str(tif_path),
        "input_polygon": str(polygon_path),
        "output_raster": str(output_path),
        "region": parse_region(tif_path),
        "year": parse_calendar_year(tif_path),
        "polygon_count": int(len(gdf)),
        "burn_pixels_before": pixels_before,
        "burn_pixels_after": pixels_after,
        "pixels_removed": int(pixels_before - pixels_after),
    }



def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    polygon_dir = Path(args.polygon_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not polygon_dir.is_dir():
        raise FileNotFoundError(f"Polygon directory not found: {polygon_dir}")

    tif_paths = sorted(input_dir.glob(args.pattern))
    if not tif_paths:
        raise RuntimeError(f"No rasters found in {input_dir} with pattern {args.pattern!r}")

    polygon_index = build_polygon_region_year_index(polygon_dir)
    polygon_files = sorted(polygon_dir.glob("*.gpkg"))

    tasks = []
    skipped = []
    pairs: list[dict] = []
    for tif_path in tif_paths:
        poly_path, match_mode = resolve_polygon_path(
            tif_path,
            polygon_dir,
            polygon_suffix=args.polygon_suffix,
            polygon_index=polygon_index,
        )
        pair_info = {
            "raster": tif_path.name,
            "polygon": poly_path.name if poly_path else None,
            "match_mode": match_mode,
        }
        pairs.append(pair_info)
        if poly_path is None:
            skipped.append(
                {
                    "raster": str(tif_path),
                    "region": parse_region(tif_path),
                    "year": parse_calendar_year(tif_path),
                    "reason": match_mode,
                }
            )
            continue
        tasks.append(
            (
                tif_path,
                poly_path,
                output_dir,
                args.burn_value,
                args.output_suffix,
            )
        )

    print(
        f"[INFO] Input rasters: {len(tif_paths)} | Polygon GPKGs: {len(polygon_files)} | "
        f"Pairs resolved: {len(tasks)} | Skipped: {len(skipped)}",
        flush=True,
    )
    for pair in pairs[:3]:
        print(
            f"[INFO] Example pair: {pair['raster']} ← {pair['polygon']} ({pair['match_mode']})",
            flush=True,
        )

    if not tasks:
        sample_gpkg = [p.name for p in polygon_files[:3]]
        sample_tif = [p.name for p in tif_paths[:3]]
        raise RuntimeError(
            "No raster/polygon pairs found.\n"
            f"  TIF dir ({input_dir}): {len(tif_paths)} files, e.g. {sample_tif}\n"
            f"  GPKG dir ({polygon_dir}): {len(polygon_files)} files, e.g. {sample_gpkg}\n"
            "  Try --polygon-suffix auto (default) or check that stems / region×year align."
        )

    if args.dry_run:
        for pair in pairs:
            if pair["polygon"]:
                print(
                    f"[DRY-RUN] {pair['raster']} ← {pair['polygon']} ({pair['match_mode']})",
                    flush=True,
                )
        print(f"[DRY-RUN] Would mask {len(tasks)} tile(s). Re-run without --dry-run to write outputs.")
        return 0

    summaries: list[dict] = []
    workers = max(1, args.workers)
    if workers == 1:
        for task in tasks:
            summary = mask_one_tile(
                task[0],
                task[1],
                task[2],
                burn_value=task[3],
                output_suffix=task[4],
            )
            summaries.append(summary)
            print(
                f"[INFO] {Path(summary['input_raster']).name}: "
                f"{summary['burn_pixels_before']} → {summary['burn_pixels_after']} burn px",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(mask_one_tile, *t): t[0] for t in tasks}
            for fut in as_completed(futures):
                summary = fut.result()
                summaries.append(summary)
                print(
                    f"[INFO] {Path(summary['input_raster']).name}: "
                    f"{summary['burn_pixels_before']} → {summary['burn_pixels_after']} burn px",
                    flush=True,
                )

    payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "polygon_dir": str(polygon_dir),
        "output_dir": str(output_dir),
        "tiles_masked": len(summaries),
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
        print(f"[WARNING] Skipped {len(skipped)} raster(s) without matching polygon.", flush=True)
    print(f"[INFO] Wrote {len(summaries)} masked raster(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
