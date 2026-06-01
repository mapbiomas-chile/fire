#!/usr/bin/env python3
"""
Temporal deduplication of burned-area rasters (per spatial tile).

For each pixel location, assigns burns to the first calendar year of detection.
Later years lose duplicate pixels at the same cell. Optionally (--spatial-merge,
default) pixels that are newly burned only in year Y but 8-connected to an existing
scar from an earlier year are merged into that origin year (e.g. Jan 2018 extent
connected to Dec 2017 fire → attributed to 2017).

Input: output of filter_classified_parallel.py (e.g. classified_filtered/).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import rasterio

YEAR_RE = re.compile(r"(20\d{2})")
_ORIGIN_NONE = np.uint16(0)
_ORIGIN_INF = np.uint32(65535)


def extract_year(path: Path) -> int:
    match = YEAR_RE.search(path.stem)
    if not match:
        raise ValueError(f"No year (20xx) in filename: {path.name}")
    return int(match.group(1))


def tile_key(path: Path) -> str:
    return YEAR_RE.sub("{YEAR}", path.stem, count=1)


def is_burned(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Binary burn mask; do not copy raw pixel values to output (may be float/noisy)."""
    burned = data > 0
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        burned &= data != nodata
    return burned


def detect_burn_value(
    originals: dict[int, np.ndarray],
    nodata_by_year: dict[int, float | None],
    years: list[int],
) -> int:
    """Use the single positive class value in inputs (expected: 1)."""
    samples: list[np.ndarray] = []
    for year in years:
        data = originals[year]
        burned = is_burned(data, nodata_by_year[year])
        if np.any(burned):
            samples.append(data[burned].ravel())
    if not samples:
        return 1
    positive = np.concatenate(samples)
    uniq = np.unique(positive)
    if uniq.size == 1:
        return int(uniq[0])
    raise ValueError(
        f"Expected one burn value (e.g. 1), got {uniq[:20].tolist()} "
        f"(check inputs in classified_filtered, not first_burn outputs)"
    )


def min_neighbor_origin(origin_year: np.ndarray, connectivity: int) -> np.ndarray:
    """Per-pixel minimum origin year among neighbors (65535 where no neighbor assigned)."""
    padded = np.pad(origin_year.astype(np.uint32), 1, constant_values=_ORIGIN_INF)
    h, w = origin_year.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            if connectivity == 4 and dy != 0 and dx != 0:
                continue
            sl = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            neighbors.append(sl)
    return np.min(np.stack(neighbors, axis=0), axis=0)


def assign_origin_year_tile(
    originals: dict[int, np.ndarray],
    nodata_by_year: dict[int, float | None],
    years: list[int],
    spatial_merge: bool,
    connectivity: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Returns (origin_year grid, stats)."""
    shape = next(iter(originals.values())).shape
    origin_year = np.zeros(shape, dtype=np.uint16)

    stats = {
        "pixels_same_cell_removed": 0,
        "pixels_spatial_merged_to_earlier_year": 0,
        "pixels_new_events": 0,
    }

    for year in years:
        data = originals[year]
        burned = is_burned(data, nodata_by_year[year])

        same_cell = burned & (origin_year > _ORIGIN_NONE)
        stats["pixels_same_cell_removed"] += int(same_cell.sum())

        new_only = burned & (origin_year == _ORIGIN_NONE)

        if spatial_merge and np.any(origin_year > _ORIGIN_NONE) and np.any(new_only):
            labeled = np.where(
                origin_year > _ORIGIN_NONE,
                origin_year.astype(np.uint32),
                _ORIGIN_INF,
            )
            min_orig = min_neighbor_origin(origin_year, connectivity)
            merge = new_only & (min_orig < _ORIGIN_INF)
            if np.any(merge):
                merged_origins = min_orig[merge].astype(np.uint16)
                origin_year[merge] = merged_origins
                stats["pixels_spatial_merged_to_earlier_year"] += int(merge.sum())
                new_only &= ~merge

        if np.any(new_only):
            origin_year[new_only] = np.uint16(year)
            stats["pixels_new_events"] += int(new_only.sum())

    return origin_year, stats


def _binary_gtiff_profile(src_profile: dict, nodata: int = 0) -> dict:
    """Clean uint8 0/1 GeoTIFF (no metadata inherited from float/classified sources)."""
    return {
        "driver": "GTiff",
        "height": src_profile["height"],
        "width": src_profile["width"],
        "transform": src_profile["transform"],
        "crs": src_profile["crs"],
        "dtype": rasterio.uint8,
        "count": 1,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }


def process_tile_group(args: tuple) -> dict:
    (
        key,
        year_to_path,
        from_year,
        to_year,
        output_dir,
        fill_value,
        target_band,
        suffix,
        spatial_merge,
        connectivity,
    ) = args

    years = sorted(y for y in year_to_path if from_year <= y <= to_year)
    if not years:
        return {"tile": key, "skipped": True, "reason": "no years in range"}

    profiles: dict[int, dict] = {}
    originals: dict[int, np.ndarray] = {}
    nodata_by_year: dict[int, float | None] = {}

    for year in years:
        path = year_to_path[year]
        with rasterio.open(path) as src:
            data = src.read(target_band)
            profile = src.profile.copy()
            nodata = src.nodata

        if originals and data.shape != next(iter(originals.values())).shape:
            raise ValueError(f"Shape mismatch in tile {key}: {path.name}")

        profiles[year] = profile
        originals[year] = data
        nodata_by_year[year] = nodata

    origin_year, assign_stats = assign_origin_year_tile(
        originals,
        nodata_by_year,
        years,
        spatial_merge=spatial_merge,
        connectivity=connectivity,
    )
    burn_value = int(detect_burn_value(originals, nodata_by_year, years))
    fill_u8 = int(fill_value)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "tile": key,
        "years": years,
        "burn_value": burn_value,
        "spatial_merge": spatial_merge,
        "assign": assign_stats,
        "pixels_written_by_year": {},
        "unique_values_by_year": {},
        "output_files": [],
    }

    for year in years:
        data = originals[year]
        keep = origin_year == year
        out = np.full(data.shape, fill_u8, dtype=np.uint8)
        out[keep] = np.uint8(burn_value)

        burned_before = is_burned(data, nodata_by_year[year])
        stats["pixels_written_by_year"][year] = int(keep.sum())
        stats.setdefault("pixels_removed_by_year", {})[year] = int(
            (burned_before & ~keep).sum()
        )

        profile = _binary_gtiff_profile(profiles[year], nodata=fill_u8)
        out_path = output_dir / f"{year_to_path[year].stem}{suffix}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)

        with rasterio.open(out_path) as verify:
            uniq = np.unique(verify.read(1))
        stats["unique_values_by_year"][year] = uniq.tolist()
        allowed = {fill_u8, burn_value}
        if not set(int(x) for x in uniq.tolist()).issubset(allowed):
            raise ValueError(f"{out_path.name}: unexpected values {uniq.tolist()}")

        stats["output_files"].append(str(out_path))

    return stats


def group_inputs(
    input_dir: Path,
    from_year: int,
    to_year: int,
    name_contains: str | None = None,
) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(input_dir.glob("*.tif")):
        if "_first_burn_year" in path.stem:
            continue
        if name_contains and name_contains not in path.name:
            continue
        try:
            year = extract_year(path)
        except ValueError:
            print(f"[WARN] Skip (no year): {path.name}")
            continue
        if from_year <= year <= to_year:
            groups[tile_key(path)][year] = path
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "First burn year per scar with optional spatial merge of new pixels "
            "connected to an earlier-year footprint (persistence filter)."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder with LULC-filtered classified GeoTIFFs (e.g. classified_filtered/).",
    )
    parser.add_argument("--output-dir", required=True, help="Output folder for deduplicated rasters.")
    parser.add_argument("--from-year", type=int, default=2013)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--fill-value", type=float, default=0)
    parser.add_argument("--target-band", type=int, default=1)
    parser.add_argument("--suffix", default="_first_burn_year")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--stats-json", default=None)
    parser.add_argument(
        "--spatial-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Merge new burns in year Y into the earliest neighboring origin year "
            "when 8-connected to an existing scar (default: on)."
        ),
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Neighbor connectivity for spatial merge (default: 8).",
    )
    parser.add_argument(
        "--name-contains",
        default=None,
        help="Only process files whose name includes this substring (e.g. 141228).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    groups = group_inputs(
        input_dir, args.from_year, args.to_year, name_contains=args.name_contains
    )
    if not groups:
        msg = f"No .tif files with years {args.from_year}-{args.to_year} in {input_dir}"
        if args.name_contains:
            msg += f" matching --name-contains {args.name_contains!r}"
        raise RuntimeError(msg)

    tasks = [
        (
            key,
            year_to_path,
            args.from_year,
            args.to_year,
            str(output_dir),
            args.fill_value,
            args.target_band,
            args.suffix,
            args.spatial_merge,
            args.connectivity,
        )
        for key, year_to_path in groups.items()
    ]

    workers = min(args.workers, len(tasks))
    print(f"[INFO] Tile groups: {len(tasks)}")
    if args.name_contains:
        print(f"[INFO] Name filter: contains {args.name_contains!r}")
    print(f"[INFO] Years: {args.from_year}-{args.to_year}")
    print(f"[INFO] Spatial merge: {args.spatial_merge} (connectivity={args.connectivity})")
    print(f"[INFO] Workers: {workers}")

    all_stats: list[dict] = []
    with Pool(processes=workers) as pool:
        for stats in pool.imap_unordered(process_tile_group, tasks):
            if stats.get("skipped"):
                print(f"[WARN] Skipped {stats['tile']}: {stats.get('reason')}")
                continue
            all_stats.append(stats)
            a = stats["assign"]
            print(
                f"[INFO] {stats['tile']}: "
                f"same_cell={a['pixels_same_cell_removed']} "
                f"spatial_merge={a['pixels_spatial_merged_to_earlier_year']} "
                f"new_events={a['pixels_new_events']}"
            )

    if args.stats_json:
        out = Path(args.stats_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "from_year": args.from_year,
                    "to_year": args.to_year,
                    "spatial_merge": args.spatial_merge,
                    "connectivity": args.connectivity,
                    "n_tiles": len(all_stats),
                    "tiles": all_stats,
                },
                f,
                indent=2,
            )
        print(f"[INFO] Stats: {out}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
