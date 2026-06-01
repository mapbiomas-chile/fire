#!/usr/bin/env python3
"""
Keep burned pixels only in the first calendar year they appear (per tile).

Example: if the same pixel is classified as burned in 2017, 2018 and 2019,
only 2017 keeps the burn; 2018 and 2019 are set to the background value.

Tiles are grouped by filename with the year token replaced (same spatial tile,
different years). Process years in order from --from-year to --to-year.
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


def extract_year(path: Path) -> int:
    match = YEAR_RE.search(path.stem)
    if not match:
        raise ValueError(f"No year (20xx) in filename: {path.name}")
    return int(match.group(1))


def tile_key(path: Path) -> str:
    """Group key: stem with the first 20xx replaced by a placeholder."""
    return YEAR_RE.sub("{YEAR}", path.stem, count=1)


def is_burned(data: np.ndarray, nodata: float | None) -> np.ndarray:
    burned = data != 0
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        burned &= data != nodata
    return burned


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
    ) = args

    years = sorted(y for y in year_to_path if from_year <= y <= to_year)
    if not years:
        return {"tile": key, "skipped": True, "reason": "no years in range"}

    first_year_grid: np.ndarray | None = None
    profiles: dict[int, dict] = {}
    originals: dict[int, np.ndarray] = {}

    for year in years:
        path = year_to_path[year]
        with rasterio.open(path) as src:
            data = src.read(target_band)
            profile = src.profile.copy()
            nodata = src.nodata

        if first_year_grid is None:
            first_year_grid = np.zeros(data.shape, dtype=np.uint16)
        elif data.shape != first_year_grid.shape:
            raise ValueError(
                f"Shape mismatch for tile {key}: {path.name} {data.shape} "
                f"vs {first_year_grid.shape}"
            )

        profiles[year] = profile
        originals[year] = data
        burned = is_burned(data, nodata)
        newly = burned & (first_year_grid == 0)
        first_year_grid[newly] = year

    assert first_year_grid is not None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "tile": key,
        "years": years,
        "pixels_removed_by_year": {},
        "output_files": [],
    }

    for year in years:
        data = originals[year]
        keep = first_year_grid == year
        out = np.where(keep, data, fill_value).astype(data.dtype)
        burned = is_burned(data, profile.get("nodata"))
        removed = int((burned & ~keep).sum())

        profile = profiles[year]
        profile.update(count=1, compress="deflate", predictor=2, tiled=True)

        out_name = f"{year_to_path[year].stem}{suffix}.tif"
        out_path = output_dir / out_name
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)

        stats["pixels_removed_by_year"][year] = removed
        stats["output_files"].append(str(out_path))

    return stats


def group_inputs(input_dir: Path, from_year: int, to_year: int) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(input_dir.glob("*.tif")):
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
            "Assign each burned pixel to its first year of detection only "
            "(removes persistence in later years)."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Folder with per-year classified GeoTIFFs.")
    parser.add_argument("--output-dir", required=True, help="Where to write deduplicated rasters.")
    parser.add_argument("--from-year", type=int, default=2013, help="First year in the series (default: 2013).")
    parser.add_argument(
        "--to-year",
        type=int,
        default=2025,
        help="Last year in the series, inclusive (default: 2025).",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0,
        help="Value for pixels burned in a later year but first seen earlier (default: 0).",
    )
    parser.add_argument("--target-band", type=int, default=1, help="Band to read (1-based, default: 1).")
    parser.add_argument(
        "--suffix",
        default="_first_burn_year",
        help="Inserted before .tif in output names (default: _first_burn_year).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Parallel workers (one spatial tile per task).",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to write aggregate JSON stats.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year")

    groups = group_inputs(input_dir, args.from_year, args.to_year)
    if not groups:
        raise RuntimeError(f"No .tif files with years {args.from_year}-{args.to_year} in {input_dir}")

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
        )
        for key, year_to_path in groups.items()
    ]

    workers = min(args.workers, len(tasks))
    print(f"[INFO] Tile groups: {len(tasks)}")
    print(f"[INFO] Years: {args.from_year}-{args.to_year}")
    print(f"[INFO] Workers: {workers}")

    all_stats: list[dict] = []
    with Pool(processes=workers) as pool:
        for stats in pool.imap_unordered(process_tile_group, tasks):
            if stats.get("skipped"):
                print(f"[WARN] Skipped {stats['tile']}: {stats.get('reason')}")
                continue
            all_stats.append(stats)
            removed = stats["pixels_removed_by_year"]
            total_removed = sum(removed.values())
            print(f"[INFO] {stats['tile']}: removed {total_removed} pixel-years total")

    if args.stats_json:
        out = Path(args.stats_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "from_year": args.from_year,
                    "to_year": args.to_year,
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
