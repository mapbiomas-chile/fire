#!/usr/bin/env python3
"""
Temporal deduplication of burned-area rasters (per spatial tile).

For each pixel (same row/col across all yearly rasters of one tile), the first
calendar year with a positive class keeps the burn; the same pixel is cleared in
later years (2013 > 2014 > … > 2025).

Optional --spatial-merge: new burns in year Y that are 8-connected to an earlier
scar are attributed to that origin year (dic 2017 / ene 2018 case). Default in the
pipeline is off; use only when you need that extra rule beyond same-pixel dedup.

Input: raw classified tiles (default pipeline) or any per-year binary masks.
Filenames should follow MapBiomas tiles, e.g.
  b14_chile_r1_2013_cog_classified.tif
with the calendar year at token index 3 (0-based) when splitting on "_".
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
from rasterio.enums import Resampling
from rasterio.warp import reproject

YEAR_RE = re.compile(r"(20\d{2})")
_ORIGIN_NONE = np.uint16(0)
_ORIGIN_INF = np.uint32(65535)
_DEFAULT_YEAR_TOKEN_INDEX = 3


def calendar_year_from_path(
    path: Path,
    year_token_index: int | None,
    from_year: int,
    to_year: int,
) -> int:
    """
    Parse calendar year from filename.

    Prefer the 4-digit token at ``year_token_index`` (MapBiomas:
    b14_chile_r1_2013_... → index 3). Fall back to the rightmost 20xx in
    ``from_year``–``to_year`` before ``_filtered_`` so run timestamps are ignored.
    """
    stem = path.stem
    if year_token_index is not None:
        parts = stem.split("_")
        if 0 <= year_token_index < len(parts):
            raw = parts[year_token_index]
            if raw.isdigit() and len(raw) == 4:
                y = int(raw)
                if from_year <= y <= to_year:
                    return y

    before_filtered = stem.split("_filtered_")[0]
    in_range = [
        int(m.group(1))
        for m in YEAR_RE.finditer(before_filtered)
        if from_year <= int(m.group(1)) <= to_year
    ]
    if len(in_range) == 1:
        return in_range[0]
    if len(in_range) > 1:
        return in_range[-1]

    match = YEAR_RE.search(stem)
    if match:
        y = int(match.group(1))
        if from_year <= y <= to_year:
            return y
    raise ValueError(f"Could not parse calendar year in {from_year}-{to_year}: {path.name}")


def _stem_for_tile_key(stem: str) -> str:
    """Drop LULC-filter run suffix so all years of a tile share one group."""
    return stem.split("_filtered_")[0]


def tile_key(path: Path, year_token_index: int | None) -> str:
    """Group all years of the same tile (year token replaced by placeholder)."""
    stem = _stem_for_tile_key(path.stem)
    if year_token_index is not None:
        parts = stem.split("_")
        if 0 <= year_token_index < len(parts) and parts[year_token_index].isdigit():
            parts[year_token_index] = "{YEAR}"
            return "_".join(parts)
    return YEAR_RE.sub("{YEAR}", stem, count=1)


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


def _profiles_match(ref: dict, other: dict) -> bool:
    return (
        ref["crs"] == other["crs"]
        and ref["width"] == other["width"]
        and ref["height"] == other["height"]
        and ref["transform"] == other["transform"]
    )


def read_burn_mask(
    path: Path,
    target_band: int,
    ref_profile: dict | None,
) -> tuple[np.ndarray, dict, float | None]:
    """Read boolean burn mask; reproject to ref grid when CRS/transform differ."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        if ref_profile is None or _profiles_match(ref_profile, profile):
            data = src.read(target_band)
            return is_burned(data, nodata), profile, nodata

        burned_src = is_burned(src.read(target_band), nodata).astype(np.uint8)
        aligned = np.zeros(
            (ref_profile["height"], ref_profile["width"]),
            dtype=np.uint8,
        )
        reproject(
            source=burned_src,
            destination=aligned,
            src_transform=profile["transform"],
            src_crs=profile["crs"],
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
        return aligned.astype(bool), ref_profile.copy(), nodata


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
    burned_by_year: dict[int, np.ndarray],
    years: list[int],
    spatial_merge: bool,
    connectivity: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    First calendar year per pixel (and optional spatial merge of new pixels).

    Returns origin_year grid (uint16, 0 = never burned) and stats.
    """
    shape = next(iter(burned_by_year.values())).shape
    origin_year = np.zeros(shape, dtype=np.uint16)

    stats = {
        "pixels_same_cell_removed": 0,
        "pixels_spatial_merged_to_earlier_year": 0,
        "pixels_new_events": 0,
    }

    for year in years:
        burned = burned_by_year[year]

        same_cell = burned & (origin_year > _ORIGIN_NONE)
        stats["pixels_same_cell_removed"] += int(same_cell.sum())

        new_only = burned & (origin_year == _ORIGIN_NONE)

        if spatial_merge and np.any(origin_year > _ORIGIN_NONE) and np.any(new_only):
            min_orig = min_neighbor_origin(origin_year, connectivity)
            merge = new_only & (min_orig < _ORIGIN_INF)
            if np.any(merge):
                origin_year[merge] = min_orig[merge].astype(np.uint16)
                stats["pixels_spatial_merged_to_earlier_year"] += int(merge.sum())
                new_only &= ~merge

        if np.any(new_only):
            origin_year[new_only] = np.uint16(year)
            stats["pixels_new_events"] += int(new_only.sum())

    return origin_year, stats


def first_burn_masks_pixel_only(
    burned_by_year: dict[int, np.ndarray],
    years: list[int],
) -> dict[int, np.ndarray]:
    """
    Same-pixel priority: output year Y keeps burn only if not burned in any earlier year.
    """
    shape = next(iter(burned_by_year.values())).shape
    ever_burned = np.zeros(shape, dtype=bool)
    keep_by_year: dict[int, np.ndarray] = {}
    for year in years:
        burned = burned_by_year[year]
        keep_by_year[year] = burned & ~ever_burned
        ever_burned |= burned
    return keep_by_year


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
        year_token_index,
    ) = args

    years = sorted(y for y in year_to_path if from_year <= y <= to_year)
    if not years:
        return {"tile": key, "skipped": True, "reason": "no years in range"}

    ref_year = years[0]
    ref_path = year_to_path[ref_year]
    with rasterio.open(ref_path) as ref_src:
        ref_profile = ref_src.profile.copy()

    profiles: dict[int, dict] = {}
    burned_by_year: dict[int, np.ndarray] = {}
    nodata_by_year: dict[int, float | None] = {}
    originals: dict[int, np.ndarray] = {}

    for year in years:
        path = year_to_path[year]
        burned, profile, nodata = read_burn_mask(
            path, target_band, ref_profile if year != ref_year else None
        )
        if year == ref_year:
            ref_profile = profile
        if burned_by_year and burned.shape != next(iter(burned_by_year.values())).shape:
            raise ValueError(f"Shape mismatch in tile {key}: {path.name}")

        with rasterio.open(path) as src:
            originals[year] = src.read(target_band)

        profiles[year] = profile
        burned_by_year[year] = burned
        nodata_by_year[year] = nodata

    if spatial_merge:
        origin_year, assign_stats = assign_origin_year_tile(
            burned_by_year,
            years,
            spatial_merge=True,
            connectivity=connectivity,
        )
        keep_by_year = {y: origin_year == y for y in years}
    else:
        keep_by_year = first_burn_masks_pixel_only(burned_by_year, years)
        assign_stats = {
            "pixels_same_cell_removed": int(
                sum(int((burned_by_year[y] & ~keep_by_year[y]).sum()) for y in years)
            ),
            "pixels_spatial_merged_to_earlier_year": 0,
            "pixels_new_events": int(sum(int(keep_by_year[y].sum()) for y in years)),
        }

    burn_value = int(detect_burn_value(originals, nodata_by_year, years))
    fill_u8 = int(fill_value)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "tile": key,
        "years": years,
        "year_to_input": {str(y): year_to_path[y].name for y in years},
        "burn_value": burn_value,
        "spatial_merge": spatial_merge,
        "assign": assign_stats,
        "pixels_written_by_year": {},
        "pixels_removed_by_year": {},
        "unique_values_by_year": {},
        "output_files": [],
    }

    for year in years:
        keep = keep_by_year[year]
        out = np.full(keep.shape, fill_u8, dtype=np.uint8)
        out[keep] = np.uint8(burn_value)

        burned_before = burned_by_year[year]
        stats["pixels_written_by_year"][year] = int(keep.sum())
        stats["pixels_removed_by_year"][year] = int((burned_before & ~keep).sum())

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
    name_contains: str | None,
    year_token_index: int | None,
) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(input_dir.glob("*.tif")):
        if "_first_burn_year" in path.stem:
            continue
        if name_contains and name_contains not in path.name:
            continue
        try:
            year = calendar_year_from_path(path, year_token_index, from_year, to_year)
        except ValueError as exc:
            print(f"[WARN] Skip: {exc}")
            continue
        key = tile_key(path, year_token_index)
        prev = groups[key].get(year)
        if prev is not None:
            print(
                f"[WARN] Duplicate year {year} for tile {key}: keeping {path.name}, "
                f"skipping {prev.name}"
            )
        groups[key][year] = path
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "First burn year per pixel: if a cell is burned in 2013 and again in 2014, "
            "only 2013 keeps it (2013 > 2014 > …). Optional spatial merge for neighbors."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder with per-year classified GeoTIFFs (e.g. raw classifier output).",
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
        "--year-token-index",
        type=int,
        default=_DEFAULT_YEAR_TOKEN_INDEX,
        help=(
            "0-based '_' token index for calendar year in MapBiomas filenames "
            f"(default: {_DEFAULT_YEAR_TOKEN_INDEX} → b14_chile_r1_2013_...)."
        ),
    )
    parser.add_argument(
        "--spatial-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also merge new burns 8-connected to an earlier scar into that year "
            "(default: off; same-pixel first year only)."
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

    year_token_index = args.year_token_index
    groups = group_inputs(
        input_dir,
        args.from_year,
        args.to_year,
        args.name_contains,
        year_token_index,
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
            year_token_index,
        )
        for key, year_to_path in groups.items()
    ]

    workers = min(args.workers, len(tasks))
    print(f"[INFO] Tile groups: {len(tasks)}")
    if args.name_contains:
        print(f"[INFO] Name filter: contains {args.name_contains!r}")
    print(f"[INFO] Years: {args.from_year}-{args.to_year}")
    print(f"[INFO] Year token index: {year_token_index}")
    print(f"[INFO] Spatial merge: {args.spatial_merge} (connectivity={args.connectivity})")
    print(f"[INFO] Workers: {workers}")

    for key, year_to_path in sorted(groups.items()):
        ys = sorted(year_to_path)
        print(f"[INFO] Group {key}: years {ys[0]}-{ys[-1]} ({len(ys)} files)")

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
            for y in stats["years"]:
                w = stats["pixels_written_by_year"][y]
                r = stats["pixels_removed_by_year"][y]
                if r > 0:
                    print(f"       {y}: kept={w} removed_vs_input={r}")

    if args.stats_json:
        out = Path(args.stats_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "from_year": args.from_year,
                    "to_year": args.to_year,
                    "year_token_index": year_token_index,
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
