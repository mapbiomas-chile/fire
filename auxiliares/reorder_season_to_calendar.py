#!/usr/bin/env python3
"""
Rebuild fire-season mosaics into calendar-year mosaics using the month band.

Input: classification_20260730 — one 3-band TIF per season, named by the
ending year ({year}.tif or {year}_remap.tif):
  band 1 = burned / not burned
  band 2 = month of occurrence (10, 11, 12, 1, 2, 3, 4)
  band 3 = surface reclassification

Season file Y covers Oct(Y-1) … Apr(Y). Calendar year Y is therefore:
  months 1-4  from season file Y          (stay)
  months 10-12 from season file Y+1       (brought back one file)

Rules agreed with the user:
  * First year (default 2013): months 10-12 (real Oct-Dec 2012) are kept in
    2013 with the month band rewritten to 1 (January).
  * Last year (default 2025): stays partial (Jan-Apr only), since no season
    file Y+1 exists to contribute Oct-Dec.
  * Conflicts (pixel burned in both contributions of the same calendar year):
    the earliest event wins, i.e. months 1-4 take precedence over 10-12.
  * Burned pixels whose month is outside {1,2,3,4,10,11,12} stay in the base
    year unchanged (counted in stats).

All 3 bands travel together from the source file of each pixel.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reorder_season_to_calendar")

DEFAULT_INPUT_DIR = Path.home() / "classification_20260730"
DEFAULT_OUTPUT_DIR = Path.home() / "classification_20260730_calendar"

EARLY_MONTHS = (1, 2, 3, 4)
LATE_MONTHS = (10, 11, 12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild season mosaics (Oct-Apr, named by ending year) into "
            "calendar-year mosaics using the month band."
        )
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--from-year", type=int, default=2013)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument(
        "--first-year-late-month",
        type=int,
        default=1,
        help=(
            "Month value written for late-season pixels (10-12) kept in the "
            "first year (default: 1 = January)."
        ),
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats-csv", type=Path, default=None)
    return p.parse_args()


def find_season_file(input_dir: Path, year: int) -> Path | None:
    for name in (f"{year}.tif", f"{year}_remap.tif"):
        path = input_dir / name
        if path.is_file():
            return path
    return None


def read_season(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"{path} has {src.count} bands; expected 3")
        fire = src.read(1)
        month = src.read(2)
        surface = src.read(3)
        profile = src.profile.copy()
    return fire, month, surface, profile


def same_grid(profile_a: dict, profile_b: dict) -> bool:
    return (
        profile_a["width"] == profile_b["width"]
        and profile_a["height"] == profile_b["height"]
        and profile_a["transform"] == profile_b["transform"]
        and profile_a["crs"] == profile_b["crs"]
    )


def read_season_aligned(
    path: Path,
    ref_profile: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the 3 season bands reprojected (nearest) onto the reference grid."""
    height = ref_profile["height"]
    width = ref_profile["width"]
    bands: list[np.ndarray] = []
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"{path} has {src.count} bands; expected 3")
        for b in (1, 2, 3):
            out = np.zeros((height, width), dtype=src.dtypes[b - 1])
            reproject(
                source=rasterio.band(src, b),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=0,
            )
            bands.append(out)
    return bands[0], bands[1], bands[2]


def process_year(
    *,
    year: int,
    args: argparse.Namespace,
) -> dict:
    base_path = find_season_file(args.input_dir, year)
    next_path = find_season_file(args.input_dir, year + 1)
    out_path = args.output_dir / f"burned_area_chile_calendar_{year}.tif"

    row = {
        "year": year,
        "season_file": str(base_path) if base_path else "",
        "next_season_file": str(next_path) if next_path else "",
        "output": str(out_path),
        "status": "pending",
    }

    if base_path is None:
        row["status"] = "missing_season"
        logger.warning("Missing season file for %s", year)
        return row

    if args.skip_existing and out_path.is_file():
        row["status"] = "skipped_existing"
        return row

    if args.dry_run:
        row["status"] = "dry_run"
        logger.info(
            "[DRY-RUN] %s <- %s (months 1-4) + %s (months 10-12)",
            year,
            base_path.name,
            next_path.name if next_path else "none",
        )
        return row

    fire, month, surface, profile = read_season(base_path)
    burned = fire > 0
    early = burned & np.isin(month, EARLY_MONTHS)
    late = burned & np.isin(month, LATE_MONTHS)
    other = burned & ~early & ~late

    is_first_year = year == args.from_year
    if is_first_year:
        # Oct-Dec of (from_year - 1): keep in the first year, month -> January.
        keep = early | other | late
        out_month = np.where(late, args.first_year_late_month, month)
        kept_late_first_year = int(late.sum())
    else:
        keep = early | other
        out_month = month
        kept_late_first_year = 0

    out_fire = np.where(keep, fire, 0)
    out_month = np.where(keep, out_month, 0).astype(month.dtype)
    out_surface = np.where(keep, surface, 0)

    received = 0
    conflicts = 0
    if next_path is not None:
        with rasterio.open(next_path) as nxt:
            next_profile = nxt.profile.copy()
        if same_grid(profile, next_profile):
            next_fire, next_month, next_surface, _ = read_season(next_path)
        else:
            logger.warning(
                "Grid mismatch %s vs %s; aligning to base grid (nearest)",
                base_path.name,
                next_path.name,
            )
            next_fire, next_month, next_surface = read_season_aligned(
                next_path, profile
            )
        incoming = (next_fire > 0) & np.isin(next_month, LATE_MONTHS)
        conflicts = int((incoming & keep).sum())
        add = incoming & ~keep  # earliest event (months 1-4) wins
        out_fire = np.where(add, next_fire, out_fire)
        out_month = np.where(add, next_month, out_month).astype(month.dtype)
        out_surface = np.where(add, next_surface, out_surface)
        received = int(add.sum())
        del next_fire, next_month, next_surface

    profile.update(
        {
            "driver": "GTiff",
            "count": 3,
            "compress": "lzw",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_fire.astype(profile["dtype"]), 1)
        dst.write(out_month.astype(profile["dtype"]), 2)
        dst.write(out_surface.astype(profile["dtype"]), 3)

    n_out = int((out_fire > 0).sum())
    row.update(
        {
            "status": "ok",
            "pixels_season_total": int(burned.sum()),
            "pixels_kept_early": int(early.sum()),
            "pixels_moved_to_prev_year": 0 if is_first_year else int(late.sum()),
            "pixels_kept_late_first_year": kept_late_first_year,
            "pixels_month_other": int(other.sum()),
            "pixels_received_from_next": received,
            "pixels_conflict_kept_early": conflicts,
            "pixels_calendar_total": n_out,
        }
    )
    logger.info(
        "%s | season=%d early=%d late_out=%d other=%d received=%d conflicts=%d -> calendar=%d",
        year,
        int(burned.sum()),
        int(early.sum()),
        0 if is_first_year else int(late.sum()),
        int(other.sum()),
        received,
        conflicts,
        n_out,
    )
    return row


def main() -> int:
    args = parse_args()
    if args.from_year > args.to_year:
        logger.error("--from-year must be <= --to-year")
        return 1
    if not args.input_dir.is_dir():
        logger.error("Input dir not found: %s", args.input_dir)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.stats_csv or (args.output_dir / "season_to_calendar_stats.csv")

    logger.info("input-dir  : %s", args.input_dir)
    logger.info("output-dir : %s", args.output_dir)
    logger.info(
        "years %d-%d | first-year late months -> month %d | earliest wins on conflict",
        args.from_year,
        args.to_year,
        args.first_year_late_month,
    )

    rows: list[dict] = []
    for year in range(args.from_year, args.to_year + 1):
        try:
            rows.append(process_year(year=year, args=args))
        except Exception:
            logger.exception("Failed %s", year)
            rows.append({"year": year, "status": "error"})

    df = pd.DataFrame(rows)
    if not args.dry_run:
        df.to_csv(stats_csv, index=False)
        logger.info("Stats: %s", stats_csv)

    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    n_err = int((df["status"] == "error").sum()) if not df.empty else 0
    logger.info("Done: ok=%d errors=%d total=%d", n_ok, n_err, len(df))
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
