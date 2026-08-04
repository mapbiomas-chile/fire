#!/usr/bin/env python3
"""
Validate MapBiomas Chile classification against UNIDOS_13_18 reference scars.

Default test case: year 2017, reference ~/validation/UNIDOS_13_18.shp,
classification from season or calendar national mosaics (band 1 = burn).

Pipeline for one calendar year:
  1. Build binary burn GeoTIFF (value 1) from classification band 1 (>0).
  2. Name stem mapbiomas_chile_nat_{year} so year token index = 3.
  3. Reproject burn raster + reference scars to Chile Albers.
  4. Polygonize classified burn.
  5. Intersect scars (Season/year = year) with classified polygons.
  6. Per-scar Jaccard CSV.

Example (2017 test on leftraru)::

  python validation/run_unidos_classification_validation.py \\
    --year 2017 \\
    --reference-shp ~/validation/UNIDOS_13_18.shp \\
    --classification-dir ~/classification_20260730_calendar \\
    --prefer-calendar \\
    --output-root ~/validation/unidos_vs_20260730
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("unidos_validate")

DEFAULT_REFERENCE = Path.home() / "validation" / "UNIDOS_13_18.shp"
DEFAULT_CLASS_SEASON = Path.home() / "classification_20260730"
DEFAULT_CLASS_CALENDAR = Path.home() / "classification_20260730_calendar"
DEFAULT_OUTPUT = Path.home() / "validation" / "unidos_vs_20260730"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UNIDOS_13_18 vs national classification: one-year validation run."
    )
    p.add_argument("--year", type=int, default=2017, help="Calendar year (default: 2017)")
    p.add_argument(
        "--reference-shp",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference scar shapefile (default: ~/validation/UNIDOS_13_18.shp)",
    )
    p.add_argument(
        "--classification-dir",
        type=Path,
        default=None,
        help="Directory with year mosaics (overrides season/calendar defaults).",
    )
    p.add_argument(
        "--prefer-calendar",
        action="store_true",
        help="Prefer burned_area_chile_calendar_{year}.tif when resolving inputs.",
    )
    p.add_argument(
        "--season-dir",
        type=Path,
        default=DEFAULT_CLASS_SEASON,
        help="Season mosaic dir (default: ~/classification_20260730)",
    )
    p.add_argument(
        "--calendar-dir",
        type=Path,
        default=DEFAULT_CLASS_CALENDAR,
        help="Calendar mosaic dir (default: ~/classification_20260730_calendar)",
    )
    p.add_argument(
        "--burn-band",
        type=int,
        default=1,
        help="Band with burn (default: 1)",
    )
    p.add_argument(
        "--year-column",
        default="Season",
        help="Reference attribute for calendar year (default: Season)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Work/output root (default: ~/validation/unidos_vs_20260730)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python for subprocess validators (default: current interpreter)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose primary output already exists",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_classification_tif(args: argparse.Namespace) -> Path:
    year = args.year
    candidates: list[Path] = []

    if args.classification_dir is not None:
        base = args.classification_dir
        candidates.extend(
            [
                base / f"burned_area_chile_calendar_{year}.tif",
                base / f"{year}_remap.tif",
                base / f"{year}.tif",
            ]
        )
    else:
        if args.prefer_calendar:
            candidates.append(
                args.calendar_dir / f"burned_area_chile_calendar_{year}.tif"
            )
        candidates.extend(
            [
                args.calendar_dir / f"burned_area_chile_calendar_{year}.tif",
                args.season_dir / f"{year}_remap.tif",
                args.season_dir / f"{year}.tif",
            ]
        )

    for path in candidates:
        if path.is_file():
            logger.info("Classification source: %s", path)
            return path

    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No classification raster for year {year}. Tried:\n  {tried}"
    )


def write_binary_burn(
    src_path: Path,
    out_path: Path,
    *,
    band: int,
    dry_run: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logger.info("[DRY-RUN] binary burn %s -> %s", src_path, out_path)
        return
    with rasterio.open(src_path) as src:
        data = src.read(band)
        profile = src.profile.copy()
        burn = (data > 0).astype(np.uint8)
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(burn, 1)
    n = int(burn.sum())
    logger.info("Binary burn written: %s (burn_px=%d)", out_path, n)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    logger.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    year = args.year
    py = str(args.python)
    out = args.output_root.expanduser().resolve()
    work = out / f"year_{year}"
    dirs = {
        "named": work / "01_named_binary",
        "albers": work / "02_class_albers",
        "poly": work / "03_class_poly",
        "ref": out / "ref_albers",
        "hits": work / "04_hits",
        "jaccard": work / "05_jaccard",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    ref = args.reference_shp.expanduser().resolve()
    if not ref.is_file():
        logger.error("Reference not found: %s", ref)
        return 1

    try:
        class_tif = resolve_classification_tif(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Stage 1 — binary named raster
    named_tif = dirs["named"] / f"mapbiomas_chile_nat_{year}.tif"
    if args.skip_existing and named_tif.is_file():
        logger.info("Skip binary (exists): %s", named_tif)
    else:
        write_binary_burn(
            class_tif, named_tif, band=args.burn_band, dry_run=args.dry_run
        )

    # Stage 2 — reproject reference (shared across years)
    ref_gpkg = dirs["ref"] / "UNIDOS_13_18_albers.gpkg"
    if args.skip_existing and ref_gpkg.is_file():
        logger.info("Skip reference reproject (exists): %s", ref_gpkg)
    else:
        run_cmd(
            [
                py,
                str(REPO_ROOT / "validation" / "reproject_vector_to_equal_area.py"),
                "--input",
                str(ref),
                "--output",
                str(ref_gpkg),
                "--preset",
                "chile_albers",
            ],
            dry_run=args.dry_run,
        )

    # Stage 3 — reproject classified burn
    albers_tif = dirs["albers"] / f"mapbiomas_chile_nat_{year}_albers.tif"
    if args.skip_existing and albers_tif.is_file():
        logger.info("Skip class reproject (exists): %s", albers_tif)
    else:
        run_cmd(
            [
                py,
                str(REPO_ROOT / "validation" / "reproject_raster_to_equal_area.py"),
                "--input-dir",
                str(dirs["named"]),
                "--output-dir",
                str(dirs["albers"]),
                "--pattern",
                f"mapbiomas_chile_nat_{year}.tif",
                "--preset",
                "chile_albers",
                "--resampling",
                "nearest",
                "--workers",
                "1",
            ],
            dry_run=args.dry_run,
        )

    # Stage 4 — polygonize
    poly_gpkg = dirs["poly"] / f"mapbiomas_chile_nat_{year}_albers.gpkg"
    if not poly_gpkg.is_file():
        # polygonize names from tif stem
        alt = dirs["poly"] / f"mapbiomas_chile_nat_{year}_albers.gpkg"
        poly_gpkg = alt
    if args.skip_existing and poly_gpkg.is_file():
        logger.info("Skip polygonize (exists): %s", poly_gpkg)
    else:
        run_cmd(
            [
                py,
                str(REPO_ROOT / "filtering" / "polygonize_mask_parallel.py"),
                "--input-dir",
                str(dirs["albers"]),
                "--output-dir",
                str(dirs["poly"]),
                "--pattern",
                f"mapbiomas_chile_nat_{year}_albers.tif",
                "--band",
                "1",
                "--mask-value",
                "1",
                "--workers",
                "1",
            ],
            dry_run=args.dry_run,
        )

    # Stage 5 — intersection for this year only
    hits_gpkg = dirs["hits"] / f"unidos_hits_{year}.gpkg"
    if args.skip_existing and hits_gpkg.is_file():
        logger.info("Skip intersect (exists): %s", hits_gpkg)
    else:
        run_cmd(
            [
                py,
                str(REPO_ROOT / "validation" / "intersect_top_n_scars_with_classified.py"),
                "--catalog",
                str(ref_gpkg),
                "--year-column",
                args.year_column,
                "--year",
                str(year),
                "--classified-dir",
                str(dirs["poly"]),
                "--output",
                str(hits_gpkg),
                "--workers",
                str(args.workers),
            ],
            dry_run=args.dry_run,
        )

    # Stage 6 — Jaccard
    jaccard_csv = dirs["jaccard"] / f"unidos_hits_{year}_jaccard.csv"
    if args.skip_existing and jaccard_csv.is_file():
        logger.info("Skip jaccard (exists): %s", jaccard_csv)
    else:
        run_cmd(
            [
                py,
                str(REPO_ROOT / "validation" / "calculate_jaccard_index.py"),
                "--hits-gpkg",
                str(hits_gpkg),
                "--output-csv",
                str(jaccard_csv),
            ],
            dry_run=args.dry_run,
        )

    # Optional spatial metrics for one file
    metrics_dir = work / "06_spatial_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics_dir / f"summary_{year}.csv"
    if not (args.skip_existing and summary.is_file()):
        # spatial_validation_metrics expects a pattern dir
        if not args.dry_run:
            # place a copy/link name matching pattern
            hits_copy = metrics_dir / f"hits_{year}.gpkg"
            if hits_gpkg.is_file() and not hits_copy.is_file():
                shutil.copy2(hits_gpkg, hits_copy)
        run_cmd(
            [
                py,
                str(REPO_ROOT / "validation" / "spatial_validation_metrics.py"),
                "--hits-dir",
                str(metrics_dir),
                "--hits-pattern",
                f"hits_{year}.gpkg",
                "--output-dir",
                str(metrics_dir),
                "--aggregate-summary-csv",
                str(summary),
            ],
            dry_run=args.dry_run,
        )

    meta = {
        "year": year,
        "classification_source": str(class_tif),
        "reference": str(ref),
        "year_column": args.year_column,
        "work_dir": str(work),
        "hits_gpkg": str(hits_gpkg),
        "jaccard_csv": str(jaccard_csv),
        "spatial_summary": str(summary),
    }
    meta_path = work / "run_manifest.json"
    if not args.dry_run:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Done year=%s | hits=%s | jaccard=%s", year, hits_gpkg, jaccard_csv)
    logger.info("Manifest: %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
