#!/usr/bin/env python3
"""
Validate MapBiomas Chile **season** classification against UNIDOS_13_18 scars.

Both sides use the **fire-season** year (season ending year):
  - Reference ``Season`` (or ``--year-column``) = fire season identity
  - Classification: ``{year}.tif`` or ``{year}_remap.tif`` under
    ``~/classification_20260730`` (band 1 = burn)

Do **not** use calendar-reordered products for this comparison: UNIDOS is
seasonal, so matching must be season-to-season.

Default smoke test: season **2017**.

Example (leftraru)::

  python validation/run_unidos_classification_validation.py --year 2017

  # or
  bash validation/run_unidos_validation_year.sh
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
DEFAULT_OUTPUT = Path.home() / "validation" / "unidos_vs_20260730_season"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "UNIDOS_13_18 vs season classification (classification_20260730): "
            "one fire-season year at a time."
        )
    )
    p.add_argument(
        "--year",
        type=int,
        default=2017,
        help="Fire-season year (Season / filename year). Default: 2017",
    )
    p.add_argument(
        "--reference-shp",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference scar shapefile (default: ~/validation/UNIDOS_13_18.shp)",
    )
    p.add_argument(
        "--classification-dir",
        type=Path,
        default=DEFAULT_CLASS_SEASON,
        help="Season mosaic dir (default: ~/classification_20260730)",
    )
    p.add_argument(
        "--burn-band",
        type=int,
        default=1,
        help="Band with burn flag (default: 1)",
    )
    p.add_argument(
        "--year-column",
        default="Season",
        help="Reference field for fire-season year (default: Season)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Work/output root (default: ~/validation/unidos_vs_20260730_season)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python for subprocess tools (default: current interpreter)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose primary output already exists",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_season_classification_tif(class_dir: Path, year: int) -> Path:
    """Pick seasonal national mosaic for fire-season year (end year)."""
    candidates = [
        class_dir / f"{year}_remap.tif",
        class_dir / f"{year}.tif",
        # avoid calendar by default; only if user pointed classification-dir there
        class_dir / f"burned_area_chile_calendar_{year}.tif",
    ]
    for path in candidates:
        if path.is_file():
            if "calendar" in path.name:
                logger.warning(
                    "Using calendar file %s — UNIDOS is seasonal; prefer "
                    "{year}.tif / {year}_remap.tif under classification_20260730",
                    path,
                )
            logger.info("Season classification source: %s", path)
            return path

    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No season classification raster for fire-season year {year}. Tried:\n  {tried}"
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
    logger.info("Binary burn written: %s (burn_px=%d)", out_path, int(burn.sum()))


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
    class_dir = args.classification_dir.expanduser().resolve()
    work = out / f"season_{year}"
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

    if not class_dir.is_dir():
        logger.error("Classification dir not found: %s", class_dir)
        return 1

    try:
        class_tif = resolve_season_classification_tif(class_dir, year)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Mode: SEASON-to-SEASON | Season/year=%s | UNIDOS.%s ↔ %s",
        year,
        args.year_column,
        class_tif.name,
    )

    # Stage 1 — binary named raster (year token index 3 for intersect scripts)
    named_tif = dirs["named"] / f"mapbiomas_chile_nat_{year}.tif"
    if args.skip_existing and named_tif.is_file():
        logger.info("Skip binary (exists): %s", named_tif)
    else:
        write_binary_burn(
            class_tif, named_tif, band=args.burn_band, dry_run=args.dry_run
        )

    # Stage 2 — reproject reference (shared)
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

    # Stage 5 — intersection for this fire-season year only
    hits_gpkg = dirs["hits"] / f"unidos_hits_season_{year}.gpkg"
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
    jaccard_csv = dirs["jaccard"] / f"unidos_hits_season_{year}_jaccard.csv"
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

    # Optional spatial metrics
    metrics_dir = work / "06_spatial_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics_dir / f"summary_season_{year}.csv"
    if not (args.skip_existing and summary.is_file()):
        if not args.dry_run and hits_gpkg.is_file():
            hits_copy = metrics_dir / f"hits_season_{year}.gpkg"
            if not hits_copy.is_file():
                shutil.copy2(hits_gpkg, hits_copy)
        run_cmd(
            [
                py,
                str(REPO_ROOT / "validation" / "spatial_validation_metrics.py"),
                "--hits-dir",
                str(metrics_dir),
                "--hits-pattern",
                f"hits_season_{year}.gpkg",
                "--output-dir",
                str(metrics_dir),
                "--aggregate-summary-csv",
                str(summary),
            ],
            dry_run=args.dry_run,
        )

    meta = {
        "mode": "season_to_season",
        "fire_season_year": year,
        "classification_source": str(class_tif),
        "classification_dir": str(class_dir),
        "reference": str(ref),
        "year_column": args.year_column,
        "note": (
            "UNIDOS Season and MapBiomas {year}.tif / {year}_remap.tif "
            "are both fire-season (season end year), not calendar remap."
        ),
        "work_dir": str(work),
        "hits_gpkg": str(hits_gpkg),
        "jaccard_csv": str(jaccard_csv),
        "spatial_summary": str(summary),
    }
    meta_path = work / "run_manifest.json"
    if not args.dry_run:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Done season=%s | hits=%s | jaccard=%s", year, hits_gpkg, jaccard_csv
    )
    logger.info("Manifest: %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
