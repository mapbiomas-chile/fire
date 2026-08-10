#!/usr/bin/env python3
"""
Validate MapBiomas Chile **season** classification against UNIDOS_13_18 scars.

Both sides use the **fire-season** year (season ending year):
  - Reference ``Season`` (or ``--year-column``) = fire season identity
  - Classification: ``{year}.tif`` or ``{year}_remap.tif`` under
    ``~/classification_20260730`` (band 1 = burn)

Do **not** use calendar-reordered products for this comparison.

Default smoke test: season **2017**.

Example (leftraru)::

  python validation/run_unidos_classification_validation.py --year 2017
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

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.ops import unary_union

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
DEFAULT_CLASS_SEASON = Path.home() / "classification_20260806"
DEFAULT_OUTPUT = Path.home() / "validation" / "unidos_vs_20260806_season"

# polygonize_mask_parallel.py always appends _mask{value}
MASK_VALUE = 1


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
    p.add_argument(
        "--clip-buffer-m",
        type=float,
        default=5000.0,
        help=(
            "Buffer (m) around season scars before clipping national burn raster "
            "for polygonize (default: 5000). Avoids full-Chile polygonize OOM."
        ),
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


def resolve_year_column(columns: list[str], preferred: str) -> str:
    if preferred in columns:
        return preferred
    lower = {c.lower(): c for c in columns}
    if preferred.lower() in lower:
        return lower[preferred.lower()]
    for candidate in ("Season", "season", "year", "Year", "YEAR"):
        if candidate in columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(
        f"Year column {preferred!r} not in reference. Columns: {columns}"
    )


def resolve_season_classification_tif(class_dir: Path, year: int) -> Path:
    """Pick seasonal national mosaic for fire-season year (end year)."""
    candidates = [
        class_dir / f"burned_area_chile_temp_10_remap_{year}.tif",
        class_dir / f"{year}_remap.tif",
        class_dir / f"{year}.tif",
        class_dir / f"burned_area_chile_calendar_{year}.tif",
        class_dir / "calendar" / f"burned_area_chile_calendar_{year}.tif",
    ]
    for path in candidates:
        if path.is_file():
            if "calendar" in path.name.lower() or path.parent.name == "calendar":
                logger.warning(
                    "Using calendar file %s — for UNIDOS prefer season "
                    "burned_area_chile_temp_10_remap_{year}.tif",
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
        del data
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
            driver="GTiff",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(burn, 1)
    logger.info("Binary burn written: %s (burn_px=%d)", out_path, int(burn.sum()))


def clip_raster_to_scars(
    src_tif: Path,
    scar_gdf: gpd.GeoDataFrame,
    out_tif: Path,
    *,
    buffer_m: float,
    dry_run: bool,
) -> None:
    """Crop burn raster to season-scar footprint (+ buffer) so polygonize is feasible."""
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logger.info(
            "[DRY-RUN] clip %s -> %s (buffer=%.0fm, n_scars=%d)",
            src_tif,
            out_tif,
            buffer_m,
            len(scar_gdf),
        )
        return

    if scar_gdf.empty:
        raise RuntimeError("Cannot clip: no scars for this season year.")

    with rasterio.open(src_tif) as src:
        if scar_gdf.crs is None:
            raise ValueError("Scar layer has no CRS for clipping.")
        scars = scar_gdf
        if scars.crs != src.crs:
            scars = scars.to_crs(src.crs)

        geom = unary_union(scars.geometry.values)
        if geom is None or geom.is_empty:
            raise RuntimeError("Scar geometries empty after union.")
        if buffer_m > 0:
            geom = geom.buffer(buffer_m)

        out_image, out_transform = rio_mask(
            src,
            [geom],
            crop=True,
            all_touched=True,
            filled=True,
            nodata=0,
        )
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
            driver="GTiff",
        )
        burn_px = int((out_image[0] > 0).sum())
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(out_image[0].astype(np.uint8, copy=False), 1)

    logger.info(
        "Clipped burn raster: %s (shape=%sx%s, burn_px=%d)",
        out_tif,
        profile["height"],
        profile["width"],
        burn_px,
    )


def load_season_scars(
    ref_gpkg: Path,
    year: int,
    year_column: str,
) -> tuple[gpd.GeoDataFrame, str]:
    gdf = gpd.read_file(ref_gpkg)
    if gdf.empty:
        raise RuntimeError(f"Reference empty: {ref_gpkg}")
    col = resolve_year_column(list(gdf.columns), year_column)

    def _as_year(v) -> int | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return int(v)
        text = str(v).strip()
        if not text:
            return None
        # "2017", "2017.0", "2016-2017" -> first 4 digits
        digits = "".join(ch if ch.isdigit() else " " for ch in text[:10]).split()
        if not digits:
            return None
        y = int(digits[0][:4])
        return y if 1900 <= y <= 2100 else None

    years = gdf[col].map(_as_year)
    counts = years.value_counts(dropna=True).sort_index()
    logger.info("Reference year counts (%s): %s", col, counts.to_dict())

    sub = gdf.loc[years == year].copy()
    if sub.empty:
        raise RuntimeError(
            f"No scars for fire-season year {year} in column {col!r}. "
            f"Available year counts: {counts.to_dict()}"
        )
    logger.info("Season %s scars: %d", year, len(sub))
    return sub, col


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
        "clip": work / "02b_class_albers_clip",
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
        "Mode: SEASON-to-SEASON | Season/year=%s | column=%s | src=%s",
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

    # Stage 2 — reproject reference (shared across seasons)
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

    if not args.dry_run:
        try:
            season_scars, resolved_year_col = load_season_scars(
                ref_gpkg, year, args.year_column
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
    else:
        season_scars = gpd.GeoDataFrame()
        resolved_year_col = args.year_column

    # Stage 3 — reproject classified burn (national)
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

    # Stage 3b — clip to season scars (required for feasible polygonize of national map)
    clip_tif = dirs["clip"] / f"mapbiomas_chile_nat_{year}_albers.tif"
    if args.skip_existing and clip_tif.is_file():
        logger.info("Skip clip (exists): %s", clip_tif)
    else:
        if not args.dry_run and not albers_tif.is_file():
            logger.error("Missing albers raster: %s", albers_tif)
            return 1
        clip_raster_to_scars(
            albers_tif,
            season_scars,
            clip_tif,
            buffer_m=args.clip_buffer_m,
            dry_run=args.dry_run,
        )

    # Stage 4 — polygonize (writes ..._albers_mask1.gpkg)
    poly_gpkg = dirs["poly"] / f"mapbiomas_chile_nat_{year}_albers_mask{MASK_VALUE}.gpkg"
    if args.skip_existing and poly_gpkg.is_file():
        logger.info("Skip polygonize (exists): %s", poly_gpkg)
    else:
        run_cmd(
            [
                py,
                str(REPO_ROOT / "filtering" / "polygonize_mask_parallel.py"),
                "--input-dir",
                str(dirs["clip"]),
                "--output-dir",
                str(dirs["poly"]),
                "--pattern",
                f"mapbiomas_chile_nat_{year}_albers.tif",
                "--band",
                "1",
                "--mask-value",
                str(MASK_VALUE),
                "--workers",
                "1",
            ],
            dry_run=args.dry_run,
        )
        if not args.dry_run and not poly_gpkg.is_file():
            found = list(dirs["poly"].glob("*.gpkg"))
            logger.error(
                "Polygonize did not produce %s. Found in dir: %s",
                poly_gpkg,
                [p.name for p in found],
            )
            return 1

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
                resolved_year_col if not args.dry_run else args.year_column,
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

    # Stage 7 — optional spatial metrics (do not fail the run if this step breaks)
    metrics_dir = work / "06_spatial_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics_dir / f"summary_season_{year}.csv"
    if not (args.skip_existing and summary.is_file()):
        try:
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
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Spatial metrics failed (Jaccard is still valid). exit=%s",
                exc.returncode,
            )

    meta = {
        "mode": "season_to_season",
        "fire_season_year": year,
        "classification_source": str(class_tif),
        "classification_dir": str(class_dir),
        "reference": str(ref),
        "year_column": resolved_year_col if not args.dry_run else args.year_column,
        "n_scars": int(len(season_scars)) if not args.dry_run else None,
        "clip_buffer_m": args.clip_buffer_m,
        "polygon_gpkg": str(poly_gpkg),
        "note": (
            "UNIDOS Season ↔ MapBiomas {year}.tif / {year}_remap.tif "
            "(fire season). National burn is clipped to season scars before "
            "polygonize; polygonize output uses _mask1 suffix."
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
