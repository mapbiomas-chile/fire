#!/usr/bin/env python3
"""
Recover burns for 2013–2018 using UNIDOS_13_18 reference scars gated by prefilter.

Base product: classification_20260713 (regional filtered v6) or national v9.
Evidence gate: classification_20260619 pre-filter (*_cog_classified.tif).
Reference: UNIDOS_13_18.shp (column Season = year).

Logic (complete polygon fill):
  Keep a reference polygon only if it overlaps ≥1 prefilter burn pixel.
  Then add the *entire* polygon (same resolution, no buffer) into the final
  product, optionally excluding strict LULC.

    accepted_ref = rasterize(UNIDOS polygons that touch prefilter)
    added = accepted_ref ∩ ~final ∩ ~LULC_strict
    out = final ∪ added

Trusted final scars are never removed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxiliares.fill_raster_from_reference_scars import (  # noqa: E402
    load_reference_by_year,
    select_reference_shapes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fill_prefilter_reference")

DEFAULT_FINAL_DIR = Path.home() / "classification_20260713"
DEFAULT_PREFILTER_DIR = Path.home() / "classification_20260619"
DEFAULT_REFERENCE_SHP = Path.home() / "validation" / "UNIDOS_13_18.shp"
DEFAULT_OUTPUT_DIR = Path.home() / "classification_20260713_prefilter_reference"
DEFAULT_MASCARAS_ROOT = (
    Path.home() / "classification_20260619" / "filtering_work" / "mascaras"
)
REGIONS = ("r1", "r2", "r4", "r6")

NATIONAL_FINAL_PATTERN = "burned_area_chile_b14_filtered_v9_{year}.tif"
REGIONAL_FINAL_PATTERN = "b14_chile_{region}_{year}_classified_filtered_v6.tif"

ACCUMULATED_STRICT_MASKS = (
    "mascara_alfloramiento_rocoso_acumulado.tif",
    "mascara_arena_playa_duna_acumulado.tif",
    "mascara_salar_acumulado.tif",
    "mascara_hielo_nieve_acumulado.tif",
    "mascara_otra_area_sin_vegetacion_acumulado.tif",
    "mascara_rio_lago_acumulado.tif",
    "mascara_infraestructura_acumulado.tif",
)
YEARLY_STRICT_STEMS = ("rio_lago", "infraestructura")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add complete UNIDOS_13_18 reference polygons that touch at least "
            "one pre-filter burn pixel into the final collection (2013–2018)."
        )
    )
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    p.add_argument("--prefilter-dir", type=Path, default=DEFAULT_PREFILTER_DIR)
    p.add_argument("--reference-shp", type=Path, default=DEFAULT_REFERENCE_SHP)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mascaras-root", type=Path, default=DEFAULT_MASCARAS_ROOT)
    p.add_argument(
        "--layout",
        choices=("regional", "national"),
        default="regional",
        help=(
            "regional: per-region final tiles (20260713); national: yearly "
            "Chile mosaics v9 (20260729), burn in --final-band."
        ),
    )
    p.add_argument("--final-band", type=int, default=1)
    p.add_argument("--regions", nargs="+", default=list(REGIONS))
    p.add_argument("--from-year", type=int, default=2013)
    p.add_argument("--to-year", type=int, default=2018)
    p.add_argument("--final-pattern", default=None)
    p.add_argument(
        "--prefilter-pattern",
        default="b14_chile_{region}_{year}_cog_classified.tif",
    )
    p.add_argument("--year-column", default="Season")
    p.add_argument("--lulc-year-fallback", type=int, default=2024)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats-csv", type=Path, default=None)
    p.add_argument("--no-lulc", action="store_true")
    return p.parse_args()


def pixel_area_ha(transform) -> float:
    return abs(transform.a * transform.e) / 10_000.0


def align_band_to_ref(
    src_path: Path,
    *,
    ref_height: int,
    ref_width: int,
    ref_transform,
    ref_crs,
    band: int = 1,
    positive_min: float = 1.0,
) -> np.ndarray:
    out = np.zeros((ref_height, ref_width), dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=0,
        )
    return out >= positive_min


def build_strict_lulc_mask(
    mascaras_root: Path,
    year: int,
    *,
    height: int,
    width: int,
    transform,
    crs,
    year_fallback: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    acum_dir = mascaras_root / "acumuladas"
    year_dir = mascaras_root / "by_year"

    for name in ACCUMULATED_STRICT_MASKS:
        path = acum_dir / name
        if not path.is_file():
            logger.warning("Missing accumulated mask: %s", path)
            continue
        mask |= align_band_to_ref(
            path,
            ref_height=height,
            ref_width=width,
            ref_transform=transform,
            ref_crs=crs,
        )

    for stem in YEARLY_STRICT_STEMS:
        path = year_dir / f"mascara_{stem}_{year}.tif"
        if not path.is_file() and year_fallback:
            path = year_dir / f"mascara_{stem}_{year_fallback}.tif"
        if not path.is_file():
            logger.warning("Missing yearly mask: mascara_%s_%s", stem, year)
            continue
        mask |= align_band_to_ref(
            path,
            ref_height=height,
            ref_width=width,
            ref_transform=transform,
            ref_crs=crs,
        )
    return mask


def merge_prefilter_regions(
    prefilter_dir: Path,
    pattern: str,
    regions: list[str],
    year: int,
    *,
    height: int,
    width: int,
    transform,
    crs,
) -> tuple[np.ndarray, list[str]]:
    merged = np.zeros((height, width), dtype=bool)
    used: list[str] = []
    for region in regions:
        region = region if str(region).startswith("r") else f"r{region}"
        path = prefilter_dir / pattern.format(region=region, year=year)
        if not path.is_file():
            logger.warning("Missing prefilter %s %s: %s", region, year, path)
            continue
        merged |= align_band_to_ref(
            path,
            ref_height=height,
            ref_width=width,
            ref_transform=transform,
            ref_crs=crs,
        )
        used.append(str(path))
    return merged, used


def write_uint8(path: Path, data: np.ndarray, profile_template: dict) -> None:
    profile = profile_template.copy()
    profile.update(
        {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "nodata": 0,
            "compress": "lzw",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8), 1)


def accepted_reference_mask(
    year_gdf: gpd.GeoDataFrame,
    *,
    prefilter_burn: np.ndarray,
    height: int,
    width: int,
    transform,
    crs,
) -> tuple[np.ndarray, int]:
    """Rasterize full UNIDOS polygons that touch ≥1 prefilter burn pixel."""
    shapes = select_reference_shapes(
        year_gdf,
        burn_mask=prefilter_burn,
        out_shape=(height, width),
        transform=transform,
        crs=crs,
        require_overlap=True,
    )
    if not shapes:
        return np.zeros((height, width), dtype=bool), 0
    labels = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return labels == 1, len(shapes)


def run_recovery(
    *,
    final_burn: np.ndarray,
    prefilter_burn: np.ndarray,
    year_gdf: gpd.GeoDataFrame,
    height: int,
    width: int,
    transform,
    crs,
    year: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ref_accepted, n_polys = accepted_reference_mask(
        year_gdf,
        prefilter_burn=prefilter_burn,
        height=height,
        width=width,
        transform=transform,
        crs=crs,
    )

    blocked = np.zeros((height, width), dtype=bool)
    if not args.no_lulc:
        blocked = build_strict_lulc_mask(
            args.mascaras_root,
            year,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
            year_fallback=args.lulc_year_fallback,
        )

    raw_added = ref_accepted & ~final_burn & ~blocked
    expanded = final_burn | raw_added
    added = expanded & ~final_burn

    stats = {
        "polygons_accepted": n_polys,
        "pixels_final_before": int(final_burn.sum()),
        "pixels_prefilter": int(prefilter_burn.sum()),
        "pixels_reference_accepted": int(ref_accepted.sum()),
        "pixels_blocked_lulc": int((ref_accepted & ~final_burn & blocked).sum()),
        "pixels_added": int(added.sum()),
        "pixels_final_after": int(expanded.sum()),
    }
    return expanded, added, stats


def process_one(
    *,
    region: str,
    year: int,
    year_gdf: gpd.GeoDataFrame | None,
    args: argparse.Namespace,
) -> dict:
    final_path = args.final_dir / args.final_pattern.format(region=region, year=year)
    prefilter_path = args.prefilter_dir / args.prefilter_pattern.format(
        region=region, year=year
    )

    stem = final_path.stem
    out_expanded = args.output_dir / f"{stem}_prefilter_reference.tif"
    out_added = args.output_dir / f"{stem}_prefilter_reference_added.tif"

    row = {
        "region": region,
        "year": year,
        "final_path": str(final_path),
        "prefilter_path": str(prefilter_path),
        "output_expanded": str(out_expanded),
        "output_added": str(out_added),
        "status": "pending",
    }

    if year_gdf is None or year_gdf.empty:
        row["status"] = "missing_reference_year"
        logger.warning("No reference polygons for year %s", year)
        return row

    for label, path in (("final", final_path), ("prefilter", prefilter_path)):
        if not path.is_file():
            row["status"] = f"missing_{label}"
            logger.warning("Missing %s: %s", label, path)
            return row

    if args.skip_existing and out_expanded.is_file() and out_added.is_file():
        row["status"] = "skipped_existing"
        return row

    if args.dry_run:
        row["status"] = "dry_run"
        logger.info("[DRY-RUN] %s %s", region, year)
        return row

    with rasterio.open(final_path) as final_src:
        profile = final_src.profile.copy()
        transform = final_src.transform
        crs = final_src.crs
        height, width = final_src.height, final_src.width
        final_burn = final_src.read(args.final_band) > 0

    prefilter_burn = align_band_to_ref(
        prefilter_path,
        ref_height=height,
        ref_width=width,
        ref_transform=transform,
        ref_crs=crs,
    )
    expanded, added, stats = run_recovery(
        final_burn=final_burn,
        prefilter_burn=prefilter_burn,
        year_gdf=year_gdf,
        height=height,
        width=width,
        transform=transform,
        crs=crs,
        year=year,
        args=args,
    )

    write_uint8(out_expanded, expanded.astype(np.uint8), profile)
    write_uint8(out_added, added.astype(np.uint8), profile)

    px_ha = pixel_area_ha(transform)
    n_final = stats["pixels_final_before"]
    n_add = stats["pixels_added"]
    row.update(stats)
    row.update(
        {
            "status": "ok",
            "area_final_before_ha": n_final * px_ha,
            "area_added_ha": n_add * px_ha,
            "area_final_after_ha": stats["pixels_final_after"] * px_ha,
            "pct_increase": (100.0 * n_add / n_final) if n_final else float("nan"),
        }
    )
    logger.info(
        "%s %s | final=%d prefilter=%d polys=%d ref=%d added=%d out=%d | +%.1f%%",
        region,
        year,
        n_final,
        stats["pixels_prefilter"],
        stats["polygons_accepted"],
        stats["pixels_reference_accepted"],
        n_add,
        stats["pixels_final_after"],
        row["pct_increase"] if n_final else 0.0,
    )
    return row


def process_year_national(
    *,
    year: int,
    year_gdf: gpd.GeoDataFrame | None,
    args: argparse.Namespace,
) -> dict:
    final_path = args.final_dir / args.final_pattern.format(year=year)
    stem = final_path.stem
    out_expanded = args.output_dir / f"{stem}_prefilter_reference.tif"
    out_added = args.output_dir / f"{stem}_prefilter_reference_added.tif"

    row = {
        "region": "chile",
        "year": year,
        "final_path": str(final_path),
        "prefilter_path": "",
        "output_expanded": str(out_expanded),
        "output_added": str(out_added),
        "final_band": args.final_band,
        "status": "pending",
    }

    if year_gdf is None or year_gdf.empty:
        row["status"] = "missing_reference_year"
        logger.warning("No reference polygons for year %s", year)
        return row

    if not final_path.is_file():
        row["status"] = "missing_final"
        logger.warning("Missing final: %s", final_path)
        return row

    if args.skip_existing and out_expanded.is_file() and out_added.is_file():
        row["status"] = "skipped_existing"
        return row

    if args.dry_run:
        row["status"] = "dry_run"
        logger.info("[DRY-RUN] chile %s", year)
        return row

    with rasterio.open(final_path) as final_src:
        if args.final_band < 1 or args.final_band > final_src.count:
            row["status"] = "bad_final_band"
            logger.error(
                "Final %s has %d bands; requested band %d",
                final_path,
                final_src.count,
                args.final_band,
            )
            return row
        profile = final_src.profile.copy()
        transform = final_src.transform
        crs = final_src.crs
        height, width = final_src.height, final_src.width
        final_burn = final_src.read(args.final_band) > 0
        logger.info(
            "National %s: %dx%d band=%d burn_px=%d",
            year,
            width,
            height,
            args.final_band,
            int(final_burn.sum()),
        )

    prefilter_burn, used = merge_prefilter_regions(
        args.prefilter_dir,
        args.prefilter_pattern,
        list(args.regions),
        year,
        height=height,
        width=width,
        transform=transform,
        crs=crs,
    )
    row["prefilter_path"] = ";".join(used)
    if not used:
        row["status"] = "missing_prefilter"
        logger.warning("No prefilter tiles for year %s", year)
        return row

    expanded, added, stats = run_recovery(
        final_burn=final_burn,
        prefilter_burn=prefilter_burn,
        year_gdf=year_gdf,
        height=height,
        width=width,
        transform=transform,
        crs=crs,
        year=year,
        args=args,
    )

    write_uint8(out_expanded, expanded.astype(np.uint8), profile)
    write_uint8(out_added, added.astype(np.uint8), profile)

    px_ha = pixel_area_ha(transform)
    n_final = stats["pixels_final_before"]
    n_add = stats["pixels_added"]
    row.update(stats)
    row.update(
        {
            "status": "ok",
            "area_final_before_ha": n_final * px_ha,
            "area_added_ha": n_add * px_ha,
            "area_final_after_ha": stats["pixels_final_after"] * px_ha,
            "pct_increase": (100.0 * n_add / n_final) if n_final else float("nan"),
        }
    )
    logger.info(
        "chile %s | final=%d prefilter=%d polys=%d ref=%d added=%d out=%d | +%.1f%%",
        year,
        n_final,
        stats["pixels_prefilter"],
        stats["polygons_accepted"],
        stats["pixels_reference_accepted"],
        n_add,
        stats["pixels_final_after"],
        row["pct_increase"] if n_final else 0.0,
    )
    return row


def main() -> int:
    args = parse_args()
    if args.final_band < 1:
        logger.error("--final-band must be >= 1")
        return 1
    if args.from_year > args.to_year:
        logger.error("--from-year must be <= --to-year")
        return 1

    if args.final_pattern is None:
        args.final_pattern = (
            NATIONAL_FINAL_PATTERN
            if args.layout == "national"
            else REGIONAL_FINAL_PATTERN
        )

    if not args.reference_shp.is_file():
        logger.error("Reference shapefile not found: %s", args.reference_shp)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.stats_csv or (args.output_dir / "prefilter_reference_stats.csv")

    logger.info("layout         : %s", args.layout)
    logger.info("final-dir      : %s", args.final_dir)
    logger.info("final-pattern  : %s (band %d)", args.final_pattern, args.final_band)
    logger.info("prefilter-dir  : %s", args.prefilter_dir)
    logger.info("reference-shp  : %s", args.reference_shp)
    logger.info("mascaras       : %s", args.mascaras_root)
    logger.info("output-dir     : %s", args.output_dir)
    logger.info(
        "years %d-%d | complete UNIDOS fill if prefilter hit | lulc_strict=%s",
        args.from_year,
        args.to_year,
        not args.no_lulc,
    )

    logger.info("Loading reference scars…")
    by_year = load_reference_by_year(
        args.reference_shp,
        year_column=args.year_column,
    )
    for y in range(args.from_year, args.to_year + 1):
        n = len(by_year[y]) if y in by_year else 0
        logger.info("  reference year %s: %d polygons", y, n)

    rows: list[dict] = []
    if args.layout == "national":
        for year in range(args.from_year, args.to_year + 1):
            try:
                rows.append(
                    process_year_national(
                        year=year,
                        year_gdf=by_year.get(year),
                        args=args,
                    )
                )
            except Exception:
                logger.exception("Failed chile %s", year)
                rows.append({"region": "chile", "year": year, "status": "error"})
    else:
        for region in args.regions:
            region = region if str(region).startswith("r") else f"r{region}"
            for year in range(args.from_year, args.to_year + 1):
                try:
                    rows.append(
                        process_one(
                            region=region,
                            year=year,
                            year_gdf=by_year.get(year),
                            args=args,
                        )
                    )
                except Exception:
                    logger.exception("Failed %s %s", region, year)
                    rows.append({"region": region, "year": year, "status": "error"})

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
