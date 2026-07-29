#!/usr/bin/env python3
"""
Recover pre-filter burn pixels confirmed by buffered MODIS.

Base product: classification_20260713 (filtered v6).
Candidates: classification_20260619 pre-filter (*_cog_classified.tif)
            AND MODIS burned (optionally dilated to soften blocky edges)
            AND NOT already in the base product.

Optional: block strict LULC (water/urban/bare/rock/salt/ice), keep ag/pasture.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fill_prefilter_modis")

DEFAULT_FINAL_DIR = Path.home() / "classification_20260713"
DEFAULT_PREFILTER_DIR = Path.home() / "classification_20260619"
DEFAULT_MODIS_DIR = Path.home() / "MODIS"
DEFAULT_OUTPUT_DIR = Path.home() / "classification_20260713_prefilter_modis"
DEFAULT_MASCARAS_ROOT = (
    Path.home() / "classification_20260619" / "filtering_work" / "mascaras"
)
REGIONS = ("r1", "r2", "r4", "r6")

# Soften MODIS 500 m blocks on the ~30 m grid (3 px ≈ 90 m)
DEFAULT_MODIS_BUFFER_PX = 3

ACCUMULATED_STRICT_MASKS = (
    "mascara_alfloramiento_rocoso_acumulado.tif",
    "mascara_arena_playa_duna_acumulado.tif",
    "mascara_salar_acumulado.tif",
    "mascara_hielo_nieve_acumulado.tif",
    "mascara_otra_area_sin_vegetacion_acumulado.tif",
)
YEARLY_STRICT_STEMS = ("rio_lago", "infraestructura")
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add pre-filter classified burn pixels that overlap buffered MODIS "
            "to the final filtered collection."
        )
    )
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    p.add_argument("--prefilter-dir", type=Path, default=DEFAULT_PREFILTER_DIR)
    p.add_argument("--modis-dir", type=Path, default=DEFAULT_MODIS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mascaras-root", type=Path, default=DEFAULT_MASCARAS_ROOT)
    p.add_argument("--regions", nargs="+", default=list(REGIONS))
    p.add_argument("--from-year", type=int, default=2019)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument(
        "--final-pattern",
        default="b14_chile_{region}_{year}_classified_filtered_v6.tif",
    )
    p.add_argument(
        "--prefilter-pattern",
        default="b14_chile_{region}_{year}_cog_classified.tif",
        help="Pre-filter classified tiles from classification_20260619",
    )
    p.add_argument(
        "--modis-pattern",
        default="modis_burned_area_chile_{year}.tif",
    )
    p.add_argument(
        "--modis-burn-min",
        type=float,
        default=1.0,
        help="MODIS values >= this count as burned",
    )
    p.add_argument(
        "--modis-buffer-px",
        type=int,
        default=DEFAULT_MODIS_BUFFER_PX,
        help=(
            "Dilate MODIS burn mask by N pixels on the final 30 m grid "
            f"(default: {DEFAULT_MODIS_BUFFER_PX}; use 0 to disable)."
        ),
    )
    p.add_argument(
        "--lulc-year-fallback",
        type=int,
        default=2024,
        help="Yearly LULC fallback if mask missing (e.g. 2025 -> 2024)",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats-csv", type=Path, default=None)
    p.add_argument(
        "--no-lulc",
        action="store_true",
        help="Do not block strict LULC classes",
    )
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


def buffer_mask(mask: np.ndarray, buffer_px: int) -> np.ndarray:
    if buffer_px <= 0 or not mask.any():
        return mask.astype(bool)
    return ndimage.binary_dilation(
        mask.astype(bool),
        structure=CONNECTIVITY_STRUCTURE,
        iterations=int(buffer_px),
    )


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


def process_one(
    *,
    region: str,
    year: int,
    args: argparse.Namespace,
) -> dict:
    final_path = args.final_dir / args.final_pattern.format(region=region, year=year)
    prefilter_path = args.prefilter_dir / args.prefilter_pattern.format(
        region=region, year=year
    )
    modis_path = args.modis_dir / args.modis_pattern.format(year=year)

    stem = final_path.stem
    out_expanded = args.output_dir / f"{stem}_prefilter_modis.tif"
    out_added = args.output_dir / f"{stem}_prefilter_modis_added.tif"

    row = {
        "region": region,
        "year": year,
        "final_path": str(final_path),
        "prefilter_path": str(prefilter_path),
        "modis_path": str(modis_path),
        "output_expanded": str(out_expanded),
        "output_added": str(out_added),
        "modis_buffer_px": args.modis_buffer_px,
        "status": "pending",
    }

    for label, path in (
        ("final", final_path),
        ("prefilter", prefilter_path),
        ("modis", modis_path),
    ):
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
        final_burn = final_src.read(1) > 0

    prefilter_burn = align_band_to_ref(
        prefilter_path,
        ref_height=height,
        ref_width=width,
        ref_transform=transform,
        ref_crs=crs,
    )
    modis_raw = align_band_to_ref(
        modis_path,
        ref_height=height,
        ref_width=width,
        ref_transform=transform,
        ref_crs=crs,
        positive_min=args.modis_burn_min,
    )
    modis = buffer_mask(modis_raw, args.modis_buffer_px)

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

    # prefilter ∩ buffered MODIS ∩ not already final ∩ not strict LULC
    added = prefilter_burn & modis & ~final_burn & ~blocked
    expanded = final_burn | added

    write_uint8(out_expanded, expanded.astype(np.uint8), profile)
    write_uint8(out_added, added.astype(np.uint8), profile)

    px_ha = pixel_area_ha(transform)
    n_final = int(final_burn.sum())
    n_pre = int(prefilter_burn.sum())
    n_modis_raw = int(modis_raw.sum())
    n_modis = int(modis.sum())
    n_add = int(added.sum())
    n_out = int(expanded.sum())
    n_blocked = int((prefilter_burn & modis & ~final_burn & blocked).sum())

    row.update(
        {
            "status": "ok",
            "pixels_final_before": n_final,
            "pixels_prefilter": n_pre,
            "pixels_modis_raw": n_modis_raw,
            "pixels_modis_buffered": n_modis,
            "pixels_blocked_lulc": n_blocked,
            "pixels_added": n_add,
            "pixels_final_after": n_out,
            "area_final_before_ha": n_final * px_ha,
            "area_added_ha": n_add * px_ha,
            "area_final_after_ha": n_out * px_ha,
            "pct_increase": (100.0 * n_add / n_final) if n_final else float("nan"),
        }
    )
    logger.info(
        "%s %s | final=%d pre=%d modis=%d(+buf %d) blocked=%d added=%d out=%d | +%.1f%%",
        region,
        year,
        n_final,
        n_pre,
        n_modis_raw,
        n_modis,
        n_blocked,
        n_add,
        n_out,
        row["pct_increase"] if n_final else 0.0,
    )
    return row


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.stats_csv or (args.output_dir / "prefilter_modis_stats.csv")

    logger.info("final-dir     : %s", args.final_dir)
    logger.info("prefilter-dir : %s", args.prefilter_dir)
    logger.info("modis-dir     : %s", args.modis_dir)
    logger.info("mascaras      : %s", args.mascaras_root)
    logger.info("output-dir    : %s", args.output_dir)
    logger.info(
        "years %d-%d | modis_buffer_px=%d | lulc_strict=%s",
        args.from_year,
        args.to_year,
        args.modis_buffer_px,
        not args.no_lulc,
    )

    rows: list[dict] = []
    for region in args.regions:
        region = region if str(region).startswith("r") else f"r{region}"
        for year in range(args.from_year, args.to_year + 1):
            try:
                rows.append(process_one(region=region, year=year, args=args))
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
