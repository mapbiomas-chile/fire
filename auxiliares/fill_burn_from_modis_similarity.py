#!/usr/bin/env python3
"""
Fill burn gaps from MODIS candidates that look like existing classified scars.

Logic (per region × year):
  1. Reference profile = dNBR of already-classified burn pixels
  2. Candidates = MODIS burn pixels
  3. Keep only candidates with dNBR very similar to the reference
  4. Block strict LULC (water, bare, rock, salt, ice, urban) — NOT agriculture/pasture
  5. Output = original scar OR accepted MODIS candidates

Original scars are always preserved.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fill_from_modis_similarity")

DEFAULT_CLASS_DIR = Path.home() / "classification_20260713"
DEFAULT_MOSAIC_DIR = Path.home() / "mosaics_cog"
DEFAULT_MODIS_DIR = Path.home() / "MODIS"
DEFAULT_OUTPUT_DIR = Path.home() / "classification_20260713_modis_similar"
DEFAULT_MASCARAS_ROOT = (
    Path.home() / "classification_20260619" / "filtering_work" / "mascaras"
)
REGIONS = ("r1", "r2", "r4", "r6")
DNBR_BAND = 13
MIN_DNBR = 0.10
MAD_K = 1.5

# Strict non-burnable LULC — agriculture (15) and pasture (18) intentionally omitted
ACCUMULATED_STRICT_MASKS = (
    "mascara_alfloramiento_rocoso_acumulado.tif",
    "mascara_arena_playa_duna_acumulado.tif",
    "mascara_salar_acumulado.tif",
    "mascara_hielo_nieve_acumulado.tif",
    "mascara_otra_area_sin_vegetacion_acumulado.tif",
)
YEARLY_STRICT_STEMS = (
    "rio_lago",
    "infraestructura",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add MODIS burn pixels that are spectrally similar to existing "
            "classified scars, excluding strict LULC classes."
        )
    )
    p.add_argument("--class-dir", type=Path, default=DEFAULT_CLASS_DIR)
    p.add_argument("--mosaic-dir", type=Path, default=DEFAULT_MOSAIC_DIR)
    p.add_argument("--modis-dir", type=Path, default=DEFAULT_MODIS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--mascaras-root",
        type=Path,
        default=DEFAULT_MASCARAS_ROOT,
        help="Root with acumuladas/ and by_year/ LULC masks",
    )
    p.add_argument("--regions", nargs="+", default=list(REGIONS))
    p.add_argument("--from-year", type=int, default=2019)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument("--dnbr-band", type=int, default=DNBR_BAND)
    p.add_argument("--min-dnbr", type=float, default=MIN_DNBR)
    p.add_argument(
        "--mad-k",
        type=float,
        default=MAD_K,
        help="Keep MODIS pixels with |dNBR-median| <= mad_k * MAD of classified burns",
    )
    p.add_argument(
        "--modis-pattern",
        default="modis_burned_area_chile_{year}.tif",
    )
    p.add_argument(
        "--class-pattern",
        default="b14_chile_{region}_{year}_classified_filtered_v6.tif",
    )
    p.add_argument(
        "--mosaic-pattern",
        default="b14_chile_{region}_{year}_cog.tif",
    )
    p.add_argument(
        "--modis-burn-min",
        type=float,
        default=1.0,
        help="MODIS values >= this count as burned (default: 1; fits day-of-burn codes)",
    )
    p.add_argument(
        "--lulc-year-fallback",
        type=int,
        default=2024,
        help="If yearly LULC mask missing (e.g. 2025), use this year (default: 2024)",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats-csv", type=Path, default=None)
    p.add_argument("--no-lulc", action="store_true", help="Skip strict LULC blocking")
    return p.parse_args()


def pixel_area_ha(transform) -> float:
    return abs(transform.a * transform.e) / 10_000.0


def reproject_band_to_grid(
    src_path: Path,
    *,
    band: int,
    height: int,
    width: int,
    transform,
    crs,
    resampling=Resampling.nearest,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=0,
        )
    return out


def read_aligned_binary(
    path: Path,
    *,
    height: int,
    width: int,
    transform,
    crs,
    positive_min: float = 1.0,
) -> np.ndarray:
    data = reproject_band_to_grid(
        path,
        band=1,
        height=height,
        width=width,
        transform=transform,
        crs=crs,
        resampling=Resampling.nearest,
    )
    return data >= positive_min


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
    """OR of accumulated bare/rock/ice/salt + yearly water/urban. No ag/pasture."""
    mask = np.zeros((height, width), dtype=bool)
    acum_dir = mascaras_root / "acumuladas"
    year_dir = mascaras_root / "by_year"

    for name in ACCUMULATED_STRICT_MASKS:
        path = acum_dir / name
        if not path.is_file():
            logger.warning("Missing accumulated mask: %s", path)
            continue
        mask |= read_aligned_binary(
            path, height=height, width=width, transform=transform, crs=crs
        )

    lulc_year = year
    for stem in YEARLY_STRICT_STEMS:
        path = year_dir / f"mascara_{stem}_{lulc_year}.tif"
        if not path.is_file() and year_fallback:
            path = year_dir / f"mascara_{stem}_{year_fallback}.tif"
            lulc_year = year_fallback
        if not path.is_file():
            logger.warning("Missing yearly mask: mascara_%s_%s", stem, year)
            continue
        mask |= read_aligned_binary(
            path, height=height, width=width, transform=transform, crs=crs
        )

    return mask


def reference_dnbr_stats(
    scar: np.ndarray,
    dnbr: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    samples = dnbr[scar.astype(bool) & valid]
    if samples.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    if mad < 1e-6:
        mad = float(np.std(samples))
    if mad < 1e-6:
        mad = 0.05
    return median, mad


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
    class_path = args.class_dir / args.class_pattern.format(region=region, year=year)
    mosaic_path = args.mosaic_dir / args.mosaic_pattern.format(region=region, year=year)
    modis_path = args.modis_dir / args.modis_pattern.format(year=year)
    stem = class_path.stem
    out_expanded = args.output_dir / f"{stem}_modis_similar.tif"
    out_added = args.output_dir / f"{stem}_modis_similar_added.tif"

    row = {
        "region": region,
        "year": year,
        "class_path": str(class_path),
        "mosaic_path": str(mosaic_path),
        "modis_path": str(modis_path),
        "output_expanded": str(out_expanded),
        "output_added": str(out_added),
        "status": "pending",
    }

    for label, path in (
        ("class", class_path),
        ("mosaic", mosaic_path),
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

    with rasterio.open(mosaic_path) as mosaic_src:
        profile = mosaic_src.profile.copy()
        transform = mosaic_src.transform
        crs = mosaic_src.crs
        height, width = mosaic_src.height, mosaic_src.width

        # Classification on mosaic grid
        original = np.zeros((height, width), dtype=np.uint8)
        with rasterio.open(class_path) as class_src:
            reproject(
                source=rasterio.band(class_src, 1),
                destination=original,
                src_transform=class_src.transform,
                src_crs=class_src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
                src_nodata=class_src.nodata,
                dst_nodata=0,
            )
        original = (original > 0).astype(bool)

        dnbr = mosaic_src.read(args.dnbr_band).astype(np.float32)
        valid = np.isfinite(dnbr)
        if mosaic_src.nodata is not None:
            valid &= dnbr != mosaic_src.nodata
        dnbr = np.where(valid, dnbr, np.nan)

        modis = read_aligned_binary(
            modis_path,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
            positive_min=args.modis_burn_min,
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

        median, mad = reference_dnbr_stats(original, dnbr, valid)
        thr_lo = thr_hi = float("nan")
        if not np.isfinite(median):
            logger.warning("%s %s: no valid dNBR on scar; copying original", region, year)
            expanded = original.copy()
            added = np.zeros_like(original)
        else:
            thr_lo = max(args.min_dnbr, median - args.mad_k * mad)
            thr_hi = median + args.mad_k * mad
            similar = valid & (dnbr >= thr_lo) & (dnbr <= thr_hi)
            candidates = modis & similar & ~blocked & ~original
            expanded = original | candidates
            added = candidates

        write_uint8(out_expanded, expanded.astype(np.uint8), profile)
        write_uint8(out_added, added.astype(np.uint8), profile)

    px_ha = pixel_area_ha(transform)
    n_orig = int(original.sum())
    n_modis = int(modis.sum())
    n_add = int(added.sum())
    n_final = int(expanded.sum())

    row.update(
        {
            "status": "ok",
            "dnbr_median": median if np.isfinite(median) else None,
            "dnbr_mad": mad if np.isfinite(mad) else None,
            "thr_lo": thr_lo if np.isfinite(thr_lo) else None,
            "thr_hi": thr_hi if np.isfinite(thr_hi) else None,
            "mad_k": args.mad_k,
            "min_dnbr": args.min_dnbr,
            "pixels_original": n_orig,
            "pixels_modis": n_modis,
            "pixels_added": n_add,
            "pixels_final": n_final,
            "pixels_blocked_lulc": int((modis & blocked & ~original).sum()),
            "area_original_ha": n_orig * px_ha,
            "area_added_ha": n_add * px_ha,
            "area_final_ha": n_final * px_ha,
            "pct_increase": (100.0 * n_add / n_orig) if n_orig else float("nan"),
        }
    )
    logger.info(
        "%s %s | med=%.3f mad=%.3f band=[%.3f,%.3f] | orig=%d modis=%d added=%d final=%d",
        region,
        year,
        median if np.isfinite(median) else -1,
        mad if np.isfinite(mad) else -1,
        thr_lo if np.isfinite(thr_lo) else -1,
        thr_hi if np.isfinite(thr_hi) else -1,
        n_orig,
        n_modis,
        n_add,
        n_final,
    )
    return row


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.stats_csv or (args.output_dir / "modis_similar_stats.csv")

    logger.info("class-dir   : %s", args.class_dir)
    logger.info("mosaic-dir  : %s", args.mosaic_dir)
    logger.info("modis-dir   : %s", args.modis_dir)
    logger.info("mascaras    : %s", args.mascaras_root)
    logger.info("output-dir  : %s", args.output_dir)
    logger.info(
        "years %d-%d | mad_k=%.2f | min_dnbr=%.2f | lulc_strict=%s",
        args.from_year,
        args.to_year,
        args.mad_k,
        args.min_dnbr,
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
