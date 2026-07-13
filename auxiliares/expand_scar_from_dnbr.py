#!/usr/bin/env python3
"""
Expand existing burn scars using connected dNBR candidates.

Seeds = original classification (> 0). Candidates = valid dNBR >= MIN_DNBR.
Keep only connected components (8-connectivity) that touch a seed.
No geometric buffer, distance cap, or iteration limit.
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
logger = logging.getLogger("expand_scar_from_dnbr")

# Defaults matching the experiment specification
DEFAULT_CLASS_DIR = Path.home() / "classification_20260713"
DEFAULT_MOSAIC_DIR = Path.home() / "mosaics_cog"
DEFAULT_OUTPUT_DIR = Path.home() / "classification_20260713_dnbr_expanded"
REGIONS = ("r1", "r2", "r4", "r6")
YEARS = range(2019, 2026)
DNBR_BAND = 13
MIN_DNBR = 0.10
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand binary burn scars via 8-connected dNBR>=threshold growth "
            "from original seeds (no buffer / radius / max iterations)."
        )
    )
    parser.add_argument(
        "--class-dir",
        type=Path,
        default=DEFAULT_CLASS_DIR,
        help=f"Binary classifications (default: {DEFAULT_CLASS_DIR})",
    )
    parser.add_argument(
        "--mosaic-dir",
        type=Path,
        default=DEFAULT_MOSAIC_DIR,
        help=f"Multiband mosaics (default: {DEFAULT_MOSAIC_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(REGIONS),
        help=f"Regions to process (default: {' '.join(REGIONS)})",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2019,
        help="First calendar year (default: 2019)",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2025,
        help="Last calendar year (default: 2025)",
    )
    parser.add_argument(
        "--dnbr-band",
        type=int,
        default=DNBR_BAND,
        help=f"1-based dNBR band index (default: {DNBR_BAND})",
    )
    parser.add_argument(
        "--min-dnbr",
        type=float,
        default=MIN_DNBR,
        help=f"Minimum dNBR for candidates (default: {MIN_DNBR})",
    )
    parser.add_argument(
        "--class-pattern",
        default="b14_chile_{region}_{year}_classified_filtered_v6.tif",
        help="Classification filename pattern with {region} and {year}",
    )
    parser.add_argument(
        "--mosaic-pattern",
        default="b14_chile_{region}_{year}_cog.tif",
        help="Mosaic filename pattern with {region} and {year}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip region/year if expanded output already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List expected inputs without writing",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional path for stats CSV (default: <output-dir>/expand_stats.csv)",
    )
    return parser.parse_args()


def pixel_area_ha(transform) -> float:
    """Pixel area in hectares from an affine transform (assumes projected metres)."""
    return abs(transform.a * transform.e) / 10_000.0


def grids_match(src_a: rasterio.DatasetReader, src_b: rasterio.DatasetReader) -> bool:
    return (
        src_a.crs == src_b.crs
        and src_a.width == src_b.width
        and src_a.height == src_b.height
        and src_a.transform == src_b.transform
    )


def read_classification_on_mosaic_grid(
    class_path: Path,
    mosaic_src: rasterio.DatasetReader,
) -> np.ndarray:
    """Read binary classification aligned to mosaic grid (uint8 0/1)."""
    with rasterio.open(class_path) as class_src:
        if grids_match(class_src, mosaic_src):
            data = class_src.read(1)
        else:
            logger.warning(
                "Reprojecting %s to mosaic grid (nearest)",
                class_path.name,
            )
            data = np.zeros(
                (mosaic_src.height, mosaic_src.width),
                dtype=np.uint8,
            )
            reproject(
                source=rasterio.band(class_src, 1),
                destination=data,
                src_transform=class_src.transform,
                src_crs=class_src.crs,
                dst_transform=mosaic_src.transform,
                dst_crs=mosaic_src.crs,
                resampling=Resampling.nearest,
                src_nodata=class_src.nodata,
                dst_nodata=0,
            )
    return (data > 0).astype(np.uint8)


def read_valid_dnbr(
    mosaic_src: rasterio.DatasetReader,
    band: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dnbr float32 with NaN nodata, valid_mask bool)."""
    if band < 1 or band > mosaic_src.count:
        raise ValueError(
            f"dNBR band {band} out of range (1..{mosaic_src.count}) "
            f"in {mosaic_src.name}"
        )
    dnbr = mosaic_src.read(band).astype(np.float32)
    valid = np.ones(dnbr.shape, dtype=bool)

    nodata = mosaic_src.nodata
    if nodata is not None:
        valid &= dnbr != nodata

    # Extra NaN / Inf guards
    valid &= np.isfinite(dnbr)
    dnbr = np.where(valid, dnbr, np.nan)
    return dnbr, valid


def expand_scar_connected(
    original_scar: np.ndarray,
    dnbr: np.ndarray,
    valid_dnbr: np.ndarray,
    *,
    min_dnbr: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Connected-component expansion from original scars through dNBR candidates.

    traversable = original_scar OR (valid & dnbr >= min_dnbr)
    Keep only components that contain at least one original_scar pixel.
    """
    original = original_scar.astype(bool)
    dnbr_candidate = valid_dnbr & (dnbr >= min_dnbr)
    traversable = original | dnbr_candidate

    labels, _n_labels = ndimage.label(traversable, structure=CONNECTIVITY_STRUCTURE)
    seed_labels = np.unique(labels[original])
    seed_labels = seed_labels[seed_labels != 0]

    if seed_labels.size == 0:
        expanded = original.copy()
        added = np.zeros_like(original)
        stats = {
            "dnbr_ge_min": int(dnbr_candidate.sum()),
            "connected_candidates": 0,
            "pixels_added": 0,
        }
        return expanded.astype(np.uint8), added.astype(np.uint8), stats

    connected_to_scar = np.isin(labels, seed_labels)
    expanded = original | connected_to_scar
    added = expanded & ~original

    # Safety: new pixels must be dNBR candidates (seeds may have low/nodata dNBR)
    if added.any() and not np.all(dnbr_candidate[added]):
        bad = int((added & ~dnbr_candidate).sum())
        raise RuntimeError(
            f"Internal error: {bad} added pixels fail dNBR >= {min_dnbr}"
        )

    stats = {
        "dnbr_ge_min": int(dnbr_candidate.sum()),
        "connected_candidates": int((connected_to_scar & dnbr_candidate).sum()),
        "pixels_added": int(added.sum()),
    }
    return expanded.astype(np.uint8), added.astype(np.uint8), stats


def validate_expansion(
    original: np.ndarray,
    expanded: np.ndarray,
    added: np.ndarray,
    dnbr: np.ndarray,
    valid_dnbr: np.ndarray,
    *,
    min_dnbr: float,
) -> None:
    original_b = original.astype(bool)
    expanded_b = expanded.astype(bool)
    added_b = added.astype(bool)

    if not np.all(expanded_b[original_b]):
        raise AssertionError("Some original scar pixels were lost")
    if np.any(added_b & original_b):
        raise AssertionError("added_pixels overlaps original_scar")
    if not np.array_equal(expanded_b, original_b | added_b):
        raise AssertionError("expanded_scar != original_scar OR added_pixels")
    if int(expanded_b.sum()) != int(original_b.sum()) + int(added_b.sum()):
        raise AssertionError("final_pixels != original_pixels + added_pixels")
    if added_b.any():
        if not np.all(valid_dnbr[added_b]):
            raise AssertionError("Some added pixels are dNBR NoData")
        if not np.all(dnbr[added_b] >= min_dnbr):
            raise AssertionError(f"Some added pixels have dNBR < {min_dnbr}")


def write_uint8_mask(
    path: Path,
    data: np.ndarray,
    *,
    profile_template: dict,
) -> None:
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
    class_dir: Path,
    mosaic_dir: Path,
    output_dir: Path,
    class_pattern: str,
    mosaic_pattern: str,
    dnbr_band: int,
    min_dnbr: float,
    skip_existing: bool,
    dry_run: bool,
) -> dict | None:
    class_name = class_pattern.format(region=region, year=year)
    mosaic_name = mosaic_pattern.format(region=region, year=year)
    class_path = class_dir / class_name
    mosaic_path = mosaic_dir / mosaic_name

    stem = Path(class_name).stem
    expanded_path = output_dir / f"{stem}_dnbr_expanded.tif"
    added_path = output_dir / f"{stem}_dnbr_added.tif"

    row = {
        "region": region,
        "year": year,
        "class_path": str(class_path),
        "mosaic_path": str(mosaic_path),
        "expanded_path": str(expanded_path),
        "added_path": str(added_path),
        "status": "pending",
    }

    if not class_path.is_file():
        row["status"] = "missing_class"
        logger.warning("Missing classification: %s", class_path)
        return row
    if not mosaic_path.is_file():
        row["status"] = "missing_mosaic"
        logger.warning("Missing mosaic: %s", mosaic_path)
        return row

    if skip_existing and expanded_path.is_file() and added_path.is_file():
        row["status"] = "skipped_existing"
        logger.info("Skip existing %s %s", region, year)
        return row

    if dry_run:
        row["status"] = "dry_run"
        logger.info("[DRY-RUN] %s %s", region, year)
        return row

    with rasterio.open(mosaic_path) as mosaic_src:
        profile = mosaic_src.profile.copy()
        transform = mosaic_src.transform
        original = read_classification_on_mosaic_grid(class_path, mosaic_src)
        dnbr, valid_dnbr = read_valid_dnbr(mosaic_src, dnbr_band)

        expanded, added, grow_stats = expand_scar_connected(
            original,
            dnbr,
            valid_dnbr,
            min_dnbr=min_dnbr,
        )
        validate_expansion(
            original,
            expanded,
            added,
            dnbr,
            valid_dnbr,
            min_dnbr=min_dnbr,
        )

        write_uint8_mask(expanded_path, expanded, profile_template=profile)
        write_uint8_mask(added_path, added, profile_template=profile)

    px_ha = pixel_area_ha(transform)
    original_px = int(original.sum())
    added_px = int(added.sum())
    final_px = int(expanded.sum())
    area_orig_ha = original_px * px_ha
    area_added_ha = added_px * px_ha
    area_final_ha = final_px * px_ha
    pct_increase = (
        100.0 * added_px / original_px if original_px > 0 else float("nan")
    )

    row.update(
        {
            "status": "ok",
            "pixels_original": original_px,
            "pixels_dnbr_ge_min": grow_stats["dnbr_ge_min"],
            "pixels_connected_candidates": grow_stats["connected_candidates"],
            "pixels_added": added_px,
            "pixels_final": final_px,
            "area_original_ha": area_orig_ha,
            "area_added_ha": area_added_ha,
            "area_final_ha": area_final_ha,
            "pct_increase": pct_increase,
            "pixel_area_ha": px_ha,
        }
    )

    logger.info(
        "%s %s | orig=%d dnbr>=%.2f=%d connected=%d added=%d final=%d | "
        "ha: %.2f + %.2f = %.2f (%%+%.1f)",
        region,
        year,
        original_px,
        min_dnbr,
        grow_stats["dnbr_ge_min"],
        grow_stats["connected_candidates"],
        added_px,
        final_px,
        area_orig_ha,
        area_added_ha,
        area_final_ha,
        pct_increase if original_px > 0 else 0.0,
    )
    return row


def main() -> int:
    args = parse_args()
    years = range(args.from_year, args.to_year + 1)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.stats_csv or (output_dir / "expand_stats.csv")

    logger.info("class-dir   : %s", args.class_dir)
    logger.info("mosaic-dir  : %s", args.mosaic_dir)
    logger.info("output-dir  : %s", output_dir)
    logger.info("regions     : %s", ", ".join(args.regions))
    logger.info("years       : %d–%d", args.from_year, args.to_year)
    logger.info("dNBR band   : %d | min_dnbr=%.3f | connectivity=8", args.dnbr_band, args.min_dnbr)

    rows: list[dict] = []
    for region in args.regions:
        # Normalize region token (accept "1" or "r1")
        region = region if str(region).startswith("r") else f"r{region}"
        for year in years:
            try:
                row = process_one(
                    region=region,
                    year=year,
                    class_dir=args.class_dir,
                    mosaic_dir=args.mosaic_dir,
                    output_dir=output_dir,
                    class_pattern=args.class_pattern,
                    mosaic_pattern=args.mosaic_pattern,
                    dnbr_band=args.dnbr_band,
                    min_dnbr=args.min_dnbr,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                )
            except Exception:
                logger.exception("Failed %s %s", region, year)
                row = {
                    "region": region,
                    "year": year,
                    "status": "error",
                }
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    if not args.dry_run:
        df.to_csv(stats_csv, index=False)
        logger.info("Stats written to %s", stats_csv)

    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    n_err = int((df["status"] == "error").sum()) if not df.empty else 0
    n_miss = int(df["status"].isin(["missing_class", "missing_mosaic"]).sum()) if not df.empty else 0
    logger.info("Done: ok=%d errors=%d missing=%d total=%d", n_ok, n_err, n_miss, len(df))
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
