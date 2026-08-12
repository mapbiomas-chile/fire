#!/usr/bin/env python3
"""
Recover pre-filter burn pixels confirmed by buffered MODIS (2019–2025).

Layouts:
  regional — final tiles b14_chile_{region}_{year}_classified_filtered_v6.tif
             (classification_20260713)
  national — yearly mosaics burned_area_chile_b14_filtered_v9_{year}.tif
             (classification_20260729); burn class in band 1; prefilter
             regions are OR-merged onto the national grid

Add rules (unchanged):
    candidates = prefilter ∩ MODIS_buffered ∩ ~final
    refine = fill_holes(final ∪ candidates) + closing
    sieved = sieve(added components < min_added_pixels)
    union = final ∪ sieved_added

Then full LULC A1 + A2 on the union:
    A1 = accumulated non-burnable (29,23,61,34,25,33,24)
    A2 = yearly agricultura (15) + pastura (18) with stability window
    out = union ∩ ~(A1 ∪ A2)
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

from lib.sieve_burn_mask import sieve_connected_components  # noqa: E402

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

DEFAULT_MODIS_BUFFER_PX = 3
DEFAULT_MIN_ADDED_PIXELS = 222
DEFAULT_CLOSING_SIZE = 3

NATIONAL_FINAL_PATTERN = "burned_area_chile_b14_filtered_v9_{year}.tif"
REGIONAL_FINAL_PATTERN = "b14_chile_{region}_{year}_classified_filtered_v6.tif"

ACCUMULATED_LULC_A1 = (
    "mascara_alfloramiento_rocoso_acumulado.tif",
    "mascara_arena_playa_duna_acumulado.tif",
    "mascara_salar_acumulado.tif",
    "mascara_hielo_nieve_acumulado.tif",
    "mascara_otra_area_sin_vegetacion_acumulado.tif",
    "mascara_rio_lago_acumulado.tif",
    "mascara_infraestructura_acumulado.tif",
)
# Paso A2 — yearly land-use (stability window already baked into these files)
YEARLY_LULC_A2 = (
    "mascara_agricultura_{year}.tif",
    "mascara_pastura_{year}.tif",
)
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add pre-filter classified burn pixels that overlap buffered MODIS "
            "to the final filtered collection, then fill holes, optionally close "
            "gaps, and sieve small added patches."
        )
    )
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    p.add_argument("--prefilter-dir", type=Path, default=DEFAULT_PREFILTER_DIR)
    p.add_argument("--modis-dir", type=Path, default=DEFAULT_MODIS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mascaras-root", type=Path, default=DEFAULT_MASCARAS_ROOT)
    p.add_argument(
        "--layout",
        choices=("regional", "national"),
        default="regional",
        help=(
            "regional: per-region final tiles; national: yearly Chile mosaics "
            "(v9 multiband, burn in --final-band)."
        ),
    )
    p.add_argument(
        "--final-band",
        type=int,
        default=1,
        help="Band index (1-based) with burn/no-burn in the final raster",
    )
    p.add_argument("--regions", nargs="+", default=list(REGIONS))
    p.add_argument("--from-year", type=int, default=2019)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument(
        "--final-pattern",
        default=None,
        help=(
            "Filename pattern. Defaults: regional "
            f"{REGIONAL_FINAL_PATTERN}; national {NATIONAL_FINAL_PATTERN}."
        ),
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
    p.add_argument("--modis-burn-min", type=float, default=1.0)
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
        "--fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill enclosed holes inward on final U candidates (default: on)",
    )
    p.add_argument(
        "--closing-size",
        type=int,
        default=DEFAULT_CLOSING_SIZE,
        help=(
            "Morphological closing kernel to reconnect near pixels "
            f"(default: {DEFAULT_CLOSING_SIZE}; 0 disables)."
        ),
    )
    p.add_argument(
        "--closing-iterations",
        type=int,
        default=1,
        help="Closing passes (default: 1)",
    )
    p.add_argument(
        "--min-added-pixels",
        type=int,
        default=DEFAULT_MIN_ADDED_PIXELS,
        help=(
            "Drop added connected components smaller than this "
            f"(default: {DEFAULT_MIN_ADDED_PIXELS} ~ 20 ha at 30 m)."
        ),
    )
    p.add_argument(
        "--lulc-year-fallback",
        type=int,
        default=2024,
        help=(
            "If mascara_agricultura/pastura_<year> is missing, try this year "
            "(default: 2024; useful for 2025)."
        ),
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats-csv", type=Path, default=None)
    p.add_argument(
        "--no-lulc",
        action="store_true",
        help="Skip LULC A1+A2 post-filter after adding pixels",
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


def build_lulc_a1_a2_mask(
    mascaras_root: Path,
    year: int,
    *,
    height: int,
    width: int,
    transform,
    crs,
    year_fallback: int | None = 2024,
) -> np.ndarray:
    """LULC A1 (accumulated) ∪ A2 (yearly agri/pasture) for filter year.

    Prefer ``mascara_total_<year>.tif`` when present (already A1∪A2).
    """
    mask = np.zeros((height, width), dtype=bool)
    year_dir = mascaras_root / "by_year"
    acum_dir = mascaras_root / "acumuladas"

    total_candidates = (
        year_dir / f"mascara_total_{year}.tif",
        mascaras_root / f"mascara_total_{year}.tif",
    )
    for total_path in total_candidates:
        if total_path.is_file():
            logger.info("LULC A1+A2 via total mask: %s", total_path.name)
            return align_band_to_ref(
                total_path,
                ref_height=height,
                ref_width=width,
                ref_transform=transform,
                ref_crs=crs,
            )

    # A1 — accumulated
    for name in ACCUMULATED_LULC_A1:
        path = acum_dir / name
        if not path.is_file():
            logger.warning("Missing accumulated mask (A1): %s", path)
            continue
        mask |= align_band_to_ref(
            path,
            ref_height=height,
            ref_width=width,
            ref_transform=transform,
            ref_crs=crs,
        )

    # A2 — yearly agri / pasture
    for pattern in YEARLY_LULC_A2:
        path = year_dir / pattern.format(year=year)
        if not path.is_file() and year_fallback is not None:
            fb = year_dir / pattern.format(year=year_fallback)
            if fb.is_file():
                logger.warning(
                    "Missing %s — using fallback year %s",
                    path.name,
                    year_fallback,
                )
                path = fb
        if not path.is_file():
            logger.warning("Missing yearly mask (A2): %s", path)
            continue
        mask |= align_band_to_ref(
            path,
            ref_height=height,
            ref_width=width,
            ref_transform=transform,
            ref_crs=crs,
        )
    return mask


def refine_union(
    final_burn: np.ndarray,
    raw_added: np.ndarray,
    blocked: np.ndarray,
    *,
    fill_holes: bool,
    closing_size: int,
    closing_iterations: int,
) -> tuple[np.ndarray, dict]:
    working = final_burn | raw_added
    refined = working.copy()
    holes_px = 0
    closed_px = 0

    if fill_holes and working.any():
        filled = ndimage.binary_fill_holes(working)
        all_holes = filled & ~working & ~blocked
        # Fill only holes belonging to scars touched by the recovered pixels.
        hole_labels, _ = ndimage.label(all_holes, structure=CONNECTIVITY_STRUCTURE)
        boundary = ndimage.binary_dilation(
            raw_added, structure=CONNECTIVITY_STRUCTURE, iterations=1
        )
        selected_ids = np.unique(hole_labels[boundary & all_holes])
        selected_ids = selected_ids[selected_ids != 0]
        holes = (
            np.isin(hole_labels, selected_ids)
            if selected_ids.size
            else np.zeros_like(all_holes)
        )
        refined |= holes
        holes_px = int(holes.sum())

    if closing_size and closing_size > 0 and refined.any():
        before = refined.copy()
        structure = np.ones((closing_size, closing_size), dtype=bool)
        closed = refined.copy()
        for _ in range(max(1, closing_iterations)):
            closed = ndimage.binary_closing(closed, structure=structure)
        # Accept closing additions only around the recovered layer, not around
        # unrelated scars already present in the trusted final collection.
        influence = ndimage.binary_dilation(
            raw_added,
            structure=CONNECTIVITY_STRUCTURE,
            iterations=max(1, closing_size // 2),
        )
        closing_added = closed & ~before & influence & ~blocked
        refined |= closing_added
        refined |= final_burn
        closed_px = int(closing_added.sum())

    refined = refined | final_burn
    return refined, {"pixels_filled_holes": holes_px, "pixels_from_closing": closed_px}


def sieve_added_only(
    final_burn: np.ndarray,
    refined: np.ndarray,
    *,
    min_added_pixels: int,
) -> tuple[np.ndarray, dict]:
    added = refined & ~final_burn
    if min_added_pixels <= 1 or not added.any():
        return refined, {
            "components_before": 0,
            "components_after": 0,
            "pixels_removed_sieve": 0,
        }

    sieved, stats = sieve_connected_components(
        added.astype(np.uint8),
        min_pixels=min_added_pixels,
        mask_value=1,
        connectivity=8,
    )
    return final_burn | (sieved == 1), {
        "components_before": stats["components_before"],
        "components_after": stats["components_after"],
        "pixels_removed_sieve": stats["pixels_removed"],
    }


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
    """OR-merge regional prefilter burn masks onto the national reference grid."""
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


def run_recovery(
    *,
    final_burn: np.ndarray,
    prefilter_burn: np.ndarray,
    modis_path: Path,
    height: int,
    width: int,
    transform,
    crs,
    year: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    modis_raw = align_band_to_ref(
        modis_path,
        ref_height=height,
        ref_width=width,
        ref_transform=transform,
        ref_crs=crs,
        positive_min=args.modis_burn_min,
    )
    modis = buffer_mask(modis_raw, args.modis_buffer_px)

    # 1) Add rules unchanged — no LULC yet (blocked empty during refine)
    blocked_none = np.zeros((height, width), dtype=bool)
    raw_added = prefilter_burn & modis & ~final_burn
    refined, refine_stats = refine_union(
        final_burn,
        raw_added,
        blocked_none,
        fill_holes=args.fill_holes,
        closing_size=args.closing_size,
        closing_iterations=args.closing_iterations,
    )
    union, sieve_stats = sieve_added_only(
        final_burn,
        refined,
        min_added_pixels=args.min_added_pixels,
    )
    pixels_added_pre_lulc = int((union & ~final_burn).sum())

    # 2) LULC A1 + A2 on the full union
    blocked = np.zeros((height, width), dtype=bool)
    if not args.no_lulc:
        blocked = build_lulc_a1_a2_mask(
            args.mascaras_root,
            year,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
            year_fallback=args.lulc_year_fallback,
        )
    expanded = union & ~blocked
    added = expanded & ~final_burn

    stats = {
        "pixels_final_before": int(final_burn.sum()),
        "pixels_prefilter": int(prefilter_burn.sum()),
        "pixels_modis_raw": int(modis_raw.sum()),
        "pixels_modis_buffered": int(modis.sum()),
        "pixels_raw_added": int(raw_added.sum()),
        "pixels_filled_holes": refine_stats["pixels_filled_holes"],
        "pixels_from_closing": refine_stats["pixels_from_closing"],
        "pixels_removed_sieve": sieve_stats["pixels_removed_sieve"],
        "sieve_components_before": sieve_stats["components_before"],
        "sieve_components_after": sieve_stats["components_after"],
        "pixels_added_pre_lulc": pixels_added_pre_lulc,
        "pixels_removed_lulc_a1_a2": int((union & blocked).sum()),
        "pixels_added": int(added.sum()),
        "pixels_final_after": int(expanded.sum()),
    }
    return expanded, added, stats


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
        "fill_holes": args.fill_holes,
        "closing_size": args.closing_size,
        "min_added_pixels": args.min_added_pixels,
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
        modis_path=modis_path,
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
        "%s %s | final=%d raw_add=%d holes=+%d close=+%d sieve=-%d added=%d out=%d | +%.1f%%",
        region,
        year,
        n_final,
        stats["pixels_raw_added"],
        stats["pixels_filled_holes"],
        stats["pixels_from_closing"],
        stats["pixels_removed_sieve"],
        n_add,
        stats["pixels_final_after"],
        row["pct_increase"] if n_final else 0.0,
    )
    return row


def process_year_national(
    *,
    year: int,
    args: argparse.Namespace,
) -> dict:
    final_path = args.final_dir / args.final_pattern.format(year=year)
    modis_path = args.modis_dir / args.modis_pattern.format(year=year)

    stem = final_path.stem
    out_expanded = args.output_dir / f"{stem}_prefilter_modis.tif"
    out_added = args.output_dir / f"{stem}_prefilter_modis_added.tif"

    row = {
        "region": "chile",
        "year": year,
        "final_path": str(final_path),
        "prefilter_path": "",
        "modis_path": str(modis_path),
        "output_expanded": str(out_expanded),
        "output_added": str(out_added),
        "modis_buffer_px": args.modis_buffer_px,
        "fill_holes": args.fill_holes,
        "closing_size": args.closing_size,
        "min_added_pixels": args.min_added_pixels,
        "final_band": args.final_band,
        "status": "pending",
    }

    for label, path in (("final", final_path), ("modis", modis_path)):
        if not path.is_file():
            row["status"] = f"missing_{label}"
            logger.warning("Missing %s: %s", label, path)
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
        modis_path=modis_path,
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
        "chile %s | final=%d raw_add=%d holes=+%d close=+%d sieve=-%d added=%d out=%d | +%.1f%%",
        year,
        n_final,
        stats["pixels_raw_added"],
        stats["pixels_filled_holes"],
        stats["pixels_from_closing"],
        stats["pixels_removed_sieve"],
        n_add,
        stats["pixels_final_after"],
        row["pct_increase"] if n_final else 0.0,
    )
    return row


def main() -> int:
    args = parse_args()
    if args.closing_size < 0:
        logger.error("--closing-size must be >= 0")
        return 1
    if args.min_added_pixels < 1:
        logger.error("--min-added-pixels must be >= 1")
        return 1
    if args.final_band < 1:
        logger.error("--final-band must be >= 1")
        return 1

    if args.final_pattern is None:
        args.final_pattern = (
            NATIONAL_FINAL_PATTERN
            if args.layout == "national"
            else REGIONAL_FINAL_PATTERN
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.stats_csv or (args.output_dir / "prefilter_modis_stats.csv")

    logger.info("layout       : %s", args.layout)
    logger.info("final-dir     : %s", args.final_dir)
    logger.info("final-pattern : %s (band %d)", args.final_pattern, args.final_band)
    logger.info("prefilter-dir : %s", args.prefilter_dir)
    logger.info("modis-dir     : %s", args.modis_dir)
    logger.info("mascaras      : %s", args.mascaras_root)
    logger.info("output-dir    : %s", args.output_dir)
    logger.info(
        "years %d-%d | modis_buffer_px=%d | fill_holes=%s | closing=%d | "
        "min_added_px=%d | LULC A1+A2 after add=%s",
        args.from_year,
        args.to_year,
        args.modis_buffer_px,
        args.fill_holes,
        args.closing_size,
        args.min_added_pixels,
        not args.no_lulc,
    )

    rows: list[dict] = []
    if args.layout == "national":
        for year in range(args.from_year, args.to_year + 1):
            try:
                rows.append(process_year_national(year=year, args=args))
            except Exception:
                logger.exception("Failed chile %s", year)
                rows.append({"region": "chile", "year": year, "status": "error"})
    else:
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
