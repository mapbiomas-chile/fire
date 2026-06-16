#!/usr/bin/env python3
"""
Fill enclosed agricultural voids inside burn scars (post-LULC step).

After LULC removes agricultural pixels, burn scars may contain holes where
MapBiomas marks cropland. This step sets burn=1 again only for fully enclosed
holes that overlap the yearly agriculture mask (mascara_agricultura_<year>.tif).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.tile_metadata import parse_calendar_year  # noqa: E402


def _binary_profile(profile: dict, nodata: int = 0) -> dict:
    return {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "transform": profile["transform"],
        "crs": profile["crs"],
        "dtype": rasterio.uint8,
        "count": 1,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }


def fill_agricultural_holes(mask: np.ndarray, ag_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return burn mask with enclosed ag holes filled; count of pixels added."""
    burn = mask.astype(bool)
    ag = ag_mask.astype(bool)
    fully_filled = ndimage.binary_fill_holes(burn)
    holes = fully_filled & ~burn
    fill_pixels = holes & ag
    refined = burn | fill_pixels
    return refined.astype(bool), int(fill_pixels.sum())


def _align_ag_mask(
    ag_path: Path,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    aligned = np.zeros(dst_shape, dtype=np.float32)
    with rasterio.open(ag_path) as ag_src:
        reproject(
            source=rasterio.band(ag_src, 1),
            destination=aligned,
            src_transform=ag_src.transform,
            src_crs=ag_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
    return aligned > 0


def process_one_file(args: tuple) -> dict:
    (
        tif_path,
        agriculture_masks_dir,
        output_dir,
        band,
        burn_value,
        fill_value,
        output_stem_suffix,
    ) = args

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    agriculture_masks_dir = Path(agriculture_masks_dir)

    year = parse_calendar_year(tif_path)
    if year is None:
        raise ValueError(f"Could not parse calendar year from: {tif_path.name}")

    ag_path = agriculture_masks_dir / f"mascara_agricultura_{year}.tif"
    if not ag_path.exists():
        raise FileNotFoundError(f"Agriculture mask not found for year {year}: {ag_path}")

    with rasterio.open(tif_path) as src:
        data = src.read(band)
        profile = src.profile.copy()
        nodata = src.nodata
        dst_shape = (src.height, src.width)
        dst_transform = src.transform
        dst_crs = src.crs

    mask = data == burn_value
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask &= data != nodata

    ag_mask = _align_ag_mask(ag_path, dst_shape, dst_transform, dst_crs)
    refined, pixels_added = fill_agricultural_holes(mask, ag_mask)

    out = np.full(refined.shape, fill_value, dtype=np.uint8)
    out[refined] = np.uint8(burn_value)

    stem = tif_path.stem
    if output_stem_suffix and not stem.endswith(output_stem_suffix):
        stem = f"{stem}{output_stem_suffix}"

    out_path = output_dir / f"{stem}.tif"
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **_binary_profile(profile, nodata=int(fill_value))) as dst:
        dst.write(out, 1)

    pixels_before = int(mask.sum())
    pixels_after = int(refined.sum())

    return {
        "input_file": str(tif_path),
        "output_file": str(out_path),
        "year": year,
        "agriculture_mask": str(ag_path),
        "pixels_burned_before": pixels_before,
        "pixels_burned_after": pixels_after,
        "pixels_added": pixels_added,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill enclosed agricultural holes inside burn scars (post-LULC)."
    )
    parser.add_argument("--input-dir", required=True, help="LULC-filtered burn rasters.")
    parser.add_argument("--agriculture-masks-dir", required=True, help="mascara_agricultura_<year>.tif folder.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument("--burn-value", type=int, default=1)
    parser.add_argument("--fill-value", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name-contains", default=None)
    parser.add_argument("--output-stem-suffix", default="")
    parser.add_argument("--stats-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    agriculture_masks_dir = Path(args.agriculture_masks_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not agriculture_masks_dir.is_dir():
        raise FileNotFoundError(f"Agriculture masks dir not found: {agriculture_masks_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if args.name_contains:
        files = [p for p in files if args.name_contains in p.name]
    if not files:
        raise RuntimeError(f"No input files found in {input_dir}")

    task_args = [
        (
            str(path),
            str(agriculture_masks_dir),
            str(output_dir),
            args.band,
            args.burn_value,
            args.fill_value,
            args.output_stem_suffix,
        )
        for path in files
    ]

    summaries: list[dict] = []
    if args.workers <= 1:
        for task in task_args:
            row = process_one_file(task)
            summaries.append(row)
            print(
                f"[INFO] {Path(row['input_file']).name}: +{row['pixels_added']} px "
                f"({row['pixels_burned_before']} → {row['pixels_burned_after']})",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one_file, task): task for task in task_args}
            for fut in as_completed(futures):
                row = fut.result()
                summaries.append(row)
                print(
                    f"[INFO] {Path(row['input_file']).name}: +{row['pixels_added']} px "
                    f"({row['pixels_burned_before']} → {row['pixels_burned_after']})",
                    flush=True,
                )

    total_added = sum(r["pixels_added"] for r in summaries)
    print(f"[INFO] Total agriculture hole pixels filled: {total_added:,}", flush=True)

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps({"files": summaries, "total_pixels_added": total_added}, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Wrote stats: {stats_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
