#!/usr/bin/env python3
"""
Filter classified tiles using a year-specific binary mask (1=remove, 0=keep).
"""

import argparse
import json
import re
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def _extract_year_from_name(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.stem)
    if not match:
        raise ValueError(f"Could not find year (20xx) in filename: {path.name}")
    return int(match.group(1))


def _filter_one_file(args):
    (
        tif_path,
        masks_dir,
        mask_band,
        target_band,
        fill_value,
        output_dir,
        run_timestamp,
    ) = args

    year = _extract_year_from_name(tif_path)
    mask_path = masks_dir / f"mascara_total_{year}.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Year mask not found for {tif_path.name}: {mask_path}")

    with rasterio.open(tif_path) as src:
        data = src.read(target_band)
        profile = src.profile.copy()
        dst_shape = (src.height, src.width)
        dst_transform = src.transform
        dst_crs = src.crs

    with rasterio.open(mask_path) as mask_src:
        aligned_mask = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(mask_src, mask_band),
            destination=aligned_mask,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )

    # Binary mask mode: only pixels equal to 1 are filtered.
    block_mask = aligned_mask == 1
    filtered = np.where(block_mask, fill_value, data).astype(data.dtype)

    profile.update(
        count=1,
        dtype=filtered.dtype,
        compress="deflate",
        predictor=2,
        tiled=True,
    )

    output_name = f"{tif_path.stem}_filtered_{run_timestamp}.tif"
    output_path = output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    pixels_filtered_to_zero = int(((data != 0) & (filtered == 0)).sum())
    summary = {
        "input_file": str(tif_path),
        "output_file": str(output_path),
        "mask_file": str(mask_path),
        "year": year,
        "run_timestamp": run_timestamp,
        "mask_mode": "binary",
        "fill_value": fill_value,
        "total_pixels": int(data.size),
        "masked_pixels": int(block_mask.sum()),
        "pixels_filtered_to_zero": pixels_filtered_to_zero,
    }
    summary_path = output_dir / f"{output_path.stem}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    return str(output_path), str(summary_path), pixels_filtered_to_zero


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply year-specific binary masks to every classified TIF in a folder "
            "using multiprocessing."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with source .tif files (e.g. /home/ecastillo/mapbiomas/output/classified).",
    )
    parser.add_argument("--masks-dir", required=True, help="Directory with mascara_total_<year>.tif files.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where filtered rasters are written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of worker processes. Default: cpu_count() - 1.",
    )
    parser.add_argument(
        "--mask-band",
        type=int,
        default=1,
        help="Band index (1-based) to read from mask raster.",
    )
    parser.add_argument(
        "--target-band",
        type=int,
        default=1,
        help="Band index (1-based) to read from each input raster.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0,
        help="Value assigned where binary mask is active (>=1).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tasks = [
        (
            tif_path,
            masks_dir,
            args.mask_band,
            args.target_band,
            args.fill_value,
            output_dir,
            run_timestamp,
        )
        for tif_path in tif_files
    ]

    workers = min(args.workers, len(tasks))
    print(f"[INFO] Found {len(tasks)} tif files.")
    print(f"[INFO] Using {workers} worker processes.")

    with Pool(processes=workers) as pool:
        for output_path, summary_path, filtered_count in pool.imap_unordered(_filter_one_file, tasks):
            print(f"[INFO] Wrote {output_path}")
            print(f"[INFO] Wrote {summary_path} (pixels_filtered_to_zero={filtered_count})")

    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
