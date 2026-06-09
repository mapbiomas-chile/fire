#!/usr/bin/env python3
"""
Refine burned-area masks: fill internal holes and/or morphological closing.

Methods:
  fill_holes  — fill fully enclosed 0-pixels inside scars (no outer boundary change)
  closing     — morphological closing (also affects edges; use if fill_holes is not enough)
  both        — fill_holes then closing

Output remains uint8 0/1 with clean GeoTIFF profile.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REFINE_METHODS = ("fill_holes", "closing", "both")


def burn_mask(data: np.ndarray, burn_value: int, nodata: float | None) -> np.ndarray:
    mask = data == burn_value
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask &= data != nodata
    return mask


def fill_holes_limited(mask: np.ndarray, max_hole_area: int | None) -> np.ndarray:
    """
    Fill enclosed background pixels inside the burn mask.

    If max_hole_area is set (>0), only holes with at most that many pixels are filled.
    Outer boundary is unchanged in all cases.
    """
    if max_hole_area is None or max_hole_area <= 0:
        return ndimage.binary_fill_holes(mask)

    fully_filled = ndimage.binary_fill_holes(mask)
    added = fully_filled & ~mask
    if not np.any(added):
        return mask.copy()

    labeled, n = ndimage.label(added)
    refined = mask.copy()
    for label_id in range(1, n + 1):
        component = labeled == label_id
        if int(component.sum()) <= max_hole_area:
            refined[component] = True
    return refined


def apply_refine(
    mask: np.ndarray,
    method: str,
    max_hole_area: int | None,
    closing_size: int,
    iterations: int,
) -> np.ndarray:
    if method not in REFINE_METHODS:
        raise ValueError(f"method must be one of {REFINE_METHODS}")
    if closing_size < 1:
        raise ValueError("closing_size must be >= 1")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    refined = mask.astype(bool)
    if method in ("fill_holes", "both"):
        refined = fill_holes_limited(refined, max_hole_area)
    if method in ("closing", "both"):
        structure = np.ones((closing_size, closing_size), dtype=bool)
        for _ in range(iterations):
            refined = ndimage.binary_closing(refined, structure=structure)
    return refined


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


def refine_one_file(args: tuple) -> dict:
    (
        tif_path,
        output_dir,
        band,
        burn_value,
        fill_value,
        method,
        max_hole_area,
        closing_size,
        iterations,
        output_stem_suffix,
    ) = args

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)

    with rasterio.open(tif_path) as src:
        data = src.read(band)
        profile = src.profile.copy()
        nodata = src.nodata

    mask = burn_mask(data, burn_value, nodata)
    refined = apply_refine(mask, method, max_hole_area, closing_size, iterations)
    out = np.full(refined.shape, fill_value, dtype=np.uint8)
    out[refined] = np.uint8(burn_value)

    pixels_before = int(mask.sum())
    pixels_after = int(refined.sum())
    pixels_added = int((refined & ~mask).sum())
    pixels_removed = int((mask & ~refined).sum())

    stem = tif_path.stem
    if output_stem_suffix and not stem.endswith(output_stem_suffix):
        stem = f"{stem}{output_stem_suffix}"

    out_path = output_dir / f"{stem}.tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **_binary_profile(profile, nodata=int(fill_value))) as dst:
        dst.write(out, 1)

    return {
        "input_file": str(tif_path),
        "output_file": str(out_path),
        "method": method,
        "max_hole_area": max_hole_area,
        "closing_size": closing_size,
        "iterations": iterations,
        "pixels_burned_before": pixels_before,
        "pixels_burned_after": pixels_after,
        "pixels_added": pixels_added,
        "pixels_removed": pixels_removed,
    }


def collect_inputs(
    input_dir: Path,
    pattern: str,
    name_contains: str | None,
    skip_suffix: str | None,
) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if name_contains and name_contains not in path.name:
            continue
        if skip_suffix and path.stem.endswith(skip_suffix):
            continue
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refine burned-area masks: fill internal holes (default) or morphological closing."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Folder with filtered uint8 masks.")
    parser.add_argument("--output-dir", required=True, help="Output folder for refined rasters.")
    parser.add_argument("--pattern", default="*.tif", help="Input glob (default: *.tif).")
    parser.add_argument(
        "--name-contains",
        default=None,
        help="Only process files whose name includes this substring.",
    )
    parser.add_argument("--band", type=int, default=1, help="Band index (1-based).")
    parser.add_argument("--burn-value", type=int, default=1, help="Burn class value (default: 1).")
    parser.add_argument("--fill-value", type=int, default=0, help="Background value (default: 0).")
    parser.add_argument(
        "--method",
        choices=REFINE_METHODS,
        default="fill_holes",
        help=(
            "fill_holes: enclosed gaps only, boundary unchanged (default). "
            "closing: morphological closing. both: fill_holes then closing."
        ),
    )
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=16,
        help=(
            "For fill_holes/both: max enclosed hole size in pixels to fill "
            "(default: 16 ≈ 4x4). Use 0 for unlimited (aggressive)."
        ),
    )
    parser.add_argument(
        "--closing-size",
        type=int,
        default=2,
        help="Closing kernel side in pixels (used for closing/both; default: 2).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Closing passes (used for closing/both; default: 1).",
    )
    parser.add_argument(
        "--output-stem-suffix",
        default="_filled",
        help="Append to output stem (default: _filled). Use '' to keep the input stem.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stats-json", default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    skip_suffix = args.output_stem_suffix or None
    paths = collect_inputs(input_dir, args.pattern, args.name_contains, skip_suffix)
    if not paths:
        msg = f"No files matching {args.pattern!r} in {input_dir}"
        if args.name_contains:
            msg += f" (name contains {args.name_contains!r})"
        raise RuntimeError(msg)

    max_hole_area = None if args.max_hole_area <= 0 else args.max_hole_area

    tasks = [
        (
            str(p),
            str(output_dir),
            args.band,
            args.burn_value,
            args.fill_value,
            args.method,
            max_hole_area,
            args.closing_size,
            args.iterations,
            args.output_stem_suffix,
        )
        for p in paths
    ]

    workers = min(args.workers, len(tasks))
    print(f"[INFO] Files: {len(tasks)} | method={args.method}", flush=True)
    if args.method in ("fill_holes", "both"):
        if max_hole_area is None:
            print("[INFO] fill_holes: unlimited (all enclosed holes)", flush=True)
        else:
            print(f"[INFO] fill_holes: max hole area = {max_hole_area} px", flush=True)
    if args.method in ("closing", "both"):
        print(
            f"[INFO] Closing: {args.closing_size}x{args.closing_size} x{args.iterations}",
            flush=True,
        )
    print(f"[INFO] Workers: {workers}", flush=True)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(refine_one_file, t): t for t in tasks}
        for fut in as_completed(futures):
            stats = fut.result()
            results.append(stats)
            print(
                f"[INFO] {Path(stats['input_file']).name}: "
                f"+{stats['pixels_added']} px "
                f"(burned {stats['pixels_burned_before']} -> {stats['pixels_burned_after']})",
                flush=True,
            )

    if args.stats_json:
        out = Path(args.stats_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "method": args.method,
            "max_hole_area": max_hole_area,
            "closing_size": args.closing_size,
            "iterations": args.iterations,
            "n_files": len(results),
            "total_pixels_added": sum(r["pixels_added"] for r in results),
            "files": results,
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Stats: {out}", flush=True)

    print("[INFO] Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
