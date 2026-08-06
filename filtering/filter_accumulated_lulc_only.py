#!/usr/bin/env python3
"""
Apply **only accumulated LULC masks** (group A) to national/season mosaics.

Blocks burn where any of these MapBiomas classes appear in *any* LULC year:
  29 rock, 23 sand, 61 salt, 34 ice/snow, 25 bare, 33 water, 24 infrastructure.

Does **not** apply yearly agriculture (15) or pasture (18).

Designed for multi-band products (e.g. classification_20260730): only
``--target-band`` (default 1 = burn) is zeroed; other bands are copied
unchanged where the mask is inactive, and left as-is for masked pixels on
non-target bands (month/surface stay unless you zero them separately).

Example::

  python filtering/filter_accumulated_lulc_only.py \\
    --input-dir ~/classification_20260730 \\
    --accumulated-dir ~/classification_20260619/filtering_work/mascaras/acumuladas \\
    --output-dir ~/classification_20260730_accA
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Must match create_accumulated_class_masks.CLASS_SPECS / create_total ACCUMULATED list
ACCUMULATED_MASK_NAMES = (
    "mascara_alfloramiento_rocoso_acumulado.tif",
    "mascara_arena_playa_duna_acumulado.tif",
    "mascara_salar_acumulado.tif",
    "mascara_hielo_nieve_acumulado.tif",
    "mascara_otra_area_sin_vegetacion_acumulado.tif",
    "mascara_rio_lago_acumulado.tif",
    "mascara_infraestructura_acumulado.tif",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter burn band with accumulated LULC masks only (group A)."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument(
        "--accumulated-dir",
        type=Path,
        required=True,
        help="Directory with mascara_*_acumulado.tif (incl. rio_lago + infraestructura).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob under input-dir (default: *.tif)",
    )
    p.add_argument(
        "--target-band",
        type=int,
        default=1,
        help="Burn band to zero under the mask (default: 1)",
    )
    p.add_argument("--fill-value", type=float, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists",
    )
    p.add_argument(
        "--name-suffix",
        default="",
        help="Optional suffix before .tif on outputs (default: keep same name)",
    )
    p.add_argument(
        "--prefer-remap",
        action="store_true",
        help=(
            "If both YYYY.tif and YYYY_remap.tif exist, process only *_remap.tif "
            "(for season products)."
        ),
    )
    return p.parse_args()


def list_inputs(input_dir: Path, pattern: str, *, prefer_remap: bool) -> list[Path]:
    paths = sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".tif")
    if not prefer_remap:
        return paths

    by_year: dict[str, list[Path]] = {}
    year_re = re.compile(r"(20\d{2})")
    for path in paths:
        m = year_re.search(path.stem)
        key = m.group(1) if m else path.stem
        by_year.setdefault(key, []).append(path)

    selected: list[Path] = []
    for key, group in sorted(by_year.items()):
        remap = [p for p in group if "_remap" in p.stem]
        if remap:
            selected.extend(remap)
        else:
            selected.extend(group)
    return selected


def build_mask_paths(accumulated_dir: Path) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for name in ACCUMULATED_MASK_NAMES:
        path = accumulated_dir / name
        if path.is_file():
            paths.append(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing accumulated mask(s). Rebuild with "
            "create_accumulated_class_masks.py (must include water 33 + infrastructure 24):\n  "
            + "\n  ".join(missing)
        )
    return paths


def _aligned_union_mask(
    mask_paths: list[Path],
    *,
    height: int,
    width: int,
    transform,
    crs,
) -> np.ndarray:
    union = np.zeros((height, width), dtype=bool)
    for path in mask_paths:
        with rasterio.open(path) as src:
            aligned = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
            union |= aligned >= 1
    return union


def filter_one(
    tif_path_str: str,
    mask_path_strs: list[str],
    output_path_str: str,
    target_band: int,
    fill_value: float,
) -> dict:
    tif_path = Path(tif_path_str)
    output_path = Path(output_path_str)
    mask_paths = [Path(p) for p in mask_path_strs]

    with rasterio.open(tif_path) as src:
        if target_band < 1 or target_band > src.count:
            raise ValueError(f"{tif_path.name}: --target-band {target_band} out of range 1..{src.count}")
        data = src.read()
        profile = src.profile.copy()
        union = _aligned_union_mask(
            mask_paths,
            height=src.height,
            width=src.width,
            transform=src.transform,
            crs=src.crs,
        )

    band_idx = target_band - 1
    before = data[band_idx]
    blocked_burn = (before != 0) & union
    removed = int(blocked_burn.sum())
    filtered_burn = before.copy()
    filtered_burn[union] = fill_value
    data[band_idx] = filtered_burn.astype(before.dtype, copy=False)

    profile.update(driver="GTiff", compress=profile.get("compress") or "lzw", tiled=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

    summary = {
        "input_file": str(tif_path),
        "output_file": str(output_path),
        "target_band": target_band,
        "mask_mode": "accumulated_only_group_A",
        "classes": "29,23,61,34,25,33,24",
        "masked_pixels": int(union.sum()),
        "pixels_filtered_to_zero": removed,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    acc_dir = args.accumulated_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not acc_dir.is_dir():
        raise FileNotFoundError(f"Accumulated dir not found: {acc_dir}")

    mask_paths = build_mask_paths(acc_dir)
    tifs = list_inputs(input_dir, args.pattern, prefer_remap=args.prefer_remap)
    if not tifs:
        raise RuntimeError(f"No TIFFs matching {args.pattern!r} in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, list[str], str, int, float]] = []
    for tif in tifs:
        stem = tif.stem + (args.name_suffix or "")
        out = output_dir / f"{stem}{tif.suffix}"
        if args.skip_existing and out.is_file():
            print(f"[INFO] Skip existing: {out.name}")
            continue
        tasks.append(
            (
                str(tif),
                [str(p) for p in mask_paths],
                str(out),
                args.target_band,
                float(args.fill_value),
            )
        )

    print(f"[INFO] Accumulated masks: {len(mask_paths)} files under {acc_dir}")
    print(f"[INFO] Inputs: {len(tifs)} | to process: {len(tasks)} | workers={args.workers}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Group A only (no agri/pasture) | run={datetime.now().isoformat(timespec='seconds')}")

    if not tasks:
        print("[INFO] Nothing to do.")
        return 0

    workers = max(1, min(args.workers, len(tasks)))
    if workers == 1:
        for task in tasks:
            summary = filter_one(*task)
            print(
                f"[INFO] {Path(summary['output_file']).name}: "
                f"removed_burn_px={summary['pixels_filtered_to_zero']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(filter_one, *t): t[0] for t in tasks}
            for fut in as_completed(futures):
                summary = fut.result()
                print(
                    f"[INFO] {Path(summary['output_file']).name}: "
                    f"removed_burn_px={summary['pixels_filtered_to_zero']}"
                )

    print("[INFO] Finished accumulated-only LULC filter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
