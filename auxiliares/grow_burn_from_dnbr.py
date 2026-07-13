#!/usr/bin/env python3
"""
Grow existing burn scars using dNBR similarity from the same-year mosaic.

For each connected burn component, iteratively dilates one pixel at a time and
adds frontier pixels whose dNBR is within ``mad_k * MAD`` of the seed median.
Growth is capped by ``max_radius`` pixels and ``max_growth_ratio`` relative to
the initial component area. Only years in ``--from-year``..``--to-year`` are
processed; other tiles are copied unchanged.

Designed for 2019–2025 where reference scars are unavailable and burns may be
under-segmented after manual false-positive cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxiliares.mosaic_dnbr import (  # noqa: E402
    mosaic_path_for_tile,
    read_dnbr_aligned_to_raster,
)
from lib.run_progress import RunProgress  # noqa: E402
from lib.tile_metadata import parse_calendar_year, parse_region  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grow burn pixels from existing scars using dNBR similarity."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with input burn rasters.")
    parser.add_argument("--output-dir", required=True, help="Directory for grown rasters.")
    parser.add_argument(
        "--mosaic-dir",
        required=True,
        help="Directory with same-year mosaic COGs (b14_chile_r<region>_<year>_cog.tif).",
    )
    parser.add_argument("--pattern", default="*.tif", help="Input glob (default: *.tif).")
    parser.add_argument("--from-year", type=int, default=2019, help="First year to grow (default: 2019).")
    parser.add_argument("--to-year", type=int, default=2025, help="Last year to grow (default: 2025).")
    parser.add_argument("--burn-value", type=float, default=1, help="Burn pixel value (default: 1).")
    parser.add_argument(
        "--output-suffix",
        default="_dnbr_grown",
        help="Suffix appended to output filename stem (default: _dnbr_grown).",
    )
    parser.add_argument(
        "--dnbr-band",
        type=int,
        default=None,
        help="1-based mosaic band index for dNBR (auto-detect if omitted).",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=8,
        help="Maximum dilation iterations per burn component (default: 8 px).",
    )
    parser.add_argument(
        "--mad-k",
        type=float,
        default=2.5,
        help="Tolerance as k * MAD around seed median dNBR (default: 2.5).",
    )
    parser.add_argument(
        "--min-dnbr",
        type=float,
        default=0.05,
        help="Minimum dNBR on frontier pixels (default: 0.05).",
    )
    parser.add_argument(
        "--min-seed-pixels",
        type=int,
        default=3,
        help="Skip components smaller than this (default: 3).",
    )
    parser.add_argument(
        "--max-growth-ratio",
        type=float,
        default=2.0,
        help="Max added pixels per component as ratio of initial area (default: 2.0). "
        "Use 0 for unlimited.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Connected-component connectivity (default: 8).",
    )
    parser.add_argument("--satellite", default="b14")
    parser.add_argument("--country", default="chile")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output TIF exists.")
    parser.add_argument("--dry-run", action="store_true", help="List actions without writing.")
    parser.add_argument("--stats-json", default=None, help="Write per-tile stats JSON to this path.")
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=float(os.environ.get("PROGRESS_HEARTBEAT_SEC", "30")),
        help="Progress heartbeat interval (default: PROGRESS_HEARTBEAT_SEC or 30).",
    )
    return parser.parse_args()


def grow_component(
    burn_mask: np.ndarray,
    dnbr: np.ndarray,
    component_mask: np.ndarray,
    *,
    max_radius: int,
    mad_k: float,
    min_dnbr: float,
    min_seed_pixels: int,
    max_growth_ratio: float,
) -> np.ndarray:
    seed_vals = dnbr[component_mask & np.isfinite(dnbr)]
    if seed_vals.size < min_seed_pixels:
        return component_mask

    median = float(np.median(seed_vals))
    mad = float(np.median(np.abs(seed_vals - median)))
    if mad < 1e-6:
        mad = float(np.std(seed_vals))
    if mad < 1e-6:
        mad = 0.1
    tolerance = mad_k * mad

    current = component_mask.copy()
    initial_area = int(current.sum())
    if initial_area == 0:
        return current

    max_add = (
        int(initial_area * max_growth_ratio)
        if max_growth_ratio and max_growth_ratio > 0
        else None
    )
    added_total = 0

    valid = np.isfinite(dnbr) & (dnbr >= min_dnbr)

    for _ in range(max_radius):
        dilated = ndimage.binary_dilation(current, iterations=1)
        frontier = dilated & ~current & valid
        if not frontier.any():
            break

        dnbr_frontier = dnbr[frontier]
        similar = np.abs(dnbr_frontier - median) <= tolerance
        add_mask = np.zeros_like(current, dtype=bool)
        add_mask[frontier] = similar
        n_add = int(add_mask.sum())
        if n_add == 0:
            break
        if max_add is not None and added_total + n_add > max_add:
            break

        current |= add_mask
        added_total += n_add

    return current


def grow_burn_mask(
    burn_mask: np.ndarray,
    dnbr: np.ndarray,
    *,
    max_radius: int,
    mad_k: float,
    min_dnbr: float,
    min_seed_pixels: int,
    max_growth_ratio: float,
    connectivity: int,
) -> tuple[np.ndarray, dict]:
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labeled, n_components = ndimage.label(burn_mask, structure=structure)
    if n_components == 0:
        return burn_mask.copy(), {
            "components": 0,
            "components_grown": 0,
            "pixels_added": 0,
        }

    grown = burn_mask.copy()
    components_grown = 0
    pixels_added = 0

    for component_id in range(1, n_components + 1):
        component_mask = labeled == component_id
        grown_component = grow_component(
            burn_mask,
            dnbr,
            component_mask,
            max_radius=max_radius,
            mad_k=mad_k,
            min_dnbr=min_dnbr,
            min_seed_pixels=min_seed_pixels,
            max_growth_ratio=max_growth_ratio,
        )
        added = int((grown_component & ~component_mask).sum())
        if added > 0:
            components_grown += 1
            pixels_added += added
        grown |= grown_component

    return grown, {
        "components": int(n_components),
        "components_grown": int(components_grown),
        "pixels_added": int(pixels_added),
    }


def grow_one_raster(
    tif_path: Path,
    output_dir: Path,
    *,
    mosaic_dir: Path,
    burn_value: float,
    output_suffix: str,
    dnbr_band: int | None,
    max_radius: int,
    mad_k: float,
    min_dnbr: float,
    min_seed_pixels: int,
    max_growth_ratio: float,
    connectivity: int,
    satellite: str,
    country: str,
    dry_run: bool,
) -> dict:
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    region = parse_region(tif_path)
    year = parse_calendar_year(tif_path)
    output_name = f"{tif_path.stem}{output_suffix}.tif"
    output_path = output_dir / output_name

    if region is None or year is None:
        raise ValueError(f"Could not parse region/year from {tif_path.name}")

    mosaic_path = mosaic_path_for_tile(
        mosaic_dir=mosaic_dir,
        region=region,
        year=year,
        satellite=satellite,
        country=country,
    )
    if not mosaic_path.is_file():
        raise FileNotFoundError(f"Mosaic not found for {tif_path.name}: {mosaic_path}")

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    burn_mask = data == burn_value
    pixels_before = int(burn_mask.sum())

    if dry_run:
        return {
            "input_raster": str(tif_path),
            "output_raster": str(output_path),
            "mosaic_raster": str(mosaic_path),
            "region": region,
            "year": year,
            "action": "grow",
            "dry_run": True,
            "burn_pixels_before": pixels_before,
        }

    dnbr, dnbr_band_name = read_dnbr_aligned_to_raster(
        mosaic_path,
        dnbr_band=dnbr_band,
        target_height=height,
        target_width=width,
        target_transform=transform,
        target_crs=crs,
    )

    grown_mask, grow_stats = grow_burn_mask(
        burn_mask,
        dnbr,
        max_radius=max_radius,
        mad_k=mad_k,
        min_dnbr=min_dnbr,
        min_seed_pixels=min_seed_pixels,
        max_growth_ratio=max_growth_ratio,
        connectivity=connectivity,
    )

    out_data = np.where(grown_mask, burn_value, data).astype(data.dtype)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile.update(
        count=1,
        dtype=out_data.dtype,
        compress="deflate",
        predictor=2,
        tiled=True,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_data, 1)

    pixels_after = int((out_data == burn_value).sum())
    return {
        "input_raster": str(tif_path),
        "output_raster": str(output_path),
        "mosaic_raster": str(mosaic_path),
        "dnbr_band_name": dnbr_band_name,
        "region": region,
        "year": year,
        "action": "grow",
        "burn_pixels_before": pixels_before,
        "burn_pixels_after": pixels_after,
        "pixels_added": int(pixels_after - pixels_before),
        **grow_stats,
    }


def passthrough_one_raster(
    tif_path: Path,
    output_dir: Path,
    *,
    output_suffix: str,
    dry_run: bool,
) -> dict:
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_path = output_dir / f"{tif_path.stem}{output_suffix}.tif"

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tif_path, output_path)

    return {
        "input_raster": str(tif_path),
        "output_raster": str(output_path),
        "region": parse_region(tif_path),
        "year": parse_calendar_year(tif_path),
        "action": "passthrough",
        "pixels_added": 0,
        "dry_run": dry_run,
    }


def _grow_task(task: tuple) -> dict:
    try:
        tif_path, output_dir, kwargs = task
        return grow_one_raster(tif_path, output_dir, **kwargs)
    except Exception as exc:
        tif_path = task[0]
        return {
            "input_raster": str(tif_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _passthrough_task(task: tuple) -> dict:
    tif_path, output_dir, kwargs = task
    return passthrough_one_raster(tif_path, output_dir, **kwargs)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mosaic_dir = Path(args.mosaic_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 1
    if not mosaic_dir.is_dir():
        print(f"ERROR: mosaic dir not found: {mosaic_dir}", file=sys.stderr)
        return 1

    tifs = sorted(input_dir.glob(args.pattern))
    if not tifs:
        print(f"ERROR: no files matching {args.pattern} in {input_dir}", file=sys.stderr)
        return 1

    grow_tasks: list[tuple] = []
    passthrough_tasks: list[tuple] = []
    skipped = 0

    for tif_path in tifs:
        year = parse_calendar_year(tif_path)
        output_path = output_dir / f"{tif_path.stem}{args.output_suffix}.tif"
        if args.skip_existing and output_path.is_file():
            skipped += 1
            continue

        if year is not None and args.from_year <= year <= args.to_year:
            grow_tasks.append(
                (
                    tif_path,
                    output_dir,
                    {
                        "mosaic_dir": mosaic_dir,
                        "burn_value": args.burn_value,
                        "output_suffix": args.output_suffix,
                        "dnbr_band": args.dnbr_band,
                        "max_radius": args.max_radius,
                        "mad_k": args.mad_k,
                        "min_dnbr": args.min_dnbr,
                        "min_seed_pixels": args.min_seed_pixels,
                        "max_growth_ratio": args.max_growth_ratio,
                        "connectivity": args.connectivity,
                        "satellite": args.satellite,
                        "country": args.country,
                        "dry_run": args.dry_run,
                    },
                )
            )
        else:
            passthrough_tasks.append(
                (tif_path, output_dir, {"output_suffix": args.output_suffix, "dry_run": args.dry_run})
            )

    total = len(grow_tasks) + len(passthrough_tasks)
    print(
        f"[INFO] {len(tifs)} input tiles: grow={len(grow_tasks)} "
        f"passthrough={len(passthrough_tasks)} skipped={skipped}",
        flush=True,
    )

    results: list[dict] = []
    progress = RunProgress(
        total=total,
        label="dNBR grow",
        heartbeat_sec=args.heartbeat_sec,
    )
    progress.start()

    all_tasks = [("grow", t) for t in grow_tasks] + [("passthrough", t) for t in passthrough_tasks]

    if args.workers <= 1 or args.dry_run:
        for kind, task in all_tasks:
            if kind == "grow":
                result = grow_one_raster(task[0], task[1], **task[2])
            else:
                result = passthrough_one_raster(task[0], task[1], **task[2])
            results.append(result)
            progress.step(Path(result.get("input_raster", "")).name)
            if kind == "grow" and result.get("action") == "grow" and not result.get("dry_run"):
                print(
                    f"[INFO] {Path(result['input_raster']).name}: "
                    f"+{result.get('pixels_added', 0)} px "
                    f"({result.get('burn_pixels_before')} → {result.get('burn_pixels_after')})",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for kind, task in all_tasks:
                if kind == "grow":
                    fut = pool.submit(_grow_task, task)
                else:
                    fut = pool.submit(_passthrough_task, task)
                futures[fut] = task[0].name

            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                progress.step(Path(result.get("input_raster", futures[fut])).name)
                if result.get("action") == "grow" and "error" not in result:
                    print(
                        f"[INFO] {Path(result['input_raster']).name}: "
                        f"+{result.get('pixels_added', 0)} px "
                        f"({result.get('burn_pixels_before')} → {result.get('burn_pixels_after')})",
                        flush=True,
                    )

    progress.finish()

    errors = [r for r in results if "error" in r]
    if errors:
        for err in errors:
            print(f"[ERROR] {err['input_raster']}: {err['error']}", file=sys.stderr)
            if "traceback" in err:
                print(err["traceback"], file=sys.stderr)

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "skipped_existing": skipped,
            "results": results,
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] stats written to {stats_path}", flush=True)

    added = sum(int(r.get("pixels_added", 0) or 0) for r in results if r.get("action") == "grow")
    print(f"[INFO] total pixels added by growth: {added:,}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
