#!/usr/bin/env python3
"""
National vectorization: merge tiles by year → polygonize → group nearby scars.

Typical flow after filtering::

    python lib/vectorize_national_by_year.py \\
      --input-dir /path/to/classified_filtered \\
      --work-root /path/to/national_vector_work
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.group_fire_events import (  # noqa: E402
    filter_fragments_by_min_area,
    filter_fragments_by_min_pixels,
    group_polygons_by_distance,
    summarize_grouping,
)
from lib.raster_by_year import merge_directory_by_year  # noqa: E402
from lib.sieve_burn_mask import pixel_area_m2_from_dataset, sieve_raster_file  # noqa: E402
from lib.vectorize import polygonize_raster_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge filtered burn rasters by year, polygonize Chile-wide mosaics, "
            "and group nearby scars into multipolygon fire events."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--name-contains", default=None)
    parser.add_argument("--from-year", type=int, default=2013)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument(
        "--merge-method",
        choices=("first", "last", "min", "max"),
        default="max",
    )
    parser.add_argument("--mosaic-stem", default="chile")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-group", action="store_true")
    parser.add_argument("--keep-raw-polygons", action="store_true")
    parser.add_argument("--group-distance-m", type=float, default=200.0)
    parser.add_argument("--metric-crs", default="EPSG:32719")
    parser.add_argument("--mask-value", type=float, default=1)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--merge-workers", type=int, default=1)
    parser.add_argument(
        "--sieve-min-pixels",
        type=int,
        default=None,
        help="Remove connected burn components smaller than N pixels (overrides --sieve-min-ha).",
    )
    parser.add_argument(
        "--sieve-min-ha",
        type=float,
        default=None,
        help="Remove components smaller than this area in ha (from raster pixel size).",
    )
    parser.add_argument(
        "--skip-sieve",
        action="store_true",
        help="Do not remove small isolated burn patches before polygonize.",
    )
    parser.add_argument(
        "--sieve-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Connectivity for the pre-vectorize sieve (default: 8).",
    )
    parser.add_argument(
        "--fragment-min-pixels",
        type=int,
        default=None,
        help=(
            "Drop polygon fragments below this pixel count before 200 m grouping. "
            "Defaults to --sieve-min-pixels when sieve is enabled."
        ),
    )
    parser.add_argument(
        "--fragment-min-ha",
        type=float,
        default=None,
        help="Legacy: drop fragments below this area in ha (use --fragment-min-pixels instead).",
    )
    parser.add_argument(
        "--skip-fragment-filter",
        action="store_true",
        help="Do not drop small polygons before proximity grouping.",
    )
    parser.add_argument("--stats-json", default=None)
    return parser.parse_args()


def _discover_existing_mosaics(
    mosaics_dir: Path,
    mosaic_stem: str,
    from_year: int,
    to_year: int,
) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(mosaics_dir.glob(f"{mosaic_stem}_*.tif")):
        token = path.stem.rsplit("_", 1)[-1]
        if not token.isdigit():
            continue
        year = int(token)
        if from_year <= year <= to_year:
            rows.append({"year": year, "output_file": str(path), "source_tiles": None})
    if not rows:
        raise RuntimeError(f"No mosaics found in {mosaics_dir}")
    return rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    work_root = Path(args.work_root)
    mosaics_dir = work_root / "mosaics_by_year"
    events_dir = work_root / "polygons_chile"
    raw_dir = work_root / "polygons_raw"
    scratch_dir = work_root / "_scratch"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for directory in (work_root, events_dir, scratch_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.keep_raw_polygons or args.skip_group:
        raw_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_merge:
        merge_summaries = _discover_existing_mosaics(
            mosaics_dir, args.mosaic_stem, args.from_year, args.to_year
        )
    else:
        merge_summaries = merge_directory_by_year(
            input_dir,
            mosaics_dir,
            pattern=args.pattern,
            name_contains=args.name_contains,
            output_stem=args.mosaic_stem,
            method=args.merge_method,
            from_year=args.from_year,
            to_year=args.to_year,
            workers=args.merge_workers,
        )
        for row in merge_summaries:
            print(
                f"[INFO] Merged year {row['year']}: {row['source_tiles']} tiles → {row['output_file']}",
                flush=True,
            )

    year_summaries: list[dict] = []
    sieve_enabled = not args.skip_sieve and (
        args.sieve_min_pixels is not None or args.sieve_min_ha is not None
    )
    if sieve_enabled:
        mosaics_sieved_dir = work_root / "mosaics_by_year_sieved"
        mosaics_sieved_dir.mkdir(parents=True, exist_ok=True)
    else:
        mosaics_sieved_dir = None

    for merge_row in sorted(merge_summaries, key=lambda r: r["year"]):
        year = merge_row["year"]
        mosaic_path = Path(merge_row["output_file"])

        polygonize_raster = mosaic_path
        sieve_stats = None
        if sieve_enabled:
            assert mosaics_sieved_dir is not None
            sieved_path = mosaics_sieved_dir / mosaic_path.name
            sieve_stats = sieve_raster_file(
                mosaic_path,
                min_pixels=args.sieve_min_pixels,
                min_area_ha=args.sieve_min_ha,
                mask_value=args.mask_value,
                connectivity=args.sieve_connectivity,
                output_path=sieved_path,
            )
            polygonize_raster = sieved_path
            print(
                f"[INFO] Year {year}: sieve "
                f"{sieve_stats['components_before']} → {sieve_stats['components_after']} components "
                f"(min_pixels={sieve_stats['min_pixels']}, "
                f"pixel_area_m2={sieve_stats.get('pixel_area_m2', '?'):.2f}, "
                f"burned {sieve_stats['burned_pixels_before']} → {sieve_stats['burned_pixels_after']} px, "
                f"removed {sieve_stats['pixels_removed']} px) → {sieved_path.name}",
                flush=True,
            )
            if (
                sieve_stats["burned_pixels_before"] > 0
                and sieve_stats["burned_pixels_after"] == 0
            ):
                print(
                    f"[WARN] Year {year}: sieve removed ALL burned pixels — "
                    f"check CRS/min_pixels (crs={sieve_stats.get('crs')}).",
                    flush=True,
                )

        raw_gpkg = raw_dir / f"{args.mosaic_stem}_{year}_raw.gpkg"
        scratch_gpkg = scratch_dir / f"{args.mosaic_stem}_{year}_raw.gpkg"
        polygonize_target = raw_gpkg if (args.keep_raw_polygons or args.skip_group) else scratch_gpkg

        raw_summary = polygonize_raster_file(
            polygonize_raster,
            polygonize_target,
            mask_value=args.mask_value,
            connectivity=args.connectivity,
            year=year,
            region=None,
            source_file=polygonize_raster.name,
        )
        print(
            f"[INFO] Year {year}: polygonized {raw_summary['polygon_count']} fragments",
            flush=True,
        )
        if raw_summary["polygon_count"] == 0:
            print(
                f"[WARN] Year {year}: no polygons — inspect {polygonize_raster}",
                flush=True,
            )

        if args.skip_group:
            year_summaries.append(
                {
                    "year": year,
                    "mosaic": str(mosaic_path),
                    "mosaic_sieved": str(polygonize_raster) if sieve_stats else None,
                    "sieve": sieve_stats,
                    "raw_gpkg": str(polygonize_target),
                    "events_gpkg": None,
                    **raw_summary,
                }
            )
            continue

        gdf_raw = gpd.read_file(polygonize_target)

        fragment_filter_stats = None
        gdf_for_grouping = gdf_raw
        fragment_min_pixels = args.fragment_min_pixels
        if fragment_min_pixels is None and not args.skip_fragment_filter:
            if args.sieve_min_pixels is not None:
                fragment_min_pixels = args.sieve_min_pixels
        fragment_min_ha = args.fragment_min_ha
        if fragment_min_ha is None and sieve_enabled and args.sieve_min_ha is not None:
            fragment_min_ha = args.sieve_min_ha

        if not args.skip_fragment_filter and fragment_min_pixels is not None:
            if sieve_stats and sieve_stats.get("pixel_area_m2"):
                pixel_area_m2 = float(sieve_stats["pixel_area_m2"])
            else:
                with rasterio.open(polygonize_raster) as src:
                    pixel_area_m2 = pixel_area_m2_from_dataset(src)
            gdf_for_grouping, fragment_filter_stats = filter_fragments_by_min_pixels(
                gdf_raw,
                min_pixels=fragment_min_pixels,
                pixel_area_m2=pixel_area_m2,
                metric_crs=args.metric_crs,
            )
            print(
                f"[INFO] Year {year}: fragment filter >= {fragment_min_pixels} px "
                f"(pixel_area_m2={pixel_area_m2:.2f}): "
                f"{fragment_filter_stats['fragments_before']} -> "
                f"{fragment_filter_stats['fragments_kept']} "
                f"(removed {fragment_filter_stats['fragments_removed']})",
                flush=True,
            )
        elif not args.skip_fragment_filter and fragment_min_ha is not None:
            gdf_for_grouping, fragment_filter_stats = filter_fragments_by_min_area(
                gdf_raw,
                min_area_ha=fragment_min_ha,
                metric_crs=args.metric_crs,
            )
            print(
                f"[INFO] Year {year}: fragment filter >= {fragment_min_ha} ha: "
                f"{fragment_filter_stats['fragments_before']} -> "
                f"{fragment_filter_stats['fragments_kept']} "
                f"(removed {fragment_filter_stats['fragments_removed']})",
                flush=True,
            )

        if fragment_filter_stats and (
            fragment_filter_stats["fragments_before"] > 0
            and fragment_filter_stats["fragments_kept"] == 0
        ):
            print(
                f"[WARN] Year {year}: fragment filter removed ALL polygons.",
                flush=True,
            )

        grouped = group_polygons_by_distance(
            gdf_for_grouping,
            max_gap_m=args.group_distance_m,
            metric_crs=args.metric_crs,
            event_id_prefix=f"{args.mosaic_stem}_{year}",
        )
        events_gpkg = events_dir / f"{args.mosaic_stem}_{year}_events.gpkg"
        grouped.to_file(events_gpkg, driver="GPKG")

        if not args.keep_raw_polygons and scratch_gpkg.exists():
            scratch_gpkg.unlink()

        grouping_stats = summarize_grouping(len(gdf_for_grouping), grouped)
        print(
            f"[INFO] Year {year}: {grouping_stats['raw_polygon_count']} fragments → "
            f"{grouping_stats['event_count']} events "
            f"(gap ≤ {args.group_distance_m} m) → {events_gpkg}",
            flush=True,
        )
        year_summaries.append(
            {
                "year": year,
                "mosaic": str(mosaic_path),
                "mosaic_sieved": str(polygonize_raster) if sieve_stats else None,
                "sieve": sieve_stats,
                "fragment_filter": fragment_filter_stats,
                "raw_gpkg": str(raw_gpkg) if raw_gpkg.exists() else None,
                "events_gpkg": str(events_gpkg),
                "group_distance_m": args.group_distance_m,
                "fragment_min_pixels": fragment_min_pixels,
                "fragment_min_ha": fragment_min_ha,
                **raw_summary,
                **grouping_stats,
            }
        )

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(input_dir),
            "work_root": str(work_root),
            "merge_method": args.merge_method,
            "group_distance_m": args.group_distance_m,
            "sieve_min_pixels": args.sieve_min_pixels,
            "sieve_min_ha": args.sieve_min_ha,
            "sieve_enabled": sieve_enabled,
            "fragment_min_pixels": args.fragment_min_pixels,
            "fragment_min_ha": args.fragment_min_ha,
            "years": year_summaries,
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote stats: {stats_path}", flush=True)

    print(f"[INFO] Finished. Years processed: {len(year_summaries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
