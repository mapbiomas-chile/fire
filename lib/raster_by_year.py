"""Merge regional burn-mask tiles into one raster per calendar year."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.merge import merge

from lib.tile_metadata import parse_calendar_year


def group_raster_paths_by_year(
    paths: list[Path],
    *,
    from_year: int = 2013,
    to_year: int = 2025,
) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        year = parse_calendar_year(path, from_year=from_year, to_year=to_year)
        if year is None:
            continue
        groups[year].append(path)
    return dict(groups)


def merge_year_rasters(
    year: int,
    paths: list[Path],
    output_path: Path,
    *,
    method: str = "max",
    nodata_override: float | None = None,
) -> dict:
    """Write one mosaic GeoTIFF for ``year`` from ``paths``."""
    paths = sorted(paths)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    srcs = [rasterio.open(p) for p in paths]
    try:
        first = srcs[0]
        for i, src in enumerate(srcs[1:], start=1):
            if src.crs != first.crs:
                raise ValueError(
                    f"CRS mismatch for year {year}: {paths[0].name} vs {paths[i].name}"
                )
            if src.count != first.count:
                raise ValueError(
                    f"Band count mismatch for year {year}: {paths[0].name} vs {paths[i].name}"
                )

        nodata = nodata_override
        if nodata is None and first.nodata is not None:
            nodata = first.nodata

        mosaic, out_transform = merge(srcs, nodata=nodata, method=method)

        profile = first.profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            crs=first.crs,
            nodata=nodata,
            compress=profile.get("compress", "deflate"),
            predictor=profile.get("predictor", 2),
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in srcs:
            src.close()

    return {
        "year": year,
        "output_file": str(output_path),
        "source_tiles": len(paths),
        "source_files": [p.name for p in paths],
    }


def merge_directory_by_year(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*.tif",
    name_contains: str | None = None,
    output_stem: str = "chile",
    method: str = "max",
    nodata_override: float | None = None,
    from_year: int = 2013,
    to_year: int = 2025,
    workers: int = 1,
) -> list[dict]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    paths: list[Path] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if name_contains and name_contains not in path.name:
            continue
        paths.append(path)

    if not paths:
        raise RuntimeError(f"No rasters found in {input_dir} with pattern {pattern!r}")

    groups = group_raster_paths_by_year(paths, from_year=from_year, to_year=to_year)
    if not groups:
        raise RuntimeError(f"No rasters with parseable years in {from_year}-{to_year}")

    years_sorted = sorted(groups.keys())
    worker_count = max(1, min(workers, len(years_sorted)))
    results: list[dict] = []

    if worker_count == 1:
        for year in years_sorted:
            out_path = output_dir / f"{output_stem}_{year}.tif"
            results.append(
                merge_year_rasters(
                    year,
                    groups[year],
                    out_path,
                    method=method,
                    nodata_override=nodata_override,
                )
            )
        return results

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                merge_year_rasters,
                year,
                groups[year],
                output_dir / f"{output_stem}_{year}.tif",
                method=method,
                nodata_override=nodata_override,
            ): year
            for year in years_sorted
        }
        for future in as_completed(futures):
            results.append(future.result())

    return sorted(results, key=lambda row: row["year"])
