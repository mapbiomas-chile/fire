"""Vectorize binary burn masks from classified / post-filter rasters."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from lib.tile_metadata import parse_calendar_year, parse_region
from lib.sieve_burn_mask import sieve_connected_components


def polygonize_burn_mask(
    data: np.ndarray,
    transform,
    crs,
    *,
    mask_value: float = 1,
    connectivity: int = 8,
) -> gpd.GeoDataFrame:
    """Convert burned pixels (``mask_value``) to a GeoDataFrame of polygons."""
    mask = data == mask_value
    if not np.any(mask):
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    raster_for_shapes = np.where(mask, 1, 0).astype(np.uint8)
    geoms = []
    for geom, val in shapes(
        raster_for_shapes,
        mask=mask,
        transform=transform,
        connectivity=connectivity,
    ):
        if int(val) == 1:
            geoms.append(shape(geom))

    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def polygonize_raster_file(
    tif_path: Path,
    output_path: Path,
    *,
    band: int = 1,
    mask_value: float = 1,
    connectivity: int = 8,
    sieve_min_pixels: int | None = None,
    sieve_connectivity: int = 8,
    year: int | None = None,
    region: str | None = None,
    source_file: str | None = None,
) -> dict:
    """
    Polygonize one burn-mask raster and write a GeoPackage.

    Returns a summary dict with polygon count and output path.
    """
    tif_path = Path(tif_path)
    output_path = Path(output_path)

    sieve_stats = None
    with rasterio.open(tif_path) as src:
        data = src.read(band)
        crs = src.crs
        if sieve_min_pixels is not None and sieve_min_pixels >= 1:
            data, sieve_stats = sieve_connected_components(
                data,
                min_pixels=sieve_min_pixels,
                mask_value=mask_value,
                connectivity=sieve_connectivity,
            )
        gdf = polygonize_burn_mask(
            data,
            src.transform,
            crs,
            mask_value=mask_value,
            connectivity=connectivity,
        )

    resolved_year = year if year is not None else parse_calendar_year(tif_path)
    resolved_region = region if region is not None else parse_region(tif_path)
    resolved_source = source_file or tif_path.name

    if gdf.empty:
        gdf = gpd.GeoDataFrame(
            {
                "source_file": [],
                "mask_value": [],
                "year": [],
                "region": [],
            },
            geometry=[],
            crs=crs,
        )
    else:
        gdf = gdf.assign(
            source_file=resolved_source,
            mask_value=int(mask_value),
            year=resolved_year,
            region=resolved_region,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")

    return {
        "input_file": str(tif_path),
        "output_file": str(output_path),
        "polygon_count": len(gdf),
        "year": resolved_year,
        "region": resolved_region,
        "sieve": sieve_stats,
    }


def _polygonize_task(args: tuple) -> dict:
    (
        tif_path,
        output_path,
        band,
        mask_value,
        connectivity,
        sieve_min_pixels,
        sieve_connectivity,
    ) = args
    return polygonize_raster_file(
        tif_path,
        output_path,
        band=band,
        mask_value=mask_value,
        connectivity=connectivity,
        sieve_min_pixels=sieve_min_pixels,
        sieve_connectivity=sieve_connectivity,
    )


def collect_raster_paths(
    input_dir: Path,
    *,
    pattern: str = "*.tif",
    name_contains: str | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if name_contains and name_contains not in path.name:
            continue
        paths.append(path)
    return paths


def polygonize_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*.tif",
    name_contains: str | None = None,
    band: int = 1,
    mask_value: float = 1,
    connectivity: int = 8,
    sieve_min_pixels: int | None = None,
    sieve_connectivity: int = 8,
    workers: int | None = None,
    output_suffix: str = "_mask1",
) -> list[dict]:
    """Polygonize every raster in ``input_dir``; one GeoPackage per input file."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    tif_files = collect_raster_paths(input_dir, pattern=pattern, name_contains=name_contains)
    if not tif_files:
        msg = f"No rasters found in {input_dir} with pattern {pattern!r}"
        if name_contains:
            msg += f" (name contains {name_contains!r})"
        raise RuntimeError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    worker_count = max(1, min(worker_count, len(tif_files)))

    tasks = [
        (
            tif_path,
            output_dir / f"{tif_path.stem}{output_suffix}.gpkg",
            band,
            mask_value,
            connectivity,
            sieve_min_pixels,
            sieve_connectivity,
        )
        for tif_path in tif_files
    ]

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_polygonize_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: row["output_file"])


def merge_polygon_outputs(
    summaries: list[dict],
    merged_gpkg: Path,
) -> dict:
    """Concatenate per-tile GeoPackages into one layer."""
    merged_gpkg = Path(merged_gpkg)
    frames: list[gpd.GeoDataFrame] = []
    for row in summaries:
        gdf = gpd.read_file(row["output_file"])
        if not gdf.empty:
            frames.append(gdf)

    if frames:
        merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    else:
        merged = gpd.GeoDataFrame(
            columns=["source_file", "mask_value", "year", "region", "geometry"],
            geometry="geometry",
        )

    merged_gpkg.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(merged_gpkg, driver="GPKG")
    return {
        "merged_file": str(merged_gpkg),
        "polygon_count": len(merged),
        "source_tiles": len(summaries),
    }
