"""GeoTIFF helpers for mask rasters (avoid inheriting VRT driver from sources)."""

from __future__ import annotations

from pathlib import Path

import rasterio


def mask_gtiff_profile(src_profile: dict) -> dict:
    """Build a write profile for single-band uint8 mask GeoTIFFs."""
    spatial = {
        k: src_profile[k]
        for k in ("height", "width", "transform", "crs")
        if k in src_profile
    }
    spatial.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    return spatial


def open_mask_writer(path: Path, src_profile: dict) -> rasterio.DatasetWriter:
    """Create (or replace) a mask GeoTIFF; never reuse an existing VRT sidecar."""
    path = Path(path)
    path.unlink(missing_ok=True)
    for sidecar in (path.with_suffix(".tfw"), path.with_suffix(".aux.xml")):
        sidecar.unlink(missing_ok=True)
    profile = mask_gtiff_profile(src_profile)
    dst = rasterio.open(path, "w", **profile)
    if dst.driver != "GTiff":
        dst.close()
        raise RuntimeError(f"Expected GTiff output at {path}, got driver={dst.driver!r}")
    return dst
