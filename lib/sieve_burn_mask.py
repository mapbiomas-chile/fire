"""Remove small connected burn components from binary masks."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from scipy import ndimage

_GEOD = Geod(ellps="WGS84")


def pixel_area_m2_from_dataset(src: rasterio.io.DatasetReader) -> float:
    """Return approximate ground area (m²) of one raster pixel."""
    transform = src.transform
    a = abs(float(transform.a))
    e = abs(float(transform.e))
    crs = src.crs

    looks_geographic = crs is not None and crs.is_geographic
    if crs is None and max(a, e) < 1.0:
        looks_geographic = True

    if not looks_geographic:
        return a * e

    row = min(max(src.height // 2, 0), src.height - 1)
    col = min(max(src.width // 2, 0), src.width - 1)
    x0, y0 = transform * (col + 0.5, row + 0.5)
    x1, y1 = transform * (col + 1.5, row + 0.5)
    x2, y2 = transform * (col + 0.5, row + 1.5)
    _, _, width_m = _GEOD.inv(x0, y0, x1, y1)
    _, _, height_m = _GEOD.inv(x0, y0, x2, y2)
    return abs(width_m * height_m)


def min_pixels_for_area_ha(src: rasterio.io.DatasetReader, area_ha: float) -> int:
    if area_ha <= 0:
        raise ValueError("area_ha must be > 0")
    pixel_area = pixel_area_m2_from_dataset(src)
    if pixel_area <= 0:
        raise ValueError(f"Invalid pixel area: {pixel_area}")
    return max(1, int(np.ceil(area_ha * 10000.0 / pixel_area)))


def sieve_connected_components(
    data: np.ndarray,
    *,
    min_pixels: int,
    mask_value: float = 1,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict]:
    """Drop burned components with fewer than ``min_pixels`` pixels."""
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1")

    burned = data == mask_value
    if mask_value == 1 and not np.any(burned) and np.any(data > 0):
        burned = data > 0

    burned_before = int(burned.sum())
    if not np.any(burned):
        return data, {
            "components_before": 0,
            "components_after": 0,
            "pixels_removed": 0,
            "burned_pixels_before": 0,
            "burned_pixels_after": 0,
        }

    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labeled, num_features = ndimage.label(burned, structure=structure)
    counts = np.bincount(labeled.ravel())
    if len(counts) <= 1:
        return data, {
            "components_before": 0,
            "components_after": 0,
            "pixels_removed": 0,
            "burned_pixels_before": burned_before,
            "burned_pixels_after": burned_before,
        }

    keep_labels = np.flatnonzero(counts >= min_pixels)
    keep_labels = keep_labels[keep_labels != 0]
    keep_mask = np.isin(labeled, keep_labels)

    out = np.where(keep_mask, mask_value, 0)
    if data.dtype != np.float64 and data.dtype != np.float32:
        out = out.astype(data.dtype)
    else:
        out = out.astype(data.dtype)

    burned_after = int((out == mask_value).sum())
    return out, {
        "components_before": int(num_features),
        "components_after": int(len(keep_labels)),
        "pixels_removed": burned_before - burned_after,
        "burned_pixels_before": burned_before,
        "burned_pixels_after": burned_after,
        "min_pixels": int(min_pixels),
    }


def sieve_raster_file(
    raster_path: Path,
    *,
    min_pixels: int | None = None,
    min_area_ha: float | None = None,
    mask_value: float = 1,
    connectivity: int = 8,
    output_path: Path | None = None,
) -> dict:
    """
    Sieve a single-band burn mask raster.

    Provide ``min_pixels`` or ``min_area_ha`` (resolved per raster geotransform).
    Writes to ``output_path`` or overwrites ``raster_path`` when omitted.
    """
    raster_path = Path(raster_path)
    if min_pixels is None and min_area_ha is None:
        raise ValueError("Provide min_pixels or min_area_ha")

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        pixel_area = pixel_area_m2_from_dataset(src)
        resolved_min_pixels = (
            int(min_pixels)
            if min_pixels is not None
            else min_pixels_for_area_ha(src, float(min_area_ha))
        )
        if resolved_min_pixels > 100_000:
            warnings.warn(
                f"Sieve min_pixels={resolved_min_pixels} looks too high "
                f"(pixel_area_m2={pixel_area:.6f}, crs={src.crs}). "
                "Check raster CRS / geotransform.",
                stacklevel=2,
            )

        sieved, stats = sieve_connected_components(
            data,
            min_pixels=resolved_min_pixels,
            mask_value=mask_value,
            connectivity=connectivity,
        )
        stats["min_area_ha"] = float(min_area_ha) if min_area_ha is not None else None
        stats["pixel_area_m2"] = pixel_area
        stats["crs"] = str(src.crs) if src.crs else None
        stats["input_file"] = str(raster_path)

        if stats["burned_pixels_before"] > 0 and stats["burned_pixels_after"] == 0:
            warnings.warn(
                f"Sieve removed all {stats['burned_pixels_before']} burned pixels from "
                f"{raster_path.name} (min_pixels={resolved_min_pixels}).",
                stacklevel=2,
            )

        dest = Path(output_path) if output_path else raster_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(sieved, 1)

    stats["output_file"] = str(dest)
    return stats
