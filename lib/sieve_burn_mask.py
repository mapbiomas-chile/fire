"""Remove small connected burn components from binary masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


def pixel_area_m2(transform) -> float:
    return abs(float(transform.a) * float(transform.e))


def min_pixels_for_area_ha(transform, area_ha: float) -> int:
    if area_ha <= 0:
        raise ValueError("area_ha must be > 0")
    return max(1, int(np.ceil(area_ha * 10000.0 / pixel_area_m2(transform))))


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
        resolved_min_pixels = (
            int(min_pixels)
            if min_pixels is not None
            else min_pixels_for_area_ha(src.transform, float(min_area_ha))
        )
        sieved, stats = sieve_connected_components(
            data,
            min_pixels=resolved_min_pixels,
            mask_value=mask_value,
            connectivity=connectivity,
        )
        stats["min_area_ha"] = float(min_area_ha) if min_area_ha is not None else None
        stats["pixel_area_m2"] = pixel_area_m2(src.transform)
        stats["input_file"] = str(raster_path)

        dest = Path(output_path) if output_path else raster_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(sieved, 1)

    stats["output_file"] = str(dest)
    return stats
