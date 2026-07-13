"""Helpers to locate dNBR/NBR bands in MapBiomas mosaics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from classification.fire_model_common import SPATIAL_BAND_PATTERNS  # noqa: E402

DNBR_PATTERNS = SPATIAL_BAND_PATTERNS


def band_descriptions(src: rasterio.DatasetReader) -> list[str]:
    return [desc if desc is not None else f"band_{i}" for i, desc in enumerate(src.descriptions)]


def find_dnbr_band_index(
    descriptions: list[str],
    *,
    explicit_band: int | None = None,
) -> int:
    """Return 1-based band index for dNBR/rNBR/NBR."""
    if explicit_band is not None:
        if explicit_band < 1 or explicit_band > len(descriptions):
            raise ValueError(f"Band {explicit_band} out of range (1..{len(descriptions)})")
        return explicit_band

    for pattern in DNBR_PATTERNS:
        for i, name in enumerate(descriptions, start=1):
            if pattern in name.lower():
                return i
    raise ValueError(
        f"No dNBR/rNBR/NBR band found in descriptions: {descriptions}. "
        "Pass --dnbr-band explicitly."
    )


def mosaic_path_for_tile(
    *,
    mosaic_dir: Path,
    region: str,
    year: int,
    satellite: str = "b14",
    country: str = "chile",
) -> Path:
    return mosaic_dir / f"{satellite}_{country}_r{region}_{year}_cog.tif"


def read_dnbr_aligned_to_raster(
    mosaic_path: Path,
    *,
    dnbr_band: int | None,
    target_height: int,
    target_width: int,
    target_transform,
    target_crs,
) -> tuple:
    """Read and nearest-neighbor align one dNBR band to a reference grid."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(mosaic_path) as src:
        descriptions = band_descriptions(src)
        band_idx = find_dnbr_band_index(descriptions, explicit_band=dnbr_band)
        dnbr = src.read(band_idx).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dnbr = np.where(dnbr == nodata, np.nan, dnbr)

        if (
            src.height == target_height
            and src.width == target_width
            and src.transform == target_transform
            and src.crs == target_crs
        ):
            return dnbr, descriptions[band_idx - 1]

        aligned = np.full((target_height, target_width), np.nan, dtype=np.float32)
        reproject(
            source=dnbr,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return aligned, descriptions[band_idx - 1]
