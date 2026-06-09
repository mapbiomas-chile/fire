"""LULC stability windows for yearly non-burnable masks (A2)."""

from __future__ import annotations


def stability_window_years(
    filter_year: int,
    *,
    lulc_min_year: int,
    lulc_max_year: int,
    window_size: int = 4,
) -> list[int]:
    """
    Calendar years used to test class stability for filtering burn year ``filter_year``.

    Default (forward): ``[Y, Y+1, Y+2, Y+3]`` when all years exist in the LULC stack.

    Near the end of the stack: ``[Y-3, Y-2, Y-1, Y]`` (e.g. filter 2025 → 2022–2025).

    The resulting mask applies only to classified/filtered rasters of ``filter_year``,
    not to the other years in the window.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if filter_year < lulc_min_year or filter_year > lulc_max_year:
        raise ValueError(
            f"filter_year {filter_year} outside LULC stack range "
            f"{lulc_min_year}-{lulc_max_year}"
        )

    forward_end = filter_year + window_size - 1
    if forward_end <= lulc_max_year:
        years = list(range(filter_year, filter_year + window_size))
    else:
        start = filter_year - window_size + 1
        if start < lulc_min_year:
            raise ValueError(
                f"Cannot build {window_size}-year stability window for filter year "
                f"{filter_year}: need LULC from {start} but stack starts at "
                f"{lulc_min_year}"
            )
        years = list(range(start, filter_year + 1))

    if any(y < lulc_min_year or y > lulc_max_year for y in years):
        raise ValueError(
            f"Stability window {years} for filter year {filter_year} exceeds LULC stack "
            f"{lulc_min_year}-{lulc_max_year}"
        )
    return years


def year_to_band(year: int, start_year_in_band_1: int) -> int:
    return year - start_year_in_band_1 + 1
