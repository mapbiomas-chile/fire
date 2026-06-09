"""Parse year, region and tile id from MapBiomas Fire raster filenames."""

from __future__ import annotations

import re
from pathlib import Path

YEAR_RE = re.compile(r"(20\d{2})")
REGION_RE = re.compile(r"_r(\d+)_")
_DEFAULT_YEAR_TOKEN_INDEX = 3


def stem_without_run_suffix(stem: str) -> str:
    """Drop LULC-filter timestamp suffix so metadata refers to the logical tile."""
    return stem.split("_filtered_")[0]


def parse_calendar_year(
    path: Path,
    *,
    year_token_index: int = _DEFAULT_YEAR_TOKEN_INDEX,
    from_year: int = 2013,
    to_year: int = 2025,
) -> int | None:
    stem = stem_without_run_suffix(path.stem)
    parts = stem.split("_")
    if 0 <= year_token_index < len(parts):
        raw = parts[year_token_index]
        if raw.isdigit() and len(raw) == 4:
            year = int(raw)
            if from_year <= year <= to_year:
                return year

    before_filtered = stem.split("_filtered_")[0]
    in_range = [
        int(match.group(1))
        for match in YEAR_RE.finditer(before_filtered)
        if from_year <= int(match.group(1)) <= to_year
    ]
    if len(in_range) == 1:
        return in_range[0]
    if len(in_range) > 1:
        return in_range[-1]

    match = YEAR_RE.search(stem)
    if match:
        year = int(match.group(1))
        if from_year <= year <= to_year:
            return year
    return None


def parse_region(path: Path) -> str | None:
    match = REGION_RE.search(path.stem)
    return match.group(1) if match else None


def tile_key(path: Path, year_token_index: int = _DEFAULT_YEAR_TOKEN_INDEX) -> str:
    """Stable id for all years of the same spatial tile."""
    stem = stem_without_run_suffix(path.stem)
    parts = stem.split("_")
    if 0 <= year_token_index < len(parts) and parts[year_token_index].isdigit():
        parts[year_token_index] = "{YEAR}"
        return "_".join(parts)
    return YEAR_RE.sub("{YEAR}", stem, count=1)
