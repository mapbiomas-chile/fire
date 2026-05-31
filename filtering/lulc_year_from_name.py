"""Parse calendar year from LULC mosaic filenames (shared by mask scripts)."""

from __future__ import annotations

import re
from pathlib import Path

YEAR_RE = re.compile(
    r"(?:collection02[_-](\d{4}))|(?:^lulc[_-](\d{4})(?:[._-]|$))|(?:_(\d{4})\d{8}-)"
)


def year_from_lulc_path(path: Path) -> int | None:
    m = YEAR_RE.search(path.stem)
    if not m:
        return None
    return int(next(g for g in m.groups() if g is not None))
