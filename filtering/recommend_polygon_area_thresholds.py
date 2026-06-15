#!/usr/bin/env python3
"""
Recommend minimum polygon area thresholds (ha) from per-tile vector outputs.

Reads GeoPackages (typically one per classified tile), groups polygons by region
and calendar year parsed from the filename (r1, r2, r4, r6 + 20xx), and writes:

- ``threshold_summary.json`` — stats + recommended thresholds (global, per region,
  and per region×year)
- ``thresholds_by_region.csv`` — region-level table (full series pooled)
- ``thresholds_by_region_year.csv`` — region×year table for manual review

Use the JSON with ``filter_polygons_by_threshold.py --stats-summary-json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.tile_metadata import parse_calendar_year, parse_region  # noqa: E402

ALLOWED_REGIONS = ("1", "2", "4", "6")

RULE_KEYS = {
    "p5": "rule_p5_threshold_ha",
    "p10": "rule_p10_threshold_ha",
    "p25": "rule_p25_threshold_ha",
    "elbow": "rule_elbow_threshold_ha",
    "bottom5_mean": "rule_bottom5_mean_threshold_ha",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend minimum area thresholds (ha) from polygon GPKGs."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with polygon GPKG files.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for threshold_summary.json and thresholds_by_region.csv.",
    )
    parser.add_argument("--pattern", default="*.gpkg", help="Input glob (default: *.gpkg).")
    parser.add_argument(
        "--target-crs",
        default="EPSG:32719",
        help="CRS for area in m²/ha (default: EPSG:32719).",
    )
    parser.add_argument(
        "--min-polygons",
        type=int,
        default=50,
        help="Skip threshold rules when a group has fewer polygons (default: 50).",
    )
    return parser.parse_args()


def load_areas_ha(gpkg_path: Path, target_crs: str) -> np.ndarray:
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return np.array([], dtype=float)
    projected = gdf.to_crs(target_crs)
    return (projected.geometry.area.to_numpy(dtype=float) / 10000.0).astype(float)


def _elbow_threshold_ha(areas_ha: np.ndarray) -> float | None:
    """Knee on the log-spaced cumulative distribution of areas."""
    valid = np.sort(areas_ha[areas_ha > 0])
    if len(valid) < 20:
        return None

    log_vals = np.log10(valid)
    x = np.linspace(0.0, 1.0, len(log_vals))
    y = np.arange(len(log_vals), dtype=float) / len(log_vals)
    line = y[0] + (y[-1] - y[0]) * x
    dist = line - y
    idx = int(np.argmax(dist))
    return float(valid[idx])


def _bottom_fraction_mean(areas_ha: np.ndarray, fraction: float = 0.05) -> float | None:
    valid = np.sort(areas_ha[areas_ha > 0])
    if len(valid) < 10:
        return None
    k = max(1, int(np.ceil(len(valid) * fraction)))
    return float(valid[:k].mean())


def recommend_for_areas(areas_ha: np.ndarray, *, min_polygons: int) -> dict:
    valid = areas_ha[areas_ha > 0]
    n = int(len(valid))
    stats = {
        "polygon_count": n,
        "area_ha_min": float(valid.min()) if n else None,
        "area_ha_p5": float(np.percentile(valid, 5)) if n else None,
        "area_ha_p10": float(np.percentile(valid, 10)) if n else None,
        "area_ha_p25": float(np.percentile(valid, 25)) if n else None,
        "area_ha_median": float(np.percentile(valid, 50)) if n else None,
        "area_ha_p75": float(np.percentile(valid, 75)) if n else None,
        "area_ha_p95": float(np.percentile(valid, 95)) if n else None,
        "area_ha_max": float(valid.max()) if n else None,
        "area_ha_mean": float(valid.mean()) if n else None,
    }

    recs: dict[str, float | None] = {key: None for key in RULE_KEYS.values()}
    if n < min_polygons:
        stats["threshold_recommendations"] = recs
        stats["threshold_note"] = f"fewer than {min_polygons} polygons; thresholds not computed"
        return stats

    recs[RULE_KEYS["p5"]] = float(np.percentile(valid, 5))
    recs[RULE_KEYS["p10"]] = float(np.percentile(valid, 10))
    recs[RULE_KEYS["p25"]] = float(np.percentile(valid, 25))
    recs[RULE_KEYS["elbow"]] = _elbow_threshold_ha(valid)
    recs[RULE_KEYS["bottom5_mean"]] = _bottom_fraction_mean(valid, fraction=0.05)

    # Back-compat aliases for filter_polygons_by_threshold.py
    recs["rule_score_threshold_ha"] = recs[RULE_KEYS["p10"]]
    recs["rule_area_cap_threshold_ha"] = recs[RULE_KEYS["p25"]]
    recs["rule_elbow_threshold_ha"] = recs[RULE_KEYS["elbow"]]

    stats["threshold_recommendations"] = {k: v for k, v in recs.items() if v is not None}
    return stats


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files found in {input_dir} with pattern {args.pattern!r}")

    by_region: dict[str, list[float]] = {r: [] for r in ALLOWED_REGIONS}
    by_region_year: dict[str, dict[str, list[float]]] = {r: {} for r in ALLOWED_REGIONS}
    skipped = 0
    for path in files:
        region = parse_region(path)
        year = parse_calendar_year(path)
        if region not in ALLOWED_REGIONS or year is None:
            skipped += 1
            continue
        areas = load_areas_ha(path, args.target_crs)
        if len(areas):
            by_region[region].extend(areas.tolist())
            year_key = str(year)
            by_region_year[region].setdefault(year_key, []).extend(areas.tolist())

    all_areas = [a for region in ALLOWED_REGIONS for a in by_region[region]]
    payload: dict = {
        "input_dir": str(input_dir),
        "target_crs": args.target_crs,
        "files_total": len(files),
        "files_skipped_no_region_or_year": skipped,
        "global": recommend_for_areas(np.array(all_areas, dtype=float), min_polygons=args.min_polygons),
        "by_region": {},
        "by_region_year": {},
    }

    rows: list[dict] = []
    for region in ALLOWED_REGIONS:
        areas = np.array(by_region[region], dtype=float)
        region_stats = recommend_for_areas(areas, min_polygons=args.min_polygons)
        payload["by_region"][region] = region_stats
        recs = region_stats.get("threshold_recommendations", {})
        rows.append(
            {
                "region": region,
                "polygon_count": region_stats["polygon_count"],
                "p5_ha": recs.get(RULE_KEYS["p5"]),
                "p10_ha": recs.get(RULE_KEYS["p10"]),
                "p25_ha": recs.get(RULE_KEYS["p25"]),
                "elbow_ha": recs.get(RULE_KEYS["elbow"]),
                "bottom5_mean_ha": recs.get(RULE_KEYS["bottom5_mean"]),
                "median_ha": region_stats.get("area_ha_median"),
            }
        )
        print(
            f"[INFO] region r{region}: n={region_stats['polygon_count']} "
            f"p5={recs.get(RULE_KEYS['p5'], 'n/a')} ha "
            f"p10={recs.get(RULE_KEYS['p10'], 'n/a')} ha",
            flush=True,
        )

    rows_ry: list[dict] = []
    for region in ALLOWED_REGIONS:
        payload["by_region_year"][region] = {}
        for year_key in sorted(by_region_year[region], key=int):
            areas = np.array(by_region_year[region][year_key], dtype=float)
            year_stats = recommend_for_areas(areas, min_polygons=args.min_polygons)
            payload["by_region_year"][region][year_key] = year_stats
            recs = year_stats.get("threshold_recommendations", {})
            rows_ry.append(
                {
                    "region": region,
                    "year": int(year_key),
                    "polygon_count": year_stats["polygon_count"],
                    "p5_ha": recs.get(RULE_KEYS["p5"]),
                    "p10_ha": recs.get(RULE_KEYS["p10"]),
                    "p25_ha": recs.get(RULE_KEYS["p25"]),
                    "elbow_ha": recs.get(RULE_KEYS["elbow"]),
                    "bottom5_mean_ha": recs.get(RULE_KEYS["bottom5_mean"]),
                    "median_ha": year_stats.get("area_ha_median"),
                }
            )
            print(
                f"[INFO] region r{region} year {year_key}: n={year_stats['polygon_count']} "
                f"p25={recs.get(RULE_KEYS['p25'], 'n/a')} ha "
                f"elbow={recs.get(RULE_KEYS['elbow'], 'n/a')} ha",
                flush=True,
            )

    summary_path = output_dir / "threshold_summary.json"
    csv_path = output_dir / "thresholds_by_region.csv"
    csv_ry_path = output_dir / "thresholds_by_region_year.csv"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(rows_ry).to_csv(csv_ry_path, index=False)

    print(f"[INFO] Wrote {summary_path}", flush=True)
    print(f"[INFO] Wrote {csv_path}", flush=True)
    print(f"[INFO] Wrote {csv_ry_path}", flush=True)
    print(
        "[INFO] Suggested next step:\n"
        "  python filtering/filter_polygons_by_threshold.py \\\n"
        f"    --input-dir {input_dir} \\\n"
        f"    --output-gpkg {output_dir / 'polygons_filtered.gpkg'} \\\n"
        f"    --stats-summary-json {summary_path} \\\n"
        "    --threshold-rule p25 --per-region-year",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
