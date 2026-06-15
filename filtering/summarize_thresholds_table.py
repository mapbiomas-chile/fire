#!/usr/bin/env python3
"""Print a readable summary of recommended area thresholds (ha) by region and year."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

METHOD_COLUMNS = [
    ("p5", "p5_ha"),
    ("p10", "p10_ha"),
    ("p25", "p25_ha"),
    ("elbow", "elbow_ha"),
    ("bottom5_mean", "bottom5_mean_ha"),
]

ELBOW_WARN_HA = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize thresholds_by_region_year.csv as a compact table."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to thresholds_by_region_year.csv from recommend_polygon_area_thresholds.py.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write a tidy long-format CSV (region, year, method, threshold_ha).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Filter to one region (1, 2, 4, or 6).",
    )
    parser.add_argument(
        "--elbow-warn-ha",
        type=float,
        default=ELBOW_WARN_HA,
        help=f"Mark elbow values above this as suspicious (default: {ELBOW_WARN_HA}).",
    )
    return parser.parse_args()


def _fmt_ha(value: float | None, *, warn: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    text = f"{float(value):.3f}"
    if warn:
        return f"{text}*"
    return text


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    if args.region is not None:
        df = df[df["region"].astype(str) == str(args.region)]
    if df.empty:
        print("No rows to display.", file=sys.stderr)
        return 1

    df = df.sort_values(["region", "year"]).reset_index(drop=True)

    header = (
        f"{'reg':>3} {'year':>4} {'n':>8}  "
        + "  ".join(f"{label:>12}" for label, _ in METHOD_COLUMNS)
    )
    print(header)
    print("-" * len(header))

    long_rows: list[dict] = []
    for _, row in df.iterrows():
        region = int(row["region"])
        year = int(row["year"])
        n = int(row["polygon_count"])
        values: list[str] = []
        for label, col in METHOD_COLUMNS:
            raw = row.get(col)
            val = None if pd.isna(raw) else float(raw)
            warn = label == "elbow" and val is not None and val > args.elbow_warn_ha
            values.append(_fmt_ha(val, warn=warn))
            if val is not None:
                long_rows.append(
                    {
                        "region": region,
                        "year": year,
                        "polygon_count": n,
                        "method": label,
                        "threshold_ha": val,
                        "elbow_suspect": warn,
                    }
                )
        print(
            f"r{region:>2} {year:>4} {n:>8}  "
            + "  ".join(f"{v:>12}" for v in values)
        )

    print()
    print("Units: hectares (ha). * = elbow > {:.1f} ha (review manually).".format(args.elbow_warn_ha))
    print("Methods: p5/p10/p25 = percentiles; elbow = knee on CDF; bottom5_mean = mean of smallest 5%.")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(long_rows).to_csv(out_path, index=False)
        print(f"Wrote long-format CSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
