#!/usr/bin/env python3
"""Plot the polygon-area distribution (in hectares) of a vector layer.

Expects the input layer to contain a numeric column with areas in hectares
(by default ``area_ha``), as produced by ``reproject_to_equal_area.py``.

One histogram is produced with a logarithmic (base-10) x-axis.  The tick
labels show linear-equivalent hectare values (e.g. 0.1, 1, 10, 100 ...).
Below the histogram a thin ruler axis is drawn on the same figure using a
linear scale over the same data range, so the two x-axes can be compared
side by side.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single-panel area-distribution histogram (log x-axis) "
            "with a parallel linear-scale ruler below it."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input vector file with an area-in-hectares column.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image file (.png, .pdf, .svg).",
    )
    parser.add_argument(
        "--area-column",
        default="area_ha",
        help="Column with polygon area in hectares (default: area_ha).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins (default: 60).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name for multi-layer formats such as GPKG.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def ha_formatter(value: float, _pos: int) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.0f}"
    return f"{value:g}"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input vector not found: {input_path}")

    read_kwargs = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(input_path, **read_kwargs)

    if args.area_column not in gdf.columns:
        raise ValueError(
            f"Column '{args.area_column}' not found in {input_path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    areas = gdf[args.area_column].astype(float).to_numpy()
    areas = areas[np.isfinite(areas) & (areas > 0)]
    if areas.size == 0:
        raise ValueError("No positive, finite area values to plot.")

    n = int(areas.size)
    a_min = float(areas.min())
    a_max = float(areas.max())
    a_median = float(np.median(areas))
    a_mean = float(areas.mean())

    print(f"[INFO] Features with valid area: {n:,}")
    print(
        f"[INFO] Min / median / mean / max (ha): "
        f"{a_min:,.2f} / {a_median:,.2f} / {a_mean:,.2f} / {a_max:,.2f}"
    )

    log_bins = np.logspace(np.log10(a_min), np.log10(a_max), args.bins + 1)

    # Layout: histogram takes the top portion; linear ruler sits below it.
    fig = plt.figure(figsize=(11, 6))
    ax_hist = fig.add_axes([0.10, 0.28, 0.86, 0.60])
    ax_ruler = fig.add_axes([0.10, 0.10, 0.86, 0.09])

    # --- Histogram (log x-axis) ---
    ax_hist.hist(
        areas,
        bins=log_bins,
        color="#3a7d44",
        edgecolor="black",
        linewidth=0.3,
    )
    ax_hist.set_xscale("log")
    ax_hist.set_xlim(a_min, a_max)
    ax_hist.set_ylabel("Number of polygons")
    ax_hist.set_xlabel("Area (ha)  —  log scale, linear-equivalent tick labels")
    ax_hist.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_hist.xaxis.set_major_formatter(FuncFormatter(ha_formatter))
    ax_hist.xaxis.set_minor_formatter(NullFormatter())

    stats_text = (
        f"n = {n:,}\n"
        f"min    = {a_min:,.2f} ha\n"
        f"median = {a_median:,.2f} ha\n"
        f"mean   = {a_mean:,.2f} ha\n"
        f"max    = {a_max:,.2f} ha"
    )
    ax_hist.text(
        0.98,
        0.97,
        stats_text,
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "#888",
        },
    )

    # --- Linear ruler ---
    ax_ruler.set_xlim(a_min, a_max)
    ax_ruler.set_ylim(0, 1)
    ax_ruler.set_xscale("linear")
    ax_ruler.yaxis.set_visible(False)
    ax_ruler.spines[["top", "right", "left"]].set_visible(False)
    ax_ruler.set_xlabel("Area (ha)  —  linear scale (same range)")
    ax_ruler.xaxis.set_major_formatter(FuncFormatter(ha_formatter))

    # Shade the region where most data sits (< median) to guide the eye.
    ax_ruler.axvspan(a_min, a_median, alpha=0.25, color="#3a7d44", label=f"≤ median ({a_median:,.2f} ha)")
    ax_ruler.axvspan(a_median, a_max, alpha=0.08, color="#888888")
    ax_ruler.axvline(a_median, color="#3a7d44", linewidth=1.2, linestyle="--")
    ax_ruler.axvline(a_mean, color="#b5651d", linewidth=1.2, linestyle="--", label=f"mean ({a_mean:,.2f} ha)")
    ax_ruler.legend(loc="upper right", fontsize=8, framealpha=0.85)

    title = args.title if args.title else f"Polygon area distribution — {input_path.name}"
    fig.suptitle(title, fontsize=13, y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
