#!/usr/bin/env python3
"""Generate polygon-area histograms by file, grouped in region folders."""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


ALLOWED_REGIONS = {"1", "2", "4", "6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read polygon files, group by region from filename (rX), and write "
            "one histogram per file under region_<X>/histogramas using file prefix."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory with polygon files (.gpkg).")
    parser.add_argument("--output-dir", required=True, help="Base output directory.")
    parser.add_argument("--pattern", default="*.gpkg", help="Input glob pattern (default: *.gpkg).")
    parser.add_argument(
        "--target-crs",
        default="EPSG:32719",
        help="Projected CRS to compute area (default: EPSG:32719).",
    )
    parser.add_argument(
        "--xscale",
        choices=["linear", "log"],
        default="log",
        help="Histogram x-axis scale (default: log).",
    )
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins (default: 40).")
    parser.add_argument(
        "--linear-ref-ticks",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000],
        help="Linear reference ticks shown as extra bottom axis when xscale=log.",
    )
    return parser.parse_args()


def extract_region(path: Path) -> str | None:
    match = re.search(r"_r([0-9]+)_", path.stem)
    if not match:
        return None
    region = match.group(1)
    if region not in ALLOWED_REGIONS:
        return None
    return region


def load_area_ha(gpkg_path: Path, target_crs: str) -> np.ndarray:
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return np.array([], dtype=float)

    if target_crs:
        gdf = gdf.to_crs(target_crs)
    elif gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError(
            f"{gpkg_path} has no projected CRS. Use --target-crs to compute area in meters."
        )

    return (gdf.geometry.area.to_numpy(dtype=float) / 10000.0).astype(float)


def save_histogram(
    area_ha: np.ndarray,
    out_png: Path,
    xscale: str,
    bins: int,
    linear_ref_ticks: list[float],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    valid = area_ha[area_ha > 0]
    if len(valid) == 0:
        ax.text(0.5, 0.5, "Sin poligonos > 0 ha", ha="center", va="center")
        ax.set_axis_off()
    else:
        if xscale == "log":
            bin_edges = np.logspace(np.log10(valid.min()), np.log10(valid.max()), bins)
            ax.hist(valid, bins=bin_edges, color="#2a9d8f", alpha=0.9)
            ax.set_xscale("log")
            valid_refs = [r for r in linear_ref_ticks if valid.min() <= r <= valid.max()]
            if valid_refs:
                secax = ax.secondary_xaxis("bottom")
                secax.spines["bottom"].set_position(("outward", 28))
                secax.set_xticks(valid_refs)
                secax.set_xticklabels([f"{r:g}" for r in valid_refs])
                secax.set_xlabel("Linear reference ticks (ha)")
        else:
            ax.hist(valid, bins=bins, color="#2a9d8f", alpha=0.9)
        ax.set_xlabel(f"Area (ha, {xscale})")
        ax.set_ylabel("Cantidad de poligonos")
        ax.set_title("Histograma de area de poligonos")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files found in {input_dir} with pattern {args.pattern}")

    grouped: dict[str, list[Path]] = {r: [] for r in sorted(ALLOWED_REGIONS)}
    skipped = 0
    for path in files:
        region = extract_region(path)
        if region is None:
            skipped += 1
            continue
        grouped[region].append(path)

    for region, region_files in grouped.items():
        region_total_polygons = 0
        for path in region_files:
            area = load_area_ha(path, args.target_crs)
            region_total_polygons += len(area)

            # Keep source stem to avoid overwrite across files/years.
            hist_path = (
                output_dir
                / f"region_{region}"
                / "histogramas"
                / f"histograma_{path.stem}.png"
            )
            save_histogram(
                area,
                hist_path,
                args.xscale,
                args.bins,
                args.linear_ref_ticks,
            )
            print(f"[INFO] Wrote {hist_path} (polygons={len(area)})")

        print(
            f"[INFO] region_{region}: files={len(region_files)} polygons={region_total_polygons}"
        )

    if skipped:
        print(f"[INFO] Skipped files without valid region token r1/r2/r4/r6: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
