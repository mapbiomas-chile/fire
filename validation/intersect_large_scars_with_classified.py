#!/usr/bin/env python3
"""Keep original classified polygons that intersect each large fire scar."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each large fire scar, keep the original classified polygons from "
            "the same year that intersect it and write one output GeoPackage."
        )
    )
    parser.add_argument(
        "--scars-dir",
        required=True,
        help="Directory with one large-scar GeoPackage per file.",
    )
    parser.add_argument(
        "--classified-dir",
        required=True,
        help="Directory with yearly classified polygon GeoPackages.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where one output GeoPackage per scar will be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Number of parallel workers (default: 15).",
    )
    return parser.parse_args()


def extract_year(value: object) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty date value.")
    try:
        return datetime.fromisoformat(text[:10]).year
    except ValueError:
        return int(text[:4])


def extract_classified_year(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    return int(parts[3])


def extract_region(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse region from filename: {path.name}")
    return parts[2]


def process_one_scar(scar_path_str: str, classified_by_year: dict[int, list[str]], output_dir_str: str):
    scar_path = Path(scar_path_str)
    output_dir = Path(output_dir_str)

    scar = gpd.read_file(scar_path)
    if scar.empty:
        raise ValueError(f"Scar file is empty: {scar_path}")

    scar_id = str(scar.iloc[0]["FireID"]).strip()
    scar_year = (
        int(scar.iloc[0]["year"])
        if "year" in scar.columns
        else extract_year(scar.iloc[0]["IgnDate"])
    )
    scar_area_ha = float(scar.iloc[0]["area_ha"]) if "area_ha" in scar.columns else None
    scar_geom = scar.geometry.iloc[0]
    scar_bounds = scar_geom.bounds

    frames: list[gpd.GeoDataFrame] = []
    for classified_path_str in classified_by_year.get(scar_year, []):
        classified_path = Path(classified_path_str)
        region = extract_region(classified_path)

        candidates = gpd.read_file(classified_path, bbox=scar_bounds)
        if candidates.empty:
            continue

        candidates = candidates.loc[candidates.geometry.intersects(scar_geom)].copy()
        if candidates.empty:
            continue

        candidates["region"] = region
        candidates["classified_file"] = classified_path.name
        candidates["scar_id"] = scar_id
        candidates["scar_year"] = scar_year
        candidates["scar_area_ha"] = scar_area_ha

        overlap_geom = candidates.geometry.intersection(scar_geom)
        candidates["intersection_area_m2"] = overlap_geom.area.astype(float)
        candidates["intersection_area_ha"] = candidates["intersection_area_m2"] / 10000.0
        frames.append(candidates)

    if frames:
        out_gdf = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            crs=frames[0].crs,
        )
    else:
        out_gdf = gpd.GeoDataFrame(
            {
                "scar_id": [],
                "scar_year": [],
                "scar_area_ha": [],
                "region": [],
                "classified_file": [],
                "intersection_area_m2": [],
                "intersection_area_ha": [],
            },
            geometry=[],
            crs=scar.crs,
        )

    out_path = output_dir / f"{scar_path.stem}_intersections.gpkg"
    out_gdf.to_file(out_path, driver="GPKG")
    return scar_path.name, scar_year, len(out_gdf), out_path.name


def main() -> int:
    args = parse_args()
    scars_dir = Path(args.scars_dir)
    classified_dir = Path(args.classified_dir)
    output_dir = Path(args.output_dir)

    if not scars_dir.exists():
        raise FileNotFoundError(f"Scars directory not found: {scars_dir}")
    if not classified_dir.exists():
        raise FileNotFoundError(f"Classified directory not found: {classified_dir}")

    scar_paths = sorted(scars_dir.glob("*.gpkg"))
    classified_paths = sorted(classified_dir.glob("*.gpkg"))
    if not scar_paths:
        raise RuntimeError(f"No GeoPackages found in: {scars_dir}")
    if not classified_paths:
        raise RuntimeError(f"No GeoPackages found in: {classified_dir}")

    classified_by_year: dict[int, list[str]] = {}
    for path in classified_paths:
        year = extract_classified_year(path)
        classified_by_year.setdefault(year, []).append(str(path))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Scars to process: {len(scar_paths)}")
    print(f"[INFO] Classified files available: {len(classified_paths)}")
    print(f"[INFO] Workers: {args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_scar, str(scar_path), classified_by_year, str(output_dir)): scar_path
            for scar_path in scar_paths
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            scar_name, scar_year, n_rows, out_name = future.result()
            print(
                f"[INFO] Completed ({idx}/{len(scar_paths)}): "
                f"{scar_name} year={scar_year} intersections={n_rows} -> {out_name}"
            )

    print(f"[INFO] Finished writing intersection GeoPackages to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
