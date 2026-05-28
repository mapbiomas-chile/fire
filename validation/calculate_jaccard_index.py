#!/usr/bin/env python3
"""Compute Jaccard index for fire validation using vector layers.

Two input modes:

1. **Hits GeoPackage** (recommended): output of ``intersect_top_n_scars_with_classified.py``
   with layers ``scar`` (reference polygon **A**) and ``classified_hits`` (polygons **bᵢ**
   that intersect **A**). For each scar, builds **B = unary_union(bᵢ)** and computes::

       J = area(A ∩ B) / area(A ∪ B)
       area(A ∪ B) = area(A) + area(B) − area(A ∩ B)

   Scars with no classified hits get **J = 0** (empty **B**).

   With ``--by-region``, **B** is built per MapBiomas tile (``region`` in ``classified_hits``,
   e.g. ``r1``), yielding one row per (scar, region). Use ``--summary-csv`` for mean/median
   counts grouped by ``scar_year`` and ``region``.

2. **Legacy**: a single intersection layer plus total reference and classified areas
   (from vectors or scalar ``--*-area-m2``), same metric as mode 1 when layers are
   one polygon each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.ops import unary_union


def _geom_for_ops(geom):
    """Repair invalid geometries so GEOS intersection/union does not raise."""
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    return make_valid(geom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Jaccard index J = intersection_area / union_area. "
            "Default input is a hits GeoPackage from intersect_top_n_scars_with_classified.py "
            "(layers scar + classified_hits). Alternatively pass --intersection with "
            "--reference/--classified or fixed areas."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hits-gpkg",
        metavar="PATH",
        help=(
            "GeoPackage from intersect_top_n_scars_with_classified.py "
            "(layers 'scar' and 'classified_hits' unless overridden)."
        ),
    )
    src.add_argument(
        "--hits-dir",
        metavar="DIR",
        help=(
            "Process every GeoPackage matching --hits-pattern under this directory "
            "(writes one CSV per file to --output-dir)."
        ),
    )
    src.add_argument(
        "--intersection",
        metavar="PATH",
        help=(
            "Legacy: single vector of A ∩ B (or equivalent). "
            "Requires --reference or --reference-area-m2, and "
            "--classified or --classified-area-m2."
        ),
    )
    parser.add_argument(
        "--scar-layer",
        default="scar",
        help="Layer name for scars when using --hits-gpkg (default: scar).",
    )
    parser.add_argument(
        "--classified-hits-layer",
        default="classified_hits",
        help=(
            "Layer name for classified hits when using --hits-gpkg "
            "(default: classified_hits)."
        ),
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Legacy: reference fire vector file path.",
    )
    parser.add_argument(
        "--classified",
        default=None,
        help="Legacy: classified fire vector file path.",
    )
    parser.add_argument(
        "--reference-area-m2",
        type=float,
        default=None,
        help="Legacy: total area of the reference layer in m2 (if --reference omitted).",
    )
    parser.add_argument(
        "--classified-area-m2",
        type=float,
        default=None,
        help="Legacy: total area of the classified layer in m2 (if --classified omitted).",
    )
    parser.add_argument(
        "--intersection-area-column",
        default=None,
        help=(
            "Legacy only: optional field with intersection area in m2. "
            "If omitted, area is computed from geometry."
        ),
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Legacy: layer name for --intersection, --reference, and --classified.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Output CSV path (required with --hits-gpkg or legacy mode; "
            "optional with --hits-dir if only --summary-csv is needed)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="With --hits-dir: directory for per-file CSV outputs (required).",
    )
    parser.add_argument(
        "--hits-pattern",
        default="*.gpkg",
        help="Glob under --hits-dir (default: *.gpkg).",
    )
    parser.add_argument(
        "--by-region",
        action="store_true",
        help=(
            "Split classified hits by 'region' and compute Jaccard per (scar, region). "
            "Requires 'region' on classified_hits (from intersect script)."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=(
            "Optional path for aggregated metrics by scar_year and region "
            "(mean/median/count of jaccard_index)."
        ),
    )
    return parser.parse_args()


def read_layer(path: Path, layer_name: str | None = None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    kwargs = {"layer": layer_name} if layer_name else {}
    return gpd.read_file(path, **kwargs)


def validate_projected(gdf: gpd.GeoDataFrame, label: str) -> None:
    if gdf.crs is None:
        raise ValueError(f"{label} has no CRS. Use a projected CRS in meters.")
    if not gdf.crs.is_projected:
        raise ValueError(f"{label} has geographic CRS. Reproject to a metric CRS first.")


def get_total_area_m2(gdf: gpd.GeoDataFrame, label: str) -> float:
    clean = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if clean.empty:
        raise ValueError(f"{label} has no valid geometries.")
    return float(clean.geometry.area.sum())


def get_intersection_area_m2(
    gdf: gpd.GeoDataFrame,
    area_column: str | None,
) -> float:
    if area_column:
        if area_column not in gdf.columns:
            raise ValueError(f"Column not found in intersection layer: {area_column}")
        values = pd.to_numeric(gdf[area_column], errors="coerce").fillna(0.0)
        return float(values.sum())

    clean = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if clean.empty:
        return 0.0
    return float(clean.geometry.area.sum())


def jaccard_from_areas(
    reference_area_m2: float,
    classified_area_m2: float,
    intersection_area_m2: float,
) -> tuple[float, float]:
    if reference_area_m2 < 0 or classified_area_m2 < 0:
        raise ValueError("Input areas must be non-negative.")
    if intersection_area_m2 < 0:
        raise ValueError("Intersection area must be non-negative.")

    union_area_m2 = reference_area_m2 + classified_area_m2 - intersection_area_m2
    if union_area_m2 <= 0:
        raise ValueError(
            "Union area is not positive. Check input layers/areas and overlap values."
        )
    jaccard_index = intersection_area_m2 / union_area_m2
    return union_area_m2, jaccard_index


def _jaccard_record_for_scar(
    scar_id: str,
    scar_row: pd.Series,
    geom_a,
    hits_sub: gpd.GeoDataFrame,
    region: str | None,
) -> dict:
    ref_area = float(geom_a.area)
    if hits_sub.empty:
        b_union = unary_union([])
    else:
        geoms = hits_sub.geometry
        geoms = geoms[geoms.notna() & ~geoms.is_empty]
        if geoms.empty:
            b_union = unary_union([])
        else:
            fixed = [_geom_for_ops(g) for g in geoms.values if g is not None and not g.is_empty]
            b_union = unary_union(fixed) if fixed else unary_union([])

    b_union = _geom_for_ops(b_union) if b_union is not None and not b_union.is_empty else b_union
    inter = geom_a.intersection(b_union)
    intersection_area_m2 = float(inter.area) if not inter.is_empty else 0.0
    classified_area_m2 = float(b_union.area) if not b_union.is_empty else 0.0

    if classified_area_m2 == 0.0 and intersection_area_m2 == 0.0:
        union_area_m2 = ref_area
        jaccard_index = 0.0
    else:
        union_area_m2, jaccard_index = jaccard_from_areas(
            ref_area,
            classified_area_m2,
            intersection_area_m2,
        )

    record: dict = {"scar_id": scar_id}
    if region is not None:
        record["region"] = region
    if "scar_year" in scar_row.index and pd.notna(scar_row["scar_year"]):
        record["scar_year"] = int(scar_row["scar_year"])
    record.update(
        {
            "reference_area_m2": ref_area,
            "classified_union_area_m2": classified_area_m2,
            "intersection_area_m2": intersection_area_m2,
            "union_area_m2": union_area_m2,
            "jaccard_index": jaccard_index,
            "jaccard_percent": jaccard_index * 100.0,
        }
    )
    return record


def summarize_by_year_region(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-scar metrics by scar_year and region."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "scar_year",
                "region",
                "n_records",
                "mean_jaccard_index",
                "median_jaccard_index",
                "mean_jaccard_percent",
            ]
        )
    group_cols = []
    if "scar_year" in df.columns:
        group_cols.append("scar_year")
    if "region" in df.columns:
        group_cols.append("region")
    if not group_cols:
        raise ValueError("Cannot summarize: need scar_year and/or region columns.")

    agg = (
        df.groupby(group_cols, dropna=False)["jaccard_index"]
        .agg(n_records="count", mean_jaccard_index="mean", median_jaccard_index="median")
        .reset_index()
    )
    agg["mean_jaccard_percent"] = agg["mean_jaccard_index"] * 100.0
    return agg.sort_values(group_cols).reset_index(drop=True)


def compute_hits_gpkg_metrics(
    hits_path: Path,
    scar_layer: str,
    hits_layer: str,
    by_region: bool = False,
) -> pd.DataFrame:
    scar_gdf = read_layer(hits_path, scar_layer)
    classified_gdf = read_layer(hits_path, hits_layer)

    validate_projected(scar_gdf, f"Layer {scar_layer!r}")
    validate_projected(classified_gdf, f"Layer {hits_layer!r}")

    if not scar_gdf.crs.equals(classified_gdf.crs):
        raise ValueError(
            f"CRS mismatch: {scar_layer!r} vs {hits_layer!r} in {hits_path}."
        )

    if "scar_id" not in scar_gdf.columns:
        raise ValueError(f"Layer {scar_layer!r} must contain column 'scar_id'.")
    if classified_gdf.empty:
        pass
    elif "scar_id" not in classified_gdf.columns:
        raise ValueError(f"Layer {hits_layer!r} must contain column 'scar_id'.")

    use_region = by_region and "region" in classified_gdf.columns
    if by_region and not use_region:
        print(
            "[WARN] --by-region set but 'region' missing on classified_hits; "
            "using one union per scar."
        )

    rows: list[dict] = []

    for _, scar_row in scar_gdf.iterrows():
        scar_id = scar_row["scar_id"]
        geom_a = _geom_for_ops(scar_row.geometry)
        if geom_a is None or geom_a.is_empty:
            raise ValueError(f"Empty scar geometry for scar_id={scar_id!r}.")

        hits_all = classified_gdf.loc[classified_gdf["scar_id"] == scar_id]

        if use_region and not hits_all.empty:
            regions = sorted(hits_all["region"].dropna().astype(str).unique())
            for region in regions:
                hits_sub = hits_all.loc[hits_all["region"].astype(str) == region]
                rows.append(
                    _jaccard_record_for_scar(scar_id, scar_row, geom_a, hits_sub, region)
                )
        elif use_region and hits_all.empty:
            rows.append(_jaccard_record_for_scar(scar_id, scar_row, geom_a, hits_all, None))
        else:
            rows.append(_jaccard_record_for_scar(scar_id, scar_row, geom_a, hits_all, None))

    return pd.DataFrame(rows)


def _write_hits_result(
    result: pd.DataFrame,
    hits_path: Path,
    output_csv: Path | None,
) -> None:
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"[INFO] Wrote metrics to: {output_csv}")
    print(f"[INFO] Processed {len(result)} row(s) from: {hits_path}")
    if not result.empty:
        mean_j = float(result["jaccard_index"].mean())
        print(f"[INFO] Mean Jaccard index (all rows): {mean_j:.6f}")
        if "region" in result.columns and "scar_year" in result.columns:
            summary = summarize_by_year_region(result)
            for _, row in summary.iterrows():
                print(
                    f"[INFO]   year={row['scar_year']} region={row['region']}: "
                    f"n={int(row['n_records'])} mean_J={row['mean_jaccard_index']:.6f}"
                )


def main() -> int:
    args = parse_args()

    if args.hits_dir:
        hits_dir = Path(args.hits_dir)
        if not hits_dir.is_dir():
            raise FileNotFoundError(f"Hits directory not found: {hits_dir}")
        if not args.output_dir:
            raise SystemExit("--hits-dir requires --output-dir.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(hits_dir.glob(args.hits_pattern))
        if not paths:
            raise RuntimeError(f"No files matching {args.hits_pattern!r} in {hits_dir}")

        combined: list[pd.DataFrame] = []
        for hits_path in paths:
            result = compute_hits_gpkg_metrics(
                hits_path,
                args.scar_layer,
                args.classified_hits_layer,
                by_region=args.by_region,
            )
            out_csv = output_dir / f"{hits_path.stem}_jaccard.csv"
            _write_hits_result(result, hits_path, out_csv)
            combined.append(result)

        all_df = pd.concat(combined, ignore_index=True)
        if args.summary_csv:
            summary_path = Path(args.summary_csv)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summarize_by_year_region(all_df).to_csv(summary_path, index=False)
            print(f"[INFO] Wrote year/region summary to: {summary_path}")
        return 0

    if args.hits_gpkg:
        if not args.output_csv:
            raise SystemExit("--output-csv is required with --hits-gpkg.")
        hits_path = Path(args.hits_gpkg)
        output_csv = Path(args.output_csv)
        result = compute_hits_gpkg_metrics(
            hits_path,
            args.scar_layer,
            args.classified_hits_layer,
            by_region=args.by_region,
        )
        _write_hits_result(result, hits_path, output_csv)
        if args.summary_csv:
            summary_path = Path(args.summary_csv)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summarize_by_year_region(result).to_csv(summary_path, index=False)
            print(f"[INFO] Wrote year/region summary to: {summary_path}")
        return 0

    if not args.output_csv:
        raise SystemExit("--output-csv is required in legacy mode.")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Legacy mode
    intersection_path = Path(args.intersection)
    intersection_gdf = read_layer(intersection_path, args.layer)
    validate_projected(intersection_gdf, "Intersection layer")
    intersection_area_m2 = get_intersection_area_m2(
        gdf=intersection_gdf,
        area_column=args.intersection_area_column,
    )

    if args.reference:
        reference_gdf = read_layer(Path(args.reference), args.layer)
        validate_projected(reference_gdf, "Reference layer")
        reference_area_m2 = get_total_area_m2(reference_gdf, "Reference layer")
    elif args.reference_area_m2 is not None:
        reference_area_m2 = float(args.reference_area_m2)
    else:
        raise ValueError(
            "Legacy mode: provide --reference or --reference-area-m2."
        )

    if args.classified:
        classified_gdf = read_layer(Path(args.classified), args.layer)
        validate_projected(classified_gdf, "Classified layer")
        classified_area_m2 = get_total_area_m2(classified_gdf, "Classified layer")
    elif args.classified_area_m2 is not None:
        classified_area_m2 = float(args.classified_area_m2)
    else:
        raise ValueError(
            "Legacy mode: provide --classified or --classified-area-m2."
        )

    union_area_m2, jaccard_index = jaccard_from_areas(
        reference_area_m2,
        classified_area_m2,
        intersection_area_m2,
    )

    result = pd.DataFrame(
        [
            {
                "intersection_area_m2": intersection_area_m2,
                "reference_area_m2": reference_area_m2,
                "classified_area_m2": classified_area_m2,
                "union_area_m2": union_area_m2,
                "jaccard_index": jaccard_index,
                "jaccard_percent": jaccard_index * 100.0,
            }
        ]
    )
    result.to_csv(output_csv, index=False)

    print(f"[INFO] Intersection area (m2): {intersection_area_m2:.3f}")
    print(f"[INFO] Reference area (m2): {reference_area_m2:.3f}")
    print(f"[INFO] Classified area (m2): {classified_area_m2:.3f}")
    print(f"[INFO] Union area (m2): {union_area_m2:.3f}")
    print(f"[INFO] Jaccard index: {jaccard_index:.6f}")
    print(f"[INFO] Jaccard (%): {jaccard_index * 100.0:.3f}")
    print(f"[INFO] Wrote metrics to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
