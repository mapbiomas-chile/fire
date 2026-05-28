#!/usr/bin/env python3
"""Spatial validation metrics between reference fire scars and classified polygons.

Implements the closeness index **D** from Singh et al. (2015) for each reference–segment
pair, plus scar-level **TP / FP / FN**, commission/omission errors, Jaccard (IoU), and Dice.

**Input (recommended):** GeoPackage from ``intersect_top_n_scars_with_classified.py`` with
layers ``scar`` (reference) and ``classified_hits`` (classified segments that intersect each scar).

Pairwise metrics (reference *i*, segment *j*)::

    OverSegmentation  = 1 - A_intersect / A_reference
    UnderSegmentation = 1 - A_intersect / A_segment
    D = sqrt(OverSegmentation^2 + UnderSegmentation^2)
    D_norm = 1 - D / sqrt(2)    # 1 = perfect match, 0 = no overlap

Scar-level metrics (union of all segments *j* that intersect scar *i*)::

    TP = area(reference ∩ union(segments))
    FP = area(union(segments) \\ reference)
    FN = area(reference \\ union(segments))
    Commission = FP / (FP + TP)
    Omission   = FN / (FN + TP)
    Jaccard = TP / (TP + FP + FN)
    Dice    = 2·TP / (2·TP + FP + FN)

Install dependencies::

    python -m pip install -r validation/requirements-spatial-validation.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.ops import unary_union

SQRT2 = math.sqrt(2.0)
M2_TO_HA = 1.0 / 10_000.0


def _geom_for_ops(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    return make_valid(geom)


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


def closeness_d_pair(
    area_reference_m2: float,
    area_segment_m2: float,
    area_intersection_m2: float,
) -> tuple[float, float, float, float]:
    """Return (over_segmentation, under_segmentation, D, D_norm)."""
    if area_reference_m2 <= 0 or area_segment_m2 <= 0:
        raise ValueError("Reference and segment areas must be positive for pairwise D.")

    a_int = max(0.0, float(area_intersection_m2))
    a_int = min(a_int, area_reference_m2, area_segment_m2)

    over = 1.0 - a_int / area_reference_m2
    under = 1.0 - a_int / area_segment_m2
    d = math.sqrt(over * over + under * under)
    d_norm = 1.0 - d / SQRT2
    return over, under, d, d_norm


def confusion_from_geometries(
    geom_reference,
    geom_classified_union,
) -> tuple[float, float, float]:
    """Return (TP, FP, FN) in m²."""
    geom_reference = _geom_for_ops(geom_reference)
    if geom_reference is None or geom_reference.is_empty:
        raise ValueError("Empty reference geometry.")

    area_ref = float(geom_reference.area)
    if geom_classified_union is None or geom_classified_union.is_empty:
        return 0.0, 0.0, area_ref

    geom_classified_union = _geom_for_ops(geom_classified_union)
    inter = geom_reference.intersection(geom_classified_union)
    tp = float(inter.area) if not inter.is_empty else 0.0
    tp = min(tp, area_ref)

    area_cls = float(geom_classified_union.area)
    fp = max(0.0, area_cls - tp)
    fn = max(0.0, area_ref - tp)
    return tp, fp, fn


def metrics_from_confusion(
    tp: float,
    fp: float,
    fn: float,
    area_reference_m2: float,
    area_classified_m2: float,
) -> dict:
    """Derive commission, omission, Jaccard, Dice, and area summaries."""
    tp = max(0.0, tp)
    fp = max(0.0, fp)
    fn = max(0.0, fn)

    union = tp + fp + fn
    jaccard = tp / union if union > 0 else 0.0
    dice_denom = 2.0 * tp + fp + fn
    dice = (2.0 * tp / dice_denom) if dice_denom > 0 else 0.0

    comm_denom = fp + tp
    commission_error = fp / comm_denom if comm_denom > 0 else float("nan")
    omit_denom = fn + tp
    omission_error = fn / omit_denom if omit_denom > 0 else float("nan")

    ref_ha = area_reference_m2 * M2_TO_HA
    cls_ha = area_classified_m2 * M2_TO_HA
    tp_ha = tp * M2_TO_HA

    pct_ref_detected = (100.0 * tp / area_reference_m2) if area_reference_m2 > 0 else 0.0
    pct_classified_is_reference = (
        (100.0 * tp / area_classified_m2) if area_classified_m2 > 0 else 0.0
    )

    return {
        "tp_m2": tp,
        "fp_m2": fp,
        "fn_m2": fn,
        "commission_error": commission_error,
        "omission_error": omission_error,
        "jaccard_index": jaccard,
        "jaccard_percent": jaccard * 100.0,
        "dice_index": dice,
        "dice_percent": dice * 100.0,
        "reference_area_ha": ref_ha,
        "classified_area_ha": cls_ha,
        "intersection_area_ha": tp_ha,
        "pct_reference_detected": pct_ref_detected,
        "pct_classified_is_reference": pct_classified_is_reference,
    }


def compute_pairwise_rows(
    scar_id: str,
    scar_row: pd.Series,
    geom_reference,
    hits_sub: gpd.GeoDataFrame,
    region: str | None,
) -> list[dict]:
    """One row per reference–segment pair with Singh et al. D metrics."""
    geom_reference = _geom_for_ops(geom_reference)
    area_ref = float(geom_reference.area)
    rows: list[dict] = []

    if hits_sub.empty:
        return rows

    for seg_idx, seg_row in hits_sub.iterrows():
        geom_seg = _geom_for_ops(seg_row.geometry)
        if geom_seg is None or geom_seg.is_empty:
            continue

        area_seg = float(geom_seg.area)
        inter = geom_reference.intersection(geom_seg)
        area_int = float(inter.area) if not inter.is_empty else 0.0

        over, under, d, d_norm = closeness_d_pair(area_ref, area_seg, area_int)

        record: dict = {
            "scar_id": scar_id,
            "segment_id": seg_idx,
            "reference_area_m2": area_ref,
            "segment_area_m2": area_seg,
            "intersection_area_m2": area_int,
            "over_segmentation": over,
            "under_segmentation": under,
            "closeness_D": d,
            "closeness_D_norm": d_norm,
        }
        if region is not None:
            record["region"] = region
        if "scar_year" in scar_row.index and pd.notna(scar_row["scar_year"]):
            record["scar_year"] = int(scar_row["scar_year"])
        if "classified_file" in seg_row.index and pd.notna(seg_row["classified_file"]):
            record["classified_file"] = seg_row["classified_file"]
        rows.append(record)

    return rows


def compute_scar_summary_row(
    scar_id: str,
    scar_row: pd.Series,
    geom_reference,
    hits_sub: gpd.GeoDataFrame,
    pairwise_df: pd.DataFrame,
    region: str | None,
) -> dict:
    """Scar-level TP/FP/FN and best/mean pairwise D_norm."""
    geom_reference = _geom_for_ops(geom_reference)
    area_ref = float(geom_reference.area)

    if hits_sub.empty:
        geom_union = unary_union([])
        area_cls = 0.0
    else:
        geoms = hits_sub.geometry
        geoms = geoms[geoms.notna() & ~geoms.is_empty]
        if geoms.empty:
            geom_union = unary_union([])
            area_cls = 0.0
        else:
            fixed = [_geom_for_ops(g) for g in geoms.values]
            geom_union = unary_union(fixed)
            area_cls = float(geom_union.area) if geom_union is not None else 0.0

    tp, fp, fn = confusion_from_geometries(geom_reference, geom_union)
    base = metrics_from_confusion(tp, fp, fn, area_ref, area_cls)

    record: dict = {
        "scar_id": scar_id,
        "n_segment_pairs": len(pairwise_df),
        **base,
    }
    if region is not None:
        record["region"] = region
    if "scar_year" in scar_row.index and pd.notna(scar_row["scar_year"]):
        record["scar_year"] = int(scar_row["scar_year"])

    if not pairwise_df.empty and "closeness_D_norm" in pairwise_df.columns:
        record["closeness_D_norm_best"] = float(pairwise_df["closeness_D_norm"].max())
        record["closeness_D_norm_mean"] = float(pairwise_df["closeness_D_norm"].mean())
        record["closeness_D_min"] = float(pairwise_df["closeness_D"].min())
    else:
        record["closeness_D_norm_best"] = 0.0
        record["closeness_D_norm_mean"] = 0.0
        record["closeness_D_min"] = SQRT2

    return record


def compute_hits_metrics(
    hits_path: Path,
    scar_layer: str,
    hits_layer: str,
    by_region: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (pairwise_df, scar_summary_df).
    """
    scar_gdf = read_layer(hits_path, scar_layer)
    classified_gdf = read_layer(hits_path, hits_layer)

    validate_projected(scar_gdf, scar_layer)
    validate_projected(classified_gdf, hits_layer)

    if not scar_gdf.crs.equals(classified_gdf.crs):
        raise ValueError(f"CRS mismatch in {hits_path}.")

    if "scar_id" not in scar_gdf.columns:
        raise ValueError(f"Layer {scar_layer!r} must contain 'scar_id'.")

    use_region = by_region and "region" in classified_gdf.columns
    if by_region and not use_region:
        print(
            "[WARN] --by-region set but 'region' missing on classified_hits; "
            "one union per scar."
        )

    pairwise_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, scar_row in scar_gdf.iterrows():
        scar_id = scar_row["scar_id"]
        geom_a = _geom_for_ops(scar_row.geometry)
        if geom_a is None or geom_a.is_empty:
            raise ValueError(f"Empty scar geometry for scar_id={scar_id!r}.")

        hits_all = classified_gdf.loc[classified_gdf["scar_id"] == scar_id]

        if use_region and not hits_all.empty:
            regions = sorted(hits_all["region"].dropna().astype(str).unique())
            groups: list[tuple[str | None, gpd.GeoDataFrame]] = [
                (r, hits_all.loc[hits_all["region"].astype(str) == r]) for r in regions
            ]
        elif use_region and hits_all.empty:
            groups = [(None, hits_all)]
        else:
            groups = [(None, hits_all)]

        for region, hits_sub in groups:
            pair_rows = compute_pairwise_rows(scar_id, scar_row, geom_a, hits_sub, region)
            pairwise_rows.extend(pair_rows)
            pair_df = pd.DataFrame(pair_rows)
            summary_rows.append(
                compute_scar_summary_row(
                    scar_id, scar_row, geom_a, hits_sub, pair_df, region
                )
            )

    return pd.DataFrame(pairwise_rows), pd.DataFrame(summary_rows)


def summarize_by_year_region(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ("scar_year", "region") if c in summary_df.columns]
    if not group_cols:
        return pd.DataFrame()

    numeric = [
        "jaccard_index",
        "dice_index",
        "closeness_D_norm_best",
        "closeness_D_norm_mean",
        "commission_error",
        "omission_error",
        "pct_reference_detected",
        "pct_classified_is_reference",
    ]
    present = [c for c in numeric if c in summary_df.columns]

    agg_dict: dict = {c: "mean" for c in present}
    agg_dict["scar_id"] = "count"

    out = summary_df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    out = out.rename(columns={"scar_id": "n_scars"})
    return out.sort_values(group_cols).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Spatial validation: Singh et al. (2015) closeness D, TP/FP/FN, "
            "commission/omission, Jaccard, Dice."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hits-gpkg", metavar="PATH", help="Single hits GeoPackage.")
    src.add_argument(
        "--hits-dir",
        metavar="DIR",
        help="Directory of hits GeoPackages (use with --hits-pattern).",
    )

    p.add_argument("--scar-layer", default="scar")
    p.add_argument("--classified-hits-layer", default="classified_hits")
    p.add_argument("--hits-pattern", default="*.gpkg")
    p.add_argument(
        "--by-region",
        action="store_true",
        help="Compute metrics per MapBiomas tile (region column on hits).",
    )
    p.add_argument(
        "--pairwise-csv",
        default=None,
        help="Output CSV for all reference–segment pairs (with --hits-gpkg).",
    )
    p.add_argument(
        "--summary-csv",
        default=None,
        help="Output CSV for scar-level metrics (with --hits-gpkg).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="With --hits-dir: write {stem}_pairwise.csv and {stem}_summary.csv here.",
    )
    p.add_argument(
        "--aggregate-summary-csv",
        default=None,
        help="With --hits-dir: combined summary by scar_year and region.",
    )
    return p.parse_args()


def _process_one_hits(
    hits_path: Path,
    scar_layer: str,
    hits_layer: str,
    by_region: bool,
    pairwise_csv: Path | None,
    summary_csv: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_df, summary_df = compute_hits_metrics(
        hits_path, scar_layer, hits_layer, by_region=by_region
    )

    if pairwise_csv is not None:
        pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
        pairwise_df.to_csv(pairwise_csv, index=False)
        print(f"[INFO] Wrote {len(pairwise_df)} pair row(s) to: {pairwise_csv}")

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        print(f"[INFO] Wrote {len(summary_df)} scar summary row(s) to: {summary_csv}")

    print(f"[INFO] Processed: {hits_path}")
    if not summary_df.empty and "jaccard_index" in summary_df.columns:
        print(f"[INFO]   Mean Jaccard: {summary_df['jaccard_index'].mean():.6f}")
        print(f"[INFO]   Mean D_norm (best pair): {summary_df['closeness_D_norm_best'].mean():.6f}")

    return pairwise_df, summary_df


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

        all_summaries: list[pd.DataFrame] = []
        for hits_path in paths:
            stem = hits_path.stem
            _, summary_df = _process_one_hits(
                hits_path,
                args.scar_layer,
                args.classified_hits_layer,
                args.by_region,
                output_dir / f"{stem}_pairwise.csv",
                output_dir / f"{stem}_summary.csv",
            )
            all_summaries.append(summary_df)

        combined = pd.concat(all_summaries, ignore_index=True)
        if args.aggregate_summary_csv:
            agg_path = Path(args.aggregate_summary_csv)
            agg_path.parent.mkdir(parents=True, exist_ok=True)
            summarize_by_year_region(combined).to_csv(agg_path, index=False)
            print(f"[INFO] Wrote aggregate summary to: {agg_path}")
        return 0

    if not args.hits_gpkg:
        raise SystemExit("Provide --hits-gpkg or --hits-dir.")

    hits_path = Path(args.hits_gpkg)
    if not args.pairwise_csv and not args.summary_csv:
        raise SystemExit("With --hits-gpkg provide --pairwise-csv and/or --summary-csv.")

    _process_one_hits(
        hits_path,
        args.scar_layer,
        args.classified_hits_layer,
        args.by_region,
        Path(args.pairwise_csv) if args.pairwise_csv else None,
        Path(args.summary_csv) if args.summary_csv else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
