"""Group nearby burn-scar polygons into fire events (multipolygons)."""

from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def components(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(self.parent)):
            grouped[self.find(idx)].append(idx)
        return list(grouped.values())


def _as_multipolygon(geom) -> MultiPolygon | Polygon:
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return MultiPolygon([])
        merged = unary_union(polys)
        if isinstance(merged, MultiPolygon):
            return merged
        return MultiPolygon([merged])
    return MultiPolygon([])


def group_polygons_by_distance(
    gdf: gpd.GeoDataFrame,
    *,
    max_gap_m: float = 200.0,
    metric_crs: str = "EPSG:32719",
    event_id_prefix: str = "event",
) -> gpd.GeoDataFrame:
    """
    Merge polygon fragments whose buffered geometries overlap within ``max_gap_m``.

    Two scars are grouped when the gap between them is at most ``max_gap_m`` meters
    (implemented as a buffer of ``max_gap_m / 2`` and connected components).
    """
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=[
                "event_id",
                "year",
                "fragment_count",
                "area_m2",
                "area_ha",
                "max_gap_m",
                "geometry",
            ],
            geometry="geometry",
            crs=gdf.crs,
        )

    if max_gap_m <= 0:
        raise ValueError("max_gap_m must be > 0")

    projected = gdf.to_crs(metric_crs)
    buffer_distance = max_gap_m / 2.0
    buffered_geoms = list(projected.geometry.buffer(buffer_distance))

    tree = STRtree(buffered_geoms)
    uf = _UnionFind(len(buffered_geoms))

    for idx, geom in enumerate(buffered_geoms):
        if geom.is_empty:
            continue
        for other_idx in tree.query(geom):
            if other_idx <= idx:
                continue
            if geom.intersects(buffered_geoms[other_idx]):
                uf.union(idx, other_idx)

    year_value = None
    if "year" in gdf.columns and gdf["year"].notna().any():
        unique_years = gdf["year"].dropna().unique()
        if len(unique_years) == 1:
            year_value = unique_years[0]

    rows: list[dict] = []
    for comp_id, component in enumerate(sorted(uf.components(), key=min), start=1):
        geoms = [projected.geometry.iloc[i] for i in component]
        merged = _as_multipolygon(unary_union(geoms))
        if merged.is_empty:
            continue
        area_m2 = float(merged.area)
        rows.append(
            {
                "event_id": f"{event_id_prefix}_{comp_id:06d}",
                "year": year_value,
                "fragment_count": len(component),
                "area_m2": area_m2,
                "area_ha": area_m2 / 10000.0,
                "max_gap_m": float(max_gap_m),
                "geometry": merged,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "event_id",
                "year",
                "fragment_count",
                "area_m2",
                "area_ha",
                "max_gap_m",
                "geometry",
            ],
            geometry="geometry",
            crs=metric_crs,
        )

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=metric_crs)
    if gdf.crs is not None and out.crs != gdf.crs:
        out = out.to_crs(gdf.crs)
    return out


def summarize_grouping(raw_count: int, grouped: gpd.GeoDataFrame) -> dict:
    return {
        "raw_polygon_count": int(raw_count),
        "event_count": int(len(grouped)),
        "fragments_in_events": int(grouped["fragment_count"].sum()) if not grouped.empty else 0,
        "total_area_ha": float(grouped["area_ha"].sum()) if not grouped.empty else 0.0,
    }
