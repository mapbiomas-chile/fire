# Validation

[Español](README.es.md) | **English**

Tools to prepare **reference fire-scar layers** and quantify agreement against mapped burns after classification / filtering / vectorization. Equal-area projections are required before area-based metrics.

Related: [classification](../classification/README.md), [filtering](../filtering/README.md), [vectorize](../vectorize/README.md).

---

## 1. Typical validation chain

```text
Reference scars (Season / year attribute)
        │
        ▼
Equal-area reprojection (Chile Albers preset)
        │
        ▼
Optional split by calendar year
        │
        ▼
Classified product polygonized (filtering/polygonize or vectorize outputs)
        │  (same CRS)
        ▼
Intersect scars × classified polygons → hits GeoPackage
        │
        ├─► Jaccard index per scar (calculate_jaccard_index.py)
        └─► Singh et al. (2015) closeness D / D_norm (spatial_validation_metrics.py)
```

Polygonization of classified rasters is implemented under **`filtering/polygonize_mask_parallel.py`**, not in this folder.

---

## 2. Module inventory

| Script | Purpose |
|--------|---------|
| `reproject_vector_to_equal_area.py` | Vector → equal-area CRS; add `area_m2`, `area_ha` |
| `reproject_raster_to_equal_area.py` | Raster warp to the same CRS (nearest for masks) |
| `merge_reprojected_tiles_by_year.py` | Mosaic regional tiles into yearly rasters |
| `split_vector_by_year.py` | One GPKG per year (default year column `Season`) |
| `plot_area_distribution.py` | Area histogram (log scale + linear ruler) |
| `intersect_top_n_scars_with_classified.py` | Scar–classified intersection (`scar`, `classified_hits`) |
| `calculate_jaccard_index.py` | Per-scar Jaccard \(J = |A∩B| / |A∪B|\) |
| `spatial_validation_metrics.py` | Over/under-segmentation, \(D\), \(D_{\mathrm{norm}}\), TP/FP/FN |
| `extract_top_fire_events.py` | Export top-\(N\) events from national `chile_YYYY_events.gpkg` |

Spatial metrics dependencies: `pip install -r validation/requirements-spatial-validation.txt`.

---

## 3. Equal-area CRS presets

| Preset | Definition |
|--------|------------|
| `chile_albers` (default) | Custom Albers for Chile (`+proj=aea +lat_1=-18 +lat_2=-55 +lat_0=-37 +lon_0=-71 +datum=WGS84 +units=m`) |
| `south_america_albers` | ESRI:102033 |

Override with any `pyproj`-valid `--target-crs`.

---

## 4. Intersection and accuracy metrics

**Intersections.** The catalog must be projected. Matching is **calendar year** from scar attributes vs year token in classified GPKG names. Optional `--top-n` selects largest scars (globally or per year with `--by-year`). Output layers:

- `scar` — reference geometry \(A\)  
- `classified_hits` — full intersecting classified polygons \(b_i\) (not clipped)

**Jaccard.** For each `scar_id`, \(B = \bigcup_i b_i\),  
\(J = \mathrm{area}(A∩B)/\mathrm{area}(A∪B)\). Unhit scars receive \(J = 0\).

**Singh et al. (2015).** Pairwise over- and under-segmentation;  
\(D = \sqrt{OS^2 + US^2}\), \(D_{\mathrm{norm}} = 1 - D/\sqrt{2}\). Scar-level commission/omission, Dice, Jaccard optional by region (`--by-region`).

---

## 5. Minimal reproducible examples

```bash
# Reference to equal-area
python validation/reproject_vector_to_equal_area.py \
  --input cicatrices.shp --output cicatrices_albers.gpkg --preset chile_albers

# Hits (one year)
python validation/intersect_top_n_scars_with_classified.py \
  --catalog cicatrices_albers.gpkg --year 2017 --top-n 50 \
  --classified-dir /path/to/polygons_2017 \
  --output hits_2017.gpkg --workers 8

# Jaccard
python validation/calculate_jaccard_index.py \
  --hits-gpkg hits_2017.gpkg --output-csv jaccard_2017.csv

# Spatial metrics suite
python validation/spatial_validation_metrics.py \
  --hits-dir /path/to/hits_by_year \
  --hits-pattern "hits_*.gpkg" \
  --output-dir /path/to/spatial_metrics --by-region
```

---

## 6. Top national events (from vectorize national products)

```bash
python validation/extract_top_fire_events.py \
  --input-dir /path/to/polygons_chile \
  --output-gpkg top50.gpkg \
  --from-year 2014 --to-year 2025 \
  --min-ha 200 --max-ha 5000 --top-n 50
```
