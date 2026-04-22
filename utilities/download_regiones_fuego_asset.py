#!/usr/bin/env python3
"""Exporta un FeatureCollection de GEE a Google Drive."""

import ee

# ====== CONFIGURACION (autocontenida) ======
ASSET_ID = "projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1"
PROJECT_ID = "mapbiomas-chile"  # o None
DRIVE_FOLDER = "regiones_fuego"  # carpeta en tu Google Drive (se crea si no existe)
FILE_NAME = "regiones_fuego_chile_v1"
OUTPUT_FORMAT = "geojson"  # "geojson" o "gpkg"
# ===========================================


def init_ee() -> None:
    try:
        if PROJECT_ID:
            ee.Initialize(project=PROJECT_ID)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if PROJECT_ID:
            ee.Initialize(project=PROJECT_ID)
        else:
            ee.Initialize()


def main() -> None:
    init_ee()
    fc = ee.FeatureCollection(ASSET_ID)

    # Earth Engine -> Drive NO soporta GPKG directo.
    # Si eliges "gpkg", se exporta como GeoJSON para luego convertirlo fuera de GEE.
    if OUTPUT_FORMAT.lower() == "gpkg":
        gee_format = "GeoJSON"
        print("[WARN] GEE no exporta GPKG directo a Drive. Exportando como GeoJSON.")
    else:
        gee_format = "GeoJSON"

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"export_{FILE_NAME}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=FILE_NAME,
        fileFormat=gee_format,
    )
    task.start()

    print("[INFO] Tarea enviada a GEE.")
    print(f"[INFO] Asset: {ASSET_ID}")
    print(f"[INFO] Formato: {gee_format}")
    print(f"[INFO] Drive folder: {DRIVE_FOLDER}")
    print("[INFO] Revisa progreso en: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
