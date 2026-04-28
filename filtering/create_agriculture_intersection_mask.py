#!/usr/bin/env python3
"""Binary mask: 1 where a LUCL class equals the target code in every band (year), else 0."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Leyenda MapBiomas uso del suelo: 18 = agricultura.
DEFAULT_OUTPUT_NAME = "mascara_agricultura_interseccion_todos_anos.tif"
DEFAULT_CLASS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una máscara binaria: 1 donde el código LUCL coincide con "
            "--class-code en todas las bandas (años), 0 en el resto (intersección "
            "temporal, equivalente a AND sobre años)."
        )
    )
    parser.add_argument(
        "--input-tif",
        type=Path,
        required=True,
        help="Raster LCLU multi-banda (una banda por año).",
    )
    parser.add_argument(
        "--output-tif",
        help=f"Ruta del GeoTIFF de salida (por defecto: --output-dir o carpeta del input / {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directorio de salida (se usa si --output-tif no está definido).",
    )
    parser.add_argument(
        "--class-code",
        type=int,
        default=DEFAULT_CLASS,
        help=f"Código de clase LUCL (default: {DEFAULT_CLASS}, agricultura).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Tamaño de bloque en píxeles (default: 2048).",
    )
    return parser.parse_args()


def iter_windows(height: int, width: int, chunk_size: int):
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def resolve_output_path(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output_tif:
        return Path(args.output_tif)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / DEFAULT_OUTPUT_NAME


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_tif)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    output_path = resolve_output_path(args, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        if src.count < 1:
            raise ValueError("Input raster has no bands.")

        n_bands = src.count
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for window in iter_windows(src.height, src.width, args.chunk_size):
                block = src.read(window=window)
                mask = np.all(block == args.class_code, axis=0).astype(np.uint8)
                dst.write(mask, 1, window=window)

    print(f"[INFO] Bands (years) intersected: {n_bands}")
    if n_bands == 1:
        print(
            "[WARN] Solo hay 1 banda: la salida es la máscara de ese año; "
            "para intersección multianual hace falta un GeoTIFF con una banda por año."
        )
    print(f"[INFO] Class code: {args.class_code}")
    print(f"[INFO] Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
