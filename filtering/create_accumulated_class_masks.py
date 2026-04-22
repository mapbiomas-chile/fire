#!/usr/bin/env python3
"""Create accumulated binary masks (OR across all bands) for selected classes."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


CLASS_SPECS = [
    (29, "mascara_alfloramiento_rocoso_acumulado.tif"),
    (23, "mascara_arena_playa_duna_acumulado.tif"),
    (61, "mascara_salar_acumulado.tif"),
    (34, "mascara_hielo_nieve_acumulado.tif"),
    (25, "mascara_otra_area_sin_vegetacion_acumulado.tif"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one binary mask per class using OR across all raster bands "
            "(accumulated across years)."
        )
    )
    parser.add_argument("--input-tif", required=True, help="Input multi-band raster path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output mask TIFFs will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Processing chunk size in pixels (default: 2048).",
    )
    return parser.parse_args()


def iter_windows(height: int, width: int, chunk_size: int):
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_tif)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        if src.count < 1:
            raise ValueError("Input raster has no bands.")

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        outputs = {}
        try:
            for _, filename in CLASS_SPECS:
                out_path = output_dir / filename
                outputs[filename] = rasterio.open(out_path, "w", **profile)

            for window in iter_windows(src.height, src.width, args.chunk_size):
                block = src.read(window=window)  # shape: (bands, rows, cols)
                for class_value, filename in CLASS_SPECS:
                    mask = np.any(block == class_value, axis=0).astype(np.uint8)
                    outputs[filename].write(mask, 1, window=window)
        finally:
            for dst in outputs.values():
                dst.close()

    for _, filename in CLASS_SPECS:
        print(f"[INFO] Saved: {output_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
