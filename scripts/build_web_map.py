#!/usr/bin/env python3
"""
Convert FIRIS FLI GeoTIFF to a transparent PNG overlay for Leaflet
and create metadata JSON containing WGS84 bounds.

Output files:
    data/web/fli_latest.png
    data/web/fli_latest.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


WGS84 = CRS.from_epsg(4326)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create PNG and JSON files for the FIRIS Leaflet web map."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input FLI GeoTIFF, e.g. data/outputs/fli_fars_2026-08-27.tif",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory, e.g. data/web",
    )
    return parser.parse_args()


def normalize_fli(values: np.ndarray) -> np.ndarray:
    """
    FLI normally ranges from 0 to 100.
    Values outside this range are clipped safely.
    """
    return np.clip(values.astype(np.float32), 0.0, 100.0)


def fli_to_rgba(fli: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Create a fire-risk color ramp:

    0-20   : green
    20-40  : yellow
    40-60  : orange
    60-80  : red
    80-100 : dark red
    """
    rgba = np.zeros((fli.shape[0], fli.shape[1], 4), dtype=np.uint8)

    # Very low risk: green
    mask = valid_mask & (fli < 20)
    rgba[mask] = [46, 125, 50, 190]

    # Low to moderate: yellow
    mask = valid_mask & (fli >= 20) & (fli < 40)
    rgba[mask] = [253, 216, 53, 200]

    # Moderate to high: orange
    mask = valid_mask & (fli >= 40) & (fli < 60)
    rgba[mask] = [251, 140, 0, 210]

    # High: red
    mask = valid_mask & (fli >= 60) & (fli < 80)
    rgba[mask] = [229, 57, 53, 220]

    # Very high / extreme: dark red
    mask = valid_mask & (fli >= 80)
    rgba[mask] = [136, 14, 79, 230]

    return rgba


def read_as_wgs84(input_path: Path):
    """Read source raster and reproject it to EPSG:4326 if required."""
    with rasterio.open(input_path) as src:
        source_data = src.read(1, masked=True)

        if src.crs is None:
            raise ValueError(
                "Input GeoTIFF has no CRS. A valid geographic coordinate system is required."
            )

        if src.crs == WGS84:
            data = source_data
            bounds = src.bounds
            return data, bounds

        transform, width, height = calculate_default_transform(
            src.crs,
            WGS84,
            src.width,
            src.height,
            *src.bounds,
        )

        destination = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=np.asarray(source_data.filled(np.nan), dtype=np.float32),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=WGS84,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        south, west, north, east = array_bounds(height, width, transform)

        return np.ma.masked_invalid(destination), rasterio.coords.BoundingBox(
            left=west,
            bottom=south,
            right=east,
            top=north,
        )


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"FLI GeoTIFF not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "fli_latest.png"
    json_path = output_dir / "fli_latest.json"

    raster, bounds = read_as_wgs84(input_path)

    raw_values = np.asarray(raster.filled(np.nan), dtype=np.float32)
    valid_mask = np.isfinite(raw_values)

    if not np.any(valid_mask):
        raise ValueError("The input FLI raster contains no valid pixels.")

    fli = normalize_fli(np.nan_to_num(raw_values, nan=0.0))
    rgba = fli_to_rgba(fli, valid_mask)

    image = Image.fromarray(rgba, mode="RGBA")
    image.save(png_path, format="PNG", optimize=True)

    valid_values = raw_values[valid_mask]

    metadata = {
        "title": "FIRIS – Fars Fire Risk Index",
        "source_file": input_path.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "crs": "EPSG:4326",
        "image": "fli_latest.png",
        "bounds": [
            [round(bounds.bottom, 8), round(bounds.left, 8)],
            [round(bounds.top, 8), round(bounds.right, 8)],
        ],
        "statistics": {
            "min": round(float(np.min(valid_values)), 2),
            "max": round(float(np.max(valid_values)), 2),
            "mean": round(float(np.mean(valid_values)), 2),
        },
        "legend": [
            {"min": 0, "max": 20, "label": "کم", "color": "#2e7d32"},
            {"min": 20, "max": 40, "label": "متوسط", "color": "#fdd835"},
            {"min": 40, "max": 60, "label": "نسبتاً زیاد", "color": "#fb8c00"},
            {"min": 60, "max": 80, "label": "زیاد", "color": "#e53935"},
            {"min": 80, "max": 100, "label": "بسیار زیاد", "color": "#880e4f"},
        ],
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Created PNG : {png_path}")
    print(f"Created JSON: {json_path}")
    print(f"Bounds      : {metadata['bounds']}")


if __name__ == "__main__":
    main()
