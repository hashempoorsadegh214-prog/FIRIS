#!/usr/bin/env python3
"""
Build a web-ready 5-class PNG overlay from the latest FIRIS FLI GeoTIFF.

Outputs:
    docs/data/fli_YYYY-MM-DD.png
    docs/data/fli_YYYY-MM-DD.json
    docs/data/fli_latest.png
    docs/data/fli_latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


WGS84 = CRS.from_epsg(4326)

# پنج کلاس خطر FLI
RISK_CLASSES = [
    {
        "id": 1,
        "name_fa": "خیلی کم",
        "name_en": "Very Low",
        "min": 0,
        "max": 20,
        "color": "#2DC653",  # سبز
        "rgba": [45, 198, 83, 255],
    },
    {
        "id": 2,
        "name_fa": "کم",
        "name_en": "Low",
        "min": 20,
        "max": 40,
        "color": "#A7C957",  # سبز روشن
        "rgba": [167, 201, 87, 255],
    },
    {
        "id": 3,
        "name_fa": "متوسط",
        "name_en": "Moderate",
        "min": 40,
        "max": 60,
        "color": "#F9C74F",  # زرد
        "rgba": [249, 199, 79, 255],
    },
    {
        "id": 4,
        "name_fa": "زیاد",
        "name_en": "High",
        "min": 60,
        "max": 80,
        "color": "#F9844A",  # نارنجی
        "rgba": [249, 132, 74, 255],
    },
    {
        "id": 5,
        "name_fa": "خیلی زیاد",
        "name_en": "Very High",
        "min": 80,
        "max": 100,
        "color": "#D62828",  # قرمز
        "rgba": [214, 40, 40, 255],
    },
]


def find_latest_fli(input_dir: Path) -> Path:
    """Find the newest dated FLI raster in data/outputs."""
    files = sorted(input_dir.glob("fli_fars_*.tif"))

    if not files:
        raise FileNotFoundError(
            f"No FLI GeoTIFF found in {input_dir}. "
            "Expected a file like fli_fars_YYYY-MM-DD.tif"
        )

    return files[-1]


def extract_date_from_filename(path: Path) -> str:
    """Extract YYYY-MM-DD from fli_fars_YYYY-MM-DD.tif."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_as_wgs84(source_path: Path) -> tuple[np.ndarray, object]:
    """
    Read FLI raster in EPSG:4326.
    If source is another CRS, reproject to WGS84 using nearest neighbour,
    which preserves discrete risk boundaries.
    """
    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError(
                f"Input raster has no CRS: {source_path}. "
                "A geographic CRS is required for web-map generation."
            )

        source_data = src.read(1).astype(np.float32)
        source_data[~np.isfinite(source_data)] = np.nan

        if src.nodata is not None:
            source_data[source_data == src.nodata] = np.nan

        # Raster already uses WGS84: no reprojection required.
        if src.crs == WGS84:
            return source_data, src.transform

        transform, width, height = calculate_default_transform(
            src.crs,
            WGS84,
            src.width,
            src.height,
            *src.bounds,
        )

        destination = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=WGS84,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        return destination, transform


def make_rgba(fli: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """
    Convert numeric FLI data into an RGBA image.
    Invalid pixels become transparent.
    """
    height, width = fli.shape
    rgba = np.zeros((4, height, width), dtype=np.uint8)

    valid = np.isfinite(fli)
    class_counts: dict[str, int] = {}

    # Alpha=0 is transparent outside valid raster coverage.
    rgba[3, valid] = 255

    for risk_class in RISK_CLASSES:
        class_id = risk_class["id"]
        lower = risk_class["min"]
        upper = risk_class["max"]

        if class_id == 5:
            mask = valid & (fli >= lower)
        else:
            mask = valid & (fli >= lower) & (fli < upper)

        red, green, blue, alpha = risk_class["rgba"]

        rgba[0, mask] = red
        rgba[1, mask] = green
        rgba[2, mask] = blue
        rgba[3, mask] = alpha

        class_counts[str(class_id)] = int(np.count_nonzero(mask))

    # مقادیر نامعمول خارج از بازهٔ 0 تا 100 هم شفاف می‌شوند.
    out_of_range = valid & ((fli < 0) | (fli > 100))
    rgba[:, out_of_range] = 0

    return rgba, class_counts


def write_png(output_path: Path, rgba: np.ndarray) -> None:
    """Write browser-readable RGBA PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, height, width = rgba.shape

    with rasterio.open(
        output_path,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=4,
        dtype=np.uint8,
    ) as dst:
        dst.write(rgba)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )


def build_metadata(
    source_path: Path,
    png_filename: str,
    transform: object,
    width: int,
    height: int,
    valid_pixels: int,
    class_counts: dict[str, int],
    data_date: str,
) -> dict:
    """Create Leaflet-ready metadata JSON."""
    west, south, east, north = array_bounds(height, width, transform)

    # Leaflet bounds format: [[south, west], [north, east]]
    leaflet_bounds = [
        [round(float(south), 8), round(float(west), 8)],
        [round(float(north), 8), round(float(east), 8)],
    ]

    return {
        "project": "FIRIS",
        "title_fa": "نقشه شاخص خطر آتش‌سوزی استان فارس",
        "title_en": "Fars Fire Risk Index Map",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_date": data_date,
        "source_geotiff": str(source_path).replace("\\", "/"),
        "image": png_filename,
        "crs": "EPSG:4326",
        "leaflet_bounds": leaflet_bounds,
        "statistics": {
            "valid_fli_pixels": valid_pixels,
            "risk_class_pixel_counts": class_counts,
        },
        "risk_classes": [
            {
                key: value
                for key, value in risk_class.items()
                if key != "rgba"
            }
            for risk_class in RISK_CLASSES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FIRIS FLI GeoTIFF to a five-class web PNG overlay."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input FLI GeoTIFF. If omitted, newest data/outputs/fli_fars_*.tif is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/data"),
        help="Directory for web PNG and metadata JSON files.",
    )
    args = parser.parse_args()

    input_path = args.input or find_latest_fli(Path("data/outputs"))
    output_dir = args.output_dir
    data_date = extract_date_from_filename(input_path)

    dated_png = output_dir / f"fli_{data_date}.png"
    dated_json = output_dir / f"fli_{data_date}.json"
    latest_png = output_dir / "fli_latest.png"
    latest_json = output_dir / "fli_latest.json"

    fli, transform = read_as_wgs84(input_path)
    rgba, class_counts = make_rgba(fli)

    valid_pixels = int(np.count_nonzero(np.isfinite(fli) & (fli >= 0) & (fli <= 100)))
    _, height, width = rgba.shape

    write_png(dated_png, rgba)

    metadata = build_metadata(
        source_path=input_path,
        png_filename=dated_png.name,
        transform=transform,
        width=width,
        height=height,
        valid_pixels=valid_pixels,
        class_counts=class_counts,
        data_date=data_date,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with dated_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=False)

    # فایل‌های latest، مسیر ثابت برای سایت خواهند بود.
    shutil.copy2(dated_png, latest_png)

    metadata["image"] = latest_png.name
    with latest_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=False)

    print("Web map files generated successfully:")
    print(f"  - {dated_png}")
    print(f"  - {dated_json}")
    print(f"  - {latest_png}")
    print(f"  - {latest_json}")
    print(f"  - Valid FLI pixels: {valid_pixels:,}")
    print(f"  - Leaflet bounds: {metadata['leaflet_bounds']}")


if __name__ == "__main__":
    main()
