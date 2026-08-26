#!/usr/bin/env python3
"""
FIRIS Web Map Builder
=====================

Convert the latest FIRIS FLI GeoTIFF into a web-ready, transparent,
five-class PNG overlay for Leaflet and generate metadata JSON.

Input:
    data/outputs/fli_fars_YYYY-MM-DD.tif

Outputs:
    docs/data/fli_YYYY-MM-DD.png
    docs/data/fli_YYYY-MM-DD.json
    docs/data/fli_latest.png
    docs/data/fli_latest.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import calculate_default_transform, reproject


# CRS استاندارد قابل استفاده در Leaflet و مرورگر
WGS84 = CRS.from_epsg(4326)


# طبقه‌بندی پنج‌گانهٔ خطر آتش‌سوزی
# بازه‌ها:
# 0 تا کمتر از 20       = خیلی کم
# 20 تا کمتر از 40      = کم
# 40 تا کمتر از 60      = متوسط
# 60 تا کمتر از 80      = زیاد
# 80 تا 100             = خیلی زیاد
RISK_CLASSES: list[dict[str, Any]] = [
    {
        "id": 1,
        "name_fa": "خیلی کم",
        "name_en": "Very Low",
        "min": 0.0,
        "max": 20.0,
        "color": "#2DC653",
        "rgba": [45, 198, 83, 255],
    },
    {
        "id": 2,
        "name_fa": "کم",
        "name_en": "Low",
        "min": 20.0,
        "max": 40.0,
        "color": "#A7C957",
        "rgba": [167, 201, 87, 255],
    },
    {
        "id": 3,
        "name_fa": "متوسط",
        "name_en": "Moderate",
        "min": 40.0,
        "max": 60.0,
        "color": "#F9C74F",
        "rgba": [249, 199, 79, 255],
    },
    {
        "id": 4,
        "name_fa": "زیاد",
        "name_en": "High",
        "min": 60.0,
        "max": 80.0,
        "color": "#F9844A",
        "rgba": [249, 132, 74, 255],
    },
    {
        "id": 5,
        "name_fa": "خیلی زیاد",
        "name_en": "Very High",
        "min": 80.0,
        "max": 100.0,
        "color": "#D62828",
        "rgba": [214, 40, 40, 255],
    },
]


def find_latest_fli(input_dir: Path) -> Path:
    """
    جدیدترین فایل FLI را از پوشهٔ خروجی پیدا می‌کند.

    انتظار:
        data/outputs/fli_fars_YYYY-MM-DD.tif
    """
    files = sorted(
        input_dir.glob("fli_fars_*.tif"),
        key=lambda item: item.name,
    )

    if not files:
        raise FileNotFoundError(
            f"هیچ فایل FLI در مسیر زیر پیدا نشد:\n"
            f"{input_dir}\n\n"
            f"فرمت مورد انتظار:\n"
            f"fli_fars_YYYY-MM-DD.tif"
        )

    return files[-1]


def extract_date_from_filename(input_path: Path) -> str:
    """
    تاریخ را از نام فایل استخراج می‌کند.

    مثال:
        fli_fars_2026-08-27.tif
        -> 2026-08-27
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", input_path.name)

    if match:
        return match.group(1)

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_raster_as_wgs84(input_path: Path) -> tuple[np.ndarray, Affine]:
    """
    رستر FLI را می‌خواند.

    اگر CRS فایل EPSG:4326 نباشد، آن را به EPSG:4326 تبدیل می‌کند؛
    زیرا Leaflet برای نمایش Overlay به مختصات طول/عرض جغرافیایی نیاز دارد.

    خروجی:
        - آرایهٔ FLI از نوع float32
        - transform مربوط به EPSG:4326
    """
    with rasterio.open(input_path) as source:
        if source.crs is None:
            raise ValueError(
                f"فایل ورودی CRS ندارد و برای نقشهٔ وب قابل استفاده نیست:\n"
                f"{input_path}"
            )

        data = source.read(1).astype(np.float32)

        # تبدیل مقادیر NoData، NaN و Infinity به NaN
        if source.nodata is not None:
            data[data == source.nodata] = np.nan

        data[~np.isfinite(data)] = np.nan

        # اگر فایل در WGS84 است، تبدیل مختصاتی نیاز نیست.
        if source.crs == WGS84:
            return data, source.transform

        # تبدیل رستر به EPSG:4326 برای Leaflet
        destination_transform, destination_width, destination_height = (
            calculate_default_transform(
                source.crs,
                WGS84,
                source.width,
                source.height,
                *source.bounds,
            )
        )

        destination = np.full(
            (destination_height, destination_width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=data,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs=WGS84,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        destination[~np.isfinite(destination)] = np.nan

        return destination, destination_transform


def classify_fli_to_rgba(
    fli_data: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], int]:
    """
    مقادیر پیوستهٔ FLI را به تصویر RGBA تبدیل می‌کند.

    پیکسل‌های خارج از محدودهٔ 0 تا 100 یا NoData:
        کاملاً شفاف می‌شوند.

    خروجی:
        rgba:
            آرایهٔ 4 باندی RGBA با ساختار (4, height, width)
        class_counts:
            تعداد پیکسل هر کلاس
        valid_pixel_count:
            تعداد کل پیکسل‌های معتبر
    """
    height, width = fli_data.shape

    # چهار باند Red, Green, Blue, Alpha
    rgba = np.zeros((4, height, width), dtype=np.uint8)

    # فقط مقادیر متناهی و بین 0 تا 100 معتبر هستند.
    valid_mask = (
        np.isfinite(fli_data)
        & (fli_data >= 0.0)
        & (fli_data <= 100.0)
    )

    valid_pixel_count = int(np.count_nonzero(valid_mask))

    # پیکسل‌های معتبر غیرشفاف هستند.
    rgba[3, valid_mask] = 255

    class_counts: dict[str, int] = {}

    for risk_class in RISK_CLASSES:
        class_id = risk_class["id"]
        minimum = float(risk_class["min"])
        maximum = float(risk_class["max"])
        red, green, blue, alpha = risk_class["rgba"]

        # کلاس آخر شامل مقدار دقیق 100 نیز هست.
        if class_id == 5:
            mask = valid_mask & (fli_data >= minimum) & (fli_data <= maximum)
        else:
            mask = valid_mask & (fli_data >= minimum) & (fli_data < maximum)

        rgba[0, mask] = red
        rgba[1, mask] = green
        rgba[2, mask] = blue
        rgba[3, mask] = alpha

        class_counts[str(class_id)] = int(np.count_nonzero(mask))

    return rgba, class_counts, valid_pixel_count


def write_png(output_path: Path, rgba: np.ndarray) -> None:
    """
    تصویر RGBA را به PNG قابل استفاده در مرورگر تبدیل می‌کند.
    """
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
    ) as destination:
        destination.write(rgba)

        destination.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )


def calculate_leaflet_bounds(
    transform: Affine,
    width: int,
    height: int,
) -> list[list[float]]:
    """
    محدودهٔ تصویر را با فرمت مورد نیاز Leaflet برمی‌گرداند.

    Leaflet format:
        [
          [south, west],
          [north, east]
        ]
    """
    west, south, east, north = array_bounds(height, width, transform)

    return [
        [round(float(south), 8), round(float(west), 8)],
        [round(float(north), 8), round(float(east), 8)],
    ]


def build_metadata(
    input_path: Path,
    image_filename: str,
    data_date: str,
    transform: Affine,
    width: int,
    height: int,
    valid_pixel_count: int,
    class_counts: dict[str, int],
) -> dict[str, Any]:
    """
    اطلاعات تکمیلی مورد نیاز سایت و Leaflet را ایجاد می‌کند.
    """
    clean_risk_classes = []

    for risk_class in RISK_CLASSES:
        clean_risk_classes.append(
            {
                "id": risk_class["id"],
                "name_fa": risk_class["name_fa"],
                "name_en": risk_class["name_en"],
                "min": risk_class["min"],
                "max": risk_class["max"],
                "color": risk_class["color"],
                "pixel_count": class_counts.get(str(risk_class["id"]), 0),
            }
        )

    return {
        "project": "FIRIS",
        "project_full_name": "Fars Fire Risk Information System",
        "title_fa": "نقشه شاخص خطر آتش‌سوزی استان فارس",
        "title_en": "Fars Fire Risk Index Map",
        "data_date": data_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_geotiff": input_path.as_posix(),
        "image": image_filename,
        "crs": "EPSG:4326",
        "leaflet_bounds": calculate_leaflet_bounds(
            transform=transform,
            width=width,
            height=height,
        ),
        "image_size": {
            "width": width,
            "height": height,
        },
        "statistics": {
            "valid_fli_pixels": valid_pixel_count,
            "risk_class_pixel_counts": class_counts,
        },
        "risk_classes": clean_risk_classes,
    }


def write_json(output_path: Path, content: dict[str, Any]) -> None:
    """
    JSON استاندارد تولید می‌کند.
    allow_nan=False تضمین می‌کند NaN یا Infinity وارد JSON نشوند.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the FIRIS FLI GeoTIFF into a five-class PNG "
            "and Leaflet metadata JSON."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "مسیر فایل ورودی FLI GeoTIFF. "
            "اگر وارد نشود، جدیدترین fli_fars_*.tif از data/outputs انتخاب می‌شود."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/data"),
        help="پوشهٔ خروجی فایل‌های مورد نیاز سایت. پیش‌فرض: docs/data",
    )

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir

    if input_path is None:
        input_path = find_latest_fli(Path("data/outputs"))

    if not input_path.exists():
        raise FileNotFoundError(
            f"فایل ورودی پیدا نشد:\n{input_path}"
        )

    data_date = extract_date_from_filename(input_path)

    dated_png_path = output_dir / f"fli_{data_date}.png"
    dated_json_path = output_dir / f"fli_{data_date}.json"

    latest_png_path = output_dir / "fli_latest.png"
    latest_json_path = output_dir / "fli_latest.json"

    print("=" * 70)
    print("FIRIS Web Map Builder")
    print("=" * 70)
    print(f"Input GeoTIFF: {input_path}")

    fli_data, transform = read_raster_as_wgs84(input_path)

    rgba, class_counts, valid_pixel_count = classify_fli_to_rgba(fli_data)

    _, height, width = rgba.shape

    write_png(dated_png_path, rgba)

    dated_metadata = build_metadata(
        input_path=input_path,
        image_filename=dated_png_path.name,
        data_date=data_date,
        transform=transform,
        width=width,
        height=height,
        valid_pixel_count=valid_pixel_count,
        class_counts=class_counts,
    )

    write_json(dated_json_path, dated_metadata)

    # فایل latest برای سایت همیشه نام ثابت دارد.
    shutil.copy2(dated_png_path, latest_png_path)

    latest_metadata = build_metadata(
        input_path=input_path,
        image_filename=latest_png_path.name,
        data_date=data_date,
        transform=transform,
        width=width,
        height=height,
        valid_pixel_count=valid_pixel_count,
        class_counts=class_counts,
    )

    write_json(latest_json_path, latest_metadata)

    print()
    print("Files created successfully:")
    print(f"  PNG dated:    {dated_png_path}")
    print(f"  JSON dated:   {dated_json_path}")
    print(f"  PNG latest:   {latest_png_path}")
    print(f"  JSON latest:  {latest_json_path}")
    print()
    print(f"Valid FLI pixels: {valid_pixel_count:,}")
    print("Risk class pixel counts:")

    for risk_class in RISK_CLASSES:
        class_id = str(risk_class["id"])
        count = class_counts.get(class_id, 0)

        print(
            f"  {risk_class['id']}. "
            f"{risk_class['name_fa']} "
            f"({risk_class['min']:.0f}-{risk_class['max']:.0f}): "
            f"{count:,}"
        )

    print()
    print(
        "Leaflet bounds: "
        f"{latest_metadata['leaflet_bounds']}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
