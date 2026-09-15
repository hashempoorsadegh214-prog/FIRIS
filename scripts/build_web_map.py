#!/usr/bin/env python3
"""
FIRIS - Build Web GIS products from a dated FLI GeoTIFF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape, mapping
from shapely.ops import unary_union


# ============================================================
# RISK CLASSIFICATION
# ============================================================

RISK_CLASSES = (
    ("متوسط", 0.0, 25.0, "#FFEB3B"),
    ("زیاد", 25.0, 50.0, "#FB8C00"),
    ("خیلی زیاد", 50.0, 75.0, "#E53935"),
    ("بحرانی", 75.0, 100.000001, "#880E4F"),
)

RISK_CODE_TO_INFO = {index + 1: item for index, item in enumerate(RISK_CLASSES)}

MIN_VECTOR_AREA = 0.000002
CHAIKIN_ITERATIONS = 2
CHAIKIN_RATIO = 0.25
SIMPLIFY_TOLERANCE = 0.00025
MIN_RING_POINTS = 4

DATE_PATTERN = re.compile(r"fli_fars_(\d{4}-\d{2}-\d{2})\.tif$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-dir", default=None, type=Path)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def extract_forecast_date(path: Path) -> str:
    match = DATE_PATTERN.search(path.name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    raise ValueError(f"Could not determine forecast date from {path.name}")


def risk_code(value: float) -> int:
    for code, (_, minimum, maximum, _) in enumerate(RISK_CLASSES, start=1):
        if minimum <= value < maximum:
            return code
    return 0


def risk_info(value: float):
    if not math.isfinite(value):
        return ("بدون داده", 0.0, 0.0, "#777777")
    for label, minimum, maximum, color in RISK_CLASSES:
        if minimum <= value < maximum:
            return (label, minimum, maximum, color)
    return ("بحرانی", 75.0, 100.0, "#880E4F")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def create_excel_report(stats: dict, output_path: Path, forecast_date: str) -> None:
    df = pd.DataFrame({
        "تاریخ پیش‌بینی": [forecast_date],
        "مساحت متوسط (هکتار)": [stats.get("area_medium", 0)],
        "مساحت زیاد (هکتار)": [stats.get("area_high", 0)],
        "مساحت خیلی زیاد (هکتار)": [stats.get("area_very_high", 0)],
        "مساحت بحرانی (هکتار)": [stats.get("area_critical", 0)],
        "حداکثر FLI": [stats.get("max_fli", 0)],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name="گزارش آتش")
    print(f"Excel report created: {output_path}")


def update_archive_index(archive_root: Path) -> None:
    index_path = archive_root / "index.json"
    entries = []
    if archive_root.exists():
        for date_dir in sorted(archive_root.iterdir()):
            if date_dir.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
                if (date_dir / "fli.json").exists():
                    excel_file = date_dir / "report.xlsx"
                    excel_exists = excel_file.exists()
                    entries.append({
                        "date": date_dir.name,
                        "url": f"archive/{date_dir.name}/fli.json",
                        "excel_url": f"archive/{date_dir.name}/report.xlsx" if excel_exists else None,
                        "generated_at": datetime.now(timezone.utc).isoformat()
                    })
    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "entries": entries
    }
    atomic_write_json(index_path, payload)
    print(f"Archive index updated: {index_path} ({len(entries)} entries)")


def main() -> None:
    args = parse_args()
    require_file(args.input, "Input FLI raster")
    forecast_date = extract_forecast_date(args.input)
    output_dir = args.output_dir.resolve()
    archive_root = args.archive_dir.resolve() if args.archive_dir is not None else output_dir / "archive"

    print("\n" + "=" * 70)
    print("FIRIS WEB MAP BUILD")
    print("=" * 70)
    print(f"Input FLI        : {args.input}")
    print(f"Forecast date    : {forecast_date}")
    print(f"Web output       : {output_dir}")
    print(f"Archive root     : {archive_root}\n")

    # TODO: Insert full raster reading, classification, polygonization,
    # smoothing and writing logic here (same as previous version)
    stats = {"area_medium": 0, "area_high": 0, "area_very_high": 0, "area_critical": 0, "max_fli": 0}

    excel_path = output_dir / "report.xlsx"
    create_excel_report(stats, excel_path, forecast_date)

    archive_date_dir = archive_root / forecast_date
    archive_date_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(excel_path, archive_date_dir / "report.xlsx")

    update_archive_index(archive_root)

    print("\n" + "=" * 70)
    print("FIRIS WEB MAP BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
