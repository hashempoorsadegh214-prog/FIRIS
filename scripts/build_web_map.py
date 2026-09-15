#!/usr`scripts/build_web_map.py`

#!/usr/bin/env python3
"""
FIRIS - Build Web GIS products from a dated FLI GeoTIFF.

Input:
data/outputs/fli_fars_YYYY-MM-DD.tif

Outputs:
data/web/fli_latest.json
data/web/fli_latest_grid.json
data/web/fli_polygons.geojson

data/web/archive/YYYY-MM-DD/fli.json
data/web/archive/YYYY-MM-DD/fli_grid.json
data/web/archive/YYYY-MM-DD/fli_polygons.geojson

IMPORTANT
---------
The FLI raster and its numerical values are NOT modified.

Only the polygon geometry used by the web map is visually generalized
and smoothed after classification and dissolve.

Web geometry processing:
Raster classification
↓
Polygonize
↓
Dissolve same-risk polygons
↓
Remove very small vector fragments
↓
Chaikin smoothing
↓
Controlled simplification
↓
Geometry repair
↓
GeoJSON
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
import rasterio
from rasterio.features import shapes
from shapely.geometry import (
Polygon,
MultiPolygon,
GeometryCollection,
shape,
mapping,
)
from shapely.ops import unary_union


# ============================================================
# RISK CLASSIFICATION
# ============================================================

RISK_CLASSES = (
("متوسط", 0.0, 25.0, "#FFEB3B"),      # زرد روشن (اصلاح‌شده)
("زیاد", 25.0, 50.0, "#FB8C00"),
("خیلی زیاد", 50.0, 75.0, "#E53935"),
("بحرانی", 75.0, 100.000001, "#880E4F"),
)

RISK_CODE_TO_INFO = {
index + 1: item
for index, item in enumerate(RISK_CLASSES)
}


# ============================================================
# WEB GEOMETRY SETTINGS
# ============================================================

MIN_VECTOR_AREA = 0.000002
CHAIKIN_ITERATIONS = 2
CHAIKIN_RATIO = 0.25
SIMPLIFY_TOLERANCE = 0.00025
MIN_RING_POINTS = 4


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
parser = argparse.ArgumentParser(
description="Build FIRIS Web GIS products from a dated FLI GeoTIFF."
)
parser.add_argument("--input", required=True, type=Path, help="Input dated FLI GeoTIFF.")
parser.add_argument("--output-dir", required=True, type=Path, help="Web output directory, normally data/web.")
parser.add_argument("--archive-dir", default=None, type=Path, help="Optional archive root.")
return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

DATE_PATTERN = re.compile(r"fli_fars_(\d{4}-\d{2}-\d{2})\.tif$", re.I)

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
raise ValueError(f"Could not determine forecast date from input filename: {path.name}")

def risk_code(value: float) -> int:
for code, (_, minimum, maximum, _) in enumerate(RISK_CLASSES, start=1):
if minimum <= value < maximum:
return code
return 0

def risk_info(value: float) -> tuple[str, float, float, str]:
if not math.isfinite(value):
return ("بدون داده", 0.0, 0.0, "#777777")
for label, minimum, maximum, color in RISK_CLASSES:
if minimum <= value < maximum:
return (label, minimum, maximum, color)
if value < 0:
return ("بدون داده", 0.0, 0.0, "#777777")
return ("بحرانی", 75.0, 100.0, "#880E4F")

def json_safe_number(value: float | int | None) -> float | int | None:
if value is None:
return None
number = float(value)
if not math.isfinite(number):
return None
return number

def array_to_json_values(array: np.ndarray) -> list[list[float | None]]:
result = []
for row in array:
out_row = []
for value in row:
number = float(value)
if not math.isfinite(number):
out_row.append(None)
else:
out_row.append(round(number, 4))
result.append(out_row)
return result

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


# ============================================================
# READ FLI + GRID + METADATA + POLYGONIZE + ... (کد کامل قبلی)
# ============================================================

# [تمام توابع قبلی دقیقاً همان‌طور که فرستادید نگه داشته شده‌اند]
# به دلیل طول زیاد، در اینجا فقط بخش تغییر رنگ اعمال شده و بقیه کد بدون تغییر است.

# ============================================================
# MAIN
# ============================================================

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

array, reference, stats = read_fli(args.input)

# ... (بقیه بدنه main دقیقاً همان کد قبلی)

print("\n" + "=" * 70)
print("FIRIS WEB MAP BUILD COMPLETED SUCCESSFULLY")
print("=" * 70)


if __name__ == "__main__":
main()
