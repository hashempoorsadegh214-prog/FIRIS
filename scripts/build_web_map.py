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
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


# ============================================================
# RISK CLASSIFICATION
# ============================================================

RISK_CLASSES = (
    ("کم", 0.0, 20.0, "#2e7d32"),
    ("متوسط", 20.0, 40.0, "#c7a900"),
    ("زیاد", 40.0, 60.0, "#fb8c00"),
    ("خیلی زیاد", 60.0, 80.0, "#e53935"),
    ("بحرانی", 80.0, 100.000001, "#880e4f"),
)

RISK_CODE_TO_INFO = {
    index + 1: item
    for index, item in enumerate(RISK_CLASSES)
}


# ============================================================
# WEB POLYGON CLEANUP
# ============================================================

# فقط روی خروجی fli_polygons.geojson اثر دارد.
# Raster اصلی و Grid اصلاً تغییر نمی‌کنند.
MIN_POLYGON_CELLS = 9


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FIRIS Web GIS products from a dated FLI GeoTIFF."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input dated FLI GeoTIFF.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Web output directory, normally data/web.",
    )

    parser.add_argument(
        "--archive-dir",
        default=None,
        type=Path,
        help="Optional archive root. Defaults to <output-dir>/archive.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

DATE_PATTERN = re.compile(
    r"fli_fars_(\d{4}-\d{2}-\d{2})\.tif$",
    re.I,
)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def extract_forecast_date(path: Path) -> str:
    match = DATE_PATTERN.search(path.name)

    if match:
        return match.group(1)

    match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        path.name,
    )

    if match:
        return match.group(1)

    raise ValueError(
        "Could not determine forecast date from input filename. "
        f"Expected a name like fli_fars_YYYY-MM-DD.tif: {path.name}"
    )


def risk_code(value: float) -> int:
    for code, (
        _label,
        minimum,
        maximum,
        _color,
    ) in enumerate(
        RISK_CLASSES,
        start=1,
    ):
        if minimum <= value < maximum:
            return code

    return 0


def risk_info(
    value: float,
) -> tuple[str, float, float, str]:
    if not math.isfinite(value):
        return (
            "بدون داده",
            0.0,
            0.0,
            "#777",
        )

    for (
        label,
        minimum,
        maximum,
        color,
    ) in RISK_CLASSES:
        if minimum <= value < maximum:
            return (
                label,
                minimum,
                maximum,
                color,
            )

    if value < 0:
        return (
            "بدون داده",
            0.0,
            0.0,
            "#777",
        )

    return (
        "بحرانی",
        80.0,
        100.0,
        "#880e4f",
    )


def json_safe_number(
    value: float | int | None,
) -> float | int | None:
    if value is None:
        return None

    number = float(value)

    if not math.isfinite(number):
        return None

    return number


def array_to_json_values(
    array: np.ndarray,
) -> list[list[float | None]]:
    result: list[list[float | None]] = []

    for row in array:
        out_row: list[float | None] = []

        for value in row:
            number = float(value)

            if not math.isfinite(number):
                out_row.append(None)
            else:
                out_row.append(
                    round(number, 4)
                )

        result.append(out_row)

    return result


def atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

            handle.write("\n")

        os.replace(
            temp_name,
            path,
        )

    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


# ============================================================
# READ FLI
# ============================================================

def read_fli(
    path: Path,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    with rasterio.open(path) as src:

        if src.count < 1:
            raise ValueError(
                "FLI raster has no bands."
            )

        if src.crs is None:
            raise ValueError(
                "FLI raster has no CRS."
            )

        if src.crs.to_epsg() != 4326:
            raise ValueError(
                "FLI raster must use EPSG:4326. "
                f"Found: {src.crs}"
            )

        data = src.read(1).astype(
            np.float32,
            copy=False,
        )

        nodata = src.nodata

        valid = np.isfinite(data)

        if nodata is not None:
            try:
                nodata_float = float(nodata)

                if math.isnan(nodata_float):
                    valid &= ~np.isnan(data)
                else:
                    valid &= ~np.isclose(
                        data,
                        nodata_float,
                        rtol=0.0,
                        atol=1e-8,
                    )

            except (
                TypeError,
                ValueError,
            ):
                pass

        valid &= data >= 0.0
        valid &= data <= 100.0

        clean = np.full_like(
            data,
            np.nan,
            dtype=np.float32,
        )

        clean[valid] = data[valid]

        bounds = src.bounds

        reference = {
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),

            "transform": [
                float(src.transform.a),
                float(src.transform.b),
                float(src.transform.c),
                float(src.transform.d),
                float(src.transform.e),
                float(src.transform.f),
            ],

            "bounds": {
                "west": float(bounds.left),
                "south": float(bounds.bottom),
                "east": float(bounds.right),
                "north": float(bounds.top),
            },

            "resolution": {
                "x": float(abs(src.res[0])),
                "y": float(abs(src.res[1])),
            },

            "nodata": (
                None
                if src.nodata is None
                else json_safe_number(src.nodata)
            ),
        }

        stats = {
            "count": int(np.sum(valid)),

            "min": (
                None
                if not np.any(valid)
                else round(
                    float(np.min(clean[valid])),
                    6,
                )
            ),

            "max": (
                None
                if not np.any(valid)
                else round(
                    float(np.max(clean[valid])),
                    6,
                )
            ),

            "mean": (
                None
                if not np.any(valid)
                else round(
                    float(np.mean(clean[valid])),
                    6,
                )
            ),
        }

    if stats["count"] == 0:
        raise ValueError(
            "FLI raster contains no valid 0-100 pixels."
        )

    return (
        clean,
        reference,
        stats,
    )


# ============================================================
# GRID JSON
# ============================================================

def build_grid_json(
    array: np.ndarray,
    reference: dict[str, Any],
    forecast_date: str,
) -> dict[str, Any]:
    return {
        "forecast_date": forecast_date,
        "target_date": forecast_date,
        "crs": reference["crs"],
        "rows": reference["height"],
        "cols": reference["width"],
        "bounds": reference["bounds"],
        "resolution": reference["resolution"],
        "values": array_to_json_values(array),
    }


# ============================================================
# METADATA JSON
# ============================================================

def build_metadata_json(
    input_path: Path,
    reference: dict[str, Any],
    stats: dict[str, Any],
    forecast_date: str,
) -> dict[str, Any]:
    return {
        "project": (
            "FIRIS - Fars Integrated Fire Information System"
        ),

        "forecast_date": forecast_date,
        "target_date": forecast_date,

        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),

        "source_file": input_path.name,
        "source_path": str(input_path),

        "crs": reference["crs"],
        "width": reference["width"],
        "height": reference["height"],
        "bounds": reference["bounds"],
        "resolution": reference["resolution"],
        "nodata": reference["nodata"],

        "statistics": stats,

        "risk_classes": [
            {
                "label": label,
                "minimum": minimum,
                "maximum": min(maximum, 100.0),
                "color": color,
            }

            for (
                label,
                minimum,
                maximum,
                color,
            ) in RISK_CLASSES
        ],

        "grid": {
            "row_order": "north_to_south",
            "column_order": "west_to_east",
            "coordinate_reference": "EPSG:4326",
        },

        "web_products": {
            "latest_metadata": "fli_latest.json",
            "latest_grid": "fli_latest_grid.json",
            "latest_polygons": "fli_polygons.geojson",
            "archive_directory": (
                f"archive/{forecast_date}"
            ),
        },
    }


# ============================================================
# CLASSIFIED RASTER
# ============================================================

def build_classified_raster(
    array: np.ndarray,
) -> np.ndarray:
    classified = np.zeros(
        array.shape,
        dtype=np.uint8,
    )

    finite = np.isfinite(array)

    for code, (
        _label,
        minimum,
        maximum,
        _color,
    ) in enumerate(
        RISK_CLASSES,
        start=1,
    ):
        mask = (
            finite
            & (array >= minimum)
            & (array < maximum)
        )

        classified[mask] = code

    return classified


# ============================================================
# POLYGONIZE
# ============================================================

def polygonize_classes(
    classified: np.ndarray,
    transform,
) -> dict[int, list[Any]]:
    """
    Polygonize risk classes.
    Class 0 / NoData is omitted.
    """

    groups: dict[int, list[Any]] = {
        code: []
        for code in RISK_CODE_TO_INFO
    }

    mask = classified > 0

    for geometry, value in shapes(
        classified,
        mask=mask,
        transform=transform,
        connectivity=4,
    ):
        code = int(value)

        if code <= 0:
            continue

        geom = shape(geometry)

        if geom.is_empty:
            continue

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            continue

        groups.setdefault(
            code,
            [],
        ).append(geom)

    return groups


# ============================================================
# FEATURE COLLECTION
# ============================================================

def make_feature_collection(
    classified: np.ndarray,
    transform,
    metadata: dict[str, Any],
) -> dict[str, Any]:

    grouped = polygonize_classes(
        classified,
        transform,
    )

    features: list[dict[str, Any]] = []

    pixel_area = abs(
        float(transform.a * transform.e)
        - float(transform.b * transform.d)
    )

    if pixel_area <= 0:
        raise ValueError(
            "Invalid raster transform: "
            "pixel area is not positive."
        )

    min_polygon_area = (
        pixel_area * MIN_POLYGON_CELLS
    )

    original_polygon_count = 0
    removed_small_polygons = 0
    remaining_polygon_count = 0

    for code, geometries in grouped.items():

        if not geometries:
            continue

        original_polygon_count += len(
            geometries
        )

        filtered_geometries = []

        for geom in geometries:

            if geom.is_empty:
                continue

            if not geom.is_valid:
                geom = geom.buffer(0)

            if geom.is_empty:
                continue

            # ------------------------------------------------
            # تنها فیلتر جدید:
            # لکه‌های کوچک‌تر از ۹ سلول حذف می‌شوند.
            #
            # خود FLI تغییر نمی‌کند.
            # ------------------------------------------------
            if geom.area < min_polygon_area:

                removed_small_polygons += 1
                continue

            filtered_geometries.append(
                geom
            )

        if not filtered_geometries:
            continue

        remaining_polygon_count += len(
            filtered_geometries
        )

        dissolved = unary_union(
            filtered_geometries
        )

        if dissolved.is_empty:
            continue

        if not dissolved.is_valid:
            dissolved = dissolved.buffer(0)

        if dissolved.is_empty:
            continue

        (
            label,
            minimum,
            maximum,
            color,
        ) = RISK_CODE_TO_INFO[code]

        features.append(
            {
                "type": "Feature",

                "properties": {
                    "risk_code": code,
                    "risk": label,
                    "label": label,
                    "color": color,
                    "minimum": minimum,
                    "maximum": min(
                        maximum,
                        100.0,
                    ),
                    "forecast_date": (
                        metadata["forecast_date"]
                    ),
                },

                "geometry": mapping(
                    dissolved
                ),
            }
        )

    features.sort(
        key=lambda item: int(
            item["properties"]["risk_code"]
        )
    )

    print()
    print("WEB POLYGON CLEANUP")
    print("-------------------")
    print(
        "Minimum polygon cells : "
        f"{MIN_POLYGON_CELLS}"
    )
    print(
        "Original polygons     : "
        f"{original_polygon_count}"
    )
    print(
        "Removed small polygons: "
        f"{removed_small_polygons}"
    )
    print(
        "Remaining polygons    : "
        f"{remaining_polygon_count}"
    )
    print(
        "Final class features  : "
        f"{len(features)}"
    )

    return {
        "type": "FeatureCollection",

        "name": "FIRIS_FLI_Risk_Zones",

        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326",
            },
        },

        "properties": {
            "forecast_date": (
                metadata["forecast_date"]
            ),
            "source_file": (
                metadata["source_file"]
            ),
            "generated_at_utc": (
                metadata["generated_at_utc"]
            ),
            "classification": (
                "FLI risk classes"
            ),
            "minimum_polygon_cells": (
                MIN_POLYGON_CELLS
            ),
        },

        "features": features,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_metadata(
    metadata_path: Path,
    expected_date: str,
    expected_source: str,
) -> None:

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    actual_date = (
        metadata.get("forecast_date")
        or metadata.get("target_date")
    )

    actual_source = metadata.get(
        "source_file",
        "",
    )

    if str(actual_date) != expected_date:
        raise RuntimeError(
            "Generated latest metadata has the wrong "
            "forecast date: "
            f"expected {expected_date}, "
            f"got {actual_date}"
        )

    if str(actual_source) != expected_source:
        raise RuntimeError(
            "Generated latest metadata has the wrong "
            "source file: "
            f"expected {expected_source}, "
            f"got {actual_source}"
        )


def validate_grid(
    grid_path: Path,
    expected_date: str,
    expected_rows: int,
    expected_cols: int,
) -> None:

    with grid_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        grid = json.load(handle)

    actual_date = (
        grid.get("forecast_date")
        or grid.get("target_date")
    )

    if str(actual_date) != expected_date:
        raise RuntimeError(
            "Generated latest grid has the wrong "
            "forecast date: "
            f"expected {expected_date}, "
            f"got {actual_date}"
        )

    if int(
        grid.get("rows", -1)
    ) != expected_rows:
        raise RuntimeError(
            "Generated grid row count is incorrect."
        )

    if int(
        grid.get("cols", -1)
    ) != expected_cols:
        raise RuntimeError(
            "Generated grid column count is incorrect."
        )

    values = grid.get("values")

    if not isinstance(values, list):
        raise RuntimeError(
            "Generated grid values are not a list."
        )

    if len(values) != expected_rows:
        raise RuntimeError(
            "Generated grid row count does not "
            "match values length."
        )

    for row in values[: min(10, len(values))]:

        if not isinstance(row, list):
            raise RuntimeError(
                "Generated grid contains an invalid row."
            )

        if len(row) != expected_cols:
            raise RuntimeError(
                "Generated grid column count does not "
                "match values width."
            )


def validate_polygons(
    polygon_path: Path,
    expected_date: str,
) -> None:

    with polygon_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        geojson = json.load(handle)

    if geojson.get(
        "type"
    ) != "FeatureCollection":
        raise RuntimeError(
            "Generated FLI polygons are not "
            "a FeatureCollection."
        )

    properties = geojson.get(
        "properties",
        {}
    )

    actual_date = properties.get(
        "forecast_date"
    )

    if str(actual_date) != expected_date:
        raise RuntimeError(
            "Generated polygon forecast date is incorrect: "
            f"expected {expected_date}, "
            f"got {actual_date}"
        )

    features = geojson.get(
        "features"
    )

    if not isinstance(features, list):
        raise RuntimeError(
            "Generated polygon features are invalid."
        )


# ============================================================
# WRITE PRODUCT SET
# ============================================================

def write_product_set(
    destination: Path,
    input_path: Path,
    array: np.ndarray,
    reference: dict[str, Any],
    stats: dict[str, Any],
    forecast_date: str,
) -> tuple[Path, Path, Path]:

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = build_metadata_json(
        input_path=input_path,
        reference=reference,
        stats=stats,
        forecast_date=forecast_date,
    )

    grid = build_grid_json(
        array=array,
        reference=reference,
        forecast_date=forecast_date,
    )

    classified = build_classified_raster(
        array
    )

    polygons = make_feature_collection(
        classified=classified,
        transform=_transform_from_reference(
            reference
        ),
        metadata=metadata,
    )

    metadata_path = (
        destination / "fli_latest.json"
    )

    grid_path = (
        destination / "fli_latest_grid.json"
    )

    polygon_path = (
        destination / "fli_polygons.geojson"
    )

    atomic_write_json(
        metadata_path,
        metadata,
    )

    atomic_write_json(
        grid_path,
        grid,
    )

    atomic_write_json(
        polygon_path,
        polygons,
    )

    return (
        metadata_path,
        grid_path,
        polygon_path,
    )


def _transform_from_reference(
    reference: dict[str, Any],
):
    from affine import Affine

    values = reference["transform"]

    return Affine(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
    )


# ============================================================
# ARCHIVE
# ============================================================

def archive_product_set(
    latest_dir: Path,
    archive_root: Path,
    forecast_date: str,
) -> Path:

    archive_dir = (
        archive_root / forecast_date
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        latest_dir / "fli_latest.json",
        archive_dir / "fli.json",
    )

    shutil.copy2(
        latest_dir / "fli_latest_grid.json",
        archive_dir / "fli_grid.json",
    )

    shutil.copy2(
        latest_dir / "fli_polygons.geojson",
        archive_dir / "fli_polygons.geojson",
    )

    return archive_dir


def validate_archive(
    archive_dir: Path,
    expected_date: str,
) -> None:

    required = {
        "fli.json",
        "fli_grid.json",
        "fli_polygons.geojson",
    }

    missing = [
        name
        for name in required
        if not (
            archive_dir / name
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Archive is incomplete. Missing: "
            + ", ".join(missing)
        )

    with (
        archive_dir / "fli.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    actual_date = (
        metadata.get("forecast_date")
        or metadata.get("target_date")
    )

    if str(actual_date) != expected_date:
        raise RuntimeError(
            "Archive metadata date is incorrect: "
            f"expected {expected_date}, "
            f"got {actual_date}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    require_file(
        args.input,
        "Input FLI raster",
    )

    forecast_date = extract_forecast_date(
        args.input
    )

    output_dir = args.output_dir.resolve()

    if args.archive_dir is not None:
        archive_root = (
            args.archive_dir.resolve()
        )
    else:
        archive_root = (
            output_dir / "archive"
        )

    print()
    print("=" * 70)
    print("FIRIS WEB MAP BUILD")
    print("=" * 70)
    print()

    print(
        f"Input FLI        : {args.input}"
    )

    print(
        f"Forecast date    : {forecast_date}"
    )

    print(
        f"Web output       : {output_dir}"
    )

    print(
        f"Archive root     : {archive_root}"
    )

    # --------------------------------------------------------
    # READ INPUT
    # --------------------------------------------------------

    (
        array,
        reference,
        stats,
    ) = read_fli(
        args.input
    )

    print()
    print("INPUT FLI")
    print("---------")

    print(
        f"CRS              : "
        f"{reference['crs']}"
    )

    print(
        f"Size             : "
        f"{reference['width']} x "
        f"{reference['height']}"
    )

    print(
        f"Resolution       : "
        f"{reference['resolution']['x']} x "
        f"{reference['resolution']['y']}"
    )

    print(
        f"Valid pixels     : "
        f"{stats['count']:,}"
    )

    print(
        f"Minimum          : "
        f"{stats['min']}"
    )

    print(
        f"Maximum          : "
        f"{stats['max']}"
    )

    print(
        f"Mean             : "
        f"{stats['mean']}"
    )

    # --------------------------------------------------------
    # STAGING
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_parent = output_dir.parent

    with tempfile.TemporaryDirectory(
        prefix=".firis-web-build-",
        dir=str(staging_parent),
    ) as temp_root_string:

        temp_root = Path(
            temp_root_string
        )

        temp_latest = (
            temp_root / "latest"
        )

        temp_archive = (
            temp_root
            / "archive"
            / forecast_date
        )

        temp_latest.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_archive.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # BUILD LATEST
        # ----------------------------------------------------

        (
            metadata_path,
            grid_path,
            polygon_path,
        ) = write_product_set(
            destination=temp_latest,
            input_path=args.input,
            array=array,
            reference=reference,
            stats=stats,
            forecast_date=forecast_date,
        )

        # ----------------------------------------------------
        # ARCHIVE SAME PRODUCT SET
        # ----------------------------------------------------

        shutil.copy2(
            metadata_path,
            temp_archive / "fli.json",
        )

        shutil.copy2(
            grid_path,
            temp_archive / "fli_grid.json",
        )

        shutil.copy2(
            polygon_path,
            temp_archive / "fli_polygons.geojson",
        )

        # ----------------------------------------------------
        # VALIDATE BEFORE LIVE OUTPUT
        # ----------------------------------------------------

        validate_metadata(
            metadata_path,
            expected_date=forecast_date,
            expected_source=args.input.name,
        )

        validate_grid(
            grid_path,
            expected_date=forecast_date,
            expected_rows=reference["height"],
            expected_cols=reference["width"],
        )

        validate_polygons(
            polygon_path,
            expected_date=forecast_date,
        )

        validate_archive(
            temp_archive,
            expected_date=forecast_date,
        )

        print()
        print(
            "✓ Staged Web GIS products validated."
        )
        print(
            "✓ Latest date validated."
        )
        print(
            "✓ Grid dimensions validated."
        )
        print(
            "✓ Polygon metadata validated."
        )
        print(
            "✓ Archive validated."
        )

        # ----------------------------------------------------
        # PUBLISH LATEST
        # ----------------------------------------------------

        for name in (
            "fli_latest.json",
            "fli_latest_grid.json",
            "fli_polygons.geojson",
        ):
            os.replace(
                temp_latest / name,
                output_dir / name,
            )

        # ----------------------------------------------------
        # PUBLISH ARCHIVE
        # ----------------------------------------------------

        live_archive = (
            archive_root / forecast_date
        )

        live_archive.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name in (
            "fli.json",
            "fli_grid.json",
            "fli_polygons.geojson",
        ):
            os.replace(
                temp_archive / name,
                live_archive / name,
            )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    live_metadata = (
        output_dir / "fli_latest.json"
    )

    live_grid = (
        output_dir / "fli_latest_grid.json"
    )

    live_polygons = (
        output_dir / "fli_polygons.geojson"
    )

    validate_metadata(
        live_metadata,
        expected_date=forecast_date,
        expected_source=args.input.name,
    )

    validate_grid(
        live_grid,
        expected_date=forecast_date,
        expected_rows=reference["height"],
        expected_cols=reference["width"],
    )

    validate_polygons(
        live_polygons,
        expected_date=forecast_date,
    )

    validate_archive(
        archive_root / forecast_date,
        expected_date=forecast_date,
    )

    print()
    print("=" * 70)
    print(
        "FIRIS WEB MAP BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)
    print()

    print(
        f"Forecast date    : "
        f"{forecast_date}"
    )

    print(
        f"Latest metadata  : "
        f"{live_metadata}"
    )

    print(
        f"Latest grid      : "
        f"{live_grid}"
    )

    print(
        f"Latest polygons  : "
        f"{live_polygons}"
    )

    print(
        f"Archive          : "
        f"{archive_root / forecast_date}"
    )

    print()
    print(
        "✓ latest products belong to "
        "the same dated FLI input."
    )
    print(
        "✓ archive products belong to "
        "the same forecast date."
    )
    print(
        "✓ no older forecast can silently "
        "become latest."
    )


if __name__ == "__main__":
    main()
