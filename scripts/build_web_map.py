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
    ("متوسط", 0.0, 25.0, "#C7A900"),
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

# حداقل مساحت قطعات کوچک در GeoJSON وب
MIN_VECTOR_AREA = 0.000002


# تعداد تکرار Chaikin
CHAIKIN_ITERATIONS = 2


# میزان نرم‌کنندگی Chaikin
CHAIKIN_RATIO = 0.25


# میزان ساده‌سازی نهایی
# مقدار کم = حفظ جزئیات بیشتر
SIMPLIFY_TOLERANCE = 0.00025


# حداقل تعداد نقاط حلقه
MIN_RING_POINTS = 4


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build FIRIS Web GIS products "
            "from a dated FLI GeoTIFF."
        )
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
        help="Optional archive root.",
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

DATE_PATTERN = re.compile(
    r"fli_fars_(\d{4}-\d{2}-\d{2})\.tif$",
    re.I,
)


def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def extract_forecast_date(
    path: Path,
) -> str:

    match = DATE_PATTERN.search(
        path.name
    )

    if match:

        return match.group(1)

    match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        path.name,
    )

    if match:

        return match.group(1)

    raise ValueError(
        "Could not determine forecast date "
        "from input filename. "
        f"Expected fli_fars_YYYY-MM-DD.tif: "
        f"{path.name}"
    )


def risk_code(
    value: float,
) -> int:

    for code, (
        _,
        minimum,
        maximum,
        _,
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
            "#777777",
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
            "#777777",
        )

    return (
        "بحرانی",
        75.0,
        100.0,
        "#880E4F",
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
                    round(
                        number,
                        4,
                    )
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

            os.unlink(
                temp_name
            )


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

        data = src.read(
            1
        ).astype(
            np.float32,
            copy=False,
        )

        nodata = src.nodata

        valid = np.isfinite(
            data
        )

        if nodata is not None:

            try:

                nodata_float = float(
                    nodata
                )

                if math.isnan(
                    nodata_float
                ):

                    valid &= ~np.isnan(
                        data
                    )

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

            "crs":
                str(src.crs),

            "width":
                int(src.width),

            "height":
                int(src.height),

            "transform": [

                float(src.transform.a),
                float(src.transform.b),
                float(src.transform.c),
                float(src.transform.d),
                float(src.transform.e),
                float(src.transform.f),

            ],

            "bounds": {

                "west":
                    float(bounds.left),

                "south":
                    float(bounds.bottom),

                "east":
                    float(bounds.right),

                "north":
                    float(bounds.top),

            },

            "resolution": {

                "x":
                    float(abs(src.res[0])),

                "y":
                    float(abs(src.res[1])),

            },

            "nodata": (

                None
                if src.nodata is None
                else json_safe_number(
                    src.nodata
                )

            ),
        }

        stats = {

            "count":
                int(np.sum(valid)),

            "min": (

                None
                if not np.any(valid)
                else round(
                    float(
                        np.min(
                            clean[valid]
                        )
                    ),
                    6,
                )

            ),

            "max": (

                None
                if not np.any(valid)
                else round(
                    float(
                        np.max(
                            clean[valid]
                        )
                    ),
                    6,
                )

            ),

            "mean": (

                None
                if not np.any(valid)
                else round(
                    float(
                        np.mean(
                            clean[valid]
                        )
                    ),
                    6,
                )

            ),
        }

    if stats["count"] == 0:

        raise ValueError(
            "FLI raster contains no valid "
            "0-100 pixels."
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

        "forecast_date":
            forecast_date,

        "target_date":
            forecast_date,

        "crs":
            reference["crs"],

        "rows":
            reference["height"],

        "cols":
            reference["width"],

        "bounds":
            reference["bounds"],

        "resolution":
            reference["resolution"],

        "values":
            array_to_json_values(
                array
            ),
    }


# ============================================================
# METADATA
# ============================================================

def build_metadata_json(
    input_path: Path,
    reference: dict[str, Any],
    stats: dict[str, Any],
    forecast_date: str,
) -> dict[str, Any]:

    return {

        "project":
            "FIRIS - Fars Integrated Fire Information System",

        "forecast_date":
            forecast_date,

        "target_date":
            forecast_date,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source_file":
            input_path.name,

        "source_path":
            str(input_path),

        "crs":
            reference["crs"],

        "width":
            reference["width"],

        "height":
            reference["height"],

        "bounds":
            reference["bounds"],

        "resolution":
            reference["resolution"],

        "nodata":
            reference["nodata"],

        "statistics":
            stats,

        "risk_classes": [

            {

                "label":
                    label,

                "minimum":
                    minimum,

                "maximum":
                    min(
                        maximum,
                        100.0,
                    ),

                "color":
                    color,

            }

            for (
                label,
                minimum,
                maximum,
                color,
            ) in RISK_CLASSES
        ],

        "grid": {

            "row_order":
                "north_to_south",

            "column_order":
                "west_to_east",

            "coordinate_reference":
                "EPSG:4326",
        },

        "web_products": {

            "latest_metadata":
                "fli_latest.json",

            "latest_grid":
                "fli_latest_grid.json",

            "latest_polygons":
                "fli_polygons.geojson",

            "archive_directory":
                f"archive/{forecast_date}",
        },

        "polygon_processing": {

            "method":
                "Dissolve + fragment removal + "
                "Chaikin smoothing + "
                "topology-preserving simplify",

            "minimum_vector_area":
                MIN_VECTOR_AREA,

            "chaikin_iterations":
                CHAIKIN_ITERATIONS,

            "chaikin_ratio":
                CHAIKIN_RATIO,

            "simplify_tolerance":
                SIMPLIFY_TOLERANCE,

            "source_fli_unchanged":
                True,
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

    finite = np.isfinite(
        array
    )

    for code, (
        _,
        minimum,
        maximum,
        _,
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
# VECTOR CLEANING
# ============================================================

def extract_polygon_parts(
    geometry,
) -> list[Polygon]:

    if geometry.is_empty:

        return []

    if isinstance(
        geometry,
        Polygon,
    ):

        return [geometry]

    if isinstance(
        geometry,
        MultiPolygon,
    ):

        return list(
            geometry.geoms
        )

    if isinstance(
        geometry,
        GeometryCollection,
    ):

        polygons = []

        for item in geometry.geoms:

            if isinstance(
                item,
                Polygon,
            ):

                polygons.append(
                    item
                )

            elif isinstance(
                item,
                MultiPolygon,
            ):

                polygons.extend(
                    list(
                        item.geoms
                    )
                )

        return polygons

    return []


def remove_small_fragments(
    geometry,
    minimum_area: float = MIN_VECTOR_AREA,
):

    parts = extract_polygon_parts(
        geometry
    )

    if not parts:

        return geometry

    kept = [

        polygon

        for polygon in parts

        if polygon.area >= minimum_area

    ]

    if not kept:

        largest = max(
            parts,
            key=lambda item: item.area,
        )

        return largest

    if len(kept) == 1:

        return kept[0]

    return MultiPolygon(
        kept
    )


# ============================================================
# CHAIKIN SMOOTHING
# ============================================================

def chaikin_ring(
    coordinates,
    iterations: int = CHAIKIN_ITERATIONS,
    ratio: float = CHAIKIN_RATIO,
):

    if len(coordinates) < MIN_RING_POINTS:

        return list(
            coordinates
        )

    points = [

        (
            float(x),
            float(y),
        )

        for x, y, *rest
        in coordinates

    ]

    if points[0] == points[-1]:

        points = points[:-1]

    if len(points) < 3:

        return list(
            coordinates
        )

    for _ in range(
        iterations
    ):

        new_points = []

        count = len(
            points
        )

        for i in range(
            count
        ):

            p0 = points[i]

            p1 = points[
                (i + 1) % count
            ]

            q = (

                (1.0 - ratio) * p0[0]
                + ratio * p1[0],

                (1.0 - ratio) * p0[1]
                + ratio * p1[1],

            )

            r = (

                ratio * p0[0]
                + (1.0 - ratio) * p1[0],

                ratio * p0[1]
                + (1.0 - ratio) * p1[1],

            )

            new_points.append(
                q
            )

            new_points.append(
                r
            )

        points = new_points

    points.append(
        points[0]
    )

    return points


def smooth_polygon(
    geometry,
    iterations: int = CHAIKIN_ITERATIONS,
    ratio: float = CHAIKIN_RATIO,
):

    if geometry.is_empty:

        return geometry

    if isinstance(
        geometry,
        Polygon,
    ):

        exterior = chaikin_ring(

            list(
                geometry.exterior.coords
            ),

            iterations=iterations,

            ratio=ratio,

        )

        holes = []

        for interior in geometry.interiors:

            hole = chaikin_ring(

                list(
                    interior.coords
                ),

                iterations=iterations,

                ratio=ratio,

            )

            if len(hole) >= MIN_RING_POINTS:

                holes.append(
                    hole
                )

        try:

            result = Polygon(
                exterior,
                holes,
            )

        except Exception:

            return geometry

        if result.is_empty:

            return geometry

        if not result.is_valid:

            repaired = result.buffer(
                0
            )

            if not repaired.is_empty:

                result = repaired

        return result

    if isinstance(
        geometry,
        MultiPolygon,
    ):

        polygons = []

        for polygon in geometry.geoms:

            smoothed = smooth_polygon(

                polygon,

                iterations=iterations,

                ratio=ratio,

            )

            if smoothed.is_empty:

                continue

            if isinstance(
                smoothed,
                Polygon,
            ):

                polygons.append(
                    smoothed
                )

            elif isinstance(
                smoothed,
                MultiPolygon,
            ):

                polygons.extend(
                    list(
                        smoothed.geoms
                    )
                )

        if not polygons:

            return geometry

        result = MultiPolygon(
            polygons
        )

        if not result.is_valid:

            repaired = result.buffer(
                0
            )

            if not repaired.is_empty:

                result = repaired

        return result

    return geometry


# ============================================================
# PROFESSIONAL WEB GENERALIZATION
# ============================================================

def professionalize_geometry(
    geometry,
):

    if geometry.is_empty:

        return geometry

    # --------------------------------------------------------
    # STEP 1
    # Repair original geometry
    # --------------------------------------------------------

    if not geometry.is_valid:

        geometry = geometry.buffer(
            0
        )

    if geometry.is_empty:

        return geometry

    # --------------------------------------------------------
    # STEP 2
    # Remove very small fragments
    # --------------------------------------------------------

    geometry = remove_small_fragments(
        geometry,
        MIN_VECTOR_AREA,
    )

    if geometry.is_empty:

        return geometry

    # --------------------------------------------------------
    # STEP 3
    # Chaikin smoothing
    # --------------------------------------------------------

    geometry = smooth_polygon(

        geometry,

        iterations=CHAIKIN_ITERATIONS,

        ratio=CHAIKIN_RATIO,

    )

    if geometry.is_empty:

        return geometry

    # --------------------------------------------------------
    # STEP 4
    # Controlled simplification
    # --------------------------------------------------------

    geometry = geometry.simplify(

        SIMPLIFY_TOLERANCE,

        preserve_topology=True,

    )

    if geometry.is_empty:

        return geometry

    # --------------------------------------------------------
    # STEP 5
    # Final geometry repair
    # --------------------------------------------------------

    if not geometry.is_valid:

        repaired = geometry.buffer(
            0
        )

        if not repaired.is_empty:

            geometry = repaired

    return geometry


# ============================================================
# POLYGONIZE
# ============================================================

def polygonize_classes(
    classified: np.ndarray,
    transform,
) -> dict[int, list[Any]]:

    groups: dict[
        int,
        list[Any],
    ] = {

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

        code = int(
            value
        )

        if code <= 0:

            continue

        geom = shape(
            geometry
        )

        if geom.is_empty:

            continue

        if not geom.is_valid:

            geom = geom.buffer(
                0
            )

        if geom.is_empty:

            continue

        groups.setdefault(
            code,
            [],
        ).append(
            geom
        )

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

    features: list[
        dict[str, Any]
    ] = []

    for code, geometries in grouped.items():

        if not geometries:

            continue

        # ----------------------------------------------------
        # DISSOLVE
        # ----------------------------------------------------

        dissolved = unary_union(
            geometries
        )

        if dissolved.is_empty:

            continue

        if not dissolved.is_valid:

            dissolved = dissolved.buffer(
                0
            )

        if dissolved.is_empty:

            continue

        original_area = float(
            dissolved.area
        )

        # ----------------------------------------------------
        # PROFESSIONAL GENERALIZATION
        # ----------------------------------------------------

        smoothed = professionalize_geometry(
            dissolved
        )

        if smoothed.is_empty:

            continue

        final_area = float(
            smoothed.area
        )

        label, minimum, maximum, color = (
            RISK_CODE_TO_INFO[code]
        )

        features.append(

            {

                "type":
                    "Feature",

                "properties": {

                    "risk_code":
                        code,

                    "risk":
                        label,

                    "label":
                        label,

                    "color":
                        color,

                    "minimum":
                        minimum,

                    "maximum":
                        min(
                            maximum,
                            100.0,
                        ),

                    "forecast_date":
                        metadata[
                            "forecast_date"
                        ],

                    "geometry_processing":
                        "Dissolve + "
                        "fragment removal + "
                        "Chaikin + "
                        "topology-preserving simplify",

                    "original_vector_area":
                        original_area,

                    "final_vector_area":
                        final_area,

                },

                "geometry":
                    mapping(
                        smoothed
                    ),

            }

        )

    features.sort(

        key=lambda item: int(

            item["properties"][

                "risk_code"

            ]

        )

    )

    return {

        "type":
            "FeatureCollection",

        "name":
            "FIRIS_FLI_Risk_Zones",

        "crs": {

            "type":
                "name",

            "properties": {

                "name":
                    "EPSG:4326",

            },

        },

        "properties": {

            "forecast_date":
                metadata[
                    "forecast_date"
                ],

            "source_file":
                metadata[
                    "source_file"
                ],

            "generated_at_utc":
                metadata[
                    "generated_at_utc"
                ],

            "classification":
                "FLI risk classes",

            "geometry_processing":
                "Professional web "
                "polygon generalization",

            "smoothing":
                "Chaikin",

            "chaikin_iterations":
                CHAIKIN_ITERATIONS,

            "chaikin_ratio":
                CHAIKIN_RATIO,

            "simplify_tolerance":
                SIMPLIFY_TOLERANCE,

            "minimum_vector_area":
                MIN_VECTOR_AREA,

            "source_raster_unchanged":
                True,

        },

        "features":
            features,

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

        metadata = json.load(
            handle
        )

    actual_date = (

        metadata.get(
            "forecast_date"
        )

        or metadata.get(
            "target_date"
        )

    )

    actual_source = metadata.get(
        "source_file",
        "",
    )

    if str(actual_date) != expected_date:

        raise RuntimeError(

            "Generated latest metadata "
            "has the wrong forecast date: "
            f"expected {expected_date}, "
            f"got {actual_date}"

        )

    if str(actual_source) != expected_source:

        raise RuntimeError(

            "Generated latest metadata "
            "has the wrong source file: "
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

        grid = json.load(
            handle
        )

    actual_date = (

        grid.get(
            "forecast_date"
        )

        or grid.get(
            "target_date"
        )

    )

    if str(actual_date) != expected_date:

        raise RuntimeError(

            "Generated latest grid "
            "has the wrong forecast date: "
            f"expected {expected_date}, "
            f"got {actual_date}"

        )

    if int(
        grid.get(
            "rows",
            -1,
        )
    ) != expected_rows:

        raise RuntimeError(
            "Generated grid row count is incorrect."
        )

    if int(
        grid.get(
            "cols",
            -1,
        )
    ) != expected_cols:

        raise RuntimeError(
            "Generated grid column count is incorrect."
        )

    values = grid.get(
        "values"
    )

    if not isinstance(
        values,
        list,
    ):

        raise RuntimeError(
            "Generated grid values are not a list."
        )

    if len(values) != expected_rows:

        raise RuntimeError(

            "Generated grid row count "
            "does not match values length."

        )

    sample_rows = values[
        : min(
            10,
            len(values),
        )
    ]

    for row in sample_rows:

        if not isinstance(
            row,
            list,
        ):

            raise RuntimeError(
                "Generated grid contains an invalid row."
            )

        if len(row) != expected_cols:

            raise RuntimeError(

                "Generated grid column count "
                "does not match values width."

            )


def validate_polygons(
    polygon_path: Path,
    expected_date: str,
) -> None:

    with polygon_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        geojson = json.load(
            handle
        )

    if geojson.get(
        "type"
    ) != "FeatureCollection":

        raise RuntimeError(

            "Generated FLI polygons "
            "are not a FeatureCollection."

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

            "Generated polygon forecast "
            "date is incorrect: "
            f"expected {expected_date}, "
            f"got {actual_date}"

        )

    features = geojson.get(
        "features"
    )

    if not isinstance(
        features,
        list,
    ):

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
) -> tuple[
    Path,
    Path,
    Path,
]:

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
        destination
        / "fli_latest.json"
    )

    grid_path = (
        destination
        / "fli_latest_grid.json"
    )

    polygon_path = (
        destination
        / "fli_polygons.geojson"
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

    values = reference[
        "transform"
    ]

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

        archive_root
        / forecast_date

    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(

        latest_dir
        / "fli_latest.json",

        archive_dir
        / "fli.json",

    )

    shutil.copy2(

        latest_dir
        / "fli_latest_grid.json",

        archive_dir
        / "fli_grid.json",

    )

    shutil.copy2(

        latest_dir
        / "fli_polygons.geojson",

        archive_dir
        / "fli_polygons.geojson",

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

            archive_dir
            / name

        ).is_file()

    ]

    if missing:

        raise RuntimeError(

            "Archive is incomplete. Missing: "
            + ", ".join(missing)

        )

    with (

        archive_dir
        / "fli.json"

    ).open(

        "r",

        encoding="utf-8",

    ) as handle:

        metadata = json.load(
            handle
        )

    actual_date = (

        metadata.get(
            "forecast_date"
        )

        or metadata.get(
            "target_date"
        )

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

    output_dir = (
        args.output_dir.resolve()
    )

    archive_root = (

        args.archive_dir.resolve()

        if args.archive_dir is not None

        else output_dir / "archive"

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

    print()
    print("WEB POLYGON GENERALIZATION")
    print("---------------------------")

    print(
        "Dissolve         : ENABLED"
    )

    print(
        "Small fragments  : REMOVED"
    )

    print(
        "Chaikin          : ENABLED"
    )

    print(
        f"Chaikin passes   : {CHAIKIN_ITERATIONS}"
    )

    print(
        f"Chaikin ratio    : {CHAIKIN_RATIO}"
    )

    print(
        f"Simplify         : {SIMPLIFY_TOLERANCE}"
    )

    print(
        f"Min area         : {MIN_VECTOR_AREA}"
    )

    print(
        "Source FLI       : UNCHANGED"
    )

    # --------------------------------------------------------
    # READ INPUT
    # --------------------------------------------------------

    array, reference, stats = read_fli(
        args.input
    )

    print()
    print("INPUT FLI")
    print("---------")

    print(
        f"CRS              : {reference['crs']}"
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
        f"Minimum          : {stats['min']}"
    )

    print(
        f"Maximum          : {stats['max']}"
    )

    print(
        f"Mean             : {stats['mean']}"
    )

    # --------------------------------------------------------
    # TEMPORARY STAGING
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
            temp_root
            / "latest"
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
        # BUILD LATEST SET
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
        # ARCHIVE
        # ----------------------------------------------------

        shutil.copy2(

            metadata_path,

            temp_archive
            / "fli.json",

        )

        shutil.copy2(

            grid_path,

            temp_archive
            / "fli_grid.json",

        )

        shutil.copy2(

            polygon_path,

            temp_archive
            / "fli_polygons.geojson",

        )

        # ----------------------------------------------------
        # VALIDATION
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

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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

            archive_root
            / forecast_date

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
        output_dir
        / "fli_latest.json"
    )

    live_grid = (
        output_dir
        / "fli_latest_grid.json"
    )

    live_polygons = (
        output_dir
        / "fli_polygons.geojson"
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
        f"Forecast date    : {forecast_date}"
    )

    print(
        f"Latest metadata  : {live_metadata}"
    )

    print(
        f"Latest grid      : {live_grid}"
    )

    print(
        f"Latest polygons  : {live_polygons}"
    )

    print(
        f"Archive          : "
        f"{archive_root / forecast_date}"
    )

    print()

    print(
        "✓ latest products belong to the same dated FLI input."
    )

    print(
        "✓ archive products belong to the same forecast date."
    )

    print(
        "✓ no older forecast can silently become latest."
    )

    print(
        "✓ web polygon geometry professionally generalized."
    )

    print(
        "✓ original FLI raster remains unchanged."
    )


if __name__ == "__main__":
    main()
