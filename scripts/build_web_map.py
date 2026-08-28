#!/usr/bin/env python3

"""
FIRIS - Web Map Builder
=======================

Build web-ready FIRIS outputs from the EXISTING FLI raster.

IMPORTANT
---------
This script does NOT recalculate FLI.

It only:
1. Reads the final FLI GeoTIFF.
2. Applies fars.geojson as the authoritative province mask.
3. Creates a classified GeoJSON layer for web display.
4. Keeps the original FLI values in a lightweight grid JSON.
5. Creates a PNG as an optional backup/preview.
6. Writes metadata.

The MAIN WEB DISPLAY layer is:

    data/web/fli_polygons.geojson

The PNG is NOT the authoritative spatial display layer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask, shapes
from rasterio.warp import transform_geom


# ============================================================
# LEGEND
# ============================================================

LEGEND = [
    {
        "min": 0,
        "max": 20,
        "label": "کم",
        "color": "#2e7d32"
    },
    {
        "min": 20,
        "max": 40,
        "label": "متوسط",
        "color": "#fdd835"
    },
    {
        "min": 40,
        "max": 60,
        "label": "زیاد",
        "color": "#fb8c00"
    },
    {
        "min": 60,
        "max": 80,
        "label": "خیلی زیاد",
        "color": "#e53935"
    },
    {
        "min": 80,
        "max": 100,
        "label": "بحرانی",
        "color": "#880e4f"
    }
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build FIRIS web map files"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--boundary",
        required=False,
        type=Path,
        default=Path("fars.geojson")
    )

    return parser.parse_args()


# ============================================================
# LOAD BOUNDARY
# ============================================================

def load_boundary(
    boundary_path: Path,
    target_crs
):

    if not boundary_path.is_file():
        raise FileNotFoundError(
            f"Boundary not found: {boundary_path}"
        )

    with boundary_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        geojson = json.load(file)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError(
            "Boundary must be a GeoJSON FeatureCollection."
        )

    features = geojson.get(
        "features",
        []
    )

    if not features:
        raise ValueError(
            "Boundary contains no features."
        )

    geometries = []

    for feature in features:

        geometry = feature.get(
            "geometry"
        )

        if geometry:
            geometries.append(
                geometry
            )

    if not geometries:
        raise ValueError(
            "Boundary contains no valid geometries."
        )

    source_crs = "EPSG:4326"

    crs_obj = geojson.get("crs")

    if isinstance(crs_obj, dict):

        props = crs_obj.get(
            "properties",
            {}
        )

        name = (
            props.get("name")
            or props.get("href")
        )

        if isinstance(name, str) and name.strip():
            source_crs = name.strip()

    if str(target_crs) != source_crs:

        geometries = [

            transform_geom(
                source_crs,
                target_crs,
                geom,
                precision=12
            )

            for geom in geometries
        ]

    return geometries


# ============================================================
# CLASSIFY FLI
# ============================================================

def classify_fli(
    values: np.ndarray
) -> np.ndarray:

    result = np.full(
        values.shape,
        -1,
        dtype=np.int8
    )

    valid = (
        np.isfinite(values)
        &
        (values >= 0)
        &
        (values <= 100)
    )

    result[
        valid & (values < 20)
    ] = 0

    result[
        valid
        & (values >= 20)
        & (values < 40)
    ] = 1

    result[
        valid
        & (values >= 40)
        & (values < 60)
    ] = 2

    result[
        valid
        & (values >= 60)
        & (values < 80)
    ] = 3

    result[
        valid & (values >= 80)
    ] = 4

    return result


# ============================================================
# CLASS INFORMATION
# ============================================================

def class_info(class_id: int):

    item = LEGEND[class_id]

    return {
        "class_id": class_id,
        "min": item["min"],
        "max": item["max"],
        "label": item["label"],
        "color": item["color"]
    }


# ============================================================
# CREATE CLASSIFIED GEOJSON
# ============================================================

def build_polygon_geojson(
    class_raster: np.ndarray,
    transform,
    fars_mask: np.ndarray
):

    features = []

    # --------------------------------------------------------
    # Only five risk classes are polygonized.
    # Each polygon is already restricted to Fars.
    # --------------------------------------------------------

    for class_id in range(5):

        class_mask = (
            (class_raster == class_id)
            &
            fars_mask
        )

        if not np.any(class_mask):
            continue

        for geometry, value in shapes(
            class_raster.astype(np.int16),
            mask=class_mask,
            transform=transform
        ):

            if int(value) != class_id:
                continue

            info = class_info(
                class_id
            )

            feature = {

                "type": "Feature",

                "properties": {

                    "class_id":
                        info["class_id"],

                    "risk":
                        info["label"],

                    "min":
                        info["min"],

                    "max":
                        info["max"],

                    "color":
                        info["color"]
                },

                "geometry":
                    geometry
            }

            features.append(
                feature
            )

    return {

        "type": "FeatureCollection",

        "name":
            "FIRIS FLI Risk Classes",

        "features":
            features
    }


# ============================================================
# COLORIZE PNG
# ============================================================

def colorize(
    values: np.ndarray,
    valid: np.ndarray
):

    rgba = np.zeros(
        (
            values.shape[0],
            values.shape[1],
            4
        ),
        dtype=np.uint8
    )

    rgba[
        valid & (values < 20)
    ] = (
        46,
        125,
        50,
        215
    )

    rgba[
        valid
        & (values >= 20)
        & (values < 40)
    ] = (
        253,
        216,
        53,
        220
    )

    rgba[
        valid
        & (values >= 40)
        & (values < 60)
    ] = (
        251,
        140,
        0,
        225
    )

    rgba[
        valid
        & (values >= 60)
        & (values < 80)
    ] = (
        229,
        57,
        53,
        230
    )

    rgba[
        valid
        & (values >= 80)
    ] = (
        136,
        14,
        79,
        235
    )

    return rgba


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir
    boundary_path = args.boundary

    if not input_path.is_file():
        raise FileNotFoundError(
            f"FLI raster not found: {input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        output_dir /
        "fli_latest.png"
    )

    metadata_path = (
        output_dir /
        "fli_latest.json"
    )

    grid_path = (
        output_dir /
        "fli_latest_grid.json"
    )

    polygons_path = (
        output_dir /
        "fli_polygons.geojson"
    )


    # ========================================================
    # OPEN FLI
    # ========================================================

    with rasterio.open(
        input_path
    ) as src:

        if src.crs is None:
            raise ValueError(
                "FLI raster has no CRS."
            )

        if src.crs.to_epsg() != 4326:
            raise ValueError(
                "FLI CRS must be EPSG:4326. "
                f"Current CRS: {src.crs}"
            )

        values = np.asarray(
            src.read(
                1,
                masked=True
            ).filled(np.nan),
            dtype=np.float32
        )

        valid = (
            np.isfinite(values)
            &
            (values >= 0)
            &
            (values <= 100)
        )

        if not np.any(valid):
            raise ValueError(
                "No valid FLI pixels found."
            )

        # ----------------------------------------------------
        # FARS MASK
        # ----------------------------------------------------

        boundary_geometries = (
            load_boundary(
                boundary_path,
                src.crs
            )
        )

        fars_mask = geometry_mask(
            boundary_geometries,
            out_shape=(
                src.height,
                src.width
            ),
            transform=src.transform,
            invert=True,
            all_touched=False
        )

        web_valid = (
            valid
            &
            fars_mask
        )

        if not np.any(web_valid):
            raise RuntimeError(
                "No valid FLI pixels remain inside Fars."
            )

        # ----------------------------------------------------
        # CLASS RASTER
        # ----------------------------------------------------

        class_raster = classify_fli(
            values
        )

        # ----------------------------------------------------
        # POLYGON GEOJSON
        # ----------------------------------------------------

        polygon_geojson = (
            build_polygon_geojson(
                class_raster,
                src.transform,
                fars_mask
            )
        )

        # ----------------------------------------------------
        # PNG PREVIEW
        # ----------------------------------------------------

        rgba = colorize(
            values,
            web_valid
        )

        Image.fromarray(
            rgba,
            "RGBA"
        ).save(
            png_path,
            optimize=True
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        province_values = values[
            web_valid
        ]

        statistics = {

            "min":
                round(
                    float(
                        province_values.min()
                    ),
                    2
                ),

            "max":
                round(
                    float(
                        province_values.max()
                    ),
                    2
                ),

            "mean":
                round(
                    float(
                        province_values.mean()
                    ),
                    2
                ),

            "valid_pixels":
                int(
                    province_values.size
                )
        }

        # ----------------------------------------------------
        # GRID JSON
        # ----------------------------------------------------

        max_dimension = 350

        row_step = max(
            1,
            int(
                np.ceil(
                    src.height /
                    max_dimension
                )
            )
        )

        col_step = max(
            1,
            int(
                np.ceil(
                    src.width /
                    max_dimension
                )
            )
        )

        sample = values[
            ::row_step,
            ::col_step
        ]

        sample_mask = web_valid[
            ::row_step,
            ::col_step
        ]

        sample = np.where(
            np.isfinite(sample)
            &
            sample_mask,
            sample,
            -9999
        )

        left = float(
            src.bounds.left
        )

        bottom = float(
            src.bounds.bottom
        )

        right = float(
            src.bounds.right
        )

        top = float(
            src.bounds.top
        )

        transform = src.transform


        grid = {

            "bounds": [

                [
                    bottom,
                    left
                ],

                [
                    top,
                    right
                ]
            ],

            "rows":
                int(
                    sample.shape[0]
                ),

            "cols":
                int(
                    sample.shape[1]
                ),

            "row_step":
                int(row_step),

            "col_step":
                int(col_step),

            "origin": {

                "left":
                    left,

                "top":
                    top
            },

            "cell_size": {

                "x":
                    float(src.res[0]),

                "y":
                    float(
                        abs(src.res[1])
                    )
            },

            "values":
                np.round(
                    sample,
                    2
                ).tolist(),

            "boundary_masked":
                True,

            "boundary_file":
                str(boundary_path)
        }


        raster_information = {

            "crs":
                str(src.crs),

            "epsg":
                src.crs.to_epsg(),

            "width":
                int(src.width),

            "height":
                int(src.height),

            "cell_size_x":
                float(src.res[0]),

            "cell_size_y":
                float(
                    abs(src.res[1])
                ),

            "bounds": {

                "left":
                    left,

                "bottom":
                    bottom,

                "right":
                    right,

                "top":
                    top
            },

            "transform": [

                float(transform.a),

                float(transform.b),

                float(transform.c),

                float(transform.d),

                float(transform.e),

                float(transform.f)
            ]
        }


    # ========================================================
    # WRITE GEOJSON
    # ========================================================

    polygons_path.write_text(
        json.dumps(
            polygon_geojson,
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            )
        ),
        encoding="utf-8"
    )


    # ========================================================
    # WRITE GRID JSON
    # ========================================================

    grid_path.write_text(
        json.dumps(
            grid,
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            )
        ),
        encoding="utf-8"
    )


    # ========================================================
    # WRITE METADATA
    # ========================================================

    metadata = {

        "title":
            "FIRIS - Fars Fire Risk Index",

        "source_file":
            input_path.name,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "crs":
            "EPSG:4326",

        "primary_web_layer":
            "fli_polygons.geojson",

        "preview_image":
            "fli_latest.png",

        "grid":
            "fli_latest_grid.json",

        "bounds": [

            [
                bottom,
                left
            ],

            [
                top,
                right
            ]
        ],

        "raster":
            raster_information,

        "statistics":
            statistics,

        "legend":
            LEGEND,

        "boundary": {

            "file":
                str(boundary_path),

            "mask_applied":
                True,

            "valid_pixels_inside_fars":
                int(
                    np.count_nonzero(
                        web_valid
                    )
                ),

            "valid_pixels_outside_fars_removed":
                int(
                    np.count_nonzero(
                        valid
                        &
                        ~fars_mask
                    )
                )
        },

        "web_display": {

            "method":
                "classified GeoJSON polygons",

            "polygon_layer":
                "fli_polygons.geojson",

            "png_is_primary":
                False,

            "outside_fars":
                "excluded",

            "spatial_resampling":
                False
        }
    }


    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


    # ========================================================
    # FINAL LOG
    # ========================================================

    print("")
    print("=" * 70)
    print("FIRIS WEB MAP BUILD")
    print("=" * 70)

    print("")
    print(
        f"Input       : {input_path}"
    )

    print(
        f"Polygon     : {polygons_path}"
    )

    print(
        f"PNG         : {png_path}"
    )

    print(
        f"Metadata    : {metadata_path}"
    )

    print(
        f"Grid JSON   : {grid_path}"
    )

    print("")
    print("RASTER GRID")

    print(
        f"Width       : {raster_information['width']}"
    )

    print(
        f"Height      : {raster_information['height']}"
    )

    print(
        f"Resolution  : "
        f"{raster_information['cell_size_x']}, "
        f"{raster_information['cell_size_y']}"
    )

    print(
        f"Bounds      : "
        f"{raster_information['bounds']}"
    )

    print("")
    print("FARS MASK")

    print(
        f"Valid inside Fars : "
        f"{statistics['valid_pixels']:,}"
    )

    print(
        f"Removed outside   : "
        f"{int(np.count_nonzero(valid & ~fars_mask)):,}"
    )

    print("")
    print(
        f"FLI statistics: {statistics}"
    )

    print("")
    print(
        "Primary web layer : fli_polygons.geojson"
    )

    print(
        "Boundary mask     : APPLIED"
    )

    print(
        "FLI calculation   : UNCHANGED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
