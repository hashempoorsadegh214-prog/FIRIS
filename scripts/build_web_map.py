
#!/usr/bin/env python3

"""
FIRIS - Web Map Builder

Purpose
-------
Convert the existing FLI GeoTIFF into web-ready files.

IMPORTANT
---------
This script does NOT recalculate or modify FLI.

It only:
1. Reads the existing FLI raster.
2. Applies fars.geojson as a WEB DISPLAY mask.
3. Makes pixels outside Fars transparent in PNG.
4. Preserves the original FLI raster grid.
5. Generates metadata and lightweight grid JSON.

Outputs
-------
fli_latest.png
fli_latest.json
fli_latest_grid.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


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
        type=Path,
        help="Existing FLI GeoTIFF"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory"
    )

    parser.add_argument(
        "--boundary",
        required=False,
        type=Path,
        default=Path("fars.geojson"),
        help="Fars Province boundary GeoJSON"
    )

    return parser.parse_args()


# ============================================================
# COLORIZE
# ============================================================

def colorize(
    values: np.ndarray,
    valid: np.ndarray
) -> np.ndarray:

    rgba = np.zeros(
        (
            values.shape[0],
            values.shape[1],
            4
        ),
        dtype=np.uint8
    )

    classes = [
        (
            valid & (values < 20),
            (46, 125, 50, 215)
        ),
        (
            valid
            & (values >= 20)
            & (values < 40),
            (253, 216, 53, 220)
        ),
        (
            valid
            & (values >= 40)
            & (values < 60),
            (251, 140, 0, 225)
        ),
        (
            valid
            & (values >= 60)
            & (values < 80),
            (229, 57, 53, 230)
        ),
        (
            valid & (values >= 80),
            (136, 14, 79, 235)
        )
    ]

    for mask, color in classes:
        rgba[mask] = color

    return rgba


# ============================================================
# LOAD BOUNDARY
# ============================================================

def load_boundary_geometries(
    boundary_path: Path,
    target_crs
):

    if not boundary_path.is_file():
        raise FileNotFoundError(
            f"Fars boundary not found: {boundary_path}"
        )

    with boundary_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        geojson = json.load(file)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError(
            "Boundary GeoJSON must be a FeatureCollection."
        )

    features = geojson.get(
        "features",
        []
    )

    if not features:
        raise ValueError(
            "Fars boundary contains no features."
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
            "Fars boundary contains no valid geometries."
        )

    source_crs = "EPSG:4326"

    crs_obj = geojson.get(
        "crs"
    )

    if isinstance(
        crs_obj,
        dict
    ):

        props = crs_obj.get(
            "properties",
            {}
        )

        name = (
            props.get("name")
            or props.get("href")
        )

        if isinstance(
            name,
            str
        ) and name.strip():

            source_crs = name.strip()

    if str(target_crs) != source_crs:

        transformed = []

        for geometry in geometries:

            transformed.append(
                transform_geom(
                    source_crs,
                    target_crs,
                    geometry,
                    precision=12
                )
            )

        geometries = transformed

    return geometries, source_crs


# ============================================================
# BUILD FARS MASK
# ============================================================

def build_fars_mask(
    boundary_path: Path,
    src
):

    geometries, source_crs = (
        load_boundary_geometries(
            boundary_path,
            src.crs
        )
    )

    mask = geometry_mask(
        geometries,
        out_shape=(
            src.height,
            src.width
        ),
        transform=src.transform,
        invert=True,
        all_touched=False
    )

    pixels_inside = int(
        np.count_nonzero(mask)
    )

    if pixels_inside == 0:

        raise RuntimeError(
            "Fars boundary does not overlap "
            "the FLI raster grid."
        )

    print("")
    print("FARS WEB MASK")
    print("-------------")
    print(f"Boundary       : {boundary_path}")
    print(f"Boundary CRS   : {source_crs}")
    print(f"Raster CRS     : {src.crs}")
    print(f"Raster size    : {src.width} x {src.height}")
    print(f"Pixels in Fars : {pixels_inside:,}")

    return mask


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

    json_path = (
        output_dir /
        "fli_latest.json"
    )

    grid_path = (
        output_dir /
        "fli_latest_grid.json"
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


        # ----------------------------------------------------
        # READ SOURCE FLI
        # ----------------------------------------------------

        data = src.read(
            1,
            masked=True
        )

        values = np.asarray(
            data.filled(np.nan),
            dtype=np.float32
        )


        # ----------------------------------------------------
        # ORIGINAL FLI VALIDITY
        # ----------------------------------------------------

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
        # BUILD FARS MASK
        # ----------------------------------------------------

        fars_mask = build_fars_mask(
            boundary_path,
            src
        )


        # ----------------------------------------------------
        # FINAL WEB VALIDITY
        #
        # ONLY FOR WEB DISPLAY.
        #
        # FLI CALCULATION IS NOT TOUCHED.
        # ----------------------------------------------------

        web_valid = (
            valid
            &
            fars_mask
        )


        inside_count = int(
            np.count_nonzero(
                web_valid
            )
        )

        original_count = int(
            np.count_nonzero(
                valid
            )
        )

        removed_count = (
            original_count
            - inside_count
        )


        if inside_count == 0:

            raise RuntimeError(
                "No valid FLI pixels remain inside "
                "the Fars boundary."
            )


        # ----------------------------------------------------
        # FULL SOURCE GRID
        # ----------------------------------------------------

        width = int(
            src.width
        )

        height = int(
            src.height
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


        # ----------------------------------------------------
        # CREATE TRANSPARENT WEB IMAGE
        # ----------------------------------------------------

        rgba = colorize(
            values,
            web_valid
        )

        image = Image.fromarray(
            rgba,
            "RGBA"
        )

        image.save(
            png_path,
            optimize=True
        )


        # ----------------------------------------------------
        # STATISTICS ONLY INSIDE FARS
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
                inside_count
        }


        # ====================================================
        # LIGHTWEIGHT GRID
        # ====================================================

        max_dimension = 350

        row_step = max(
            1,
            int(
                np.ceil(
                    height /
                    max_dimension
                )
            )
        )

        col_step = max(
            1,
            int(
                np.ceil(
                    width /
                    max_dimension
                )
            )
        )

        sample = values[
            ::row_step,
            ::col_step
        ]

        sample_valid = web_valid[
            ::row_step,
            ::col_step
        ]

        sample = np.where(
            np.isfinite(sample)
            & sample_valid,
            sample,
            -9999
        )


        # ====================================================
        # GRID JSON
        # ====================================================

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
                int(
                    row_step
                ),

            "col_step":
                int(
                    col_step
                ),

            "origin": {

                "left":
                    left,

                "top":
                    top
            },

            "cell_size": {

                "x":
                    float(
                        src.res[0]
                    ),

                "y":
                    float(
                        abs(src.res[1])
                    )
            },

            "full_raster": {

                "width":
                    width,

                "height":
                    height
            },

            "boundary_masked":
                True,

            "boundary_file":
                str(boundary_path),

            "values":
                np.round(
                    sample,
                    2
                ).tolist()
        }


        # ====================================================
        # RASTER INFORMATION
        # ====================================================

        transform = src.transform

        raster_information = {

            "crs":
                str(src.crs),

            "epsg":
                src.crs.to_epsg(),

            "width":
                width,

            "height":
                height,

            "cell_size_x":
                float(
                    src.res[0]
                ),

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
    # METADATA
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

        "image":
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

            "original_valid_pixels":
                original_count,

            "valid_pixels_inside_fars":
                inside_count,

            "valid_pixels_removed_outside_fars":
                removed_count
        },

        "spatial_policy": {

            "reference_grid":
                "FLI source raster",

            "web_boundary":
                str(boundary_path),

            "cropping":
                False,

            "full_raster_extent_preserved":
                True,

            "outside_fars_transparent":
                True,

            "nodata_transparent":
                True,

            "coordinate_system":
                "EPSG:4326"
        }
    }


    # ========================================================
    # WRITE JSON
    # ========================================================

    json_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

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
    # REPORT
    # ========================================================

    print("")
    print("=" * 70)
    print("FIRIS WEB MAP BUILD")
    print("=" * 70)

    print("")
    print(f"Input : {input_path}")
    print(f"PNG   : {png_path}")
    print(f"JSON  : {json_path}")
    print(f"GRID  : {grid_path}")

    print("")
    print("FULL RASTER GRID")
    print(f"Width : {width}")
    print(f"Height: {height}")
    print(
        f"Cell  : "
        f"{float(raster_information['cell_size_x'])}, "
        f"{float(raster_information['cell_size_y'])}"
    )

    print("")
    print("FULL RASTER BOUNDS")
    print(f"South: {bottom:.8f}")
    print(f"West : {left:.8f}")
    print(f"North: {top:.8f}")
    print(f"East : {right:.8f}")

    print("")
    print("WEB MASK RESULTS")
    print(f"Original valid pixels : {original_count:,}")
    print(f"Valid inside Fars     : {inside_count:,}")
    print(f"Removed outside Fars  : {removed_count:,}")

    print("")
    print(f"Stats : {statistics}")

    print("")
    print("Fars boundary mask : APPLIED")
    print("Outside Fars       : TRANSPARENT")
    print("Grid cropping      : DISABLED")
    print("FLI calculation     : UNCHANGED")

    print("=" * 70)


if __name__ == "__main__":
    main()
