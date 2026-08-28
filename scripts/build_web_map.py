
#!/usr/bin/env python3

"""
FIRIS - Web Map Builder

Purpose
-------
Convert the final FLI GeoTIFF into web-ready files.

IMPORTANT
---------
This script does NOT modify or recalculate the FLI index.

The FLI raster itself is the authoritative spatial grid.

The complete raster extent is preserved.
No spatial cropping is performed.

Outputs
-------
fli_latest.png
fli_latest.json
fli_latest_grid.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


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
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    return parser.parse_args()


# ============================================================
# COLORIZE
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
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    png_path = (
        output_dir
        / "fli_latest.png"
    )

    json_path = (
        output_dir
        / "fli_latest.json"
    )

    grid_path = (
        output_dir
        / "fli_latest_grid.json"
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
        # READ COMPLETE RASTER
        # ----------------------------------------------------

        data = src.read(
            1,
            masked=True
        )

        values = np.asarray(
            data.filled(np.nan),
            dtype=np.float32
        )

        valid = (
            np.isfinite(values)
            & (values >= 0)
            & (values <= 100)
        )

        if not np.any(valid):

            raise ValueError(
                "No valid FLI pixels found."
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT CROP THE RASTER.
        #
        # The complete FLI grid is retained.
        # ----------------------------------------------------

        height = int(
            src.height
        )

        width = int(
            src.width
        )

        # ----------------------------------------------------
        # AUTHORITATIVE GEOGRAPHIC EXTENT
        #
        # Taken directly from the source raster.
        # ----------------------------------------------------

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
        # COLORIZE COMPLETE RASTER
        # ----------------------------------------------------

        rgba = colorize(
            values,
            valid
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
        # STATISTICS
        # ----------------------------------------------------

        valid_values = values[
            valid
        ]

        statistics = {

            "min":
                round(
                    float(
                        valid_values.min()
                    ),
                    2
                ),

            "max":
                round(
                    float(
                        valid_values.max()
                    ),
                    2
                ),

            "mean":
                round(
                    float(
                        valid_values.mean()
                    ),
                    2
                )
        }

        # ====================================================
        # LIGHTWEIGHT GRID
        # ====================================================

        max_dimension = 350

        row_step = max(
            1,
            int(
                np.ceil(
                    height
                    / max_dimension
                )
            )
        )

        col_step = max(
            1,
            int(
                np.ceil(
                    width
                    / max_dimension
                )
            )
        )

        sample = values[
            ::row_step,
            ::col_step
        ]

        sample = np.where(
            np.isfinite(sample),
            sample,
            -9999
        )

        # ----------------------------------------------------
        # Grid coordinates
        #
        # These describe the SAME full raster grid.
        # ----------------------------------------------------

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

            "values":
                np.round(
                    sample,
                    2
                ).tolist()
        }

        # ----------------------------------------------------
        # Spatial validation information
        # ----------------------------------------------------

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

            "transform":
                [
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

        # ----------------------------------------------------
        # IMPORTANT:
        # Full source raster bounds.
        # ----------------------------------------------------

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

        "spatial_policy": {

            "reference_grid":
                "FLI source raster",

            "cropping":
                False,

            "full_raster_extent_preserved":
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

    print(
        "============================================================"
    )

    print(
        "FIRIS WEB MAP BUILD"
    )

    print(
        "============================================================"
    )

    print(
        f"Input : {input_path}"
    )

    print(
        f"PNG   : {png_path}"
    )

    print(
        f"JSON  : {json_path}"
    )

    print(
        f"GRID  : {grid_path}"
    )

    print("")

    print(
        "FULL RASTER GRID"
    )

    print(
        f"Width : {width}"
    )

    print(
        f"Height: {height}"
    )

    print(
        f"Cell  : "
        f"{float(raster_information['cell_size_x'])}, "
        f"{float(raster_information['cell_size_y'])}"
    )

    print("")

    print(
        "FULL RASTER BOUNDS"
    )

    print(
        f"South: {bottom:.8f}"
    )

    print(
        f"West : {left:.8f}"
    )

    print(
        f"North: {top:.8f}"
    )

    print(
        f"East : {right:.8f}"
    )

    print("")

    print(
        f"Stats : {statistics}"
    )

    print("")

    print(
        "Spatial cropping : DISABLED"
    )

    print(
        "Full grid extent : PRESERVED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":

    main()
