#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window


LEGEND = [
    {"min": 0, "max": 20, "label": "کم", "color": "#2e7d32"},
    {"min": 20, "max": 40, "label": "متوسط", "color": "#fdd835"},
    {"min": 40, "max": 60, "label": "زیاد", "color": "#fb8c00"},
    {"min": 60, "max": 80, "label": "خیلی زیاد", "color": "#e53935"},
    {"min": 80, "max": 100, "label": "بحرانی", "color": "#880e4f"},
]


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


def colorize(values, valid):

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
            valid & (values >= 20) & (values < 40),
            (253, 216, 53, 220)
        ),
        (
            valid & (values >= 40) & (values < 60),
            (251, 140, 0, 225)
        ),
        (
            valid & (values >= 60) & (values < 80),
            (229, 57, 53, 230)
        ),
        (
            valid & (values >= 80),
            (136, 14, 79, 235)
        ),
    ]

    for mask, color in classes:
        rgba[mask] = color

    return rgba


def main():

    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

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

    with rasterio.open(input_path) as src:

        if src.crs is None:
            raise ValueError(
                "FLI raster has no CRS."
            )

        if src.crs.to_epsg() != 4326:
            raise ValueError(
                f"FLI CRS must be EPSG:4326. "
                f"Current CRS: {src.crs}"
            )

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

        rows, cols = np.where(valid)

        row_min = int(rows.min())
        row_max = int(rows.max())

        col_min = int(cols.min())
        col_max = int(cols.max())

        height = (
            row_max -
            row_min +
            1
        )

        width = (
            col_max -
            col_min +
            1
        )

        window = Window(
            col_min,
            row_min,
            width,
            height
        )

        cropped = values[
            row_min:row_max + 1,
            col_min:col_max + 1
        ]

        cropped_valid = valid[
            row_min:row_max + 1,
            col_min:col_max + 1
        ]

        rgba = colorize(
            cropped,
            cropped_valid
        )

        image = Image.fromarray(
            rgba,
            "RGBA"
        )

        image.save(
            png_path,
            optimize=True
        )

        transform = (
            src.window_transform(window)
        )

        left = float(
            transform.c
        )

        top = float(
            transform.f
        )

        right = (
            left +
            width * transform.a
        )

        bottom = (
            top +
            height * transform.e
        )

        valid_values = values[valid]

        statistics = {
            "min": round(
                float(valid_values.min()),
                2
            ),
            "max": round(
                float(valid_values.max()),
                2
            ),
            "mean": round(
                float(valid_values.mean()),
                2
            )
        }

        # ----------------------------------------------------
        # Lightweight grid for pixel popup
        # ----------------------------------------------------

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

        sample = cropped[
            ::row_step,
            ::col_step
        ]

        sample = np.where(
            np.isfinite(sample),
            sample,
            -9999
        )

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
            "rows": int(
                sample.shape[0]
            ),
            "cols": int(
                sample.shape[1]
            ),
            "row_step": int(
                row_step
            ),
            "col_step": int(
                col_step
            ),
            "origin": {
                "left": left,
                "top": top
            },
            "cell_size": {
                "x": float(
                    src.res[0]
                ),
                "y": float(
                    abs(src.res[1])
                )
            },
            "values": (
                np.round(
                    sample,
                    2
                ).tolist()
            )
        }

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

        "raster": {

            "width":
                int(width),

            "height":
                int(height),

            "cell_size_x":
                float(src.res[0]),

            "cell_size_y":
                float(
                    abs(src.res[1])
                )
        },

        "statistics":
            statistics,

        "legend":
            LEGEND
    }

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

    print(
        f"Bounds: "
        f"{bottom:.8f}, "
        f"{left:.8f}, "
        f"{top:.8f}, "
        f"{right:.8f}"
    )

    print(
        f"Stats : {statistics}"
    )


if __name__ == "__main__":
    main()
