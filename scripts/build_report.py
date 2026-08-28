```python
#!/usr/bin/env python3

"""
FIRIS - Detailed Excel Fire Risk Report
=======================================

این فایل FLI را محاسبه نمی‌کند.

فقط FLI نهایی تولیدشده را می‌خواند و یک گزارش
منطقه‌ای و طبقه‌بندی‌شده Excel تولید می‌کند.

خروجی شامل:

1) خلاصه
2) مناطق_شکار_ممنوع
3) مناطق_چهارگانه
4) طبقات_خطر
5) اطلاعات_فنی

نگاشت اسامی طبق درخواست فعلی پروژه:

protected_areas.geojson
    -> مناطق شکار ممنوع

hunting_banned.geojson
    -> مناطق چهارگانه
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from openpyxl import Workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter
from rasterio.mask import mask


# ============================================================
# RISK CLASS
# ============================================================

RISK_CLASSES = [
    ("کم", 0.0, 20.0),
    ("متوسط", 20.0, 40.0),
    ("زیاد", 40.0, 60.0),
    ("خیلی زیاد", 60.0, 80.0),
    ("بحرانی", 80.0, 100.000001),
]


RISK_COLORS = {
    "کم": "2E7D32",
    "متوسط": "FDD835",
    "زیاد": "FB8C00",
    "خیلی زیاد": "E53935",
    "بحرانی": "880E4F",
    "بدون داده": "777777",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build detailed FIRIS Excel "
            "report from existing FLI raster."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Existing FLI raster."
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Excel file."
    )

    parser.add_argument(
        "--run-date",
        required=False,
        default=None,
        help="Forecast date YYYY-MM-DD."
    )

    parser.add_argument(
        "--protected",
        required=False,
        type=Path,
        default=Path(
            "protected_areas.geojson"
        ),
        help="Protected areas GeoJSON."
    )

    parser.add_argument(
        "--hunting",
        required=False,
        type=Path,
        default=Path(
            "hunting_banned.geojson"
        ),
        help="Hunting banned GeoJSON."
    )

    return parser.parse_args()


# ============================================================
# FILE CHECK
# ============================================================

def require_file(
    path: Path,
    label: str
):

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


# ============================================================
# RISK CLASS
# ============================================================

def risk_class(
    value: float | None
) -> str:

    if value is None:

        return "بدون داده"

    value = float(value)

    if not np.isfinite(value):

        return "بدون داده"

    if value < 20:

        return "کم"

    if value < 40:

        return "متوسط"

    if value < 60:

        return "زیاد"

    if value < 80:

        return "خیلی زیاد"

    return "بحرانی"


# ============================================================
# RISK COLOR
# ============================================================

def risk_fill(
    label: str
) -> PatternFill:

    return PatternFill(
        fill_type="solid",
        fgColor=RISK_COLORS.get(
            label,
            RISK_COLORS["بدون داده"]
        )
    )


# ============================================================
# LOAD GEOJSON
# ============================================================

def load_geojson(
    path: Path
) -> dict[str, Any]:

    require_file(
        path,
        "GeoJSON"
    )

    import json

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    if (
        data.get("type")
        != "FeatureCollection"
    ):

        raise ValueError(
            f"Invalid GeoJSON FeatureCollection: "
            f"{path}"
        )

    features = data.get(
        "features",
        []
    )

    if not features:

        raise ValueError(
            f"No features found in: {path}"
        )

    return data


# ============================================================
# FEATURE NAME
# ============================================================

def get_feature_name(
    feature: dict[str, Any],
    fallback: str
) -> str:

    properties = (
        feature.get(
            "properties"
        )
        or {}
    )

    candidates = [

        "name",
        "NAME",
        "Name",
        "shapeName",
        "ShapeName",
        "NAME_1",
        "NAME_FA",
        "name_fa",
        "title",
        "TITLE",
        "نام",
        "نام منطقه",
        "نام_منطقه",

    ]

    for key in candidates:

        value = properties.get(
            key
        )

        if (
            value is not None
            and str(value).strip()
        ):

            return str(
                value
            ).strip()


    # Secondary fallback:
    # search any text-like property that looks like a name.

    for key, value in properties.items():

        if not isinstance(
            value,
            str
        ):
            continue

        if not value.strip():
            continue

        key_lower = str(
            key
        ).strip().lower()

        if (
            "name" in key_lower
            or
            "title" in key_lower
            or
            "نام" in key_lower
        ):

            return value.strip()


    return fallback


# ============================================================
# READ FLI RASTER
# ============================================================

def read_fli_raster(
    path: Path
):

    with rasterio.open(
        path
    ) as src:

        if src.crs is None:

            raise ValueError(
                "FLI raster has no CRS."
            )


        data = src.read(
            1
        ).astype(
            np.float32
        )


        nodata = src.nodata

        if nodata is not None:

            data[
                np.isclose(
                    data,
                    float(nodata)
                )
            ] = np.nan


        data[
            ~np.isfinite(data)
        ] = np.nan


        data[
            (data < 0)
            |
            (data > 100)
        ] = np.nan


        valid = np.isfinite(
            data
        )


        if not np.any(
            valid
        ):

            raise RuntimeError(
                "FLI raster contains no valid pixels."
            )


        valid_values = data[
            valid
        ]


        statistics = {

            "min":
                float(
                    np.min(
                        valid_values
                    )
                ),

            "max":
                float(
                    np.max(
                        valid_values
                    )
                ),

            "mean":
                float(
                    np.mean(
                        valid_values
                    )
                ),

            "count":
                int(
                    valid_values.size
                ),
        }


        metadata = {

            "crs":
                str(
                    src.crs
                ),

            "width":
                int(
                    src.width
                ),

            "height":
                int(
                    src.height
                ),

            "resolution_x":
                float(
                    src.res[0]
                ),

            "resolution_y":
                float(
                    abs(
                        src.res[1]
                    )
                ),

            "bounds": {

                "left":
                    float(
                        src.bounds.left
                    ),

                "bottom":
                    float(
                        src.bounds.bottom
                    ),

                "right":
                    float(
                        src.bounds.right
                    ),

                "top":
                    float(
                        src.bounds.top
                    ),
            },

            "transform": [

                float(
                    src.transform.a
                ),

                float(
                    src.transform.b
                ),

                float(
                    src.transform.c
                ),

                float(
                    src.transform.d
                ),

                float(
                    src.transform.e
                ),

                float(
                    src.transform.f
                ),
            ],
        }


        return (
            data,
            statistics,
            metadata
        )


# ============================================================
# REGION FLI STATISTICS
# ============================================================

def calculate_region_statistics(
    src,
    feature: dict[str, Any]
):

    geometry = feature.get(
        "geometry"
    )


    if not geometry:

        return {

            "count":
                0,

            "min":
                None,

            "max":
                None,

            "mean":
                None,

            "risk":
                "بدون داده",
        }


    try:

        masked_data, _ = mask(

            src,

            [geometry],

            crop=True,

            filled=False,

            all_touched=False

        )


        band = masked_data[0]


        if np.ma.isMaskedArray(
            band
        ):

            values = (
                band.compressed()
                .astype(
                    np.float32
                )
            )

        else:

            values = np.asarray(
                band,
                dtype=np.float32
            )


        values = values[
            np.isfinite(values)
        ]


        values = values[
            (values >= 0)
            &
            (values <= 100)
        ]


        if values.size == 0:

            return {

                "count":
                    0,

                "min":
                    None,

                "max":
                    None,

                "mean":
                    None,

                "risk":
                    "بدون داده",
            }


        mean_value = float(
            np.mean(
                values
            )
        )


        return {

            "count":
                int(
                    values.size
                ),

            "min":
                float(
                    np.min(
                        values
                    )
                ),

            "max":
                float(
                    np.max(
                        values
                    )
                ),

            "mean":
                mean_value,

            "risk":
                risk_class(
                    mean_value
                ),
        }


    except Exception as error:

        print(
            "WARNING: "
            "Could not calculate region statistics:"
        )

        print(
            f"  {error}"
        )


        return {

            "count":
                0,

            "min":
                None,

            "max":
                None,

            "mean":
                None,

            "risk":
                "بدون داده",
        }


# ============================================================
# RISK CLASS STATISTICS FOR WHOLE PROVINCE
# ============================================================

def calculate_class_statistics(
    values: np.ndarray
):

    valid = (

        np.isfinite(
            values
        )

        &

        (values >= 0)

        &

        (values <= 100)

    )


    total = int(
        np.count_nonzero(
            valid
        )
    )


    rows = []


    for label, minimum, maximum in RISK_CLASSES:

        class_mask = (

            valid

            &

            (
                values >=
                minimum
            )

            &

            (
                values <
                maximum
            )

        )


        count = int(
            np.count_nonzero(
                class_mask
            )
        )


        percent = (

            100.0 *
            count /
            total

            if total > 0

            else 0.0

        )


        rows.append({

            "label":
                label,

            "min":
                minimum,

            "max":
                min(
                    maximum,
                    100.0
                ),

            "count":
                count,

            "percent":
                percent,
        })


    return rows


# ============================================================
# EXCEL STYLES
# ============================================================

def create_styles():

    thin = Side(
        style="thin",
        color="C9CED3"
    )


    return {

        "border":
            Border(
                left=thin,
                right=thin,
                top=thin,
                bottom=thin
            ),

        "header_fill":
            PatternFill(
                fill_type="solid",
                fgColor="8B0000"
            ),

        "section_fill":
            PatternFill(
                fill_type="solid",
                fgColor="E9EEF2"
            ),

        "title_font":
            Font(
                name="B Nazanin",
                size=16,
                bold=True
            ),

        "header_font":
            Font(
                name="B Nazanin",
                size=11,
                bold=True,
                color="FFFFFF"
            ),

        "normal_font":
            Font(
                name="B Nazanin",
                size=11
            ),

        "bold_font":
            Font(
                name="B Nazanin",
                size=11,
                bold=True
            ),

    }


# ============================================================
# APPLY GENERAL FORMAT
# ============================================================

def apply_sheet_format(
    worksheet,
    styles
):

    worksheet.sheet_view.rightToLeft = True

    worksheet.sheet_properties.pageSetUpPr.fitToPage = True

    worksheet.page_setup.fitToWidth = 1

    worksheet.page_setup.fitToHeight = 0


    for row in worksheet.iter_rows():

        for cell in row:

            if isinstance(
                cell,
                MergedCell
            ):

                continue


            if cell.value is None:

                continue


            cell.font = styles[
                "normal_font"
            ]


            cell.alignment = Alignment(
                horizontal="right",
                vertical="center",
                wrap_text=True
            )


            cell.border = styles[
                "border"
            ]


# ============================================================
# HEADER ROW STYLE
# ============================================================

def style_header_row(
    worksheet,
    row_number,
    styles
):

    for cell in worksheet[
        row_number
    ]:

        if isinstance(
            cell,
            MergedCell
        ):

            continue


        if cell.value is None:

            continue


        cell.fill = styles[
            "header_fill"
        ]


        cell.font = styles[
            "header_font"
        ]


        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )


        cell.border = styles[
            "border"
        ]


# ============================================================
# AUTO FIT
# ============================================================

def auto_fit_columns(
    worksheet,
    minimum=12,
    maximum=42
):

    column_widths = {}


    for row in worksheet.iter_rows():

        for cell in row:

            if isinstance(
                cell,
                MergedCell
            ):

                continue


            if cell.value is None:

                continue


            try:

                column_letter = (
                    get_column_letter(
                        cell.column
                    )
                )

            except Exception:

                continue


            text = str(
                cell.value
            )


            current = column_widths.get(
                column_letter,
                0
            )


            column_widths[
                column_letter
            ] = max(
                current,
                len(text) + 2
            )


    for column_letter, width in (
        column_widths.items()
    ):

        worksheet.column_dimensions[
            column_letter
        ].width = min(
            max(
                width,
                minimum
            ),
            maximum
        )


# ============================================================
# SUMMARY SHEET
# ============================================================

def build_summary_sheet(
    workbook,
    run_date,
    input_path,
    statistics,
    metadata,
    styles
):

    worksheet = workbook.active

    worksheet.title = "خلاصه"


    worksheet.merge_cells(
        "A1:C1"
    )


    title_cell =
        worksheet["A1"]


    title_cell.value = (
        "گزارش خطر حریق "
        "استان فارس"
    )


    title_cell.font = styles[
        "title_font"
    ]


    title_cell.alignment = Alignment(
        horizontal="center",
        vertical="center"
    )


    worksheet.row_dimensions[
        1
    ].height = 30


    rows = [

        (
            "تاریخ پیش‌بینی",
            run_date,
        ),

        (
            "فایل FLI",
            input_path.name,
        ),

        (
            "سیستم مختصات",
            metadata["crs"],
        ),

        (
            "تعداد سلول معتبر",
            statistics["count"],
        ),

        (
            "حداقل FLI",
            round(
                statistics["min"],
                2
            ),
        ),

        (
            "حداکثر FLI",
            round(
                statistics["max"],
                2
            ),
        ),

        (
            "میانگین FLI",
            round(
                statistics["mean"],
                2
            ),
        ),

        (
            "طبقه خطر استان",
            risk_class(
                statistics["mean"]
            ),
        ),

        (
            "رزولوشن X",
            metadata["resolution_x"],
        ),

        (
            "رزولوشن Y",
            metadata["resolution_y"],
        ),

    ]


    start_row = 3


    for row_number, (
        label,
        value
    ) in enumerate(
        rows,
        start=start_row
    ):

        worksheet.cell(
            row_number,
            1,
            label
        )


        worksheet.cell(
            row_number,
            2,
            value
        )


    style_header_row(
        worksheet,
        0,
        styles
    )


    risk_row = (
        start_row
        +
        7
    )


    risk_cell =
        worksheet.cell(
            risk_row,
            2
        )


    risk_label =
        risk_class(
            statistics["mean"]
        )


    risk_cell.fill =
        risk_fill(
            risk_label
        )


    risk_cell.font = Font(
        name="B Nazanin",
        size=11,
        bold=True,
        color=(
            "000000"
            if risk_label == "متوسط"
            else "FFFFFF"
        )
    )


    apply_sheet_format(
        worksheet,
        styles
    )


    auto_fit_columns(
        worksheet
    )


    worksheet.freeze_panes = "A3"


# ============================================================
# REGIONAL SHEET
# ============================================================

def build_region_sheet(
    workbook,
    title,
    report_label,
    geojson,
    src,
    styles
):

    worksheet = workbook.create_sheet(
        title=title
    )


    columns = [

        "ردیف",

        "نام منطقه",

        "نوع منطقه",

        "میانگین FLI",

        "حداقل FLI",

        "حداکثر FLI",

        "تعداد سلول معتبر",

        "طبقه خطر",

    ]


    for column_number, value in enumerate(
        columns,
        start=1
    ):

        worksheet.cell(
            1,
            column_number,
            value
        )


    style_header_row(
        worksheet,
        1,
        styles
    )


    features = geojson[
        "features"
    ]


    row_number = 2


    for index, feature in enumerate(
        features,
        start=1
    ):

        fallback = (
            f"{report_label} "
            f"{index}"
        )


        name = get_feature_name(
            feature,
            fallback
        )


        stats = calculate_region_statistics(
            src,
            feature
        )


        values = [

            index,

            name,

            report_label,

            (
                round(
                    stats["mean"],
                    2
                )
                if stats["mean"]
                is not None
                else None
            ),

            (
                round(
                    stats["min"],
                    2
                )
                if stats["min"]
                is not None
                else None
            ),

            (
                round(
                    stats["max"],
                    2
                )
                if stats["max"]
                is not None
                else None
            ),

            stats["count"],

            stats["risk"],

        ]


        for column_number, value in enumerate(
            values,
            start=1
        ):

            worksheet.cell(
                row_number,
                column_number,
                value
            )


        risk_cell =
            worksheet.cell(
                row_number,
                8
            )


        risk_label =
            stats["risk"]


        risk_cell.fill =
            risk_fill(
                risk_label
            )


        risk_cell.font = Font(
            name="B Nazanin",
            size=11,
            bold=True,
            color=(
                "000000"
                if risk_label == "متوسط"
                else "FFFFFF"
            )
        )


        row_number += 1


    apply_sheet_format(
        worksheet,
        styles
    )


    style_header_row(
        worksheet,
        1,
        styles
    )


    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = (
        worksheet.dimensions
    )


    auto_fit_columns(
        worksheet
    )


# ============================================================
# CLASS SHEET
# ============================================================

def build_class_sheet(
    workbook,
    values,
    styles
):

    worksheet =
        workbook.create_sheet(
            title="طبقات_خطر"
        )


    headers = [

        "طبقه خطر",

        "حداقل FLI",

        "حداکثر FLI",

        "تعداد سلول",

        "درصد از کل",

    ]


    for column_number, value in enumerate(
        headers,
        start=1
    ):

        worksheet.cell(
            1,
            column_number,
            value
        )


    style_header_row(
        worksheet,
        1,
        styles
    )


    rows =
        calculate_class_statistics(
            values
        )


    for row_number, item in enumerate(
        rows,
        start=2
    ):

        row_values = [

            item["label"],

            item["min"],

            item["max"],

            item["count"],

            round(
                item["percent"],
                2
            ),

        ]


        for column_number, value in enumerate(
            row_values,
            start=1
        ):

            worksheet.cell(
                row_number,
                column_number,
                value
            )


        risk_label =
            item["label"]


        cell =
            worksheet.cell(
                row_number,
                1
            )


        cell.fill =
            risk_fill(
                risk_label
            )


        cell.font = Font(
            name="B Nazanin",
            size=11,
            bold=True,
            color=(
                "000000"
                if risk_label == "متوسط"
                else "FFFFFF"
            )
        )


    apply_sheet_format(
        worksheet,
        styles
    )


    style_header_row(
        worksheet,
        1,
        styles
    )


    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = (
        worksheet.dimensions
    )


    auto_fit_columns(
        worksheet
    )


# ============================================================
# TECHNICAL SHEET
# ============================================================

def build_technical_sheet(
    workbook,
    metadata,
    styles
):

    worksheet =
        workbook.create_sheet(
            title="اطلاعات_فنی"
        )


    rows = [

        (
            "CRS",
            metadata["crs"]
        ),

        (
            "Width",
            metadata["width"]
        ),

        (
            "Height",
            metadata["height"]
        ),

        (
            "Resolution X",
            metadata["resolution_x"]
        ),

        (
            "Resolution Y",
            metadata["resolution_y"]
        ),

        (
            "Left",
            metadata["bounds"]["left"]
        ),

        (
            "Bottom",
            metadata["bounds"]["bottom"]
        ),

        (
            "Right",
            metadata["bounds"]["right"]
        ),

        (
            "Top",
            metadata["bounds"]["top"]
        ),

        (
            "Transform a",
            metadata["transform"][0]
        ),

        (
            "Transform b",
            metadata["transform"][1]
        ),

        (
            "Transform c",
            metadata["transform"][2]
        ),

        (
            "Transform d",
            metadata["transform"][3]
        ),

        (
            "Transform e",
            metadata["transform"][4]
        ),

        (
            "Transform f",
            metadata["transform"][5]
        ),

    ]


    for row_number, (
        label,
        value
    ) in enumerate(
        rows,
        start=1
    ):

        worksheet.cell(
            row_number,
            1,
            label
        )


        worksheet.cell(
            row_number,
            2,
            value
        )


    apply_sheet_format(
        worksheet,
        styles
    )


    auto_fit_columns(
        worksheet
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args =
        parse_args()


    input_path =
        args.input


    output_path =
        args.output


    protected_path =
        args.protected


    hunting_path =
        args.hunting


    require_file(
        input_path,
        "FLI raster"
    )


    require_file(
        protected_path,
        "protected areas GeoJSON"
    )


    require_file(
        hunting_path,
        "hunting banned GeoJSON"
    )


    if args.run_date:

        run_date =
            args.run_date

    else:

        parts =
            input_path.stem.split(
                "_"
            )


        run_date =
            parts[-1]


    print("")
    print("=" * 70)
    print("FIRIS DETAILED EXCEL REPORT")
    print("=" * 70)

    print(
        f"FLI raster       : {input_path}"
    )

    print(
        f"Forecast date    : {run_date}"
    )

    print(
        f"Protected source : {protected_path}"
    )

    print(
        f"Hunting source   : {hunting_path}"
    )


    # ========================================================
    # READ RASTER
    # ========================================================

    print("")
    print(
        "Reading existing FLI raster..."
    )


    with rasterio.open(
        input_path
    ) as src:

        data =
            src.read(
                1
            ).astype(
                np.float32
            )


        if src.nodata is not None:

            data[
                np.isclose(
                    data,
                    float(
                        src.nodata
                    )
                )
            ] = np.nan


        data[
            ~np.isfinite(data)
        ] = np.nan


        data[
            (data < 0)
            |
            (data > 100)
        ] = np.nan


        valid =
            np.isfinite(
                data
            )


        if not np.any(
            valid
        ):

            raise RuntimeError(
                "No valid FLI pixels found."
            )


        valid_values =
            data[
                valid
            ]


        statistics = {

            "min":
                float(
                    valid_values.min()
                ),

            "max":
                float(
                    valid_values.max()
                ),

            "mean":
                float(
                    valid_values.mean()
                ),

            "count":
                int(
                    valid_values.size
                ),
        }


        metadata = {

            "crs":
                str(
                    src.crs
                ),

            "width":
                int(
                    src.width
                ),

            "height":
                int(
                    src.height
                ),

            "resolution_x":
                float(
                    src.res[0]
                ),

            "resolution_y":
                float(
                    abs(
                        src.res[1]
                    )
                ),

            "bounds": {

                "left":
                    float(
                        src.bounds.left
                    ),

                "bottom":
                    float(
                        src.bounds.bottom
                    ),

                "right":
                    float(
                        src.bounds.right
                    ),

                "top":
                    float(
                        src.bounds.top
                    ),
            },

            "transform": [

                float(
                    src.transform.a
                ),

                float(
                    src.transform.b
                ),

                float(
                    src.transform.c
                ),

                float(
                    src.transform.d
                ),

                float(
                    src.transform.e
                ),

                float(
                    src.transform.f
                ),

            ],
        }


        # Keep Rasterio dataset open
        # while calculating regional statistics.

        protected_geojson =
            load_geojson(
                protected_path
            )


        hunting_geojson =
            load_geojson(
                hunting_path
            )


        # ====================================================
        # CREATE WORKBOOK
        # ====================================================

        workbook =
            Workbook()


        styles =
            create_styles()


        # ====================================================
        # SUMMARY
        # ====================================================

        build_summary_sheet(

            workbook,

            run_date,

            input_path,

            statistics,

            metadata,

            styles
        )


        # ====================================================
        # IMPORTANT NAMING CONVENTION
        #
        # protected_areas.geojson
        #      -> مناطق شکار ممنوع
        #
        # hunting_banned.geojson
        #      -> مناطق چهارگانه
        # ====================================================

        build_region_sheet(

            workbook,

            "مناطق_شکار_ممنوع",

            "مناطق شکار ممنوع",

            protected_geojson,

            src,

            styles
        )


        build_region_sheet(

            workbook,

            "مناطق_چهارگانه",

            "مناطق چهارگانه",

            hunting_geojson,

            src,

            styles
        )


        # ====================================================
        # RISK CLASSES
        # ====================================================

        build_class_sheet(

            workbook,

            data,

            styles
        )


        # ====================================================
        # TECHNICAL
        # ====================================================

        build_technical_sheet(

            workbook,

            metadata,

            styles
        )


    # ========================================================
    # FINAL GLOBAL FORMATTING
    # ========================================================

    for worksheet in workbook.worksheets:

        worksheet.sheet_view.rightToLeft = True


        for row in worksheet.iter_rows():

            for cell in row:

                if isinstance(
                    cell,
                    MergedCell
                ):

                    continue


                if cell.value is None:

                    continue


                /*
                 * Font:
                 * B Nazanin
                 */

                cell.font = Font(
                    name="B Nazanin",
                    size=(
                        cell.font.sz
                        or
                        11
                    ),
                    bold=bool(
                        cell.font.bold
                    ),
                    italic=bool(
                        cell.font.italic
                    ),
                    color=(
                        cell.font.color
                        if cell.font.color
                        and cell.font.color.type == "rgb"
                        else None
                    )
                )


        auto_fit_columns(
            worksheet
        )


    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    workbook.save(
        output_path
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("")
    print("=" * 70)
    print("FIRIS EXCEL REPORT CREATED")
    print("=" * 70)

    print(
        f"Output : {output_path}"
    )

    print("")
    print(
        "Sheets:"
    )

    print(
        "  1. خلاصه"
    )

    print(
        "  2. مناطق_شکار_ممنوع"
    )

    print(
        "  3. مناطق_چهارگانه"
    )

    print(
        "  4. طبقات_خطر"
    )

    print(
        "  5. اطلاعات_فنی"
    )

    print("")
    print(
        f"Province mean FLI: "
        f"{statistics['mean']:.2f}"
    )

    print(
        f"Province risk     : "
        f"{risk_class(statistics['mean'])}"
    )

    print("")
    print(
        "✓ Excel report completed successfully."
    )


if __name__ == "__main__":

    main()
```
