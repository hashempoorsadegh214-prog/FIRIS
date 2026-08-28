#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import rasterio

from openpyxl import Workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build FIRIS Excel report from existing FLI raster"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Existing FLI raster"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output Excel file"
    )

    parser.add_argument(
        "--run-date",
        required=False,
        default=None,
        help="Run date YYYY-MM-DD"
    )

    return parser.parse_args()


# ============================================================
# RISK CLASS
# ============================================================

def risk_class(value):

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
# READ EXISTING FLI
# ============================================================

def read_fli_raster(path):

    with rasterio.open(path) as src:

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

            raise RuntimeError(
                "No valid FLI pixels found."
            )

        valid_values = values[valid]

        statistics = {

            "min":
                float(valid_values.min()),

            "max":
                float(valid_values.max()),

            "mean":
                float(valid_values.mean()),

            "count":
                int(valid_values.size)
        }

        return {
            "statistics": statistics,
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "resolution_x": float(src.res[0]),
            "resolution_y": float(abs(src.res[1])),
            "bounds": [
                float(src.bounds.bottom),
                float(src.bounds.left),
                float(src.bounds.top),
                float(src.bounds.right)
            ]
        }


# ============================================================
# BUILD EXCEL
# ============================================================

def build_excel(
    output_path,
    input_path,
    raster_info,
    run_date
):

    workbook = Workbook()

    worksheet = workbook.active

    worksheet.title = "گزارش FIRIS"

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    worksheet.merge_cells(
        "A1:B1"
    )

    worksheet["A1"] = (
        "گزارش سامانه FIRIS "
        "استان فارس"
    )

    worksheet["A1"].font = Font(
        bold=True,
        size=16
    )

    worksheet["A1"].alignment = Alignment(
        horizontal="center",
        vertical="center"
    )

    worksheet.row_dimensions[1].height = 30

    # --------------------------------------------------------
    # GENERAL INFORMATION
    # --------------------------------------------------------

    rows = [

        ("تاریخ اجرا", run_date),

        (
            "فایل ورودی",
            input_path.name
        ),

        (
            "سیستم مختصات",
            raster_info["crs"]
        ),

        (
            "عرض Raster",
            raster_info["width"]
        ),

        (
            "ارتفاع Raster",
            raster_info["height"]
        ),

        (
            "اندازه سلول X",
            raster_info["resolution_x"]
        ),

        (
            "اندازه سلول Y",
            raster_info["resolution_y"]
        ),

    ]

    start_row = 3

    for index, (label, value) in enumerate(
        rows,
        start=start_row
    ):

        worksheet.cell(
            index,
            1,
            label
        )

        worksheet.cell(
            index,
            2,
            value
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    stats_start = (
        start_row +
        len(rows) +
        2
    )

    worksheet.merge_cells(
        start_row=stats_start,
        start_column=1,
        end_row=stats_start,
        end_column=2
    )

    worksheet.cell(
        stats_start,
        1,
        "آمار شاخص FLI"
    )

    worksheet.cell(
        stats_start,
        1
    ).font = Font(
        bold=True,
        size=14
    )

    worksheet.cell(
        stats_start,
        1
    ).alignment = Alignment(
        horizontal="center"
    )

    statistics = raster_info["statistics"]

    stat_rows = [

        (
            "حداقل FLI",
            round(
                statistics["min"],
                2
            )
        ),

        (
            "حداکثر FLI",
            round(
                statistics["max"],
                2
            )
        ),

        (
            "میانگین FLI",
            round(
                statistics["mean"],
                2
            )
        ),

        (
            "تعداد سلول معتبر",
            statistics["count"]
        ),

    ]

    for index, (label, value) in enumerate(
        stat_rows,
        start=stats_start + 1
    ):

        worksheet.cell(
            index,
            1,
            label
        )

        worksheet.cell(
            index,
            2,
            value
        )

    # --------------------------------------------------------
    # RISK INTERPRETATION
    # --------------------------------------------------------

    risk_row = (
        stats_start +
        len(stat_rows) +
        2
    )

    worksheet.merge_cells(
        start_row=risk_row,
        start_column=1,
        end_row=risk_row,
        end_column=2
    )

    worksheet.cell(
        risk_row,
        1,
        "طبقه‌بندی خطر"
    )

    worksheet.cell(
        risk_row,
        1
    ).font = Font(
        bold=True,
        size=14
    )

    mean_value = statistics["mean"]

    worksheet.cell(
        risk_row + 1,
        1,
        "طبقه خطر بر اساس میانگین"
    )

    worksheet.cell(
        risk_row + 1,
        2,
        risk_class(mean_value)
    )

    # --------------------------------------------------------
    # BOUNDS
    # --------------------------------------------------------

    bounds_row = (
        risk_row +
        4
    )

    worksheet.merge_cells(
        start_row=bounds_row,
        start_column=1,
        end_row=bounds_row,
        end_column=2
    )

    worksheet.cell(
        bounds_row,
        1,
        "محدوده مکانی"
    )

    worksheet.cell(
        bounds_row,
        1
    ).font = Font(
        bold=True,
        size=14
    )

    bottom, left, top, right = (
        raster_info["bounds"]
    )

    bounds = [

        ("جنوب", bottom),

        ("غرب", left),

        ("شمال", top),

        ("شرق", right),

    ]

    for index, (label, value) in enumerate(
        bounds,
        start=bounds_row + 1
    ):

        worksheet.cell(
            index,
            1,
            label
        )

        worksheet.cell(
            index,
            2,
            value
        )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    thin = Side(
        style="thin"
    )

    border = Border(
        left=thin,
        right=thin,
        top=thin,
        bottom=thin
    )

    for row in worksheet.iter_rows():

        for cell in row:

            if isinstance(
                cell,
                MergedCell
            ):
                continue

            if cell.value is None:
                continue

            cell.border = border

            cell.alignment = Alignment(
                vertical="center",
                horizontal="right"
            )

    # --------------------------------------------------------
    # AUTO-FIT COLUMNS
    # --------------------------------------------------------

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

            value = str(
                cell.value
            )

            current_width = (
                column_widths.get(
                    column_letter,
                    0
                )
            )

            column_widths[
                column_letter
            ] = max(
                current_width,
                len(value) + 2
            )

    for column_letter, width in (
        column_widths.items()
    ):

        worksheet.column_dimensions[
            column_letter
        ].width = min(
            max(width, 12),
            45
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    workbook.save(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"FLI raster not found: "
            f"{input_path}"
        )

    run_date = (
        args.run_date
        or datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d"
        )
    )

    print(
        "============================================================"
    )

    print(
        "FIRIS EXCEL REPORT"
    )

    print(
        "============================================================"
    )

    print(
        f"Input  : {input_path}"
    )

    print(
        f"Output : {output_path}"
    )

    print(
        f"Date   : {run_date}"
    )

    print(
        "Reading existing FLI raster..."
    )

    raster_info = read_fli_raster(
        input_path
    )

    print(
        f"Min    : {raster_info['statistics']['min']:.2f}"
    )

    print(
        f"Max    : {raster_info['statistics']['max']:.2f}"
    )

    print(
        f"Mean   : {raster_info['statistics']['mean']:.2f}"
    )

    print(
        "Building Excel report..."
    )

    build_excel(
        output_path,
        input_path,
        raster_info,
        run_date
    )

    print(
        "Excel report created successfully."
    )

    print(
        f"FILE: {output_path}"
    )


if __name__ == "__main__":
    main()
