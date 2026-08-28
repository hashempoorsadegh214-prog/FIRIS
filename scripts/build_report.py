#!/usr/bin/env python3

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build FIRIS Excel report"
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--run-date",
        required=True
    )

    return parser.parse_args()


def classify(value):

    if value < 20:
        return "کم"

    if value < 40:
        return "متوسط"

    if value < 60:
        return "زیاد"

    if value < 80:
        return "خیلی زیاد"

    return "بحرانی"


def main():

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with rasterio.open(
        input_path
    ) as src:

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

        valid_values = values[
            valid
        ]

        minimum = float(
            valid_values.min()
        )

        maximum = float(
            valid_values.max()
        )

        mean = float(
            valid_values.mean()
        )

        classes = {

            "کم": int(
                np.sum(
                    (valid_values >= 0)
                    &
                    (valid_values < 20)
                )
            ),

            "متوسط": int(
                np.sum(
                    (valid_values >= 20)
                    &
                    (valid_values < 40)
                )
            ),

            "زیاد": int(
                np.sum(
                    (valid_values >= 40)
                    &
                    (valid_values < 60)
                )
            ),

            "خیلی زیاد": int(
                np.sum(
                    (valid_values >= 60)
                    &
                    (valid_values < 80)
                )
            ),

            "بحرانی": int(
                np.sum(
                    valid_values >= 80
                )
            )
        }

        total_pixels = len(
            valid_values
        )

        resolution_x = float(
            src.res[0]
        )

        resolution_y = float(
            abs(src.res[1])
        )

        bounds = src.bounds

        crs = str(
            src.crs
        )


    # ========================================================
    # WORKBOOK
    # ========================================================

    wb = Workbook()

    ws = wb.active

    ws.title = "FIRIS Report"


    # ========================================================
    # TITLE
    # ========================================================

    ws["A1"] = (
        "FIRIS - Fars Fire Risk Index"
    )

    ws["A1"].font = Font(
        bold=True,
        size=16
    )

    ws.merge_cells(
        "A1:D1"
    )


    ws["A3"] = "تاریخ اجرا"
    ws["B3"] = args.run_date

    ws["A4"] = "زمان تولید UTC"
    ws["B4"] = datetime.now(
        timezone.utc
    ).isoformat()

    ws["A5"] = "Raster"
    ws["B5"] = input_path.name

    ws["A6"] = "CRS"
    ws["B6"] = crs


    # ========================================================
    # STATISTICS
    # ========================================================

    ws["A8"] = "آمار شاخص FLI"

    ws["A8"].font = Font(
        bold=True
    )

    rows = [

        ("حداقل", round(minimum, 2)),

        ("حداکثر", round(maximum, 2)),

        ("میانگین", round(mean, 2)),

        ("تعداد سلول معتبر", total_pixels),

        ("تفکیک X", resolution_x),

        ("تفکیک Y", resolution_y),

    ]

    row = 9

    for label, value in rows:

        ws.cell(
            row=row,
            column=1,
            value=label
        )

        ws.cell(
            row=row,
            column=2,
            value=value
        )

        row += 1


    # ========================================================
    # RISK CLASSES
    # ========================================================

    row += 1

    ws.cell(
        row=row,
        column=1,
        value="طبقات خطر"
    )

    ws.cell(
        row=row,
        column=1
    ).font = Font(
        bold=True
    )

    row += 1

    ws.cell(
        row=row,
        column=1,
        value="طبقه"
    )

    ws.cell(
        row=row,
        column=2,
        value="تعداد سلول"
    )

    ws.cell(
        row=row,
        column=3,
        value="درصد"
    )

    row += 1

    for name, count in classes.items():

        percentage = (
            count /
            total_pixels *
            100
        )

        ws.cell(
            row=row,
            column=1,
            value=name
        )

        ws.cell(
            row=row,
            column=2,
            value=count
        )

        ws.cell(
            row=row,
            column=3,
            value=round(
                percentage,
                2
            )
        )

        row += 1


    # ========================================================
    # BOUNDS
    # ========================================================

    row += 1

    ws.cell(
        row=row,
        column=1,
        value="محدوده مکانی"
    )

    ws.cell(
        row=row,
        column=1
    ).font = Font(
        bold=True
    )

    row += 1

    bounds_rows = [

        ("غرب", float(bounds.left)),

        ("شرق", float(bounds.right)),

        ("جنوب", float(bounds.bottom)),

        ("شمال", float(bounds.top)),

    ]

    for label, value in bounds_rows:

        ws.cell(
            row=row,
            column=1,
            value=label
        )

        ws.cell(
            row=row,
            column=2,
            value=value
        )

        row += 1


    # ========================================================
    # FORMATTING
    # ========================================================

    for column in ws.columns:

        max_length = 0

        column_letter = (
            column[0].column_letter
        )

        for cell in column:

            cell.alignment = Alignment(
                horizontal="right"
            )

            if cell.value is not None:

                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        ws.column_dimensions[
            column_letter
        ].width = min(
            max_length + 4,
            45
        )


    # ========================================================
    # SAVE
    # ========================================================

    wb.save(
        output_path
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
        f"Input : {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Min   : {minimum:.2f}"
    )

    print(
        f"Max   : {maximum:.2f}"
    )

    print(
        f"Mean  : {mean:.2f}"
    )

    print(
        "Excel report created successfully."
    )


if __name__ == "__main__":

    main()
