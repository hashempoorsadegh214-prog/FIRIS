# ============================================================
# AUTO-FIT EXCEL COLUMNS
# ============================================================

from openpyxl.cell.cell import MergedCell
from openpyxl.utils import get_column_letter

for worksheet in workbook.worksheets:

    column_widths = {}

    for row in worksheet.iter_rows():

        for cell in row:

            if isinstance(cell, MergedCell):
                continue

            if cell.value is None:
                continue

            try:
                column_letter = get_column_letter(
                    cell.column
                )
            except Exception:
                continue

            value = str(cell.value)

            current_width = column_widths.get(
                column_letter,
                0
            )

            column_widths[column_letter] = max(
                current_width,
                len(value) + 2
            )

    for column_letter, width in column_widths.items():

        worksheet.column_dimensions[
            column_letter
        ].width = min(
            max(width, 10),
            45
        )
