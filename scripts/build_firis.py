#!/usr/bin/env python3
"""
build_firis.py
==============

Build FIRIS / Fire Likelihood Index (FLI) for Fars province.

Formula:
    FLI = 100 * (
        0.45 * FWI_normalized
        + 0.35 * Fuel_normalized
        + 0.20 * Topography_normalized
    )

Input:
- FWI raster
- Fuel-class raster
- DEM raster
- Global_fuelbeds_parameters_v1.2.xlsx

Output:
- data/outputs/fli_fars_YYYY-MM-DD.tif
- data/outputs/firis_report_YYYY-MM-DD.json

Example:
python scripts/build_firis.py \
  --fwi-raster data/raw/fwi/fwi_ecmwf_fars_2026-08-27.tif \
  --fuel-raster data/raw/fuel/fars_fuel.tif \
  --dem-raster data/raw/topography/dem_fars.tif \
  --fuel-excel data/raw/fuel/Global_fuelbeds_parameters_v1.2.xlsx \
  --fuel-code-column JOIN_VALUE \
  --fuel-score-column AUTO \
  --output-dir data/outputs \
  --run-date 2026-08-27
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

OUTPUT_NODATA = -9999.0

# ستون‌های واقعی بار سوخت در شیت Fuelbeds_metric
FUEL_LOAD_COLUMNS = [
    "G_Load (Mg/ha)",
    "W_1hLoad (Mg/ha)",
    "W_10h Load (Mg/ha)",
    "W_100h Load (Mg/ha)",
    "W_1000h Load (Mg/ha)",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index raster and JSON report."
    )

    parser.add_argument(
        "--fwi-raster",
        required=True,
        type=Path,
        help="Path to FWI GeoTIFF raster.",
    )
    parser.add_argument(
        "--fuel-raster",
        required=True,
        type=Path,
        help="Path to fuel-class GeoTIFF raster.",
    )
    parser.add_argument(
        "--dem-raster",
        required=True,
        type=Path,
        help="Path to DEM GeoTIFF raster.",
    )
    parser.add_argument(
        "--fuel-excel",
        required=True,
        type=Path,
        help="Path to Global_fuelbeds_parameters_v1.2.xlsx.",
    )
    parser.add_argument(
        "--fuel-code-column",
        default="JOIN_VALUE",
        help="Excel column containing the fuel-class code. Default: JOIN_VALUE",
    )
    parser.add_argument(
        "--fuel-score-column",
        default="AUTO",
        help=(
            "Excel column containing an existing fuel score, or AUTO to calculate "
            "fuel score from fuel-load columns in Fuelbeds_metric. Default: AUTO"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where GeoTIFF and JSON report will be written.",
    )
    parser.add_argument(
        "--run-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date used in output filenames, format YYYY-MM-DD.",
    )

    return parser.parse_args()


def ensure_file_exists(file_path: Path, label: str) -> None:
    """Stop with a useful error when an input file is missing."""
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} was not found: {file_path}")


def safe_float(value: Any) -> float | None:
    """Convert finite numeric values to ordinary Python floats for JSON."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(converted):
        return None

    return converted


def array_stats(array: np.ndarray) -> dict[str, Any]:
    """
    Calculate JSON-safe descriptive statistics.

    NaN, infinity and nodata values must already be represented by np.nan.
    """
    values = np.asarray(array, dtype=np.float64)
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "count": int(valid.size),
        "min": safe_float(np.min(valid)),
        "max": safe_float(np.max(valid)),
        "mean": safe_float(np.mean(valid)),
        "std": safe_float(np.std(valid)),
    }


def normalize_to_0_1(array: np.ndarray) -> np.ndarray:
    """
    Min-max normalize valid values to [0, 1].

    Invalid values remain NaN.
    If all valid values are equal, valid pixels become 0.5.
    """
    source = np.asarray(array, dtype=np.float32)
    result = np.full(source.shape, np.nan, dtype=np.float32)

    valid_mask = np.isfinite(source)

    if not np.any(valid_mask):
        return result

    valid_values = source[valid_mask]
    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))

    if math.isclose(minimum, maximum):
        result[valid_mask] = 0.5
        return result

    result[valid_mask] = (source[valid_mask] - minimum) / (maximum - minimum)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def classify_fli(fli: np.ndarray) -> np.ndarray:
    """
    Convert FLI score to classes:
        1 = Very Low   (0-20)
        2 = Low        (20-40)
        3 = Moderate   (40-60)
        4 = High       (60-80)
        5 = Very High  (80-100)
    """
    classes = np.full(fli.shape, 0, dtype=np.uint8)
    valid = np.isfinite(fli)

    classes[valid & (fli <= 20)] = 1
    classes[valid & (fli > 20) & (fli <= 40)] = 2
    classes[valid & (fli > 40) & (fli <= 60)] = 3
    classes[valid & (fli > 60) & (fli <= 80)] = 4
    classes[valid & (fli > 80)] = 5

    return classes


# ---------------------------------------------------------------------
# Raster functions
# ---------------------------------------------------------------------

def read_reference_raster(
    raster_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read FWI raster. Its grid becomes the reference grid for all outputs.
    """
    with rasterio.open(raster_path) as source:
        data = source.read(1).astype(np.float32)

        if source.nodata is not None:
            data[np.isclose(data, source.nodata)] = np.nan

        data[~np.isfinite(data)] = np.nan

        profile = source.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=OUTPUT_NODATA,
            compress="deflate",
            predictor=3,
        )

        metadata = {
            "crs": source.crs,
            "transform": source.transform,
            "width": source.width,
            "height": source.height,
            "profile": profile,
            "resolution_x": abs(source.transform.a),
            "resolution_y": abs(source.transform.e),
        }

    return data, metadata


def align_raster_to_reference(
    raster_path: Path,
    reference_metadata: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject/resample one raster to the FWI reference grid.
    """
    height = int(reference_metadata["height"])
    width = int(reference_metadata["width"])

    destination = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as source:
        source_array = source.read(1).astype(np.float32)

        if source.nodata is not None:
            source_array[np.isclose(source_array, source.nodata)] = np.nan

        source_array[~np.isfinite(source_array)] = np.nan

        reproject(
            source=source_array,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=reference_metadata["transform"],
            dst_crs=reference_metadata["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

    destination[~np.isfinite(destination)] = np.nan
    return destination


# ---------------------------------------------------------------------
# Fuel mapping functions
# ---------------------------------------------------------------------

def load_fuel_mapping(
    excel_path: Path,
    code_column: str,
    score_column: str,
) -> dict[float, float]:
    """
    Create mapping: fuel class code -> raw fuel score.

    AUTO mode:
    Reads the Fuelbeds_metric sheet and calculates a weighted combustible
    fuel load from real physical fuel-load columns.

    Manual mode:
    Reads a user-provided score column from whichever Excel sheet contains
    both code_column and score_column.
    """
    workbook = pd.ExcelFile(excel_path)

    # -------------------------------------------------------------
    # AUTO mode: this fixes the previous error where code tried
    # to find a literal Excel column called "AUTO".
    # -------------------------------------------------------------
    if score_column.strip().upper() == "AUTO":
        sheet_name = "Fuelbeds_metric"

        if sheet_name not in workbook.sheet_names:
            raise ValueError(
                f"Required sheet '{sheet_name}' was not found in {excel_path}. "
                f"Available sheets: {workbook.sheet_names}"
            )

        dataframe = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
        )

        required_columns = [code_column] + FUEL_LOAD_COLUMNS

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Required columns for AUTO fuel-score calculation are missing. "
                f"Missing: {missing_columns}. "
                f"Available: {list(dataframe.columns)}"
            )

        working = dataframe[required_columns].copy()

        # JOIN_VALUE / fuel code
        working[code_column] = pd.to_numeric(
            working[code_column],
            errors="coerce",
        )

        # Numeric cleanup for fuel-load columns
        for column in FUEL_LOAD_COLUMNS:
            working[column] = pd.to_numeric(
                working[column],
                errors="coerce",
            ).fillna(0.0)

            # Negative loads are physically invalid for this purpose.
            working.loc[working[column] < 0, column] = 0.0

        # Weighted combustible fuel load.
        # Fine fuels receive higher weights because they ignite/spread faster.
        working["calculated_fuel_score"] = (
            1.00 * working["G_Load (Mg/ha)"]
            + 1.00 * working["W_1hLoad (Mg/ha)"]
            + 0.75 * working["W_10h Load (Mg/ha)"]
            + 0.45 * working["W_100h Load (Mg/ha)"]
            + 0.20 * working["W_1000h Load (Mg/ha)"]
        )

        working = working.dropna(subset=[code_column])
        working = working.drop_duplicates(
            subset=[code_column],
            keep="last",
        )

        mapping = {
            float(fuel_code): float(fuel_score)
            for fuel_code, fuel_score in zip(
                working[code_column],
                working["calculated_fuel_score"],
            )
            if math.isfinite(float(fuel_code))
            and math.isfinite(float(fuel_score))
        }

        if not mapping:
            raise ValueError(
                "AUTO fuel mapping failed: no valid JOIN_VALUE / score records exist."
            )

        print(f"Fuel mapping loaded from sheet: {sheet_name}")
        print("Fuel score mode: AUTO")
        print(f"Fuel mapping entries: {len(mapping)}")

        return mapping

    # -------------------------------------------------------------
    # Manual mode: use an existing Excel score column.
    # -------------------------------------------------------------
    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)

        if code_column not in dataframe.columns:
            continue

        if score_column not in dataframe.columns:
            continue

        working = dataframe[[code_column, score_column]].copy()

        working[code_column] = pd.to_numeric(
            working[code_column],
            errors="coerce",
        )
        working[score_column] = pd.to_numeric(
            working[score_column],
            errors="coerce",
        )

        working = working.dropna()
        working = working.drop_duplicates(
            subset=[code_column],
            keep="last",
        )

        mapping = {
            float(fuel_code): float(fuel_score)
            for fuel_code, fuel_score in zip(
                working[code_column],
                working[score_column],
            )
            if math.isfinite(float(fuel_code))
            and math.isfinite(float(fuel_score))
        }

        if mapping:
            print(f"Fuel mapping loaded from sheet: {sheet_name}")
            print(f"Fuel score mode: Excel column '{score_column}'")
            print(f"Fuel mapping entries: {len(mapping)}")
            return mapping

    raise ValueError(
        "Required Excel columns were not found together in any sheet. "
        f"Required columns: '{code_column}', '{score_column}'. "
        "For automatic calculation use: --fuel-score-column AUTO"
    )


def create_fuel_score_raster(
    fuel_classes: np.ndarray,
    fuel_mapping: dict[float, float],
) -> tuple[np.ndarray, int]:
    """
    Replace each fuel-class code in raster with its corresponding raw score.

    Returns:
        (raw_fuel_score_raster, unmatched_valid_fuel_pixel_count)
    """
    result = np.full(fuel_classes.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(fuel_classes)
    if not np.any(valid):
        return result, 0

    unique_codes = np.unique(fuel_classes[valid])
    unmatched_codes: list[float] = []

    for code in unique_codes:
        code_as_float = float(code)

        if code_as_float in fuel_mapping:
            result[fuel_classes == code] = fuel_mapping[code_as_float]
        else:
            unmatched_codes.append(code_as_float)

    unmatched_pixel_count = int(
        np.sum(
            valid
            & ~np.isin(
                fuel_classes,
                np.array(list(fuel_mapping.keys()), dtype=np.float32),
            )
        )
    )

    if unmatched_codes:
        preview = unmatched_codes[:20]
        warnings.warn(
            "Some fuel raster class codes have no Excel mapping. "
            f"Count of unmatched class codes: {len(unmatched_codes)}. "
            f"First values: {preview}"
        )

    return result, unmatched_pixel_count


# ---------------------------------------------------------------------
# Topography functions
# ---------------------------------------------------------------------

def calculate_slope_degrees(
    dem: np.ndarray,
    resolution_x: float,
    resolution_y: float,
) -> np.ndarray:
    """
    Calculate slope in degrees from DEM using NumPy gradients.

    DEM nodata is filled temporarily with the median valid elevation only for
    gradient calculation; final invalid DEM cells return NaN.
    """
    result = np.full(dem.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(dem)

    if not np.any(valid):
        return result

    valid_dem_values = dem[valid]
    fill_value = float(np.median(valid_dem_values))

    dem_filled = np.where(valid, dem, fill_value).astype(np.float32)

    gradient_y, gradient_x = np.gradient(
        dem_filled,
        float(resolution_y),
        float(resolution_x),
    )

    slope_radians = np.arctan(
        np.sqrt(
            gradient_x**2 + gradient_y**2
        )
    )

    slope_degrees = np.degrees(slope_radians).astype(np.float32)
    slope_degrees[~valid] = np.nan

    return slope_degrees


# ---------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------

def write_geotiff(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    """Write a float32 GeoTIFF with explicit nodata."""
    output_array = np.where(
        np.isfinite(array),
        array,
        OUTPUT_NODATA,
    ).astype(np.float32)

    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(output_array, 1)


def write_json_report(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    """
    Write strict standard JSON.

    allow_nan=False guarantees that NaN/Infinity cannot be written.
    """
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete FIRIS pipeline."""
    args = parse_arguments()

    ensure_file_exists(args.fwi_raster, "FWI raster")
    ensure_file_exists(args.fuel_raster, "Fuel raster")
    ensure_file_exists(args.dem_raster, "DEM raster")
    ensure_file_exists(args.fuel_excel, "Fuel Excel file")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_fli_path = args.output_dir / f"fli_fars_{args.run_date}.tif"
    output_report_path = args.output_dir / f"firis_report_{args.run_date}.json"

    print("Reading FWI raster...")
    fwi_raw, reference = read_reference_raster(args.fwi_raster)

    if reference["crs"] is None:
        raise ValueError("FWI raster has no CRS. A valid CRS is required.")

    print("Aligning fuel raster to FWI grid...")
    fuel_classes = align_raster_to_reference(
        raster_path=args.fuel_raster,
        reference_metadata=reference,
        resampling=Resampling.nearest,
    )

    print("Aligning DEM raster to FWI grid...")
    dem = align_raster_to_reference(
        raster_path=args.dem_raster,
        reference_metadata=reference,
        resampling=Resampling.bilinear,
    )

    print("Loading fuel mapping from Excel...")
    fuel_mapping = load_fuel_mapping(
        excel_path=args.fuel_excel,
        code_column=args.fuel_code_column,
        score_column=args.fuel_score_column,
    )

    print("Creating fuel-score raster...")
    fuel_raw, unmatched_fuel_pixels = create_fuel_score_raster(
        fuel_classes=fuel_classes,
        fuel_mapping=fuel_mapping,
    )

    print("Calculating DEM slope...")
    slope_degrees = calculate_slope_degrees(
        dem=dem,
        resolution_x=float(reference["resolution_x"]),
        resolution_y=float(reference["resolution_y"]),
    )

    print("Normalizing FWI, Fuel and Topography components...")
    fwi_normalized = normalize_to_0_1(fwi_raw)
    fuel_normalized = normalize_to_0_1(fuel_raw)
    topo_normalized = normalize_to_0_1(slope_degrees)

    # فقط پیکسل‌هایی که هر سه مؤلفه معتبر دارند در FLI نهایی وارد می‌شوند.
    valid_mask = (
        np.isfinite(fwi_normalized)
        & np.isfinite(fuel_normalized)
        & np.isfinite(topo_normalized)
    )

    fli = np.full(fwi_raw.shape, np.nan, dtype=np.float32)

    fli[valid_mask] = 100.0 * (
        FWI_WEIGHT * fwi_normalized[valid_mask]
        + FUEL_WEIGHT * fuel_normalized[valid_mask]
        + TOPO_WEIGHT * topo_normalized[valid_mask]
    )

    fli = np.clip(fli, 0.0, 100.0).astype(np.float32)
    fli_classes = classify_fli(fli)

    print("Writing FLI GeoTIFF...")
    write_geotiff(
        output_path=output_fli_path,
        array=fli,
        reference_profile=reference["profile"],
    )

    total_pixels = int(fli.size)
    valid_pixels = int(np.sum(valid_mask))

    class_counts = {
        "very_low_0_20": int(np.sum(fli_classes == 1)),
        "low_20_40": int(np.sum(fli_classes == 2)),
        "moderate_40_60": int(np.sum(fli_classes == 3)),
        "high_60_80": int(np.sum(fli_classes == 4)),
        "very_high_80_100": int(np.sum(fli_classes == 5)),
    }

    report = {
        "project": "FIRIS",
        "product": "Fire Likelihood Index",
        "province": "Fars",
        "run_date": args.run_date,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formula": {
            "expression": "FLI = 100 * (0.45 * FWI + 0.35 * Fuel + 0.20 * Topography)",
            "weights": {
                "fwi": FWI_WEIGHT,
                "fuel": FUEL_WEIGHT,
                "topography": TOPO_WEIGHT,
            },
        },
        "inputs": {
            "fwi_raster": str(args.fwi_raster),
            "fuel_raster": str(args.fuel_raster),
            "dem_raster": str(args.dem_raster),
            "fuel_excel": str(args.fuel_excel),
            "fuel_code_column": args.fuel_code_column,
            "fuel_score_column": args.fuel_score_column,
        },
        "outputs": {
            "fli_raster": str(output_fli_path),
        },
        "grid": {
            "crs": str(reference["crs"]),
            "width": int(reference["width"]),
            "height": int(reference["height"]),
            "resolution_x": safe_float(reference["resolution_x"]),
            "resolution_y": safe_float(reference["resolution_y"]),
            "total_pixels": total_pixels,
            "valid_fli_pixels": valid_pixels,
            "valid_fli_percent": safe_float(
                100.0 * valid_pixels / total_pixels if total_pixels else 0.0
            ),
        },
        "fuel_mapping": {
            "mode": args.fuel_score_column,
            "mapping_entries": int(len(fuel_mapping)),
            "unmatched_fuel_raster_pixels": unmatched_fuel_pixels,
        },
        "statistics": {
            "fwi_raw": array_stats(fwi_raw),
            "fuel_raw_score": array_stats(fuel_raw),
            "slope_degrees": array_stats(slope_degrees),
            "fli": array_stats(fli),
        },
        "fli_class_pixel_counts": class_counts,
        "fli_classes": {
            "1": "Very Low (0-20)",
            "2": "Low (20-40)",
            "3": "Moderate (40-60)",
            "4": "High (60-80)",
            "5": "Very High (80-100)",
        },
    }

    print("Writing JSON report...")
    write_json_report(
        output_path=output_report_path,
        report=report,
    )

    print("")
    print("FIRIS build completed successfully.")
    print(f"FLI GeoTIFF: {output_fli_path}")
    print(f"JSON report: {output_report_path}")
    print(f"Valid FLI pixels: {valid_pixels:,} / {total_pixels:,}")
    print(f"Fuel mapping entries: {len(fuel_mapping):,}")
    print(f"Unmatched fuel pixels: {unmatched_fuel_pixels:,}")


if __name__ == "__main__":
    main()
