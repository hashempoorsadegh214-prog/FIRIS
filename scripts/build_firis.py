#!/usr/bin/env python3
"""
FIRIS - Fars Integrated Fire Information System
------------------------------------------------

Build Fire Likelihood Index (FLI):

    FLI = 100 * (
        0.45 * F_FWI
        + 0.35 * F_Fuel
        + 0.20 * F_Topo
    )

Normalization:
    F_FWI  = clip(FWI / 100.0, 0, 1)
    F_Topo = clip(slope_degrees / 45.0, 0, 1)
    F_Fuel = weighted normalized fuelbed parameters from Fuelbeds_metric

Outputs:
    data/outputs/f_fwi_fars_YYYY-MM-DD.tif
    data/outputs/f_fuel_fars_YYYY-MM-DD.tif
    data/outputs/slope_fars_YYYY-MM-DD.tif
    data/outputs/f_topo_fars_YYYY-MM-DD.tif
    data/outputs/fli_fars_YYYY-MM-DD.tif
    data/outputs/firis_report_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ============================================================
# Model constants
# ============================================================

WEIGHT_FWI = 0.45
WEIGHT_FUEL = 0.35
WEIGHT_TOPO = 0.20

FWI_SCALE = 100.0
SLOPE_SCALE_DEGREES = 45.0

OUTPUT_NODATA = -9999.0

FUEL_COMPONENT_WEIGHTS = {
    "FineFuel": 0.35,
    "DeadWood": 0.30,
    "ShrubStructure": 0.15,
    "Litter": 0.10,
    "CanopyStructure": 0.10,
}


# ============================================================
# Arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index rasters and JSON report."
    )

    parser.add_argument(
        "--fwi-raster",
        type=Path,
        required=True,
        help="Input FWI GeoTIFF.",
    )
    parser.add_argument(
        "--fuel-raster",
        type=Path,
        required=True,
        help="Input fuel-code GeoTIFF.",
    )
    parser.add_argument(
        "--dem-raster",
        type=Path,
        required=True,
        help="Input DEM GeoTIFF.",
    )
    parser.add_argument(
        "--fuel-excel",
        type=Path,
        required=True,
        help="Global_fuelbeds_parameters_v1.2.xlsx path.",
    )
    parser.add_argument(
        "--fuel-code-column",
        default="JOIN_VALUE",
        help="Fuel code column in Excel. Default: JOIN_VALUE",
    )
    parser.add_argument(
        "--fuel-score-column",
        default="AUTO",
        help="Use AUTO for calculated weighted fuel score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date, for example 2026-08-27.",
    )

    return parser.parse_args()


# ============================================================
# General helpers
# ============================================================

def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def to_json_number(value: Any) -> float | None:
    """
    Convert a numeric value to JSON-safe finite float.
    NaN / +Infinity / -Infinity become None.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    return round(numeric_value, 6)


def calculate_statistics(array: np.ndarray) -> dict[str, Any]:
    """
    Calculate statistics only from finite values.

    Important:
    - np.min(array) on an array containing NaN returns NaN.
    - This function first removes NaN / Inf.
    - Output is always valid standard JSON.
    """
    values = np.asarray(array, dtype=np.float64)
    valid_values = values[np.isfinite(values)]

    if valid_values.size == 0:
        return {
            "valid_pixel_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "valid_pixel_count": int(valid_values.size),
        "min": to_json_number(np.min(valid_values)),
        "max": to_json_number(np.max(valid_values)),
        "mean": to_json_number(np.mean(valid_values)),
        "std": to_json_number(np.std(valid_values)),
    }


def normalize_min_max(values: pd.Series) -> pd.Series:
    """
    Normalize numeric values to [0, 1].
    Invalid values are converted to zero.
    """
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_values = numeric_values.clip(lower=0.0)

    minimum = float(numeric_values.min())
    maximum = float(numeric_values.max())

    if math.isclose(minimum, maximum):
        return pd.Series(
            np.zeros(len(numeric_values), dtype=np.float64),
            index=numeric_values.index,
        )

    return (numeric_values - minimum) / (maximum - minimum)


# ============================================================
# Raster read / alignment
# ============================================================

def read_reference_raster(
    raster_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read FWI raster. Its grid becomes the target grid of all output files.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)

        if src.nodata is not None:
            data[np.isclose(data, src.nodata)] = np.nan

        data[~np.isfinite(data)] = np.nan

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=OUTPUT_NODATA,
            compress="deflate",
            predictor=3,
        )

        reference = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": profile,
            "resolution_x": abs(float(src.transform.a)),
            "resolution_y": abs(float(src.transform.e)),
        }

    if reference["crs"] is None:
        raise ValueError("FWI raster has no CRS.")

    return data, reference


def align_to_reference(
    raster_path: Path,
    reference: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject/resample source raster to FWI target grid.
    """
    destination = np.full(
        (reference["height"], reference["width"]),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as src:
        source = src.read(1).astype(np.float32)

        if src.nodata is not None:
            source[np.isclose(source, src.nodata)] = np.nan

        source[~np.isfinite(source)] = np.nan

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=reference["transform"],
            dst_crs=reference["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

    destination[~np.isfinite(destination)] = np.nan
    return destination


# ============================================================
# Fuel mapping
# ============================================================

def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Find first exact available column name from a candidate list.
    """
    for column_name in candidates:
        if column_name in dataframe.columns:
            return column_name
    return None


def load_fuel_mapping_auto(
    excel_path: Path,
    code_column: str,
) -> tuple[dict[float, float], dict[str, Any]]:
    """
    Read Fuelbeds_metric and calculate F_Fuel in AUTO mode.

    The code supports common Fuelbeds_metric column names. It combines
    five normalized components:

        FineFuel        35%
        DeadWood        30%
        ShrubStructure  15%
        Litter          10%
        CanopyStructure 10%
    """
    sheet_name = "Fuelbeds_metric"

    workbook = pd.ExcelFile(excel_path)

    if sheet_name not in workbook.sheet_names:
        raise ValueError(
            f"Sheet '{sheet_name}' not found. "
            f"Available sheets: {workbook.sheet_names}"
        )

    dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)

    if code_column not in dataframe.columns:
        raise ValueError(
            f"Fuel code column '{code_column}' not found in '{sheet_name}'. "
            f"Available columns: {list(dataframe.columns)}"
        )

    # Candidate columns in Global Fuelbeds parameter tables.
    # The first available name is selected for each component.
    fine_fuel_column = find_first_existing_column(
        dataframe,
        [
            "G_Load (Mg/ha)",
            "W_1hLoad (Mg/ha)",
            "W_1h Load (Mg/ha)",
        ],
    )

    dead_wood_columns = [
        col for col in [
            find_first_existing_column(
                dataframe,
                ["W_10h Load (Mg/ha)", "W_10hLoad (Mg/ha)"],
            ),
            find_first_existing_column(
                dataframe,
                ["W_100h Load (Mg/ha)", "W_100hLoad (Mg/ha)"],
            ),
            find_first_existing_column(
                dataframe,
                ["W_1000h Load (Mg/ha)", "W_1000hLoad (Mg/ha)"],
            ),
        ]
        if col is not None
    ]

    shrub_column = find_first_existing_column(
        dataframe,
        [
            "Shrub_Load (Mg/ha)",
            "Shrub Load (Mg/ha)",
            "S_Load (Mg/ha)",
            "ShrubCover",
            "Shrub Cover (%)",
        ],
    )

    litter_column = find_first_existing_column(
        dataframe,
        [
            "Litter_Load (Mg/ha)",
            "Litter Load (Mg/ha)",
            "L_Load (Mg/ha)",
        ],
    )

    canopy_column = find_first_existing_column(
        dataframe,
        [
            "CanopyCover",
            "Canopy Cover (%)",
            "Canopy_Cover (%)",
            "C_Cover (%)",
            "CrownCover",
        ],
    )

    # Basic required values for AUTO score.
    # Fine fuel and dead wood are the most important fire-spread parameters.
    if fine_fuel_column is None:
        raise ValueError(
            "No fine-fuel column was found in Fuelbeds_metric. "
            "Expected one of: G_Load (Mg/ha), W_1hLoad (Mg/ha)."
        )

    if not dead_wood_columns:
        raise ValueError(
            "No dead-wood columns were found in Fuelbeds_metric. "
            "Expected W_10h / W_100h / W_1000h load columns."
        )

    working = dataframe.copy()

    working[code_column] = pd.to_numeric(
        working[code_column],
        errors="coerce",
    )

    # FineFuel
    fine_fuel = normalize_min_max(working[fine_fuel_column])

    # DeadWood: weighted composition of 10h, 100h and 1000h fuels.
    dead_wood_raw = pd.Series(
        np.zeros(len(working), dtype=np.float64),
        index=working.index,
    )

    dead_wood_weights = {
        "W_10h": 0.50,
        "W_100h": 0.30,
        "W_1000h": 0.20,
    }

    for column in dead_wood_columns:
        numeric = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
        numeric = numeric.clip(lower=0.0)

        if "10h" in column and "100h" not in column and "1000h" not in column:
            weight = dead_wood_weights["W_10h"]
        elif "1000h" in column:
            weight = dead_wood_weights["W_1000h"]
        else:
            weight = dead_wood_weights["W_100h"]

        dead_wood_raw = dead_wood_raw + (weight * numeric)

    dead_wood = normalize_min_max(dead_wood_raw)

    # If detailed structure columns are absent, use zero contribution.
    # This preserves total score logic and avoids invalid values.
    shrub_structure = (
        normalize_min_max(working[shrub_column])
        if shrub_column is not None
        else pd.Series(0.0, index=working.index)
    )

    litter = (
        normalize_min_max(working[litter_column])
        if litter_column is not None
        else pd.Series(0.0, index=working.index)
    )

    canopy_structure = (
        normalize_min_max(working[canopy_column])
        if canopy_column is not None
        else pd.Series(0.0, index=working.index)
    )

    working["F_Fuel"] = (
        FUEL_COMPONENT_WEIGHTS["FineFuel"] * fine_fuel
        + FUEL_COMPONENT_WEIGHTS["DeadWood"] * dead_wood
        + FUEL_COMPONENT_WEIGHTS["ShrubStructure"] * shrub_structure
        + FUEL_COMPONENT_WEIGHTS["Litter"] * litter
        + FUEL_COMPONENT_WEIGHTS["CanopyStructure"] * canopy_structure
    )

    working["F_Fuel"] = working["F_Fuel"].clip(lower=0.0, upper=1.0)

    before_cleanup = len(working)

    working = working.dropna(subset=[code_column])
    working = working.drop_duplicates(subset=[code_column], keep="last")

    duplicates_removed = before_cleanup - len(working)

    mapping = {
        float(code): float(score)
        for code, score in zip(working[code_column], working["F_Fuel"])
        if math.isfinite(float(code)) and math.isfinite(float(score))
    }

    if not mapping:
        raise ValueError("No valid fuel mapping records were created.")

    mapping_metadata = {
        "reference_table_rows_after_cleanup": int(len(working)),
        "duplicate_join_values_removed": int(duplicates_removed),
        "fuel_component_weights": FUEL_COMPONENT_WEIGHTS,
        "selected_source_columns": {
            "FineFuel": fine_fuel_column,
            "DeadWood": dead_wood_columns,
            "ShrubStructure": shrub_column,
            "Litter": litter_column,
            "CanopyStructure": canopy_column,
        },
        "f_fuel_statistics_in_reference_table": {
            "min": to_json_number(working["F_Fuel"].min()),
            "max": to_json_number(working["F_Fuel"].max()),
            "mean": to_json_number(working["F_Fuel"].mean()),
        },
    }

    print(f"Fuel mapping loaded from sheet: {sheet_name}")
    print("Fuel score mode: AUTO")
    print(f"Fuel mapping entries: {len(mapping)}")

    return mapping, mapping_metadata


def create_fuel_raster(
    fuel_codes: np.ndarray,
    mapping: dict[float, float],
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Convert fuel-class raster to F_Fuel raster.

    Returns:
        f_fuel_raster,
        unique_codes_in_raster,
        unmapped_codes
    """
    result = np.full(fuel_codes.shape, np.nan, dtype=np.float32)

    valid_mask = np.isfinite(fuel_codes)

    if not np.any(valid_mask):
        return result, [], []

    unique_codes_array = np.unique(fuel_codes[valid_mask])
    unique_codes = [float(code) for code in unique_codes_array.tolist()]

    unmapped_codes: list[float] = []

    for fuel_code in unique_codes:
        if fuel_code in mapping:
            result[fuel_codes == fuel_code] = mapping[fuel_code]
        else:
            unmapped_codes.append(fuel_code)

    return result, unique_codes, unmapped_codes


# ============================================================
# Topography / slope
# ============================================================

def calculate_slope_degrees(
    dem: np.ndarray,
    resolution_x: float,
    resolution_y: float,
) -> np.ndarray:
    """
    Calculate slope in degrees.

    Key correction:
    Before np.gradient(), DEM NaN cells are filled temporarily with the
    median elevation of valid DEM cells. Without this, one NaN spreads into
    neighbouring gradient calculations and can cause NaN statistics.

    Original invalid DEM cells are returned as NaN in final slope raster.
    """
    slope = np.full(dem.shape, np.nan, dtype=np.float32)

    valid_mask = np.isfinite(dem)

    if not np.any(valid_mask):
        return slope

    median_elevation = float(np.nanmedian(dem))

    dem_for_gradient = np.where(
        valid_mask,
        dem,
        median_elevation,
    ).astype(np.float32)

    gradient_y, gradient_x = np.gradient(
        dem_for_gradient,
        resolution_y,
        resolution_x,
    )

    slope_radians = np.arctan(
        np.sqrt(
            np.square(gradient_x) + np.square(gradient_y)
        )
    )

    calculated_slope = np.degrees(slope_radians).astype(np.float32)

    # Preserve original DEM nodata locations.
    calculated_slope[~valid_mask] = np.nan

    # Final safety cleanup.
    calculated_slope[~np.isfinite(calculated_slope)] = np.nan

    return calculated_slope


# ============================================================
# Output functions
# ============================================================

def write_geotiff(
    output_path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
) -> None:
    """
    Write a float32 GeoTIFF and convert NaN to standard raster nodata.
    """
    output_array = np.where(
        np.isfinite(array),
        array,
        OUTPUT_NODATA,
    ).astype(np.float32)

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(output_array, 1)


def write_json_report(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    """
    Write strict valid JSON.

    allow_nan=False is intentionally used:
    if a NaN somehow remains in report, program fails rather than writing
    invalid JSON.
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


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    args = parse_arguments()

    require_file(args.fwi_raster, "FWI raster")
    require_file(args.fuel_raster, "Fuel raster")
    require_file(args.dem_raster, "DEM raster")
    require_file(args.fuel_excel, "Fuel Excel")

    if args.fuel_score_column.strip().upper() != "AUTO":
        raise ValueError(
            "This version is configured for AUTO fuel scoring. "
            "Set FUEL_SCORE_COLUMN to AUTO in GitHub Actions."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading FWI raster...")
    fwi_original, reference = read_reference_raster(args.fwi_raster)

    print("Aligning fuel raster to FWI grid...")
    fuel_codes = align_to_reference(
        args.fuel_raster,
        reference,
        Resampling.nearest,
    )

    print("Aligning DEM raster to FWI grid...")
    dem = align_to_reference(
        args.dem_raster,
        reference,
        Resampling.bilinear,
    )

    print("Loading fuel mapping from Excel...")
    fuel_mapping, fuel_mapping_metadata = load_fuel_mapping_auto(
        args.fuel_excel,
        args.fuel_code_column,
    )

    print("Creating fuel-score raster...")
    f_fuel, unique_fuel_codes, unmapped_codes = create_fuel_raster(
        fuel_codes,
        fuel_mapping,
    )

    print("Calculating DEM slope...")
    slope_degrees = calculate_slope_degrees(
        dem,
        reference["resolution_x"],
        reference["resolution_y"],
    )

    print("Normalizing FWI, Fuel and Topography components...")

    # F_FWI = FWI / 100, clipped to 0..1
    f_fwi = np.full(fwi_original.shape, np.nan, dtype=np.float32)
    fwi_valid = np.isfinite(fwi_original)
    f_fwi[fwi_valid] = np.clip(
        fwi_original[fwi_valid] / FWI_SCALE,
        0.0,
        1.0,
    )

    # F_Topo = slope / 45, clipped to 0..1
    f_topo = np.full(slope_degrees.shape, np.nan, dtype=np.float32)
    slope_valid = np.isfinite(slope_degrees)
    f_topo[slope_valid] = np.clip(
        slope_degrees[slope_valid] / SLOPE_SCALE_DEGREES,
        0.0,
        1.0,
    )

    # Final valid pixels require all three components.
    final_valid_mask = (
        np.isfinite(f_fwi)
        & np.isfinite(f_fuel)
        & np.isfinite(f_topo)
    )

    fli = np.full(fwi_original.shape, np.nan, dtype=np.float32)

    fli[final_valid_mask] = 100.0 * (
        WEIGHT_FWI * f_fwi[final_valid_mask]
        + WEIGHT_FUEL * f_fuel[final_valid_mask]
        + WEIGHT_TOPO * f_topo[final_valid_mask]
    )

    fli = np.clip(fli, 0.0, 100.0).astype(np.float32)

    run_date = args.run_date

    output_f_fwi = args.output_dir / f"f_fwi_fars_{run_date}.tif"
    output_f_fuel = args.output_dir / f"f_fuel_fars_{run_date}.tif"
    output_slope = args.output_dir / f"slope_fars_{run_date}.tif"
    output_f_topo = args.output_dir / f"f_topo_fars_{run_date}.tif"
    output_fli = args.output_dir / f"fli_fars_{run_date}.tif"
    output_report = args.output_dir / f"firis_report_{run_date}.json"

    print("Writing component GeoTIFF files...")
    write_geotiff(output_f_fwi, f_fwi, reference["profile"])
    write_geotiff(output_f_fuel, f_fuel, reference["profile"])
    write_geotiff(output_slope, slope_degrees, reference["profile"])
    write_geotiff(output_f_topo, f_topo, reference["profile"])

    print("Writing FLI GeoTIFF...")
    write_geotiff(output_fli, fli, reference["profile"])

    report = {
        "project": "FIRIS - Fars Integrated Fire Information System",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "formula": (
            "FLI = 100 * "
            "(0.45 * F_FWI + 0.35 * F_Fuel + 0.20 * F_Topo)"
        ),
        "weights": {
            "F_FWI": WEIGHT_FWI,
            "F_Fuel": WEIGHT_FUEL,
            "F_Topo": WEIGHT_TOPO,
        },
        "normalization": {
            "F_FWI": (
                "clip(FWI / 100.0, 0, 1); "
                "FWI values at or above the scale receive 1."
            ),
            "F_Fuel": (
                "Weighted normalized fuelbed parameters from Fuelbeds_metric "
                "using FineFuel=35%, DeadWood=30%, ShrubStructure=15%, "
                "Litter=10%, CanopyStructure=10%."
            ),
            "F_Topo": (
                "clip(slope_degrees / 45.0, 0, 1); "
                "slopes at or above the scale receive 1."
            ),
        },
        "inputs": {
            "fwi_raster": str(args.fwi_raster),
            "fuel_raster": str(args.fuel_raster),
            "fuel_table": str(args.fuel_excel),
            "dem_raster": str(args.dem_raster),
        },
        "target_grid": {
            "crs": str(reference["crs"]),
            "width": int(reference["width"]),
            "height": int(reference["height"]),
            "transform": [
                float(value)
                for value in reference["transform"][:6]
            ],
        },
        "fuel_mapping": {
            "unique_fuel_codes_in_raster": unique_fuel_codes,
            "unique_fuel_code_count": int(len(unique_fuel_codes)),
            "mapped_code_count": int(
                len(unique_fuel_codes) - len(unmapped_codes)
            ),
            "unmapped_codes": unmapped_codes,
            "unmapped_code_count": int(len(unmapped_codes)),
            **fuel_mapping_metadata,
        },
        "statistics": {
            "fwi_original": calculate_statistics(fwi_original),
            "f_fwi": calculate_statistics(f_fwi),
            "f_fuel": calculate_statistics(f_fuel),
            "slope_degrees": calculate_statistics(slope_degrees),
            "f_topo": calculate_statistics(f_topo),
            "fli": calculate_statistics(fli),
            "final_model_valid_pixels": int(np.sum(final_valid_mask)),
        },
        "outputs": {
            "f_fwi": str(output_f_fwi),
            "f_fuel": str(output_f_fuel),
            "slope": str(output_slope),
            "f_topo": str(output_f_topo),
            "fli": str(output_fli),
        },
    }

    print("Writing JSON report...")
    write_json_report(output_report, report)

    print("")
    print("FIRIS build completed successfully.")
    print(f"FLI GeoTIFF: {output_fli}")
    print(f"JSON report: {output_report}")
    print(
        f"Valid FLI pixels: "
        f"{int(np.sum(final_valid_mask)):,} / {fli.size:,}"
    )
    print(f"Fuel mapping entries: {len(fuel_mapping):,}")
    print(f"Unmatched fuel pixels: {int(np.sum(np.isfinite(fuel_codes) & ~np.isfinite(f_fuel))):,}")


if __name__ == "__main__":
    main()
