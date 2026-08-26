#!/usr/bin/env python3
"""
Build FIRIS Fire Likelihood Index (FLI) raster for Fars province.

Formula:
    FLI = 100 * (
        0.45 * normalized_FWI +
        0.35 * normalized_Fuel +
        0.20 * normalized_Topography
    )
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index raster and JSON report."
    )

    parser.add_argument(
        "--fwi-raster",
        required=True,
        help="Path to input FWI GeoTIFF raster.",
    )
    parser.add_argument(
        "--fuel-raster",
        required=True,
        help="Path to input fuel-code GeoTIFF raster.",
    )
    parser.add_argument(
        "--dem-raster",
        required=True,
        help="Path to input DEM GeoTIFF raster.",
    )
    parser.add_argument(
        "--fuel-excel",
        required=True,
        help="Path to Excel file that maps fuel codes to fuel scores.",
    )
    parser.add_argument(
        "--fuel-code-column",
        required=True,
        help="Excel column containing fuel category/code values.",
    )
    parser.add_argument(
        "--fuel-score-column",
        required=True,
        help="Excel column containing normalized or raw fuel risk scores.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs",
        help="Directory where GeoTIFF and JSON results will be created.",
    )
    parser.add_argument(
        "--run-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date used in output filenames, format YYYY-MM-DD.",
    )

    return parser.parse_args()


def ensure_file(path: str | Path, label: str) -> Path:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {file_path}")

    return file_path


def array_stats(array: np.ndarray) -> dict[str, float | int | None]:
    """Return JSON-safe statistics while excluding NaN and infinity values."""
    values = np.asarray(array, dtype=np.float64)
    valid_values = values[np.isfinite(values)]

    if valid_values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "count": int(valid_values.size),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values)),
    }


def normalize_0_1(
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Min-max normalize valid finite values to range [0, 1].

    Invalid cells become NaN.
    """
    output = np.full(values.shape, np.nan, dtype=np.float32)
    finite_mask = np.isfinite(values)

    if valid_mask is not None:
        finite_mask &= valid_mask

    valid_values = values[finite_mask]

    if valid_values.size == 0:
        return output

    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))

    if np.isclose(min_value, max_value):
        output[finite_mask] = 0.5
        return output

    output[finite_mask] = (
        (values[finite_mask] - min_value) / (max_value - min_value)
    ).astype(np.float32)

    return np.clip(output, 0.0, 1.0)


def read_raster_as_float(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read first band as float32 and replace NoData with NaN."""
    with rasterio.open(path) as dataset:
        array = dataset.read(1).astype(np.float32)
        profile = dataset.profile.copy()
        nodata = dataset.nodata

    if nodata is not None:
        array[np.isclose(array, nodata)] = np.nan

    return array, profile


def align_to_reference(
    source_path: Path,
    reference_profile: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray:
    """
    Read and align a raster to FWI raster grid.

    FWI is the reference grid. Fuel uses nearest-neighbor resampling,
    DEM uses bilinear resampling.
    """
    destination_height = reference_profile["height"]
    destination_width = reference_profile["width"]

    destination = np.full(
        (destination_height, destination_width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(source_path) as source:
        source_data = source.read(1).astype(np.float32)

        source_nodata = source.nodata
        if source_nodata is not None:
            source_data[np.isclose(source_data, source_nodata)] = np.nan

        reproject(
            source=source_data,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

    return destination


def load_fuel_mapping(
    excel_path: Path,
    code_column: str,
    score_column: str,
) -> dict[float, float]:
    """Load fuel-code to fuel-score mapping from Excel."""
    workbook = pd.ExcelFile(excel_path)

    selected_df: pd.DataFrame | None = None
    selected_sheet: str | None = None

    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)

        if code_column in dataframe.columns and score_column in dataframe.columns:
            selected_df = dataframe
            selected_sheet = sheet_name
            break

    if selected_df is None:
        available_columns: dict[str, list[str]] = {}

        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=1)
            available_columns[sheet_name] = [str(c) for c in dataframe.columns]

        raise ValueError(
            "Required Excel columns were not found together in any sheet. "
            f"Required columns: '{code_column}', '{score_column}'. "
            f"Available columns: {available_columns}"
        )

    mapping_frame = selected_df[[code_column, score_column]].copy()
    mapping_frame[code_column] = pd.to_numeric(
        mapping_frame[code_column],
        errors="coerce",
    )
    mapping_frame[score_column] = pd.to_numeric(
        mapping_frame[score_column],
        errors="coerce",
    )

    mapping_frame = mapping_frame.dropna()
    mapping_frame = mapping_frame.drop_duplicates(subset=[code_column], keep="last")

    if mapping_frame.empty:
        raise ValueError(
            f"No valid numeric fuel mappings found in Excel sheet '{selected_sheet}'."
        )

    mapping = dict(
        zip(
            mapping_frame[code_column].astype(float),
            mapping_frame[score_column].astype(float),
        )
    )

    print(f"Fuel mapping loaded from sheet: {selected_sheet}")
    print(f"Fuel mapping entries: {len(mapping)}")

    return mapping


def fuel_codes_to_scores(
    fuel_codes: np.ndarray,
    mapping: dict[float, float],
) -> np.ndarray:
    """Convert categorical fuel-code raster values to risk scores."""
    result = np.full(fuel_codes.shape, np.nan, dtype=np.float32)

    for fuel_code, fuel_score in mapping.items():
        code_mask = np.isfinite(fuel_codes) & np.isclose(
            fuel_codes,
            fuel_code,
            atol=0.0001,
        )
        result[code_mask] = fuel_score

    return result


def calculate_slope_degrees(
    dem: np.ndarray,
    transform: Any,
) -> np.ndarray:
    """
    Estimate terrain slope in degrees from DEM using NumPy gradients.

    Pixel resolutions are read from the raster affine transform.
    """
    slope = np.full(dem.shape, np.nan, dtype=np.float32)

    valid_dem = np.isfinite(dem)
    if not np.any(valid_dem):
        return slope

    filled_dem = dem.copy()
    median_elevation = np.nanmedian(filled_dem)
    filled_dem[~valid_dem] = median_elevation

    pixel_x = abs(float(transform.a))
    pixel_y = abs(float(transform.e))

    if pixel_x == 0 or pixel_y == 0:
        raise ValueError("Invalid raster transform: pixel size cannot be zero.")

    gradient_y, gradient_x = np.gradient(
        filled_dem,
        pixel_y,
        pixel_x,
    )

    slope_radians = np.arctan(
        np.sqrt((gradient_x**2) + (gradient_y**2))
    )

    slope[valid_dem] = np.degrees(slope_radians[valid_dem]).astype(np.float32)

    return slope


def main() -> None:
    args = parse_args()

    fwi_path = ensure_file(args.fwi_raster, "FWI raster")
    fuel_path = ensure_file(args.fuel_raster, "Fuel raster")
    dem_path = ensure_file(args.dem_raster, "DEM raster")
    excel_path = ensure_file(args.fuel_excel, "Fuel Excel")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_tif = output_dir / f"fli_fars_{args.run_date}.tif"
    output_json = output_dir / f"firis_report_{args.run_date}.json"

    print("Reading FWI raster...")
    fwi_raw, fwi_profile = read_raster_as_float(fwi_path)

    if fwi_profile.get("crs") is None:
        raise ValueError("FWI raster has no CRS. A valid CRS is required.")

    print("Aligning fuel raster to FWI grid...")
    fuel_codes = align_to_reference(
        source_path=fuel_path,
        reference_profile=fwi_profile,
        resampling=Resampling.nearest,
    )

    print("Aligning DEM raster to FWI grid...")
    dem = align_to_reference(
        source_path=dem_path,
        reference_profile=fwi_profile,
        resampling=Resampling.bilinear,
    )

    print("Loading fuel mapping from Excel...")
    fuel_mapping = load_fuel_mapping(
        excel_path=excel_path,
        code_column=args.fuel_code_column,
        score_column=args.fuel_score_column,
    )

    print("Converting fuel codes to fuel scores...")
    fuel_raw = fuel_codes_to_scores(fuel_codes, fuel_mapping)

    print("Calculating terrain slope...")
    slope_degrees = calculate_slope_degrees(
        dem=dem,
        transform=fwi_profile["transform"],
    )

    print("Normalizing FWI, fuel and slope components...")
    fwi_normalized = normalize_0_1(fwi_raw)
    fuel_normalized = normalize_0_1(fuel_raw)
    topo_normalized = normalize_0_1(slope_degrees)

    valid_mask = (
        np.isfinite(fwi_normalized)
        & np.isfinite(fuel_normalized)
        & np.isfinite(topo_normalized)
    )

    fli = np.full(fwi_raw.shape, np.nan, dtype=np.float32)

    fli[valid_mask] = 100.0 * (
        (FWI_WEIGHT * fwi_normalized[valid_mask])
        + (FUEL_WEIGHT * fuel_normalized[valid_mask])
        + (TOPO_WEIGHT * topo_normalized[valid_mask])
    )

    fli = np.clip(fli, 0.0, 100.0).astype(np.float32)

    output_nodata = -9999.0
    fli_for_output = np.where(
        np.isfinite(fli),
        fli,
        output_nodata,
    ).astype(np.float32)

    output_profile = fwi_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=output_nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
    )

    print(f"Writing GeoTIFF: {output_tif}")
    with rasterio.open(output_tif, "w", **output_profile) as destination:
        destination.write(fli_for_output, 1)
        destination.set_band_description(1, "Fire Likelihood Index (FLI)")
        destination.update_tags(
            PRODUCT="FIRIS Fire Likelihood Index",
            FORMULA="FLI=100*(0.45*FWI+0.35*Fuel+0.20*Topo)",
            FWI_WEIGHT=str(FWI_WEIGHT),
            FUEL_WEIGHT=str(FUEL_WEIGHT),
            TOPO_WEIGHT=str(TOPO_WEIGHT),
            RUN_DATE=args.run_date,
        )

    report = {
        "product": "FIRIS Fire Likelihood Index",
        "run_date": args.run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "FLI = 100 * (0.45 * FWI + 0.35 * Fuel + 0.20 * Topography)",
        "weights": {
            "fwi": FWI_WEIGHT,
            "fuel": FUEL_WEIGHT,
            "topography": TOPO_WEIGHT,
        },
        "inputs": {
            "fwi_raster": str(fwi_path),
            "fuel_raster": str(fuel_path),
            "dem_raster": str(dem_path),
            "fuel_excel": str(excel_path),
            "fuel_code_column": args.fuel_code_column,
            "fuel_score_column": args.fuel_score_column,
            "fuel_mapping_count": len(fuel_mapping),
        },
        "outputs": {
            "fli_raster": str(output_tif),
            "report_json": str(output_json),
        },
        "raster": {
            "width": int(fwi_profile["width"]),
            "height": int(fwi_profile["height"]),
            "crs": str(fwi_profile["crs"]),
            "transform": [float(value) for value in fwi_profile["transform"][:6]],
            "nodata": output_nodata,
        },
        "statistics": {
            "fwi_raw": array_stats(fwi_raw),
            "fuel_codes": array_stats(fuel_codes),
            "fuel_scores_raw": array_stats(fuel_raw),
            "slope_degrees": array_stats(slope_degrees),
            "fwi_normalized": array_stats(fwi_normalized),
            "fuel_normalized": array_stats(fuel_normalized),
            "topography_normalized": array_stats(topo_normalized),
            "fli": array_stats(fli),
        },
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "invalid_pixel_count": int(valid_mask.size - np.count_nonzero(valid_mask)),
    }

    print(f"Writing JSON report: {output_json}")
    with output_json.open("w", encoding="utf-8") as report_file:
        json.dump(
            report,
            report_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("FIRIS build completed successfully.")
    print(f"FLI raster: {output_tif}")
    print(f"JSON report: {output_json}")


if __name__ == "__main__":
    main()
