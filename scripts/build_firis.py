#!/usr/bin/env python3
"""
FIRIS / FARS-HRI Fire Likelihood Index builder.

Model:
    FLI = 100 * (
        0.45 * normalized_FWI +
        0.35 * normalized_fuel_risk +
        0.20 * normalized_topography
    )

Outputs:
    data/outputs/f_fwi_fars_<date>.tif
    data/outputs/f_fuel_fars_<date>.tif
    data/outputs/slope_fars_<date>.tif
    data/outputs/f_topo_fars_<date>.tif
    data/outputs/fli_fars_<date>.tif
    data/outputs/firis_report_<date>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ---------------------------------------------------------------------
# تنظیمات مدل
# ---------------------------------------------------------------------

WEIGHT_FWI = 0.45
WEIGHT_FUEL = 0.35
WEIGHT_TOPO = 0.20

FLOAT_NODATA = -9999.0
EPSILON = 1e-12

LOGGER = logging.getLogger("firis")


# ---------------------------------------------------------------------
# ابزارهای عمومی
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index rasters and JSON report."
    )

    parser.add_argument(
        "--fwi-raster",
        required=True,
        type=Path,
        help="Path to the input Fire Weather Index GeoTIFF raster.",
    )
    parser.add_argument(
        "--fuel-raster",
        required=True,
        type=Path,
        help="Path to categorical fuel-bed GeoTIFF raster.",
    )
    parser.add_argument(
        "--dem-raster",
        required=True,
        type=Path,
        help="Path to DEM GeoTIFF raster used to calculate slope.",
    )
    parser.add_argument(
        "--fuel-excel",
        required=True,
        type=Path,
        help="Path to Excel file containing fuel code and fuel risk score.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/outputs"),
        type=Path,
        help="Output directory. Default: data/outputs",
    )
    parser.add_argument(
        "--run-date",
        default=date.today().isoformat(),
        help="Date used in output filenames, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or sheet index. Default: first sheet.",
    )
    parser.add_argument(
        "--fuel-code-column",
        default="fuel_code",
        help="Excel column containing categorical codes matching fuel raster values.",
    )
    parser.add_argument(
        "--fuel-score-column",
        default="fuel_score",
        help=(
            "Excel column containing fuel danger/risk values. "
            "Values may be in 0..1 or 0..100 scale."
        ),
    )
    parser.add_argument(
        "--fwi-min",
        type=float,
        default=None,
        help=(
            "Optional fixed lower FWI bound for normalization. "
            "If omitted, minimum valid FWI value is used."
        ),
    )
    parser.add_argument(
        "--fwi-max",
        type=float,
        default=None,
        help=(
            "Optional fixed upper FWI bound for normalization. "
            "If omitted, maximum valid FWI value is used."
        ),
    )
    parser.add_argument(
        "--topo-slope-cap",
        type=float,
        default=45.0,
        help=(
            "Slope in degrees mapped to a topography risk of 1. "
            "Higher slopes are clipped. Default: 45."
        ),
    )

    return parser.parse_args()


def setup_logging() -> None:
    """Configure console logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_run_date(value: str) -> str:
    """Validate output date format."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(
            f"Invalid --run-date '{value}'. Expected format: YYYY-MM-DD"
        ) from error

    return value


def ensure_file_exists(path: Path, label: str) -> None:
    """Raise a clear error if a required input file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")


def json_safe(value: Any) -> Any:
    """
    Convert NumPy/Pandas values recursively into valid JSON values.

    NaN and Infinity become None, which serializes as JSON null.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if pd.isna(value):
        return None

    return value


def array_stats(array: np.ndarray) -> dict[str, int | float | None]:
    """
    Return JSON-safe descriptive statistics for an array.

    All NaN, +Infinity and -Infinity values are removed before the
    calculations. If no finite values exist, numeric fields become None,
    which becomes `null` in valid JSON.
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
        "min": round(float(np.min(valid_values)), 6),
        "max": round(float(np.max(valid_values)), 6),
        "mean": round(float(np.mean(valid_values)), 6),
        "std": round(float(np.std(valid_values)), 6),
    }


def valid_mask(array: np.ndarray) -> np.ndarray:
    """Return True only for finite numeric values."""
    return np.isfinite(np.asarray(array, dtype=np.float64))


def normalize(
    array: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """
    Normalize finite raster values to range 0..1.

    Invalid values remain NaN. Values outside supplied bounds are clipped.
    """
    values = np.asarray(array, dtype=np.float32)
    result = np.full(values.shape, np.nan, dtype=np.float32)

    mask = valid_mask(values)
    if not np.any(mask):
        return result

    finite_values = values[mask]
    actual_lower = float(np.min(finite_values)) if lower is None else float(lower)
    actual_upper = float(np.max(finite_values)) if upper is None else float(upper)

    if not math.isfinite(actual_lower) or not math.isfinite(actual_upper):
        return result

    if actual_upper - actual_lower <= EPSILON:
        result[mask] = 0.0
        return result

    normalized = (values[mask] - actual_lower) / (actual_upper - actual_lower)
    result[mask] = np.clip(normalized, 0.0, 1.0)

    return result


# ---------------------------------------------------------------------
# خواندن و هم‌ترازسازی رسترها
# ---------------------------------------------------------------------

def read_raster_as_float(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read first raster band as float32 and convert nodata to NaN.
    """
    with rasterio.open(path) as source:
        array = source.read(1).astype(np.float32)
        profile = source.profile.copy()
        source_nodata = source.nodata

    if source_nodata is not None:
        array[np.isclose(array, source_nodata)] = np.nan

    array[~np.isfinite(array)] = np.nan
    return array, profile


def align_raster_to_reference(
    source_path: Path,
    reference_profile: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject/resample source raster to reference raster geometry.

    This makes all raster inputs have identical CRS, transform, width and height.
    """
    with rasterio.open(source_path) as source:
        destination = np.full(
            (reference_profile["height"], reference_profile["width"]),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

    destination[~np.isfinite(destination)] = np.nan
    return destination


# ---------------------------------------------------------------------
# سوخت، شیب و مؤلفه‌های مدل
# ---------------------------------------------------------------------

def normalize_column_name(value: str) -> str:
    """Normalize Excel headers for more robust matching."""
    return re.sub(r"[\s_\-]+", "", str(value).strip().lower())


def resolve_excel_column(
    dataframe: pd.DataFrame,
    requested_column: str,
) -> str:
    """
    Find an Excel column by direct or normalized name match.
    """
    if requested_column in dataframe.columns:
        return requested_column

    requested_normalized = normalize_column_name(requested_column)

    for column in dataframe.columns:
        if normalize_column_name(column) == requested_normalized:
            return str(column)

    available = ", ".join(map(str, dataframe.columns))
    raise KeyError(
        f"Column '{requested_column}' was not found in Excel file. "
        f"Available columns: {available}"
    )


def load_fuel_mapping(
    excel_path: Path,
    sheet_name: str | int,
    fuel_code_column: str,
    fuel_score_column: str,
) -> tuple[dict[int, float], dict[str, Any]]:
    """
    Load fuel code -> normalized fuel risk mapping from Excel.

    Fuel risk values can be provided in:
      - 0..1 scale
      - 0..100 scale

    Any other positive range is normalized using its min/max valid values.
    """
    try:
        if isinstance(sheet_name, str) and sheet_name.isdigit():
            sheet_name = int(sheet_name)

        dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as error:
        raise RuntimeError(f"Could not read fuel Excel file: {excel_path}") from error

    code_column = resolve_excel_column(dataframe, fuel_code_column)
    score_column = resolve_excel_column(dataframe, fuel_score_column)

    table = dataframe[[code_column, score_column]].copy()
    table.columns = ["fuel_code", "fuel_score"]

    table["fuel_code"] = pd.to_numeric(table["fuel_code"], errors="coerce")
    table["fuel_score"] = pd.to_numeric(table["fuel_score"], errors="coerce")
    table = table.dropna(subset=["fuel_code", "fuel_score"])

    if table.empty:
        raise ValueError(
            "No usable fuel mapping rows were found after reading Excel columns."
        )

    table["fuel_code"] = table["fuel_code"].astype(int)

    # اگر یک کد در اکسل چند بار آمده باشد، میانگین آن استفاده می‌شود.
    table = table.groupby("fuel_code", as_index=False)["fuel_score"].mean()

    scores = table["fuel_score"].to_numpy(dtype=np.float32)

    if np.min(scores) >= 0.0 and np.max(scores) <= 1.0:
        normalized_scores = scores
        score_scale = "0_to_1"
    elif np.min(scores) >= 0.0 and np.max(scores) <= 100.0:
        normalized_scores = scores / 100.0
        score_scale = "0_to_100"
    else:
        normalized_scores = normalize(scores)
        score_scale = "min_max_normalized"

    table["normalized_fuel_score"] = np.clip(normalized_scores, 0.0, 1.0)

    mapping = {
        int(row.fuel_code): float(row.normalized_fuel_score)
        for row in table.itertuples(index=False)
    }

    details = {
        "excel_file": str(excel_path),
        "sheet_name": sheet_name,
        "fuel_code_column": code_column,
        "fuel_score_column": score_column,
        "input_score_scale": score_scale,
        "mapping_row_count": int(len(table)),
        "mapped_codes": sorted(int(code) for code in mapping),
    }

    return mapping, details


def map_fuel_codes(
    fuel_codes: np.ndarray,
    mapping: dict[int, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert categorical fuel raster to normalized fuel-risk raster.
    """
    fuel_values = np.asarray(fuel_codes, dtype=np.float32)
    fuel_factor = np.full(fuel_values.shape, np.nan, dtype=np.float32)

    finite_mask = valid_mask(fuel_values)
    observed_codes = np.unique(fuel_values[finite_mask].astype(np.int64))

    for code, score in mapping.items():
        fuel_factor[fuel_values == code] = np.float32(score)

    mapped_mask = valid_mask(fuel_factor)
    mapped_codes = sorted(
        int(code) for code in observed_codes if int(code) in mapping
    )
    unmapped_codes = sorted(
        int(code) for code in observed_codes if int(code) not in mapping
    )

    report = {
        "observed_code_count": int(len(observed_codes)),
        "mapped_code_count": int(len(mapped_codes)),
        "unmapped_code_count": int(len(unmapped_codes)),
        "observed_codes": [int(code) for code in observed_codes.tolist()],
        "mapped_codes": mapped_codes,
        "unmapped_codes": unmapped_codes,
        "mapped_pixel_count": int(np.count_nonzero(mapped_mask)),
    }

    return fuel_factor, report


def calculate_slope_degrees(
    dem: np.ndarray,
    transform: rasterio.Affine,
) -> np.ndarray:
    """
    Calculate slope in degrees from DEM using NumPy gradients.

    The DEM must already be in the target grid. For geographic CRS (degrees),
    this is an approximation. Prefer a projected metric CRS for best accuracy.
    """
    elevation = np.asarray(dem, dtype=np.float32)
    slope = np.full(elevation.shape, np.nan, dtype=np.float32)

    finite = valid_mask(elevation)
    if np.count_nonzero(finite) < 4:
        return slope

    # برای محاسبه gradient، NaNها با نزدیک‌ترین مقدار آماری پر می‌شوند،
    # اما در خروجی نهایی همان نواحی نامعتبر دوباره NaN باقی می‌مانند.
    fill_value = float(np.nanmedian(elevation[finite]))
    working_dem = np.where(finite, elevation, fill_value).astype(np.float32)

    pixel_width = abs(float(transform.a))
    pixel_height = abs(float(transform.e))

    if pixel_width <= EPSILON or pixel_height <= EPSILON:
        raise ValueError("Invalid raster resolution; cannot calculate slope.")

    gradient_y, gradient_x = np.gradient(
        working_dem,
        pixel_height,
        pixel_width,
    )

    slope_radians = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
    slope[finite] = np.degrees(slope_radians[finite]).astype(np.float32)

    slope[~np.isfinite(slope)] = np.nan
    return slope


def calculate_topographic_factor(
    slope_degrees: np.ndarray,
    slope_cap: float,
) -> np.ndarray:
    """
    Convert slope to normalized topographic danger factor in range 0..1.
    """
    if slope_cap <= 0:
        raise ValueError("--topo-slope-cap must be greater than zero.")

    topo_factor = np.full(slope_degrees.shape, np.nan, dtype=np.float32)
    mask = valid_mask(slope_degrees)

    topo_factor[mask] = np.clip(
        slope_degrees[mask] / float(slope_cap),
        0.0,
        1.0,
    )

    return topo_factor


def calculate_fli(
    fwi_factor: np.ndarray,
    fuel_factor: np.ndarray,
    topo_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate final Fire Likelihood Index in range 0..100.

    A final FLI pixel is valid only when all three component factors are valid.
    """
    combined_mask = (
        valid_mask(fwi_factor)
        & valid_mask(fuel_factor)
        & valid_mask(topo_factor)
    )

    fli = np.full(fwi_factor.shape, np.nan, dtype=np.float32)

    fli[combined_mask] = 100.0 * (
        WEIGHT_FWI * fwi_factor[combined_mask]
        + WEIGHT_FUEL * fuel_factor[combined_mask]
        + WEIGHT_TOPO * topo_factor[combined_mask]
    )

    fli[combined_mask] = np.clip(fli[combined_mask], 0.0, 100.0)
    return fli, combined_mask


# ---------------------------------------------------------------------
# نوشتن خروجی‌ها
# ---------------------------------------------------------------------

def output_profile(reference_profile: dict[str, Any]) -> dict[str, Any]:
    """Prepare a consistent single-band float GeoTIFF profile."""
    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=FLOAT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    # برخی پروفایل‌ها ممکن است block size نامعتبر داشته باشند.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    return profile


def write_float_raster(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    """Write float32 GeoTIFF; invalid values are written as FLOAT_NODATA."""
    profile = output_profile(reference_profile)

    output = np.asarray(array, dtype=np.float32).copy()
    output[~np.isfinite(output)] = FLOAT_NODATA

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(output, 1)


def write_json_report(report_path: Path, report: dict[str, Any]) -> None:
    """
    Write strictly valid JSON.

    allow_nan=False blocks accidental NaN / Infinity output.
    """
    safe_report = json_safe(report)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            safe_report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


# ---------------------------------------------------------------------
# اجرای اصلی
# ---------------------------------------------------------------------

def main() -> int:
    """Build FIRIS products."""
    setup_logging()
    args = parse_arguments()

    run_date = validate_run_date(args.run_date)

    ensure_file_exists(args.fwi_raster, "FWI raster")
    ensure_file_exists(args.fuel_raster, "Fuel raster")
    ensure_file_exists(args.dem_raster, "DEM raster")
    ensure_file_exists(args.fuel_excel, "Fuel Excel file")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading FWI raster: %s", args.fwi_raster)
    fwi_raw, reference_profile = read_raster_as_float(args.fwi_raster)

    if reference_profile.get("crs") is None:
        raise ValueError("FWI raster has no CRS. A valid CRS is required.")

    LOGGER.info("Aligning fuel raster to FWI raster grid.")
    fuel_codes = align_raster_to_reference(
        args.fuel_raster,
        reference_profile,
        Resampling.nearest,
    )

    LOGGER.info("Aligning DEM raster to FWI raster grid.")
    dem = align_raster_to_reference(
        args.dem_raster,
        reference_profile,
        Resampling.bilinear,
    )

    LOGGER.info("Loading fuel mapping from Excel.")
    fuel_mapping, mapping_details = load_fuel_mapping(
        excel_path=args.fuel_excel,
        sheet_name=args.sheet_name,
        fuel_code_column=args.fuel_code_column,
        fuel_score_column=args.fuel_score_column,
    )

    LOGGER.info("Mapping fuel codes to fuel risk values.")
    fuel_factor, fuel_mapping_report = map_fuel_codes(fuel_codes, fuel_mapping)

    LOGGER.info("Normalizing FWI values.")
    fwi_factor = normalize(
        fwi_raw,
        lower=args.fwi_min,
        upper=args.fwi_max,
    )

    LOGGER.info("Calculating slope in degrees.")
    slope_degrees = calculate_slope_degrees(
        dem=dem,
        transform=reference_profile["transform"],
    )

    LOGGER.info("Calculating topographic factor.")
    topo_factor = calculate_topographic_factor(
        slope_degrees=slope_degrees,
        slope_cap=args.topo_slope_cap,
    )

    LOGGER.info("Calculating final FLI.")
    fli, final_valid_mask = calculate_fli(
        fwi_factor=fwi_factor,
        fuel_factor=fuel_factor,
        topo_factor=topo_factor,
    )

    output_paths = {
        "fwi_factor": args.output_dir / f"f_fwi_fars_{run_date}.tif",
        "fuel_factor": args.output_dir / f"f_fuel_fars_{run_date}.tif",
        "slope_degrees": args.output_dir / f"slope_fars_{run_date}.tif",
        "topographic_factor": args.output_dir / f"f_topo_fars_{run_date}.tif",
        "fli": args.output_dir / f"fli_fars_{run_date}.tif",
        "report": args.output_dir / f"firis_report_{run_date}.json",
    }

    LOGGER.info("Writing output GeoTIFF files.")
    write_float_raster(output_paths["fwi_factor"], fwi_factor, reference_profile)
    write_float_raster(output_paths["fuel_factor"], fuel_factor, reference_profile)
    write_float_raster(output_paths["slope_degrees"], slope_degrees, reference_profile)
    write_float_raster(
        output_paths["topographic_factor"],
        topo_factor,
        reference_profile,
    )
    write_float_raster(output_paths["fli"], fli, reference_profile)

    report = {
        "project": "FIRIS / FARS-HRI",
        "run_date": run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "formula": (
                "FLI = 100 * (0.45 * FWI + 0.35 * Fuel + 0.20 * Topo)"
            ),
            "weights": {
                "fwi": WEIGHT_FWI,
                "fuel": WEIGHT_FUEL,
                "topography": WEIGHT_TOPO,
            },
            "fwi_normalization_bounds": {
                "requested_min": args.fwi_min,
                "requested_max": args.fwi_max,
            },
            "topographic_slope_cap_degrees": args.topo_slope_cap,
        },
        "inputs": {
            "fwi_raster": str(args.fwi_raster),
            "fuel_raster": str(args.fuel_raster),
            "dem_raster": str(args.dem_raster),
            "fuel_excel": str(args.fuel_excel),
        },
        "grid": {
            "crs": str(reference_profile["crs"]),
            "width": int(reference_profile["width"]),
            "height": int(reference_profile["height"]),
            "transform": [
                float(value)
                for value in tuple(reference_profile["transform"])
            ],
        },
        "fuel_mapping": {
            **mapping_details,
            **fuel_mapping_report,
        },
        "statistics": {
            "fwi_raw": array_stats(fwi_raw),
            "f_fwi": array_stats(fwi_factor),
            "fuel_codes": array_stats(fuel_codes),
            "f_fuel": array_stats(fuel_factor),
            "dem": array_stats(dem),
            "slope_degrees": array_stats(slope_degrees),
            "f_topo": array_stats(topo_factor),
            "fli": array_stats(fli),
        },
        "validity": {
            "fwi_valid_pixels": int(np.count_nonzero(valid_mask(fwi_factor))),
            "fuel_valid_pixels": int(np.count_nonzero(valid_mask(fuel_factor))),
            "topo_valid_pixels": int(np.count_nonzero(valid_mask(topo_factor))),
            "final_model_valid_pixels": int(np.count_nonzero(final_valid_mask)),
        },
        "outputs": {
            key: str(path)
            for key, path in output_paths.items()
            if key != "report"
        },
    }

    LOGGER.info("Writing JSON report: %s", output_paths["report"])
    write_json_report(output_paths["report"], report)

    LOGGER.info("FIRIS build completed successfully.")
    LOGGER.info(
        "Final valid FLI pixels: %s",
        f"{int(np.count_nonzero(final_valid_mask)):,}",
    )
    LOGGER.info("FLI output: %s", output_paths["fli"])
    LOGGER.info("Report output: %s", output_paths["report"])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        LOGGER.exception("FIRIS build failed: %s", error)
        raise SystemExit(1) from error
