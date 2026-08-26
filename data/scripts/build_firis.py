#!/usr/bin/env python3
"""
FIRIS - Fars Integrated Fire Information System
Build F_Fuel, F_Topo, F_FWI and final Fire Likelihood Index (FLI).

Formula:
    FLI = 100 * (0.45 * F_FWI + 0.35 * F_Fuel + 0.20 * F_Topo)

Fuel factor:
    F_Fuel = 0.35 * FineFuel
           + 0.30 * DeadWood
           + 0.15 * ShrubStructure
           + 0.10 * Litter
           + 0.10 * CanopyStructure

Input files:
    data/raw/fuel/fars_fuel.tif
    data/raw/fuel/Global_fuelbeds_parameters_v1.2.xlsx
    data/raw/topography/dem_fars.tif
    data/raw/fwi/fwi_ecmwf_fars_YYYY-MM-DD.tif

Output files:
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
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FUEL_RASTER = PROJECT_ROOT / "data/raw/fuel/fars_fuel.tif"
DEFAULT_FUEL_TABLE = (
    PROJECT_ROOT / "data/raw/fuel/Global_fuelbeds_parameters_v1.2.xlsx"
)
DEFAULT_DEM_RASTER = PROJECT_ROOT / "data/raw/topography/dem_fars.tif"
DEFAULT_FWI_DIRECTORY = PROJECT_ROOT / "data/raw/fwi"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data/outputs"

NODATA_FLOAT = -9999.0
EPSILON = 1e-12

# Weights approved for the FIRIS model.
WEIGHT_FWI = 0.45
WEIGHT_FUEL = 0.35
WEIGHT_TOPO = 0.20

WEIGHT_FINE_FUEL = 0.35
WEIGHT_DEAD_WOOD = 0.30
WEIGHT_SHRUB = 0.15
WEIGHT_LITTER = 0.10
WEIGHT_CANOPY = 0.10


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def find_latest_fwi_file(fwi_directory: Path) -> Path:
    """Return the most recent dated FWI GeoTIFF in the FWI directory."""
    candidates = sorted(fwi_directory.glob("fwi_ecmwf_fars_*.tif"))

    if not candidates:
        raise FileNotFoundError(
            f"No FWI GeoTIFF was found in: {fwi_directory}\n"
            "Expected name pattern: fwi_ecmwf_fars_YYYY-MM-DD.tif"
        )

    return candidates[-1]


def extract_date_from_fwi_name(fwi_path: Path) -> str:
    """Extract YYYY-MM-DD from the standard FWI filename."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", fwi_path.stem)
    if not match:
        raise ValueError(
            f"Could not find a YYYY-MM-DD date in FWI filename: {fwi_path.name}"
        )
    return match.group(1)


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a table column safely to numeric values; invalid values become zero."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def min_max_normalize(values: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to [0, 1].
    If all source values are identical, return zeros to avoid division by zero.
    """
    values = clean_numeric(values)
    minimum = float(values.min())
    maximum = float(values.max())

    if math.isclose(minimum, maximum, abs_tol=EPSILON):
        return pd.Series(0.0, index=values.index, dtype="float64")

    return (values - minimum) / (maximum - minimum)


def array_statistics(array: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    """Calculate summary statistics only for valid pixels."""
    values = array[valid_mask]

    if values.size == 0:
        return {
            "valid_pixel_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "valid_pixel_count": int(values.size),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
    }


# ---------------------------------------------------------------------
# Fuel-table processing
# ---------------------------------------------------------------------

def build_fuel_lookup(fuel_table_path: Path) -> tuple[dict[int, float], dict[str, Any]]:
    """
    Read the Fuelbeds_metric Excel sheet and create:
        JOIN_VALUE -> F_Fuel lookup dictionary.

    Component definitions:
      FineFuel:
        G_Load (Mg/ha) + W_1hLoad (Mg/ha)

      DeadWood:
        W_10h + W_100h + W_1000h loads

      ShrubStructure:
        average of normalized Shrub Cover and S_Height

      Litter:
        average of normalized Litter_Cover and L_depth

      CanopyStructure:
        average of normalized Tree Cover, TO_Height, TO_HLC and T_Ladder
    """
    logging.info("Reading fuel parameter table: %s", fuel_table_path)

    table = pd.read_excel(
        fuel_table_path,
        sheet_name="Fuelbeds_metric",
        engine="openpyxl",
    )
    table.columns = [str(column).strip() for column in table.columns]

    required_columns = [
        "JOIN_VALUE",
        "FUELBED",
        "G_Load (Mg/ha)",
        "W_1hLoad (Mg/ha)",
        "W_10h Load (Mg/ha)",
        "W_100h Load (Mg/ha)",
        "W_1000h Load (Mg/ha)",
        "Shrub Cover (%)",
        "S_Height (m)",
        "Litter_Cover (%)",
        "L_depth (cm)",
        "Tree Cover (%)",
        "TO_Height (m)",
        "TO_HLC (m)",
        "T_Ladder",
    ]

    missing_columns = [
        column for column in required_columns if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            "The Fuelbeds_metric sheet does not contain required columns:\n"
            + "\n".join(f" - {column}" for column in missing_columns)
        )

    table["JOIN_VALUE_NUMERIC"] = pd.to_numeric(
        table["JOIN_VALUE"], errors="coerce"
    )
    table = table.dropna(subset=["JOIN_VALUE_NUMERIC"]).copy()
    table["JOIN_VALUE_NUMERIC"] = table["JOIN_VALUE_NUMERIC"].astype(np.int64)

    # Remove duplicate fuel-code rows, retaining the first occurrence.
    duplicate_count = int(table["JOIN_VALUE_NUMERIC"].duplicated().sum())
    table = table.drop_duplicates(subset=["JOIN_VALUE_NUMERIC"], keep="first").copy()

    # 1. Fine fuel = grass load + one-hour dead wood load
    fine_fuel_raw = (
        clean_numeric(table["G_Load (Mg/ha)"])
        + clean_numeric(table["W_1hLoad (Mg/ha)"])
    )

    # 2. Dead wood = 10-hour + 100-hour + 1000-hour wood loads
    dead_wood_raw = (
        clean_numeric(table["W_10h Load (Mg/ha)"])
        + clean_numeric(table["W_100h Load (Mg/ha)"])
        + clean_numeric(table["W_1000h Load (Mg/ha)"])
    )

    # 3. Shrub structure = equal contribution of cover and height
    shrub_structure_raw = (
        min_max_normalize(table["Shrub Cover (%)"])
        + min_max_normalize(table["S_Height (m)"])
    ) / 2.0

    # 4. Litter = equal contribution of cover and litter depth
    litter_raw = (
        min_max_normalize(table["Litter_Cover (%)"])
        + min_max_normalize(table["L_depth (cm)"])
    ) / 2.0

    # 5. Canopy structure = equal contribution of four canopy indicators
    canopy_structure_raw = (
        min_max_normalize(table["Tree Cover (%)"])
        + min_max_normalize(table["TO_Height (m)"])
        + min_max_normalize(table["TO_HLC (m)"])
        + min_max_normalize(table["T_Ladder"])
    ) / 4.0

    # Normalize the raw fuel-load components across the complete reference table.
    table["FineFuel"] = min_max_normalize(fine_fuel_raw)
    table["DeadWood"] = min_max_normalize(dead_wood_raw)
    table["ShrubStructure"] = min_max_normalize(shrub_structure_raw)
    table["Litter"] = min_max_normalize(litter_raw)
    table["CanopyStructure"] = min_max_normalize(canopy_structure_raw)

    table["F_Fuel"] = (
        WEIGHT_FINE_FUEL * table["FineFuel"]
        + WEIGHT_DEAD_WOOD * table["DeadWood"]
        + WEIGHT_SHRUB * table["ShrubStructure"]
        + WEIGHT_LITTER * table["Litter"]
        + WEIGHT_CANOPY * table["CanopyStructure"]
    ).clip(0.0, 1.0)

    lookup = {
        int(row["JOIN_VALUE_NUMERIC"]): float(row["F_Fuel"])
        for _, row in table.iterrows()
    }

    component_summary = {
        "reference_table_rows_after_cleanup": int(len(table)),
        "duplicate_join_values_removed": duplicate_count,
        "fuel_component_weights": {
            "FineFuel": WEIGHT_FINE_FUEL,
            "DeadWood": WEIGHT_DEAD_WOOD,
            "ShrubStructure": WEIGHT_SHRUB,
            "Litter": WEIGHT_LITTER,
            "CanopyStructure": WEIGHT_CANOPY,
        },
        "f_fuel_statistics_in_reference_table": {
            "min": round(float(table["F_Fuel"].min()), 6),
            "max": round(float(table["F_Fuel"].max()), 6),
            "mean": round(float(table["F_Fuel"].mean()), 6),
        },
    }

    logging.info(
        "Fuel lookup created successfully for %d JOIN_VALUE codes.", len(lookup)
    )
    return lookup, component_summary


# ---------------------------------------------------------------------
# Raster processing
# ---------------------------------------------------------------------

def read_reprojected_raster(
    source_path: Path,
    target_profile: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray:
    """Read one raster band and reproject/resample it to the target FWI grid."""
    destination = np.full(
        (target_profile["height"], target_profile["width"]),
        NODATA_FLOAT,
        dtype=np.float32,
    )

    with rasterio.open(source_path) as source:
        source_data = source.read(1).astype(np.float32)
        source_nodata = source.nodata

        if source_nodata is not None:
            source_data[np.isclose(source_data, float(source_nodata))] = NODATA_FLOAT

        reproject(
            source=source_data,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=NODATA_FLOAT,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            dst_nodata=NODATA_FLOAT,
            resampling=resampling,
        )

    return destination


def slope_in_degrees(
    dem: np.ndarray,
    transform: Affine,
    crs: Any,
) -> np.ndarray:
    """
    Calculate terrain slope in degrees from a DEM.

    The calculation uses pixel dimensions in metres for projected CRS.
    For geographic CRS (longitude/latitude), degree dimensions are converted
    approximately to metres using the raster's central latitude.
    """
    valid = np.isfinite(dem) & (dem != NODATA_FLOAT)
    safe_dem = np.where(valid, dem, np.nan)

    pixel_x = abs(float(transform.a))
    pixel_y = abs(float(transform.e))

    if crs is not None and getattr(crs, "is_geographic", False):
        center_y = transform.f + (transform.e * dem.shape[0] / 2.0)
        latitude_radians = math.radians(center_y)
        pixel_x *= 111_320.0 * max(math.cos(latitude_radians), 0.01)
        pixel_y *= 110_574.0

    if pixel_x <= 0 or pixel_y <= 0:
        raise ValueError("DEM has invalid pixel dimensions; slope cannot be calculated.")

    gradient_y, gradient_x = np.gradient(safe_dem, pixel_y, pixel_x)
    slope = np.degrees(np.arctan(np.sqrt(gradient_x**2 + gradient_y**2)))
    slope[~valid] = NODATA_FLOAT

    return slope.astype(np.float32)


def write_geotiff(
    output_path: Path,
    array: np.ndarray,
    base_profile: dict[str, Any],
    description: str,
) -> None:
    """Write a single-band compressed Float32 GeoTIFF."""
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA_FLOAT,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype(np.float32), 1)
        dataset.set_band_description(1, description)
        dataset.update_tags(
            product="FIRIS",
            description=description,
            nodata=str(NODATA_FLOAT),
        )


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FIRIS fire-risk factor rasters and final FLI map."
    )
    parser.add_argument(
        "--fwi",
        type=Path,
        default=None,
        help=(
            "Path to an input FWI GeoTIFF. "
            "If omitted, the newest fwi_ecmwf_fars_*.tif is selected."
        ),
    )
    parser.add_argument(
        "--fuel-raster",
        type=Path,
        default=DEFAULT_FUEL_RASTER,
        help="Path to categorical fuel raster.",
    )
    parser.add_argument(
        "--fuel-table",
        type=Path,
        default=DEFAULT_FUEL_TABLE,
        help="Path to Global fuelbeds Excel parameter table.",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM_RASTER,
        help="Path to the DEM raster.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for generated FIRIS outputs.",
    )
    parser.add_argument(
        "--fwi-scale",
        type=float,
        default=100.0,
        help=(
            "FWI value considered equal to F_FWI=1. "
            "Default is 100; values above it are clipped to 1."
        ),
    )
    parser.add_argument(
        "--slope-scale",
        type=float,
        default=45.0,
        help=(
            "Slope in degrees considered equal to F_Topo=1. "
            "Default is 45; steeper slopes are clipped to 1."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logs.",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    if args.fwi_scale <= 0:
        raise ValueError("--fwi-scale must be greater than zero.")
    if args.slope_scale <= 0:
        raise ValueError("--slope-scale must be greater than zero.")

    fwi_path = args.fwi if args.fwi else find_latest_fwi_file(DEFAULT_FWI_DIRECTORY)
    output_dir = args.output_dir
    run_date = extract_date_from_fwi_name(fwi_path)

    required_paths = {
        "FWI raster": fwi_path,
        "Fuel raster": args.fuel_raster,
        "Fuel Excel table": args.fuel_table,
        "DEM raster": args.dem,
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} was not found: {path}")

    logging.info("Starting FIRIS build for date: %s", run_date)
    logging.info("FWI target grid: %s", fwi_path)

    # The FWI raster is selected as the common target grid.
    with rasterio.open(fwi_path) as fwi_dataset:
        if fwi_dataset.crs is None:
            raise ValueError("FWI raster has no CRS; alignment cannot be performed.")

        target_profile = fwi_dataset.profile.copy()
        target_profile.update(
            height=fwi_dataset.height,
            width=fwi_dataset.width,
            transform=fwi_dataset.transform,
            crs=fwi_dataset.crs,
        )

        fwi = fwi_dataset.read(1).astype(np.float32)
        fwi_nodata = fwi_dataset.nodata

    fwi_valid = np.isfinite(fwi)
    if fwi_nodata is not None:
        fwi_valid &= ~np.isclose(fwi, float(fwi_nodata))

    fwi = np.where(fwi_valid, fwi, NODATA_FLOAT).astype(np.float32)

    # Convert original FWI values to [0, 1]. Values >= fwi_scale are high risk.
    f_fwi = np.full(fwi.shape, NODATA_FLOAT, dtype=np.float32)
    f_fwi[fwi_valid] = np.clip(
        fwi[fwi_valid] / float(args.fwi_scale),
        0.0,
        1.0,
    )

    # Build JOIN_VALUE -> F_Fuel dictionary from the Excel reference data.
    fuel_lookup, fuel_table_summary = build_fuel_lookup(args.fuel_table)

    # Categorical fuel classes must use nearest-neighbour resampling.
    logging.info("Aligning categorical fuel raster to FWI grid.")
    fuel_codes = read_reprojected_raster(
        args.fuel_raster,
        target_profile,
        Resampling.nearest,
    )

    fuel_valid = np.isfinite(fuel_codes) & (fuel_codes != NODATA_FLOAT)
    fuel_codes_integer = np.rint(fuel_codes).astype(np.int64)

    f_fuel = np.full(fuel_codes.shape, NODATA_FLOAT, dtype=np.float32)
    mapped_code_count = 0

    for code, fuel_value in fuel_lookup.items():
        code_mask = fuel_valid & (fuel_codes_integer == code)
        if np.any(code_mask):
            f_fuel[code_mask] = np.float32(fuel_value)
            mapped_code_count += 1

    mapped_fuel_mask = f_fuel != NODATA_FLOAT
    unique_raster_codes = np.unique(fuel_codes_integer[fuel_valid])
    unmapped_codes = sorted(
        int(code) for code in unique_raster_codes if int(code) not in fuel_lookup
    )

    # Continuous DEM is aligned with bilinear resampling.
    logging.info("Aligning DEM to FWI grid and calculating terrain slope.")
    dem = read_reprojected_raster(
        args.dem,
        target_profile,
        Resampling.bilinear,
    )

    slope = slope_in_degrees(
        dem=dem,
        transform=target_profile["transform"],
        crs=target_profile["crs"],
    )

    slope_valid = np.isfinite(slope) & (slope != NODATA_FLOAT)
    f_topo = np.full(slope.shape, NODATA_FLOAT, dtype=np.float32)
    f_topo[slope_valid] = np.clip(
        slope[slope_valid] / float(args.slope_scale),
        0.0,
        1.0,
    )

    # A final pixel is valid only if all model factors are valid.
    final_valid = (
        (f_fwi != NODATA_FLOAT)
        & (f_fuel != NODATA_FLOAT)
        & (f_topo != NODATA_FLOAT)
    )

    fli = np.full(fwi.shape, NODATA_FLOAT, dtype=np.float32)
    fli[final_valid] = (
        100.0
        * (
            WEIGHT_FWI * f_fwi[final_valid]
            + WEIGHT_FUEL * f_fuel[final_valid]
            + WEIGHT_TOPO * f_topo[final_valid]
        )
    )

    # Output filenames.
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "f_fwi": output_dir / f"f_fwi_fars_{run_date}.tif",
        "f_fuel": output_dir / f"f_fuel_fars_{run_date}.tif",
        "slope": output_dir / f"slope_fars_{run_date}.tif",
        "f_topo": output_dir / f"f_topo_fars_{run_date}.tif",
        "fli": output_dir / f"fli_fars_{run_date}.tif",
        "report": output_dir / f"firis_report_{run_date}.json",
    }

    logging.info("Writing FIRIS GeoTIFF outputs.")
    write_geotiff(
        output_paths["f_fwi"],
        f_fwi,
        target_profile,
        "FIRIS normalized FWI factor (0 to 1)",
    )
    write_geotiff(
        output_paths["f_fuel"],
        f_fuel,
        target_profile,
        "FIRIS normalized fuel factor (0 to 1)",
    )
    write_geotiff(
        output_paths["slope"],
        slope,
        target_profile,
        "Terrain slope in degrees",
    )
    write_geotiff(
        output_paths["f_topo"],
        f_topo,
        target_profile,
        "FIRIS normalized topographic slope factor (0 to 1)",
    )
    write_geotiff(
        output_paths["fli"],
        fli,
        target_profile,
        "FIRIS Fire Likelihood Index (0 to 100)",
    )

    report = {
        "project": "FIRIS - Fars Integrated Fire Information System",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "formula": (
            "FLI = 100 * (0.45 * F_FWI + 0.35 * F_Fuel + 0.20 * F_Topo)"
        ),
        "weights": {
            "F_FWI": WEIGHT_FWI,
            "F_Fuel": WEIGHT_FUEL,
            "F_Topo": WEIGHT_TOPO,
        },
        "normalization": {
            "F_FWI": (
                f"clip(FWI / {args.fwi_scale}, 0, 1); "
                "FWI values at or above the scale receive 1."
            ),
            "F_Fuel": (
                "Weighted normalized fuelbed parameters from Fuelbeds_metric "
                "using FineFuel=35%, DeadWood=30%, ShrubStructure=15%, "
                "Litter=10%, CanopyStructure=10%."
            ),
            "F_Topo": (
                f"clip(slope_degrees / {args.slope_scale}, 0, 1); "
                "slopes at or above the scale receive 1."
            ),
        },
        "inputs": {
            "fwi_raster": str(fwi_path.relative_to(PROJECT_ROOT)),
            "fuel_raster": str(args.fuel_raster.relative_to(PROJECT_ROOT)),
            "fuel_table": str(args.fuel_table.relative_to(PROJECT_ROOT)),
            "dem_raster": str(args.dem.relative_to(PROJECT_ROOT)),
        },
        "target_grid": {
            "crs": str(target_profile["crs"]),
            "width": int(target_profile["width"]),
            "height": int(target_profile["height"]),
            "transform": list(target_profile["transform"])[:6],
        },
        "fuel_mapping": {
            "unique_fuel_codes_in_raster": [int(code) for code in unique_raster_codes],
            "unique_fuel_code_count": int(len(unique_raster_codes)),
            "mapped_code_count": mapped_code_count,
            "unmapped_codes": unmapped_codes,
            "unmapped_code_count": len(unmapped_codes),
            **fuel_table_summary,
        },
        "statistics": {
            "fwi_original": array_statistics(fwi, fwi_valid),
            "f_fwi": array_statistics(f_fwi, f_fwi != NODATA_FLOAT),
            "f_fuel": array_statistics(f_fuel, f_fuel != NODATA_FLOAT),
            "slope_degrees": array_statistics(slope, slope != NODATA_FLOAT),
            "f_topo": array_statistics(f_topo, f_topo != NODATA_FLOAT),
            "fli": array_statistics(fli, fli != NODATA_FLOAT),
            "final_model_valid_pixels": int(np.count_nonzero(final_valid)),
        },
        "outputs": {
            key: str(path.relative_to(PROJECT_ROOT))
            for key, path in output_paths.items()
            if key != "report"
        },
    }

    with output_paths["report"].open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    logging.info("FIRIS build completed successfully.")
    logging.info("Final FLI raster: %s", output_paths["fli"])
    logging.info("JSON report: %s", output_paths["report"])

    if unmapped_codes:
        logging.warning(
            "Some raster fuel codes were not found in Excel JOIN_VALUE: %s",
            unmapped_codes,
        )
    else:
        logging.info("All raster fuel codes were mapped successfully.")


if __name__ == "__main__":
    main()
