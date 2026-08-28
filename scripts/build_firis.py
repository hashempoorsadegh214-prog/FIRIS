
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

Important raster rule
---------------------
FWI is the final target grid.

Fuel:
    nearest-neighbour -> FWI grid

DEM:
    slope is calculated FIRST on the native DEM grid
    using metric cell spacing
    THEN slope is resampled to the FWI grid using bilinear

This avoids calculating slope from an already coarsened DEM.

Outputs:
    f_fwi_fars_YYYY-MM-DD.tif
    f_fuel_fars_YYYY-MM-DD.tif
    slope_fars_YYYY-MM-DD.tif
    f_topo_fars_YYYY-MM-DD.tif
    fli_fars_YYYY-MM-DD.tif
    firis_report_YYYY-MM-DD.json
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
from rasterio.transform import array_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.crs import CRS


# ============================================================
# MODEL CONSTANTS
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
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index."
    )

    parser.add_argument(
        "--fwi-raster",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fuel-raster",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dem-raster",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fuel-excel",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fuel-code-column",
        default="JOIN_VALUE",
    )

    parser.add_argument(
        "--fuel-score-column",
        default="AUTO",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-date",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def require_file(path: Path, label: str) -> None:

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def finite_values(array: np.ndarray) -> np.ndarray:

    values = np.asarray(array, dtype=np.float64)

    return values[np.isfinite(values)]


def to_json_number(value: Any) -> float | None:

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return round(number, 6)


def calculate_statistics(array: np.ndarray) -> dict[str, Any]:

    values = finite_values(array)

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
        "min": to_json_number(np.min(values)),
        "max": to_json_number(np.max(values)),
        "mean": to_json_number(np.mean(values)),
        "std": to_json_number(np.std(values)),
    }


def normalize_series(values: pd.Series) -> pd.Series:

    numeric = pd.to_numeric(
        values,
        errors="coerce"
    ).fillna(0.0)

    numeric = numeric.clip(lower=0.0)

    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if math.isclose(minimum, maximum):
        return pd.Series(
            np.zeros(len(numeric), dtype=np.float64),
            index=numeric.index,
        )

    return (
        (numeric - minimum)
        / (maximum - minimum)
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    normalized = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# FWI REFERENCE GRID
# ============================================================

def read_fwi_reference(
    raster_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError(
                "FWI raster has no CRS."
            )

        data = src.read(1).astype(np.float32)

        if src.nodata is not None:
            data[
                np.isclose(
                    data,
                    src.nodata
                )
            ] = np.nan

        data[
            ~np.isfinite(data)
        ] = np.nan

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
            "resolution_x": abs(
                float(src.transform.a)
            ),
            "resolution_y": abs(
                float(src.transform.e)
            ),
            "bounds": src.bounds,
        }

    return data, reference


# ============================================================
# GENERIC RASTER ALIGNMENT
# ============================================================

def align_to_reference(
    raster_path: Path,
    reference: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray:

    destination = np.full(
        (
            reference["height"],
            reference["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError(
                f"Raster has no CRS: {raster_path}"
            )

        source = src.read(1).astype(
            np.float32
        )

        if src.nodata is not None:
            source[
                np.isclose(
                    source,
                    src.nodata
                )
            ] = np.nan

        source[
            ~np.isfinite(source)
        ] = np.nan

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

    destination[
        ~np.isfinite(destination)
    ] = np.nan

    return destination


# ============================================================
# DEM METRIC SLOPE
# ============================================================

def metric_spacing_from_crs(
    src,
) -> tuple[float, float]:

    crs = src.crs

    xres = abs(float(src.transform.a))
    yres = abs(float(src.transform.e))

    if crs is None:
        raise ValueError(
            "DEM has no CRS."
        )

    # Projected CRS: units are normally metres.
    if crs.is_projected:

        unit_factor = 1.0

        try:
            axis_units = [
                axis.unit_name
                for axis in crs.axis_info
            ]

            if axis_units:
                first_unit = (
                    axis_units[0] or ""
                ).lower()

                if "foot" in first_unit:
                    unit_factor = 0.3048

        except Exception:
            pass

        return (
            xres * unit_factor,
            yres * unit_factor,
        )

    # Geographic CRS:
    # Convert degree resolution to metres
    # using latitude-dependent scale.
    if crs.is_geographic:

        height = src.height

        row_center = height / 2.0

        center_y = (
            src.transform.f
            + (
                row_center
                * src.transform.e
            )
        )

        lat_rad = math.radians(
            float(center_y)
        )

        meters_per_degree_lat = (
            111132.92
            - 559.82 * math.cos(
                2 * lat_rad
            )
            + 1.175 * math.cos(
                4 * lat_rad
            )
            - 0.0023 * math.cos(
                6 * lat_rad
            )
        )

        meters_per_degree_lon = (
            111412.84 * math.cos(
                lat_rad
            )
            - 93.5 * math.cos(
                3 * lat_rad
            )
            + 0.118 * math.cos(
                5 * lat_rad
            )
        )

        return (
            xres * meters_per_degree_lon,
            yres * meters_per_degree_lat,
        )

    raise ValueError(
        "Unsupported DEM CRS type."
    )


def calculate_native_dem_slope(
    dem_path: Path,
) -> np.ndarray:

    with rasterio.open(dem_path) as src:

        if src.crs is None:
            raise ValueError(
                "DEM has no CRS."
            )

        dem = src.read(1).astype(
            np.float32
        )

        if src.nodata is not None:

            dem[
                np.isclose(
                    dem,
                    src.nodata
                )
            ] = np.nan

        dem[
            ~np.isfinite(dem)
        ] = np.nan

        valid = np.isfinite(dem)

        slope = np.full(
            dem.shape,
            np.nan,
            dtype=np.float32,
        )

        if not np.any(valid):
            return slope

        x_spacing, y_spacing = (
            metric_spacing_from_crs(src)
        )

        median_elevation = float(
            np.nanmedian(dem)
        )

        dem_gradient = np.where(
            valid,
            dem,
            median_elevation,
        ).astype(np.float32)

        gradient_y, gradient_x = (
            np.gradient(
                dem_gradient,
                y_spacing,
                x_spacing,
            )
        )

        slope_radians = np.arctan(
            np.sqrt(
                np.square(gradient_x)
                + np.square(gradient_y)
            )
        )

        slope = np.degrees(
            slope_radians
        ).astype(np.float32)

        slope[~valid] = np.nan

        slope[
            ~np.isfinite(slope)
        ] = np.nan

        return slope


def read_native_dem_slope_and_align(
    dem_path: Path,
    reference: dict[str, Any],
) -> np.ndarray:

    native_slope = calculate_native_dem_slope(
        dem_path
    )

    destination = np.full(
        (
            reference["height"],
            reference["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(dem_path) as src:

        reproject(
            source=native_slope,
            destination=destination,

            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,

            dst_transform=reference["transform"],
            dst_crs=reference["crs"],
            dst_nodata=np.nan,

            resampling=Resampling.bilinear,
        )

    destination[
        ~np.isfinite(destination)
    ] = np.nan

    return destination


# ============================================================
# FUEL TABLE
# ============================================================

def load_fuel_mapping(
    excel_path: Path,
    code_column: str,
) -> tuple[dict[float, float], dict[str, Any]]:

    sheet_name = "Fuelbeds_metric"

    workbook = pd.ExcelFile(
        excel_path
    )

    if sheet_name not in workbook.sheet_names:

        raise ValueError(
            f"Sheet '{sheet_name}' not found. "
            f"Available sheets: "
            f"{workbook.sheet_names}"
        )

    dataframe = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
    )

    code_col = find_column(
        dataframe,
        [code_column],
    )

    if code_col is None:

        raise ValueError(
            f"Fuel code column '{code_column}' "
            f"not found. Available columns: "
            f"{list(dataframe.columns)}"
        )

    # --------------------------------------------------------
    # Actual Fuelbeds fields
    # --------------------------------------------------------

    woody_col = find_column(
        dataframe,
        [
            "Woody Cover (%)",
            "Woody Cover",
            "Woody_Cover",
        ],
    )

    w1_col = find_column(
        dataframe,
        [
            "W_1hLoad (Mg/ha)",
            "W_1h Load (Mg/ha)",
            "W_1hLoad",
        ],
    )

    w10_col = find_column(
        dataframe,
        [
            "W_10hLoad (Mg/ha)",
            "W_10h Load (Mg/ha)",
            "W_10hLoad",
        ],
    )

    w100_col = find_column(
        dataframe,
        [
            "W_100hLoad (Mg/ha)",
            "W_100h Load (Mg/ha)",
            "W_100hLoad",
        ],
    )

    w1000_col = find_column(
        dataframe,
        [
            "W_1000hLoad (Mg/ha)",
            "W_1000h Load (Mg/ha)",
            "W_1000hLoad",
        ],
    )

    litter_cover_col = find_column(
        dataframe,
        [
            "Litter Cover (%)",
            "Litter Cover",
            "Litter_Cover",
        ],
    )

    litter_depth_col = find_column(
        dataframe,
        [
            "L_depth (cm)",
            "L_depth",
            "Litter Depth (cm)",
        ],
    )

    if w1_col is None:
        raise ValueError(
            "W_1hLoad column was not found "
            "in Fuelbeds_metric."
        )

    if not any(
        col is not None
        for col in [
            w10_col,
            w100_col,
            w1000_col,
        ]
    ):
        raise ValueError(
            "No W_10h/W_100h/W_1000h "
            "columns were found."
        )

    working = dataframe.copy()

    working[code_col] = pd.to_numeric(
        working[code_col],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Fine fuel
    # --------------------------------------------------------

    fine_fuel = normalize_series(
        working[w1_col]
    )

    # --------------------------------------------------------
    # Dead wood
    # --------------------------------------------------------

    dead_components = []

    if w10_col is not None:
        dead_components.append(
            (
                0.50,
                normalize_series(
                    working[w10_col]
                ),
            )
        )

    if w100_col is not None:
        dead_components.append(
            (
                0.30,
                normalize_series(
                    working[w100_col]
                ),
            )
        )

    if w1000_col is not None:
        dead_components.append(
            (
                0.20,
                normalize_series(
                    working[w1000_col]
                ),
            )
        )

    dead_wood = pd.Series(
        0.0,
        index=working.index,
    )

    weight_sum = 0.0

    for weight, values in dead_components:

        dead_wood += (
            weight * values
        )

        weight_sum += weight

    if weight_sum > 0:
        dead_wood /= weight_sum

    # --------------------------------------------------------
    # Woody structure
    # --------------------------------------------------------

    if woody_col is not None:

        woody = normalize_series(
            working[woody_col]
        )

    else:

        woody = pd.Series(
            0.0,
            index=working.index,
        )

    # --------------------------------------------------------
    # Litter
    # --------------------------------------------------------

    litter_components = []

    if litter_cover_col is not None:
        litter_components.append(
            normalize_series(
                working[litter_cover_col]
            )
        )

    if litter_depth_col is not None:
        litter_components.append(
            normalize_series(
                working[litter_depth_col]
            )
        )

    if litter_components:

        litter = sum(
            litter_components
        ) / len(litter_components)

    else:

        litter = pd.Series(
            0.0,
            index=working.index,
        )

    # --------------------------------------------------------
    # Preserve original model structure
    #
    # ShrubStructure and CanopyStructure are represented
    # by woody structural cover when that field is available.
    # --------------------------------------------------------

    shrub_structure = woody.copy()
    canopy_structure = woody.copy()

    working["F_Fuel"] = (

        FUEL_COMPONENT_WEIGHTS[
            "FineFuel"
        ] * fine_fuel

        + FUEL_COMPONENT_WEIGHTS[
            "DeadWood"
        ] * dead_wood

        + FUEL_COMPONENT_WEIGHTS[
            "ShrubStructure"
        ] * shrub_structure

        + FUEL_COMPONENT_WEIGHTS[
            "Litter"
        ] * litter

        + FUEL_COMPONENT_WEIGHTS[
            "CanopyStructure"
        ] * canopy_structure
    )

    working["F_Fuel"] = (
        working["F_Fuel"]
        .clip(0.0, 1.0)
    )

    working = working.dropna(
        subset=[code_col]
    )

    working = working.drop_duplicates(
        subset=[code_col],
        keep="last",
    )

    mapping = {}

    for code, score in zip(
        working[code_col],
        working["F_Fuel"],
    ):

        code_float = float(code)
        score_float = float(score)

        if (
            math.isfinite(code_float)
            and math.isfinite(score_float)
        ):
            mapping[
                code_float
            ] = score_float

    if not mapping:
        raise ValueError(
            "No valid fuel mapping was created."
        )

    metadata = {

        "sheet": sheet_name,

        "fuel_code_column": str(
            code_col
        ),

        "selected_source_columns": {
            "WoodyCover": woody_col,
            "W_1hLoad": w1_col,
            "W_10hLoad": w10_col,
            "W_100hLoad": w100_col,
            "W_1000hLoad": w1000_col,
            "LitterCover": litter_cover_col,
            "LitterDepth": litter_depth_col,
        },

        "fuel_component_weights":
            FUEL_COMPONENT_WEIGHTS,

        "mapping_entries":
            len(mapping),

        "reference_fuel_statistics":
            calculate_statistics(
                working["F_Fuel"].to_numpy()
            ),
    }

    print(
        f"Fuel mapping entries: "
        f"{len(mapping):,}"
    )

    return mapping, metadata


# ============================================================
# FUEL RASTER -> F_FUEL
# ============================================================

def create_fuel_raster(
    fuel_codes: np.ndarray,
    mapping: dict[float, float],
) -> tuple[
    np.ndarray,
    list[float],
    list[float],
]:

    result = np.full(
        fuel_codes.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(
        fuel_codes
    )

    if not np.any(valid):

        return result, [], []

    unique_codes = [
        float(code)
        for code in np.unique(
            fuel_codes[valid]
        )
    ]

    unmapped = []

    for code in unique_codes:

        mask = (
            fuel_codes == code
        )

        if code in mapping:

            result[mask] = (
                mapping[code]
            )

        else:

            unmapped.append(code)

    return (
        result,
        unique_codes,
        unmapped,
    )


# ============================================================
# OUTPUT
# ============================================================

def write_geotiff(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
) -> None:

    output = np.where(
        np.isfinite(array),
        array,
        OUTPUT_NODATA,
    ).astype(np.float32)

    out_profile = profile.copy()

    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(
        path,
        "w",
        **out_profile,
    ) as dst:

        dst.write(
            output,
            1,
        )


def write_json(
    path: Path,
    report: dict[str, Any],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        file.write("\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_arguments()

    require_file(
        args.fwi_raster,
        "FWI raster",
    )

    require_file(
        args.fuel_raster,
        "Fuel raster",
    )

    require_file(
        args.dem_raster,
        "DEM raster",
    )

    require_file(
        args.fuel_excel,
        "Fuel Excel",
    )

    if (
        args.fuel_score_column
        .strip()
        .upper()
        != "AUTO"
    ):
        raise ValueError(
            "FIRIS uses AUTO fuel scoring."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. FWI reference
    # --------------------------------------------------------

    print(
        "============================================================"
    )
    print(
        "FIRIS FIRE LIKELIHOOD INDEX"
    )
    print(
        "============================================================"
    )

    print(
        "Reading FWI reference raster..."
    )

    fwi, reference = (
        read_fwi_reference(
            args.fwi_raster
        )
    )

    print(
        f"FWI CRS       : "
        f"{reference['crs']}"
    )

    print(
        f"FWI dimensions: "
        f"{reference['width']} x "
        f"{reference['height']}"
    )

    print(
        f"FWI cell size : "
        f"{reference['resolution_x']} x "
        f"{reference['resolution_y']}"
    )

    # --------------------------------------------------------
    # 2. Fuel -> FWI grid
    # --------------------------------------------------------

    print(
        "Aligning fuel raster to FWI grid..."
    )

    fuel_codes = align_to_reference(
        args.fuel_raster,
        reference,
        Resampling.nearest,
    )

    # --------------------------------------------------------
    # 3. Native DEM slope -> FWI grid
    # --------------------------------------------------------

    print(
        "Calculating slope on native DEM grid..."
    )

    slope_degrees = (
        read_native_dem_slope_and_align(
            args.dem_raster,
            reference,
        )
    )

    # --------------------------------------------------------
    # 4. Fuel mapping
    # --------------------------------------------------------

    print(
        "Loading Fuelbeds_metric..."
    )

    fuel_mapping, fuel_metadata = (
        load_fuel_mapping(
            args.fuel_excel,
            args.fuel_code_column,
        )
    )

    print(
        "Creating F_Fuel raster..."
    )

    f_fuel, unique_codes, unmapped_codes = (
        create_fuel_raster(
            fuel_codes,
            fuel_mapping,
        )
    )

    # --------------------------------------------------------
    # 5. FWI normalization
    # --------------------------------------------------------

    f_fwi = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32,
    )

    fwi_valid = np.isfinite(fwi)

    f_fwi[fwi_valid] = np.clip(
        fwi[fwi_valid]
        / FWI_SCALE,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 6. Topography normalization
    # --------------------------------------------------------

    f_topo = np.full(
        slope_degrees.shape,
        np.nan,
        dtype=np.float32,
    )

    slope_valid = np.isfinite(
        slope_degrees
    )

    f_topo[slope_valid] = np.clip(
        slope_degrees[slope_valid]
        / SLOPE_SCALE_DEGREES,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 7. Final common valid mask
    # --------------------------------------------------------

    valid_mask = (
        np.isfinite(f_fwi)
        & np.isfinite(f_fuel)
        & np.isfinite(f_topo)
    )

    valid_count = int(
        np.sum(valid_mask)
    )

    total_count = int(
        fwi.size
    )

    if valid_count == 0:

        raise RuntimeError(
            "No common valid pixels exist "
            "between FWI, Fuel and Topography."
        )

    # --------------------------------------------------------
    # 8. FLI
    # --------------------------------------------------------

    fli = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32,
    )

    fli[valid_mask] = (
        100.0
        * (
            WEIGHT_FWI
            * f_fwi[valid_mask]

            + WEIGHT_FUEL
            * f_fuel[valid_mask]

            + WEIGHT_TOPO
            * f_topo[valid_mask]
        )
    )

    fli = np.clip(
        fli,
        0.0,
        100.0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # 9. Outputs
    # --------------------------------------------------------

    run_date = args.run_date

    output_fwi = (
        args.output_dir
        / f"f_fwi_fars_{run_date}.tif"
    )

    output_fuel = (
        args.output_dir
        / f"f_fuel_fars_{run_date}.tif"
    )

    output_slope = (
        args.output_dir
        / f"slope_fars_{run_date}.tif"
    )

    output_topo = (
        args.output_dir
        / f"f_topo_fars_{run_date}.tif"
    )

    output_fli = (
        args.output_dir
        / f"fli_fars_{run_date}.tif"
    )

    output_report = (
        args.output_dir
        / f"firis_report_{run_date}.json"
    )

    print(
        "Writing component rasters..."
    )

    write_geotiff(
        output_fwi,
        f_fwi,
        reference["profile"],
    )

    write_geotiff(
        output_fuel,
        f_fuel,
        reference["profile"],
    )

    write_geotiff(
        output_slope,
        slope_degrees,
        reference["profile"],
    )

    write_geotiff(
        output_topo,
        f_topo,
        reference["profile"],
    )

    print(
        "Writing final FLI..."
    )

    write_geotiff(
        output_fli,
        fli,
        reference["profile"],
    )

    # --------------------------------------------------------
    # 10. Report
    # --------------------------------------------------------

    report = {

        "project":
            "FIRIS - Fars Integrated Fire Information System",

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "run_date":
            run_date,

        "formula":
            "FLI = 100 * "
            "(0.45 * F_FWI + "
            "0.35 * F_Fuel + "
            "0.20 * F_Topo)",

        "weights": {
            "F_FWI": WEIGHT_FWI,
            "F_Fuel": WEIGHT_FUEL,
            "F_Topo": WEIGHT_TOPO,
        },

        "normalization": {

            "F_FWI":
                "clip(FWI / 100, 0, 1)",

            "F_Fuel":
                "Weighted normalized Fuelbeds_metric variables",

            "F_Topo":
                "clip(slope_degrees / 45, 0, 1)",
        },

        "alignment": {

            "reference":
                "FWI",

            "fuel":
                "Nearest neighbour to FWI grid",

            "dem":
                "Native DEM slope calculation, "
                "then bilinear resampling to FWI grid",

            "final_grid":
                "FWI grid",
        },

        "target_grid": {

            "crs":
                str(reference["crs"]),

            "width":
                int(reference["width"]),

            "height":
                int(reference["height"]),

            "cell_size_x":
                float(
                    reference["resolution_x"]
                ),

            "cell_size_y":
                float(
                    reference["resolution_y"]
                ),

            "transform":
                [
                    float(value)
                    for value in
                    reference["transform"][:6]
                ],
        },

        "inputs": {

            "fwi_raster":
                str(args.fwi_raster),

            "fuel_raster":
                str(args.fuel_raster),

            "dem_raster":
                str(args.dem_raster),

            "fuel_excel":
                str(args.fuel_excel),
        },

        "fuel_mapping": {

            "unique_codes":
                unique_codes,

            "unique_code_count":
                len(unique_codes),

            "mapped_code_count":
                len(unique_codes)
                - len(unmapped_codes),

            "unmapped_codes":
                unmapped_codes,

            "unmapped_code_count":
                len(unmapped_codes),

            **fuel_metadata,
        },

        "statistics": {

            "FWI":
                calculate_statistics(fwi),

            "F_FWI":
                calculate_statistics(f_fwi),

            "F_Fuel":
                calculate_statistics(f_fuel),

            "Slope_degrees":
                calculate_statistics(
                    slope_degrees
                ),

            "F_Topo":
                calculate_statistics(f_topo),

            "FLI":
                calculate_statistics(fli),

            "common_valid_pixels":
                valid_count,

            "total_target_pixels":
                total_count,

            "valid_percentage":
                round(
                    100.0
                    * valid_count
                    / total_count,
                    4,
                ),
        },

        "outputs": {

            "f_fwi":
                str(output_fwi),

            "f_fuel":
                str(output_fuel),

            "slope":
                str(output_slope),

            "f_topo":
                str(output_topo),

            "fli":
                str(output_fli),
        },
    }

    write_json(
        output_report,
        report,
    )

    print(
        "============================================================"
    )

    print(
        "FIRIS build completed successfully."
    )

    print(
        f"Target pixels : "
        f"{total_count:,}"
    )

    print(
        f"Valid pixels  : "
        f"{valid_count:,}"
    )

    print(
        f"Valid percent : "
        f"{100.0 * valid_count / total_count:.2f}%"
    )

    print(
        f"FLI output    : "
        f"{output_fli}"
    )

    print(
        f"Report        : "
        f"{output_report}"
    )

    print(
        f"Unmapped fuel codes: "
        f"{len(unmapped_codes):,}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
