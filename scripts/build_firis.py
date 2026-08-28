
#!/usr/bin/env python3

"""
FIRIS - Fars Integrated Fire Information System

Fire Likelihood Index:

    FLI = 100 * (
        0.45 * F_FWI
        + 0.35 * F_Fuel
        + 0.20 * F_Topo
    )

Spatial rule
------------
FWI is the final reference grid.

Fuel:
    categorical raster
    -> nearest-neighbour
    -> FWI grid

DEM:
    native DEM
    -> calculate slope on native resolution
    -> metric cell spacing
    -> bilinear resampling
    -> FWI grid

This prevents calculating slope after degrading
the original DEM to the coarse FWI grid.
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
# MODEL
# ============================================================

FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

FWI_MAX = 100.0
SLOPE_REFERENCE = 45.0

OUTPUT_NODATA = -9999.0


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index"
    )

    parser.add_argument(
        "--fwi-raster",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--fuel-raster",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--dem-raster",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--fuel-excel",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--fuel-code-column",
        default="JOIN_VALUE",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--run-date",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# GENERAL HELPERS
# ============================================================

def require_file(path: Path, label: str):

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def clean_array(
    array: np.ndarray,
    nodata: Any = None,
) -> np.ndarray:

    result = np.asarray(
        array,
        dtype=np.float32,
    ).copy()

    if nodata is not None:

        try:

            if np.isnan(nodata):

                result[
                    np.isnan(result)
                ] = np.nan

            else:

                result[
                    np.isclose(
                        result,
                        float(nodata),
                    )
                ] = np.nan

        except (TypeError, ValueError):

            pass

    result[
        ~np.isfinite(result)
    ] = np.nan

    return result


def stats(array: np.ndarray):

    valid = array[
        np.isfinite(array)
    ]

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
        "min": round(float(np.min(valid)), 6),
        "max": round(float(np.max(valid)), 6),
        "mean": round(float(np.mean(valid)), 6),
        "std": round(float(np.std(valid)), 6),
    }


# ============================================================
# FWI REFERENCE
# ============================================================

def read_fwi(
    path: Path,
):

    with rasterio.open(path) as src:

        if src.crs is None:

            raise ValueError(
                "FWI raster has no CRS."
            )

        data = clean_array(
            src.read(1),
            src.nodata,
        )

        profile = src.profile.copy()

        reference = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": profile,
            "bounds": src.bounds,
            "res": src.res,
        }

    print("")
    print("FWI REFERENCE GRID")
    print("------------------")
    print(f"CRS        : {reference['crs']}")
    print(f"Width      : {reference['width']}")
    print(f"Height     : {reference['height']}")
    print(f"Cell size  : {reference['res']}")
    print(f"Bounds     : {reference['bounds']}")
    print(f"Statistics : {stats(data)}")

    return data, reference


# ============================================================
# ALIGN RASTER TO FWI GRID
# ============================================================

def align_to_fwi(
    path: Path,
    reference: dict,
    resampling: Resampling,
):

    destination = np.full(
        (
            reference["height"],
            reference["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(path) as src:

        if src.crs is None:

            raise ValueError(
                f"Raster has no CRS: {path}"
            )

        source = clean_array(
            src.read(1),
            src.nodata,
        )

        print("")
        print(f"Aligning: {path}")
        print(f"Source CRS      : {src.crs}")
        print(f"Source size     : {src.width} x {src.height}")
        print(f"Source cell     : {src.res}")
        print(f"Target CRS      : {reference['crs']}")
        print(
            "Target size     : "
            f"{reference['width']} x "
            f"{reference['height']}"
        )
        print(
            f"Resampling      : {resampling.name}"
        )

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

    print(
        f"Aligned statistics: "
        f"{stats(destination)}"
    )

    return destination


# ============================================================
# DEM -> SLOPE
# ============================================================

def metric_cell_size(src):

    if src.crs is None:

        raise ValueError(
            "DEM CRS is missing."
        )

    xres = abs(float(src.transform.a))
    yres = abs(float(src.transform.e))

    # Projected CRS
    if src.crs.is_projected:

        factor = 1.0

        try:

            units = [
                axis.unit_name
                for axis in src.crs.axis_info
            ]

            if units:

                unit = (
                    units[0] or ""
                ).lower()

                if "foot" in unit:

                    factor = 0.3048

        except Exception:

            pass

        return (
            xres * factor,
            yres * factor,
        )

    # Geographic CRS
    if src.crs.is_geographic:

        center_row = src.height / 2.0

        latitude = (
            src.transform.f
            + center_row
            * src.transform.e
        )

        lat = math.radians(
            float(latitude)
        )

        meters_lat = (
            111132.92
            - 559.82 * math.cos(2 * lat)
            + 1.175 * math.cos(4 * lat)
            - 0.0023 * math.cos(6 * lat)
        )

        meters_lon = (
            111412.84 * math.cos(lat)
            - 93.5 * math.cos(3 * lat)
            + 0.118 * math.cos(5 * lat)
        )

        return (
            xres * meters_lon,
            yres * meters_lat,
        )

    raise ValueError(
        "Unsupported DEM coordinate system."
    )


def calculate_native_slope(
    dem_path: Path,
):

    print("")
    print("CALCULATING SLOPE ON NATIVE DEM")
    print("--------------------------------")

    with rasterio.open(dem_path) as src:

        if src.crs is None:

            raise ValueError(
                "DEM raster has no CRS."
            )

        dem = clean_array(
            src.read(1),
            src.nodata,
        )

        valid = np.isfinite(dem)

        if not np.any(valid):

            raise ValueError(
                "DEM contains no valid pixels."
            )

        dx, dy = metric_cell_size(src)

        print(f"DEM CRS       : {src.crs}")
        print(f"DEM size      : {src.width} x {src.height}")
        print(f"DEM cell      : {src.res}")
        print(
            f"Metric spacing: "
            f"X={dx:.3f} m, Y={dy:.3f} m"
        )

        # Fill only for numerical gradient calculation.
        # Original NoData pixels are restored afterwards.
        fill_value = float(
            np.nanmedian(dem)
        )

        working = np.where(
            valid,
            dem,
            fill_value,
        ).astype(np.float32)

        gradient_y, gradient_x = np.gradient(
            working,
            dy,
            dx,
        )

        slope_rad = np.arctan(
            np.sqrt(
                gradient_x ** 2
                + gradient_y ** 2
            )
        )

        slope = np.degrees(
            slope_rad
        ).astype(np.float32)

        slope[~valid] = np.nan

        slope[
            ~np.isfinite(slope)
        ] = np.nan

        print(
            f"Native slope statistics: "
            f"{stats(slope)}"
        )

        return slope, src.transform, src.crs


def align_slope_to_fwi(
    dem_path: Path,
    reference: dict,
):

    slope, dem_transform, dem_crs = (
        calculate_native_slope(
            dem_path
        )
    )

    destination = np.full(
        (
            reference["height"],
            reference["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    print("")
    print("ALIGNING SLOPE TO FWI GRID")
    print("--------------------------")

    reproject(
        source=slope,
        destination=destination,

        src_transform=dem_transform,
        src_crs=dem_crs,
        src_nodata=np.nan,

        dst_transform=reference["transform"],
        dst_crs=reference["crs"],
        dst_nodata=np.nan,

        resampling=Resampling.bilinear,
    )

    destination[
        ~np.isfinite(destination)
    ] = np.nan

    print(
        f"FWI-grid slope statistics: "
        f"{stats(destination)}"
    )

    return destination


# ============================================================
# FUEL EXCEL
# ============================================================

def find_column(
    dataframe,
    candidates,
):

    lookup = {
        str(c).strip().lower(): c
        for c in dataframe.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lookup:

            return lookup[key]

    return None


def normalize_column(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    values = values.clip(
        lower=0.0
    )

    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )

    if math.isclose(
        minimum,
        maximum,
    ):

        return pd.Series(
            np.zeros(
                len(values),
                dtype=np.float64,
            ),
            index=values.index,
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def load_fuel_mapping(
    excel_path: Path,
    requested_code_column: str,
):

    print("")
    print("LOADING FUEL TABLE")
    print("------------------")

    workbook = pd.ExcelFile(
        excel_path
    )

    if "Fuelbeds_metric" not in (
        workbook.sheet_names
    ):

        raise ValueError(
            "Sheet 'Fuelbeds_metric' "
            "not found. Available sheets: "
            f"{workbook.sheet_names}"
        )

    df = pd.read_excel(
        excel_path,
        sheet_name="Fuelbeds_metric",
    )

    print(
        "Fuel columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    code_col = find_column(
        df,
        [
            requested_code_column,
            "JOIN_VALUE",
            "Join_Value",
            "FUELBED",
            "Fuelbed",
            "FUELBED_ID",
            "Fuelbed_ID",
            "FUEL_CODE",
            "Fuel_Code",
        ],
    )

    if code_col is None:

        raise ValueError(
            "Could not identify the fuel-code "
            "column in Fuelbeds_metric."
        )

    woody_col = find_column(
        df,
        [
            "Woody Cover (%)",
            "Woody Cover",
            "Woody_Cover",
        ],
    )

    w1_col = find_column(
        df,
        [
            "W_1hLoad (Mg/ha)",
            "W_1h Load (Mg/ha)",
            "W_1hLoad",
        ],
    )

    w10_col = find_column(
        df,
        [
            "W_10hLoad (Mg/ha)",
            "W_10h Load (Mg/ha)",
            "W_10hLoad",
        ],
    )

    w100_col = find_column(
        df,
        [
            "W_100hLoad (Mg/ha)",
            "W_100h Load (Mg/ha)",
            "W_100hLoad",
        ],
    )

    w1000_col = find_column(
        df,
        [
            "W_1000hLoad (Mg/ha)",
            "W_1000h Load (Mg/ha)",
            "W_1000hLoad",
        ],
    )

    litter_cover_col = find_column(
        df,
        [
            "Litter Cover (%)",
            "Litter Cover",
            "Litter_Cover",
        ],
    )

    litter_depth_col = find_column(
        df,
        [
            "L_depth (cm)",
            "L_depth",
            "Litter Depth (cm)",
        ],
    )

    if w1_col is None:

        raise ValueError(
            "W_1hLoad column was not found."
        )

    df[code_col] = pd.to_numeric(
        df[code_col],
        errors="coerce",
    )

    fine = normalize_column(
        df[w1_col]
    )

    dead_parts = []

    if w10_col is not None:

        dead_parts.append(
            (
                0.50,
                normalize_column(
                    df[w10_col]
                ),
            )
        )

    if w100_col is not None:

        dead_parts.append(
            (
                0.30,
                normalize_column(
                    df[w100_col]
                ),
            )
        )

    if w1000_col is not None:

        dead_parts.append(
            (
                0.20,
                normalize_column(
                    df[w1000_col]
                ),
            )
        )

    dead = pd.Series(
        0.0,
        index=df.index,
    )

    total_weight = 0.0

    for weight, values in dead_parts:

        dead += weight * values
        total_weight += weight

    if total_weight > 0:

        dead /= total_weight

    if woody_col is not None:

        woody = normalize_column(
            df[woody_col]
        )

    else:

        woody = pd.Series(
            0.0,
            index=df.index,
        )

    litter_parts = []

    if litter_cover_col is not None:

        litter_parts.append(
            normalize_column(
                df[litter_cover_col]
            )
        )

    if litter_depth_col is not None:

        litter_parts.append(
            normalize_column(
                df[litter_depth_col]
            )
        )

    if litter_parts:

        litter = (
            sum(litter_parts)
            / len(litter_parts)
        )

    else:

        litter = pd.Series(
            0.0,
            index=df.index,
        )

    fuel_score = (

        0.35 * fine
        + 0.30 * dead
        + 0.15 * woody
        + 0.10 * litter
        + 0.10 * woody
    )

    fuel_score = fuel_score.clip(
        0.0,
        1.0,
    )

    df["_F_Fuel"] = fuel_score

    df = df.dropna(
        subset=[code_col]
    )

    df = df.drop_duplicates(
        subset=[code_col],
        keep="last",
    )

    mapping = {}

    for code, score in zip(
        df[code_col],
        df["_F_Fuel"],
    ):

        try:

            code_value = float(code)
            score_value = float(score)

            if (
                math.isfinite(code_value)
                and math.isfinite(score_value)
            ):

                mapping[
                    code_value
                ] = score_value

        except (
            TypeError,
            ValueError,
        ):

            continue

    if not mapping:

        raise ValueError(
            "Fuel mapping is empty."
        )

    print(
        f"Fuel-code column: {code_col}"
    )

    print(
        f"Fuel mapping entries: "
        f"{len(mapping)}"
    )

    return mapping


# ============================================================
# FUEL RASTER -> SCORE
# ============================================================

def fuel_to_score(
    fuel_codes,
    mapping,
):

    output = np.full(
        fuel_codes.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(
        fuel_codes
    )

    if not np.any(valid):

        raise ValueError(
            "Aligned Fuel raster contains "
            "no valid pixels."
        )

    unique_codes = np.unique(
        fuel_codes[valid]
    )

    unmapped = []

    for code in unique_codes:

        code_float = float(code)

        mask = (
            fuel_codes == code
        )

        if code_float in mapping:

            output[mask] = (
                mapping[code_float]
            )

        else:

            unmapped.append(
                code_float
            )

    print("")
    print("FUEL MAPPING")
    print("------------")
    print(
        f"Unique raster codes : "
        f"{len(unique_codes)}"
    )

    print(
        f"Mapped codes        : "
        f"{len(unique_codes) - len(unmapped)}"
    )

    print(
        f"Unmapped codes      : "
        f"{len(unmapped)}"
    )

    if unmapped:

        print(
            "First unmapped codes:"
        )

        for code in unmapped[:20]:

            print(
                f"  - {code}"
            )

    print(
        f"F_Fuel statistics   : "
        f"{stats(output)}"
    )

    return output, unmapped


# ============================================================
# WRITE RASTER
# ============================================================

def write_raster(
    path: Path,
    array: np.ndarray,
    reference: dict,
):

    profile = reference[
        "profile"
    ].copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        width=reference["width"],
        height=reference["height"],
        crs=reference["crs"],
        transform=reference["transform"],
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
    )

    output = np.where(
        np.isfinite(array),
        array,
        OUTPUT_NODATA,
    ).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            output,
            1,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("=" * 70)
    print("FIRIS BUILD START")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. FWI
    # --------------------------------------------------------

    fwi, reference = read_fwi(
        args.fwi_raster
    )

    # --------------------------------------------------------
    # 2. Fuel -> FWI grid
    # --------------------------------------------------------

    fuel_codes = align_to_fwi(
        args.fuel_raster,
        reference,
        Resampling.nearest,
    )

    # --------------------------------------------------------
    # 3. DEM native slope -> FWI grid
    # --------------------------------------------------------

    slope = align_slope_to_fwi(
        args.dem_raster,
        reference,
    )

    # --------------------------------------------------------
    # 4. Fuel Excel
    # --------------------------------------------------------

    fuel_mapping = load_fuel_mapping(
        args.fuel_excel,
        args.fuel_code_column,
    )

    # --------------------------------------------------------
    # 5. Fuel score
    # --------------------------------------------------------

    f_fuel, unmapped_codes = (
        fuel_to_score(
            fuel_codes,
            fuel_mapping,
        )
    )

    # --------------------------------------------------------
    # 6. FWI normalization
    # --------------------------------------------------------

    f_fwi = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32,
    )

    fwi_valid = np.isfinite(
        fwi
    )

    f_fwi[fwi_valid] = np.clip(
        fwi[fwi_valid]
        / FWI_MAX,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 7. Topography normalization
    # --------------------------------------------------------

    f_topo = np.full(
        slope.shape,
        np.nan,
        dtype=np.float32,
    )

    slope_valid = np.isfinite(
        slope
    )

    f_topo[slope_valid] = np.clip(
        slope[slope_valid]
        / SLOPE_REFERENCE,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 8. Common valid pixels
    # --------------------------------------------------------

    valid = (
        np.isfinite(f_fwi)
        & np.isfinite(f_fuel)
        & np.isfinite(f_topo)
    )

    valid_count = int(
        np.sum(valid)
    )

    total_count = int(
        fwi.size
    )

    print("")
    print("COMMON GRID VALIDATION")
    print("----------------------")
    print(
        f"Total target pixels : "
        f"{total_count:,}"
    )

    print(
        f"Valid common pixels : "
        f"{valid_count:,}"
    )

    print(
        f"Valid percentage    : "
        f"{100.0 * valid_count / total_count:.2f}%"
    )

    if valid_count == 0:

        raise RuntimeError(
            "There are no common valid pixels "
            "between FWI, Fuel and Topography."
        )

    # --------------------------------------------------------
    # 9. Final FLI
    # --------------------------------------------------------

    fli = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32,
    )

    fli[valid] = (

        100.0
        * (
            FWI_WEIGHT
            * f_fwi[valid]

            + FUEL_WEIGHT
            * f_fuel[valid]

            + TOPO_WEIGHT
            * f_topo[valid]
        )
    )

    fli = np.clip(
        fli,
        0.0,
        100.0,
    )

    print("")
    print("FINAL FLI")
    print("---------")
    print(
        f"Statistics: {stats(fli)}"
    )

    # --------------------------------------------------------
    # 10. Output paths
    # --------------------------------------------------------

    date = args.run_date

    f_fwi_path = (
        args.output_dir
        / f"f_fwi_fars_{date}.tif"
    )

    f_fuel_path = (
        args.output_dir
        / f"f_fuel_fars_{date}.tif"
    )

    slope_path = (
        args.output_dir
        / f"slope_fars_{date}.tif"
    )

    f_topo_path = (
        args.output_dir
        / f"f_topo_fars_{date}.tif"
    )

    fli_path = (
        args.output_dir
        / f"fli_fars_{date}.tif"
    )

    report_path = (
        args.output_dir
        / f"firis_report_{date}.json"
    )

    # --------------------------------------------------------
    # 11. Write rasters
    # --------------------------------------------------------

    print("")
    print("WRITING OUTPUTS")
    print("----------------")

    write_raster(
        f_fwi_path,
        f_fwi,
        reference,
    )

    write_raster(
        f_fuel_path,
        f_fuel,
        reference,
    )

    write_raster(
        slope_path,
        slope,
        reference,
    )

    write_raster(
        f_topo_path,
        f_topo,
        reference,
    )

    write_raster(
        fli_path,
        fli,
        reference,
    )

    # --------------------------------------------------------
    # 12. Report
    # --------------------------------------------------------

    report = {

        "project":
            "FIRIS - Fars Integrated Fire Information System",

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "run_date":
            date,

        "formula":
            "FLI = 100 * "
            "(0.45 * F_FWI + "
            "0.35 * F_Fuel + "
            "0.20 * F_Topo)",

        "weights": {
            "F_FWI": FWI_WEIGHT,
            "F_Fuel": FUEL_WEIGHT,
            "F_Topo": TOPO_WEIGHT,
        },

        "target_grid": {

            "reference":
                "FWI",

            "crs":
                str(reference["crs"]),

            "width":
                int(reference["width"]),

            "height":
                int(reference["height"]),

            "cell_size_x":
                float(reference["res"][0]),

            "cell_size_y":
                float(reference["res"][1]),
        },

        "alignment": {

            "FWI":
                "reference grid",

            "Fuel":
                "nearest-neighbour to FWI",

            "DEM":
                "native slope calculation "
                "then bilinear to FWI",
        },

        "normalization": {

            "F_FWI":
                "clip(FWI / 100, 0, 1)",

            "F_Fuel":
                "Fuelbeds_metric weighted score",

            "F_Topo":
                "clip(slope_degrees / 45, 0, 1)",
        },

        "statistics": {

            "FWI":
                stats(fwi),

            "F_FWI":
                stats(f_fwi),

            "F_Fuel":
                stats(f_fuel),

            "Slope_degrees":
                stats(slope),

            "F_Topo":
                stats(f_topo),

            "FLI":
                stats(fli),

            "total_pixels":
                total_count,

            "common_valid_pixels":
                valid_count,

            "common_valid_percent":
                round(
                    100.0
                    * valid_count
                    / total_count,
                    4,
                ),
        },

        "fuel": {

            "unmapped_code_count":
                len(unmapped_codes),

            "unmapped_codes":
                unmapped_codes[:100],
        },

        "inputs": {

            "FWI":
                str(args.fwi_raster),

            "Fuel":
                str(args.fuel_raster),

            "DEM":
                str(args.dem_raster),

            "FuelExcel":
                str(args.fuel_excel),
        },

        "outputs": {

            "F_FWI":
                str(f_fwi_path),

            "F_Fuel":
                str(f_fuel_path),

            "Slope":
                str(slope_path),

            "F_Topo":
                str(f_topo_path),

            "FLI":
                str(fli_path),
        },
    }

    with report_path.open(
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

    print("")
    print("=" * 70)
    print("FIRIS BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"FLI output: {fli_path}"
    )

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":

    main()
