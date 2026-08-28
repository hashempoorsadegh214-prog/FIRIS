#!/usr/bin/env python3

"""
FIRIS - Fars Integrated Fire Information System

FLI = 100 * (
    0.45 * F_FWI
    + 0.35 * F_Fuel
    + 0.20 * F_Topo
)

Spatial rules:
- FWI is the reference grid.
- Fuel -> FWI grid using nearest neighbour.
- Slope is calculated on native DEM first, then aligned to FWI.
- All final calculations are restricted to fars.geojson.
- NoData is never artificially filled.
- Coverage inside Fars is explicitly reported.
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
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_geom


FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

FWI_MAX = 100.0
SLOPE_REFERENCE = 45.0
OUTPUT_NODATA = -9999.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index"
    )

    parser.add_argument("--fwi-raster", required=True, type=Path)
    parser.add_argument("--fuel-raster", required=True, type=Path)
    parser.add_argument("--dem-raster", required=True, type=Path)
    parser.add_argument("--fuel-excel", required=True, type=Path)

    parser.add_argument(
        "--fuel-code-column",
        default="JOIN_VALUE"
    )

    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("fars.geojson")
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--run-date",
        required=True
    )

    return parser.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def clean_array(
    array: np.ndarray,
    nodata: Any = None
) -> np.ndarray:

    result = np.asarray(
        array,
        dtype=np.float32
    ).copy()

    if nodata is not None:

        try:

            if np.isnan(nodata):
                result[np.isnan(result)] = np.nan

            else:
                result[
                    np.isclose(
                        result,
                        float(nodata)
                    )
                ] = np.nan

        except (TypeError, ValueError):
            pass

    result[~np.isfinite(result)] = np.nan

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
            "std": None
        }

    return {
        "count": int(valid.size),
        "min": round(float(np.min(valid)), 6),
        "max": round(float(np.max(valid)), 6),
        "mean": round(float(np.mean(valid)), 6),
        "std": round(float(np.std(valid)), 6)
    }


def bounds_dict(bounds):

    return {
        "left": float(bounds.left),
        "bottom": float(bounds.bottom),
        "right": float(bounds.right),
        "top": float(bounds.top)
    }


def raster_metadata(path: Path):

    with rasterio.open(path) as src:

        return {
            "crs": str(src.crs) if src.crs else None,
            "width": int(src.width),
            "height": int(src.height),
            "cell_size_x": float(src.res[0]),
            "cell_size_y": float(src.res[1]),
            "bounds": bounds_dict(src.bounds),
            "nodata": (
                None
                if src.nodata is None
                else float(src.nodata)
            )
        }


# ============================================================
# FARS BOUNDARY
# ============================================================

def load_boundary_mask(
    boundary_path: Path,
    reference: dict
):

    with boundary_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        geojson = json.load(f)

    features = geojson.get(
        "features",
        []
    )

    if not features:
        raise ValueError(
            f"Boundary contains no features: {boundary_path}"
        )

    geometries = []

    for feature in features:

        geometry = feature.get(
            "geometry"
        )

        if geometry:
            geometries.append(
                geometry
            )

    if not geometries:
        raise ValueError(
            f"Boundary contains no geometries: {boundary_path}"
        )

    source_crs = "EPSG:4326"

    crs_obj = geojson.get("crs")

    if isinstance(crs_obj, dict):

        props = crs_obj.get(
            "properties",
            {}
        )

        name = (
            props.get("name")
            or props.get("href")
        )

        if isinstance(name, str) and name:
            source_crs = name

    target_crs = reference["crs"]

    if str(target_crs) != source_crs:

        geometries = [
            transform_geom(
                source_crs,
                target_crs,
                geom,
                precision=12
            )
            for geom in geometries
        ]

    mask = geometry_mask(
        geometries,
        out_shape=(
            reference["height"],
            reference["width"]
        ),
        transform=reference["transform"],
        invert=True,
        all_touched=False
    )

    count = int(
        np.sum(mask)
    )

    if count == 0:
        raise ValueError(
            "Fars boundary does not overlap the FWI grid."
        )

    print()
    print("FARS BOUNDARY")
    print("-------------")
    print(
        f"Boundary file       : {boundary_path}"
    )
    print(
        f"Boundary CRS        : {source_crs}"
    )
    print(
        f"Pixels inside Fars  : {count:,}"
    )

    return mask


# ============================================================
# FWI
# ============================================================

def read_fwi(path: Path):

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError(
                "FWI raster has no CRS."
            )

        data = clean_array(
            src.read(1),
            src.nodata
        )

        reference = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": src.profile.copy(),
            "bounds": src.bounds,
            "res": src.res
        }

    print()
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
# ALIGN RASTER
# ============================================================

def align_to_fwi(
    path: Path,
    reference: dict,
    resampling: Resampling
):

    destination = np.full(
        (
            reference["height"],
            reference["width"]
        ),
        np.nan,
        dtype=np.float32
    )

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError(
                f"Raster has no CRS: {path}"
            )

        source = clean_array(
            src.read(1),
            src.nodata
        )

        print()
        print(f"Aligning: {path}")
        print(f"Source CRS      : {src.crs}")
        print(
            f"Source size     : "
            f"{src.width} x {src.height}"
        )
        print(
            f"Source cell     : {src.res}"
        )
        print(
            f"Source bounds   : {src.bounds}"
        )
        print(
            f"Target CRS      : {reference['crs']}"
        )
        print(
            f"Target size     : "
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
            resampling=resampling
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
# DEM / SLOPE
# ============================================================

def metric_cell_size(src):

    if src.crs is None:
        raise ValueError(
            "DEM CRS is missing."
        )

    xres = abs(
        float(src.transform.a)
    )

    yres = abs(
        float(src.transform.e)
    )

    if src.crs.is_projected:

        return xres, yres

    if src.crs.is_geographic:

        center_row = src.height / 2.0

        latitude = (
            src.transform.f
            + center_row * src.transform.e
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
            yres * meters_lat
        )

    raise ValueError(
        "Unsupported DEM coordinate system."
    )


def calculate_native_slope(
    dem_path: Path
):

    print()
    print(
        "CALCULATING SLOPE ON NATIVE DEM"
    )
    print(
        "--------------------------------"
    )

    with rasterio.open(dem_path) as src:

        if src.crs is None:
            raise ValueError(
                "DEM raster has no CRS."
            )

        dem = clean_array(
            src.read(1),
            src.nodata
        )

        valid = np.isfinite(dem)

        if not np.any(valid):
            raise ValueError(
                "DEM contains no valid pixels."
            )

        dx, dy = metric_cell_size(src)

        print(
            f"DEM CRS       : {src.crs}"
        )
        print(
            f"DEM size      : "
            f"{src.width} x {src.height}"
        )
        print(
            f"DEM cell      : {src.res}"
        )
        print(
            f"Metric spacing: "
            f"X={dx:.3f} m, Y={dy:.3f} m"
        )

        fill_value = float(
            np.nanmedian(dem)
        )

        working = np.where(
            valid,
            dem,
            fill_value
        ).astype(np.float32)

        gradient_y, gradient_x = np.gradient(
            working,
            dy,
            dx
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
        slope[~np.isfinite(slope)] = np.nan

        print(
            f"Native slope statistics: "
            f"{stats(slope)}"
        )

        return (
            slope,
            src.transform,
            src.crs
        )


def align_slope_to_fwi(
    dem_path: Path,
    reference: dict
):

    slope, dem_transform, dem_crs = (
        calculate_native_slope(
            dem_path
        )
    )

    destination = np.full(
        (
            reference["height"],
            reference["width"]
        ),
        np.nan,
        dtype=np.float32
    )

    print()
    print(
        "ALIGNING SLOPE TO FWI GRID"
    )
    print(
        "--------------------------"
    )

    reproject(
        source=slope,
        destination=destination,
        src_transform=dem_transform,
        src_crs=dem_crs,
        src_nodata=np.nan,
        dst_transform=reference["transform"],
        dst_crs=reference["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
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
# FUEL
# ============================================================

def find_column(
    dataframe,
    candidates
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

    values = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )

    if math.isclose(
        minimum,
        maximum
    ):

        return pd.Series(
            np.zeros(
                len(values),
                dtype=np.float64
            ),
            index=values.index
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def load_fuel_mapping(
    excel_path: Path,
    requested_code_column: str
):

    print()
    print("LOADING FUEL TABLE")
    print("------------------")

    workbook = pd.ExcelFile(
        excel_path
    )

    if "Fuelbeds_metric" not in workbook.sheet_names:

        raise ValueError(
            "Sheet 'Fuelbeds_metric' not found. "
            f"Available sheets: {workbook.sheet_names}"
        )

    df = pd.read_excel(
        excel_path,
        sheet_name="Fuelbeds_metric"
    )

    print("Fuel columns:")

    for column in df.columns:
        print(f"  - {column}")

    code_col = find_column(
        df,
        [
            requested_code_column,
            "JOIN_VALUE",
            "FUELBED",
            "FUELBED_ID",
            "FUEL_CODE"
        ]
    )

    if code_col is None:

        raise ValueError(
            "Could not identify fuel-code column."
        )

    woody_col = find_column(
        df,
        [
            "Woody Cover (%)",
            "Woody Cover"
        ]
    )

    w1_col = find_column(
        df,
        [
            "W_1hLoad (Mg/ha)",
            "W_1h Load (Mg/ha)",
            "W_1hLoad"
        ]
    )

    w10_col = find_column(
        df,
        [
            "W_10hLoad (Mg/ha)",
            "W_10h Load (Mg/ha)",
            "W_10hLoad"
        ]
    )

    w100_col = find_column(
        df,
        [
            "W_100hLoad (Mg/ha)",
            "W_100h Load (Mg/ha)",
            "W_100hLoad"
        ]
    )

    w1000_col = find_column(
        df,
        [
            "W_1000hLoad (Mg/ha)",
            "W_1000h Load (Mg/ha)",
            "W_1000hLoad"
        ]
    )

    litter_cover_col = find_column(
        df,
        [
            "Litter Cover (%)",
            "Litter Cover"
        ]
    )

    litter_depth_col = find_column(
        df,
        [
            "L_depth (cm)",
            "L_depth"
        ]
    )

    if w1_col is None:

        raise ValueError(
            "W_1hLoad column was not found."
        )

    df[code_col] = pd.to_numeric(
        df[code_col],
        errors="coerce"
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
                )
            )
        )

    if w100_col is not None:

        dead_parts.append(
            (
                0.30,
                normalize_column(
                    df[w100_col]
                )
            )
        )

    if w1000_col is not None:

        dead_parts.append(
            (
                0.20,
                normalize_column(
                    df[w1000_col]
                )
            )
        )

    dead = pd.Series(
        0.0,
        index=df.index
    )

    total_weight = 0.0

    for weight, values in dead_parts:

        dead += (
            weight * values
        )

        total_weight += weight

    if total_weight > 0:
        dead /= total_weight

    woody = (
        normalize_column(
            df[woody_col]
        )
        if woody_col is not None
        else pd.Series(
            0.0,
            index=df.index
        )
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
            index=df.index
        )

    fuel_score = (
        0.35 * fine
        + 0.30 * dead
        + 0.15 * woody
        + 0.10 * litter
        + 0.10 * woody
    ).clip(0.0, 1.0)

    df["_F_Fuel"] = fuel_score

    df = (
        df
        .dropna(subset=[code_col])
        .drop_duplicates(
            subset=[code_col],
            keep="last"
        )
    )

    mapping = {}

    for code, score in zip(
        df[code_col],
        df["_F_Fuel"]
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
            ValueError
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


def fuel_to_score(
    fuel_codes,
    mapping
):

    output = np.full(
        fuel_codes.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(
        fuel_codes
    )

    if not np.any(valid):

        raise ValueError(
            "Aligned Fuel raster contains no valid pixels."
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

    print()
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
# OUTPUT
# ============================================================

def write_raster(
    path: Path,
    array: np.ndarray,
    reference: dict
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
        predictor=3
    )

    output = np.where(
        np.isfinite(array),
        array,
        OUTPUT_NODATA
    ).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            output,
            1
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    require_file(
        args.fwi_raster,
        "FWI raster"
    )

    require_file(
        args.fuel_raster,
        "Fuel raster"
    )

    require_file(
        args.dem_raster,
        "DEM raster"
    )

    require_file(
        args.fuel_excel,
        "Fuel Excel"
    )

    require_file(
        args.boundary,
        "Fars boundary"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("FIRIS BUILD START")
    print("=" * 70)

    # --------------------------------------------------------
    # FWI reference
    # --------------------------------------------------------

    fwi, reference = read_fwi(
        args.fwi_raster
    )

    # --------------------------------------------------------
    # Fars mask
    # --------------------------------------------------------

    fars_mask = load_boundary_mask(
        args.boundary,
        reference
    )

    # --------------------------------------------------------
    # Fuel alignment
    # --------------------------------------------------------

    fuel_codes = align_to_fwi(
        args.fuel_raster,
        reference,
        Resampling.nearest
    )

    # --------------------------------------------------------
    # Native DEM slope
    # --------------------------------------------------------

    slope = align_slope_to_fwi(
        args.dem_raster,
        reference
    )

    # --------------------------------------------------------
    # Fuel mapping
    # --------------------------------------------------------

    fuel_mapping = load_fuel_mapping(
        args.fuel_excel,
        args.fuel_code_column
    )

    f_fuel, unmapped_codes = fuel_to_score(
        fuel_codes,
        fuel_mapping
    )

    # --------------------------------------------------------
    # FWI normalization
    # --------------------------------------------------------

    f_fwi = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32
    )

    fwi_valid = np.isfinite(
        fwi
    )

    f_fwi[fwi_valid] = np.clip(
        fwi[fwi_valid] / FWI_MAX,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Topography normalization
    # --------------------------------------------------------

    f_topo = np.full(
        slope.shape,
        np.nan,
        dtype=np.float32
    )

    slope_valid = np.isfinite(
        slope
    )

    f_topo[slope_valid] = np.clip(
        slope[slope_valid]
        / SLOPE_REFERENCE,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Coverage inside Fars
    # --------------------------------------------------------

    province_pixels = int(
        np.sum(fars_mask)
    )

    fwi_inside = (
        fars_mask
        & np.isfinite(fwi)
    )

    fuel_inside = (
        fars_mask
        & np.isfinite(f_fuel)
    )

    topo_inside = (
        fars_mask
        & np.isfinite(f_topo)
    )

    common = (
        fwi_inside
        & fuel_inside
        & topo_inside
    )

    fwi_count = int(
        np.sum(fwi_inside)
    )

    fuel_count = int(
        np.sum(fuel_inside)
    )

    topo_count = int(
        np.sum(topo_inside)
    )

    common_count = int(
        np.sum(common)
    )

    print()
    print(
        "FARS COVERAGE VALIDATION"
    )
    print(
        "------------------------"
    )

    print(
        f"Province pixels      : "
        f"{province_pixels:,}"
    )

    print(
        f"FWI valid in Fars    : "
        f"{fwi_count:,} "
        f"({100*fwi_count/province_pixels:.2f}%)"
    )

    print(
        f"Fuel valid in Fars   : "
        f"{fuel_count:,} "
        f"({100*fuel_count/province_pixels:.2f}%)"
    )

    print(
        f"Topo valid in Fars   : "
        f"{topo_count:,} "
        f"({100*topo_count/province_pixels:.2f}%)"
    )

    print(
        f"Common valid in Fars : "
        f"{common_count:,} "
        f"({100*common_count/province_pixels:.2f}%)"
    )

    if fuel_count < province_pixels:

        print()
        print(
            "WARNING: Fuel does not cover all "
            "of Fars."
        )

        print(
            f"Missing Fuel pixels: "
            f"{province_pixels-fuel_count:,}"
        )

        print(
            "NoData will remain NoData."
        )

    if common_count == 0:

        raise RuntimeError(
            "No common valid pixels exist inside Fars."
        )

    # --------------------------------------------------------
    # FLI
    # --------------------------------------------------------

    fli = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32
    )

    fli[common] = (
        100.0
        * (
            FWI_WEIGHT * f_fwi[common]
            + FUEL_WEIGHT * f_fuel[common]
            + TOPO_WEIGHT * f_topo[common]
        )
    )

    fli = np.clip(
        fli,
        0.0,
        100.0
    )

    print()
    print("FINAL FLI")
    print("---------")
    print(
        f"Statistics: "
        f"{stats(fli)}"
    )

    # --------------------------------------------------------
    # Outputs
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

    coverage_path = (
        args.output_dir
        / f"fuel_coverage_fars_{date}.tif"
    )

    report_path = (
        args.output_dir
        / f"firis_report_{date}.json"
    )

    f_fwi_out = np.where(
        fars_mask,
        f_fwi,
        np.nan
    )

    f_fuel_out = np.where(
        fars_mask,
        f_fuel,
        np.nan
    )

    slope_out = np.where(
        fars_mask,
        slope,
        np.nan
    )

    f_topo_out = np.where(
        fars_mask,
        f_topo,
        np.nan
    )

    coverage = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32
    )

    coverage[fars_mask] = np.where(
        fuel_inside[fars_mask],
        1.0,
        0.0
    )

    print()
    print("WRITING OUTPUTS")
    print("----------------")

    write_raster(
        f_fwi_path,
        f_fwi_out,
        reference
    )

    write_raster(
        f_fuel_path,
        f_fuel_out,
        reference
    )

    write_raster(
        slope_path,
        slope_out,
        reference
    )

    write_raster(
        f_topo_path,
        f_topo_out,
        reference
    )

    write_raster(
        fli_path,
        fli,
        reference
    )

    write_raster(
        coverage_path,
        coverage,
        reference
    )

    # --------------------------------------------------------
    # Report
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

        "boundary":
            str(args.boundary),

        "formula":
            "FLI = 100 * "
            "(0.45 * F_FWI + "
            "0.35 * F_Fuel + "
            "0.20 * F_Topo)",

        "weights": {
            "F_FWI": FWI_WEIGHT,
            "F_Fuel": FUEL_WEIGHT,
            "F_Topo": TOPO_WEIGHT
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

            "bounds":
                bounds_dict(
                    reference["bounds"]
                )
        },

        "input_metadata": {

            "FWI":
                raster_metadata(
                    args.fwi_raster
                ),

            "Fuel":
                raster_metadata(
                    args.fuel_raster
                ),

            "DEM":
                raster_metadata(
                    args.dem_raster
                )
        },

        "alignment": {

            "FWI":
                "reference grid",

            "Fuel":
                "nearest-neighbour to FWI",

            "DEM":
                "native slope calculation "
                "then bilinear to FWI"
        },

        "normalization": {

            "F_FWI":
                "clip(FWI / 100, 0, 1)",

            "F_Fuel":
                "Fuelbeds_metric weighted score",

            "F_Topo":
                "clip(slope_degrees / 45, 0, 1)"
        },

        "coverage_inside_fars": {

            "province_pixels":
                province_pixels,

            "FWI_valid_pixels":
                fwi_count,

            "FWI_valid_percent":
                round(
                    100 * fwi_count
                    / province_pixels,
                    4
                ),

            "Fuel_valid_pixels":
                fuel_count,

            "Fuel_valid_percent":
                round(
                    100 * fuel_count
                    / province_pixels,
                    4
                ),

            "Topo_valid_pixels":
                topo_count,

            "Topo_valid_percent":
                round(
                    100 * topo_count
                    / province_pixels,
                    4
                ),

            "common_valid_pixels":
                common_count,

            "common_valid_percent":
                round(
                    100 * common_count
                    / province_pixels,
                    4
                ),

            "common_valid_percent_of_FWI":
                round(
                    100 * common_count
                    / max(fwi_count, 1),
                    4
                )
        },

        "statistics": {

            "FWI":
                stats(f_fwi_out),

            "F_FWI":
                stats(f_fwi_out),

            "F_Fuel":
                stats(f_fuel_out),

            "Slope_degrees":
                stats(slope_out),

            "F_Topo":
                stats(f_topo_out),

            "FLI":
                stats(fli)
        },

        "fuel": {

            "unmapped_code_count":
                len(unmapped_codes),

            "unmapped_codes":
                unmapped_codes[:100]
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

            "Boundary":
                str(args.boundary)
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

            "FuelCoverage":
                str(coverage_path)
        },

        "interpretation": {

            "NoData_policy":
                "NoData is preserved; "
                "missing source coverage "
                "is never extrapolated.",

            "coverage_warning":
                bool(
                    fuel_count
                    < province_pixels
                )
        }
    }

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

        file.write("\n")

    print()
    print("=" * 70)
    print(
        "FIRIS BUILD COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        f"FLI output    : {fli_path}"
    )

    print(
        f"Fuel coverage : {coverage_path}"
    )

    print(
        f"Report        : {report_path}"
    )


if __name__ == "__main__":
    main()
