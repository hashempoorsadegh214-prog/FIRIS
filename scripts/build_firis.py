```python
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
- Slope is calculated on the native DEM.
- DEM NoData values are NEVER artificially filled.
- Slope is calculated only where the required neighbouring
  DEM cells are valid.
- Native slope is then aligned to the FWI grid.
- All final calculations are restricted to fars.geojson.
- NoData is preserved.
- Coverage inside Fars is explicitly reported.
- All final outputs use the exact FWI reference grid.
- Main FLI weights remain unchanged.
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


# ============================================================
# FLI PARAMETERS
# ============================================================

FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

FWI_MAX = 100.0

# 45 degrees is the reference point at which
# the topographic component reaches 1.0.
SLOPE_REFERENCE = 45.0

OUTPUT_NODATA = -9999.0


# ============================================================
# FUEL COMPONENT WEIGHTS
# ============================================================

# Available variables in Fuelbeds_metric:
#
# W_1hLoad       -> Fine Fuel
# W_10hLoad      \
# W_100hLoad      > Dead Wood
# W_1000hLoad    /
# Woody Cover    -> Woody Cover
# Litter Cover   \
# L_depth         > Litter
#
# Canopy Structure is NOT used because it is
# not present in the supplied Fuelbeds_metric table.

FINE_FUEL_WEIGHT = 0.35
DEAD_WOOD_WEIGHT = 0.30
WOODY_COVER_WEIGHT = 0.15
LITTER_WEIGHT = 0.20


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
        type=Path
    )

    parser.add_argument(
        "--fuel-raster",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--dem-raster",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--fuel-excel",
        required=True,
        type=Path
    )

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


# ============================================================
# FILE CHECK
# ============================================================

def require_file(
    path: Path,
    label: str
):

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


# ============================================================
# CLEAN ARRAY
# ============================================================

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

                result[
                    np.isnan(result)
                ] = np.nan

            else:

                result[
                    np.isclose(
                        result,
                        float(nodata)
                    )
                ] = np.nan

        except (
            TypeError,
            ValueError
        ):

            pass

    result[
        ~np.isfinite(result)
    ] = np.nan

    return result


# ============================================================
# STATISTICS
# ============================================================

def stats(
    array: np.ndarray
):

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

        "count":
            int(valid.size),

        "min":
            round(
                float(np.min(valid)),
                6
            ),

        "max":
            round(
                float(np.max(valid)),
                6
            ),

        "mean":
            round(
                float(np.mean(valid)),
                6
            ),

        "std":
            round(
                float(np.std(valid)),
                6
            )
    }


# ============================================================
# BOUNDS
# ============================================================

def bounds_dict(bounds):

    return {

        "left":
            float(bounds.left),

        "bottom":
            float(bounds.bottom),

        "right":
            float(bounds.right),

        "top":
            float(bounds.top)
    }


# ============================================================
# RASTER METADATA
# ============================================================

def raster_metadata(
    path: Path
):

    with rasterio.open(path) as src:

        return {

            "crs":
                str(src.crs)
                if src.crs
                else None,

            "width":
                int(src.width),

            "height":
                int(src.height),

            "cell_size_x":
                float(src.res[0]),

            "cell_size_y":
                float(src.res[1]),

            "bounds":
                bounds_dict(src.bounds),

            "nodata":
                (
                    None
                    if src.nodata is None
                    else float(src.nodata)
                ),

            "transform": [

                float(src.transform.a),

                float(src.transform.b),

                float(src.transform.c),

                float(src.transform.d),

                float(src.transform.e),

                float(src.transform.f)
            ]
        }


# ============================================================
# GRID VALIDATION
# ============================================================

def transform_values(
    transform
):

    return np.array(
        [

            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f

        ],
        dtype=np.float64
    )


def grid_matches(
    reference: dict,
    metadata: dict,
    tolerance: float = 1e-9
):

    reasons = []

    if metadata["crs"] != str(
        reference["crs"]
    ):

        reasons.append(
            "CRS mismatch"
        )

    if metadata["width"] != int(
        reference["width"]
    ):

        reasons.append(
            "Width mismatch"
        )

    if metadata["height"] != int(
        reference["height"]
    ):

        reasons.append(
            "Height mismatch"
        )

    reference_transform = (
        transform_values(
            reference["transform"]
        )
    )

    output_transform = np.array(
        metadata["transform"],
        dtype=np.float64
    )

    if not np.allclose(
        reference_transform,
        output_transform,
        rtol=0.0,
        atol=tolerance
    ):

        reasons.append(
            "Transform mismatch"
        )

    reference_bounds = np.array(
        [

            reference["bounds"].left,
            reference["bounds"].bottom,
            reference["bounds"].right,
            reference["bounds"].top

        ],
        dtype=np.float64
    )

    output_bounds = np.array(
        [

            metadata["bounds"]["left"],
            metadata["bounds"]["bottom"],
            metadata["bounds"]["right"],
            metadata["bounds"]["top"]

        ],
        dtype=np.float64
    )

    if not np.allclose(
        reference_bounds,
        output_bounds,
        rtol=0.0,
        atol=tolerance
    ):

        reasons.append(
            "Bounds mismatch"
        )

    return (
        len(reasons) == 0,
        reasons
    )


def validate_output_grids(
    output_paths: dict,
    reference: dict
):

    print()
    print("=" * 70)
    print("FINAL GRID VALIDATION")
    print("=" * 70)

    print()
    print("REFERENCE = FWI")

    print(
        f"CRS       : {reference['crs']}"
    )

    print(
        f"SIZE      : "
        f"{reference['width']} x "
        f"{reference['height']}"
    )

    print(
        f"RES       : {reference['res']}"
    )

    print(
        f"BOUNDS    : {reference['bounds']}"
    )

    failures = []

    validation = {}

    for name, path in output_paths.items():

        metadata = raster_metadata(
            path
        )

        ok, reasons = grid_matches(
            reference,
            metadata,
            tolerance=1e-9
        )

        validation[name] = {

            "path":
                str(path),

            "matches_fwi_grid":
                bool(ok),

            "reasons":
                reasons,

            "metadata":
                metadata
        }

        if ok:

            print(
                f"✓ {name:<16} GRID MATCH"
            )

        else:

            print(
                f"✗ {name:<16} GRID MISMATCH"
            )

            for reason in reasons:

                print(
                    f"    - {reason}"
                )

            failures.append(
                name
            )

    print()

    if failures:

        print(
            "FINAL GRID VALIDATION FAILED"
        )

        for name in failures:

            print(
                f"  - {name}"
            )

        raise RuntimeError(
            "One or more output rasters do not "
            "match the FWI reference grid."
        )

    print(
        "✓ ALL OUTPUT RASTERS MATCH THE FWI GRID"
    )

    return validation


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
    ) as file:

        geojson = json.load(
            file
        )

    features = geojson.get(
        "features",
        []
    )

    if not features:

        raise ValueError(
            f"Boundary contains no features: "
            f"{boundary_path}"
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
            f"Boundary contains no geometries: "
            f"{boundary_path}"
        )

    source_crs = "EPSG:4326"

    crs_obj = geojson.get(
        "crs"
    )

    if isinstance(
        crs_obj,
        dict
    ):

        props = crs_obj.get(
            "properties",
            {}
        )

        name = (
            props.get("name")
            or props.get("href")
        )

        if (
            isinstance(name, str)
            and name.strip()
        ):

            source_crs = name.strip()

    target_crs = reference[
        "crs"
    ]

    if str(target_crs) != source_crs:

        geometries = [

            transform_geom(
                source_crs,
                target_crs,
                geometry,
                precision=12
            )

            for geometry in geometries
        ]

    mask = geometry_mask(

        geometries,

        out_shape=(
            reference["height"],
            reference["width"]
        ),

        transform=reference[
            "transform"
        ],

        invert=True,

        all_touched=False
    )

    count = int(
        np.sum(mask)
    )

    if count == 0:

        raise ValueError(
            "Fars boundary does not overlap "
            "the FWI grid."
        )

    print()
    print("FARS BOUNDARY")
    print("-------------")

    print(
        f"Boundary file       : "
        f"{boundary_path}"
    )

    print(
        f"Boundary CRS        : "
        f"{source_crs}"
    )

    print(
        f"Target CRS          : "
        f"{reference['crs']}"
    )

    print(
        f"Pixels inside Fars  : "
        f"{count:,}"
    )

    return mask


# ============================================================
# FWI
# ============================================================

def read_fwi(
    path: Path
):

    with rasterio.open(
        path
    ) as src:

        if src.crs is None:

            raise ValueError(
                "FWI raster has no CRS."
            )

        data = clean_array(
            src.read(1),
            src.nodata
        )

        reference = {

            "crs":
                src.crs,

            "transform":
                src.transform,

            "width":
                src.width,

            "height":
                src.height,

            "profile":
                src.profile.copy(),

            "bounds":
                src.bounds,

            "res":
                src.res
        }

    print()
    print("FWI REFERENCE GRID")
    print("------------------")

    print(
        f"CRS        : "
        f"{reference['crs']}"
    )

    print(
        f"Width      : "
        f"{reference['width']}"
    )

    print(
        f"Height     : "
        f"{reference['height']}"
    )

    print(
        f"Cell size  : "
        f"{reference['res']}"
    )

    print(
        f"Bounds     : "
        f"{reference['bounds']}"
    )

    print(
        f"Transform  : "
        f"{reference['transform']}"
    )

    print(
        f"Statistics : "
        f"{stats(data)}"
    )

    return (
        data,
        reference
    )


# ============================================================
# ALIGN RASTER TO FWI
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

    with rasterio.open(
        path
    ) as src:

        if src.crs is None:

            raise ValueError(
                f"Raster has no CRS: {path}"
            )

        source = clean_array(
            src.read(1),
            src.nodata
        )

        print()
        print(
            f"Aligning: {path}"
        )

        print(
            f"Source CRS      : {src.crs}"
        )

        print(
            f"Source size     : "
            f"{src.width} x {src.height}"
        )

        print(
            f"Source cell     : "
            f"{src.res}"
        )

        print(
            f"Source bounds   : "
            f"{src.bounds}"
        )

        print(
            f"Target CRS      : "
            f"{reference['crs']}"
        )

        print(
            f"Target size     : "
            f"{reference['width']} x "
            f"{reference['height']}"
        )

        print(
            f"Target cell     : "
            f"{reference['res']}"
        )

        print(
            f"Target bounds   : "
            f"{reference['bounds']}"
        )

        print(
            f"Resampling      : "
            f"{resampling.name}"
        )

        reproject(

            source=source,

            destination=destination,

            src_transform=src.transform,

            src_crs=src.crs,

            src_nodata=np.nan,

            dst_transform=reference[
                "transform"
            ],

            dst_crs=reference[
                "crs"
            ],

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
# METRIC CELL SIZE
# ============================================================

def metric_cell_size(
    src
):

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

        center_row = (
            src.height / 2.0
        )

        latitude = (
            src.transform.f
            +
            center_row *
            src.transform.e
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


# ============================================================
# NATIVE SLOPE
# ============================================================

def calculate_native_slope(
    dem_path: Path
):

    """
    Calculate slope on the native DEM.

    Method:
        Central finite differences.

    Important:
        NoData pixels are NOT filled.

        A slope value is calculated only where:
        - center DEM pixel is valid
        - north pixel is valid
        - south pixel is valid
        - west pixel is valid
        - east pixel is valid

    Slope output unit:
        degrees
    """

    print()
    print(
        "CALCULATING SLOPE ON NATIVE DEM"
    )
    print(
        "--------------------------------"
    )

    with rasterio.open(
        dem_path
    ) as src:

        if src.crs is None:

            raise ValueError(
                "DEM raster has no CRS."
            )

        dem = clean_array(
            src.read(1),
            src.nodata
        )

        valid = np.isfinite(
            dem
        )

        if not np.any(valid):

            raise ValueError(
                "DEM contains no valid pixels."
            )

        dx, dy = metric_cell_size(
            src
        )

        print(
            f"DEM CRS       : {src.crs}"
        )

        print(
            f"DEM size      : "
            f"{src.width} x {src.height}"
        )

        print(
            f"DEM cell      : "
            f"{src.res}"
        )

        print(
            f"DEM bounds    : "
            f"{src.bounds}"
        )

        print(
            f"Metric spacing: "
            f"X={dx:.3f} m, "
            f"Y={dy:.3f} m"
        )

        # ----------------------------------------------------
        # OUTPUT SLOPE
        # ----------------------------------------------------

        slope = np.full(
            dem.shape,
            np.nan,
            dtype=np.float32
        )

        rows, cols = dem.shape

        if (
            rows >= 3
            and
            cols >= 3
        ):

            center = dem[
                1:-1,
                1:-1
            ]

            north = dem[
                :-2,
                1:-1
            ]

            south = dem[
                2:,
                1:-1
            ]

            west = dem[
                1:-1,
                :-2
            ]

            east = dem[
                1:-1,
                2:
            ]

            # ------------------------------------------------
            # VALIDITY MASK
            # ------------------------------------------------

            local_valid = (

                np.isfinite(center)
                &
                np.isfinite(north)
                &
                np.isfinite(south)
                &
                np.isfinite(west)
                &
                np.isfinite(east)
            )

            # ------------------------------------------------
            # GRADIENTS
            # ------------------------------------------------

            dzdx = np.full(
                center.shape,
                np.nan,
                dtype=np.float32
            )

            dzdy = np.full(
                center.shape,
                np.nan,
                dtype=np.float32
            )

            dzdx[local_valid] = (

                east[local_valid]
                -
                west[local_valid]

            ) / (
                2.0 * dx
            )

            dzdy[local_valid] = (

                south[local_valid]
                -
                north[local_valid]

            ) / (
                2.0 * dy
            )

            # ------------------------------------------------
            # SLOPE
            # ------------------------------------------------

            gradient = np.sqrt(

                dzdx ** 2
                +
                dzdy ** 2

            )

            local_slope = np.degrees(
                np.arctan(
                    gradient
                )
            )

            slope[
                1:-1,
                1:-1
            ][local_valid] = (
                local_slope[local_valid]
            )

        slope[
            ~np.isfinite(slope)
        ] = np.nan

        print(
            f"Native slope statistics: "
            f"{stats(slope)}"
        )

        valid_slope_count = int(
            np.sum(
                np.isfinite(slope)
            )
        )

        print(
            f"Valid native slope pixels: "
            f"{valid_slope_count:,}"
        )

        return (
            slope,
            src.transform,
            src.crs
        )


# ============================================================
# ALIGN SLOPE TO FWI
# ============================================================

def align_slope_to_fwi(
    dem_path: Path,
    reference: dict
):

    (
        slope,
        dem_transform,
        dem_crs
    ) = calculate_native_slope(
        dem_path
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

        dst_transform=
            reference["transform"],

        dst_crs=
            reference["crs"],

        dst_nodata=np.nan,

        resampling=
            Resampling.bilinear
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
# FUEL HELPERS
# ============================================================

def find_column(
    dataframe,
    candidates
):

    lookup = {

        str(c).strip().lower():
            c

        for c in dataframe.columns
    }

    for candidate in candidates:

        key = (
            candidate.strip().lower()
        )

        if key in lookup:

            return lookup[key]

    return None


def normalize_column(
    series
):

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
        /
        (maximum - minimum)
    )


# ============================================================
# FUEL MAPPING
# ============================================================

def load_fuel_mapping(
    excel_path: Path,
    requested_code_column: str
):

    print()
    print(
        "LOADING FUEL TABLE"
    )
    print(
        "------------------"
    )

    workbook = pd.ExcelFile(
        excel_path
    )

    if (
        "Fuelbeds_metric"
        not in workbook.sheet_names
    ):

        raise ValueError(
            "Sheet 'Fuelbeds_metric' not found. "
            f"Available sheets: "
            f"{workbook.sheet_names}"
        )

    df = pd.read_excel(
        excel_path,
        sheet_name="Fuelbeds_metric"
    )

    print(
        "Fuel columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # FUEL CODE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # WOODY COVER
    # --------------------------------------------------------

    woody_col = find_column(

        df,

        [
            "Woody Cover (%)",
            "Woody Cover"
        ]
    )

    # --------------------------------------------------------
    # FINE FUEL
    # --------------------------------------------------------

    w1_col = find_column(

        df,

        [
            "W_1hLoad (Mg/ha)",
            "W_1h Load (Mg/ha)",
            "W_1hLoad"
        ]
    )

    # --------------------------------------------------------
    # DEAD WOOD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LITTER
    # --------------------------------------------------------

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

    # ========================================================
    # FINE FUEL
    # ========================================================

    fine = normalize_column(
        df[w1_col]
    )

    # ========================================================
    # DEAD WOOD
    # ========================================================

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
            weight *
            values
        )

        total_weight += weight

    if total_weight > 0:

        dead /= total_weight

    # ========================================================
    # WOODY COVER
    # ========================================================
    #
    # IMPORTANT:
    # Woody Cover is used ONLY ONCE.
    # ========================================================

    if woody_col is not None:

        woody = normalize_column(
            df[woody_col]
        )

    else:

        woody = pd.Series(
            0.0,
            index=df.index
        )

    # ========================================================
    # LITTER
    # ========================================================

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
            /
            len(litter_parts)
        )

    else:

        litter = pd.Series(
            0.0,
            index=df.index
        )

    # ========================================================
    # CORRECTED FUEL FORMULA
    # ========================================================
    #
    # Fine Fuel     = 35%
    # Dead Wood     = 30%
    # Woody Cover   = 15%
    # Litter        = 20%
    #
    # Total         = 100%
    #
    # Woody Cover is NOT duplicated.
    # Canopy Structure is NOT used.
    # ========================================================

    fuel_score = (

        FINE_FUEL_WEIGHT * fine
        +
        DEAD_WOOD_WEIGHT * dead
        +
        WOODY_COVER_WEIGHT * woody
        +
        LITTER_WEIGHT * litter

    ).clip(
        0.0,
        1.0
    )

    df["_F_Fuel"] = fuel_score

    df = (
        df
        .dropna(
            subset=[code_col]
        )
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

            code_value = float(
                code
            )

            score_value = float(
                score
            )

            if (
                math.isfinite(
                    code_value
                )
                and
                math.isfinite(
                    score_value
                )
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

    print()
    print(
        f"Fuel-code column: "
        f"{code_col}"
    )

    print(
        f"Fuel mapping entries: "
        f"{len(mapping)}"
    )

    print()
    print(
        "FUEL COMPONENT WEIGHTS"
    )
    print(
        "----------------------"
    )

    print(
        f"Fine Fuel     : "
        f"{FINE_FUEL_WEIGHT:.2f}"
    )

    print(
        f"Dead Wood     : "
        f"{DEAD_WOOD_WEIGHT:.2f}"
    )

    print(
        f"Woody Cover   : "
        f"{WOODY_COVER_WEIGHT:.2f}"
    )

    print(
        f"Litter        : "
        f"{LITTER_WEIGHT:.2f}"
    )

    print(
        "Canopy        : NOT USED"
    )

    print(
        "TOTAL         : "
        f"{FINE_FUEL_WEIGHT + DEAD_WOOD_WEIGHT + WOODY_COVER_WEIGHT + LITTER_WEIGHT:.2f}"
    )

    return mapping


# ============================================================
# FUEL TO SCORE
# ============================================================

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
            "Aligned Fuel raster contains "
            "no valid pixels."
        )

    unique_codes = np.unique(
        fuel_codes[valid]
    )

    unmapped = []

    for code in unique_codes:

        code_float = float(
            code
        )

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
    print(
        "FUEL MAPPING"
    )
    print(
        "------------"
    )

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

    return (
        output,
        unmapped
    )


# ============================================================
# WRITE RASTER
# ============================================================

def write_raster(
    path: Path,
    array: np.ndarray,
    reference: dict
):

    profile = (
        reference["profile"].copy()
    )

    profile.update(

        driver="GTiff",

        dtype="float32",

        count=1,

        width=
            reference["width"],

        height=
            reference["height"],

        crs=
            reference["crs"],

        transform=
            reference["transform"],

        nodata=
            OUTPUT_NODATA,

        compress="deflate",

        predictor=3
    )

    output = np.where(

        np.isfinite(array),

        array,

        OUTPUT_NODATA

    ).astype(
        np.float32
    )

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

    # ========================================================
    # FWI REFERENCE
    # ========================================================

    fwi, reference = read_fwi(
        args.fwi_raster
    )

    # ========================================================
    # FARS MASK
    # ========================================================

    fars_mask = load_boundary_mask(
        args.boundary,
        reference
    )

    # ========================================================
    # FUEL ALIGNMENT
    # ========================================================

    fuel_codes = align_to_fwi(

        args.fuel_raster,

        reference,

        Resampling.nearest
    )

    # ========================================================
    # DEM / SLOPE
    # ========================================================

    slope = align_slope_to_fwi(

        args.dem_raster,

        reference
    )

    # ========================================================
    # FUEL MAPPING
    # ========================================================

    fuel_mapping = load_fuel_mapping(

        args.fuel_excel,

        args.fuel_code_column
    )

    (
        f_fuel,
        unmapped_codes
    ) = fuel_to_score(

        fuel_codes,

        fuel_mapping
    )

    # ========================================================
    # FWI NORMALIZATION
    # ========================================================

    f_fwi = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32
    )

    fwi_valid = np.isfinite(
        fwi
    )

    f_fwi[fwi_valid] = np.clip(

        fwi[fwi_valid]
        /
        FWI_MAX,

        0.0,
        1.0
    )

    # ========================================================
    # TOPOGRAPHY NORMALIZATION
    # ========================================================

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
        /
        SLOPE_REFERENCE,

        0.0,
        1.0
    )

    # ========================================================
    # COVERAGE INSIDE FARS
    # ========================================================

    province_pixels = int(
        np.sum(fars_mask)
    )

    fwi_inside = (
        fars_mask
        &
        np.isfinite(fwi)
    )

    fuel_inside = (
        fars_mask
        &
        np.isfinite(f_fuel)
    )

    topo_inside = (
        fars_mask
        &
        np.isfinite(f_topo)
    )

    common = (
        fwi_inside
        &
        fuel_inside
        &
        topo_inside
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
            "No common valid pixels exist "
            "inside Fars."
        )

    # ========================================================
    # FLI CALCULATION
    # ========================================================

    fli = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32
    )

    fli[common] = (

        100.0

        *

        (
            FWI_WEIGHT * f_fwi[common]
            +
            FUEL_WEIGHT * f_fuel[common]
            +
            TOPO_WEIGHT * f_topo[common]
        )
    )

    fli = np.clip(
        fli,
        0.0,
        100.0
    )

    print()
    print(
        "FINAL FLI"
    )
    print(
        "---------"
    )

    print(
        f"Statistics: "
        f"{stats(fli)}"
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    date = args.run_date

    f_fwi_path = (
        args.output_dir
        /
        f"f_fwi_fars_{date}.tif"
    )

    f_fuel_path = (
        args.output_dir
        /
        f"f_fuel_fars_{date}.tif"
    )

    slope_path = (
        args.output_dir
        /
        f"slope_fars_{date}.tif"
    )

    f_topo_path = (
        args.output_dir
        /
        f"f_topo_fars_{date}.tif"
    )

    fli_path = (
        args.output_dir
        /
        f"fli_fars_{date}.tif"
    )

    coverage_path = (
        args.output_dir
        /
        f"fuel_coverage_fars_{date}.tif"
    )

    report_path = (
        args.output_dir
        /
        f"firis_report_{date}.json"
    )

    # ========================================================
    # MASK FINAL OUTPUTS TO FARS
    # ========================================================

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

    # ========================================================
    # FUEL COVERAGE
    # ========================================================

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

    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    print()
    print(
        "WRITING OUTPUTS"
    )
    print(
        "---------------"
    )

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

    # ========================================================
    # FINAL GRID VALIDATION
    # ========================================================

    output_paths = {

        "F_FWI":
            f_fwi_path,

        "F_Fuel":
            f_fuel_path,

        "Slope":
            slope_path,

        "F_Topo":
            f_topo_path,

        "FLI":
            fli_path,

        "FuelCoverage":
            coverage_path
    }

    grid_validation = (
        validate_output_grids(
            output_paths,
            reference
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

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

            "F_FWI":
                FWI_WEIGHT,

            "F_Fuel":
                FUEL_WEIGHT,

            "F_Topo":
                TOPO_WEIGHT
        },

        "fuel_formula": {

            "FineFuel":
                FINE_FUEL_WEIGHT,

            "DeadWood":
                DEAD_WOOD_WEIGHT,

            "WoodyCover":
                WOODY_COVER_WEIGHT,

            "Litter":
                LITTER_WEIGHT,

            "CanopyStructure":
                None,

            "WoodyCover_used_once":
                True,

            "weights_sum":
                (
                    FINE_FUEL_WEIGHT
                    +
                    DEAD_WOOD_WEIGHT
                    +
                    WOODY_COVER_WEIGHT
                    +
                    LITTER_WEIGHT
                )
        },

        "slope_method": {

            "calculation":
                "Native DEM central finite differences",

            "nodata_filling":
                False,

            "nodata_policy":
                "Slope is calculated only where "
                "center, north, south, west and east "
                "DEM cells are valid.",

            "units":
                "degrees",

            "reference_degrees":
                SLOPE_REFERENCE,

            "normalization":
                "clip(slope_degrees / 45, 0, 1)"
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
                ),

            "transform": [

                float(reference["transform"].a),

                float(reference["transform"].b),

                float(reference["transform"].c),

                float(reference["transform"].d),

                float(reference["transform"].e),

                float(reference["transform"].f)
            ]
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
                "without artificial NoData filling, "
                "then bilinear alignment to FWI"
        },

        "normalization": {

            "F_FWI":
                "clip(FWI / 100, 0, 1)",

            "F_Fuel":
                "0.35 FineFuel + "
                "0.30 DeadWood + "
                "0.15 WoodyCover + "
                "0.20 Litter",

            "F_Topo":
                "clip(slope_degrees / 45, 0, 1)"
        },

        "grid_validation":
            grid_validation,

        "coverage_inside_fars": {

            "province_pixels":
                province_pixels,

            "FWI_valid_pixels":
                fwi_count,

            "FWI_valid_percent":
                round(
                    100 *
                    fwi_count /
                    province_pixels,
                    4
                ),

            "Fuel_valid_pixels":
                fuel_count,

            "Fuel_valid_percent":
                round(
                    100 *
                    fuel_count /
                    province_pixels,
                    4
                ),

            "Topo_valid_pixels":
                topo_count,

            "Topo_valid_percent":
                round(
                    100 *
                    topo_count /
                    province_pixels,
                    4
                ),

            "common_valid_pixels":
                common_count,

            "common_valid_percent":
                round(
                    100 *
                    common_count /
                    province_pixels,
                    4
                ),

            "common_valid_percent_of_FWI":
                round(
                    100 *
                    common_count /
                    max(
                        fwi_count,
                        1
                    ),
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
                    fuel_count <
                    province_pixels
                ),

            "grid_policy":
                "All final rasters use the "
                "exact FWI reference profile.",

            "woody_policy":
                "Woody Cover is included exactly once.",

            "canopy_policy":
                "Canopy Structure is not included "
                "because it is absent from "
                "Fuelbeds_metric."
        }
    }

    # ========================================================
    # WRITE JSON REPORT
    # ========================================================

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

        file.write(
            "\n"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FIRIS BUILD COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"FLI output    : "
        f"{fli_path}"
    )

    print(
        f"Fuel coverage : "
        f"{coverage_path}"
    )

    print(
        f"Report        : "
        f"{report_path}"
    )


if __name__ == "__main__":

    main()
```
