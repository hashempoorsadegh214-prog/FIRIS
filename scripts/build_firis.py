#!/usr/bin/env python3
"""
FIRIS - Fars Integrated Fire Information System

FLI = 100 * (
    0.45 * F_FWI
    + 0.35 * F_Fuel
    + 0.20 * F_Topo
)

NEW FUEL MODEL
--------------
Fuel is a continuous Sentinel-2-derived raster
with values normalized to the range 0-1.

F_Fuel = direct value from the Fuel raster.

No Excel table is used.
No JOIN_VALUE is used.
No Fuelbeds_metric lookup is used.

Spatial rules:
- FWI is the reference grid.
- Fuel -> FWI grid using nearest neighbour.
- Fuel values are expected in the range 0-1.
- Slope is calculated on the native DEM.
- DEM NoData values are NEVER artificially filled.
- Slope is calculated only where the required neighbouring
  DEM cells are valid.
- Native slope is then aligned to the FWI grid.
- All final calculations are restricted to fars.geojson.
- NoData is preserved.
- Missing Fuel coverage is never artificially extrapolated.
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
SLOPE_REFERENCE = 45.0
OUTPUT_NODATA = -9999.0


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FIRIS Fire Likelihood Index"
    )

    parser.add_argument("--fwi-raster", required=True, type=Path)
    parser.add_argument("--fuel-raster", required=True, type=Path)
    parser.add_argument("--dem-raster", required=True, type=Path)
    parser.add_argument("--boundary", type=Path, default=Path("fars.geojson"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-date", required=True)

    return parser.parse_args()


# ============================================================
# FILE CHECK
# ============================================================

def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================
# CLEAN ARRAY
# ============================================================

def clean_array(array: np.ndarray, nodata: Any = None) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32).copy()

    if nodata is not None:
        try:
            if np.isnan(nodata):
                result[np.isnan(result)] = np.nan
            else:
                result[np.isclose(result, float(nodata))] = np.nan
        except (TypeError, ValueError):
            pass

    result[~np.isfinite(result)] = np.nan
    return result


# ============================================================
# STATISTICS
# ============================================================

def stats(array: np.ndarray):
    valid = array[np.isfinite(array)]

    if valid.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

    return {
        "count": int(valid.size),
        "min": round(float(np.min(valid)), 6),
        "max": round(float(np.max(valid)), 6),
        "mean": round(float(np.mean(valid)), 6),
        "std": round(float(np.std(valid)), 6),
    }


# ============================================================
# BOUNDS
# ============================================================

def bounds_dict(bounds):
    return {
        "left": float(bounds.left),
        "bottom": float(bounds.bottom),
        "right": float(bounds.right),
        "top": float(bounds.top),
    }


# ============================================================
# RASTER METADATA
# ============================================================

def raster_metadata(path: Path):
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs) if src.crs else None,
            "width": int(src.width),
            "height": int(src.height),
            "cell_size_x": float(src.res[0]),
            "cell_size_y": float(src.res[1]),
            "bounds": bounds_dict(src.bounds),
            "nodata": None if src.nodata is None else float(src.nodata),
            "transform": [
                float(src.transform.a),
                float(src.transform.b),
                float(src.transform.c),
                float(src.transform.d),
                float(src.transform.e),
                float(src.transform.f),
            ],
        }


# ============================================================
# GRID VALIDATION
# ============================================================

def transform_values(transform):
    return np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype=np.float64,
    )


def grid_matches(reference: dict, metadata: dict, tolerance: float = 1e-9):
    reasons = []

    if metadata["crs"] != str(reference["crs"]):
        reasons.append("CRS mismatch")

    if metadata["width"] != int(reference["width"]):
        reasons.append("Width mismatch")

    if metadata["height"] != int(reference["height"]):
        reasons.append("Height mismatch")

    reference_transform = transform_values(reference["transform"])
    output_transform = np.array(metadata["transform"], dtype=np.float64)

    if not np.allclose(reference_transform, output_transform, rtol=0.0, atol=tolerance):
        reasons.append("Transform mismatch")

    reference_bounds = np.array(
        [
            reference["bounds"].left,
            reference["bounds"].bottom,
            reference["bounds"].right,
            reference["bounds"].top,
        ],
        dtype=np.float64,
    )

    output_bounds = np.array(
        [
            metadata["bounds"]["left"],
            metadata["bounds"]["bottom"],
            metadata["bounds"]["right"],
            metadata["bounds"]["top"],
        ],
        dtype=np.float64,
    )

    if not np.allclose(reference_bounds, output_bounds, rtol=0.0, atol=tolerance):
        reasons.append("Bounds mismatch")

    return len(reasons) == 0, reasons


def validate_output_grids(output_paths: dict, reference: dict):
    print()
    print("=" * 70)
    print("FINAL GRID VALIDATION")
    print("=" * 70)

    print()
    print("REFERENCE = FWI")
    print(f"CRS       : {reference['crs']}")
    print(f"SIZE      : {reference['width']} x {reference['height']}")
    print(f"RES       : {reference['res']}")
    print(f"BOUNDS    : {reference['bounds']}")

    failures = []
    validation = {}

    for name, path in output_paths.items():
        metadata = raster_metadata(path)
        ok, reasons = grid_matches(reference, metadata, tolerance=1e-9)

        validation[name] = {
            "path": str(path),
            "matches_fwi_grid": bool(ok),
            "reasons": reasons,
            "metadata": metadata,
        }

        if ok:
            print(f"✓ {name:<16} GRID MATCH")
        else:
            print(f"✗ {name:<16} GRID MISMATCH")
            for reason in reasons:
                print(f"    - {reason}")
            failures.append(name)

    print()

    if failures:
        print("FINAL GRID VALIDATION FAILED")
        for name in failures:
            print(f"  - {name}")
        raise RuntimeError(
            "One or more output rasters do not match the FWI reference grid."
        )

    print("✓ ALL OUTPUT RASTERS MATCH THE FWI GRID")
    return validation


# ============================================================
# FARS BOUNDARY
# ============================================================

def load_boundary_mask(boundary_path: Path, reference: dict):
    with boundary_path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson.get("features", [])
    if not features:
        raise ValueError(f"Boundary contains no features: {boundary_path}")

    geometries = []
    for feature in features:
        geometry = feature.get("geometry")
        if geometry:
            geometries.append(geometry)

    if not geometries:
        raise ValueError(f"Boundary contains no geometries: {boundary_path}")

    source_crs = "EPSG:4326"
    crs_obj = geojson.get("crs")

    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties", {})
        name = props.get("name") or props.get("href")
        if isinstance(name, str) and name.strip():
            source_crs = name.strip()

    target_crs = reference["crs"]
    if str(target_crs) != source_crs:
        geometries = [
            transform_geom(source_crs, target_crs, geometry, precision=12)
            for geometry in geometries
        ]

    mask = geometry_mask(
        geometries,
        out_shape=(reference["height"], reference["width"]),
        transform=reference["transform"],
        invert=True,
        all_touched=False,
    )

    count = int(np.sum(mask))
    if count == 0:
        raise ValueError("Fars boundary does not overlap the FWI grid.")

    print()
    print("FARS BOUNDARY")
    print("-------------")
    print(f"Boundary file       : {boundary_path}")
    print(f"Boundary CRS        : {source_crs}")
    print(f"Target CRS          : {reference['crs']}")
    print(f"Pixels inside Fars  : {count:,}")

    return mask


# ============================================================
# FWI
# ============================================================

def read_fwi(path: Path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("FWI raster has no CRS.")

        data = clean_array(src.read(1), src.nodata)

        reference = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": src.profile.copy(),
            "bounds": src.bounds,
            "res": src.res,
        }

    print()
    print("FWI REFERENCE GRID")
    print("------------------")
    print(f"CRS        : {reference['crs']}")
    print(f"Width      : {reference['width']}")
    print(f"Height     : {reference['height']}")
    print(f"Cell size  : {reference['res']}")
    print(f"Bounds     : {reference['bounds']}")
    print(f"Transform  : {reference['transform']}")
    print(f"Statistics : {stats(data)}")

    return data, reference


# ============================================================
# ALIGN RASTER TO FWI
# ============================================================

def align_to_fwi(path: Path, reference: dict, resampling: Resampling):
    destination = np.full((reference["height"], reference["width"]), np.nan, dtype=np.float32)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")

        source = clean_array(src.read(1), src.nodata)

        print()
        print(f"Aligning: {path}")
        print(f"Source CRS      : {src.crs}")
        print(f"Source size     : {src.width} x {src.height}")
        print(f"Source cell     : {src.res}")
        print(f"Source bounds   : {src.bounds}")
        print(f"Target CRS      : {reference['crs']}")
        print(f"Target size     : {reference['width']} x {reference['height']}")
        print(f"Target cell     : {reference['res']}")
        print(f"Target bounds   : {reference['bounds']}")
        print(f"Resampling      : {resampling.name}")

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
    print(f"Aligned statistics: {stats(destination)}")
    return destination


# ============================================================
# FUEL VALIDATION
# ============================================================

def validate_fuel_range(fuel: np.ndarray):
    valid = np.isfinite(fuel)
    if not np.any(valid):
        raise ValueError("Fuel raster contains no valid pixels.")

    minimum = float(np.nanmin(fuel))
    maximum = float(np.nanmax(fuel))

    print()
    print("FUEL VALUE VALIDATION")
    print("---------------------")
    print(f"Minimum Fuel value : {minimum:.6f}")
    print(f"Maximum Fuel value : {maximum:.6f}")

    if minimum < 0.0:
        raise ValueError("Fuel raster contains values below 0.")
    if maximum > 1.0:
        raise ValueError("Fuel raster contains values above 1.")

    print("✓ Fuel values are within the expected 0-1 range.")


# ============================================================
# METRIC CELL SIZE
# ============================================================

def metric_cell_size(src):
    if src.crs is None:
        raise ValueError("DEM CRS is missing.")

    xres = abs(float(src.transform.a))
    yres = abs(float(src.transform.e))

    if src.crs.is_projected:
        return xres, yres

    if src.crs.is_geographic:
        center_row = src.height / 2.0
        latitude = src.transform.f + center_row * src.transform.e
        lat = math.radians(float(latitude))

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

        return xres * meters_lon, yres * meters_lat

    raise ValueError("Unsupported DEM coordinate system.")


# ============================================================
# NATIVE SLOPE
# ============================================================

def calculate_native_slope(dem_path: Path):
    print()
    print("CALCULATING SLOPE ON NATIVE DEM")
    print("--------------------------------")

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError("DEM raster has no CRS.")

        dem = clean_array(src.read(1), src.nodata)
        valid = np.isfinite(dem)

        if not np.any(valid):
            raise ValueError("DEM contains no valid pixels.")

        dx, dy = metric_cell_size(src)

        print(f"DEM CRS       : {src.crs}")
        print(f"DEM size      : {src.width} x {src.height}")
        print(f"DEM cell      : {src.res}")
        print(f"DEM bounds    : {src.bounds}")
        print(f"Metric spacing: X={dx:.3f} m, Y={dy:.3f} m")

        slope = np.full(dem.shape, np.nan, dtype=np.float32)
        rows, cols = dem.shape

        if rows >= 3 and cols >= 3:
            center = dem[1:-1, 1:-1]
            north = dem[:-2, 1:-1]
            south = dem[2:, 1:-1]
            west = dem[1:-1, :-2]
            east = dem[1:-1, 2:]

            local_valid = (
                np.isfinite(center)
                & np.isfinite(north)
                & np.isfinite(south)
                & np.isfinite(west)
                & np.isfinite(east)
            )

            dzdx = np.full(center.shape, np.nan, dtype=np.float32)
            dzdy = np.full(center.shape, np.nan, dtype=np.float32)

            dzdx[local_valid] = (east[local_valid] - west[local_valid]) / (2.0 * dx)
            dzdy[local_valid] = (south[local_valid] - north[local_valid]) / (2.0 * dy)

            gradient = np.sqrt(dzdx**2 + dzdy**2)
            local_slope = np.degrees(np.arctan(gradient))

            slope[1:-1, 1:-1][local_valid] = local_slope[local_valid]

        slope[~np.isfinite(slope)] = np.nan

        print(f"Native slope statistics: {stats(slope)}")
        print(f"Valid native slope pixels: {int(np.sum(np.isfinite(slope))):,}")

        return slope, src.transform, src.crs


# ============================================================
# ALIGN SLOPE TO FWI
# ============================================================

def align_slope_to_fwi(dem_path: Path, reference: dict):
    slope, dem_transform, dem_crs = calculate_native_slope(dem_path)

    destination = np.full((reference["height"], reference["width"]), np.nan, dtype=np.float32)

    print()
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

    destination[~np.isfinite(destination)] = np.nan
    print(f"FWI-grid slope statistics: {stats(destination)}")
    return destination


# ============================================================
# WRITE RASTER
# ============================================================

def write_raster(path: Path, array: np.ndarray, reference: dict):
    profile = reference["profile"].copy()
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

    output = np.where(np.isfinite(array), array, OUTPUT_NODATA).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    require_file(args.fwi_raster, "FWI raster")
    require_file(args.fuel_raster, "Fuel raster")
    require_file(args.dem_raster, "DEM raster")
    require_file(args.boundary, "Fars boundary")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("FIRIS BUILD START")
    print("=" * 70)

    print()
    print("NEW FUEL MODEL")
    print("--------------")
    print("Fuel source: Sentinel-2-derived continuous raster")
    print("Fuel range : 0-1")
    print("Excel      : NOT USED")
    print("JOIN_VALUE : NOT USED")

    fwi, reference = read_fwi(args.fwi_raster)
    fars_mask = load_boundary_mask(args.boundary, reference)
    f_fuel = align_to_fwi(args.fuel_raster, reference, Resampling.nearest)
    validate_fuel_range(f_fuel)
    slope = align_slope_to_fwi(args.dem_raster, reference)

    f_fwi = np.full(fwi.shape, np.nan, dtype=np.float32)
    fwi_valid = np.isfinite(fwi)
    f_fwi[fwi_valid] = np.clip(fwi[fwi_valid] / FWI_MAX, 0.0, 1.0)

    f_topo = np.full(slope.shape, np.nan, dtype=np.float32)
    slope_valid = np.isfinite(slope)
    f_topo[slope_valid] = np.clip(slope[slope_valid] / SLOPE_REFERENCE, 0.0, 1.0)

    province_pixels = int(np.sum(fars_mask))

    fwi_inside = fars_mask & np.isfinite(fwi)
    fuel_inside = fars_mask & np.isfinite(f_fuel)
    topo_inside = fars_mask & np.isfinite(f_topo)
    common = fwi_inside & fuel_inside & topo_inside

    fwi_count = int(np.sum(fwi_inside))
    fuel_count = int(np.sum(fuel_inside))
    topo_count = int(np.sum(topo_inside))
    common_count = int(np.sum(common))

    print()
    print("FARS COVERAGE VALIDATION")
    print("------------------------")
    print(f"Province pixels      : {province_pixels:,}")
    print(f"FWI valid in Fars    : {fwi_count:,} ({100*fwi_count/province_pixels:.2f}%)")
    print(f"Fuel valid in Fars   : {fuel_count:,} ({100*fuel_count/province_pixels:.2f}%)")
    print(f"Topo valid in Fars   : {topo_count:,} ({100*topo_count/province_pixels:.2f}%)")
    print(f"Common valid in Fars : {common_count:,} ({100*common_count/province_pixels:.2f}%)")

    if fuel_count < province_pixels:
        print()
        print("WARNING: Fuel does not cover all of Fars.")
        print(f"Missing Fuel pixels: {province_pixels - fuel_count:,}")
        print("NoData will remain NoData.")
    else:
        print()
        print("✓ Fuel coverage is complete inside Fars.")

    if common_count == 0:
        raise RuntimeError("No common valid pixels exist inside Fars.")

    fli = np.full(fwi.shape, np.nan, dtype=np.float32)
    fli[common] = 100.0 * (
        FWI_WEIGHT * f_fwi[common]
        + FUEL_WEIGHT * f_fuel[common]
        + TOPO_WEIGHT * f_topo[common]
    )
    fli = np.clip(fli, 0.0, 100.0)
    fli = np.where(fars_mask, fli, np.nan)

    print()
    print("FINAL FLI")
    print("---------")
    print(f"Statistics: {stats(fli)}")

    date = args.run_date

    f_fwi_path = args.output_dir / f"f_fwi_fars_{date}.tif"
    f_fuel_path = args.output_dir / f"f_fuel_fars_{date}.tif"
    slope_path = args.output_dir / f"slope_fars_{date}.tif"
    f_topo_path = args.output_dir / f"f_topo_fars_{date}.tif"
    fli_path = args.output_dir / f"fli_fars_{date}.tif"
    coverage_path = args.output_dir / f"fuel_coverage_fars_{date}.tif"
    report_path = args.output_dir / f"firis_report_{date}.json"

    f_fwi_out = np.where(fars_mask, f_fwi, np.nan)
    f_fuel_out = np.where(fars_mask, f_fuel, np.nan)
    slope_out = np.where(fars_mask, slope, np.nan)
    f_topo_out = np.where(fars_mask, f_topo, np.nan)

    coverage = np.full(fwi.shape, np.nan, dtype=np.float32)
    coverage[fars_mask] = np.where(np.isfinite(f_fuel[fars_mask]), 1.0, 0.0)

    print()
    print("WRITING OUTPUTS")
    print("---------------")

    write_raster(f_fwi_path, f_fwi_out, reference)
    write_raster(f_fuel_path, f_fuel_out, reference)
    write_raster(slope_path, slope_out, reference)
    write_raster(f_topo_path, f_topo_out, reference)
    write_raster(fli_path, fli, reference)
    write_raster(coverage_path, coverage, reference)

    output_paths = {
        "F_FWI": f_fwi_path,
        "F_Fuel": f_fuel_path,
        "Slope": slope_path,
        "F_Topo": f_topo_path,
        "FLI": fli_path,
        "FuelCoverage": coverage_path,
    }

    grid_validation = validate_output_grids(output_paths, reference)

    report = {
        "project": "FIRIS - Fars Integrated Fire Information System",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": date,
        "boundary": str(args.boundary),
        "formula": "FLI = 100 * (0.45 * F_FWI + 0.35 * F_Fuel + 0.20 * F_Topo)",
        "weights": {
            "F_FWI": FWI_WEIGHT,
            "F_Fuel": FUEL_WEIGHT,
            "F_Topo": TOPO_WEIGHT,
        },
        "fuel_model": {
            "source": "Sentinel-2-derived continuous Fuel raster",
            "input_path": str(args.fuel_raster),
            "input_range": "0-1",
            "fuel_definition": "Vegetation Availability x Fuel Dryness",
            "normalization": "Direct use of continuous 0-1 fuel raster",
            "excel_used": False,
            "JOIN_VALUE_used": False,
            "Fuelbeds_metric_used": False,
            "resampling_to_FWI": "nearest neighbour",
        },
        "slope_method": {
            "calculation": "Native DEM central finite differences",
            "nodata_filling": False,
            "nodata_policy": "Slope is calculated only where center, north, south, west and east DEM cells are valid.",
            "units": "degrees",
            "reference_degrees": SLOPE_REFERENCE,
            "normalization": "clip(slope_degrees / 45, 0, 1)",
        },
        "target_grid": {
            "reference": "FWI",
            "crs": str(reference["crs"]),
            "width": int(reference["width"]),
            "height": int(reference["height"]),
            "cell_size_x": float(reference["res"][0]),
            "cell_size_y": float(reference["res"][1]),
            "bounds": bounds_dict(reference["bounds"]),
            "transform": [
                float(reference["transform"].a),
                float(reference["transform"].b),
                float(reference["transform"].c),
                float(reference["transform"].d),
                float(reference["transform"].e),
                float(reference["transform"].f),
            ],
        },
        "input_metadata": {
            "FWI": raster_metadata(args.fwi_raster),
            "Fuel": raster_metadata(args.fuel_raster),
            "DEM": raster_metadata(args.dem_raster),
        },
        "alignment": {
            "FWI": "reference grid",
            "Fuel": "nearest-neighbour to FWI",
            "DEM": "native slope calculation without artificial NoData filling, then bilinear alignment to FWI",
        },
        "normalization": {
            "F_FWI": "clip(FWI / 100, 0, 1)",
            "F_Fuel": "direct continuous Fuel raster value in the range 0-1",
            "F_Topo": "clip(slope_degrees / 45, 0, 1)",
        },
        "grid_validation": grid_validation,
        "coverage_inside_fars": {
            "province_pixels": province_pixels,
            "FWI_valid_pixels": fwi_count,
            "FWI_valid_percent": round(100 * fwi_count / province_pixels, 4),
            "Fuel_valid_pixels": fuel_count,
            "Fuel_valid_percent": round(100 * fuel_count / province_pixels, 4),
            "Topo_valid_pixels": topo_count,
            "Topo_valid_percent": round(100 * topo_count / province_pixels, 4),
            "common_valid_pixels": common_count,
            "common_valid_percent": round(100 * common_count / province_pixels, 4),
            "common_valid_percent_of_FWI": round(100 * common_count / max(fwi_count, 1), 4),
        },
        "statistics": {
            "FWI": stats(f_fwi_out),
            "F_FWI": stats(f_fwi_out),
            "Fuel": stats(f_fuel_out),
            "F_Fuel": stats(f_fuel_out),
            "Slope_degrees": stats(slope_out),
            "F_Topo": stats(f_topo_out),
            "FLI": stats(fli),
        },
        "inputs": {
            "FWI": str(args.fwi_raster),
            "Fuel": str(args.fuel_raster),
            "DEM": str(args.dem_raster),
            "Boundary": str(args.boundary),
        },
        "outputs": {
            "F_FWI": str(f_fwi_path),
            "F_Fuel": str(f_fuel_path),
            "Slope": str(slope_path),
            "F_Topo": str(f_topo_path),
            "FLI": str(fli_path),
            "FuelCoverage": str(coverage_path),
        },
        "interpretation": {
            "NoData_policy": "NoData is preserved; missing source coverage is never extrapolated.",
            "coverage_warning": bool(fuel_count < province_pixels),
            "grid_policy": "All final rasters use the exact FWI reference profile.",
            "fuel_policy": "Fuel is a continuous Sentinel-2-derived 0-1 index and is used directly.",
            "excel_policy": "No Excel fuel table is used.",
            "classification_policy": "No external fuel classification lookup is used.",
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

    print()
    print("=" * 70)
    print("FIRIS BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"FLI output    : {fli_path}")
    print(f"Fuel output   : {f_fuel_path}")
    print(f"Fuel coverage : {coverage_path}")
    print(f"Report        : {report_path}")


if __name__ == "__main__":
    main()
