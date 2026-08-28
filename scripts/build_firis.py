#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_geom
from shapely.geometry import shape
from shapely.ops import transform

FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

FWI_MAX = 100.0
SLOPE_REFERENCE = 45.0
OUTPUT_NODATA = -9999.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Integrated Fire Likelihood Index"
    )
    parser.add_argument("--fwi-raster", required=True, type=Path)
    parser.add_argument("--fuel-raster", required=True, type=Path)
    parser.add_argument("--dem-raster", required=True, type=Path)
    parser.add_argument("--fuel-excel", required=True, type=Path)
    parser.add_argument("--fuel-code-column", default="JOIN_VALUE")
    parser.add_argument("--boundary", type=Path, default=Path("fars.geojson"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-date", required=True)

    return parser.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


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


def stats(array: np.ndarray):
    valid = array[np.isfinite(array)]
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


def bounds_dict(bounds):
    return {
        "left": float(bounds.left),
        "bottom": float(bounds.bottom),
        "right": float(bounds.right),
        "top": float(bounds.top),
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


def transform_values(transform):
    return np.array(
        [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        dtype=np.float64,
    )


def grid_matches(reference: dict, metadata: dict, tolerance: float = 1e-10):
    reasons = []
    if metadata["crs"] != str(reference["crs"]):
        reasons.append(f"CRS mismatch: {metadata['crs']} != {reference['crs']}")

    if metadata["width"] != int(reference["width"]):
        reasons.append(
            f"Width mismatch: {metadata['width']} != {reference['width']}"
        )

    if metadata["height"] != int(reference["height"]):
        reasons.append(
            f"Height mismatch: {metadata['height']} != {reference['height']}"
        )

    reference_transform = transform_values(reference["transform"])
    output_transform = np.array(metadata["transform"], dtype=np.float64)

    if not np.allclose(
        reference_transform, output_transform, rtol=0.0, atol=tolerance
    ):
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

    if not np.allclose(
        reference_bounds, output_bounds, rtol=0.0, atol=tolerance
    ):
        reasons.append("Bounds mismatch")

    return len(reasons) == 0, reasons


def validate_output_grids(output_paths: dict, reference: dict):
    print("\n" + "=" * 70)
    print("FINAL GRID VALIDATION")
    print("=" * 70)

    failures = []
    validation = {}

    for name, path in output_paths.items():
        metadata = raster_metadata(path)
        ok, reasons = grid_matches(reference, metadata)
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

    if failures:
        raise RuntimeError(
            "One or more output rasters do not match the reference grid."
        )

    print("✓ ALL OUTPUT RASTERS MATCH THE REFERENCE GRID")
    return validation


def load_boundary_mask(boundary_path: Path, reference: dict):
    with boundary_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        raise ValueError(f"Boundary contains no features: {boundary_path}")

    geometries = []
    for feature in features:
        geom = feature.get("geometry")
        if geom:
            geometries.append(geom)

    if not geometries:
        raise ValueError(f"Boundary contains no geometries: {boundary_path}")

    source_crs = "EPSG:4326"
    crs_obj = geojson.get("crs")
    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties", {})
        name = props.get("name") or props.get("href")
        if isinstance(name, str) and name:
            source_crs = name

    target_crs = reference["crs"]

    # تبدیل دقیق ژئومتری به CRS مرجع
    if str(target_crs).upper() != str(source_crs).upper():
        project_func = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        ).transform
        reprojected_geoms = []
        for g in geometries:
            s_geom = shape(g)
            s_reproj = transform(project_func, s_geom)
            reprojected_geoms.append(s_reproj.__geo_interface__)
        geometries = reprojected_geoms

    # تولید ماسک درون مرز - all_touched=True تضمین می‌کند حاشیه‌ها حذف نشوند
    mask = geometry_mask(
        geometries,
        out_shape=(reference["height"], reference["width"]),
        transform=reference["transform"],
        invert=True,
        all_touched=True,
    )

    count = int(np.sum(mask))
    if count == 0:
        raise ValueError(
            "Region boundary does not overlap the reference grid."
        )

    print("\nREGION BOUNDARY MASK")
    print("--------------------")
    print(f"Boundary file      : {boundary_path}")
    print(f"Boundary CRS       : {source_crs}")
    print(f"Pixels inside mask : {count:,}")

    return mask


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

    return data, reference


def align_to_reference(path: Path, reference: dict, resampling: Resampling):
    destination = np.full(
        (reference["height"], reference["width"]), np.nan, dtype=np.float32
    )

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")

        source = clean_array(src.read(1), src.nodata)

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
        return (xres * meters_lon, yres * meters_lat)

    raise ValueError("Unsupported DEM coordinate system.")


def calculate_native_slope(dem_path: Path):
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError("DEM raster has no CRS.")

        dem = clean_array(src.read(1), src.nodata)
        valid = np.isfinite(dem)
        if not np.any(valid):
            raise ValueError("DEM contains no valid pixels.")

        dx, dy = metric_cell_size(src)
        fill_value = float(np.nanmedian(dem))
        working = np.where(valid, dem, fill_value).astype(np.float32)

        gradient_y, gradient_x = np.gradient(working, dy, dx)
        slope_rad = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
        slope = np.degrees(slope_rad).astype(np.float32)

        slope[~valid] = np.nan
        slope[~np.isfinite(slope)] = np.nan

        return slope, src.transform, src.crs


def align_slope_to_reference(dem_path: Path, reference: dict):
    slope, dem_transform, dem_crs = calculate_native_slope(dem_path)
    destination = np.full(
        (reference["height"], reference["width"]), np.nan, dtype=np.float32
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
        resampling=Resampling.bilinear,
    )

    destination[~np.isfinite(destination)] = np.nan
    return destination


def find_column(dataframe, candidates):
    lookup = {str(c).strip().lower(): c for c in dataframe.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def normalize_column(series):
    values = (
        pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    minimum = float(values.min())
    maximum = float(values.max())

    if math.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
    return (values - minimum) / (maximum - minimum)


def load_fuel_mapping(excel_path: Path, requested_code_column: str):
    workbook = pd.ExcelFile(excel_path)
    if "Fuelbeds_metric" not in workbook.sheet_names:
        raise ValueError("Sheet 'Fuelbeds_metric' not found in Excel file.")

    df = pd.read_excel(excel_path, sheet_name="Fuelbeds_metric")
    code_col = find_column(
        df,
        [
            requested_code_column,
            "JOIN_VALUE",
            "FUELBED",
            "FUELBED_ID",
            "FUEL_CODE",
        ],
    )
    if code_col is None:
        raise ValueError("Could not identify fuel-code column.")

    woody_col = find_column(df, ["Woody Cover (%)", "Woody Cover"])
    w1_col = find_column(
        df, ["W_1hLoad (Mg/ha)", "W_1h Load (Mg/ha)", "W_1hLoad"]
    )
    w10_col = find_column(
        df, ["W_10hLoad (Mg/ha)", "W_10h Load (Mg/ha)", "W_10hLoad"]
    )
    w100_col = find_column(
        df, ["W_100hLoad (Mg/ha)", "W_100h Load (Mg/ha)", "W_100hLoad"]
    )
    w1000_col = find_column(
        df, ["W_1000hLoad (Mg/ha)", "W_1000h Load (Mg/ha)", "W_1000hLoad"]
    )
    litter_cover_col = find_column(df, ["Litter Cover (%)", "Litter Cover"])
    litter_depth_col = find_column(df, ["L_depth (cm)", "L_depth"])

    if w1_col is None:
        raise ValueError("W_1hLoad column was not found.")

    df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
    fine = normalize_column(df[w1_col])

    dead_parts = []
    if w10_col is not None:
        dead_parts.append((0.50, normalize_column(df[w10_col])))
    if w100_col is not None:
        dead_parts.append((0.30, normalize_column(df[w100_col])))
    if w1000_col is not None:
        dead_parts.append((0.20, normalize_column(df[w1000_col])))

    dead = pd.Series(0.0, index=df.index)
    total_weight = sum(weight for weight, _ in dead_parts)
    for weight, values in dead_parts:
        dead += weight * values
    if total_weight > 0:
        dead /= total_weight

    woody = (
        normalize_column(df[woody_col])
        if woody_col is not None
        else pd.Series(0.0, index=df.index)
    )

    litter_parts = []
    if litter_cover_col is not None:
        litter_parts.append(normalize_column(df[litter_cover_col]))
    if litter_depth_col is not None:
        litter_parts.append(normalize_column(df[litter_depth_col]))

    litter = (
        sum(litter_parts) / len(litter_parts)
        if litter_parts
        else pd.Series(0.0, index=df.index)
    )

    fuel_score = (
        0.35 * fine + 0.30 * dead + 0.15 * woody + 0.10 * litter + 0.10 * woody
    ).clip(0.0, 1.0)
    df["_F_Fuel"] = fuel_score
    df = df.dropna(subset=[code_col]).drop_duplicates(
        subset=[code_col], keep="last"
    )

    mapping = {}
    for code, score in zip(df[code_col], df["_F_Fuel"]):
        try:
            cv, sv = float(code), float(score)
            if math.isfinite(cv) and math.isfinite(sv):
                mapping[cv] = sv
        except (TypeError, ValueError):
            continue

    return mapping


def fuel_to_score(fuel_codes, mapping):
    output = np.full(fuel_codes.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(fuel_codes)
    if not np.any(valid):
        raise ValueError("Aligned Fuel raster contains no valid pixels.")

    unique_codes = np.unique(fuel_codes[valid])
    unmapped = []

    for code in unique_codes:
        code_float = float(code)
        mask = fuel_codes == code
        if code_float in mapping:
            output[mask] = mapping[code_float]
        else:
            unmapped.append(code_float)

    return output, unmapped


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

    output = np.where(np.isfinite(array), array, OUTPUT_NODATA).astype(
        np.float32
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)


def main():
    args = parse_args()

    require_file(args.fwi_raster, "FWI raster")
    require_file(args.fuel_raster, "Fuel raster")
    require_file(args.dem_raster, "DEM raster")
    require_file(args.fuel_excel, "Fuel Excel")
    require_file(args.boundary, "Boundary")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. خواندن FWI به عنوان رفرنس
    fwi, reference = read_fwi(args.fwi_raster)

    # 2. ساخت ماسک مرز منطقه
    boundary_mask = load_boundary_mask(args.boundary, reference)

    # 3. هم‌پوشانی لایه‌های رستر روی FWI
    fuel_codes = align_to_reference(
        args.fuel_raster, reference, Resampling.nearest
    )
    slope = align_slope_to_reference(args.dem_raster, reference)

    # 4. نگاشت و نرمال‌سازی سوخت
    fuel_mapping = load_fuel_mapping(args.fuel_excel, args.fuel_code_column)
    f_fuel, unmapped_codes = fuel_to_score(fuel_codes, fuel_mapping)

    # 5. نرمال‌سازی شاخص‌ها
    f_fwi = np.clip(fwi / FWI_MAX, 0.0, 1.0)
    f_topo = np.clip(slope / SLOPE_REFERENCE, 0.0, 1.0)

    # 6. ترکیب لایه‌ها صرفاً درون مرز (Clamping to Boundary Mask)
    common_mask = (
        boundary_mask
        & np.isfinite(f_fwi)
        & np.isfinite(f_fuel)
        & np.isfinite(f_topo)
    )

    if int(np.sum(common_mask)) == 0:
        raise RuntimeError("No valid data pixels exist inside the boundary.")

    # 7. محاسبه FLI نهایی
    fli = np.full(fwi.shape, np.nan, dtype=np.float32)
    fli[common_mask] = 100.0 * (
        FWI_WEIGHT * f_fwi[common_mask]
        + FUEL_WEIGHT * f_fuel[common_mask]
        + TOPO_WEIGHT * f_topo[common_mask]
    )
    fli = np.clip(fli, 0.0, 100.0)

    # 8. ذخیره خروجی‌ها (اعمال دقیق ماسک مرز روی همه لایه‌ها)
    date = args.run_date
    fli_path = args.output_dir / f"fli_fars_{date}.tif"
    f_fwi_path = args.output_dir / f"f_fwi_fars_{date}.tif"
    f_fuel_path = args.output_dir / f"f_fuel_fars_{date}.tif"
    slope_path = args.output_dir / f"slope_fars_{date}.tif"
    f_topo_path = args.output_dir / f"f_topo_fars_{date}.tif"

    write_raster(fli_path, fli, reference)
    write_raster(
        f_fwi_path, np.where(boundary_mask, f_fwi, np.nan), reference
    )
    write_raster(
        f_fuel_path, np.where(boundary_mask, f_fuel, np.nan), reference
    )
    write_raster(slope_path, np.where(boundary_mask, slope, np.nan), reference)
    write_raster(
        f_topo_path, np.where(boundary_mask, f_topo, np.nan), reference
    )

    # 9. اعتبارسنجی نهایی
    output_paths = {
        "FLI": fli_path,
        "F_FWI": f_fwi_path,
        "F_Fuel": f_fuel_path,
        "Slope": slope_path,
        "F_Topo": f_topo_path,
    }
    validate_output_grids(output_paths, reference)

    print("\n" + "=" * 70)
    print("SUCCESS: Output layers strictly clamped to boundary.")
    print("=" * 70)


if __name__ == "__main__":
    main()
