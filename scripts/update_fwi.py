
#!/usr/bin/env python3

"""
FIRIS - ECMWF FWI Downloader
============================

Downloads ECMWF Fire Weather Index (FWI) from
Copernicus GWIS WMS.

IMPORTANT
---------
The spatial BBOX is calculated directly from:

    fars.geojson

A small margin is added around the actual province
extent so the complete Fars Province is covered.

This script does NOT calculate FLI.
It only downloads the FWI raster used by FIRIS.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import rasterio
import requests


# ============================================================
# COPERNICUS GWIS
# ============================================================

WMS_URL = (
    "https://maps.effis.emergency.copernicus.eu/gwis"
)

LAYER_NAME = "ecmwf.fwi"

WMS_VERSION = "1.1.1"

SRS = "EPSG:4326"

WIDTH = 2000
HEIGHT = 2000

REQUEST_TIMEOUT_SECONDS = 180


# ============================================================
# BOUNDARY MARGIN
# ============================================================

# Margin in geographic degrees.
#
# The current Fars boundary extends farther east than
# the old fixed BBOX. A 0.10 degree margin provides
# safe coverage around the complete province.
#
# This does not alter FLI calculations.
# ============================================================

BOUNDARY_MARGIN_DEG = 0.10


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Download ECMWF Fire Weather Index "
            "(FWI) for Fars from Copernicus GWIS WMS."
        )
    )

    parser.add_argument(
        "--date",
        dest="target_date",
        default=None,
        help=(
            "Date in YYYY-MM-DD format. "
            "Default: tomorrow UTC."
        ),
    )

    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("fars.geojson"),
        help=(
            "Fars Province GeoJSON boundary."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Download again even if output already exists."
        ),
    )

    return parser.parse_args()


# ============================================================
# DATE
# ============================================================

def get_target_date(
    value: str | None
) -> str:

    if value is not None:

        try:

            return datetime.strptime(
                value,
                "%Y-%m-%d"
            ).date().isoformat()

        except ValueError as error:

            raise SystemExit(
                f"Invalid date: {value}. "
                "Expected format YYYY-MM-DD."
            ) from error

    tomorrow = (
        datetime.now(
            timezone.utc
        ).date()
        +
        timedelta(days=1)
    )

    return tomorrow.isoformat()


# ============================================================
# LOAD BOUNDARY
# ============================================================

def load_boundary(
    boundary_path: Path
) -> dict[str, Any]:

    if not boundary_path.is_file():

        raise FileNotFoundError(
            f"Boundary not found: {boundary_path}"
        )

    with boundary_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        geojson = json.load(file)

    if (
        geojson.get("type")
        != "FeatureCollection"
    ):

        raise ValueError(
            "Boundary GeoJSON must be a "
            "FeatureCollection."
        )

    features = geojson.get(
        "features",
        []
    )

    if not features:

        raise ValueError(
            "Boundary contains no features."
        )

    return geojson


# ============================================================
# EXTRACT COORDINATES
# ============================================================

def extract_coordinates(
    value,
    output: list[tuple[float, float]]
) -> None:

    if (
        isinstance(value, list)
        and
        len(value) >= 2
        and
        isinstance(
            value[0],
            (int, float)
        )
        and
        isinstance(
            value[1],
            (int, float)
        )
    ):

        output.append(
            (
                float(value[0]),
                float(value[1])
            )
        )

        return

    if isinstance(
        value,
        list
    ):

        for item in value:

            extract_coordinates(
                item,
                output
            )


# ============================================================
# CALCULATE REAL BBOX
# ============================================================

def calculate_boundary_bbox(
    geojson: dict[str, Any],
    margin_deg: float
) -> tuple[
    float,
    float,
    float,
    float
]:

    coordinates = []

    for feature in geojson.get(
        "features",
        []
    ):

        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            continue

        geometry_coordinates = (
            geometry.get(
                "coordinates"
            )
        )

        extract_coordinates(
            geometry_coordinates,
            coordinates
        )

    if not coordinates:

        raise ValueError(
            "No coordinates found in "
            "Fars boundary."
        )

    longitudes = [
        p[0]
        for p in coordinates
    ]

    latitudes = [
        p[1]
        for p in coordinates
    ]

    west = min(
        longitudes
    )

    east = max(
        longitudes
    )

    south = min(
        latitudes
    )

    north = max(
        latitudes
    )

    # Add margin around actual boundary.
    west -= margin_deg
    south -= margin_deg
    east += margin_deg
    north += margin_deg

    return (
        west,
        south,
        east,
        north
    )


# ============================================================
# FORMAT BBOX
# ============================================================

def format_bbox(
    west: float,
    south: float,
    east: float,
    north: float
) -> str:

    return (
        f"{west:.8f},"
        f"{south:.8f},"
        f"{east:.8f},"
        f"{north:.8f}"
    )


# ============================================================
# WMS ERROR DETECTION
# ============================================================

def response_is_error_payload(
    response: requests.Response
) -> bool:

    content_type = (
        response.headers
        .get(
            "Content-Type",
            ""
        )
        .lower()
    )

    sample = (
        response.content[:1000]
        .lstrip()
        .lower()
    )

    error_content_types = (
        "text/xml",
        "application/xml",
        "text/html",
        "application/json",
        "text/plain",
    )

    if any(
        item in content_type
        for item in error_content_types
    ):

        return True

    error_prefixes = (
        b"<?xml",
        b"<serviceexception",
        b"<serviceexceptionreport",
        b"<html",
        b"<!doctype html",
        b"{",
        b"[",
    )

    return sample.startswith(
        error_prefixes
    )


# ============================================================
# SAVE ERROR RESPONSE
# ============================================================

def write_error_response(
    output_dir: Path,
    target_date: str,
    response: requests.Response
) -> Path:

    error_path = (
        output_dir
        /
        f"fwi_download_error_{target_date}.txt"
    )

    header = (
        f"HTTP status: {response.status_code}\n"
        f"Content-Type: "
        f"{response.headers.get('Content-Type', '')}\n"
        f"Request URL: {response.url}\n"
        "\n"
        "Response body:\n"
        "------------------------------------------------------------\n"
    ).encode(
        "utf-8"
    )

    error_path.write_bytes(
        header
        +
        response.content
    )

    return error_path


# ============================================================
# VALIDATE GEOTIFF
# ============================================================

def validate_geotiff(
    tif_path: Path
) -> dict[str, Any]:

    try:

        with rasterio.open(
            tif_path
        ) as dataset:

            if dataset.driver != "GTiff":

                raise RuntimeError(
                    "Downloaded file is not GTiff."
                )

            if dataset.count < 1:

                raise RuntimeError(
                    "GeoTIFF contains no raster bands."
                )

            if (
                dataset.width < 1
                or
                dataset.height < 1
            ):

                raise RuntimeError(
                    "GeoTIFF dimensions are invalid."
                )

            first_band = dataset.read(
                1,
                masked=True
            )

            valid_pixel_count = int(
                first_band.count()
            )

            if valid_pixel_count == 0:

                raise RuntimeError(
                    "GeoTIFF contains no "
                    "valid FWI pixels."
                )

            return {

                "driver":
                    dataset.driver,

                "crs":
                    (
                        str(dataset.crs)
                        if dataset.crs
                        else None
                    ),

                "width":
                    dataset.width,

                "height":
                    dataset.height,

                "band_count":
                    dataset.count,

                "dtype":
                    dataset.dtypes[0],

                "nodata":
                    dataset.nodata,

                "bounds": {

                    "left":
                        float(
                            dataset.bounds.left
                        ),

                    "bottom":
                        float(
                            dataset.bounds.bottom
                        ),

                    "right":
                        float(
                            dataset.bounds.right
                        ),

                    "top":
                        float(
                            dataset.bounds.top
                        )
                },

                "resolution": {

                    "x":
                        float(
                            dataset.res[0]
                        ),

                    "y":
                        float(
                            abs(
                                dataset.res[1]
                            )
                        )
                },

                "valid_pixel_count":
                    valid_pixel_count,

                "fwi_min":
                    float(
                        first_band.min()
                    ),

                "fwi_max":
                    float(
                        first_band.max()
                    ),

                "fwi_mean":
                    float(
                        first_band.mean()
                    )
            }

    except Exception as error:

        raise RuntimeError(
            "Downloaded file cannot be validated "
            f"as a readable GeoTIFF: {error}"
        ) from error


# ============================================================
# WRITE METADATA
# ============================================================

def write_metadata(
    path: Path,
    metadata: dict[str, Any]
) -> None:

    path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    target_date = get_target_date(
        args.target_date
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_dir = (
        project_root
        /
        "data"
        /
        "raw"
        /
        "fwi"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_tif = (
        output_dir
        /
        f"fwi_ecmwf_fars_{target_date}.tif"
    )

    output_json = (
        output_dir
        /
        f"fwi_ecmwf_fars_{target_date}.json"
    )


    # ========================================================
    # LOAD REAL FARS BOUNDARY
    # ========================================================

    geojson = load_boundary(
        args.boundary
    )

    (
        west,
        south,
        east,
        north
    ) = calculate_boundary_bbox(
        geojson,
        BOUNDARY_MARGIN_DEG
    )

    fars_bbox = format_bbox(
        west,
        south,
        east,
        north
    )


    # ========================================================
    # EXISTING FILE CHECK
    # ========================================================

    if (
        output_tif.exists()
        and
        not args.overwrite
    ):

        print(
            f"FWI file already exists: "
            f"{output_tif}"
        )

        print(
            "Use --overwrite to download again."
        )

        return


    # ========================================================
    # WMS PARAMETERS
    # ========================================================

    params = {

        "SERVICE":
            "WMS",

        "VERSION":
            WMS_VERSION,

        "REQUEST":
            "GetMap",

        "LAYERS":
            LAYER_NAME,

        "STYLES":
            "",

        "SRS":
            SRS,

        "BBOX":
            fars_bbox,

        "WIDTH":
            str(WIDTH),

        "HEIGHT":
            str(HEIGHT),

        "FORMAT":
            "image/tiff",

        "TRANSPARENT":
            "FALSE",

        "TIME":
            target_date
    }


    # ========================================================
    # LOG
    # ========================================================

    print("")
    print("=" * 70)
    print("FIRIS - ECMWF FWI DOWNLOADER")
    print("=" * 70)

    print(
        f"Target date : {target_date}"
    )

    print(
        f"Layer       : {LAYER_NAME}"
    )

    print(
        f"Boundary    : {args.boundary}"
    )

    print(
        f"Margin      : "
        f"{BOUNDARY_MARGIN_DEG} degree"
    )

    print(
        f"BBOX        : {fars_bbox}"
    )

    print(
        f"Output TIFF : {output_tif}"
    )

    print("")


    # ========================================================
    # DOWNLOAD
    # ========================================================

    try:

        response = requests.get(
            WMS_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={
                "User-Agent":
                    (
                        "FIRIS/1.0 "
                        "(GitHub Actions; "
                        "Fars Fire Risk System)"
                    )
            }
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise SystemExit(
            f"FWI download failed: {error}"
        ) from error


    print(
        f"HTTP status     : "
        f"{response.status_code}"
    )

    print(
        "Content-Type    : "
        f"{response.headers.get('Content-Type', '')}"
    )

    print(
        f"Downloaded bytes: "
        f"{len(response.content)}"
    )


    if not response.content:

        raise SystemExit(
            "FWI download failed: "
            "empty response."
        )


    # ========================================================
    # WMS ERROR DOCUMENT
    # ========================================================

    if response_is_error_payload(
        response
    ):

        error_path = (
            write_error_response(
                output_dir,
                target_date,
                response
            )
        )

        raise SystemExit(
            "FWI download failed: WMS returned "
            "an error document instead of GeoTIFF.\n"
            f"Saved to: {error_path}"
        )


    # ========================================================
    # TEMPORARY FILE
    # ========================================================

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".tif",
            prefix=(
                f".fwi_ecmwf_fars_"
                f"{target_date}_"
            ),
            dir=output_dir,
            delete=False
        ) as temporary_file:

            temporary_file.write(
                response.content
            )

            temporary_path = Path(
                temporary_file.name
            )


        # ----------------------------------------------------
        # Validate before replacing existing output.
        # ----------------------------------------------------

        raster_info = validate_geotiff(
            temporary_path
        )


        # ----------------------------------------------------
        # Make the new file authoritative.
        # ----------------------------------------------------

        os.replace(
            temporary_path,
            output_tif
        )

        temporary_path = None


    except RuntimeError as error:

        invalid_file = (
            output_dir
            /
            f"fwi_invalid_response_"
            f"{target_date}.tif"
        )

        if (
            temporary_path is not None
            and
            temporary_path.exists()
        ):

            os.replace(
                temporary_path,
                invalid_file
            )

            temporary_path = None

        raise SystemExit(
            f"FWI validation failed: {error}\n"
            f"Invalid file saved to: {invalid_file}"
        ) from error


    finally:

        if (
            temporary_path is not None
            and
            temporary_path.exists()
        ):

            temporary_path.unlink()


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "project":
            "FIRIS",

        "indicator":
            "ECMWF Fire Weather Index (FWI)",

        "source":
            "Copernicus GWIS WMS",

        "source_url":
            WMS_URL,

        "layer":
            LAYER_NAME,

        "target_date":
            target_date,

        "downloaded_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "boundary_file":
            str(args.boundary),

        "boundary_margin_degree":
            BOUNDARY_MARGIN_DEG,

        "boundary_bbox_with_margin": {

            "west":
                west,

            "south":
                south,

            "east":
                east,

            "north":
                north
        },

        "wms_version":
            WMS_VERSION,

        "srs":
            SRS,

        "width":
            WIDTH,

        "height":
            HEIGHT,

        "request_parameters":
            params,

        "request_url":
            response.url,

        "http_status":
            response.status_code,

        "content_type":
            response.headers.get(
                "Content-Type",
                ""
            ),

        "output_file":
            output_tif.name,

        "output_file_size_bytes":
            output_tif.stat().st_size,

        "raster":
            raster_info
    }


    write_metadata(
        output_json,
        metadata
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("")
    print("=" * 70)
    print("FWI DOWNLOAD COMPLETED")
    print("=" * 70)

    print(
        f"GeoTIFF : {output_tif}"
    )

    print(
        f"Metadata: {output_json}"
    )

    print("")

    print("REQUEST BBOX")

    print(
        f"West  : {west:.8f}"
    )

    print(
        f"South : {south:.8f}"
    )

    print(
        f"East  : {east:.8f}"
    )

    print(
        f"North : {north:.8f}"
    )

    print("")

    print("OUTPUT RASTER")

    print(
        f"CRS   : {raster_info['crs']}"
    )

    print(
        f"Size  : "
        f"{raster_info['width']} x "
        f"{raster_info['height']}"
    )

    print(
        "Bounds:"
    )

    print(
        f"  West  = "
        f"{raster_info['bounds']['left']:.8f}"
    )

    print(
        f"  South = "
        f"{raster_info['bounds']['bottom']:.8f}"
    )

    print(
        f"  East  = "
        f"{raster_info['bounds']['right']:.8f}"
    )

    print(
        f"  North = "
        f"{raster_info['bounds']['top']:.8f}"
    )

    print("")

    print(
        "FWI statistics: "
        f"min={raster_info['fwi_min']:.3f}, "
        f"max={raster_info['fwi_max']:.3f}, "
        f"mean={raster_info['fwi_mean']:.3f}"
    )

    print(
        f"Valid pixels: "
        f"{raster_info['valid_pixel_count']}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
