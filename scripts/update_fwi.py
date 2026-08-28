#!/usr/bin/env python3

"""
FIRIS - ECMWF FWI Downloader
============================

Downloads ECMWF Fire Weather Index (FWI)
from Copernicus GWIS WMS.

The requested spatial extent is calculated directly
from fars.geojson.

No FLI calculation is performed here.
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
# GWIS WMS
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

# Extra margin around the exact Fars boundary.
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
        help="Fars Province GeoJSON boundary."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FWI for the target date."
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
                "Expected YYYY-MM-DD."
            ) from error

    tomorrow = (
        datetime.now(
            timezone.utc
        ).date()
        + timedelta(days=1)
    )

    return tomorrow.isoformat()


# ============================================================
# LOAD FARS GEOJSON
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

    if geojson.get("type") != "FeatureCollection":

        raise ValueError(
            "fars.geojson must be a "
            "GeoJSON FeatureCollection."
        )

    features = geojson.get(
        "features",
        []
    )

    if not features:

        raise ValueError(
            "fars.geojson contains no features."
        )

    return geojson


# ============================================================
# EXTRACT ALL COORDINATES
# ============================================================

def extract_coordinates(
    value,
    output: list[tuple[float, float]]
) -> None:

    # A coordinate pair.
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(
            value[0],
            (int, float)
        )
        and isinstance(
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

    # Nested coordinate arrays.
    if isinstance(value, list):

        for item in value:

            extract_coordinates(
                item,
                output
            )


# ============================================================
# CALCULATE BBOX FROM FARS
# ============================================================

def calculate_fars_bbox(
    geojson: dict[str, Any],
    margin_deg: float
) -> tuple[
    float,
    float,
    float,
    float
]:

    coordinates: list[
        tuple[float, float]
    ] = []

    for feature in geojson.get(
        "features",
        []
    ):

        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            continue

        coords = geometry.get(
            "coordinates"
        )

        extract_coordinates(
            coords,
            coordinates
        )

    if not coordinates:

        raise ValueError(
            "No coordinates found in "
            "fars.geojson."
        )

    longitudes = [
        point[0]
        for point in coordinates
    ]

    latitudes = [
        point[1]
        for point in coordinates
    ]

    west = min(longitudes)
    east = max(longitudes)

    south = min(latitudes)
    north = max(latitudes)

    # Add safety margin.
    west -= margin_deg
    east += margin_deg
    south -= margin_deg
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

    error_types = (
        "text/xml",
        "application/xml",
        "text/html",
        "application/json",
        "text/plain",
    )

    if any(
        item in content_type
        for item in error_types
    ):

        return True

    prefixes = (
        b"<?xml",
        b"<serviceexception",
        b"<serviceexceptionreport",
        b"<html",
        b"<!doctype html",
        b"{",
        b"[",
    )

    return sample.startswith(
        prefixes
    )


# ============================================================
# SAVE ERROR RESPONSE
# ============================================================

def save_error_response(
    output_dir: Path,
    target_date: str,
    response: requests.Response
) -> Path:

    path = (
        output_dir
        /
        f"fwi_download_error_{target_date}.txt"
    )

    header = (
        f"HTTP status: {response.status_code}\n"
        f"Content-Type: "
        f"{response.headers.get('Content-Type', '')}\n"
        f"URL: {response.url}\n"
        "\n"
        "Response body:\n"
        "------------------------------------------------------------\n"
    ).encode("utf-8")

    path.write_bytes(
        header + response.content
    )

    return path


# ============================================================
# VALIDATE GEOTIFF
# ============================================================

def validate_geotiff(
    tif_path: Path
) -> dict[str, Any]:

    try:

        with rasterio.open(
            tif_path
        ) as src:

            if src.driver != "GTiff":

                raise RuntimeError(
                    f"Expected GTiff, got "
                    f"{src.driver}"
                )

            if src.count < 1:

                raise RuntimeError(
                    "GeoTIFF has no raster bands."
                )

            if (
                src.width <= 0
                or
                src.height <= 0
            ):

                raise RuntimeError(
                    "Invalid raster dimensions."
                )

            data = src.read(
                1,
                masked=True
            )

            valid_count = int(
                data.count()
            )

            if valid_count == 0:

                raise RuntimeError(
                    "FWI GeoTIFF contains no "
                    "valid pixels."
                )

            return {

                "driver":
                    src.driver,

                "crs":
                    str(src.crs)
                    if src.crs
                    else None,

                "width":
                    int(src.width),

                "height":
                    int(src.height),

                "resolution": {

                    "x":
                        float(src.res[0]),

                    "y":
                        float(
                            abs(src.res[1])
                        )
                },

                "bounds": {

                    "left":
                        float(
                            src.bounds.left
                        ),

                    "bottom":
                        float(
                            src.bounds.bottom
                        ),

                    "right":
                        float(
                            src.bounds.right
                        ),

                    "top":
                        float(
                            src.bounds.top
                        )
                },

                "nodata":
                    src.nodata,

                "dtype":
                    src.dtypes[0],

                "valid_pixel_count":
                    valid_count,

                "min":
                    float(data.min()),

                "max":
                    float(data.max()),

                "mean":
                    float(data.mean())
            }

    except Exception as error:

        raise RuntimeError(
            "Downloaded file could not be "
            f"validated as GeoTIFF: {error}"
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
        / "data"
        / "raw"
        / "fwi"
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
    # READ FARS BOUNDARY
    # ========================================================

    geojson = load_boundary(
        args.boundary
    )

    (
        west,
        south,
        east,
        north
    ) = calculate_fars_bbox(
        geojson,
        BOUNDARY_MARGIN_DEG
    )

    bbox = format_bbox(
        west,
        south,
        east,
        north
    )


    # ========================================================
    # PRINT REQUEST EXTENT
    # ========================================================

    print("")
    print("=" * 70)
    print("FIRIS - ECMWF FWI DOWNLOAD")
    print("=" * 70)

    print("")
    print(
        f"Target date : {target_date}"
    )

    print(
        f"Boundary    : {args.boundary}"
    )

    print(
        f"Margin      : "
        f"{BOUNDARY_MARGIN_DEG} degree"
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

    print(
        f"BBOX string: {bbox}"
    )


    # ========================================================
    # EXISTING OUTPUT
    # ========================================================

    if (
        output_tif.exists()
        and
        not args.overwrite
    ):

        print("")
        print(
            "FWI file already exists:"
        )

        print(
            output_tif
        )

        print(
            "Use --overwrite to replace it."
        )

        return


    # ========================================================
    # WMS REQUEST
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
            bbox,

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


    print("")
    print(
        "Downloading FWI..."
    )


    # ========================================================
    # REQUEST
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
                        "(GitHub Actions)"
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
        f"Content-Type    : "
        f"{response.headers.get('Content-Type', '')}"
    )

    print(
        f"Downloaded bytes: "
        f"{len(response.content)}"
    )


    # ========================================================
    # EMPTY RESPONSE
    # ========================================================

    if not response.content:

        raise SystemExit(
            "FWI download failed: "
            "empty response."
        )


    # ========================================================
    # WMS ERROR
    # ========================================================

    if response_is_error_payload(
        response
    ):

        error_path = (
            save_error_response(
                output_dir,
                target_date,
                response
            )
        )

        raise SystemExit(
            "FWI download failed: "
            "WMS returned an error document.\n"
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
        ) as temp:

            temp.write(
                response.content
            )

            temporary_path = Path(
                temp.name
            )


        # ----------------------------------------------------
        # VALIDATE BEFORE REPLACING FINAL FILE
        # ----------------------------------------------------

        raster_info = validate_geotiff(
            temporary_path
        )


        # ----------------------------------------------------
        # ATOMIC REPLACEMENT
        # ----------------------------------------------------

        os.replace(
            temporary_path,
            output_tif
        )

        temporary_path = None


    except RuntimeError as error:

        invalid_path = (
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
                invalid_path
            )

            temporary_path = None

        raise SystemExit(
            f"FWI validation failed: {error}\n"
            f"Invalid file saved to: {invalid_path}"
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

        "boundary_bbox":

            {
                "west":
                    west,

                "south":
                    south,

                "east":
                    east,

                "north":
                    north
            },

        "request_bbox":
            bbox,

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
    # FINAL OUTPUT
    # ========================================================

    print("")
    print("=" * 70)
    print("FWI DOWNLOAD COMPLETED SUCCESSFULLY")
    print("=" * 70)

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
        f"CRS        : "
        f"{raster_info['crs']}"
    )

    print(
        f"Size       : "
        f"{raster_info['width']} x "
        f"{raster_info['height']}"
    )

    print(
        f"Resolution : "
        f"{raster_info['resolution']['x']}, "
        f"{raster_info['resolution']['y']}"
    )

    print(
        "Bounds     : "
        f"W={raster_info['bounds']['left']:.8f}, "
        f"S={raster_info['bounds']['bottom']:.8f}, "
        f"E={raster_info['bounds']['right']:.8f}, "
        f"N={raster_info['bounds']['top']:.8f}"
    )

    print("")
    print(
        f"FWI min    : "
        f"{raster_info['min']:.3f}"
    )

    print(
        f"FWI max    : "
        f"{raster_info['max']:.3f}"
    )

    print(
        f"FWI mean   : "
        f"{raster_info['mean']:.3f}"
    )

    print(
        f"Valid pixels: "
        f"{raster_info['valid_pixel_count']}"
    )

    print("")
    print(
        f"GeoTIFF : {output_tif}"
    )

    print(
        f"Metadata: {output_json}"
    )

    print("")
    print(
        "IMPORTANT: FLI calculation was not modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
