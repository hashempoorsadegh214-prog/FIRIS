#!/usr/bin/env python3

"""
FIRIS - ECMWF FWI Downloader
============================

Automatically downloads ECMWF FWI for TOMORROW
according to Iran local date/time.

Timezone:
    Asia/Tehran

The requested BBOX is calculated directly from:
    fars.geojson

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
from zoneinfo import ZoneInfo

import rasterio
import requests


# ============================================================
# CONFIGURATION
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

BOUNDARY_MARGIN_DEG = 0.10

IRAN_TIMEZONE = ZoneInfo(
    "Asia/Tehran"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Automatically download ECMWF FWI "
            "for tomorrow in Iran."
        )
    )

    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("fars.geojson"),
        help="Fars Province GeoJSON."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target-date FWI."
    )

    return parser.parse_args()


# ============================================================
# AUTOMATIC FORECAST DATE
# ============================================================

def get_forecast_date():

    now_iran = datetime.now(
        IRAN_TIMEZONE
    )

    forecast_datetime = (
        now_iran
        + timedelta(days=1)
    )

    return (
        now_iran,
        forecast_datetime.date()
    )


# ============================================================
# LOAD GEOJSON
# ============================================================

def load_boundary(
    path: Path
) -> dict[str, Any]:

    if not path.is_file():

        raise FileNotFoundError(
            f"Boundary not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        geojson = json.load(file)

    if (
        geojson.get("type")
        != "FeatureCollection"
    ):

        raise ValueError(
            "Boundary must be a "
            "GeoJSON FeatureCollection."
        )

    if not geojson.get(
        "features"
    ):

        raise ValueError(
            "Boundary contains no features."
        )

    return geojson


# ============================================================
# EXTRACT COORDINATES
# ============================================================

def extract_coordinates(
    value,
    output
):

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
# BBOX FROM FARS
# ============================================================

def calculate_bbox(
    geojson,
    margin_deg
):

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

        extract_coordinates(
            geometry.get(
                "coordinates"
            ),
            coordinates
        )

    if not coordinates:

        raise ValueError(
            "No coordinates found in "
            "fars.geojson."
        )

    west = min(
        p[0]
        for p in coordinates
    )

    east = max(
        p[0]
        for p in coordinates
    )

    south = min(
        p[1]
        for p in coordinates
    )

    north = max(
        p[1]
        for p in coordinates
    )

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
# BBOX STRING
# ============================================================

def bbox_string(
    west,
    south,
    east,
    north
):

    return (
        f"{west:.8f},"
        f"{south:.8f},"
        f"{east:.8f},"
        f"{north:.8f}"
    )


# ============================================================
# DETECT WMS ERROR
# ============================================================

def response_is_error(
    response
):

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
        x in content_type
        for x in error_types
    ):

        return True

    return sample.startswith(
        (
            b"<?xml",
            b"<serviceexception",
            b"<html",
            b"<!doctype html",
            b"{",
            b"["
        )
    )


# ============================================================
# SAVE WMS ERROR
# ============================================================

def save_error(
    output_dir,
    target_date,
    response
):

    path = (
        output_dir
        /
        f"fwi_download_error_{target_date}.txt"
    )

    header = (
        f"HTTP status: "
        f"{response.status_code}\n"
        f"Content-Type: "
        f"{response.headers.get('Content-Type', '')}\n"
        f"URL: {response.url}\n"
        "\n"
        "Response body:\n"
        "------------------------------------------------------------\n"
    ).encode(
        "utf-8"
    )

    path.write_bytes(
        header
        +
        response.content
    )

    return path


# ============================================================
# VALIDATE GEOTIFF
# ============================================================

def validate_geotiff(
    path
):

    try:

        with rasterio.open(
            path
        ) as src:

            if src.driver != "GTiff":

                raise RuntimeError(
                    "Downloaded file is not GeoTIFF."
                )

            if src.crs is None:

                raise RuntimeError(
                    "Downloaded FWI has no CRS."
                )

            if src.crs.to_epsg() != 4326:

                raise RuntimeError(
                    f"FWI CRS must be EPSG:4326. "
                    f"Got {src.crs}"
                )

            if src.width <= 0 or src.height <= 0:

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
                    "FWI contains no valid pixels."
                )

            return {

                "crs":
                    str(src.crs),

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

                    "west":
                        float(
                            src.bounds.left
                        ),

                    "south":
                        float(
                            src.bounds.bottom
                        ),

                    "east":
                        float(
                            src.bounds.right
                        ),

                    "north":
                        float(
                            src.bounds.top
                        )
                },

                "transform": [

                    float(src.transform.a),
                    float(src.transform.b),
                    float(src.transform.c),
                    float(src.transform.d),
                    float(src.transform.e),
                    float(src.transform.f)
                ],

                "nodata":
                    src.nodata,

                "valid_pixels":
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
            f"GeoTIFF validation failed: {error}"
        ) from error


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

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


    # --------------------------------------------------------
    # AUTOMATIC IRAN DATE
    # --------------------------------------------------------

    now_iran, forecast_date = (
        get_forecast_date()
    )

    target_date = (
        forecast_date.isoformat()
    )


    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BOUNDARY
    # --------------------------------------------------------

    geojson = load_boundary(
        args.boundary
    )

    (
        west,
        south,
        east,
        north
    ) = calculate_bbox(

        geojson,

        BOUNDARY_MARGIN_DEG
    )

    bbox = bbox_string(
        west,
        south,
        east,
        north
    )


    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("FIRIS - AUTOMATIC ECMWF FWI")
    print("=" * 70)

    print("")
    print(
        "Iran current date : "
        f"{now_iran.strftime('%Y-%m-%d')}"
    )

    print(
        "Iran current time : "
        f"{now_iran.strftime('%H:%M:%S')}"
    )

    print(
        "Forecast date     : "
        f"{target_date}"
    )

    print(
        "Timezone          : Asia/Tehran"
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
        f"BBOX   : {bbox}"
    )


    # --------------------------------------------------------
    # EXISTING FILE
    # --------------------------------------------------------

    if (
        output_tif.exists()
        and
        not args.overwrite
    ):

        print("")
        print(
            f"FWI already exists:"
        )

        print(
            output_tif
        )

        print(
            "Use --overwrite to download again."
        )

        return


    # --------------------------------------------------------
    # WMS PARAMETERS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    print("")
    print(
        "Downloading FWI for forecast date..."
    )

    try:

        response = requests.get(

            WMS_URL,

            params=params,

            timeout=
                REQUEST_TIMEOUT_SECONDS,

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


    if not response.content:

        raise SystemExit(
            "FWI response is empty."
        )


    if response_is_error(
        response
    ):

        error_path = save_error(
            output_dir,
            target_date,
            response
        )

        raise SystemExit(
            "WMS returned an error document.\n"
            f"Saved: {error_path}"
        )


    # --------------------------------------------------------
    # TEMPORARY DOWNLOAD
    # --------------------------------------------------------

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(

            mode="wb",

            suffix=".tif",

            prefix=
                f".fwi_{target_date}_",

            dir=output_dir,

            delete=False

        ) as temp:

            temp.write(
                response.content
            )

            temporary_path = Path(
                temp.name
            )


        raster_info = validate_geotiff(
            temporary_path
        )


        os.replace(
            temporary_path,
            output_tif
        )

        temporary_path = None


    except RuntimeError as error:

        invalid_path = (
            output_dir
            /
            f"fwi_invalid_{target_date}.tif"
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
            f"{error}\n"
            f"Invalid file: {invalid_path}"
        ) from error


    finally:

        if (
            temporary_path is not None
            and
            temporary_path.exists()
        ):

            temporary_path.unlink()


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "project":
            "FIRIS",

        "indicator":
            "ECMWF Fire Weather Index",

        "source":
            "Copernicus GWIS WMS",

        "source_url":
            WMS_URL,

        "layer":
            LAYER_NAME,

        "timezone":
            "Asia/Tehran",

        "current_iran_datetime":
            now_iran.isoformat(),

        "forecast_date":
            target_date,

        "forecast_definition":
            "Tomorrow according to Iran local date",

        "boundary_file":
            str(args.boundary),

        "boundary_margin_degree":
            BOUNDARY_MARGIN_DEG,

        "request_bbox": {

            "west":
                west,

            "south":
                south,

            "east":
                east,

            "north":
                north
        },

        "request_parameters":
            params,

        "request_url":
            response.url,

        "downloaded_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "raster":
            raster_info,

        "output_file":
            output_tif.name
    }


    output_json.write_text(

        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"
    )


    # --------------------------------------------------------
    # FINAL LOG
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("FWI DOWNLOAD COMPLETED")
    print("=" * 70)

    print(
        f"Forecast date : {target_date}"
    )

    print(
        f"Output        : {output_tif}"
    )

    print(
        f"Metadata      : {output_json}"
    )

    print("")
    print("FINAL RASTER BOUNDS")

    print(
        f"West  : "
        f"{raster_info['bounds']['west']:.8f}"
    )

    print(
        f"South : "
        f"{raster_info['bounds']['south']:.8f}"
    )

    print(
        f"East  : "
        f"{raster_info['bounds']['east']:.8f}"
    )

    print(
        f"North : "
        f"{raster_info['bounds']['north']:.8f}"
    )

    print("")
    print(
        "FLI calculation: NOT performed here."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
