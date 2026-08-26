from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import rasterio
import requests


WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

# محدوده تقریبی استان فارس در EPSG:4326
# ترتیب BBOX در WMS 1.1.1:
# min_longitude, min_latitude, max_longitude, max_latitude
DEFAULT_FARS_BBOX = "50.0,27.0,55.0,32.0"

LAYER_NAME = "ecmwf.fwi"
CRS = "EPSG:4326"
DEFAULT_WIDTH = 2000
DEFAULT_HEIGHT = 2000
REQUEST_TIMEOUT_SECONDS = 180


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate an ECMWF Fire Weather Index (FWI) GeoTIFF "
            "for Fars from Copernicus GWIS WMS."
        )
    )

    parser.add_argument(
        "--date",
        dest="target_date",
        help="Forecast date in YYYY-MM-DD format. Default: tomorrow (UTC).",
    )

    parser.add_argument(
        "--bbox",
        default=DEFAULT_FARS_BBOX,
        help=(
            "WMS 1.1.1 BBOX in EPSG:4326 as "
            "min_lon,min_lat,max_lon,max_lat. "
            f"Default: {DEFAULT_FARS_BBOX}"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output raster width in pixels. Default: {DEFAULT_WIDTH}",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output raster height in pixels. Default: {DEFAULT_HEIGHT}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing GeoTIFF for the requested date.",
    )

    return parser.parse_args()


def get_target_date(value: str | None) -> str:
    """Return the requested ISO date, or tomorrow in UTC."""
    if value:
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as error:
            raise SystemExit(
                f"Invalid --date value '{value}'. Expected YYYY-MM-DD."
            ) from error

    return (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()


def validate_bbox(bbox: str) -> str:
    """Validate and normalize a WMS 1.1.1 geographic bounding box."""
    try:
        values = [float(value.strip()) for value in bbox.split(",")]
    except ValueError as error:
        raise SystemExit(
            "Invalid --bbox. Expected: min_lon,min_lat,max_lon,max_lat"
        ) from error

    if len(values) != 4:
        raise SystemExit(
            "Invalid --bbox. Expected exactly four values: "
            "min_lon,min_lat,max_lon,max_lat"
        )

    min_lon, min_lat, max_lon, max_lat = values

    if not (-180 <= min_lon < max_lon <= 180):
        raise SystemExit("Invalid longitude values in --bbox.")

    if not (-90 <= min_lat < max_lat <= 90):
        raise SystemExit("Invalid latitude values in --bbox.")

    return ",".join(f"{value:g}" for value in values)


def validate_downloaded_geotiff(file_path: Path) -> dict[str, Any]:
    """
    Open the downloaded file with rasterio and return useful raster metadata.

    Raises SystemExit if the server response is not a readable GeoTIFF or
    contains no valid numeric values.
    """
    try:
        with rasterio.open(file_path) as dataset:
            if dataset.driver != "GTiff":
                raise SystemExit(
                    f"Downloaded file is not a GeoTIFF. Detected driver: {dataset.driver}"
                )

            if dataset.count < 1:
                raise SystemExit("Downloaded GeoTIFF does not contain any raster band.")

            if dataset.width < 1 or dataset.height < 1:
                raise SystemExit("Downloaded GeoTIFF has invalid dimensions.")

            band = dataset.read(1, masked=True)
            valid_value_count = int(band.count())

            if valid_value_count == 0:
                raise SystemExit(
                    "Downloaded GeoTIFF contains no valid FWI pixel values."
                )

            minimum = float(band.min())
            maximum = float(band.max())
            mean = float(band.mean())

            return {
                "driver": dataset.driver,
                "crs": str(dataset.crs) if dataset.crs else None,
                "width": dataset.width,
                "height": dataset.height,
                "band_count": dataset.count,
                "dtype": dataset.dtypes[0],
                "nodata": dataset.nodata,
                "bounds": {
                    "left": dataset.bounds.left,
                    "bottom": dataset.bounds.bottom,
                    "right": dataset.bounds.right,
                    "top": dataset.bounds.top,
                },
                "valid_pixel_count": valid_value_count,
                "fwi_min": minimum,
                "fwi_max": maximum,
                "fwi_mean": mean,
            }

    except rasterio.errors.RasterioError as error:
        raise SystemExit(
            "The response was not a readable GeoTIFF. "
            f"Raster validation error: {error}"
        ) from error


def response_looks_like_error(response: requests.Response) -> bool:
    """Detect common WMS error payloads before writing a TIFF file."""
    content_type = response.headers.get("Content-Type", "").lower()
    first_bytes = response.content[:500].lstrip().lower()

    if any(
        media_type in content_type
        for media_type in ("xml", "html", "json", "text/plain")
    ):
        return True

    return (
        first_bytes.startswith(b"<?xml")
        or first_bytes.startswith(b"<serviceexception")
        or first_bytes.startswith(b"<html")
        or first_bytes.startswith(b"{")
        or first_bytes.startswith(b"[")
    )


def save_error_response(
    output_dir: Path,
    target_date: str,
    response: requests.Response,
) -> Path:
    """Save a non-raster WMS response for inspection."""
    error_file = output_dir / f"fwi_download_error_{target_date}.txt"

    error_file.write_bytes(response.content)

    return error_file


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 formatted JSON."""
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Download, validate, and document one FWI raster."""
    args = parse_args()

    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must both be greater than zero.")

    target_date = get_target_date(args.target_date)
    bbox = validate_bbox(args.bbox)

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "fwi"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"fwi_ecmwf_fars_{target_date}.tif"
    metadata_file = output_dir / f"fwi_ecmwf_fars_{target_date}.json"

    if output_file.exists() and not args.overwrite:
        print(f"FWI file already exists: {output_file}")
        print("Use --overwrite to download it again.")
        return

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": LAYER_NAME,
        "STYLES": "",
        "SRS": CRS,
        "BBOX": bbox,
        "WIDTH": str(args.width),
        "HEIGHT": str(args.height),
        "FORMAT": "image/tiff",
        "TRANSPARENT": "TRUE",
        "TIME": target_date,
    }

    print(f"Downloading ECMWF FWI for Fars: {target_date}")
    print(f"WMS URL: {WMS_URL}")
    print(f"Layer: {LAYER_NAME}")
    print(f"BBOX: {bbox}")
    print(f"Output: {output_file}")

    try:
        response = requests.get(
            WMS_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"FWI download failed: {error}") from error

    content_type = response.headers.get("Content-Type", "").lower()

    print(f"HTTP status: {response.status_code}")
    print(f"Content-Type: {content_type or 'not provided'}")
    print(f"Downloaded bytes: {len(response.content)}")

    if not response.content:
        raise SystemExit("The WMS server returned an empty response.")

    if response_looks_like_error(response):
        error_file = save_error_response(output_dir, target_date, response)

        raise SystemExit(
            "The WMS server returned an error payload instead of a GeoTIFF. "
            f"Saved response for inspection: {error_file}"
        )

    temporary_file: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".tif",
            prefix=f".fwi_ecmwf_fars_{target_date}_",
            dir=output_dir,
            delete=False,
        ) as temporary_output:
            temporary_output.write(response.content)
            temporary_file = Path(temporary_output.name)

        raster_metadata = validate_downloaded_geotiff(temporary_file)

        os.replace(temporary_file, output_file)
        temporary_file = None

        download_metadata = {
            "source": "Copernicus GWIS WMS / ECMWF",
            "wms_url": WMS_URL,
            "layer": LAYER_NAME,
            "request_date": target_date,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_parameters": params,
            "http_status": response.status_code,
            "content_type": content_type,
            "file_name": output_file.name,
            "file_size_bytes": output_file.stat().st_size,
            "raster": raster_metadata,
        }

        write_json(metadata_file, download_metadata)

    finally:
        if temporary_file is not None and temporary_file.exists():
            temporary_file.unlink()

    print("FWI download and GeoTIFF validation completed successfully.")
    print(f"GeoTIFF: {output_file}")
    print(f"Metadata: {metadata_file}")
    print(
        "FWI statistics: "
        f"min={raster_metadata['fwi_min']:.3f}, "
        f"max={raster_metadata['fwi_max']:.3f}, "
        f"mean={raster_metadata['fwi_mean']:.3f}, "
        f"valid_pixels={raster_metadata['valid_pixel_count']}"
    )


if __name__ == "__main__":
    main()
