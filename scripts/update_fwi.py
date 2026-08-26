from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import rasterio
from rasterio.errors import RasterioError
import requests


# سرویس رسمی WMS سامانه Global Wildfire Information System
WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

# لایه Fire Weather Index از ECMWF
LAYER_NAME = "ecmwf.fwi"

# محدودهٔ جغرافیایی فارس، ترتیب مختصات در WMS 1.1.1:
# min_longitude, min_latitude, max_longitude, max_latitude
FARS_BBOX = "50.0,27.0,55.0,32.0"

# WMS 1.1.1 برای جلوگیری از تغییر ترتیب محور مختصات EPSG:4326
WMS_VERSION = "1.1.1"
SRS = "EPSG:4326"

# اندازهٔ خروجی. حداکثر سرویس 5120 × 5120 است.
WIDTH = 2000
HEIGHT = 2000

REQUEST_TIMEOUT_SECONDS = 180


def parse_args() -> argparse.Namespace:
    """خواندن پارامترهای خط فرمان."""
    parser = argparse.ArgumentParser(
        description=(
            "Download ECMWF Fire Weather Index (FWI) GeoTIFF for Fars "
            "from Copernicus GWIS WMS."
        )
    )

    parser.add_argument(
        "--date",
        dest="target_date",
        default=None,
        help=(
            "Date in YYYY-MM-DD format. "
            "Default: tomorrow in UTC."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if output TIFF already exists.",
    )

    return parser.parse_args()


def get_target_date(value: str | None) -> str:
    """
    تاریخ هدف را برمی‌گرداند.
    در صورت نداشتن آرگومان، تاریخ فردا بر مبنای UTC انتخاب می‌شود.
    """
    if value is not None:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
        except ValueError as error:
            raise SystemExit(
                f"Invalid date: {value}. Expected format: YYYY-MM-DD"
            ) from error

    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    return tomorrow.isoformat()


def response_is_error_payload(response: requests.Response) -> bool:
    """
    تشخیص پاسخ خطا از سرویس WMS.

    گاهی پاسخ HTTP موفق است ولی بدنهٔ پاسخ XML/HTML شامل ServiceException
    است؛ بنابراین صرفاً HTTP 200 کافی نیست.
    """
    content_type = response.headers.get("Content-Type", "").lower()
    sample = response.content[:1000].lstrip().lower()

    error_content_types = (
        "text/xml",
        "application/xml",
        "text/html",
        "application/json",
        "text/plain",
    )

    if any(item in content_type for item in error_content_types):
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

    return sample.startswith(error_prefixes)


def write_error_response(
    output_dir: Path,
    target_date: str,
    response: requests.Response,
) -> Path:
    """ذخیرهٔ پاسخ خطای WMS برای بررسی در GitHub Actions."""
    error_path = output_dir / f"fwi_download_error_{target_date}.txt"

    header = (
        f"HTTP status: {response.status_code}\n"
        f"Content-Type: {response.headers.get('Content-Type', '')}\n"
        f"Request URL: {response.url}\n"
        "\n"
        "Response body:\n"
        "------------------------------------------------------------\n"
    ).encode("utf-8")

    error_path.write_bytes(header + response.content)
    return error_path


def validate_geotiff(tif_path: Path) -> dict[str, Any]:
    """
    بررسی می‌کند فایل دانلودی واقعاً GeoTIFF رستری خوانا و دارای داده باشد.
    """
    try:
        with rasterio.open(tif_path) as dataset:
            if dataset.driver != "GTiff":
                raise RuntimeError(
                    f"Expected GTiff, received driver: {dataset.driver}"
                )

            if dataset.count < 1:
                raise RuntimeError("GeoTIFF contains no raster band.")

            if dataset.width < 1 or dataset.height < 1:
                raise RuntimeError("GeoTIFF has invalid dimensions.")

            first_band = dataset.read(1, masked=True)
            valid_pixel_count = int(first_band.count())

            if valid_pixel_count == 0:
                raise RuntimeError("GeoTIFF has no valid FWI pixels.")

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
                "valid_pixel_count": valid_pixel_count,
                "fwi_min": float(first_band.min()),
                "fwi_max": float(first_band.max()),
                "fwi_mean": float(first_band.mean()),
            }

    except RasterioError as error:
        raise RuntimeError(
            f"Downloaded file cannot be opened as GeoTIFF: {error}"
        ) from error


def write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """ثبت متادیتای کامل برای بازتولیدپذیری محاسبهٔ FIRIS."""
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    target_date = get_target_date(args.target_date)

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "fwi"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_tif = output_dir / f"fwi_ecmwf_fars_{target_date}.tif"
    output_json = output_dir / f"fwi_ecmwf_fars_{target_date}.json"

    if output_tif.exists() and not args.overwrite:
        print(f"FWI file already exists: {output_tif}")
        print("Use --overwrite to download it again.")
        return

    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": LAYER_NAME,
        "STYLES": "",
        "SRS": SRS,
        "BBOX": FARS_BBOX,
        "WIDTH": str(WIDTH),
        "HEIGHT": str(HEIGHT),
        "FORMAT": "image/tiff",
        "TRANSPARENT": "FALSE",
        "TIME": target_date,
    }

    print("=" * 70)
    print("FIRIS - ECMWF FWI downloader")
    print("=" * 70)
    print(f"Target date : {target_date}")
    print(f"Layer       : {LAYER_NAME}")
    print(f"BBOX        : {FARS_BBOX}")
    print(f"Output TIFF : {output_tif}")
    print()

    try:
        response = requests.get(
            WMS_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={
                "User-Agent": "FIRIS/1.0 (GitHub Actions; Fars Fire Risk System)"
            },
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise SystemExit(f"FWI download failed: {error}") from error

    print(f"HTTP status     : {response.status_code}")
    print(
        "Content-Type    : "
        f"{response.headers.get('Content-Type', 'not provided')}"
    )
    print(f"Downloaded bytes: {len(response.content)}")

    if not response.content:
        raise SystemExit("FWI download failed: server returned an empty response.")

    if response_is_error_payload(response):
        error_path = write_error_response(output_dir, target_date, response)

        raise SystemExit(
            "FWI download failed: WMS returned an error document, not GeoTIFF. "
            f"Error response saved to: {error_path}"
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".tif",
            prefix=f".fwi_ecmwf_fars_{target_date}_",
            dir=output_dir,
            delete=False,
        ) as temporary_file:
            temporary_file.write(response.content)
            temporary_path = Path(temporary_file.name)

        raster_info = validate_geotiff(temporary_path)

        # جایگزینی اتمیک؛ فایل ناقص در خروجی نهایی باقی نمی‌ماند.
        os.replace(temporary_path, output_tif)
        temporary_path = None

    except RuntimeError as error:
        invalid_file = output_dir / f"fwi_invalid_response_{target_date}.tif"

        if temporary_path is not None and temporary_path.exists():
            os.replace(temporary_path, invalid_file)
            temporary_path = None

        raise SystemExit(
            f"FWI download failed: {error}\n"
            f"Invalid response saved to: {invalid_file}"
        ) from error

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    metadata = {
        "project": "FIRIS",
        "indicator": "ECMWF Fire Weather Index (FWI)",
        "source": "Copernicus GWIS WMS",
        "source_url": WMS_URL,
        "layer": LAYER_NAME,
        "target_date": target_date,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_parameters": params,
        "request_url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "output_file": output_tif.name,
        "output_file_size_bytes": output_tif.stat().st_size,
        "raster": raster_info,
    }

    write_metadata(output_json, metadata)

    print()
    print("FWI GeoTIFF downloaded and validated successfully.")
    print(f"GeoTIFF : {output_tif}")
    print(f"Metadata: {output_json}")
    print(
        "FWI statistics: "
        f"min={raster_info['fwi_min']:.3f}, "
        f"max={raster_info['fwi_max']:.3f}, "
        f"mean={raster_info['fwi_mean']:.3f}"
    )
    print(f"Valid pixels: {raster_info['valid_pixel_count']}")


if __name__ == "__main__":
    main()
