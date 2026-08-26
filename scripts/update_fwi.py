from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import requests


WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"

# محدودهٔ درخواست داده برای استان فارس:
# BBOX در EPSG:4326 و ترتیب آن در WMS 1.1.1:
# min_longitude, min_latitude, max_longitude, max_latitude
FARS_BBOX = "50.0,27.0,55.0,32.0"

LAYER_NAME = "ecmwf.fwi"
CRS = "EPSG:4326"
WIDTH = 2000
HEIGHT = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real ECMWF FWI raster for Fars from Copernicus GWIS WMS."
    )

    parser.add_argument(
        "--date",
        dest="target_date",
        help="Forecast date in YYYY-MM-DD format. Default: tomorrow (UTC).",
    )

    return parser.parse_args()


def get_target_date(value: str | None) -> str:
    if value:
        # فقط اعتبار فرمت تاریخ را بررسی می‌کنیم.
        # تاریخ را به تاریخ دلخواه دیگری تغییر نمی‌دهیم.
        return date.fromisoformat(value).isoformat()

    return (date.today() + timedelta(days=1)).isoformat()


def main() -> None:
    args = parse_args()
    target_date = get_target_date(args.target_date)

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "fwi"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"fwi_ecmwf_fars_{target_date}.tif"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": LAYER_NAME,
        "STYLES": "",
        "SRS": CRS,
        "BBOX": FARS_BBOX,
        "WIDTH": str(WIDTH),
        "HEIGHT": str(HEIGHT),
        "FORMAT": "image/tiff",
        "TRANSPARENT": "TRUE",
        "TIME": target_date,
    }

    print(f"Downloading FWI for Fars: {target_date}")
    print(f"Output file: {output_file}")

    try:
        response = requests.get(
            WMS_URL,
            params=params,
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"Download failed: {error}") from error

    content_type = response.headers.get("Content-Type", "").lower()

    print("HTTP:", response.status_code)
    print("Content-Type:", content_type)
    print("Downloaded bytes:", len(response.content))

    # اگر WMS به جای GeoTIFF پیام خطا برگرداند،
    # فایل خطا را با نام tif ذخیره نمی‌کنیم.
    if "tiff" not in content_type:
        error_file = output_dir / f"fwi_download_error_{target_date}.txt"
        error_file.write_bytes(response.content)

        raise SystemExit(
            "The server did not return a GeoTIFF. "
            f"Response saved for inspection: {error_file}"
        )

    if not response.content:
        raise SystemExit("The server returned an empty file.")

    output_file.write_bytes(response.content)

    print("Download completed successfully.")
    print(output_file)


if __name__ == "__main__":
    main()
