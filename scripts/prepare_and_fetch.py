import os
import sys
import glob
import time
import subprocess
from datetime import datetime, timedelta
import zoneinfo

def main():
    # ۱. محاسبه تاریخ فردا به وقت تهران
    tehran = zoneinfo.ZoneInfo("Asia/Tehran")
    target_date = (datetime.now(tehran) + timedelta(days=1)).strftime("%Y-%m-%d")

    def find_file(patterns):
        for pattern in patterns:
            matches = sorted(glob.glob(pattern, recursive=True))
            if matches:
                return matches[0]
        return ""

    # ۲. شناسایی خودکار لایه‌های ورودی
    dem_path = find_file(["data/*dem*.tif", "**/*dem*.tif", "data/dem.tif"])
    fuel_path = find_file(["data/*fuel*.tif", "**/*fuel*.tif", "data/fuel.tif"])
    excel_path = find_file(["data/*.xlsx", "data/*.xls", "**/*.xlsx"])
    boundary_path = find_file(["data/*fars*.geojson", "**/*fars*.geojson", "data/*.geojson"])

    # ۳. تلاش برای دانلود داده‌های هواشناسی با مکانیزم Retry (مدیریت خطای ۵۰۳)
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if os.path.exists("scripts/fetch_weather.py"):
        success = False
        for attempt in range(1, 4):
            print(f"[Attempt {attempt}/3] Fetching weather data...")
            res = subprocess.run([sys.executable, "scripts/fetch_weather.py"])
            if res.returncode == 0:
                print("Weather data successfully downloaded.")
                success = True
                break
            print("Weather server error (503/timeout). Retrying in 15 seconds...")
            time.sleep(15)
        if not success:
            print("Warning: Could not fetch new weather data. Checking local files...")

    # ۴. شناسایی رستر FWI
    fwi_path = find_file([
        f"data/*{target_date}*.tif",
        "data/*fwi*.tif",
        "**/*fwi*.tif",
        "data/*.tif"
    ])

    print("--- Detected Parameters ---")
    print(f"RUN_DATE      : {target_date}")
    print(f"DEM_PATH      : {dem_path}")
    print(f"FUEL_PATH     : {fuel_path}")
    print(f"EXCEL_PATH    : {excel_path}")
    print(f"BOUNDARY_PATH : {boundary_path}")
    print(f"FWI_PATH      : {fwi_path}")

    # ۵. تزریق متغیرها به GitHub Actions Environment
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as f:
            f.write(f"RUN_DATE={target_date}\n")
            f.write(f"DEM_PATH={dem_path}\n")
            f.write(f"FUEL_PATH={fuel_path}\n")
            f.write(f"EXCEL_PATH={excel_path}\n")
            f.write(f"BOUNDARY_PATH={boundary_path}\n")
            f.write(f"FWI_PATH={fwi_path}\n")

if __name__ == "__main__":
    main()
