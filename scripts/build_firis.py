name: Update FIRIS FWI

on:

  workflow_dispatch:

  schedule:

    # ----------------------------------------------------------
    # RUN 1
    #
    # 18:00 UTC
    # = 21:30 Iran
    #
    # Prepare tomorrow's forecast in advance.
    # ----------------------------------------------------------
    - cron: "0 18 * * *"

    # ----------------------------------------------------------
    # RUN 2
    #
    # 20:35 UTC
    # = 00:05 Iran
    #
    # After midnight in Iran, tomorrow changes to the new date.
    # ----------------------------------------------------------
    - cron: "35 20 * * *"

  push:

    paths:

      - "scripts/update_fwi.py"
      - "fars.geojson"
      - ".github/workflows/update_fwi.yml"


permissions:
  contents: write


concurrency:

  group: firis-fwi-update

  cancel-in-progress: false


jobs:

  update-fwi:

    runs-on: ubuntu-latest

    steps:

      # ========================================================
      # CHECKOUT
      # ========================================================

      - name: Checkout repository

        uses: actions/checkout@v4

        with:

          fetch-depth: 0
          persist-credentials: true


      # ========================================================
      # SETUP PYTHON
      # ========================================================

      - name: Setup Python

        uses: actions/setup-python@v5

        with:

          python-version: "3.11"


      # ========================================================
      # INSTALL DEPENDENCIES
      # ========================================================

      - name: Install dependencies

        shell: bash

        run: |

          set -euo pipefail

          python -m pip install --upgrade pip

          pip install \
            numpy \
            rasterio \
            requests \
            shapely \
            pyproj


      # ========================================================
      # CALCULATE IRAN DATE
      # ========================================================

      - name: Calculate Iran dates

        shell: bash

        run: |

          set -euo pipefail

          python - <<'PY'

          import os

          from datetime import (
              datetime,
              timedelta
          )

          from zoneinfo import ZoneInfo


          IRAN = ZoneInfo(
              "Asia/Tehran"
          )


          now_iran = datetime.now(
              IRAN
          )


          today = now_iran.date()

          tomorrow = (
              today
              +
              timedelta(days=1)
          )


          today_str = today.isoformat()

          tomorrow_str = tomorrow.isoformat()


          print("")
          print("=" * 70)
          print("FIRIS IRAN DATE CALCULATION")
          print("=" * 70)

          print("")
          print("Current Iran datetime:")
          print(now_iran.isoformat())

          print("")
          print("Today in Iran:")
          print(today_str)

          print("")
          print("Forecast target:")
          print(tomorrow_str)


          with open(
              os.environ["GITHUB_ENV"],
              "a",
              encoding="utf-8"
          ) as env:

              env.write(
                  f"TODAY_IRAN={today_str}\n"
              )

              env.write(
                  f"EXPECTED_DATE={tomorrow_str}\n"
              )

          PY


      # ========================================================
      # DOWNLOAD TOMORROW FWI
      # ========================================================

      - name: Download FWI for tomorrow

        shell: bash

        run: |

          set -euo pipefail

          echo ""
          echo "=" * 70
          echo "DOWNLOADING FIRIS FWI"
          echo "=" * 70

          echo ""
          echo "Today in Iran:"
          echo "${TODAY_IRAN}"

          echo ""
          echo "Target forecast:"
          echo "${EXPECTED_DATE}"


          python scripts/update_fwi.py \
            --boundary "fars.geojson" \
            --overwrite


      # ========================================================
      # VERIFY TOMORROW FWI
      # ========================================================

      - name: Verify tomorrow FWI

        shell: bash

        run: |

          set -euo pipefail

          FWI_FILE="data/raw/fwi/fwi_ecmwf_fars_${EXPECTED_DATE}.tif"

          FWI_METADATA="data/raw/fwi/fwi_ecmwf_fars_${EXPECTED_DATE}.json"


          if [ ! -f "${FWI_FILE}" ]; then

            echo ""
            echo "ERROR: Tomorrow FWI raster was not created:"
            echo "${FWI_FILE}"

            exit 1

          fi


          if [ ! -f "${FWI_METADATA}" ]; then

            echo ""
            echo "ERROR: Tomorrow FWI metadata was not created:"
            echo "${FWI_METADATA}"

            exit 1

          fi


          echo ""
          echo "✓ Tomorrow FWI raster exists."

          echo "✓ Tomorrow FWI metadata exists."


      # ========================================================
      # VERIFY METADATA DATE
      # ========================================================

      - name: Verify FWI metadata date

        shell: bash

        run: |

          set -euo pipefail

          python - <<'PY'

          import json
          import os
          import sys


          expected = os.environ["EXPECTED_DATE"]


          path = (
              "data/raw/fwi/"
              f"fwi_ecmwf_fars_{expected}.json"
          )


          with open(
              path,
              "r",
              encoding="utf-8"
          ) as file:

              metadata = json.load(file)


          actual = (
              metadata.get("target_date")
              or
              metadata.get("forecast_date")
          )


          print("")
          print("=" * 70)
          print("FWI DATE VALIDATION")
          print("=" * 70)

          print("")
          print("Expected:")
          print(expected)

          print("")
          print("Metadata:")
          print(actual)


          if actual != expected:

              print("")
              print(
                  "ERROR: FWI target date does not "
                  "match EXPECTED_DATE."
              )

              sys.exit(1)


          print("")
          print("✓ FWI target date is correct.")

          PY


      # ========================================================
      # VALIDATE GEOTIFF
      # ========================================================

      - name: Validate FWI GeoTIFF

        shell: bash

        run: |

          set -euo pipefail

          python - <<'PY'

          import os
          import rasterio


          expected = os.environ["EXPECTED_DATE"]


          path = (
              "data/raw/fwi/"
              f"fwi_ecmwf_fars_{expected}.tif"
          )


          with rasterio.open(path) as src:

              print("")
              print("=" * 70)
              print("FWI GEOTIFF VALIDATION")
              print("=" * 70)

              print("")
              print("CRS:")
              print(src.crs)

              print("")
              print("Size:")
              print(
                  src.width,
                  "x",
                  src.height
              )

              print("")
              print("Resolution:")
              print(src.res)

              print("")
              print("Bounds:")
              print(src.bounds)


              if src.crs is None:

                  raise SystemExit(
                      "ERROR: FWI has no CRS."
                  )


              if src.crs.to_epsg() != 4326:

                  raise SystemExit(
                      "ERROR: FWI CRS is not EPSG:4326."
                  )


              data = src.read(
                  1,
                  masked=True
              )


              if data.count() == 0:

                  raise SystemExit(
                      "ERROR: FWI contains no valid pixels."
                  )


              print("")
              print("Valid pixels:")
              print(int(data.count()))

              print("")
              print("Minimum:")
              print(float(data.min()))

              print("")
              print("Maximum:")
              print(float(data.max()))

              print("")
              print("Mean:")
              print(float(data.mean()))


          print("")
          print("✓ FWI GeoTIFF is valid.")

          PY


      # ========================================================
      # FINAL FORECAST VALIDATION
      # ========================================================

      - name: Final forecast validation

        shell: bash

        run: |

          set -euo pipefail

          python - <<'PY'

          import json
          import os
          import sys


          expected = os.environ["EXPECTED_DATE"]


          path = (
              "data/raw/fwi/"
              f"fwi_ecmwf_fars_{expected}.json"
          )


          with open(
              path,
              "r",
              encoding="utf-8"
          ) as file:

              metadata = json.load(file)


          target = (
              metadata.get("target_date")
              or
              metadata.get("forecast_date")
          )


          print("")
          print("=" * 70)
          print("FINAL FIRIS FWI FORECAST")
          print("=" * 70)

          print("")
          print("Iran datetime:")
          print(
              metadata.get(
                  "current_iran_datetime"
              )
          )

          print("")
          print("Today in Iran:")
          print(
              os.environ["TODAY_IRAN"]
          )

          print("")
          print("Target forecast:")
          print(target)


          if target != expected:

              print("")
              print(
                  "ERROR: Final target date "
                  "is not tomorrow."
              )

              sys.exit(1)


          print("")
          print("✓ Forecast target is correct.")
          print("✓ Tomorrow FWI validated.")

          PY


      # ========================================================
      # COMMIT
      # ========================================================

      - name: Commit FWI

        shell: bash

        run: |

          set -euo pipefail

          git config \
            user.name \
            "github-actions[bot]"

          git config \
            user.email \
            "41898282+github-actions[bot]@users.noreply.github.com"


          git add \
            data/raw/fwi


          if git diff --cached --quiet; then

            echo ""
            echo "No FWI changes to commit."

            exit 0

          fi


          git commit \
            -m \
            "chore(fwi): update forecast ${EXPECTED_DATE} [skip ci]"


      # ========================================================
      # PUSH
      # ========================================================

      - name: Push FWI

        shell: bash

        run: |

          set -euo pipefail

          git fetch origin main

          git merge origin/main \
            -X ours \
            --no-edit

          git push origin HEAD:main


          echo ""
          echo "=" * 70
          echo "FIRIS FWI UPDATE COMPLETED"
          echo "=" * 70

          echo ""
          echo "Today in Iran:"
          echo "${TODAY_IRAN}"

          echo ""
          echo "Forecast target:"
          echo "${EXPECTED_DATE}"
