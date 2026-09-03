"use strict";

// ============================================================================
// APP STATE - مدیریت وضعیت کل برنامه
// ============================================================================
const APP_STATE = {
  map: null,
  layers: {
    fars: null,      // مرز استان فارس
    fli: null,       // شاخص خطر حریق
    protected: null, // مناطق تحت مدیریت
    hunting: null    // مناطق شکار ممنوع
  },
  data: {
    fliGrid: null,      // داده های شطرنجی FLI
    fliMetadata: null   // متادیتای FLI
  },
  ui: {
    nightMode: false,
    currentPopup: null,
    activeLayer: "fli"
  }
};


// ============================================================================
// تابع های UTILITY - توابع کمکی
// ============================================================================

/**
 * تنظیم تاریخ پیش‌بینی در هدر برنامه
 *
 * IMPORTANT:
 * تاریخ دیگر از ساعت سیستم + یک روز محاسبه نمی‌شود.
 *
 * تنها منبع معتبر تاریخ:
 * data/web/fli_latest.json
 *
 * بنابراین تاریخ نمایش‌داده‌شده در سایت دقیقاً همان تاریخی است
 * که برای FLI واقعی تولید شده است.
 */
async function setupTomorrowDate() {

  const badge =
    document.getElementById(
      "forecastDateBadge"
    );

  if (!badge) {
    return;
  }

  try {

    badge.textContent =
      "در حال دریافت تاریخ پیش‌بینی...";

    /*
     * metadata از قبل توسط loadFLIData()
     * در APP_STATE ذخیره شده است.
     *
     * اگر موجود نباشد، یک بار دیگر از فایل
     * دریافت می‌شود تا تاریخ هرگز مستقل
     * از داده واقعی محاسبه نشود.
     */

    let metadata =
      APP_STATE.data.fliMetadata;

    if (!metadata) {

      const response =
        await fetchWithTimeout(
          "data/web/fli_latest.json"
        );

      metadata =
        await response.json();

      APP_STATE.data.fliMetadata =
        metadata;
    }

    const forecastDate =
      metadata?.forecast_date;

    if (!forecastDate) {

      throw new Error(
        "forecast_date not found in fli_latest.json"
      );
    }

    /*
     * زمان ثابت ظهر انتخاب می‌شود تا timezone
     * روی تبدیل تاریخ اثر نگذارد.
     */
    const date =
      new Date(
        `${forecastDate}T12:00:00`
      );

    if (
      Number.isNaN(
        date.getTime()
      )
    ) {

      throw new Error(
        `Invalid forecast date: ${forecastDate}`
      );
    }

    const options = {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric"
    };

    const dateStr =
      date.toLocaleDateString(
        "fa-IR",
        options
      );

    badge.textContent =
      `پیش‌بینی برای تاریخ: ${dateStr}`;

  } catch (error) {

    console.error(
      "Forecast date error:",
      error
    );

    badge.textContent =
      "تاریخ پیش‌بینی در دسترس نیست";
  }
}


/**
 * fetch با timeout - جلوگیری از درخواست های آویزان
 *
 * @param {string} url - آدرس درخواست
 * @param {number} timeout - مدت زمان timeout (میلی ثانیه)
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(
  url,
  timeout = 7000
) {

  const controller =
    new AbortController();

  const id =
    setTimeout(
      () => controller.abort(),
      timeout
    );

  try {

    const response =
      await fetch(
        url,
        {
          signal:
            controller.signal,

          cache:
            "no-store"
        }
      );

    clearTimeout(id);

    if (!response.ok) {

      throw new Error(
        `HTTP ${response.status}`
      );
    }

    return response;

  } catch (error) {

    clearTimeout(id);

    throw error;
  }
}


/**
 * تعیین سطح خطر بر اساس مقدار عددی
 *
 * @param {number} value - مقدار شاخص خطر (0-100)
 * @returns {Object} - {label, color}
 */
function getRisk(value) {

  const num =
    Number(value);

  if (
    !Number.isFinite(num)
    ||
    num < 0
  ) {

    return {
      label: "بدون داده",
      color: "#777"
    };
  }

  const risks = [

    {
      min: 0,
      max: 20,
      label: "کم",
      color: "#2e7d32"
    },

    {
      min: 20,
      max: 40,
      label: "متوسط",
      color: "#c7a900"
    },

    {
      min: 40,
      max: 60,
      label: "زیاد",
      color: "#fb8c00"
    },

    {
      min: 60,
      max: 80,
      label: "خیلی زیاد",
      color: "#e53935"
    },

    {
      min: 80,
      max: 101,
      label: "بحرانی",
      color: "#880e4f"
    }
  ];

  return (
    risks.find(
      r =>
        num >= r.min
        &&
        num < r.max
    )
    ||
    risks[risks.length - 1]
  );
}


/**
 * تجزیه bounds از فرمت GeoJSON به object
 *
 * @param {Array} bounds - [[south, west], [north, east]]
 * @returns {Object|null}
 */
function parseBounds(bounds) {

  if (
    !Array.isArray(bounds)
    ||
    bounds.length < 2
  ) {

    return null;
  }

  return {

    south:
      Number(
        bounds[0][0]
      ),

    west:
      Number(
        bounds[0][1]
      ),

    north:
      Number(
        bounds[1][0]
      ),

    east:
      Number(
        bounds[1][1]
      )
  };
}


/**
 * اعتبارسنجی مختصات جغرافیایی
 *
 * @param {number} lat - عرض جغرافیایی
 * @param {number} lon - طول جغرافیایی
 * @returns {boolean}
 */
function validateCoords(
  lat,
  lon
) {

  return (

    Number.isFinite(lat)
    &&
    Number.isFinite(lon)

    &&

    lat >= -90
    &&
    lat <= 90

    &&

    lon >= -180
    &&
    lon <= 180
  );
}


/**
 * بروزرسانی گیج وضعیت
 *
 * @param {number} value - مقدار شاخص
 * @param {string} title - عنوان منطقه
 * @param {string} subtitle - زیرعنوان
 */
function updateGauge(
  value,
  title,
  subtitle
) {

  const labelElement =
    document.getElementById(
      "gaugeLabel"
    );

  const locationElement =
    document.getElementById(
      "gaugeLocation"
    );

  if (
    !Number.isFinite(
      Number(value)
    )
  ) {

    labelElement.textContent =
      "بدون داده";

    labelElement.style.color =
      "#777";

    locationElement.textContent =
      subtitle || "—";

    return;
  }

  const risk =
    getRisk(value);

  labelElement.textContent =
    risk.label;

  labelElement.style.color =
    risk.color;

  locationElement.textContent =
    `${title || ""}${subtitle ? " | " + subtitle : ""}`;
}


// ============================================================================
// تابع های نقشه - بارگذاری و مدیریت نقشه
// ============================================================================

/**
 * مقداردهی اولیه نقشه Leaflet
 */
function initMap() {

  APP_STATE.map =
    L.map(
      "map",
      {
        zoomControl: false,
        preferCanvas: true
      }
    );

  L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap"
    }
  ).addTo(
    APP_STATE.map
  );
}


/**
 * بارگذاری مرز استان فارس
 */
async function loadFars() {

  const response =
    await fetchWithTimeout(
      "fars.geojson"
    );

  const geojson =
    await response.json();

  APP_STATE.layers.fars =
    L.geoJSON(
      geojson,
      {
        style: {
          color: "#20252a",
          weight: 2.2,
          opacity: 0.95,
          fill: false
        },

        interactive: false
      }
    ).addTo(
      APP_STATE.map
    );

  return geojson;
}


/**
 * بارگذاری لایه شاخص خطر حریق (FLI)
 */
async function loadFLI() {

  const response =
    await fetchWithTimeout(
      "data/web/fli_polygons.geojson"
    );

  const geojson =
    await response.json();

  APP_STATE.layers.fli =
    L.geoJSON(
      geojson,
      {
        style: (feature) => ({

          color:
            feature.properties?.color
            ||
            "#777",

          weight: 0.15,

          opacity: 0.85,

          fillColor:
            feature.properties?.color
            ||
            "#777",

          fillOpacity: 0.72
        }),

        interactive: false
      }
    ).addTo(
      APP_STATE.map
    );

  return APP_STATE.layers.fli;
}


/**
 * بارگذاری متادیتای FLI
 */
async function loadFLIData() {

  try {

    const response =
      await fetchWithTimeout(
        "data/web/fli_latest.json"
      );

    APP_STATE.data.fliMetadata =
      await response.json();

  } catch (error) {

    console.warn(
      "Metadata error:",
      error
    );
  }
}


/**
 * بارگذاری داده های شطرنجی FLI
 */
async function loadGrid() {

  try {

    const response =
      await fetchWithTimeout(
        "data/web/fli_latest_grid.json"
      );

    const grid =
      await response.json();

    if (
      grid?.bounds
      &&
      Array.isArray(
        grid.values
      )
    ) {

      APP_STATE.data.fliGrid =
        grid;
    }

  } catch (error) {

    console.warn(
      "Grid error:",
      error
    );
  }
}


/**
 * بارگذاری لایه مناطق چهارگانه تحت مدیریت
 */
async function loadProtected() {

  const response =
    await fetchWithTimeout(
      "protected_areas.geojson"
    );

  const geojson =
    await response.json();

  APP_STATE.layers.protected =
    L.geoJSON(
      geojson,
      {

        style: {

          color: "#1565c0",

          weight: 1.8,

          opacity: 0.95,

          fillColor: "#42a5f5",

          fillOpacity: 0.15
        },

        onEachFeature:
          (
            feature,
            layer
          ) =>
            bindRegion(
              feature,
              layer,
              "protected"
            )
      }
    );

  return APP_STATE.layers.protected;
}


/**
 * بارگذاری لایه مناطق شکار ممنوع
 */
async function loadHunting() {

  const response =
    await fetchWithTimeout(
      "hunting_banned.geojson"
    );

  const geojson =
    await response.json();

  APP_STATE.layers.hunting =
    L.geoJSON(
      geojson,
      {

        style: {

          color: "#7b1fa2",

          weight: 1.8,

          opacity: 0.95,

          fillColor: "#ab47bc",

          fillOpacity: 0.15
        },

        onEachFeature:
          (
            feature,
            layer
          ) =>
            bindRegion(
              feature,
              layer,
              "hunting"
            )
      }
    );

  return APP_STATE.layers.hunting;
}


// ============================================================================
// تابع های داده - استخراج و محاسبه داده ها
// ============================================================================

/**
 * گرفتن مقدار FLI برای یک نقطه خاص
 *
 * @param {number} lat - عرض جغرافیایی
 * @param {number} lon - طول جغرافیایی
 * @returns {number|null} - مقدار FLI یا null
 */
function getPointValue(
  lat,
  lon
) {

  if (
    !validateCoords(
      lat,
      lon
    )
    ||
    !APP_STATE.data.fliGrid
  ) {

    return null;
  }

  const bounds =
    parseBounds(
      APP_STATE.data.fliGrid.bounds
    );

  if (!bounds) {
    return null;
  }

  const {
    south,
    west,
    north,
    east
  } = bounds;

  // بررسی آیا نقطه در محدوده bounds قرار دارد
  if (
    lat < south
    ||
    lat > north
    ||
    lon < west
    ||
    lon > east
  ) {

    return null;
  }

  const rows =
    Number(
      APP_STATE.data.fliGrid.rows
    );

  const cols =
    Number(
      APP_STATE.data.fliGrid.cols
    );

  if (
    rows <= 0
    ||
    cols <= 0
  ) {

    return null;
  }

  // محاسبه شاخص سطر و ستون
  let row =
    Math.floor(
      (
        (north - lat)
        /
        (north - south)
      )
      *
      rows
    );

  let col =
    Math.floor(
      (
        (lon - west)
        /
        (east - west)
      )
      *
      cols
    );

  // اطمینان از اینکه شاخص ها در محدوده معتبر هستند
  row =
    Math.max(
      0,
      Math.min(
        rows - 1,
        row
      )
    );

  col =
    Math.max(
      0,
      Math.min(
        cols - 1,
        col
      )
    );

  const value =
    Number(
      APP_STATE
        .data
        .fliGrid
        .values?.[row]?.[col]
    );

  return (
    Number.isFinite(value)
    &&
    value >= 0
    &&
    value <= 100
  )
    ? value
    : null;
}


/**
 * بررسی آیا یک نقطه درون مرز استان فارس قرار دارد
 *
 * @param {number} lat - عرض جغرافیایی
 * @param {number} lon - طول جغرافیایی
 * @returns {boolean}
 */
function pointInsideFars(
  lat,
  lon
) {

  if (
    !APP_STATE.layers.fars
    ||
    !validateCoords(
      lat,
      lon
    )
  ) {

    return false;
  }

  const point =
    turf.point(
      [
        lon,
        lat
      ]
    );

  for (
    const layer
    of APP_STATE.layers.fars.getLayers()
  ) {

    try {

      if (
        turf.booleanPointInPolygon(
          point,
          layer.toGeoJSON()
        )
      ) {

        return true;
      }

    } catch (error) {

      // ادامه به منطقه بعدی اگر خطا رخ داد
    }
  }

  return false;
}


/**
 * محاسبه میانگین FLI برای یک منطقه (Feature)
 * با نمونه برداری از داده های شطرنجی
 *
 * @param {Object} feature - GeoJSON Feature
 * @returns {number|null} - میانگین مقدار یا null
 */
function regionMean(
  feature
) {

  if (
    !APP_STATE.data.fliGrid
  ) {

    return null;
  }

  const bounds =
    parseBounds(
      APP_STATE.data.fliGrid.bounds
    );

  if (!bounds) {
    return null;
  }

  const {
    south,
    west,
    north,
    east
  } = bounds;

  const rows =
    Number(
      APP_STATE.data.fliGrid.rows
    );

  const cols =
    Number(
      APP_STATE.data.fliGrid.cols
    );

  if (
    rows <= 0
    ||
    cols <= 0
  ) {

    return null;
  }

  let sum = 0;

  let count = 0;

  // نمونه برداری هر 2 سطر و ستون برای بهبود عملکرد
  for (
    let row = 0;
    row < rows;
    row += 2
  ) {

    const lat =
      north
      -
      (
        (row + 0.5)
        /
        rows
      )
      *
      (
        north - south
      );

    for (
      let col = 0;
      col < cols;
      col += 2
    ) {

      const lon =
        west
        +
        (
          (col + 0.5)
          /
          cols
        )
        *
        (
          east - west
        );

      // بررسی آیا نقطه درون منطقه است
      if (
        turf.booleanPointInPolygon(
          turf.point(
            [
              lon,
              lat
            ]
          ),
          feature
        )
      ) {

        const value =
          Number(
            APP_STATE
              .data
              .fliGrid
              .values?.[row]?.[col]
          );

        if (
          Number.isFinite(value)
        ) {

          sum += value;

          count++;
        }
      }
    }
  }

  return (
    count > 0
      ? sum / count
      : null
  );
}


/**
 * اتصال popup و رویدادهای کلیک برای یک منطقه
 *
 * @param {Object} feature - GeoJSON Feature
 * @param {Object} layer - Leaflet Layer
 * @param {string} regionType - نوع منطقه
 */
function bindRegion(
  feature,
  layer,
  regionType
) {

  const name =
    feature.properties?.name
    ||
    `منطقه ${regionType}`;

  const mean =
    regionMean(
      feature
    );

  const risk =
    getRisk(
      mean
    );

  const regionLabel =
    regionType === "protected"
      ? "منطقه تحت مدیریت"
      : "منطقه شکار ممنوع";

  const popup = `
    <div class="region-popup">
      <div class="popup-title">
        ${name}
      </div>

      <div
        class="popup-risk-badge"
        style="background:${risk.color}"
      >
        ${risk.label}
      </div>

      <div class="popup-coord">
        (${regionLabel})
      </div>
    </div>
  `;

  layer.bindPopup(
    popup,
    {
      maxWidth: 260
    }
  );

  layer.on(
    "click",
    (e) => {

      L.DomEvent.stopPropagation(
        e
      );

      updateGauge(
        mean,
        name,
        regionLabel
      );
    }
  );
}


// ============================================================================
// تابع های تعامل کاربر - رویدادهای کلیک و انتخاب
// ============================================================================

/**
 * تنظیم رویداد کلیک روی نقشه
 */
function setupMapClick() {

  APP_STATE.map.on(
    "click",
    (event) => {

      const {
        lat,
        lng: lon
      } = event.latlng;

      // فقط اگر کلیک درون فارس باشد
      if (
        !pointInsideFars(
          lat,
          lon
        )
      ) {

        return;
      }

      const value =
        getPointValue(
          lat,
          lon
        );

      updateGauge(
        value,
        "استان فارس",
        `عرض: ${lat.toFixed(3)} | طول: ${lon.toFixed(3)}`
      );

      const risk =
        getRisk(
          value
        );

      const popup = `
        <div class="fli-popup">

          <div class="popup-title">
            شاخص خطر حریق
          </div>

          <div
            class="popup-risk-badge"
            style="background:${risk.color}"
          >
            ${risk.label}
          </div>

          <div class="popup-coord">
            عرض: ${lat.toFixed(4)}°
            |
            طول: ${lon.toFixed(4)}°
          </div>

        </div>
      `;

      // بستن popup قبلی اگر وجود داشت
      if (
        APP_STATE.ui.currentPopup
      ) {

        APP_STATE.map.removeLayer(
          APP_STATE.ui.currentPopup
        );
      }

      APP_STATE.ui.currentPopup =
        L.popup(
          {
            maxWidth: 260
          }
        )
        .setLatLng(
          event.latlng
        )
        .setContent(
          popup
        )
        .openOn(
          APP_STATE.map
        );
    }
  );
}


/**
 * تنظیم نمایش/پنهان کردن یک لایه
 *
 * @param {Object} layer - Leaflet Layer
 * @param {boolean} visible - آیا نمایش یابد یا خیر
 */
function setLayerVisible(
  layer,
  visible
) {

  if (!layer) {
    return;
  }

  if (visible) {

    if (
      !APP_STATE.map.hasLayer(
        layer
      )
    ) {

      layer.addTo(
        APP_STATE.map
      );
    }

  } else {

    if (
      APP_STATE.map.hasLayer(
        layer
      )
    ) {

      APP_STATE.map.removeLayer(
        layer
      );
    }
  }
}


/**
 * تنظیم dropdown منوی انتخاب لایه ها
 */
function setupDropdown() {

  const btn =
    document.getElementById(
      "layersBtn"
    );

  const menu =
    document.getElementById(
      "layersMenu"
    );

  const items =
    menu.querySelectorAll(
      ".dropdown-item"
    );

  // باز و بسته کردن منو
  btn.addEventListener(
    "click",
    (e) => {

      e.stopPropagation();

      menu.classList.toggle(
        "show"
      );
    }
  );

  // بستن منو هنگام کلیک خارج از آن
  document.addEventListener(
    "click",
    (e) => {

      if (
        !btn.contains(
          e.target
        )
        &&
        !menu.contains(
          e.target
        )
      ) {

        menu.classList.remove(
          "show"
        );
      }
    }
  );

  // رویداد انتخاب لایه
  items.forEach(
    item => {

      item.addEventListener(
        "click",
        (e) => {

          e.stopPropagation();

          const layer =
            item.dataset.layer;

          APP_STATE.ui.activeLayer =
            layer;

          // بروزرسانی active state
          items.forEach(
            i =>
              i.classList.remove(
                "active"
              )
          );

          item.classList.add(
            "active"
          );

          // تنظیم کدام لایه ها نمایش داده شوند
          const layerConfig = {

            fli: {
              fli: true,
              protected: false,
              hunting: false
            },

            protected: {
              fli: true,
              protected: true,
              hunting: false
            },

            hunting: {
              fli: true,
              protected: false,
              hunting: true
            },

            all: {
              fli: true,
              protected: true,
              hunting: true
            }
          };

          const config =
            layerConfig[layer]
            ||
            layerConfig.fli;

          setLayerVisible(
            APP_STATE.layers.fli,
            config.fli
          );

          setLayerVisible(
            APP_STATE.layers.protected,
            config.protected
          );

          setLayerVisible(
            APP_STATE.layers.hunting,
            config.hunting
          );

          menu.classList.remove(
            "show"
          );

          // تطبیق نقشه بر روی لایه انتخابی
          if (
            layer === "protected"
            &&
            APP_STATE.layers.protected
          ) {

            const bounds =
              APP_STATE.layers.protected.getBounds();

            if (
              bounds.isValid()
            ) {

              setTimeout(
                () => {

                  APP_STATE.map.fitBounds(
                    bounds,
                    {
                      padding: [
                        60,
                        60
                      ],
                      maxZoom: 10
                    }
                  );

                },
                100
              );
            }

          } else if (
            layer === "hunting"
            &&
            APP_STATE.layers.hunting
          ) {

            const bounds =
              APP_STATE.layers.hunting.getBounds();

            if (
              bounds.isValid()
            ) {

              setTimeout(
                () => {

                  APP_STATE.map.fitBounds(
                    bounds,
                    {
                      padding: [
                        60,
                        60
                      ],
                      maxZoom: 10
                    }
                  );

                },
                100
              );
            }
          }
        }
      );
    }
  );
}


/**
 * تنظیم حالت شب
 */
function setupNightMode() {

  document
    .getElementById(
      "modeButton"
    )
    .addEventListener(
      "click",
      function() {

        APP_STATE.ui.nightMode =
          !APP_STATE.ui.nightMode;

        document.body.classList.toggle(
          "night",
          APP_STATE.ui.nightMode
        );

        this.textContent =
          APP_STATE.ui.nightMode
            ? "🌙"
            : "☀️";
      }
    );
}


/**
 * تطبیق نقشه بر روی کل استان فارس
 */
function fitFars() {

  if (
    !APP_STATE.layers.fars
  ) {

    return;
  }

  const bounds =
    APP_STATE.layers.fars.getBounds();

  if (
    !bounds.isValid()
  ) {

    return;
  }

  APP_STATE.map.fitBounds(
    bounds,
    {
      padding: [
        40,
        40
      ],

      maxZoom: 8,

      animate: true
    }
  );

  setTimeout(
    () =>
      APP_STATE.map.invalidateSize(),
    100
  );
}


/**
 * تنظیم دکمه تطبیق نقشه
 */
function setupFitButton() {

  document
    .getElementById(
      "fitButton"
    )
    .addEventListener(
      "click",
      () =>
        fitFars()
    );
}


/**
 * تنظیم پنجره "درباره سامانه"
 */
function setupAbout() {

  const aboutOverlay =
    document.getElementById(
      "aboutOverlay"
    );

  document
    .getElementById(
      "aboutButton"
    )
    .addEventListener(
      "click",
      () =>
        aboutOverlay.classList.add(
          "open"
        )
    );

  document
    .getElementById(
      "aboutClose"
    )
    .addEventListener(
      "click",
      () =>
        aboutOverlay.classList.remove(
          "open"
        )
    );

  aboutOverlay.addEventListener(
    "click",
    (event) => {

      if (
        event.target ===
        aboutOverlay
      ) {

        aboutOverlay.classList.remove(
          "open"
        );
      }
    }
  );
}


// ============================================================================
// شروع برنامه
// ============================================================================

/**
 * تابع اصلی برای شروع کل برنامه
 *
 * ترتیب مهم:
 *
 * 1. نقشه ایجاد می‌شود
 * 2. FLI metadata و grid دریافت می‌شوند
 * 3. تاریخ از metadata واقعی تعیین می‌شود
 * 4. سپس سایر لایه‌ها بارگذاری می‌شوند
 *
 * بنابراین تاریخ هدر هرگز مستقل از FLI محاسبه نمی‌شود.
 */
async function startApp() {

  try {

    // ==========================================================
    // ۱. مقداردهی اولیه نقشه
    // ==========================================================

    initMap();


    // ==========================================================
    // ۲. دریافت metadata و grid
    // ==========================================================
    //
    // metadata حاوی forecast_date واقعی است.
    //
    // بنابراین قبل از ساخت هدر باید دریافت شود.
    // ==========================================================

    await Promise.all(
      [
        loadFLIData(),
        loadGrid()
      ]
    );


    // ==========================================================
    // ۳. تنظیم تاریخ بر اساس FLI واقعی
    // ==========================================================

    await setupTomorrowDate();


    // ==========================================================
    // ۴. بارگذاری مرز فارس
    // ==========================================================

    await loadFars();


    // ==========================================================
    // ۵. بارگذاری لایه FLI
    // ==========================================================

    await loadFLI();


    // ==========================================================
    // ۶. بارگذاری لایه های اختیاری
    // ==========================================================

    try {

      await Promise.all(
        [
          loadProtected(),
          loadHunting()
        ]
      );

    } catch (error) {

      console.warn(
        "Optional layers warning:",
        error
      );
    }


    // ==========================================================
    // ۷. تنظیم نمایش لایه ها
    // ==========================================================

    setLayerVisible(
      APP_STATE.layers.fli,
      true
    );

    setLayerVisible(
      APP_STATE.layers.protected,
      false
    );

    setLayerVisible(
      APP_STATE.layers.hunting,
      false
    );


    // ==========================================================
    // ۸. تطبیق نقشه و تنظیم رویدادهای کاربر
    // ==========================================================

    fitFars();

    setupMapClick();

    setupDropdown();

    setupNightMode();

    setupFitButton();

    setupAbout();


    // ==========================================================
    // ۹. تنظیم مجدد اندازه نقشه
    // ==========================================================

    setTimeout(
      () => {

        APP_STATE.map.invalidateSize();

        fitFars();

      },
      400
    );


  } catch (error) {

    console.error(
      "Startup error:",
      error
    );
  }
}


// ============================================================================
// شروع برنامه هنگام بارگذاری کامل DOM
// ============================================================================

document.addEventListener(
  "DOMContentLoaded",
  startApp
);
