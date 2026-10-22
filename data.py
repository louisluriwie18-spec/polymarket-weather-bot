"""
data.py — Data fetching module

Improvements in this version:
  - Fix 1: Retry with exponential backoff on all HTTP calls (2 retries, 1s/2s)
  - Fix 2: In-memory city coordinate cache — same city is never geocoded twice
            in the same process lifetime
"""
import asyncio
import logging
import math
import zoneinfo
import httpx
import json
import os
import numpy as np
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from config import WEATHERAPI_KEY, FORECAST_SIGMA

logger = logging.getLogger(__name__)


# NOTE: Historical data is cached in historical_cache.json to avoid repeated archive API calls.
# Do NOT add a proxy here — use the cache layer instead.


# ── City coordinate cache ──────────────────────────────────────────────
# In-memory cache persisted to disk so geocoding survives bot restarts.
# Without persistence, 30+ cities are re-geocoded on every startup (~30s delay,
# risk of rate-limiting the geocoding API).
_CITY_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_cache.json")
_CITY_CACHE: dict[str, dict | None] = {}
_city_cache_loaded = False
# asyncio.Lock protecting concurrent writes to _CITY_CACHE and city_cache.json.
# Python 3.10+ allows construction outside a running loop.
_city_cache_lock = asyncio.Lock()

# ── Met.no in-process cache ────────────────────────────────────────────────────
# Met.no's forecast won't change meaningfully within one bot run. Cache by
# (lat, lon, date_str, target_type) to avoid hammering the API for the same
# city twice (e.g. during the two-pass normalisation scan).
_metno_cache: dict[str, dict | None] = {}


def _load_city_cache() -> None:
    """Load city coordinate cache from disk (called once on first geocode request)."""
    global _city_cache_loaded
    if _city_cache_loaded:
        return
    _city_cache_loaded = True
    if os.path.exists(_CITY_CACHE_FILE):
        try:
            with open(_CITY_CACHE_FILE, "r", encoding="utf-8") as _f:
                _CITY_CACHE.update(json.load(_f))
            logger.debug("City cache loaded: %d entries from disk", len(_CITY_CACHE))
        except Exception as _e:
            logger.warning("Could not load city cache from disk: %s", _e)


def _save_city_cache() -> None:
    """Persist city coordinate cache to disk after each new geocode."""
    try:
        with open(_CITY_CACHE_FILE, "w", encoding="utf-8") as _f:
            json.dump(_CITY_CACHE, _f, indent=2)
    except Exception as _e:
        logger.warning("Could not save city cache to disk: %s", _e)

# ── City name aliases ──────────────────────────────────────────────────────────
# Maps Polymarket event display names to geocodable city names.
# Add entries here when Polymarket introduces new city title formats.
CITY_NAME_ALIASES: dict[str, str] = {
    "Seoul (Incheon)":  "Seoul",
    "NYC":              "New York City",
    "Mexico City":      "Mexico City, Mexico",  # disambiguate from Mexico City, Philippines
    "Panama City":      "Panama City, Panama",  # disambiguate from Panama City, Florida
    "Sao Paulo":        "São Paulo",
    "Buenos Aires":     "Buenos Aires",
    "Cape Town":        "Cape Town",
    "Tel Aviv":         "Tel Aviv",
    "Kuala Lumpur":     "Kuala Lumpur",
}


def _adjacent_day_suffixes(month: int, day: int, window: int = 1) -> set[str]:
    """
    Return a set of '-MM-DD' strings for the target date ±window days.

    Year 2000 is used as reference (leap year) so Feb 27–Mar 2 is handled
    correctly. The year portion is omitted — these are matched as suffixes
    against 'YYYY-MM-DD' strings from the Open-Meteo archive API.
    """
    ref = date(2000, month, day)
    return {
        (ref + timedelta(days=d)).strftime("-%m-%d")
        for d in range(-window, window + 1)
    }


# PolyNimbus city key mapping — keys are the EXACT city names extracted from
# Polymarket event titles; values are the PolyNimbus API city slugs.
# Source: https://www.polynimbus.club/#api
POLYNIMBUS_CITIES = {
    # Europe
    "London":           "london",
    "Paris":            "paris",
    "Madrid":           "madrid",
    "Milan":            "milan",
    "Munich":           "munich",
    "Warsaw":           "warsaw",
    "Istanbul":         "istanbul",
    "Ankara":           "ankara",
    # Asia-Pacific
    "Tokyo":            "tokyo",
    "Seoul":            "seoul",
    "Seoul (Incheon)":  "seoul",    # Polymarket title format — PolyNimbus key is "seoul"
    "Hong Kong":        "hong-kong",
    "Shanghai":         "shanghai",
    "Shenzhen":         "shenzhen",
    "Beijing":          "beijing",
    "Singapore":        "singapore",
    "Taipei":           "taipei",
    "Chengdu":          "chengdu",
    "Chongqing":        "chongqing",
    "Wuhan":            "wuhan",
    "Lucknow":          "lucknow",
    "Wellington":       "wellington",
    # Middle East / Africa
    "Tel Aviv":         "tel-aviv",
    # Americas
    "NYC":              "nyc",      # Polymarket uses "NYC", not "New York"
    "New York":         "nyc",
    "New York City":    "nyc",
    "Dallas":           "dallas",
    "Miami":            "miami",
    "Chicago":          "chicago",
    "Houston":          "houston",
    "Atlanta":          "atlanta",
    "Austin":           "austin",
    "Denver":           "denver",
    "Los Angeles":      "los-angeles",
    "San Francisco":    "san-francisco",
    "Seattle":          "seattle",
    "Toronto":          "toronto",
    "Buenos Aires":     "buenos-aires",
    "Sao Paulo":        "sao-paulo",  # Polymarket spells without accent
    "São Paulo":        "sao-paulo",
}


# ── Fix 1: HTTP retry helper ───────────────────────────────────────────────────
async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 15.0,
    headers: dict | None = None,
    retries: int = 2,
) -> httpx.Response | None:
    """
    GET with exponential-backoff retry.

    Retries on:
      - httpx transport errors (connect reset, timeout, DNS failure)
      - HTTP 429 (rate limit) and 5xx server errors

    Does NOT retry on 4xx client errors (bad key, not found, etc.).

    Returns the Response on success, None if all attempts fail.
    """
    delay = 1.0
    kwargs: dict = {"timeout": timeout}
    if headers:
        kwargs["headers"] = headers

    for attempt in range(1, retries + 2):  # retries+1 total attempts
        try:
            resp = await client.get(url, **kwargs)
            if resp.status_code < 400:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                # Check for Retry-After header
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                elif resp.status_code == 429:
                    # Check JSON body retryAfter (used by PolyNimbus API per their spec)
                    try:
                        body = resp.json()
                        json_retry = body.get("retryAfter")
                        # Use `is not None` so retryAfter=0 ("retry immediately") is
                        # honoured rather than falling through to the 15-second floor.
                        wait_time = float(json_retry) if json_retry is not None else max(delay, 15.0)
                    except Exception:
                        # No Retry-After header and no parseable JSON body:
                        # use a meaningful minimum. Open-Meteo archive needs ~15-30s.
                        wait_time = max(delay, 15.0)
                else:
                    wait_time = delay

                if attempt <= retries:
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d) — retrying in %.0fs",
                        resp.status_code, url[:60], attempt, retries + 1, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    if not retry_after:
                        delay *= 2
                else:
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d) — failed",
                        resp.status_code, url[:60], attempt, retries + 1,
                    )
            else:
                # 4xx — not retriable
                return resp
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.RequestError) as exc:
            if attempt <= retries:
                logger.warning(
                    "Network error fetching %s (attempt %d/%d): %s — retrying in %.0fs",
                    url[:60], attempt, retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.warning(
                    "Network error fetching %s (attempt %d/%d): %s — failed",
                    url[:60], attempt, retries + 1, exc,
                )
        except Exception as exc:
            logger.error("Unexpected error fetching %s: %s", url[:60], exc)
            return None

    logger.error("All %d attempts failed for %s", retries + 1, url[:60])
    return None


# ── Geocoding ──────────────────────────────────────────────────────────────────

async def get_city_coordinates(city_name: str) -> dict | None:
    # Normalise known Polymarket display names to geocodable names.
    # The original name is still used as the cache key so the alias fires only once.
    geocode_name = CITY_NAME_ALIASES.get(city_name, city_name)

    # Load from disk on first call — survives bot restarts, no repeat geocoding
    _load_city_cache()

    # Return cached result immediately — no API call needed
    if city_name in _CITY_CACHE:
        cached = _CITY_CACHE[city_name]
        if cached is not None:
            logger.debug("City cache hit: %s", city_name)
        return cached

    safe_city = urllib.parse.quote(geocode_name)
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={safe_city}&count=1&language=en&format=json"
    )
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    city = results[0]
                    result = {
                        "name": city["name"],
                        "lat": city["latitude"],
                        "lon": city["longitude"],
                        "timezone": city.get("timezone", "UTC"),
                    }
                    async with _city_cache_lock:
                        _CITY_CACHE[city_name] = result
                        _save_city_cache()
                    logger.debug("City geocoded and cached to disk: %s", city_name)
                    return result
        except Exception as e:
            logger.error("Error looking up city '%s': %s", city_name, e)

    # Cache the None result so we don't keep hammering the API for unknown cities
    async with _city_cache_lock:
        _CITY_CACHE[city_name] = None
        _save_city_cache()
    return None


# ── Market data ────────────────────────────────────────────────────────────────

async def get_market_resolution(market_id: str) -> dict | None:
    """Fetch market resolution from Gamma API.

    Returns:
        {"closed": True, "winner": "YES"|"NO"} on a normal resolution.
        {"closed": True, "winner": None, "voided": True} on a cancelled/voided market.
        None if the market is not yet resolved or an error occurred.
    """
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                if data.get("closed"):
                    resolution = (data.get("resolution") or "").upper()
                    if resolution in ("CANCELLED", "VOID", "INVALID"):
                        return {"closed": True, "winner": None, "voided": True}
                    
                    if resolution:
                        return {"closed": True, "winner": resolution}
                    return None  # closed=True but still settling
        except Exception as e:
            logger.error("Error fetching resolution for market %s: %s", market_id, e)
    return None


async def get_active_temperature_events() -> list[dict]:
    """Fetch active temperature-prediction events from the Polymarket Gamma API."""
    events = []
    cursor = None
    limit = 100
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        async def fetch_detail(evt):
            eid = evt.get("id")
            if not eid:
                return evt
            async with sem:
                r = await _get_with_retry(client, f"https://gamma-api.polymarket.com/events/{eid}")
                if r and r.status_code == 200:
                    return r.json()
                return evt

        while True:
            url = (
                f"https://gamma-api.polymarket.com/events/keyset"
                f"?active=true&closed=false&limit={limit}"
            )
            if cursor and cursor != "null":
                url += f"&after_cursor={cursor}"
                
            try:
                resp = await _get_with_retry(client, url)
                if resp is None or resp.status_code != 200:
                    break
                data = resp.json()
                if not data:
                    break
                    
                events_list = data.get("events", [])
                # Only break if the API returned no events at all (true end of pagination).
                # Do NOT break on an empty temperature sub-list — non-temperature events
                # may dominate a page while temperature events appear on later pages.
                if not events_list:
                    break
                    
                matching_events = [e for e in events_list if "temperature" in e.get("title", "").lower()]
                if matching_events:
                    detailed_results = await asyncio.gather(
                        *[fetch_detail(e) for e in matching_events],
                        return_exceptions=True,
                    )
                    # Filter out exceptions so one malformed event JSON cannot
                    # cause the entire page of events to be abandoned.
                    events.extend(
                        r for r in detailed_results if not isinstance(r, BaseException)
                    )
                        
                cursor = data.get("next_cursor")
                if not cursor or cursor == "null":
                    break
            except Exception as e:
                logger.error("Error fetching Gamma API events: %s", e)
                break
    return events


# ── Weather forecasts ──────────────────────────────────────────────────────────

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_cache.json")
_historical_cache = None
_historical_cache_lock = asyncio.Lock()

def _load_historical_cache():
    global _historical_cache
    if _historical_cache is not None:
        return
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _historical_cache = json.load(f)
        except Exception as e:
            logger.error("Failed to load historical cache: %s", e)
            _historical_cache = {}
    else:
        _historical_cache = {}

async def _save_historical_cache_async():
    """Asynchronously and safely save historical cache to disk."""
    async with _historical_cache_lock:
        try:
            tmp_file = f"{CACHE_FILE}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(_historical_cache, f)
            os.replace(tmp_file, CACHE_FILE)
        except Exception as e:
            logger.error("Failed to save historical cache: %s", e)

def _save_historical_cache():
    try:
        tmp_file = f"{CACHE_FILE}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(_historical_cache, f)
        os.replace(tmp_file, CACHE_FILE)
    except Exception as e:
        logger.error("Failed to save historical cache: %s", e)

async def get_historical_temps(
    lat: float, lon: float, event_date: date, target_type: str = "max"
) -> list[tuple[int, float]]:
    """
    Fetch historical daily max/min temperatures for the target calendar date ±2 days.

    Accuracy improvements:
      - Fix F: Archive extended from 2014 → 2000 (~2× more history)
      - Fix B: ±2 calendar day window (~5× more data points per year)
      - Returns (year, temp_celsius) pairs so the model can apply climate
        trend correction (Fix C) without a second API call

    The user's concern about climate-change outliers is handled in model.py:
    _apply_climate_correction() detrends the data first, then applies MAD
    filtering on the residuals — so a year that is 'hot due to climate warming'
    is captured by the trend, NOT removed as an outlier.

    Returns: list[(year, temp_celsius)] sorted by year ascending.
    """
    _load_historical_cache()
    
    # We round lat/lon slightly for the cache key to increase hit rate on tiny variations
    lat_r, lon_r = round(lat, 2), round(lon, 2)
    cache_key = f"{lat_r},{lon_r}_{event_date.month:02d}-{event_date.day:02d}_{target_type}"
    
    if cache_key in _historical_cache:
        logger.debug("Cache hit for %s", cache_key)
        # JSON loads lists, convert back to tuples
        return [tuple(x) for x in _historical_cache[cache_key]]

    suffixes = _adjacent_day_suffixes(event_date.month, event_date.day, window=2)
    end_date_limit = event_date - timedelta(days=1)
    end_date_str = min(datetime.now(timezone.utc).date(), end_date_limit).strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date=2000-01-01&end_date={end_date_str}"
        f"&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
    )
    # Brief sleep before each archive call to spread load and avoid rate-limiting.
    # Without this, parallel asyncio.gather() calls across events can burst the archive API.
    await asyncio.sleep(2.0)
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url, timeout=35.0, retries=4)
            if resp is None or resp.status_code != 200:
                logger.error(
                    "Historical data fetch failed: HTTP %s — will try NASA POWER fallback",
                    resp.status_code if resp else "no response",
                )
                # fall through to NASA POWER fallback (no return here)
            else:
                data = resp.json()
                daily = data.get("daily", {})
                times = daily.get("time", [])  # list of "YYYY-MM-DD" strings

                target_key = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
                target_data = daily.get(target_key, [])

                # Collect (year, temp) pairs for all dates in the ±1-day window
                result: list[tuple[int, float]] = []
                for i, t in enumerate(times):
                    day_suffix = t[4:]  # "-MM-DD" portion
                    if day_suffix in suffixes:
                        if target_data and i < len(target_data) and target_data[i] is not None:
                            year = int(t[:4])
                            result.append((year, float(target_data[i])))

                result.sort(key=lambda x: x[0])
                logger.debug(
                    "Historical: %d points across %d unique years (±2-day window, 2000–present)",
                    len(result),
                    len({y for y, _ in result}),
                )
                # Save to cache and return — success path exits here
                _historical_cache[cache_key] = result
                _save_historical_cache()
                return result
        except Exception as e:
            logger.error("Error fetching historical temps from Open-Meteo: %s", e)
            # fall through to NASA POWER fallback

    # ── NASA POWER fallback ─────────────────────────────────────────────────
    # Triggered if Open-Meteo archive returned [] (all retries failed/429) or threw.
    logger.warning(
        "Open-Meteo archive failed for (%.2f, %.2f) — trying NASA POWER fallback",
        lat, lon,
    )
    nasa_result = await _get_nasa_power_historical(lat, lon, event_date, target_type)
    if nasa_result:
        _historical_cache[cache_key] = nasa_result
        _save_historical_cache()
        logger.info(
            "  ✓ NASA POWER supplied %d points — cached for future runs", len(nasa_result)
        )
    return nasa_result

async def _get_nasa_power_historical(
    lat: float, lon: float, event_date: date, target_type: str = "max"
) -> list[tuple[int, float]]:
    """
    Fallback historical temperature source: NASA POWER API.

    Why NASA POWER:
      - Completely free, no API key required
      - Global coverage (satellite-derived, works anywhere on Earth)
      - Daily Tmax/Tmin from 1981 to yesterday
      - Very liberal rate limits (NASA public science data)
      - No registration or credential management needed

    API endpoint: https://power.larc.nasa.gov/api/temporal/daily/point
    Parameters: T2M_MAX (daily max 2m temp °C), T2M_MIN (daily min 2m temp °C)
    Missing value sentinel: -999  (filtered out)

    Returns: list[(year, temp_celsius)] for dates in the ±1-day window, sorted
             ascending by year, exactly matching the format of get_historical_temps.
    """
    param = "T2M_MAX" if target_type == "max" else "T2M_MIN"
    end_date_str = (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y%m%d")
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={param}&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start=20000101&end={end_date_str}"
        f"&format=JSON"
    )
    # Use the same ±2-day window as the primary source (get_historical_temps) so
    # cities served by the NASA POWER fallback get the same number of data points
    # per year and the same prior quality as Open-Meteo-served cities.
    suffixes = _adjacent_day_suffixes(event_date.month, event_date.day, window=2)
    logger.info("  NASA POWER fallback: fetching %s historical data for (%.2f, %.2f)", param, lat, lon)
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url, retries=2, timeout=60.0)
            if resp is None or resp.status_code != 200:
                logger.error("NASA POWER API returned HTTP %s",
                             resp.status_code if resp else "no response")
                return []
            data = resp.json()
            raw = data.get("properties", {}).get("parameter", {}).get(param, {})
            if not raw:
                logger.warning("NASA POWER: no data in response for %s", param)
                return []
            result: list[tuple[int, float]] = []
            for date_str_key, val in raw.items():
                # date_str_key format: "YYYYMMDD"
                if len(date_str_key) != 8:
                    continue
                year  = int(date_str_key[:4])
                month = int(date_str_key[4:6])
                day   = int(date_str_key[6:8])
                suffix = f"-{month:02d}-{day:02d}"
                if suffix in suffixes and isinstance(val, (int, float)) and val > -990:
                    result.append((year, float(val)))
            result.sort(key=lambda x: x[0])
            logger.info(
                "  NASA POWER: %d historical points retrieved (%.2f, %.2f)",
                len(result), lat, lon,
            )
            return result
        except Exception as e:
            logger.error("NASA POWER fetch failed: %s", e)
            return []

async def get_openmeteo_ensemble_std(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> float | None:
    """
    Return the ICON ensemble forecast spread as the true forecast uncertainty
    for the target date. Uses Open-Meteo's free ICON seamless ensemble (~40 members).

    Why this matters (Fix A):
      The previous std was std(hourly readings within the day) — this measures
      the diurnal temperature swing (~4–6°C), NOT how uncertain the forecast is.
      The ensemble spread is the standard meteorological measure of next-day
      forecast uncertainty and is typically 0.5–2°C for max/min temp.

    Returns None if the ensemble API is unavailable — the caller falls back to
    the diurnal std from get_openmeteo_forecast.
    """
    url = (
        f"https://ensemble-api.open-meteo.com/v1/ensemble"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m"
        f"&models=icon_seamless"
        f"&timezone=auto"
    )
    async with httpx.AsyncClient() as client:
        try:
            # Only 1 retry — this is enhancement, not critical path
            resp = await _get_with_retry(client, url, retries=1, timeout=15.0)
            if resp is None or resp.status_code != 200:
                logger.debug(
                    "Ensemble API unavailable (HTTP %s) — using diurnal std fallback",
                    resp.status_code if resp else "no response",
                )
                return None

            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            # Collect all member columns (e.g. temperature_2m_member01 … member40)
            member_cols = sorted(
                k for k in hourly.keys() if k.startswith("temperature_2m_member")
            )
            if not member_cols:
                logger.warning(
                    "Ensemble API: no member columns found in response — "
                    "falling back to diurnal std (reduced uncertainty accuracy)"
                )
                return None

            # Per member: compute daily max or min for the target date
            member_vals: list[float] = []
            for col in member_cols:
                member_temps = hourly[col]
                day_temps = [
                    member_temps[i]
                    for i, t in enumerate(times)
                    if t.startswith(date_str)
                    and i < len(member_temps)
                    and member_temps[i] is not None
                ]
                if day_temps:
                    val = max(day_temps) if target_type == "max" else min(day_temps)
                    member_vals.append(val)

            if len(member_vals) < 5:
                logger.debug("Ensemble: insufficient members (%d)", len(member_vals))
                return None

            ens_std = float(np.std(member_vals))
            ens_std = max(0.3, ens_std)  # floor: even perfect forecasts have ≥0.3°C uncertainty

            logger.info(
                "  Ensemble (%d members): mean=%.1f°C  spread=%.2f°C",
                len(member_vals), float(np.mean(member_vals)), ens_std,
            )
            return ens_std

        except Exception as e:
            logger.debug("Ensemble fetch error: %s", e)
            return None


# ── Batched Open-Meteo NWP forecast ───────────────────────────────────────────

async def get_openmeteo_all_models(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """
    Fetch ECMWF IFS, ICON seamless, GFS seamless, and UKMO seamless from
    Open-Meteo in ONE batched HTTP request.

    Returns: (forecast_ec, forecast_om, forecast_gfs, forecast_ukmet)
    Any model whose column is absent or null is returned as None.

    Why batched:
      Previously 4 separate requests to api.open-meteo.com (same server pool).
      One multi-model call returns all four in a single JSON payload, reducing
      Open-Meteo API calls by 75% and nearly eliminating burst-limit risk.

    Column format from Open-Meteo multi-model response:
      temperature_2m_max_{model_id}  e.g. temperature_2m_max_ecmwf_ifs025
    """
    key  = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
    # Model ids in the order Open-Meteo expects them
    _MODEL_IDS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless", "ukmo_seamless"]
    # Map model_id -> (source_key_in_FORECAST_SIGMA, return_position)
    _MODEL_MAP = {
        "ecmwf_ifs025":  "ecmwf",
        "icon_seamless": "openmeteo",
        "gfs_seamless":  "gfs",
        "ukmo_seamless": "ukmet",
    }
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily={key}"
        f"&models={',' .join(_MODEL_IDS)}"
        f"&timezone=auto"
    )
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is None or resp.status_code != 200:
                logger.warning(
                    "Open-Meteo batch fetch failed (HTTP %s) — models will be None",
                    resp.status_code if resp else "no response",
                )
                return None, None, None, None

            data  = resp.json()
            daily = data.get("daily", {})

            results: dict[str, dict | None] = {}
            for model_id in _MODEL_IDS:
                source_key = _MODEL_MAP[model_id]
                col  = f"{key}_{model_id}"
                vals = daily.get(col, [])
                if vals and vals[0] is not None:
                    sigma = FORECAST_SIGMA.get(source_key, 2.0)
                    results[source_key] = {"mean": float(vals[0]), "std": sigma}
                    logger.debug(
                        "  OM-batch %s: %.1f°C ± %.1f°C",
                        source_key.upper(), float(vals[0]), sigma,
                    )
                else:
                    results[source_key] = None
                    logger.debug("  OM-batch %s: no data", source_key.upper())

            return (
                results.get("ecmwf"),
                results.get("openmeteo"),
                results.get("gfs"),
                results.get("ukmet"),
            )

        except Exception as e:
            logger.error("Error in Open-Meteo batch fetch: %s", e)
            return None, None, None, None


async def get_openmeteo_forecast(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> dict | None:
    """
    Fetch daily max/min temperature from Open-Meteo's daily endpoint (ICON only).

    Kept as a standalone fallback in case the batched multi-model request fails.
    The primary call path uses get_openmeteo_all_models() instead.
    std defaults to 2.0°C (ICON seamless day-ahead RMSE); replaced by ensemble
    spread when get_openmeteo_ensemble_std succeeds.
    """
    key = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily={key}&timezone=auto"
    )
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                temps = daily.get(key, [])
                if temps and temps[0] is not None:
                    return {"mean": float(temps[0]), "std": 2.0}
        except Exception as e:
            logger.error("Error fetching Open-Meteo forecast: %s", e)
    return None


async def get_ecmwf_forecast(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> dict | None:
    """
    Fetch daily max/min temperature from ECMWF IFS via Open-Meteo.

    Uses the dedicated ECMWF endpoint (/v1/ecmwf) which is the most reliable
    path. Falls back to /v1/forecast?models=ecmwf_ifs025 if needed.
    IMPORTANT: ecmwf_ifs04 and ecmwf_aifs025 both return None for daily
    temperature_2m_max/min — do NOT use those model names.

    ECMWF IFS is globally ranked #1 for medium-range temperature skill.
    std defaults to 1.8°C (ECMWF day-ahead RMSE is lower than ICON/GFS).
    """
    key = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
    # Primary: dedicated ECMWF endpoint (most reliable, verified returning data)
    primary_url = (
        f"https://api.open-meteo.com/v1/ecmwf"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily={key}&timezone=auto"
    )
    # Fallback: standard forecast endpoint with ifs025 model parameter
    fallback_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily={key}&timezone=auto"
        f"&models=ecmwf_ifs025"
    )
    async with httpx.AsyncClient() as client:
        for url in (primary_url, fallback_url):
            try:
                resp = await _get_with_retry(client, url)
                if resp is not None and resp.status_code == 200:
                    data = resp.json()
                    daily = data.get("daily", {})
                    temps = daily.get(key, [])
                    if temps and temps[0] is not None:
                        logger.debug("  ECMWF IFS: %.1f°C (%s)", float(temps[0]), key)
                        return {"mean": float(temps[0]), "std": 1.8}
            except Exception as e:
                logger.debug("ECMWF fetch attempt failed (%s): %s", url[:50], e)
    logger.warning("ECMWF forecast unavailable for (%.2f, %.2f) %s", lat, lon, date_str)
    return None


async def get_gfs_forecast(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> dict | None:
    """
    Fetch daily max/min temperature from GFS (NOAA/NWS) via Open-Meteo.

    GFS seamless combines GFS and GFS-HRRR for the best available resolution.
    Particularly strong for North/South America. std=2.0°C (GFS day-ahead RMSE
    is slightly higher than ECMWF but still excellent).
    Verified: returns data for European coordinates too.
    """
    key = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily={key}&timezone=auto"
        f"&models=gfs_seamless"
    )
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                temps = daily.get(key, [])
                if temps and temps[0] is not None:
                    logger.debug("  GFS: %.1f°C (%s)", float(temps[0]), key)
                    return {"mean": float(temps[0]), "std": 2.0}
        except Exception as e:
            logger.error("Error fetching GFS forecast: %s", e)
    return None


async def get_ukmet_forecast(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> dict | None:
    """
    Fetch daily max/min temperature from UK Met Office model via Open-Meteo.

    UKMET (ukmo_seamless) is particularly accurate for European cities and
    provides a valuable independent NWP signal. std=1.9°C.
    Verified: returns data globally (not just UK coverage).
    """
    key = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily={key}&timezone=auto"
        f"&models=ukmo_seamless"
    )
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                temps = daily.get(key, [])
                if temps and temps[0] is not None:
                    logger.debug("  UKMET: %.1f°C (%s)", float(temps[0]), key)
                    return {"mean": float(temps[0]), "std": 1.9}
        except Exception as e:
            logger.error("Error fetching UKMET forecast: %s", e)
    return None


async def get_metno_forecast(
    lat: float, lon: float, date_str: str, tz_name: str, target_type: str = "max"
) -> dict | None:
    """Fetch forecast from Met.no (Norwegian Meteorological Institute).

    Results are cached in-memory by (lat, lon, date_str, target_type).
    Met.no's API terms of service ask callers to respect If-Modified-Since;
    caching within a run is the simplest compliant approach.
    """
    lat_r, lon_r = round(lat, 4), round(lon, 4)
    cache_key = f"{lat_r},{lon_r}_{date_str}_{target_type}"
    if cache_key in _metno_cache:
        logger.debug("Met.no cache hit: %s", cache_key)
        return _metno_cache[cache_key]

    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat_r}&lon={lon_r}"
    headers = {
        "User-Agent": "PolymarketWeatherBot/1.0 github.com/lordoshin/polymarket-weather-bot"
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url, headers=headers)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                timeseries = data.get("properties", {}).get("timeseries", [])
                day_temps = []

                try:
                    tz = zoneinfo.ZoneInfo(tz_name)
                except Exception:
                    tz = zoneinfo.ZoneInfo("UTC")

                for ts in timeseries:
                    time_str = ts.get("time", "")
                    if time_str:
                        try:
                            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                            local_dt = dt.astimezone(tz)
                            if local_dt.strftime("%Y-%m-%d") == date_str:
                                t = (
                                    ts.get("data", {})
                                    .get("instant", {})
                                    .get("details", {})
                                    .get("air_temperature")
                                )
                                if t is not None:
                                    day_temps.append(t)
                        except ValueError:
                            pass

                if day_temps:
                    val = max(day_temps) if target_type == "max" else min(day_temps)
                    # Fixed 2.0°C: realistic day-ahead RMSE for a single NWP model.
                    # Previously used diurnal std (~4–6°C) which made Met.no's likelihood
                    # near-uniform, wasting its 10% source weight entirely.
                    result = {"mean": val, "std": 2.0}
                    _metno_cache[cache_key] = result
                    return result
        except Exception as e:
            logger.error("Error fetching Met.no forecast: %s", e)
    _metno_cache[cache_key] = None
    return None


# In-memory snapshot cache for PolyNimbus: (city_key, date_str) -> (cached_at_timestamp, loc_dict)
# PolyNimbus snapshots contain both min, max, hourly points, and METAR.
# Caching for 15 minutes (900s) eliminates redundant requests across High/Low markets.
_POLYNIMBUS_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_POLYNIMBUS_CACHE_TTL = 900.0


async def get_polynimbus_forecast(
    city_name: str, date_str: str, target_type: str = "max"
) -> dict | None:
    """
    Fetch aggregated ensemble forecast from PolyNimbus API.

    Returns a dict with:
      - "mean"         float: predicted max or min temp for date_str (°C)
      - "std"          float: hourly spread (used as uncertainty proxy)
      - "metar_temp_c" float | None: most recent METAR observation temperature
                       for TODAY (same-day ground truth anchor); None if unavailable.
    """
    city_key = POLYNIMBUS_CITIES.get(city_name)
    if not city_key:
        logger.debug("City %s not tracked by PolyNimbus (covers 36 cities).", city_name)
        return None

    now_ts = datetime.now(timezone.utc).timestamp()
    cache_entry = _POLYNIMBUS_CACHE.get((city_key, date_str))
    loc: dict | None = None
    if cache_entry and (now_ts - cache_entry[0]) < _POLYNIMBUS_CACHE_TTL:
        loc = cache_entry[1]

    if loc is None:
        # Use explicit date= parameter (overrides day=) to avoid server-side timezone ambiguity.
        url = f"https://www.polynimbus.club/api/weather/snapshots?city={city_key}&date={date_str}"
        headers = {
            "User-Agent": "WeatherTradingBot/2.0 (contact@weatherbot.local)",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                # 25s timeout accommodates Vercel serverless cold starts
                resp = await _get_with_retry(client, url, timeout=25.0, headers=headers)
                if resp is not None and resp.status_code == 200:
                    data = resp.json()
                    if "locations" in data and len(data["locations"]) > 0:
                        loc = data["locations"][0]
                        _POLYNIMBUS_CACHE[(city_key, date_str)] = (now_ts, loc)
                elif resp is not None and resp.status_code != 200:
                    logger.debug("PolyNimbus API returned HTTP %d for %s", resp.status_code, city_key)
            except Exception as e:
                logger.error("Error fetching PolyNimbus API for %s: %s", city_key, e)

    if loc is None:
        return None

    points = loc.get("points", [])

    # Forecast temps for the requested date
    day_temps = [
        pt["tempC"]
        for pt in points
        if pt.get("isoLocal", "").startswith(date_str) and "tempC" in pt
    ]

    # Extract latest METAR observation (ground truth anchor)
    metar_temp_c: float | None = None
    metar_raw = loc.get("metar") or loc.get("obs")
    if isinstance(metar_raw, dict):
        latest = metar_raw.get("latest")
        if isinstance(latest, dict) and "tempC" in latest:
            metar_temp_c = float(latest["tempC"])
        elif "tempC" in metar_raw or "temp_c" in metar_raw:
            metar_temp_c = float(metar_raw.get("tempC") or metar_raw.get("temp_c"))
    elif isinstance(metar_raw, (int, float)):
        metar_temp_c = float(metar_raw)

    val: float | None = None
    std: float = 2.0
    if day_temps:
        val = min(day_temps) if target_type == "min" else max(day_temps)
        std = float(np.std(day_temps)) if len(day_temps) > 1 else 1.5
        std = max(0.5, std)
    else:
        # Fallback: check loc-level summary fields if points list is empty
        loc_val = loc.get("minTempC") if target_type == "min" else loc.get("maxTempC")
        if loc_val is not None:
            val = float(loc_val)

    if val is not None and math.isfinite(val):
        result = {"mean": val, "std": std, "metar_temp_c": metar_temp_c}
        if metar_temp_c is not None:
            logger.debug(
                "  PolyNimbus METAR anchor: %.1f°C (today, ground truth)",
                metar_temp_c,
            )
        return result

    return None


# == Recent daily series (for QuantEnsemble) ==================================

async def get_recent_daily_series(
    lat: float,
    lon: float,
    target_date: date,
    days_back: int = 60,
    target_type: str = "max",
) -> list[float]:
    """
    Fetch the last `days_back` calendar days of daily max/min temperature as a
    sequential list ordered oldest -> newest.
    series[-1] = yesterday's actual, series[-2] = two days ago, etc.

    Used exclusively by QuantEnsemble as its input time series.

    Note: this is an additional archive API call issued concurrently with the
    NWP forecast batch via asyncio.gather(), so it adds no serial latency.
    A 1-second delay spreads the load across the two archive calls.
    """
    key   = "temperature_2m_max" if target_type == "max" else "temperature_2m_min"
    start = (target_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end   = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily={key}&timezone=auto"
    )
    await asyncio.sleep(1.0)   # brief stagger from the main historical archive call
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is None or resp.status_code != 200:
                logger.warning(
                    "Recent daily series fetch failed: HTTP %s",
                    resp.status_code if resp else "no response",
                )
                return []
            data   = resp.json()
            vals   = data.get("daily", {}).get(key, [])
            result = [float(v) for v in vals if v is not None]
            logger.debug(
                "Recent %s series: %d days fetched (%.2f, %.2f)",
                target_type, len(result), lat, lon,
            )
            return result
        except Exception as e:
            logger.error("Error fetching recent daily series: %s", e)
            return []


async def get_weatherapi_forecast(
    lat: float, lon: float, date_str: str, target_type: str = "max"
) -> dict | None:
    if not WEATHERAPI_KEY:
        return None
    url = (
        f"https://api.weatherapi.com/v1/forecast.json"
        f"?key={WEATHERAPI_KEY}&q={lat},{lon}&days=2"
    )
    async with httpx.AsyncClient() as client:
        try:
            resp = await _get_with_retry(client, url)
            if resp is None or resp.status_code != 200:
                logger.error(
                    "WeatherAPI fetch failed: HTTP %s",
                    resp.status_code if resp else "no response",
                )
                return None
            data = resp.json()
            forecast = data.get("forecast", {}).get("forecastday", [])
            for day in forecast:
                if day.get("date") == date_str:
                    key = "maxtemp_c" if target_type == "max" else "mintemp_c"
                    temp = day.get("day", {}).get(key)
                    if temp is not None:
                        # WeatherAPI is a commercial proprietary model with higher
                        # day-ahead RMSE than ECMWF/ICON. Using 2.5°C (vs 2.0°C)
                        # gives it appropriately lower precision weight in IVW.
                        return {"mean": temp, "std": 2.5}
            logger.warning(
                "WeatherAPI: target date %s not found in %d forecast days returned",
                date_str, len(forecast),
            )
        except Exception as e:
            logger.error("Error fetching WeatherAPI forecast: %s", e)
    return None