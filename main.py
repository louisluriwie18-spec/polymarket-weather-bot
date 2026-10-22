"""
main.py  Entry point, scheduler, and main trading loop.

Robustness improvements applied here:
  - Fix 3: Robust city regex with 3 fallback patterns
  - Fix 4: Minimum 5-year historical sample guard per market
  - Fix 7: Scheduler while-loop catches unexpected exceptions
  - Fix 8: Startup WeatherAPI key validation

Orchestrates the full weather-trading pipeline:
  1. Discover active temperature EVENTS from Polymarket
  2. Filter to TOMORROW's events only
  3. For each event: look up city coords → fetch data → process sub-markets
  4. For each sub-market: parse groupItemTitle → run ensemble model → detect edge
  5. Rank ALL edges across all events by score
  6. Record the top N recommendations in trades.db (prevents duplicates)
  7. Print a clean recommendations table to the terminal
  8. Daily 12:00 UTC: check resolved trades and update PnL

Manual Trading:
  - Review the TRADE RECOMMENDATIONS table printed each cycle
  - Place trades manually on polymarket.com
  - Run `python view_db.py` to see recorded positions and PnL history

Safety: Simulation-only (no live orders). Trades.db tracks recommendations
to prevent the same market being recommended twice while OPEN.
"""

import asyncio
import json
import logging
import math
import re
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import zoneinfo
from datetime import date, datetime, timedelta, timezone
from typing import Any

import schedule
import httpx
import numpy as np

from config import (
    DRY_RUN,
    DRY_RUN_STARTING_BANKROLL,
    MAX_FORECAST_DISAGREEMENT,
    MAX_TRADES_PER_RUN,
    MIN_EDGE_THRESHOLD,
    API_CONCURRENCY_LIMIT,
    API_REQUEST_DELAY,
    WEATHERAPI_KEY,
    THRESHOLD_BY_TYPE,
    STATION_COORDS,
    CITY_ALLOWLIST,
)
from data import (
    POLYNIMBUS_CITIES,
    _get_with_retry,
    get_active_temperature_events,
    get_historical_temps,
    get_openmeteo_ensemble_std,
    get_openmeteo_all_models,
    get_recent_daily_series,
    get_weatherapi_forecast,
    get_metno_forecast,
    get_polynimbus_forecast,
    get_city_coordinates,
)
from quant_ensemble import QuantEnsemble
from model import (
    check_forecast_agreement,
    detect_edge,
    parse_group_item_title,
    run_forecast_model,
)
from trader import (
    check_trade_cooldown,
    execute_trade,
    get_bankroll,
    get_current_stake,
    get_kelly_stake,
    log_trade,
)
from database import init_db, get_total_open_count, get_open_trades

from logging.handlers import RotatingFileHandler


class SafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe rotating file handler that ignores file locking during rollover."""

    def doRollover(self):
        try:
            super().doRollover()
        except (PermissionError, OSError) as e:
            # Log to stderr rather than silently swallowing: a failed rollover
            # means the log file will grow unboundedly until disk is full.
            print(f"WARNING: Log rotation failed: {e}", file=sys.stderr)


# ── Logging setup ──────────────────────────────────────────────────────────────
import os as _os
_LOG_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(module)-12s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        SafeRotatingFileHandler(_LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=2, encoding="utf-8", delay=True),
    ],
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── Date Parsing ───────────────────────────────────────────────────────────────

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_event_date(date_str: str) -> date | None:
    """Parse a date string like 'June 8' into a date object (current year)."""
    date_str = date_str.strip().rstrip("?")
    m = re.match(r"(\w+)\s+(\d+)", date_str)
    if not m:
        logger.warning("Could not parse event date: '%s'", date_str)
        return None

    month_name = m.group(1).lower()
    day = int(m.group(2))
    month = _MONTHS.get(month_name)
    if not month:
        logger.warning("Unknown month: '%s'", month_name)
        return None

    try:
        current_date = datetime.now(timezone.utc).date()
        target_year = current_date.year
        # Year boundary: if we are in Q4 (Oct–Dec) and the event month is in Q1
        # (Jan–Mar), the event belongs to next year. The previous "gap >= 6 months"
        # check incorrectly promoted mid-year past events to next year
        # (e.g. today=July, event month=January → gap=6 ≥ 6 → wrongly +1 year).
        if month <= 3 and current_date.month >= 10:
            target_year += 1
        result = date(target_year, month, day)
    except ValueError:
        logger.warning("Invalid date: %s %d", month_name, day)
        return None

    return result


def get_tomorrow() -> date:
    """Get tomorrow's date in UTC."""
    return datetime.now(timezone.utc).date() + timedelta(days=1)


# ── Terminal Output Helpers ────────────────────────────────────────────────────

def _divider(char: str = "=", width: int = 72) -> str:
    return char * width


def _print_header(run_number: int, tomorrow: date) -> None:
    """Print the scan-cycle header banner."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    print()
    print(_divider("="))
    print(f"  [Bot] POLYMARKET WEATHER BOT — Run #{run_number}")
    print(f"  Scanning for:  {tomorrow.strftime('%A, %B %d %Y')} (tomorrow)")
    print(f"  Time (UTC):    {now_str}")
    print(f"  Edge threshold: {MIN_EDGE_THRESHOLD * 100:.0f}%  |  Max open: {MAX_TRADES_PER_RUN}")
    print(_divider("="))
    print()


def _print_recommendations(
    candidates: list,
    total_open: int,
    bankroll: float,
    executed: list[dict],
    run_number: int,
    open_market_ids: set[str] | None = None,
) -> None:
    """
    Print a clean, well-aligned TRADE RECOMMENDATIONS table to the terminal.
    Always calculates and shows Kelly stakes and highlights the top 2 surest picks.
    """
    if open_market_ids is None:
        open_market_ids = set()

    COL_RANK =  4
    COL_MKT  = 30
    COL_SIDE =  5
    COL_OURS =  6
    COL_MKTP =  6
    COL_EDGE =  6
    COL_CONF =  5
    COL_STKE =  7
    COL_NOTE = 18

    header = (
        f"  {'#':<{COL_RANK}}"
        f"{'Market':<{COL_MKT}}"
        f"{'Side':<{COL_SIDE}}"
        f"{'Ours':>{COL_OURS}}"
        f"{'Mkt%':>{COL_MKTP}}"
        f"{'Edge':>{COL_EDGE}}"
        f"{'Conf':>{COL_CONF}}"
        f"{'Stake':>{COL_STKE}}"
        f"  {'Status':<{COL_NOTE}}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print()
    print(_divider("="))
    if not candidates:
        print("  NO EDGE CANDIDATES FOUND THIS CYCLE")
        print(f"  Bankroll: ${bankroll:.2f} pUSD  |  Open in DB: {total_open}/{MAX_TRADES_PER_RUN}")
        print(_divider("="))
        return

    # Sort all candidates by conviction score descending
    candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)

    # City-level deduplication for top picks priority
    top_candidates = []
    seen_cities = set()
    for c in candidates_sorted:
        if c.city_name not in seen_cities:
            top_candidates.append(c)
            seen_cities.add(c.city_name)
    for c in candidates_sorted:
        if c not in top_candidates:
            top_candidates.append(c)

    print(f"  TRADE RECOMMENDATIONS  —  Run #{run_number}")
    print(_divider("-"))
    print(header)
    print(sep)

    executed_map = {r["market_id"]: r["stake"] for r in executed}

    # Display top 15 ranked candidates
    for rank, c in enumerate(top_candidates[:15], 1):
        # Calculate stake (Kelly) always so operator sees sizing even if slots are full
        calc_stake = get_kelly_stake(
            bankroll,
            edge=c.edge,
            mkt_price=c.mkt_price,
            confidence=c.confidence,
        )
        if c.market_id in executed_map:
            calc_stake = executed_map[c.market_id]
            note = "[✓] RECORDED IN DB"
        elif c.market_id in open_market_ids:
            note = "[OPEN IN DB]"
        elif rank == 1:
            note = "⭐ TOP PICK #1"
        elif rank == 2:
            note = "⭐ TOP PICK #2"
        else:
            note = f"  (ranked #{rank})"

        # Construct specific market label with City and Bracket
        raw_mkt = f"{c.city_name} — {c.group_title}"
        mkt_label = raw_mkt[:COL_MKT-1] if len(raw_mkt) >= COL_MKT else raw_mkt
        our_pct = f"{c.our_prob * 100:.1f}%"
        mkt_pct = f"{c.mkt_price * 100:.1f}%"
        edge_str = f"{c.edge * 100:.1f}%"
        conf_str = f"{c.confidence:.2f}"
        stake_str = f"${calc_stake:.2f}" if calc_stake > 0 else "—"

        row = (
            f"  {rank:<{COL_RANK}}"
            f"{mkt_label:<{COL_MKT}}"
            f"{c.side:<{COL_SIDE}}"
            f"{our_pct:>{COL_OURS}}"
            f"{mkt_pct:>{COL_MKTP}}"
            f"{edge_str:>{COL_EDGE}}"
            f"{conf_str:>{COL_CONF}}"
            f"{stake_str:>{COL_STKE}}"
            f"  {note:<{COL_NOTE}}"
        )
        print(row)

    print(sep)
    print(
        f"  Bankroll: ${bankroll:.2f} pUSD  |  "
        f"Open in DB: {total_open}/{MAX_TRADES_PER_RUN}  |  "
        f"Total edges found: {len(candidates)}"
    )
    print(_divider("-"))

    # Highlight Top 2 Actionable Picks for manual betting
    top_2 = top_candidates[:2]
    if top_2:
        print()
        print("  " + "═" * 74)
        print("  ⭐ TOP 2 ACTIONABLE PICKS (RECOMMENDED FOR MANUAL PLACEMENT)")
        print("  " + "═" * 74)
        for i, pick in enumerate(top_2, 1):
            p_stake = get_kelly_stake(bankroll, edge=pick.edge, mkt_price=pick.mkt_price, confidence=pick.confidence)
            if pick.market_id in executed_map:
                p_stake = executed_map[pick.market_id]
                status_txt = "RECORDED IN DB"
            elif pick.market_id in open_market_ids:
                status_txt = "ALREADY OPEN IN DB"
            else:
                status_txt = "READY TO PLACE MANUALLY"

            print(f"  [PICK #{i}] {pick.city_name} — {pick.group_title}  (Side: {pick.side})")
            print(f"            Event:       {pick.event_title}")
            print(f"            Action:      Buy {pick.side} @ {pick.mkt_price * 100:.1f}¢  |  Our Model: {pick.our_prob * 100:.1f}%  |  Edge: +{pick.edge * 100:.1f}%")
            print(f"            Rec. Stake:  ${p_stake:.2f} pUSD  (Confidence: {pick.confidence:.2f})  |  Status: {status_txt}")
            if i < len(top_2):
                print("  " + "-" * 74)
        print("  " + "═" * 74)
    print()


def _print_resolution_result(market_id: str, status: str, pnl: float) -> None:
    """Print a single trade resolution result."""
    icon = "[WON]" if status == "WON" else "[LOST]"
    sign = "+" if pnl >= 0 else ""
    print(f"  {icon}  market={market_id[:12]}...  PnL: {sign}${pnl:.2f} pUSD")


# ── Edge Candidate ─────────────────────────────────────────────────────────────


class EdgeCandidate:
    """Represents a potential trade that passed edge detection."""

    def __init__(
        self,
        event_title: str,
        city_name: str,
        group_title: str,
        question: str,
        condition_id: str,
        market_id: str,
        side: str,
        edge: float,
        our_prob: float,
        mkt_price: float,
        confidence: float,
        token_id: str,
        forecast_om_mean: float | None,
        forecast_wa_mean: float | None,
        forecast_mn_mean: float | None,
        forecast_pn_mean: float | None,
        forecast_ec_mean: float | None,
        forecast_gfs_mean: float | None,
        forecast_ukmet_mean: float | None,
        historical_count: int,
        sub_type: str,
        original_unit: str,
    ):
        self.event_title = event_title
        self.city_name = city_name
        self.group_title = group_title
        self.question = question
        self.condition_id = condition_id
        self.market_id = market_id
        self.side = side
        self.edge = edge
        self.our_prob = our_prob
        self.mkt_price = mkt_price
        self.confidence = confidence
        self.token_id = token_id
        self.forecast_om_mean = forecast_om_mean
        self.forecast_wa_mean = forecast_wa_mean
        self.forecast_mn_mean = forecast_mn_mean
        self.forecast_pn_mean = forecast_pn_mean
        self.forecast_ec_mean = forecast_ec_mean
        self.forecast_gfs_mean = forecast_gfs_mean
        self.forecast_ukmet_mean = forecast_ukmet_mean
        self.historical_count = historical_count
        self.sub_type = sub_type
        self.original_unit = original_unit

    @property
    def score(self) -> float:
        """
        Kelly-normalised conviction score.
        kelly_fraction = edge / (1 - mkt_price)  [normalised Kelly numerator]
        prob_modifier   = our_prob^0.5            [square-root — rewards high
                                                   probability without over-
                                                   penalising very confident bets]
        score = kelly_fraction * confidence * prob_modifier
        """
        kelly_fraction = self.edge / max(0.05, 1.0 - self.mkt_price)
        prob_modifier  = self.our_prob ** 0.5
        return kelly_fraction * self.confidence * prob_modifier

    def __repr__(self) -> str:
        return (
            f"Edge({self.city_name}/{self.group_title} "
            f"{self.side} edge={self.edge:.1%} conf={self.confidence:.2f})"
        )


# ── Input Validation Layer ────────────────────────────────────────────────────

_TEMP_ABS_MIN: float = -60.0   # All-time world record coldest: -89.2 °C (buffered)
_TEMP_ABS_MAX: float =  65.0   # All-time world record warmest:  56.7 °C (buffered)
_MAX_DAY_JUMP: float =  30.0   # Max plausible 24-h temperature change (°C)


def _sanitize_series(series: list[float]) -> list[float]:
    """
    Clean the QuantEnsemble's 60-day sequential input series:
    1. Clip values to absolute physical bounds [-60, 65] °C.
       Eliminates -999 / 9999 sentinel fill-values some archive APIs emit.
    2. Cap implausible day-to-day jumps (> 30 °C in 24 h).
       Even the most extreme cold-front passage on record is ~20 °C / day.
    """
    if not series:
        return series
    cleaned = [max(_TEMP_ABS_MIN, min(_TEMP_ABS_MAX, v)) for v in series]
    for i in range(1, len(cleaned)):
        delta = cleaned[i] - cleaned[i - 1]
        if abs(delta) > _MAX_DAY_JUMP:
            logger.warning(
                "Series sanitize: index %d jump %.1f°C (%.1f->%.1f) capped at ±%.0f°C",
                i, delta, cleaned[i - 1], cleaned[i], _MAX_DAY_JUMP,
            )
            cleaned[i] = cleaned[i - 1] + _MAX_DAY_JUMP * (1.0 if delta > 0 else -1.0)
    return cleaned


def _validate_forecasts(
    forecasts: dict[str, dict | None],
    historical_temps: list[float],
) -> dict[str, dict | None]:
    """
    Validate all forecast dicts before they enter the IVW combiner.

    Checks applied per source:
      1. `mean` is a finite float within absolute physical bounds [-60, 65] °C.
      2. `mean` is within the city's own 4σ historical envelope.
         (An ECMWF value of 55 °C for London is physically impossible.)
      3. `std` is a positive finite float <= 15 °C.
         Invalid std is repaired to 3.0 °C rather than dropping the forecast.
      4. Cross-source MAD z-score outlier detection (threshold z > 2.5).
         Outlier sources are NOT dropped — they may be the only correct one
         during a genuine extreme event.  Instead their `std` is doubled,
         which halves their IVW precision weight automatically.

    Returns the cleaned dict (corrupted forecasts replaced with None).
    """
    if historical_temps and len(historical_temps) > 3:
        hist_mean = float(np.mean(historical_temps))
        hist_std  = float(np.std(historical_temps, ddof=1))
    else:
        hist_mean, hist_std = 20.0, 15.0   # wide fallback — no information to bound with

    hist_lo = hist_mean - 4.0 * hist_std
    hist_hi = hist_mean + 4.0 * hist_std

    validated: dict[str, dict | None] = {}
    for key, fc in forecasts.items():
        if fc is None:
            validated[key] = None
            continue

        mu    = fc.get("mean")
        sigma = fc.get("std")

        # Check 1: mean must be finite
        if mu is None or not math.isfinite(mu):
            logger.warning("Forecast validation: %s mean=%s is not finite — dropping", key, mu)
            validated[key] = None
            continue

        # Check 2: absolute physical bounds
        if not (_TEMP_ABS_MIN <= mu <= _TEMP_ABS_MAX):
            logger.warning(
                "Forecast validation: %s mean=%.1f°C outside absolute bounds [%.0f, %.0f] — dropping",
                key, mu, _TEMP_ABS_MIN, _TEMP_ABS_MAX,
            )
            validated[key] = None
            continue

        # Check 3: historical plausibility (4σ envelope)
        if not (hist_lo <= mu <= hist_hi):
            logger.warning(
                "Forecast validation: %s mean=%.1f°C outside 4σ historical bounds "
                "[%.1f, %.1f] (hist_mean=%.1f hist_std=%.1f) — dropping",
                key, mu, hist_lo, hist_hi, hist_mean, hist_std,
            )
            validated[key] = None
            continue

        # Check 4: std sanity — repair rather than drop
        if sigma is None or not math.isfinite(sigma) or sigma <= 0.0 or sigma > 15.0:
            logger.warning(
                "Forecast validation: %s std=%s is invalid — replacing with 3.0°C fallback",
                key, sigma,
            )
            fc = {**fc, "std": 3.0}

        validated[key] = fc

    # Cross-source outlier detection via Median Absolute Deviation
    available = {k: v for k, v in validated.items() if v is not None}
    if len(available) >= 3:
        mu_vals = [f["mean"] for f in available.values()]
        median  = float(np.median(mu_vals))
        mad     = float(np.median([abs(m - median) for m in mu_vals]))
        mad_sc  = max(0.10, mad * 1.4826)   # MAD -> sigma equivalent (Gaussian)
        for key, fc in list(available.items()):
            z = abs(fc["mean"] - median) / mad_sc
            if z > 2.5:
                logger.warning(
                    "Forecast validation: %s is a cross-source outlier "
                    "(z=%.1f, mean=%.1f°C vs median=%.1f°C) — doubling sigma to halve IVW weight",
                    key, z, fc["mean"], median,
                )
                validated[key] = {**fc, "std": fc["std"] * 2.0}

    return validated


# ── Scan Event ─────────────────────────────────────────────────────────────────


async def scan_event_for_edges(
    event: dict[str, Any],
    tomorrow: date,
) -> list[EdgeCandidate]:
    """
    Scan a temperature event for trading edges WITHOUT executing trades.

    Steps:
      1. Verify event date is tomorrow
      2. Look up city coordinates
      3. Fetch historical data + all 3 forecasts concurrently
      4. Per sub-market: parse → ensemble model → detect edge
      5. Return list of EdgeCandidates
    """
    title = event.get("title", "Unknown Event")

    # Extract city from title (e.g. "Highest temperature in Seoul on June 8?")
    city_match = re.search(r"in\s+(.*?)\s+(?:on|for)\b", title)
    city_name = city_match.group(1).strip() if city_match else "Unknown"

    # Extract date from title (e.g. "on June 8")
    m = re.search(r"on\s+([A-Z][a-z]+\s+\d+)", title)
    date_str = m.group(1) if m else ""

    if "highest" in title.lower():
        market_type = "highest"
    elif "lowest" in title.lower():
        market_type = "lowest"
    else:
        logger.warning("SKIP EVENT: %s — cannot determine market type (max/min)", title)
        return []
    sub_markets = event.get("markets", [])
    candidates: list[EdgeCandidate] = []

    try:
        # Step 1: Look up city coordinates to determine timezone first
        # Fix 3: Multiple regex fallbacks for different Polymarket title formats
        if city_name == "Unknown" or not city_name:
            # Fallback A: "Seoul: Highest temperature on June 8"
            m2 = re.match(r"^([^:]+):", title)
            if m2:
                city_name = m2.group(1).strip()
            else:
                # Fallback B: first capitalised word sequence before a date-like word
                m3 = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:on|for|–)", title)
                if m3:
                    city_name = m3.group(1).strip()
            if city_name and city_name != "Unknown":
                logger.info("  City regex fallback resolved: '%s'", city_name)
            else:
                logger.warning("SKIP EVENT: %s — could not extract city name", title)
                return []

        # CITY_ALLOWLIST filter — skip cities not in the high-forecastability set.
        # This is checked BEFORE geocoding to save API quota on non-allowlisted events.
        if CITY_ALLOWLIST is not None and city_name not in CITY_ALLOWLIST:
            logger.debug(
                "SKIP EVENT: %s — '%s' not in CITY_ALLOWLIST", title, city_name
            )
            return []

        city = await get_city_coordinates(city_name)
        if city is None:
            logger.warning("SKIP EVENT: %s — unknown city '%s'", title, city_name)
            return []

        lat, lon = city["lat"], city["lon"]
        tz_name = city.get("timezone", "UTC")
        
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            tz = zoneinfo.ZoneInfo("UTC")
        local_tomorrow = (datetime.now(tz) + timedelta(days=1)).date()

        # Step 2: Verify event is for TOMORROW (local timezone)
        event_date = parse_event_date(date_str)
        if event_date is None:
            logger.warning("SKIP EVENT: %s — could not parse date '%s'", title, date_str)
            return []

        if event_date != local_tomorrow:
            logger.info(
                "SKIP EVENT: %s — date %s is not tomorrow in %s (%s)",
                title, event_date, city_name, local_tomorrow,
            )
            return []

        temp_type = "max" if market_type == "highest" else "min"
        date_fmt = event_date.strftime("%Y-%m-%d")

        logger.info(
            "SCANNING: %s | City: %s (%.2f, %.2f) | Date: %s | Subs: %d",
            title, city_name, lat, lon, event_date, len(sub_markets),
        )

        # Station coordinate override: use exact ICAO station lat/lon instead of
        # city-centre geocoding. This eliminates the urban heat-island bias
        # (0.5-2°C) that corrupts forecasts when the resolution station is an
        # airport on the outskirts (e.g. EGLC for London, JFK for NYC).
        if city_name in STATION_COORDS:
            s_lat, s_lon, s_icao, s_desc = STATION_COORDS[city_name]
            if abs(s_lat - lat) > 0.05 or abs(s_lon - lon) > 0.05:  # only log if meaningful override
                logger.info(
                    "  Station override: %s → %s (%s) (%.4f,%.4f) → (%.4f,%.4f)",
                    city_name, s_icao, s_desc, lat, lon, s_lat, s_lon,
                )
            lat, lon = s_lat, s_lon

        # Step 3+4: Fetch historical + all forecasts + ensemble uncertainty CONCURRENTLY.
        # Open-Meteo NWP models (ECMWF, ICON, GFS, UKMET) are batched into ONE HTTP request
        # via get_openmeteo_all_models(), cutting Open-Meteo API calls from 4 to 1.
        # get_recent_daily_series() fetches the last 60 days for the QuantEnsemble and
        # runs concurrently — no added serial latency.
        (
            historical_data,
            om_batch,
            forecast_wa,
            forecast_mn,
            ensemble_std,
            forecast_pn,
            recent_series,
        ) = await asyncio.gather(
            get_historical_temps(lat, lon, event_date, target_type=temp_type),
            get_openmeteo_all_models(lat, lon, date_fmt, target_type=temp_type),
            get_weatherapi_forecast(lat, lon, date_fmt, target_type=temp_type),
            get_metno_forecast(lat, lon, date_fmt, tz_name, target_type=temp_type),
            get_openmeteo_ensemble_std(lat, lon, date_fmt, target_type=temp_type),
            get_polynimbus_forecast(city_name, date_fmt, target_type=temp_type),
            get_recent_daily_series(lat, lon, event_date, days_back=60, target_type=temp_type),
        )
        # Unpack the batched Open-Meteo tuple
        forecast_ec, forecast_om, forecast_gfs, forecast_ukmet = om_batch

        # Sanitize the recent series BEFORE passing to QuantEnsemble:
        # clips to absolute physical bounds and caps implausible 24-h jumps.
        # This prevents -999 fill-values or API glitches from corrupting
        # the Kalman/ETS/OU models which are sensitive to initial conditions.
        recent_series = _sanitize_series(recent_series)

        # Quantitative Prediction Group — pure local computation (no API call here).
        # Runs OU + Holt-Winters ETS + Kalman DLM on the 60-day sequential series
        # fetched above, then fuses via min-variance portfolio weighting.
        # Regime (STABLE/CHAOTIC) is detected from the NWP ensemble spread.
        forecast_local = QuantEnsemble().predict(
            series=recent_series,
            target_date=event_date,
            ensemble_spread=ensemble_std,
        )

        # Extract plain list[float] for checks and logging
        historical_temps = [t for _, t in historical_data] if historical_data else []

        # Sanitize historical data — strip physically impossible archive values
        # before they corrupt the prior. Open-Meteo archive occasionally emits
        # sentinel values (-999, 9999) for missing station records.
        historical_data  = [
            (y, t) for y, t in historical_data
            if _TEMP_ABS_MIN <= t <= _TEMP_ABS_MAX and math.isfinite(t)
        ] if historical_data else []
        historical_temps = [t for _, t in historical_data]

        if not historical_temps:
            logger.warning("SKIP EVENT: %s — no historical data", title)
            return []

        # Minimum sample size guard
        unique_years = {y for y, _ in historical_data} if historical_data else set()
        if len(unique_years) < 5:
            logger.warning(
                "SKIP EVENT: %s — insufficient historical data (%d years, need >=5)",
                title, len(unique_years),
            )
            return []

        # Historical data freshness check — stale archive data distorts the prior.
        most_recent_year = max(int(y) for y, _ in historical_data)
        current_year     = datetime.now(timezone.utc).year
        if most_recent_year < current_year - 2:
            logger.warning(
                "  VALIDATION: Historical data for %s is stale "
                "(most recent year %d vs current %d) — prior may be biased",
                city_name, most_recent_year, current_year,
            )

        logger.info(
            "  Historical: %d points | +-2-day window | 2000-present | "
            "mean=%.1f°C range=[%.1f, %.1f] | most_recent_year=%d",
            len(historical_temps),
            sum(historical_temps) / len(historical_temps),
            min(historical_temps),
            max(historical_temps),
            most_recent_year,
        )

        # Validate all 8 forecast sources against physical and historical bounds.
        # This is the gate that ensures no corrupted API response contaminates
        # the IVW combiner.  Outlier sources get doubled sigma (halved IVW weight)
        # rather than silent removal — a single outlier may be the correct one
        # during extreme events.
        _raw = {
            "ecmwf":          forecast_ec,
            "openmeteo":      forecast_om,
            "gfs":            forecast_gfs,
            "ukmet":          forecast_ukmet,
            "metno":          forecast_mn,
            "local_ensemble": forecast_local,
            "polynimbus":     forecast_pn,
            "weatherapi":     forecast_wa,
        }
        _validated = _validate_forecasts(_raw, historical_temps)
        forecast_ec     = _validated["ecmwf"]
        forecast_om     = _validated["openmeteo"]
        forecast_gfs    = _validated["gfs"]
        forecast_ukmet  = _validated["ukmet"]
        forecast_mn     = _validated["metno"]
        forecast_local  = _validated["local_ensemble"]
        forecast_pn     = _validated["polynimbus"]
        forecast_wa     = _validated["weatherapi"]


        # Replace Open-Meteo's diurnal/default std with the true ensemble spread.
        # Do NOT add the 1.2°C station floor here — calculate_likelihood() applies
        # it once in quadrature. Applying it twice caused a ~17% overestimate of
        # forecast uncertainty: sqrt(ens² + 2×1.44) instead of sqrt(ens² + 1.44).
        if ensemble_std is not None and forecast_om is not None:
            forecast_om = {**forecast_om, "std": ensemble_std}
            logger.info(
                "  Open-Meteo std -> ensemble spread: %.2f°C (station floor applied in likelihood)",
                ensemble_std,
            )

        if not any([forecast_om, forecast_wa, forecast_mn,
                    forecast_ec, forecast_gfs, forecast_ukmet, forecast_pn]):
            logger.warning("SKIP EVENT: %s — all forecast sources failed", title)
            return []

        if forecast_ec:
            logger.info("  ECMWF IFS:   %.1f°C ± %.2f°C", forecast_ec["mean"], forecast_ec["std"])
        else:
            logger.warning("  ECMWF IFS:   UNAVAILABLE")
        if forecast_gfs:
            logger.info("  GFS:         %.1f°C ± %.2f°C", forecast_gfs["mean"], forecast_gfs["std"])
        else:
            logger.warning("  GFS:         UNAVAILABLE")
        if forecast_ukmet:
            logger.info("  UKMET:       %.1f°C ± %.2f°C", forecast_ukmet["mean"], forecast_ukmet["std"])
        else:
            logger.warning("  UKMET:       UNAVAILABLE")
        if forecast_om:
            logger.info("  Open-Meteo:  %.1f°C ± %.2f°C", forecast_om["mean"], forecast_om["std"])
        else:
            logger.warning("  Open-Meteo:  FAILED")
        if forecast_wa:
            logger.info("  WeatherAPI:  %.1f°C ± %.2f°C", forecast_wa["mean"], forecast_wa["std"])
        else:
            logger.warning("  WeatherAPI:  UNAVAILABLE (check WEATHERAPI_KEY)")
        if forecast_mn:
            logger.info("  Met.no:      %.1f°C ± %.2f°C", forecast_mn["mean"], forecast_mn["std"])
        else:
            logger.warning("  Met.no:      UNAVAILABLE")
        if forecast_pn:
            logger.info("  PolyNimbus:  %.1f°C +/- %.2f°C", forecast_pn["mean"], forecast_pn["std"])
        elif city_name in POLYNIMBUS_CITIES:
            logger.warning("  PolyNimbus:  UNAVAILABLE")
        else:
            logger.info("  PolyNimbus:  N/A (city not tracked)")
        if forecast_local:
            logger.info(
                "  QuantGroup:  %.1f°C +/- %.2f°C  (OU+ETS+Kalman  regime=%s)",
                forecast_local["mean"], forecast_local["std"],
                "CHAOTIC" if ensemble_std is not None and ensemble_std >= 1.2 else "STABLE",
            )
        else:
            logger.warning("  QuantGroup:  SKIPPED (insufficient recent series or models failed)")

        # METAR soft bound: blend today's observed temperature into forecast means.
        # Evening METAR anchors tomorrow's conditions (after 7 PM local, the day's
        # high/low is largely decided). Blend at 20% for max markets, 15% for min.
        if forecast_pn and isinstance(forecast_pn.get("metar_temp_c"), float):
            metar_now = forecast_pn["metar_temp_c"]
            try:
                tz_obj = zoneinfo.ZoneInfo(tz_name)
                local_hour = datetime.now(tz_obj).hour
            except Exception:
                local_hour = datetime.now(timezone.utc).hour

            # Use a fixed climatological diurnal range, NOT std(historical daily maxima).
            # std(daily maxima over 25 years) measures interannual variability (~2-5°C),
            # NOT the within-day temperature swing (~8-15°C). The old formula
            # max(4.0, hist_std * 1.5) was conceptually wrong and produced anchors
            # that were too close to the current METAR reading.
            if temp_type == "max":
                metar_anchor = metar_now + 11.0 * 0.5   # typical diurnal range for max markets
                blend = 0.20 if local_hour >= 19 else 0.10
            else:
                metar_anchor = metar_now - 9.0 * 0.3    # typical diurnal range for min markets
                blend = 0.20 if local_hour >= 20 else 0.08

            def _blend(f, anchor, w):
                return {**f, "mean": (1.0 - w) * f["mean"] + w * anchor} if f else None

            forecast_om = _blend(forecast_om, metar_anchor, blend)
            forecast_wa = _blend(forecast_wa, metar_anchor, blend)
            forecast_mn = _blend(forecast_mn, metar_anchor, blend)
            forecast_pn = _blend(forecast_pn, metar_anchor, blend)
            forecast_ec = _blend(forecast_ec, metar_anchor, blend)
            forecast_gfs    = _blend(forecast_gfs,    metar_anchor, blend)
            forecast_ukmet  = _blend(forecast_ukmet,  metar_anchor, blend)
            logger.info(
                "  METAR anchor: %.1f°C → anchor=%.1f°C  blend=%.0f%%  (local hour %02d)",
                metar_now, metar_anchor, blend * 100, local_hour,
            )

        # Check forecast agreement — pass hist_std for dynamic threshold
        # ddof=1: sample std (unbiased) — consistent with calculate_likelihood and
        # the rest of the codebase. Population std (ddof=0) underestimates variability
        # for small samples and was used here inconsistently.
        hist_std_val = float(np.std(historical_temps, ddof=1)) if len(historical_temps) > 3 else None
        agreement_ok, disagreement = check_forecast_agreement(
            forecast_om, forecast_wa, forecast_mn, forecast_pn,
            MAX_FORECAST_DISAGREEMENT,
            hist_std=hist_std_val,
            forecast_ec=forecast_ec,
            forecast_gfs=forecast_gfs,
            forecast_ukmet=forecast_ukmet,
            forecast_local=forecast_local,
        )
        if not agreement_ok:
            logger.warning(
                "SKIP EVENT: %s — forecast disagreement %.1f°C (threshold: dynamic)",
                title, disagreement,
            )
            return []

        # Two-pass normalisation — bins are mutually exclusive and exhaustive
        # (exactly one bin resolves YES). Independent Bayesian runs can sum to >1,
        # creating phantom edges. Pass 1 collects raw posteriors; Pass 2 normalises.

        bin_results = []  # (sub, group_title, cond_id, mkt_id, low_b, high_b,
                          #  sub_type, orig_unit, raw_yes, confidence, yes_p, no_p, tok_id)

        for sub in sub_markets:
            if sub.get("enableOrderBook") is False or sub.get("acceptingOrders") is False:
                continue
            if sub.get("closed") is True or sub.get("active") is False:
                continue

            group_title = sub.get("groupItemTitle") or sub.get("group_item_title", "")
            if not group_title:
                group_title = sub.get("question") or sub.get("title", "")

            condition_id = sub.get("conditionId") or sub.get("condition_id", "")
            market_id = sub.get("id", "")

            parsed = parse_group_item_title(group_title)
            if parsed is None:
                continue

            low_bound, high_bound, sub_type, original_unit = parsed


            # Extract market prices
            yes_price = None
            no_price = None
            try:
                outcomes = json.loads(sub.get("outcomes") or "[]")
                prices_str = sub.get("outcomePrices")
                prices = json.loads(prices_str) if prices_str else []
                if "Yes" in outcomes and "No" in outcomes:
                    yes_idx = outcomes.index("Yes")
                    no_idx = outcomes.index("No")
                    if prices and len(prices) > max(yes_idx, no_idx):
                        yes_price = float(prices[yes_idx])
                        no_price = float(prices[no_idx])
                    else:
                        clob_url = f"https://clob.polymarket.com/markets/{condition_id}"
                        async with httpx.AsyncClient() as client:
                            # Rate-limiting delay placed BEFORE the request, not after
                            # receiving the response (the response should be parsed immediately).
                            await asyncio.sleep(API_REQUEST_DELAY)
                            resp = await _get_with_retry(client, clob_url)
                            if resp is not None and resp.status_code == 200:
                                clob_data = resp.json()
                                for token in clob_data.get("tokens", []):
                                    if token.get("outcome") == "Yes":
                                        yes_price = float(token.get("price", 0.0))
                                    elif token.get("outcome") == "No":
                                        no_price = float(token.get("price", 0.0))
            except Exception as e:
                logger.error("Error extracting prices for %s: %s", market_id, e, exc_info=True)

            if yes_price is None or no_price is None or (yes_price == 0.0 and no_price == 0.0):
                continue

            # Resolve token_id (YES side)
            token_id = None
            if sub.get("clobTokenIds"):
                try:
                    c_tokens = json.loads(sub.get("clobTokenIds") or "[]")
                    if c_tokens:
                        token_id = c_tokens[0]
                except Exception:
                    pass
            if not token_id:
                for t in sub.get("tokens", []):
                    if t.get("outcome", "").upper() == "YES":
                        token_id = t.get("token_id")
                        break
            if not token_id:
                token_id = market_id

            # T2-C: Stale/uninitialised price guard.
            # If yes + no deviates significantly from 1.0, the market has no real
            # bids (stale default prices) — edge calculations against them are meaningless.
            price_sum = yes_price + no_price
            if not (0.85 <= price_sum <= 1.15):
                logger.debug(
                    "  SKIP bin '%s': stale prices (YES=%.3f + NO=%.3f = %.3f)",
                    group_title, yes_price, no_price, price_sum,
                )
                continue

            # Run Bayesian model — raw (un-normalised) posterior for this bin
            raw_yes, raw_no, confidence = run_forecast_model(
                historical_data=historical_data,
                forecast_om=forecast_om,
                forecast_wa=forecast_wa,
                forecast_mn=forecast_mn,
                forecast_pn=forecast_pn,
                low_bound=low_bound,
                high_bound=high_bound,
                ensemble_std=ensemble_std,
                forecast_ec=forecast_ec,
                forecast_gfs=forecast_gfs,
                forecast_ukmet=forecast_ukmet,
                forecast_local=forecast_local,
            )

            bin_results.append((
                sub, group_title, condition_id, market_id,
                low_bound, high_bound, sub_type, original_unit,
                raw_yes, confidence, yes_price, no_price, token_id,
            ))

        # Pass 2: normalise bin probabilities so they sum to 1, then detect edges.
        total_raw_yes = sum(r[8] for r in bin_results)
        norm_factor = (1.0 / total_raw_yes) if total_raw_yes > 1e-9 else 1.0
        if bin_results and abs(total_raw_yes - 1.0) > 0.05:
            logger.info(
                "  Bin normalisation: raw total=%.3f → normalised (factor=%.3f)",
                total_raw_yes, norm_factor,
            )

        # T1-C: Market coherence filter.
        # If all YES prices sum to > 1.0, the market-maker's overround is eating
        # our edge. High overround = distorted prices = unreliable edge signal.
        if bin_results:
            total_market_yes = sum(r[10] for r in bin_results)  # yes_price at index 10
            overround = total_market_yes - 1.0
            if overround > 0.30:
                logger.warning(
                    "SKIP EVENT: %s — market overround %.1f%% (>30%%) — prices too distorted",
                    title, overround * 100,
                )
                return candidates
            elif overround > 0.15:
                edge_penalty = overround / 2.0
                logger.warning(
                    "  ⚠ High overround %.1f%% — edge estimates penalised by %.1f%%",
                    overround * 100, edge_penalty * 100,
                )
            else:
                edge_penalty = 0.0
        else:
            edge_penalty = 0.0

        for (
            sub, group_title, condition_id, market_id,
            low_bound, high_bound, sub_type, original_unit,
            raw_yes, confidence, yes_price, no_price, token_id,
        ) in bin_results:
            # Cooldown check AFTER normalisation so all bins contribute to the
            # sum-to-1 constraint even when we already hold a position in one of them.
            # Moving this earlier (as was the case) caused the remaining bins to be
            # inflated, creating phantom edges.
            if not await check_trade_cooldown(market_id):
                logger.info("  COOLDOWN: market %s already OPEN in DB", market_id[:12])
                continue

            final_yes = max(0.01, min(0.99, raw_yes * norm_factor))
            final_no  = max(0.01, min(0.99, 1.0 - final_yes))

            # T3-B: Per-type edge threshold
            type_threshold = THRESHOLD_BY_TYPE.get(sub_type, MIN_EDGE_THRESHOLD)

            side, edge = detect_edge(
                final_yes, final_no, yes_price, no_price, type_threshold
            )

            # T1-C: Apply overround edge penalty — if market is high-overround,
            # reduce detected edge by the penalty before accepting it.
            if side is not None and edge_penalty > 0:
                edge = max(0.0, edge - edge_penalty)
                if edge < type_threshold:
                    logger.debug(
                        "  Edge cancelled after overround penalty: %s net edge=%.1f%%",
                        group_title, edge * 100,
                    )
                    side = None

            if side is not None:
                our_prob = final_yes if side == "YES" else final_no
                mkt_price = yes_price if side == "YES" else no_price

                candidate = EdgeCandidate(
                    event_title=title,
                    city_name=city_name,
                    group_title=group_title,
                    question=sub.get("question", group_title),
                    condition_id=condition_id,
                    market_id=market_id,
                    side=side,
                    edge=edge,
                    our_prob=our_prob,
                    mkt_price=mkt_price,
                    confidence=confidence,
                    token_id=token_id,
                    forecast_om_mean=forecast_om["mean"] if forecast_om else None,
                    forecast_wa_mean=forecast_wa["mean"] if forecast_wa else None,
                    forecast_mn_mean=forecast_mn["mean"] if forecast_mn else None,
                    forecast_pn_mean=forecast_pn["mean"] if forecast_pn else None,
                    forecast_ec_mean=forecast_ec["mean"] if forecast_ec else None,
                    forecast_gfs_mean=forecast_gfs["mean"] if forecast_gfs else None,
                    forecast_ukmet_mean=forecast_ukmet["mean"] if forecast_ukmet else None,
                    historical_count=len(historical_temps),
                    sub_type=sub_type,
                    original_unit=original_unit,
                )
                candidates.append(candidate)
                logger.info(
                    "  [EDGE] %s %s | edge=%+.1f%% | ours=%.1f%% mkt=%.1f%% | conf=%.2f",
                    side, group_title, edge * 100, our_prob * 100, mkt_price * 100, confidence,
                )
            else:
                best_yes_edge = final_yes - yes_price
                best_no_edge  = final_no  - no_price
                best_side = "YES" if best_yes_edge > best_no_edge else "NO"
                best_edge = max(best_yes_edge, best_no_edge)
                logger.info(
                    "  [PASS] %s | %s best edge=%+.1f%% (needs %.1f%%) | "
                    "ours=%.1f%% mkt=%.1f%%",
                    group_title, best_side, best_edge * 100, MIN_EDGE_THRESHOLD * 100,
                    (final_yes if best_side == "YES" else final_no) * 100,
                    (yes_price if best_side == "YES" else no_price) * 100,
                )

    except Exception as e:
        logger.error("Error scanning event '%s': %s", title, e, exc_info=True)
        raise

    return candidates


# ── Execute Top Trades ─────────────────────────────────────────────────────────


async def execute_top_trades(
    candidates: list[EdgeCandidate],
    max_trades: int,
    bankroll: float,
) -> list[dict]:
    """
    Sort edge candidates by score and record the top N in trades.db.
    Returns list of result dicts (one per successfully recorded recommendation).
    """
    results = []

    if not candidates:
        return results

    max_per_city = 3 if bankroll >= 500 else 1

    # Sort by score descending
    candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Apply per-city/event limit
    filtered: list[EdgeCandidate] = []
    city_counts: dict[str, int] = {}
    for c in candidates_sorted:
        # Key by city_name (not event_title) to enforce the per-city limit correctly.
        count = city_counts.get(c.city_name, 0)
        if count < max_per_city:
            filtered.append(c)
            city_counts[c.city_name] = count + 1

    # Execute top N
    for i, candidate in enumerate(filtered[:max_trades], 1):
        if bankroll <= 0:
            logger.warning("Bankroll exhausted — stopping execution.")
            break
        stake = get_kelly_stake(
                bankroll,
                edge=candidate.edge,
                mkt_price=candidate.mkt_price,
                confidence=candidate.confidence,
            )
        if stake <= 0:
            logger.warning(
                "Stake is zero for candidate %d (%s) — insufficient edge or bankroll, skipping.",
                i, candidate.group_title,
            )
            continue

        trade_result = await execute_trade(candidate, stake)
        if not trade_result:
            continue
            
        bankroll -= stake

        results.append({
            "market_id": candidate.market_id,
            "question": candidate.question,
            "side": candidate.side,
            "edge": candidate.edge,
            "stake": stake,
            "status": trade_result.get("status", "error"),
            "rank": i,
        })

    return results


# ── Main Trading Loop ──────────────────────────────────────────────────────────

_LOOP_LOCK: asyncio.Lock | None = None
_RESOLUTION_LOCK: asyncio.Lock | None = None
_run_counter = 0


async def run_trading_loop():
    global _LOOP_LOCK
    if _LOOP_LOCK is None:
        _LOOP_LOCK = asyncio.Lock()
    if _LOOP_LOCK.locked():
        logger.warning("Previous trading loop still running. Skipping this cycle.")
        return
    async with _LOOP_LOCK:
        await _run_trading_loop_inner()


async def _run_trading_loop_inner():
    """
    Main trading loop — discover tomorrow's events, scan for edges,
    record top candidates, print recommendations table.
    Called once per scheduled run (every 5 minutes).
    """
    global _run_counter
    _run_counter += 1

    tomorrow = get_tomorrow()
    _print_header(_run_counter, tomorrow)

    all_candidates: list[EdgeCandidate] = []
    events_scanned = 0
    sub_markets_scanned = 0
    errors = 0
    trade_results: list[dict] = []

    try:
        # Step 1: Discover active temperature events
        events = await get_active_temperature_events()

        if not events:
            logger.info("No active temperature events found")
        else:
            logger.info(
                "Found %d temperature events total — filtering for tomorrow (%s)",
                len(events), tomorrow,
            )

        # Step 2: Scan each event for edges
        for i, event in enumerate(events, 1):
            logger.info("")
            logger.info("  — Analyzing event %d/%d: %s", i, len(events), event.get("title", "?"))
            result = []  # ensure 'result' is always bound even if scan_event_for_edges raises
            try:
                result = await scan_event_for_edges(event, tomorrow)
                all_candidates.extend(result)
            except Exception as e:
                errors += 1
                logger.error("Error scanning event %d: %s", i, e)
            sub_markets_scanned += len(event.get("markets", []))
            events_scanned += 1

            # Rate limiting: only sleep after events that triggered real API calls.
            # Skipped events (wrong date / unknown city) return [] immediately
            # with no network requests — sleeping here wastes ~2s × 100+ skipped events per cycle.
            if result:
                await asyncio.sleep(API_REQUEST_DELAY)

        logger.info(
            "Scan complete: %d events | %d sub-markets | %d edges found | %d errors",
            events_scanned, sub_markets_scanned, len(all_candidates), errors,
        )

        # Step 3: Record top trades up to available slots
        open_trades = await get_open_trades()
        open_market_ids = {t["market_id"] for t in open_trades}
        total_open = len(open_trades)
        available_slots = max(0, MAX_TRADES_PER_RUN - total_open)
        bankroll = await get_bankroll()

        trade_results = []
        if available_slots > 0 and all_candidates:
            trade_results = await execute_top_trades(all_candidates, available_slots, bankroll)
            total_open += len(trade_results)
        elif available_slots == 0:
            logger.info(
                "Trade slot limit reached (%d open positions in DB, max %d).",
                total_open, MAX_TRADES_PER_RUN,
            )

        # Step 4: Print the recommendations table
        bankroll_final = await get_bankroll()
        _print_recommendations(
            candidates=all_candidates,
            total_open=total_open,
            bankroll=bankroll_final,
            executed=trade_results,
            run_number=_run_counter,
            open_market_ids=open_market_ids,
        )

    except Exception as e:
        logger.error("Trading loop crashed: %s", e, exc_info=True)

    logger.info(
        "Loop #%d complete — Events: %d | Subs: %d | Edges: %d | Recorded: %d",
        _run_counter, events_scanned, sub_markets_scanned,
        len(all_candidates), len(trade_results),
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────


def _task_done_callback(task: asyncio.Task):
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Background task crashed: %s", e, exc_info=True)


def schedule_job():
    """Wrapper to safely launch the async trading loop from sync scheduler."""
    logger.info("Triggering scheduled trading loop...")
    task = asyncio.create_task(run_trading_loop())
    task.add_done_callback(_task_done_callback)


async def run_resolution_check():
    """
    Check open trades against Polymarket resolution data and update PnL.
    Runs daily at 12:00 UTC. Guarded by _RESOLUTION_LOCK to prevent two
    concurrent calls from double-writing PnL for the same trade.
    """
    global _RESOLUTION_LOCK
    if _RESOLUTION_LOCK is None:
        _RESOLUTION_LOCK = asyncio.Lock()
    if _RESOLUTION_LOCK.locked():
        logger.warning("Resolution check already running — skipping this cycle.")
        return

    from database import get_open_trades, update_trade_resolution
    from data import get_market_resolution

    async with _RESOLUTION_LOCK:
        logger.info("Starting daily resolution check...")
        open_trades = await get_open_trades()
        if not open_trades:
            logger.info("No open trades to check.")
            return

        print()
        print(_divider("═"))
        print("  DAILY RESOLUTION CHECK")
        print(_divider("═"))

        resolved = 0
        for trade in open_trades:
            res = await get_market_resolution(trade["market_id"])
            if res and res["closed"]:
                if res.get("voided"):
                    try:
                        await update_trade_resolution(trade["market_id"], "VOID", 0.0)
                        print(f"  [!] VOID  market={trade['market_id'][:12]}...  (market voided)")
                        resolved += 1
                    except Exception as db_err:
                        logger.error("DB write failed marking VOID for %s: %s", trade["market_id"], db_err)
                    continue

                if res.get("winner"):
                    if trade["price"] <= 0:
                        # Guard against division-by-zero in the PnL calculation below.
                        # A price of 0 in the DB means the trade was logged with a bad price.
                        # Mark VOID rather than crashing the resolution loop.
                        logger.error(
                            "Trade %s has invalid price (%.4f). Marking VOID.",
                            trade["market_id"], trade["price"],
                        )
                        try:
                            await update_trade_resolution(trade["market_id"], "VOID", 0.0)
                            print(f"  [!] VOID  market={trade['market_id'][:12]}...  (invalid price in DB)")
                            resolved += 1
                        except Exception as db_err:
                            logger.error("DB write failed marking VOID for %s: %s", trade["market_id"], db_err)
                        continue

                    if trade["price"] >= 1.0:
                        # A price ≥ 1.0 is invalid for Polymarket and would produce
                        # zero or negative net profit. Mark VOID rather than recording
                        # nonsensical PnL.
                        logger.error(
                            "Trade %s has invalid price ≥ 1.0 (%.4f). Marking VOID.",
                            trade["market_id"], trade["price"],
                        )
                        try:
                            await update_trade_resolution(trade["market_id"], "VOID", 0.0)
                            print(f"  [!] VOID  market={trade['market_id'][:12]}...  (invalid price ≥ 1.0)")
                            resolved += 1
                        except Exception as db_err:
                            logger.error("DB write failed marking VOID for %s: %s", trade["market_id"], db_err)
                        continue

                    if trade["side"].upper() == res["winner"].upper():
                        status = "WON"
                        shares = trade["stake"] / trade["price"]
                        pnl = shares - trade["stake"]
                    else:
                        status = "LOST"
                        pnl = -trade["stake"]

                    logger.info("Trade %s resolved %s (PnL: $%.2f)", trade["market_id"][:12], status, pnl)
                    try:
                        await update_trade_resolution(trade["market_id"], status, pnl)
                        _print_resolution_result(trade["market_id"], status, pnl)
                        resolved += 1
                    except Exception as db_err:
                        logger.error(
                            "DB write failed updating resolution for trade %s: %s. "
                            "Trade remains OPEN in DB — fix manually.",
                            trade["market_id"], db_err,
                        )

        if resolved == 0:
            print("  No trades resolved yet.")
        print(_divider("-"))
        print()


def schedule_resolution_job():
    """Wrapper to launch async resolution check."""
    logger.info("Triggering scheduled resolution check...")
    task = asyncio.create_task(run_resolution_check())
    task.add_done_callback(_task_done_callback)


_last_resolution_date = None

def check_resolution_time():
    global _last_resolution_date
    now = datetime.now(timezone.utc)
    if now.hour == 12 and now.date() != _last_resolution_date:
        _last_resolution_date = now.date()
        schedule_resolution_job()

async def run_scheduler():
    """
    Run the scheduler loop.
      Weather trading:  every 5 minutes
      Resolution check: daily at 12:00 UTC
    """
    schedule.every(5).minutes.do(schedule_job)
    schedule.every(5).minutes.do(check_resolution_time)

    print(_divider("-"))
    print("  SCHEDULER ACTIVE")
    print("  * Weather Trading Loop  — every 5 minutes")
    print("  * Resolution Check      — 12:00 UTC daily")
    print("  Press Ctrl+C to stop.")
    print(_divider("-"))
    print()

    # Fix 7: Scheduler error recovery
    # A raw while True with no outer try/except would silently die on any
    # unexpected exception, leaving the bot alive but doing nothing.
    while True:
        try:
            schedule.run_pending()
        except Exception as sched_err:
            logger.error(
                "Scheduler error (run_pending): %s - bot continues running.",
                sched_err, exc_info=True,
            )
        await asyncio.sleep(30)


# -- Entry Point ----------------------------------------------------------------


async def main():
    """
    Bot entry point.
      - Init DB
      - Print startup banner
      - Run first trading loop immediately
      - Start recurring scheduler
    """
    tomorrow = get_tomorrow()

    print()
    print(_divider("*"))
    print("  POLYMARKET WEATHER TRADING BOT  (Manual Trade Entry Mode)")
    print(f"  Bankroll:   ${DRY_RUN_STARTING_BANKROLL:.2f} pUSD (configured)")
    print(f"  Target:     {tomorrow.strftime('%A, %B %d %Y')}")
    print(f"  Edge:       >= {MIN_EDGE_THRESHOLD * 100:.0f}%  |  Max open: {MAX_TRADES_PER_RUN}")
    print()
    print("  This bot RECOMMENDS trades. You place them manually on polymarket.com.")
    print("  Run `python view_db.py` to check recorded positions and PnL.")
    print(_divider("*"))
    print()

    await init_db()

    # Fix 8: Startup WeatherAPI key validation
    # Catches a missing or revoked key immediately rather than failing silently
    # 5 minutes into the first run.
    if not WEATHERAPI_KEY:
        print()
        print("  [!] WARNING: WEATHERAPI_KEY is not set in .env")
        print("       WeatherAPI forecasts will be skipped. The bot will still run")
        print("       using Open-Meteo and Met.no only (reduced accuracy).")
        print("       Set WEATHERAPI_KEY in weather_bot/.env to enable all 3 sources.")
        print()
    else:
        # Fire a quick test call to catch bad/expired keys before the first trade cycle
        try:
            async with httpx.AsyncClient() as _c:
                _r = await _c.get(
                    f"https://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q=London",
                    timeout=8.0,
                )
            if _r.status_code == 401:
                print()
                print("  ⚠️  WARNING: WEATHERAPI_KEY returned 401 Unauthorized.")
                print("       The key may be invalid or expired. WeatherAPI source will be skipped.")
                print("       Rotate the key at weatherapi.com and update weather_bot/.env.")
                print()
            elif _r.status_code == 200:
                logger.info("WeatherAPI key validated OK")
            else:
                logger.warning("WeatherAPI key check returned HTTP %d", _r.status_code)
        except Exception as _val_err:
            logger.warning("WeatherAPI key validation skipped (network error: %s)", _val_err)

    logger.info("Running initial trading loop...")
    await run_trading_loop()

    logger.info("Starting scheduler...")
    await run_scheduler()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print(_divider("-"))
        print("  Bot stopped by user (Ctrl+C). Goodbye.")
        print(_divider("-"))
    except Exception as e:
        logger.error("Bot crashed: %s", e, exc_info=True)
        sys.exit(1)
