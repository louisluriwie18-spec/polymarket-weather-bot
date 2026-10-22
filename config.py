"""
config.py — Central configuration for the Polymarket Weather Bot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── DRY RUN MODE ──────────────────────────────────────────────────────────────
# Accepts: "true" / "1" / "yes" / "on"  →  simulation (safe)
# Accepts: "false" / "0" / "no" / "off" →  record trades as live
# Any other value → warning emitted + defaults to True (safe)
_dry_run_raw = os.getenv("DRY_RUN", "true").strip().lower()
_TRUTHY = {"true", "1", "yes", "on"}
_FALSY  = {"false", "0", "no", "off"}
if _dry_run_raw not in _TRUTHY | _FALSY:
    import warnings as _warnings
    _warnings.warn(
        f"Unrecognised DRY_RUN value '{_dry_run_raw}' — defaulting to True. "
        "Check your .env file for typos.",
        stacklevel=2,
    )
DRY_RUN = _dry_run_raw not in _FALSY

# ── BANKROLL ──────────────────────────────────────────────────────────────────
# Starting bankroll used to simulate position sizing.
# Set this to match your actual Polymarket balance for accurate stake sizing.
try:
    DRY_RUN_STARTING_BANKROLL = float(os.getenv("DRY_RUN_STARTING_BANKROLL", "20.0"))
except ValueError:
    import warnings as _warnings
    _warnings.warn(
        "Invalid DRY_RUN_STARTING_BANKROLL value in .env (not a number) — defaulting to 20.0 pUSD.",
        stacklevel=2,
    )
    DRY_RUN_STARTING_BANKROLL = 20.0

# ── WEATHER API ───────────────────────────────────────────────────────────────
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")

# ── EDGE & MODEL THRESHOLDS ───────────────────────────────────────────────────
MIN_EDGE_THRESHOLD      = 0.07   # 7% minimum edge to flag a YES trade
MIN_NO_EDGE_THRESHOLD   = 0.10   # NO trades need higher conviction (counters NO-spammer noise)

# Per sub-market type thresholds.
# Tail markets (at_or_above / at_or_below) cover a wide temperature range
# → the model CAN achieve strong confidence on them → require higher minimum edge.
# Exact 1°C bins are the narrowest and riskiest sub-market type: any 1°C NWP
# error causes a miss → the model rarely achieves high confidence → accept lower
# minimum edge to still generate recommendations on these markets.
THRESHOLD_BY_TYPE: dict[str, float] = {
    "at_or_above": 0.08,  # open-ended tail — model can be confident
    "at_or_below": 0.08,  # open-ended tail — model can be confident
    "range":       0.07,  # bounded range
    "exact":       0.06,  # single 1°C bin — riskiest; model rarely > 50%
}
YES_BIAS_MULTIPLIER     = 1.10   # YES edge is treated as 10% stronger (structural YES-value bias)
MAX_FORECAST_DISAGREEMENT = 3.0  # °C — skip event if forecasts disagree by more than this
MAX_TRADES_PER_RUN = 2           # Hard cap on open positions at any time

# ── API RATE LIMITING ─────────────────────────────────────────────────────────
API_CONCURRENCY_LIMIT = 5        # Maximum concurrent external API requests
API_REQUEST_DELAY = 2.0          # Delay (seconds) after API calls to prevent 429

# ── FORECAST SOURCE WEIGHTS ───────────────────────────────────────────────────
# Weights reflect objective model quality, not equal treatment.
# Calibrate via backtesting RMSE once resolved trades accumulate:
#   weight_i ∝ 1 / RMSE_i²  (inverse-variance weighting)
SOURCE_WEIGHTS = {
    "ecmwf":         0.25,  # ECMWF IFS 0.25 -- world's best medium-range NWP
    "openmeteo":     0.20,  # ICON seamless -- excellent day-ahead accuracy
    "gfs":           0.16,  # GFS seamless -- US NWS model, strong Americas coverage
    "ukmet":         0.13,  # UK Met Office -- best for Europe
    "metno":         0.09,  # Met.no NWP -- good European + global coverage
    "local_ensemble":0.10,  # Quantitative Prediction Group (OU + ETS + Kalman)
    "polynimbus":    0.04,  # Aggregated ensemble incl. METAR anchor
    "weatherapi":    0.03,  # Single commercial model -- lowest weight
}
# Validate at import time — a zero or negative weight silently corrupts the ensemble
if not all(v > 0 for v in SOURCE_WEIGHTS.values()):
    raise ValueError(
        f"All SOURCE_WEIGHTS must be positive. Got: {SOURCE_WEIGHTS}"
    )
_weight_sum = sum(SOURCE_WEIGHTS.values())
if abs(_weight_sum - 1.0) > 0.001:
    raise ValueError(
        f"SOURCE_WEIGHTS must sum to 1.0, got {_weight_sum:.4f}. "
        "Adjust the weights so they sum to exactly 1.0."
    )
del _weight_sum

# ── PER-SOURCE FORECAST UNCERTAINTY PRIORS ────────────────────────────────────
# σᵢ = expected day-ahead RMSE (°C) for each NWP source, used as the
# denominator in Inverse-Variance Weighting (IVW) ensemble combination:
#
#   precision_i = SOURCE_WEIGHTS[i] / σᵢ²
#   μ_combined  = Σ(precision_i × μᵢ) / Σ(precision_i)
#   σ_combined  = 1 / sqrt(Σ(precision_i))
#
# Sources: WMO NWP Verification reports + ECMWF model performance archive.
# Calibrate against resolved trades once sufficient data accumulates.
FORECAST_SIGMA: dict[str, float] = {
    "ecmwf":         1.8,   # ECMWF IFS 0.25: world-best NWP, RMSE ~1.8 degC day-1
    "openmeteo":     2.0,   # ICON seamless: good day-ahead skill, RMSE ~2.0 degC
    "gfs":           2.0,   # GFS seamless: ~2.0 degC globally, stronger for Americas
    "ukmet":         1.9,   # UKMO seamless: ~1.9 degC, particularly strong in Europe
    "metno":         2.0,   # Met.no MEPS blend: ~2.0 degC day-ahead
    "local_ensemble":2.2,   # OU + ETS + Kalman: initial estimate; calibrate from resolved trades
    "polynimbus":    2.5,   # Aggregated ensemble (incl. METAR anchor): wider uncertainty
    "weatherapi":    2.5,   # Commercial proprietary model: conservative sigma estimate
}

# Confidence spread divisor: the °C spread at which inter-source confidence drops to ~0.1.
# E.g. 10.0 means a 10°C disagreement between forecast sources → confidence = 0.1.
CONFIDENCE_SPREAD_DIVISOR = 10.0

# ── STAKING PLAN ──────────────────────────────────────────────────────────────
# Maps bankroll tiers to fixed stake amounts.
# Format: (min_bankroll_pUSD, stake_per_trade_pUSD)
STAKING_PLAN = [
    (10, 1.0),
    (20, 2.0),
    (50, 4.0),
    (100, 5.0),
    (500, 10.0),
]
MAX_STAKE_FRACTION = 0.10  # Never risk more than 10% of bankroll on a single trade

# ── KELLY STAKING ─────────────────────────────────────────────────────────────
# Half-Kelly fraction: multiplies raw Kelly to reduce variance from model error.
# 0.5 = half-Kelly (standard safe choice); range 0.25–0.5 recommended.
KELLY_FRACTION     = 0.5
MIN_KELLY_STAKE    = 0.50   # Minimum stake in pUSD (avoid micro-bets)

# ── STATION COORDINATES ──────────────────────────────────────────────
# Maps Polymarket city names to the exact ICAO weather station coordinates used
# for market resolution (source: PolyNimbus dashboard + aviationweather.gov).
# Using city-centre geocoding introduces 0.5–2°C urban heat-island bias.
# Station coordinates eliminate this systematic error.
# Format: city_name -> (lat, lon, icao, description)
STATION_COORDS: dict[str, tuple[float, float, str, str]] = {
    "London":        (51.5053,   0.0553, "EGLC", "London City Airport"),
    "Paris":         (48.7233,   2.3794, "LFPO", "Paris Orly Airport"),
    "Madrid":        (40.4983,  -3.5676, "LEMD", "Madrid Barajas Airport"),
    "Milan":         (45.6306,   8.7231, "LIMC", "Milan Malpensa Airport"),
    "Munich":        (48.3537,  11.7750, "EDDM", "Munich Airport"),
    "Istanbul":      (40.9769,  28.8146, "LTBA", "Istanbul Ataturk Airport"),
    "Ankara":        (40.1281,  32.9951, "LTAC", "Ankara Esenboga Airport"),
    "Warsaw":        (52.1657,  20.9671, "EPWA", "Warsaw Chopin Airport"),
    "Tel Aviv":      (32.0114,  34.8867, "LLBG", "Ben Gurion Airport"),
    "New York":      (40.6413, -73.7781, "KJFK", "JFK International Airport"),
    "NYC":           (40.6413, -73.7781, "KJFK", "JFK International Airport"),  # Polymarket title variant
    "Chicago":       (41.9742, -87.9073, "KORD", "O'Hare International Airport"),
    "Dallas":        (32.8998, -97.0403, "KDFW", "Dallas/Fort Worth Airport"),
    "Miami":         (25.7959, -80.2870, "KMIA", "Miami International Airport"),
    "Houston":       (29.9902, -95.3368, "KIAH", "Houston Intercontinental Airport"),
    "Atlanta":       (33.6407, -84.4277, "KATL", "Atlanta Hartsfield Airport"),
    "Denver":        (39.8561, -104.6737,"KDEN", "Denver International Airport"),
    "Seattle":       (47.4502, -122.3088,"KSEA", "Seattle-Tacoma Airport"),
    "San Francisco": (37.6213, -122.3790,"KSFO", "SFO International Airport"),
    "Los Angeles":   (33.9425, -118.4081,"KLAX", "LAX International Airport"),
    "Austin":        (30.1975, -97.6664, "KAUS", "Austin-Bergstrom Airport"),
    "Toronto":       (43.6772, -79.6306, "CYYZ", "Toronto Pearson Airport"),
    "Tokyo":         (35.5494, 139.7798, "RJTT", "Tokyo Haneda Airport"),
    "Seoul":         (37.4692, 126.4510, "RKSI", "Seoul Incheon Airport"),
    "Seoul (Incheon)": (37.4692, 126.4510, "RKSI", "Seoul Incheon Airport"),  # Polymarket title variant
    "Beijing":       (40.0799, 116.5853, "ZBAA", "Beijing Capital Airport"),
    "Shanghai":      (31.1443, 121.8083, "ZSPD", "Shanghai Pudong Airport"),
    "Shenzhen":      (22.6393, 113.8107, "ZGSZ", "Shenzhen Bao'an Airport"),
    "Chengdu":       (30.5785, 103.9470, "ZUUU", "Chengdu Shuangliu Airport"),
    "Chongqing":     (29.7192, 106.6419, "ZUCK", "Chongqing Jiangbei Airport"),
    "Wuhan":         (30.7838, 114.2081, "ZHHH", "Wuhan Tianhe Airport"),
    "Hong Kong":     (22.3080, 113.9185, "VHHH", "Hong Kong Intl Airport"),
    "Taipei":        (25.0777, 121.2322, "RCTP", "Taoyuan International Airport"),
    "Singapore":     ( 1.3592, 103.9894, "WSSS", "Changi Airport"),
    "Wellington":    (-41.3272, 174.8050,"NZWN", "Wellington Airport"),
    "Buenos Aires":  (-34.8222, -58.5358,"SAEZ", "Ezeiza International Airport"),
    "Sao Paulo":     (-23.4356, -46.4731,"SBGR", "Guarulhos International Airport"),
    "Lucknow":       (26.7606,  80.8893, "VILK", "Chaudhary Charan Singh Airport"),
}
if not all(
    STAKING_PLAN[i][0] < STAKING_PLAN[i + 1][0]
    for i in range(len(STAKING_PLAN) - 1)
):
    raise ValueError("STAKING_PLAN tiers must be in strictly ascending bankroll order")
if not all(s > 0 for _, s in STAKING_PLAN):
    raise ValueError("All STAKING_PLAN stakes must be positive")


# ── CITY ALLOWLIST ────────────────────────────────────────────────────────────
# The 15 cities that are historically forecastable to ≥80% accuracy for
# next-day max/min temperature. Restricting to these cities:
#   1. Dramatically reduces API call volume (fewer events scanned)
#   2. Maximises edge-detection signal quality (high-confidence priors)
#   3. Avoids low-accuracy markets where model edge is illusory
#
# Selection criteria (NWP skill literature + meteorological logic):
#   • Low day-to-day temperature variability (forecast RMSE < 2°C typically)
#   • Stable synoptic regimes: semi-arid, subtropical maritime, equatorial
#   • Flat terrain — no orographic enhancement of forecast uncertainty
#   • Dense observational networks (radiosonde + surface stations + METAR)
#   • NWP skill ≥ 80% within 2°C for 24-hour max/min temperature
#
# Set CITY_ALLOWLIST = None to disable the filter and trade all available cities.
CITY_ALLOWLIST: set[str] | None = {
    # ── Original 15 Americas / Europe / Asia-Pacific (lowest forecast error) ──
    "Los Angeles",    # Arid Mediterranean, cold California Current anchors temps (~0.8°C variability)
    "Miami",          # Subtropical maritime, SST-controlled summer heat, >85% NWP accuracy
    "Houston",        # Gulf Coast subtropical, stable maritime air mass year-round
    "Dallas",         # Southern Plains, consistent pressure patterns, low fog/cloud
    "Atlanta",        # Warm humid subtropical, flat Piedmont, excellent US NWP coverage
    "Austin",         # Central Texas dry subtropical, strong clear-sky radiative predictability
    "Buenos Aires",   # Humid Pampa, flat terrain, South Atlantic anticyclone dominance
    "San Francisco",  # Marine climate, cold California Current stabilises temps daily
    "Madrid",         # Castilian plateau semi-arid, clear-sky dominance, very predictable
    "Milan",          # Po Valley enclosed basin, stable valley-floor weather patterns
    "Tel Aviv",       # Eastern Mediterranean, Levantine sea-breeze regularity
    "Singapore",      # Equatorial maritime, <1°C interannual max-temp variability, ~32°C always
    "Hong Kong",      # Subtropical coastal, stable Pearl River delta marine patterns
    "Tokyo",          # Kanto plain, excellent JMA observational data, consistently high NWP skill
    "Beijing",        # North China plain, continental dry, very predictable in winter/summer

    # ── 15 Additional High-Accuracy Cities (≥80% 24h NWP Skill) ──
    "London",         # Maritime temperate, EGLC station, ECMWF & UKMO top global verification (>86%)
    "Paris",          # Flat Paris basin, LFPO station, Météo-France & ECMWF consensus (>85%)
    "Shenzhen",       # South China Sea subtropical coastal plain, ZGSZ station, low diurnal noise (>86%)
    "Seoul",          # Yellow Sea coastal island buffer, RKSI station, high KMA/ECMWF skill (>85%)
    "Seoul (Incheon)",# Polymarket event title variant for Seoul
    "Shanghai",       # Yangtze delta coastal plain, ZSPD station, maritime temperature cushion (>85%)
    "Taipei",         # Subtropical coastal plain, RCTP station, dense CWA radar network (>84%)
    "Munich",         # Bavarian plateau, EDDM station, DWD ICON & ECMWF core skill domain (>83%)
    "NYC",            # Coastal Atlantic, KJFK station, dense NWS Doppler radar network (>83%)
    "New York",       # Polymarket event title variant for NYC
    "New York City",  # Polymarket event title variant for NYC
    "Seattle",        # Puget Sound marine moderation, KSEA station, low temperature volatility (>83%)
    "Ankara",         # Central Anatolian plateau, LTAC station, stable clear-sky radiative regime (>83%)
    "Chicago",        # Flat interior plains, KORD station, well-modelled lake-breeze regime (>82%)
    "Toronto",        # Great Lakes plain, CYYZ station, Environment Canada/ECMWF high skill (>82%)
    "Warsaw",         # Flat Central European plain, EPWA station, regular frontal synoptics (>82%)
    "Sao Paulo",      # Tropical plateau, SBGR station, CPTEC/ECMWF seasonal consistency (>82%)
    "São Paulo",      # Polymarket title variant with accent
    "Wuhan",          # Flat Jianghan Plain, ZHHH station, consistent radiative heating cycle (>81%)
}