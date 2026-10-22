"""
model.py  Bayesian engine and edge detection.

This module is pure computation  no I/O, no side effects.
Every step is commented for auditability.

Handles the REAL Polymarket temperature market structure:
  - Events: "Highest temperature in Seoul on June 8?"
  - Sub-markets with groupItemTitle like:
      "24C or below"     P(T  24)
      "25C"              P(24.5 < T  25.5)
      "34C or higher"    P(T  34)
      "62-63F"           P(62  T  63)  (converted to C)
      "61F or below"     P(T  61F)

Probability Pipeline:
  1. Bayesian posterior from historical prior + forecast likelihood
  2. Skew-Normal likelihood calculation
  3. Weighted ensemble of multiple API sources
  4. Edge detection against market prices

Temperature unit handling:
  - All internal calculations are done in C
  - F thresholds from sub-market titles are converted to C
  - Forecasts and historical data are always in C
"""

import logging
import math
import re

import numpy as np
from datetime import datetime, timezone
from scipy.stats import linregress, skewnorm, skew

from config import (
    MIN_EDGE_THRESHOLD,
    MIN_NO_EDGE_THRESHOLD,
    YES_BIAS_MULTIPLIER,
    MAX_FORECAST_DISAGREEMENT,
    SOURCE_WEIGHTS,
    CONFIDENCE_SPREAD_DIVISOR,
    FORECAST_SIGMA,
)

logger = logging.getLogger(__name__)


# ── Log-odds helpers for logit pooling ────────────────────────────────
def _logit(p: float) -> float:
    """Log-odds of p, clamped to avoid -inf/+inf."""
    p = max(1e-7, min(1.0 - 1e-7, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid / inverse-logit."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# 
# 1. TEMPERATURE UNIT HANDLING
# 


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


# 
# 2. SUB-MARKET PARSING  Extract temp bounds from groupItemTitle
# 

# Regex patterns for groupItemTitle:
# Handles both "25C" and "25°C" notation.
_PAT_OR_BELOW = re.compile(
    r"(-?\d+\.?\d*)\s*°?\s*([cfCF])\s+or\s+(?:below|lower|less)",
    re.IGNORECASE,
)
_PAT_OR_ABOVE = re.compile(
    r"(-?\d+\.?\d*)\s*°?\s*([cfCF])\s+or\s+(?:higher|above|more)",
    re.IGNORECASE,
)
_PAT_RANGE = re.compile(
    r"(-?\d+\.?\d*)\s*[-]\s*(-?\d+\.?\d*)\s*°?\s*([cfCF])",
    re.IGNORECASE,
)
_PAT_EXACT = re.compile(
    r"(-?\d+\.?\d*)\s*°?\s*([cfCF])(?:\s*\?|)$",
    re.IGNORECASE,
)


def parse_group_item_title(
    title: str,
) -> tuple[float | None, float | None, str, str] | None:
    """
    Parse a sub-market's groupItemTitle to extract temperature bounds.

    Args:
        title: e.g. "25C", "24C or below", "34C or higher", "62-63F"

    Returns:
        (low_bound_C, high_bound_C, sub_type, original_unit) or None.
        - For "at_or_below": low=None, high=threshold_C
        - For "exact": low=val-0.5, high=val+0.5
        - For "range": low=low_C, high=high_C (inclusive)
        - For "at_or_above": low=threshold_C, high=None
    """
    title = title.strip()

    # Try "XC or below" / "XF or below"
    m = _PAT_OR_BELOW.search(title)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        val_c = fahrenheit_to_celsius(val + 0.5) if unit == "F" else val + 0.5
        # Add 0.5 to upper bound for "or below" to match inclusive integer buckets
        return None, val_c, "at_or_below", unit

    # Try "XC or higher" / "XF or higher"
    m = _PAT_OR_ABOVE.search(title)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        val_c = fahrenheit_to_celsius(val - 0.5) if unit == "F" else val - 0.5
        # Subtract 0.5 from lower bound for "or higher" to match inclusive integer buckets
        return val_c, None, "at_or_above", unit

    # Try "62-63F" or "78-79F" range format
    m = _PAT_RANGE.search(title)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        unit = m.group(3).upper()
        if unit == "F":
            # For F ranges like "62-63F", the range is [62, 63] inclusive
            # Convert to C: the full range maps to [F2C(lo) - 0.5, F2C(hi) + 0.5]
            # But since market resolves to integer F, use exact conversion
            lo_c = fahrenheit_to_celsius(lo - 0.5)
            hi_c = fahrenheit_to_celsius(hi + 0.5)
        else:
            lo_c = lo - 0.5
            hi_c = hi + 0.5
        return lo_c, hi_c, "range", unit

    # Try exact "25C" or "80F"
    m = _PAT_EXACT.search(title)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        val_c_low = fahrenheit_to_celsius(val - 0.5) if unit == "F" else val - 0.5
        val_c_high = fahrenheit_to_celsius(val + 0.5) if unit == "F" else val + 0.5
        # Exact means within 0.5C (or 0.5F converted)
        return val_c_low, val_c_high, "exact", unit

    logger.warning("Could not parse groupItemTitle: '%s'", title)
    return None


# 
# 3. PRIOR CALCULATION  Based on historical frequency
# 


# ─────────────────────────────────────────────────────────────────────────────
# 2b. CLIMATE CORRECTION  Detrend + MAD filter + trend shift to current year
# ─────────────────────────────────────────────────────────────────────────────


def _apply_climate_correction(
    historical_data: list[tuple[int, float]],
    current_year: int,
    mad_threshold: float = 3.5,
) -> list[float]:
    """
    Convert raw (year, temp) historical pairs into a cleaned, trend-corrected
    list of temperatures expressed in current-year equivalent units.

    Algorithm (Fix C + outlier concern):
      1. Sort by year ascending.
      2. Fit linear trend: temp = intercept + slope × year.
      3. Compute detrended residuals: residual_i = temp_i - trend(year_i).
      4. MAD filter on residuals (threshold = 3.5 × MAD by default).
         Critical: detrending FIRST means a year that is hot because of
         climate warming is NOT flagged — only genuine anomalies are removed.
      5. Shift each clean temp to current-year equivalent:
             corrected_i = temp_i + slope × (current_year − year_i)
         All historical observations are now expressed as if measured today,
         giving the prior the correct expected range for the current season.
      6. Sort corrected temps by year (ascending) so downstream exponential
         decay weighting in calculate_prior works correctly.

    Returns: list[float] sorted oldest → newest year.
    """
    if not historical_data:
        return []

    data = sorted(historical_data, key=lambda x: x[0])
    years_f = [float(d[0]) for d in data]
    temps = [d[1] for d in data]

    if len(data) < 10:
        # Insufficient data for reliable linear trend estimation.
        # Apply a lightweight 2.5×IQR outlier filter so a single record-breaking
        # year does not bias the prior when we have limited history.
        temps_all = [d[1] for d in data]
        if len(temps_all) >= 4:
            q1   = float(np.percentile(temps_all, 25))
            q3   = float(np.percentile(temps_all, 75))
            iqr  = q3 - q1
            lo_f = q1 - 2.5 * iqr
            hi_f = q3 + 2.5 * iqr
            filtered = [(float(d[0]), d[1]) for d in data if lo_f <= d[1] <= hi_f]
            if len(filtered) >= 3:   # keep filter only if it leaves enough points
                n_removed = len(data) - len(filtered)
                if n_removed > 0:
                    logger.info(
                        "  Small-dataset IQR filter: %d/%d points removed",
                        n_removed, len(data),
                    )
                filtered.sort(key=lambda x: x[0])
                return filtered
        # Fallback: not enough points to filter, return as-is (sorted)
        unfiltered = [(float(d[0]), d[1]) for d in data]
        unfiltered.sort(key=lambda x: x[0])
        return unfiltered

    # Step 1: Fit linear climate trend
    slope, intercept, r_value, p_value, _ = linregress(years_f, temps)
    r2 = r_value ** 2

    # Apply correction only when trend is statistically meaningful
    # (R² > 0.05 avoids noise-driven "corrections")
    use_trend = (r2 > 0.05) and (p_value < 0.05)
    if use_trend:
        logger.info(
            "  Climate trend: %.3f\u00b0C/year (R\u00b2=%.3f, p=%.3f) — correction applied",
            slope, r2, p_value,
        )
    else:
        slope = 0.0
        logger.debug("  No significant climate trend (R\u00b2=%.3f) — no correction", r2)

    # Step 2: Detrended residuals
    predicted = [intercept + slope * y for y in years_f]
    residuals = [t - p for t, p in zip(temps, predicted)]

    # Step 3: MAD filter on residuals
    median_res = float(np.median(residuals))
    mad = float(np.median([abs(r - median_res) for r in residuals]))

    if mad < 0.01:
        clean_mask = [True] * len(data)  # near-zero variance — skip filtering
    else:
        clean_mask = [
            abs(r - median_res) <= mad_threshold * mad
            for r in residuals
        ]

    n_removed = sum(1 for ok in clean_mask if not ok)
    if n_removed > 0:
        logger.info(
            "  Outlier removal: %d/%d data points removed (MAD threshold=%.1f×)",
            n_removed, len(data), mad_threshold,
        )

    # Step 4: Shift clean temps to current-year equivalent
    corrected: list[tuple[float, float]] = []
    for i, (year_f, temp) in enumerate(zip(years_f, temps)):
        if clean_mask[i]:
            corrected.append((year_f, temp + slope * (current_year - year_f)))

    corrected.sort(key=lambda x: x[0])  # oldest → newest for decay weighting
    return corrected


def calculate_prior(
    historical_data_corrected: list[tuple[float, float]],
    low_bound: float | None,
    high_bound: float | None,
    current_year: int,
) -> tuple[float, float]:
    """
    Calculate prior probability from historical data.

    Counts what fraction of historical temps fell within [low_bound, high_bound].
    Uses Laplace smoothing to avoid 0/1 extremes.

    Returns:
        (prior_yes, prior_no) with Laplace smoothing.
    """
    if not historical_data_corrected:
        logger.warning("No historical data  using uninformative prior 0.5/0.5")
        return 0.5, 0.5
    
    # Exponentially Decayed Weights — most recent years get highest weight.
    # Guard: for sparse datasets (<10 points or <8-year span), use uniform weights
    # to avoid discarding scarce early data via near-zero decay factors
    # (e.g. e^(-0.15×21) ≈ 0.04 for data from 2005 — almost thrown away).
    _year_span = (
        max(y for y, _ in historical_data_corrected)
        - min(y for y, _ in historical_data_corrected)
    ) if len(historical_data_corrected) > 1 else 0.0
    if len(historical_data_corrected) < 10 or _year_span < 8:
        weights = [1.0] * len(historical_data_corrected)
    else:
        weights = [np.exp(-0.15 * (current_year - y)) for y, _ in historical_data_corrected]
    
    weighted_yes = 0.0
    total_weight = sum(weights)

    for i, (_, t) in enumerate(historical_data_corrected):
        in_range = True
        if low_bound is not None and t < low_bound:
            in_range = False
        if high_bound is not None and t > high_bound:
            in_range = False
        if in_range:
            weighted_yes += weights[i]

    # Laplace smoothing with fixed pseudo_count = 1.0 observation equivalent.
    # A fixed pseudo_count ensures smoothing weakens proportionally as real data
    # accumulates (larger datasets → less relative smoothing). The previous avg_w
    # implementation incorrectly scaled pseudo_count with dataset size, giving
    # the same relative smoothing to 5-year and 50-year datasets.
    pseudo_count = 1.0
    prior_yes = (weighted_yes + pseudo_count) / (total_weight + 2.0 * pseudo_count)
    prior_no = 1.0 - prior_yes

    logger.debug(
        "Prior: %.2f/%.2f in [%s, %s]  P(yes)=%.3f",
        weighted_yes, total_weight,
        f"{low_bound:.1f}" if low_bound is not None else "-",
        f"{high_bound:.1f}" if high_bound is not None else "+",
        prior_yes,
    )

    return prior_yes, prior_no


# 
# 4. LIKELIHOOD CALCULATION  Based on forecast distribution
# 


def calculate_likelihood(
    forecast_mean: float,
    forecast_std: float,
    low_bound: float | None,
    high_bound: float | None,
    historical_temps: list[float]
) -> tuple[float, float]:
    """
    Calculate likelihood of YES given the forecast distribution.
    Uses Skew-Normal distribution and Confidence Scaling.
    """
    # Enforce meteorological station observation uncertainty floor (minimum 1.4°C effective std)
    forecast_std = math.sqrt(max(0.5, forecast_std)**2 + 1.2**2)

    # Improvement F: Confidence Scaling (Dynamic STD)
    if historical_temps and len(historical_temps) > 3:
        # ddof=1: sample std (unbiased estimator) — population std underestimates
        # variability for small historical samples (5-15 points).
        hist_std = float(np.std(historical_temps, ddof=1))
        # If history is highly volatile, trust API std less
        if hist_std > 2.0:
            forecast_std = max(forecast_std, hist_std * 0.4)
            
    # T2-B: Climatological anomaly-aware uncertainty boost.
    # NWP models perform worst on anomalous days. If the forecast is >1.5σ from
    # the historical mean, inflate std proportionally to reflect higher model error.
    if historical_temps and len(historical_temps) > 5:
        hist_mean_val = float(np.mean(historical_temps))
        hist_std_val  = float(np.std(historical_temps, ddof=1))  # sample std
        if hist_std_val > 0.5:
            anomaly_z = abs(forecast_mean - hist_mean_val) / hist_std_val
            if anomaly_z > 1.5:
                anomaly_boost = (anomaly_z - 1.5) * 0.30  # +0.3°C per σ beyond 1.5
                forecast_std = math.sqrt(forecast_std**2 + anomaly_boost**2)
                logger.debug(
                    "  Anomaly boost: forecast z=%.2fσ → std boosted +%.2f°C",
                    anomaly_z, anomaly_boost,
                )

    # Improvement H: Skew-Normal Distribution
    alpha = 0.0
    if historical_temps and len(historical_temps) > 3:
        # Use a tolerance rather than exact equality: floating-point variance of
        # near-constant data can be 1e-30 rather than exactly 0, which would pass
        # the check and produce a numerically undefined skew value.
        if np.var(historical_temps) < 1e-6:
            alpha = 0.0
        else:
            raw_alpha = skew(historical_temps)
            # Clamp alpha to prevent extreme distribution distortion.
            # skewnorm is numerically stable in [-10, 10]; beyond that
            # The CDF degenerates toward a step function.
            alpha = max(-5.0, min(5.0, raw_alpha))
        
    delta = alpha / math.sqrt(1 + alpha**2)
    adjusted_loc = forecast_mean - forecast_std * delta * math.sqrt(2 / math.pi)

    # Bin boundary smoothing: station readings are rounded to nearest integer, so
    # the effective bin boundary has ±0.5°C fuzziness. We convolve each CDF
    # evaluation with a Gaussian (σ=0.3°C) by averaging at ±0.3°C — this spreads
    # probability mass near boundaries and prevents overconfident calls when the
    # forecast lands within 0.5°C of a bin edge.
    BOUNDARY_SIGMA = 0.3  # °C — half the typical station rounding uncertainty

    def _smooth_cdf(x):
        return 0.5 * (
            skewnorm.cdf(x + BOUNDARY_SIGMA, a=alpha, loc=adjusted_loc, scale=forecast_std)
            + skewnorm.cdf(x - BOUNDARY_SIGMA, a=alpha, loc=adjusted_loc, scale=forecast_std)
        )

    if high_bound is not None:
        cdf_high = _smooth_cdf(high_bound)
    else:
        cdf_high = 1.0

    if low_bound is not None:
        cdf_low = _smooth_cdf(low_bound)
    else:
        cdf_low = 0.0

    likelihood_yes = cdf_high - cdf_low
    likelihood_yes = max(0.001, min(0.999, likelihood_yes))
    likelihood_no = 1.0 - likelihood_yes

    return likelihood_yes, likelihood_no


# 
# 5. POSTERIOR CALCULATION  Bayes' theorem
# 


def calculate_posterior(
    prior_yes: float,
    prior_no: float,
    likelihood_yes: float,
    likelihood_no: float,
) -> tuple[float, float]:
    """
    Apply Bayes' theorem to compute posterior probabilities.

    P(YES | data) = P(data | YES)  P(YES) / P(data)

    Returns:
        (posterior_yes, posterior_no) normalized to sum to 1.
    """
    unnorm_yes = prior_yes * likelihood_yes
    unnorm_no = prior_no * likelihood_no
    evidence = unnorm_yes + unnorm_no

    if evidence < 1e-10:
        logger.warning("Evidence near zero (%.2e) — returning uniform posterior", evidence)
        return 0.5, 0.5

    posterior_yes = unnorm_yes / evidence
    posterior_no = unnorm_no / evidence

    return posterior_yes, posterior_no


# 
# 6. FULL PIPELINE  Multi-Source Bayesian Model
# 


def run_bayesian_model(
    historical_data_corrected: list[tuple[float, float]],
    forecast: dict[str, float],
    low_bound: float | None,
    high_bound: float | None,
    current_year: int,
) -> tuple[float, float]:
    """
    Run the full Bayesian pipeline for one forecast source.

    Pipeline:
        1. calculate_prior(historical_temps, low_bound, high_bound)
        2. calculate_likelihood(forecast_mean, forecast_std, low_bound, high_bound)
        3. calculate_posterior(prior, likelihood)

    Returns:
        (posterior_yes, posterior_no) for this forecast source.
    """
    prior_yes, prior_no = calculate_prior(historical_data_corrected, low_bound, high_bound, current_year)
    historical_temps = [t for _, t in historical_data_corrected]
    likelihood_yes, likelihood_no = calculate_likelihood(
        forecast["mean"], forecast["std"], low_bound, high_bound, historical_temps
    )
    posterior_yes, posterior_no = calculate_posterior(
        prior_yes, prior_no, likelihood_yes, likelihood_no
    )
    return posterior_yes, posterior_no


# 
# 6b. IVW SUPER-FORECAST  Optimal linear combination of NWP sources
# 


def ivw_combine_forecasts(
    forecast_ec:     dict[str, float] | None,
    forecast_om:     dict[str, float] | None,
    forecast_gfs:    dict[str, float] | None,
    forecast_ukmet:  dict[str, float] | None,
    forecast_mn:     dict[str, float] | None,
    forecast_pn:     dict[str, float] | None,
    forecast_wa:     dict[str, float] | None,
    forecast_local:  dict[str, float] | None,
    weights: dict[str, float],
) -> tuple[float, float]:
    """
    Inverse-Variance Weighting (IVW) combination of independent NWP forecasts.

    For independent Gaussian distributions N(μᵢ, σᵢ²), the optimal linear
    unbiased estimator (BLUE) is:

        precision_i = source_weight_i / σᵢ²
        μ_combined  = Σ(precision_i × μᵢ) / Σ(precision_i)
        σ_combined  = 1 / sqrt(Σ(precision_i))

    source_weight_i acts as a domain-knowledge scaling factor encoding our prior
    belief that ECMWF outperforms GFS globally, without altering the fundamental
    IVW structure. Each source's σ comes directly from its forecast dict (set
    at fetch time from FORECAST_SIGMA priors in config.py).

    Returns: (combined_mean_°C, combined_sigma_°C)

    Extensibility: to add a third model ensemble, simply extend the _sources
    list with ("new_source_key", forecast_new, default_weight) — no other
    changes needed.
    """
    _sources = [
        ("ecmwf",         forecast_ec),
        ("openmeteo",     forecast_om),
        ("gfs",           forecast_gfs),
        ("ukmet",         forecast_ukmet),
        ("metno",         forecast_mn),
        ("local_ensemble",forecast_local),  # Quantitative Prediction Group
        ("polynimbus",    forecast_pn),
        ("weatherapi",    forecast_wa),
    ]

    precision_total = 0.0
    weighted_mean   = 0.0
    active: list[str] = []

    for source_key, fc in _sources:
        if fc is None:
            continue
        w = weights.get(source_key, 0.0)
        if w <= 0:
            continue
        mu    = fc["mean"]
        # Use the forecast's own σ (set from FORECAST_SIGMA at fetch time).
        # Clamp to a minimum to prevent division by near-zero from dominating.
        sigma = max(0.3, float(fc.get("std") or FORECAST_SIGMA.get(source_key, 2.0)))
        p     = w / (sigma ** 2)
        precision_total += p
        weighted_mean   += p * mu
        active.append(f"{source_key}({mu:.1f}°C±{sigma:.1f})")

    if precision_total < 1e-10:
        logger.warning("IVW: no valid forecast sources — returning fallback (0.0, 5.0)")
        return 0.0, 5.0

    combined_mean  = weighted_mean / precision_total
    combined_sigma = math.sqrt(1.0 / precision_total)

    logger.info(
        "IVW super-forecast: μ=%.2f°C  σ=%.2f°C  [%s]",
        combined_mean, combined_sigma, "  ".join(active),
    )
    return combined_mean, combined_sigma


def compute_event_super_forecast(
    historical_data: list[tuple[int, float]],
    forecast_ec: dict[str, float] | None = None,
    forecast_om: dict[str, float] | None = None,
    forecast_gfs: dict[str, float] | None = None,
    forecast_ukmet: dict[str, float] | None = None,
    forecast_mn: dict[str, float] | None = None,
    forecast_pn: dict[str, float] | None = None,
    forecast_wa: dict[str, float] | None = None,
    forecast_local: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    ensemble_std: float | None = None,
) -> tuple[dict[str, float] | None, float, list[tuple[float, float]]]:
    """
    Pre-compute the IVW super-forecast, event confidence score, and
    climate-corrected historical data ONCE for an entire event.
    Avoids re-running IVW and climate-correction redundantly across N sub-market bins.

    Returns:
        (combined_forecast_dict or None, confidence, historical_data_corrected)
    """
    if weights is None:
        weights = SOURCE_WEIGHTS

    current_year = datetime.now(timezone.utc).year
    historical_data_corrected = _apply_climate_correction(historical_data, current_year)
    if not historical_data_corrected:
        logger.warning("No clean historical data after climate correction — returning None")
        return None, 0.1, []

    combined_mean, combined_sigma = ivw_combine_forecasts(
        forecast_ec=forecast_ec,
        forecast_om=forecast_om,
        forecast_gfs=forecast_gfs,
        forecast_ukmet=forecast_ukmet,
        forecast_mn=forecast_mn,
        forecast_pn=forecast_pn,
        forecast_wa=forecast_wa,
        forecast_local=forecast_local,
        weights=weights,
    )

    if combined_mean == 0.0 and combined_sigma == 5.0:
        return None, 0.1, historical_data_corrected

    if ensemble_std is not None:
        # Blend IVW combined uncertainty with ICON ensemble spread rather than throwing IVW out
        combined_sigma = math.sqrt(0.5 * (combined_sigma ** 2) + 0.5 * (ensemble_std ** 2))
        combined_sigma = max(0.5, combined_sigma)
        logger.info(
            "  Ensemble spread (%.2f°C) blended with IVW σ → combined σ=%.2f°C",
            ensemble_std, combined_sigma,
        )

    combined_forecast = {"mean": combined_mean, "std": combined_sigma}

    available = [
        f for f in [forecast_ec, forecast_om, forecast_gfs, forecast_ukmet,
                    forecast_mn, forecast_local, forecast_pn, forecast_wa]
        if f is not None
    ]
    n_sources = len(available)
    means = [f["mean"] for f in available]

    if n_sources >= 2:
        spread = max(means) - min(means)
        spread_factor = max(0.0, 1.0 - spread / CONFIDENCE_SPREAD_DIVISOR)
    elif n_sources == 1:
        spread_factor = 0.5
    else:
        return None, 0.1, historical_data_corrected

    if ensemble_std is not None:
        ens_factor = max(0.0, 1.0 - ensemble_std / 3.0)
    else:
        ens_factor = None

    unique_hist_years = len({int(y) for y, _ in historical_data_corrected})
    hist_factor = min(1.0, unique_hist_years / 20.0)

    if ens_factor is not None:
        confidence = max(0.1, min(0.95,
            spread_factor * 0.50 + ens_factor * 0.35 + hist_factor * 0.15
        ))
    else:
        confidence = max(0.1, min(0.95,
            spread_factor * 0.85 + hist_factor * 0.15
        ))

    logger.info(
        "Super-forecast pre-computed: μ=%.2f°C  σ=%.2f°C  conf=%.2f  (sources=%d)",
        combined_mean, combined_sigma, confidence, n_sources,
    )
    return combined_forecast, confidence, historical_data_corrected


def run_forecast_model(
    historical_data: list[tuple[int, float]],
    forecast_om: dict[str, float] | None,
    forecast_wa: dict[str, float] | None,
    forecast_mn: dict[str, float] | None,
    forecast_pn: dict[str, float] | None,
    low_bound: float | None,
    high_bound: float | None,
    weights: dict[str, float] | None = None,
    ensemble_std: float | None = None,
    forecast_ec: dict[str, float] | None = None,
    forecast_gfs: dict[str, float] | None = None,
    forecast_ukmet: dict[str, float] | None = None,
    forecast_local: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """
    Run the full forecasting pipeline using IVW super-forecast + one Bayesian run.
    """
    combined_forecast, confidence, hist_corr = compute_event_super_forecast(
        historical_data=historical_data,
        forecast_ec=forecast_ec,
        forecast_om=forecast_om,
        forecast_gfs=forecast_gfs,
        forecast_ukmet=forecast_ukmet,
        forecast_mn=forecast_mn,
        forecast_pn=forecast_pn,
        forecast_wa=forecast_wa,
        forecast_local=forecast_local,
        weights=weights,
        ensemble_std=ensemble_std,
    )
    if combined_forecast is None or not hist_corr:
        return 0.5, 0.5, 0.1

    current_year = datetime.now(timezone.utc).year
    posterior_yes, posterior_no = run_bayesian_model(
        hist_corr, combined_forecast, low_bound, high_bound, current_year
    )
    return posterior_yes, posterior_no, confidence


# 
# 7. COMBINE POSTERIORS  Logit-pooled weighted combination
# 
# NOTE: combine_posteriors is preserved for future hybrid use.
# When a structurally different third model ensemble (e.g. ML-based NWP) is
# added, it may produce posteriors that differ fundamentally from the Gaussian
# IVW super-forecast. combine_posteriors provides the mechanism to merge those
# two posteriors at the probability level without re-running history.
# For the current all-Gaussian NWP ensemble, IVW in run_forecast_model() is used.
# 


def combine_posteriors(
    posterior_om: tuple[float, float] | None,
    posterior_wa: tuple[float, float] | None,
    posterior_mn: tuple[float, float] | None,
    posterior_pn: tuple[float, float] | None,
    weights: dict[str, float] | None = None,
    posterior_ec: tuple[float, float] | None = None,
    posterior_gfs: tuple[float, float] | None = None,
    posterior_ukmet: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Combine posteriors from multiple forecast sources via log-odds (logit) pooling.
    """
    if weights is None:
        weights = SOURCE_WEIGHTS

    # Weights — fallbacks mirror SOURCE_WEIGHTS for consistency
    w_ec    = weights.get("ecmwf",      SOURCE_WEIGHTS.get("ecmwf",      0.28))
    w_om    = weights.get("openmeteo",  SOURCE_WEIGHTS.get("openmeteo",  0.22))
    w_gfs   = weights.get("gfs",        SOURCE_WEIGHTS.get("gfs",        0.18))
    w_ukmet = weights.get("ukmet",      SOURCE_WEIGHTS.get("ukmet",      0.14))
    w_mn    = weights.get("metno",      SOURCE_WEIGHTS.get("metno",      0.10))
    w_pn    = weights.get("polynimbus", SOURCE_WEIGHTS.get("polynimbus", 0.05))
    w_wa    = weights.get("weatherapi", SOURCE_WEIGHTS.get("weatherapi", 0.03))

    logit_sum = 0.0
    total_w   = 0.0

    if posterior_ec is not None:
        logit_sum += w_ec * _logit(posterior_ec[0]);    total_w += w_ec
    if posterior_om is not None:
        logit_sum += w_om * _logit(posterior_om[0]);    total_w += w_om
    if posterior_gfs is not None:
        logit_sum += w_gfs * _logit(posterior_gfs[0]);  total_w += w_gfs
    if posterior_ukmet is not None:
        logit_sum += w_ukmet * _logit(posterior_ukmet[0]); total_w += w_ukmet
    if posterior_mn is not None:
        logit_sum += w_mn * _logit(posterior_mn[0]);    total_w += w_mn
    if posterior_pn is not None:
        logit_sum += w_pn * _logit(posterior_pn[0]);    total_w += w_pn
    if posterior_wa is not None:
        logit_sum += w_wa * _logit(posterior_wa[0]);    total_w += w_wa

    if total_w < 1e-9:
        logger.warning("All forecast APIs failed. Returning 0.5/0.5")
        return 0.5, 0.5

    final_yes = _sigmoid(logit_sum / total_w)
    final_no  = 1.0 - final_yes
    return final_yes, final_no


# 
# 9. EDGE DETECTION
# 


def detect_edge(
    final_yes: float,
    final_no: float,
    market_yes_price: float,
    market_no_price: float,
    threshold: float = MIN_EDGE_THRESHOLD,
    no_threshold: float = MIN_NO_EDGE_THRESHOLD,
    yes_bias: float = YES_BIAS_MULTIPLIER,
) -> tuple[str | None, float]:
    """
    Detect a tradeable edge between model probabilities and market prices.

    YES vs NO asymmetry (structural):
    ─────────────────────────────────
    On Polymarket temperature markets, a subset of traders systematically buys
    NO on every bin regardless of forecast. This 'NO-spamming' artificially
    inflates NO prices and deflates YES prices, creating structural value on
    the YES side. We compensate with two mechanisms:

      1. YES_BIAS: multiply raw YES edge by yes_bias (1.10) before comparing
         to threshold. A 6.4% raw YES edge is treated as a 7% qualifying edge.
      2. NO threshold: NO trades require a higher bar (no_threshold) to filter
         out edges that are merely the result of NO-spam rather than genuine
         model conviction.

    NO trades are NOT removed — they are accepted when the model genuinely
    predicts a bin will miss. They just require stronger evidence.

    Returns:
        (side, raw_edge)  — side is "YES", "NO", or None.
                           raw_edge is always the un-biased actual edge.
    """
    yes_edge = final_yes - market_yes_price
    no_edge  = final_no  - market_no_price

    logger.info(
        "Edge: YES=%+.1f%% (ours=%.1f%% mkt=%.1f%%) | NO=%+.1f%% (ours=%.1f%% mkt=%.1f%%)",
        yes_edge * 100, final_yes * 100, market_yes_price * 100,
        no_edge  * 100, final_no  * 100, market_no_price  * 100,
    )

    # Apply YES bias: a biased_yes_edge ≥ threshold is the YES qualifying bar.
    # This effectively lowers the raw YES threshold to (threshold / yes_bias).
    biased_yes_edge = yes_edge * yes_bias
    # Round to 6dp to avoid IEEE 754 floating-point precision artefacts
    # (e.g. 0.90 - 0.80 = 0.09999999999999998, not 0.10)
    yes_qualifies = round(biased_yes_edge, 6) >= threshold
    no_qualifies  = round(no_edge, 6)         >= no_threshold

    if yes_qualifies and no_qualifies:
        # Both qualify — prefer the side with the larger raw edge.
        # YES_BIAS serves as tiebreaker only when edges are within 2% of each other.
        if no_edge > yes_edge + 0.02:
            logger.info(
                "NO preferred over YES: NO edge=%.1f%% exceeds YES edge=%.1f%% by >2%%",
                no_edge * 100, yes_edge * 100,
            )
            return "NO", no_edge
        logger.info(
            "YES preferred over NO: biased YES edge=%.1f%% vs NO edge=%.1f%%",
            biased_yes_edge * 100, no_edge * 100,
        )
        return "YES", yes_edge

    if yes_qualifies:
        logger.info("YES edge: %.1f%% (biased: %.1f%%)", yes_edge * 100, biased_yes_edge * 100)
        return "YES", yes_edge

    if no_qualifies:
        logger.info("NO edge: %.1f%% (genuine model conviction)", no_edge * 100)
        return "NO", no_edge

    logger.info("No edge (best: %.2f%%)", max(yes_edge, no_edge) * 100)
    return None, max(yes_edge, no_edge)


# 
# 10. FORECAST AGREEMENT CHECK
# 


def check_forecast_agreement(
    forecast_om: dict[str, float] | None,
    forecast_wa: dict[str, float] | None,
    forecast_mn: dict[str, float] | None = None,
    forecast_pn: dict[str, float] | None = None,
    max_disagreement: float = MAX_FORECAST_DISAGREEMENT,
    hist_std: float | None = None,
    forecast_ec: dict[str, float] | None = None,
    forecast_gfs: dict[str, float] | None = None,
    forecast_ukmet: dict[str, float] | None = None,
    forecast_local: dict[str, float] | None = None,
) -> tuple[bool, float]:
    """
    Check if physical NWP forecast sources agree within acceptable bounds.
    Note: forecast_local (QuantEnsemble) is an auto-regressive statistical model
    without atmospheric physics; it is logged for observability but excluded from
    the spread calculation to avoid false rejections during frontal airmass transitions.
    """
    nwp_sources = [
        f for f in (forecast_ec, forecast_gfs, forecast_ukmet,
                    forecast_om, forecast_wa, forecast_mn, forecast_pn)
        if f is not None
    ]
    if len(nwp_sources) < 2:
        return True, 0.0  # Can't disagree with fewer than 2 physical sources

    means = [f["mean"] for f in nwp_sources]
    spread = max(means) - min(means)

    # Fix 5: dynamic threshold
    if hist_std is not None and hist_std > 0:
        effective_threshold = max(1.5, hist_std * 0.8)
        logger.debug(
            "  Dynamic disagreement threshold: %.2f°C (hist_std=%.2f°C × 0.8, floor=1.5°C)",
            effective_threshold, hist_std,
        )
    else:
        effective_threshold = max_disagreement

    agreement_ok = spread <= effective_threshold

    if not agreement_ok:
        source_labels = []
        if forecast_ec is not None:
            source_labels.append(f"EC={forecast_ec['mean']:.1f}°C")
        if forecast_gfs is not None:
            source_labels.append(f"GFS={forecast_gfs['mean']:.1f}°C")
        if forecast_ukmet is not None:
            source_labels.append(f"UK={forecast_ukmet['mean']:.1f}°C")
        if forecast_om is not None:
            source_labels.append(f"OM={forecast_om['mean']:.1f}°C")
        if forecast_wa is not None:
            source_labels.append(f"WA={forecast_wa['mean']:.1f}°C")
        if forecast_mn is not None:
            source_labels.append(f"MN={forecast_mn['mean']:.1f}°C")
        if forecast_pn is not None:
            source_labels.append(f"PN={forecast_pn['mean']:.1f}°C")
        if forecast_local is not None:
            source_labels.append(f"QUANT={forecast_local['mean']:.1f}°C (stat)")
        logger.warning(
            "⚠ NWP forecast disagreement: %s (NWP spread=%.1f°C, threshold=%.2f°C)",
            " | ".join(source_labels), spread, effective_threshold,
        )

    return agreement_ok, spread
