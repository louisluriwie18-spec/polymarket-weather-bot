import math
import numpy as np
import pytest

from model import (
    fahrenheit_to_celsius,
    parse_group_item_title,
    calculate_prior,
    calculate_likelihood,
    calculate_posterior,
    ivw_combine_forecasts,
    compute_event_super_forecast,
    run_bayesian_model,
    detect_edge,
    check_forecast_agreement,
    _apply_climate_correction,
)


def test_temperature_unit_conversions():
    assert abs(fahrenheit_to_celsius(32.0) - 0.0) < 1e-4
    assert abs(fahrenheit_to_celsius(212.0) - 100.0) < 1e-4
    assert abs(fahrenheit_to_celsius(68.0) - 20.0) < 1e-4


def test_parse_group_item_titles():
    # Exact celsius
    res = parse_group_item_title("25°C")
    assert res is not None
    lo, hi, stype, unit = res
    assert abs(lo - 24.5) < 1e-4 and abs(hi - 25.5) < 1e-4
    assert stype == "exact" and unit == "C"

    # Exact without degree symbol
    res = parse_group_item_title("25C")
    assert res is not None
    assert res[2] == "exact" and res[3] == "C"

    # At or below
    res = parse_group_item_title("24°C or below")
    assert res is not None
    lo, hi, stype, unit = res
    assert lo is None
    assert abs(hi - 24.5) < 1e-4
    assert stype == "at_or_below" and unit == "C"

    # At or above
    res = parse_group_item_title("34°C or higher")
    assert res is not None
    lo, hi, stype, unit = res
    assert abs(lo - 33.5) < 1e-4
    assert hi is None
    assert stype == "at_or_above" and unit == "C"

    # Range Fahrenheit
    res = parse_group_item_title("62-63°F")
    assert res is not None
    lo, hi, stype, unit = res
    assert lo is not None and hi is not None
    assert lo < hi
    assert stype == "range" and unit == "F"


def test_prior_calculation():
    # Synthetic historical data: 10 years, 1 observation per year
    hist = [(float(2010 + i), 20.0 + i * 0.5) for i in range(10)]
    current_year = 2026

    # Target bin [22.0, 24.0]
    p_yes, p_no = calculate_prior(hist, low_bound=22.0, high_bound=24.0, current_year=current_year)
    assert 0.0 < p_yes < 1.0
    assert abs((p_yes + p_no) - 1.0) < 1e-6

    # Non-overlapping bin (Laplace smoothed > 0)
    p_yes_empty, p_no_empty = calculate_prior(hist, low_bound=50.0, high_bound=60.0, current_year=current_year)
    assert p_yes_empty > 0.0
    assert p_yes_empty < 0.35


def test_likelihood_calculation():
    # Forecast mean 22.0, std 1.5
    hist_temps = [20.0, 21.0, 22.0, 23.0, 24.0]
    # Bin centered on forecast mean
    l_yes, l_no = calculate_likelihood(
        forecast_mean=22.0,
        forecast_std=1.5,
        low_bound=21.5,
        high_bound=22.5,
        historical_temps=hist_temps,
    )
    assert 0.0 < l_yes < 1.0
    assert abs((l_yes + l_no) - 1.0) < 1e-6

    # Far away bin should have much lower likelihood
    l_yes_far, _ = calculate_likelihood(
        forecast_mean=22.0,
        forecast_std=1.5,
        low_bound=35.0,
        high_bound=36.0,
        historical_temps=hist_temps,
    )
    assert l_yes_far < l_yes
    assert l_yes_far < 0.05


def test_posterior_calculation():
    p_yes, p_no = calculate_posterior(prior_yes=0.2, prior_no=0.8, likelihood_yes=0.8, likelihood_no=0.2)
    assert abs(p_yes + p_no - 1.0) < 1e-6
    # 0.2*0.8 = 0.16, 0.8*0.2 = 0.16 -> 50%
    assert abs(p_yes - 0.5) < 1e-6


def test_ivw_combine_forecasts():
    f_ec = {"mean": 20.0, "std": 1.5}
    f_gfs = {"mean": 21.0, "std": 2.0}
    weights = {"ecmwf": 0.6, "gfs": 0.4}

    mu, sigma = ivw_combine_forecasts(
        forecast_ec=f_ec,
        forecast_om=None,
        forecast_gfs=f_gfs,
        forecast_ukmet=None,
        forecast_mn=None,
        forecast_pn=None,
        forecast_wa=None,
        forecast_local=None,
        weights=weights,
    )
    # Combined mean must lie strictly between 20.0 and 21.0
    assert 20.0 <= mu <= 21.0
    # Combined sigma must be positive and reasonable
    assert 0.5 <= sigma <= 2.5


def test_compute_event_super_forecast_and_bayesian_bin():
    hist = [(2000 + i, 20.0 + (i % 5)) for i in range(20)]
    f_ec = {"mean": 22.0, "std": 1.5}
    f_om = {"mean": 22.5, "std": 1.8}

    combined, conf, hist_corr = compute_event_super_forecast(
        historical_data=hist,
        forecast_ec=f_ec,
        forecast_om=f_om,
        ensemble_std=1.0,
    )
    assert combined is not None
    assert 21.5 <= combined["mean"] <= 23.0
    assert 0.1 <= conf <= 0.95
    assert len(hist_corr) > 0

    # Evaluate bin
    yes_p, no_p = run_bayesian_model(
        historical_data_corrected=hist_corr,
        forecast=combined,
        low_bound=21.5,
        high_bound=22.5,
        current_year=2026,
    )
    assert 0.0 < yes_p < 1.0
    assert abs(yes_p + no_p - 1.0) < 1e-6


def test_check_forecast_agreement_nwp():
    f_ec = {"mean": 22.0, "std": 1.5}
    f_om = {"mean": 22.5, "std": 1.8}
    f_gfs = {"mean": 23.0, "std": 2.0}
    # Local statistical model is an outlier due to front transition
    f_local = {"mean": 15.0, "std": 2.0}

    # NWP spread is 23.0 - 22.0 = 1.0°C <= 3.0°C threshold
    ok, spread = check_forecast_agreement(
        forecast_om=f_om,
        forecast_wa=None,
        forecast_ec=f_ec,
        forecast_gfs=f_gfs,
        forecast_local=f_local,
        max_disagreement=3.0,
    )
    assert ok is True
    assert abs(spread - 1.0) < 1e-4


def test_detect_edge():
    # Significant YES edge
    side, edge = detect_edge(final_yes=0.40, final_no=0.60, market_yes_price=0.20, market_no_price=0.80)
    assert side == "YES"
    assert edge > 0.15

    # Significant NO edge
    side, edge = detect_edge(final_yes=0.10, final_no=0.90, market_yes_price=0.40, market_no_price=0.60)
    assert side == "NO"
    assert edge > 0.15

    # No edge
    side, edge = detect_edge(final_yes=0.51, final_no=0.49, market_yes_price=0.50, market_no_price=0.50)
    assert side is None


def test_climate_correction_sorting_guarantee():
    # Unordered small dataset
    raw = [(2022, 21.0), (2018, 19.5), (2020, 20.0), (2015, 18.0)]
    corrected = _apply_climate_correction(raw, current_year=2026)
    assert len(corrected) == len(raw)
    # Must be sorted ascending by year
    years = [y for y, _ in corrected]
    assert years == sorted(years)
