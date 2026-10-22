import pytest
from config import (
    SOURCE_WEIGHTS,
    FORECAST_SIGMA,
    STAKING_PLAN,
    CITY_ALLOWLIST,
    STATION_COORDS,
    MIN_EDGE_THRESHOLD,
    MIN_NO_EDGE_THRESHOLD,
)


def test_source_weights_sum_to_one():
    total = sum(SOURCE_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001, f"SOURCE_WEIGHTS must sum to 1.0, got {total}"


def test_source_weights_all_positive():
    for name, weight in SOURCE_WEIGHTS.items():
        assert weight > 0, f"Weight for {name} must be positive, got {weight}"


def test_forecast_sigmas_positive():
    for name, sigma in FORECAST_SIGMA.items():
        assert sigma > 0, f"Sigma for {name} must be positive, got {sigma}"
        assert sigma < 10.0, f"Sigma for {name} unreasonably large: {sigma}"


def test_staking_plan_order_and_positive():
    for i in range(len(STAKING_PLAN) - 1):
        assert STAKING_PLAN[i][0] < STAKING_PLAN[i + 1][0], "STAKING_PLAN bankroll tiers must be strictly ascending"
    for min_br, stake in STAKING_PLAN:
        assert min_br >= 0, "Bankroll tier threshold must be >= 0"
        assert stake > 0, "Stake must be positive"


def test_thresholds_sane():
    assert 0.01 <= MIN_EDGE_THRESHOLD <= 0.20
    assert 0.01 <= MIN_NO_EDGE_THRESHOLD <= 0.30
    assert MIN_NO_EDGE_THRESHOLD >= MIN_EDGE_THRESHOLD


def test_station_coords_format():
    for city, coord in STATION_COORDS.items():
        lat, lon, icao, desc = coord
        assert -90.0 <= lat <= 90.0, f"Invalid lat {lat} for {city}"
        assert -180.0 <= lon <= 180.0, f"Invalid lon {lon} for {city}"
        assert len(icao) == 4, f"Invalid ICAO {icao} for {city}"
