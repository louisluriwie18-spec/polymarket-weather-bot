from datetime import date
import numpy as np
import pytest

from quant_ensemble import (
    OUFourierModel,
    HoltWintersETS,
    KalmanDLM,
    QuantEnsemble,
    _min_variance_weights,
)


def test_min_variance_weights():
    # Diagonal covariance matrix: inversely proportional to variances
    cov = np.diag([1.0, 2.0, 4.0])
    w = _min_variance_weights(cov)
    assert len(w) == 3
    assert abs(sum(w) - 1.0) < 1e-4
    # Lower variance must receive higher weight
    assert w[0] > w[1] > w[2]
    # No negative weights
    assert (w >= 0.0).all()


def test_ou_fourier_model():
    model = OUFourierModel()
    # 30-day sinusoidal synthetic temperature series
    doys = list(range(100, 130))
    series = [20.0 + 5.0 * np.sin(2.0 * np.pi * d / 365.0) for d in doys]

    model.fit(series, doys)
    assert model._fitted is True
    assert 0.05 <= model.theta <= 0.95

    pred = model.predict(target_doy=130, yesterday_actual=series[-1], yesterday_doy=doys[-1])
    assert pred is not None
    assert 15.0 <= pred <= 30.0


def test_holt_winters_ets():
    model = HoltWintersETS()
    # Linear upward trend
    series = [15.0 + 0.5 * i for i in range(25)]
    model.fit(series)
    assert model._fitted is True

    pred = model.predict()
    assert pred is not None
    # Next step should continue trend around 27.0 - 28.0
    assert 25.0 <= pred <= 30.0


def test_kalman_dlm():
    model = KalmanDLM()
    # Series with a sudden step change
    series = [20.0] * 15 + [25.0] * 10
    model.fit(series)
    assert model._fitted is True
    assert model.posterior_var > 0.0

    pred = model.predict()
    assert pred is not None
    # Kalman should rapidly adapt toward 25.0
    assert 23.0 <= pred <= 27.0


def test_quant_ensemble_pipeline():
    ensemble = QuantEnsemble()
    target_date = date(2026, 10, 23)
    # 40-day series
    series = [18.0 + 0.1 * i + np.sin(i / 3.0) for i in range(40)]

    res = ensemble.predict(series=series, target_date=target_date, ensemble_spread=0.8)
    assert res is not None
    assert "mean" in res and "std" in res
    assert 15.0 <= res["mean"] <= 30.0
    assert 0.5 <= res["std"] <= 5.0
