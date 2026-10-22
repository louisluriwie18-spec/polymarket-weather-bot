"""
quant_ensemble.py — Quantitative Prediction Group

A three-model minimum-variance statistical ensemble that acts as a single,
independent forecast input alongside the 7 NWP API sources.

Models
------
1. OUFourierModel   — Ornstein-Uhlenbeck with Fourier seasonal baseline
                      Captures the gravitational pull of the calendar cycle.
                      City-specific mean-reversion speed theta estimated from data.

2. HoltWintersETS   — Holt linear (double exponential smoothing)
                      Captures short-term atmospheric momentum.
                      No seasonal component (OU already owns that signal).

3. KalmanDLM        — 2-state Dynamic Linear Model (level + trend)
                      Adaptive anomaly tracker: Kalman Gain spikes on cold-front /
                      sudden-block events, rapidly realigning the state vector.
                      Posterior variance -> sigma output.

Fusion pipeline
---------------
  Walk-forward debiasing (14 days)
  -> sample error covariance (3 x 14, p/n = 0.21, no Ledoit-Wolf needed)
  -> regime detection (STABLE / CHAOTIC) via NWP ensemble spread threshold
  -> min-variance portfolio weights  w* = Sigma^-1 1 / (1^T Sigma^-1 1)  clipped [0.05, 0.90]
  -> weighted forecast mean
  -> sigma from Kalman posterior variance (60%) + rolling RMSE (40%)

Output
------
  {"mean": float, "std": float}  -- plugs directly into IVW combiner as "local_ensemble"

Adding a fourth quant model: extend _quick-predict helpers and the full-fit block.
No other changes required.
"""

from __future__ import annotations

import math
import logging
import numpy as np
from datetime import date, timedelta

logger = logging.getLogger(__name__)


# == Utilities =================================================================

def _quick_holt(series: list[float], alpha: float = 0.30, beta: float = 0.10) -> float | None:
    """
    Fast Holt linear smoothing with FIXED parameters.
    Used in the walk-forward debiasing loop to avoid running scipy.optimize
    14 times per event.  Returns 1-step-ahead forecast or None.
    """
    if len(series) < 2:
        return None
    level = series[0]
    trend = series[1] - series[0]
    for t in range(1, len(series)):
        level_new = alpha * series[t] + (1.0 - alpha) * (level + trend)
        trend     = beta  * (level_new - level) + (1.0 - beta) * trend
        level     = level_new
    return level + trend


def _quick_kalman(series: list[float]) -> float | None:
    """
    Fast Kalman DLM with default noise ratios for the debiasing loop.
    Returns 1-step-ahead forecast.
    """
    if len(series) < 3:
        return None
    s = np.array(series, dtype=float)
    R = max(0.10, float(np.var(np.diff(s), ddof=1)) * 0.5)
    Q = np.array([[R * 0.01, 0.0], [0.0, R * 0.001]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    x = np.array([s[0], s[1] - s[0]])
    P = np.eye(2) * R * 10.0
    for t in range(1, len(s)):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        S  = float((H @ Pp @ H.T)[0, 0]) + R
        K  = (Pp @ H.T) / S
        x  = xp + K.flatten() * (s[t] - float((H @ xp)[0]))
        P  = (np.eye(2) - np.outer(K.flatten(), H[0])) @ Pp
    return float((H @ (F @ x))[0])


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """
    Global minimum-variance portfolio:
        w* = Sigma^-1 * 1 / (1^T * Sigma^-1 * 1)

    Weights clipped to [0.05, 0.90] and renormalised.
    Prevents negative weights (no short-selling for forecast combination).
    Falls back to equal weights if the covariance matrix is singular.
    """
    n = cov.shape[0]
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-6)
        raw     = inv_cov @ np.ones(n)
        raw     = np.clip(raw, 0.05, 0.90)
        return raw / raw.sum()
    except np.linalg.LinAlgError:
        logger.warning("QuantEnsemble: covariance inversion failed -- using equal weights")
        return np.ones(n) / n


# == Model 1: Ornstein-Uhlenbeck with Fourier Seasonal Baseline ================

class OUFourierModel:
    """
    Discretized Ornstein-Uhlenbeck process:

        T_t = mu_t + (1 - theta) * (T_{t-1} - mu_{t-1}) + eps_t

    mu_t is the Fourier seasonal baseline (2 harmonics: annual + semi-annual).

    theta (mean-reversion speed) is estimated via AR(1) on deseasonalized residuals:
        theta = 1 - AR(1)_coefficient
    High theta -> fast mean-reversion (continental interiors, e.g. Dallas, Moscow).
    Low  theta -> slow  mean-reversion (marine climates, e.g. Los Angeles, London).
    """

    N_HARMONICS: int = 2

    def __init__(self) -> None:
        self._coeffs: np.ndarray | None = None
        self.theta:   float = 0.30
        self._fitted: bool  = False

    def _design_row(self, doy: float) -> list[float]:
        row: list[float] = [1.0]
        for k in range(1, self.N_HARMONICS + 1):
            row.append(float(np.cos(2.0 * np.pi * k * doy / 365.0)))
            row.append(float(np.sin(2.0 * np.pi * k * doy / 365.0)))
        return row

    def _baseline(self, doy: int) -> float:
        if self._coeffs is None:
            return 0.0
        return float(np.dot(self._coeffs, self._design_row(float(doy))))

    def fit(self, series: list[float], doys: list[int]) -> None:
        """Fit Fourier coefficients and estimate theta. series and doys must be same length (>=10)."""
        if len(series) < 10 or len(series) != len(doys):
            self._fitted = False
            return
        s = np.array(series, dtype=float)
        X = np.array([self._design_row(d) for d in doys])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, s, rcond=None)
            self._coeffs    = coeffs
        except Exception as exc:
            logger.debug("OUFourierModel: lstsq failed -- %s", exc)
            self._fitted = False
            return
        residuals = s - X @ coeffs
        if len(residuals) > 5:
            r          = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            self.theta = float(np.clip(1.0 - r, 0.05, 0.95))
        self._fitted = True

    def predict(self, target_doy: int, yesterday_actual: float, yesterday_doy: int) -> float | None:
        if not self._fitted:
            return None
        return (
            self._baseline(target_doy)
            + (1.0 - self.theta) * (yesterday_actual - self._baseline(yesterday_doy))
        )


def _quick_ou(
    series: list[float], doys: list[int], target_doy: int, yesterday_doy: int
) -> float | None:
    """Quick OU prediction for the debiasing walk-forward loop."""
    ou = OUFourierModel()
    ou.fit(series, doys)
    return ou.predict(target_doy, series[-1], yesterday_doy) if series else None


# == Model 2: Holt-Winters ETS =================================================

class HoltWintersETS:
    """
    Holt linear method (double exponential smoothing), 1-step-ahead forecast.
    alpha and beta optimised jointly via Nelder-Mead minimising SSE.
    No seasonal component -- seasonality is owned by OUFourierModel.
    """

    def __init__(self) -> None:
        self.alpha:   float = 0.30
        self.beta:    float = 0.10
        self._level:  float | None = None
        self._trend:  float | None = None
        self._fitted: bool  = False

    def _smooth(self, s: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
        level = s[0]
        trend = s[1] - s[0]
        for t in range(1, len(s)):
            level_new = alpha * s[t] + (1.0 - alpha) * (level + trend)
            trend     = beta  * (level_new - level) + (1.0 - beta) * trend
            level     = level_new
        return level, trend

    def fit(self, series: list[float]) -> None:
        if len(series) < 4:
            self._fitted = False
            return
        s = np.array(series, dtype=float)

        def _sse(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            if not (0.01 <= a <= 0.99 and 0.01 <= b <= 0.99):
                return 1e9
            lv = s[0]; tr = s[1] - s[0]; sse = 0.0
            for t in range(1, len(s)):
                err = s[t] - (lv + tr); sse += err * err
                lv_new = a * s[t] + (1.0 - a) * (lv + tr)
                tr = b * (lv_new - lv) + (1.0 - b) * tr
                lv = lv_new
            return sse

        try:
            from scipy.optimize import minimize
            res = minimize(
                _sse, [self.alpha, self.beta], method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 300},
            )
            if res.success:
                self.alpha = float(np.clip(res.x[0], 0.01, 0.99))
                self.beta  = float(np.clip(res.x[1], 0.01, 0.99))
        except Exception as exc:
            logger.debug("HoltWintersETS: optimisation failed (%s) -- using defaults", exc)

        self._level, self._trend = self._smooth(s, self.alpha, self.beta)
        self._fitted = True

    def predict(self) -> float | None:
        if not self._fitted or self._level is None:
            return None
        return self._level + self._trend


# == Model 3: Kalman Filter DLM ================================================

class KalmanDLM:
    """
    2-state Dynamic Linear Model:
        state = [level, trend]
        F     = [[1, 1], [0, 1]]   transition (level advances by trend each step)
        H     = [[1, 0]]           observation (only level is directly observed)

    R (measurement noise) and Q (process noise) are data-driven:
        R = 50% of variance of first-differences
        Q = R * diag([0.01, 0.001])

    posterior_var is exposed after fit() as the one-step-ahead predictive variance.
    This is used as sigma in the IVW combiner -- a principled, not heuristic, estimate.
    """

    F: np.ndarray = np.array([[1.0, 1.0], [0.0, 1.0]])
    H: np.ndarray = np.array([[1.0, 0.0]])

    def __init__(self) -> None:
        self._x:    np.ndarray | None = None
        self._P:    np.ndarray | None = None
        self._R:    float = 1.0
        self._Q:    np.ndarray | None = None
        self._fitted:       bool  = False
        self.posterior_var: float = 4.0

    def fit(self, series: list[float]) -> None:
        if len(series) < 5:
            self._fitted = False
            return
        s = np.array(series, dtype=float)
        self._R = max(0.10, float(np.var(np.diff(s), ddof=1)) * 0.5)
        self._Q = np.array([[self._R * 0.01, 0.0], [0.0, self._R * 0.001]])
        x = np.array([s[0], s[1] - s[0]])
        P = np.eye(2) * self._R * 10.0
        for t in range(1, len(s)):
            xp = self.F @ x
            Pp = self.F @ P @ self.F.T + self._Q
            S  = float((self.H @ Pp @ self.H.T)[0, 0]) + self._R
            K  = (Pp @ self.H.T) / S
            x  = xp + K.flatten() * (s[t] - float((self.H @ xp)[0]))
            P  = (np.eye(2) - np.outer(K.flatten(), self.H[0])) @ Pp
        self._x = x
        self._P = P
        self._fitted = True
        Pp_next = self.F @ P @ self.F.T + self._Q
        self.posterior_var = float((self.H @ Pp_next @ self.H.T)[0, 0]) + self._R

    def predict(self) -> float | None:
        if not self._fitted or self._x is None:
            return None
        return float((self.H @ (self.F @ self._x))[0])


# == Ensemble Orchestrator =====================================================

class QuantEnsemble:
    """
    Quantitative Prediction Group -- three-model minimum-variance ensemble.

    Public API:
        predict(series, target_date, ensemble_spread) -> {"mean": float, "std": float} | None

    Slots directly into the IVW combiner as source key "local_ensemble".
    """

    DEBIAS_WINDOW:    int   = 14
    STABLE_THRESHOLD: float = 1.2   # degrees C NWP spread; above this -> CHAOTIC regime
    MIN_SERIES_LEN:   int   = 24    # DEBIAS_WINDOW + 10

    def predict(
        self,
        series: list[float],
        target_date: date,
        ensemble_spread: float | None = None,
    ) -> dict | None:
        """
        Args:
            series:          Sequential daily temps, ordered oldest -> newest.
                             series[-1] = yesterday's actual.
                             Minimum length: 24.
            target_date:     The date being forecast (tomorrow).
            ensemble_spread: NWP ensemble spread in degrees C for regime detection.

        Returns:
            {"mean": float, "std": float}  or  None.
        """
        n = len(series)
        if n < self.MIN_SERIES_LEN:
            logger.warning(
                "QuantEnsemble: series too short (%d < %d) -- skipping", n, self.MIN_SERIES_LEN
            )
            return None

        # Day-of-year for every element. series[i] = target_date - (n-1-i) days.
        doys: list[int] = [
            (target_date - timedelta(days=(n - 1 - i))).timetuple().tm_yday
            for i in range(n)
        ]
        target_doy    = target_date.timetuple().tm_yday
        yesterday_doy = doys[-1]

        # -- Step 1: Walk-forward debiasing ------------------------------------
        errs: dict[str, list[float]] = {"ou": [], "ets": [], "kf": []}
        for d in range(n - self.DEBIAS_WINDOW, n):
            tr_s    = series[:d]
            tr_doys = doys[:d]
            actual  = series[d]
            if not tr_s:
                continue
            p_ou  = _quick_ou(tr_s, tr_doys, doys[d], tr_doys[-1])
            p_ets = _quick_holt(tr_s)
            p_kf  = _quick_kalman(tr_s)
            for key, pred in [("ou", p_ou), ("ets", p_ets), ("kf", p_kf)]:
                errs[key].append((pred - actual) if pred is not None else 0.0)

        bias: dict[str, float] = {
            k: float(np.mean(v)) if v else 0.0 for k, v in errs.items()
        }

        # -- Step 2: Full-fit predictions on the complete series ---------------
        ou = OUFourierModel()
        ou.fit(series, doys)
        raw_ou = ou.predict(target_doy, series[-1], yesterday_doy)

        ets = HoltWintersETS()
        ets.fit(series)
        raw_ets = ets.predict()

        kf = KalmanDLM()
        kf.fit(series)
        raw_kf = kf.predict()

        preds: dict[str, float] = {}
        if raw_ou  is not None: preds["ou"]  = raw_ou  - bias["ou"]
        if raw_ets is not None: preds["ets"] = raw_ets - bias["ets"]
        if raw_kf  is not None: preds["kf"]  = raw_kf  - bias["kf"]

        if len(preds) < 2:
            logger.warning("QuantEnsemble: fewer than 2 models succeeded -- skipping")
            return None

        # -- Step 3: Regime classification ------------------------------------
        is_chaotic = (
            ensemble_spread is not None and ensemble_spread >= self.STABLE_THRESHOLD
        )

        # -- Step 4: Error covariance matrix ----------------------------------
        active_keys = [k for k in ["ou", "ets", "kf"] if k in preds]
        E = np.column_stack([
            np.array(errs[k][-self.DEBIAS_WINDOW:], dtype=float) for k in active_keys
        ])
        if E.ndim == 1 or E.shape[1] == 1:
            cov_sample = np.array([[max(0.01, float(np.var(E, ddof=1)))]])
        else:
            cov_sample = np.cov(E.T)

        if is_chaotic:
            # 50% shrinkage toward diagonal:
            # cross-model error correlations are less reliable under high volatility.
            diag_cov   = np.diag(np.diag(cov_sample))
            cov_active = 0.5 * cov_sample + 0.5 * diag_cov
            logger.info(
                "QuantEnsemble: CHAOTIC regime (ens_spread=%.2f degrees C)", ensemble_spread
            )
        else:
            cov_active = cov_sample
            logger.info(
                "QuantEnsemble: STABLE regime (ens_spread=%s)",
                f"{ensemble_spread:.2f} degrees C" if ensemble_spread is not None else "N/A",
            )

        # -- Step 5: Min-variance portfolio weights ---------------------------
        weights = _min_variance_weights(cov_active)

        # -- Step 6: Weighted combination -------------------------------------
        pred_vec      = np.array([preds[k] for k in active_keys])
        combined_mean = float(np.dot(weights, pred_vec))

        kf_var       = kf.posterior_var if kf._fitted else 4.0
        all_errors   = np.concatenate([np.array(errs[k]) for k in active_keys if errs[k]])
        rolling_rmse = float(np.sqrt(np.mean(all_errors ** 2))) if len(all_errors) > 0 else 2.0
        combined_std = max(0.5, math.sqrt(0.6 * kf_var + 0.4 * rolling_rmse ** 2))

        logger.info(
            "QuantEnsemble: mu=%.2f sigma=%.2f  weights={%s}  bias={ou:%.2f ets:%.2f kf:%.2f}",
            combined_mean, combined_std,
            "  ".join(f"{k}:{w:.2f}" for k, w in zip(active_keys, weights)),
            bias["ou"], bias["ets"], bias["kf"],
        )
        return {"mean": combined_mean, "std": combined_std}
