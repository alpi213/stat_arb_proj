"""Hedge-ratio estimators: periodically refit OLS, and a Kalman filter.

Both take log prices and return a DataFrame with columns alpha, beta and
spread, where

    spread_t = log_y_t - beta_t * log_x_t - alpha_t

The one rule: row t may only use data up to and including t. No smoothing,
no same-bar fitting. The backtester adds its own execution lag on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


class StaticHedge:
    """OLS hedge ratio, refit every `refit_every_days` on all past data.

    This is what a simple production system would actually do, which makes
    it the fair baseline for the Kalman version.
    """

    def __init__(self, refit_every_days: int = 63, min_obs: int = 126):
        self.refit_every_days = refit_every_days
        self.min_obs = min_obs

    def estimate(self, log_y: pd.Series, log_x: pd.Series) -> pd.DataFrame:
        df = pd.concat({"y": log_y, "x": log_x}, axis=1).dropna()
        n = len(df)
        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)

        alpha, beta = np.nan, np.nan
        for i in range(self.min_obs, n):
            if (i - self.min_obs) % self.refit_every_days == 0:
                # Fit on bars 0..i-1, then use the fit from bar i onward.
                past = df.iloc[:i]
                X = sm.add_constant(past["x"].to_numpy())
                fit = sm.OLS(past["y"].to_numpy(), X).fit()
                alpha, beta = float(fit.params[0]), float(fit.params[1])
            alphas[i], betas[i] = alpha, beta

        out = pd.DataFrame({"alpha": alphas, "beta": betas}, index=df.index)
        out["spread"] = df["y"] - out["beta"] * df["x"] - out["alpha"]
        return out


class KalmanHedge:
    """Hedge ratio that drifts: a random-walk state [alpha, beta].

        state:        s_t = s_{t-1} + w_t,          w_t ~ N(0, delta/(1-delta) I)
        observation:  log_y_t = [1, log_x_t] . s_t + v_t,   v_t ~ N(0, obs_var)

    `delta` sets how fast beta may move. Zero recovers static OLS; too large
    and beta chases the very mispricing you want to trade. 1e-5 to 1e-4 is
    the usual range for daily bars.

    `spread_mode` picks what gets traded. Both are standard, and they behave
    differently enough that results for both belong in the writeup:

    - "innovation": the one-step prediction error y_t - E[y_t | t-1]. The
      classic choice, but a well-specified filter drives its innovations
      toward white noise, so z-scores flip sign fast and turnover is high.
    - "residual": y_t - beta_t x_t - alpha_t using the filtered state. Looks
      like the OLS spread, just with a hedge ratio that adapts.
    """

    def __init__(
        self,
        delta: float = 1e-5,
        obs_var: float = 1e-3,
        burn_in: int = 30,
        spread_mode: str = "innovation",
    ):
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if spread_mode not in {"innovation", "residual"}:
            raise ValueError("spread_mode must be 'innovation' or 'residual'")
        self.delta = delta
        self.obs_var = obs_var
        self.burn_in = burn_in
        self.spread_mode = spread_mode

    def estimate(self, log_y: pd.Series, log_x: pd.Series) -> pd.DataFrame:
        df = pd.concat({"y": log_y, "x": log_x}, axis=1).dropna()
        y = df["y"].to_numpy()
        x = df["x"].to_numpy()
        n = len(df)

        trans_cov = self.delta / (1.0 - self.delta) * np.eye(2)
        state = np.array([0.0, 1.0])  # beta near 1 is the natural prior for similar assets
        P = np.eye(2)

        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)
        spreads = np.full(n, np.nan)

        for t in range(n):
            # Predict: state unchanged, uncertainty grows.
            P = P + trans_cov
            H = np.array([1.0, x[t]])

            # Innovation against the prior. This is the tradeable surprise.
            e = y[t] - H @ state
            S = H @ P @ H + self.obs_var

            # Update.
            K = P @ H / S
            state = state + K * e
            P = P - np.outer(K, H @ P)

            if t >= self.burn_in:
                alphas[t] = state[0]
                betas[t] = state[1]
                if self.spread_mode == "innovation":
                    spreads[t] = e
                else:
                    spreads[t] = y[t] - state[1] * x[t] - state[0]

        return pd.DataFrame({"alpha": alphas, "beta": betas, "spread": spreads}, index=df.index)


Hedge = StaticHedge | KalmanHedge


def make_hedge(method: str, **kwargs) -> Hedge:
    if method == "static":
        return StaticHedge(**kwargs)
    if method == "kalman":
        return KalmanHedge(**kwargs)
    raise ValueError(f"unknown hedge method {method!r}")
