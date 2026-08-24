"""Kalman-filter dynamic hedge ratio.

State-space model:

    state:        s_t = [alpha_t, beta_t]',   s_t = s_{t-1} + w_t   (random walk)
    observation:  log_y_t = [1, log_x_t] . s_t + v_t

    w_t ~ N(0, delta/(1-delta) * I)   (process noise — how fast beta may drift)
    v_t ~ N(0, obs_var)               (observation noise)

`delta` is THE hyperparameter: 0 recovers static OLS; too large and beta
chases noise, absorbing the very mispricing you want to trade. Values around
1e-5..1e-4 are standard for daily bars. Sensitivity to delta belongs in the
validation report, not swept under the rug.

Implementation notes:
- Filtered (not smoothed) estimates only: the smoother uses future data and
  is look-ahead by construction.
- The spread at t uses the PREDICTED state (prior) rather than the updated
  (posterior) state. The posterior at t has already "seen" y_t, which shrinks
  the innovation you trade on. The prior innovation e_t = y_t - y_hat_t|t-1
  is the classic tradeable signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.hedge.base import HedgeEstimator


class KalmanHedge(HedgeEstimator):
    """Dynamic hedge ratio via Kalman filter.

    `spread_mode` selects which quantity is traded — a genuine specification
    choice, not a tuning knob, and both are standard in the literature:

    - "innovation": the one-step prediction error e_t = y_t - E[y_t | t-1].
      This is the classic formulation (Chan, *Algorithmic Trading*). Caveat:
      a well-specified filter drives its own innovations toward white noise,
      so the z-score of e_t flips sign quickly and produces short holding
      periods and high turnover.
    - "residual": the level spread y_t - beta_t*x_t - alpha_t built from the
      FILTERED state. Persistent and mean-reverting like the classic OLS
      spread, but with a hedge ratio that adapts over time.

    Both use only information available at t (filtered, never smoothed).
    Report results for both; do not silently keep whichever wins.
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
        # Diffuse-ish prior centered at [0, 1]: beta near 1 is the natural
        # prior for log prices of economically similar assets.
        state = np.array([0.0, 1.0])
        P = np.eye(2) * 1.0

        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)
        spreads = np.full(n, np.nan)
        innovation_vars = np.full(n, np.nan)

        for t in range(n):
            # Predict (random walk: state unchanged, covariance grows)
            P = P + trans_cov
            H = np.array([1.0, x[t]])

            # Innovation from the PRIOR — this is the tradeable spread
            y_hat = H @ state
            e = y[t] - y_hat
            S = H @ P @ H + self.obs_var

            # Update
            K = P @ H / S
            state = state + K * e
            P = P - np.outer(K, H @ P)

            if t >= self.burn_in:
                alphas[t] = state[0]
                betas[t] = state[1]
                # Both modes use only data up to t (state is the FILTERED
                # posterior at t, not a smoothed estimate).
                spreads[t] = e if self.spread_mode == "innovation" else (
                    y[t] - state[1] * x[t] - state[0]
                )
                innovation_vars[t] = S

        out = pd.DataFrame(
            {
                "alpha": alphas,
                "beta": betas,
                "spread": spreads,
                # sqrt(S) is a model-implied spread vol — an alternative to a
                # rolling z-score denominator; exposed for experimentation.
                "innovation_std": np.sqrt(innovation_vars),
            },
            index=df.index,
        )
        return out
