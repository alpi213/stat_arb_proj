"""Static (rolling-refit) OLS hedge ratio — the baseline.

Not literally one regression for all time: the ratio is refit every
`refit_every_days` on an expanding window of PAST data only, then held
constant until the next refit. This is what a simple production
implementation would actually do, and it is the honest comparator for
the Kalman upgrade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statarb.hedge.base import HedgeEstimator


class StaticHedge(HedgeEstimator):
    def __init__(self, refit_every_days: int = 63, min_obs: int = 126):
        self.refit_every_days = refit_every_days
        self.min_obs = min_obs

    def estimate(self, log_y: pd.Series, log_x: pd.Series) -> pd.DataFrame:
        df = pd.concat({"y": log_y, "x": log_x}, axis=1).dropna()
        n = len(df)
        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)

        cur_alpha, cur_beta = np.nan, np.nan
        for i in range(self.min_obs, n):
            if (i - self.min_obs) % self.refit_every_days == 0:
                # Fit on data strictly up to and including bar i-1 …
                past = df.iloc[:i]
                X = sm.add_constant(past["x"].to_numpy())
                fit = sm.OLS(past["y"].to_numpy(), X).fit()
                cur_alpha, cur_beta = float(fit.params[0]), float(fit.params[1])
            # … and apply it from bar i onward: no same-bar fitting.
            alphas[i], betas[i] = cur_alpha, cur_beta

        out = pd.DataFrame({"alpha": alphas, "beta": betas}, index=df.index)
        out["spread"] = df["y"] - out["beta"] * df["x"] - out["alpha"]
        return out
