"""Cointegration tests and mean-reversion diagnostics.

Engle-Granger two-step:
    1. OLS: log(y) = alpha + beta * log(x) + eps
    2. ADF test on the residual eps. If eps is stationary, x and y are
       cointegrated and eps is the tradeable spread.

We work in LOG prices: the hedge ratio is then interpretable as an
elasticity, the spread is symmetric in percentage terms, and position
sizing in dollars follows naturally.

Caveats encoded here:
- The EG test is asymmetric (regressing y on x vs x on y can disagree);
  we test both directions and keep the stronger one.
- The ADF critical values for a *residual* are not the plain ADF ones
  (parameter estimation shifts the distribution); statsmodels' `coint`
  uses the correct MacKinnon surfaces, so we use it rather than rolling
  our own ADF on residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class CointResult:
    y: str                 # dependent leg (long 1 unit when spread is low)
    x: str                 # hedge leg
    alpha: float           # OLS intercept
    beta: float            # static hedge ratio (log-price regression coef)
    pvalue: float          # EG cointegration p-value (MacKinnon)
    coint_t: float         # EG test statistic
    half_life_days: float  # OU half-life of the residual
    spread_std: float      # residual std over the formation window


def engle_granger(log_y: pd.Series, log_x: pd.Series) -> CointResult:
    """Run EG in both directions, return the direction with lower p-value."""
    df = pd.concat({"y": log_y, "x": log_x}, axis=1).dropna()
    if len(df) < 100:
        raise ValueError(f"Too few overlapping observations: {len(df)}")

    best: CointResult | None = None
    for a_name, b_name in ((log_y.name, log_x.name), (log_x.name, log_y.name)):
        a = df["y"] if a_name == log_y.name else df["x"]
        b = df["x"] if a_name == log_y.name else df["y"]
        t_stat, pvalue, _ = coint(a, b)
        X = sm.add_constant(b.to_numpy())
        ols = sm.OLS(a.to_numpy(), X).fit()
        resid = pd.Series(ols.resid, index=df.index)
        result = CointResult(
            y=str(a_name),
            x=str(b_name),
            alpha=float(ols.params[0]),
            beta=float(ols.params[1]),
            pvalue=float(pvalue),
            coint_t=float(t_stat),
            half_life_days=half_life(resid),
            spread_std=float(resid.std()),
        )
        if best is None or result.pvalue < best.pvalue:
            best = result
    assert best is not None
    return best


def half_life(spread: pd.Series) -> float:
    """OU half-life via the AR(1) discretization.

    d(spread)_t = theta * spread_{t-1} + noise  =>  HL = -ln(2)/ln(1+theta).
    A negative theta (mean reversion) gives a positive finite half-life;
    theta >= 0 means no reversion — return +inf so filters reject it.
    """
    s = spread.dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    X = sm.add_constant(lag.to_numpy())
    theta = sm.OLS(ds.to_numpy(), X).fit().params[1]
    if theta >= 0:
        return float("inf")
    return float(-np.log(2.0) / np.log(1.0 + theta))


def johansen_test(log_prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    """Johansen trace test — robustness check on top of EG.

    Symmetric in the assets (no y-vs-x choice) and extends to baskets of
    3+ names. Returns the statsmodels result object; the first column of
    `evec` is the cointegrating vector, comparable to [1, -beta].
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    clean = log_prices.dropna()
    return coint_johansen(clean, det_order=det_order, k_ar_diff=k_ar_diff)
