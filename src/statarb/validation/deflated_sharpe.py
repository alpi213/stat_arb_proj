"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

You tested N pairs and tried M configurations; the best in-sample Sharpe is
upward-biased by construction. The DSR asks: what is the probability the
observed Sharpe exceeds the Sharpe you'd expect from the BEST of N tries
on pure noise?

DSR = Phi( ((SR_hat - SR0) * sqrt(T-1)) /
           sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )

where SR_hat is the observed (per-period) Sharpe, T the number of return
observations, g3/g4 skewness/kurtosis of returns, and SR0 the expected
max Sharpe under the null across n_trials.

Report DSR next to the headline Sharpe. DSR > 0.95 = the result is unlikely
to be pure selection bias. Undercounting n_trials is the classic way to
cheat here — count every configuration you evaluated, not just the ones
you liked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """E[max of n_trials sharpes] under the null (per-period units).

    Extreme-value approximation:
      E[max] ~ sqrt(var) * ((1-g)*Z^{-1}(1-1/n) + g*Z^{-1}(1-1/(n*e)))
    """
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    return float(
        np.sqrt(var_sharpe)
        * (
            (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n)
            + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n * np.e))
        )
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    var_sharpe_across_trials: float | None = None,
) -> dict:
    """Compute PSR against the expected-max-Sharpe benchmark.

    var_sharpe_across_trials: variance of the per-period Sharpes across your
    trials. If you didn't record it (be honest), we default to the variance
    of a null SR estimator, 1/T — a lenient choice; recording the real
    spread across trials is the rigorous path.
    """
    r = returns.dropna()
    T = len(r)
    if T < 30:
        raise ValueError("Too few observations for a meaningful DSR")

    sr = float(r.mean() / r.std())  # per-period, NOT annualized
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))

    var_trials = var_sharpe_across_trials if var_sharpe_across_trials is not None else 1.0 / T
    sr0 = expected_max_sharpe(n_trials, var_trials)

    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    dsr = float(stats.norm.cdf(z))

    return {
        "sharpe_annual": sr * np.sqrt(252),
        "sharpe_per_period": sr,
        "expected_max_sharpe_null": sr0,
        "expected_max_sharpe_null_annual": sr0 * np.sqrt(252),
        "n_trials": n_trials,
        "T": T,
        "skew": g3,
        "kurtosis": g4,
        "dsr": dsr,
        "passes_95pct": dsr > 0.95,
    }
