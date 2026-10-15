"""Walk-forward windows, the Deflated Sharpe Ratio, and regime slices.

These are the checks that separate "found a pattern" from "found an edge".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from statarb.backtest.metrics import performance_summary

# --- walk-forward ----------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardWindow:
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


def make_windows(
    dates: pd.DatetimeIndex,
    formation_days: int = 504,
    trading_days: int = 126,
    step_days: int = 126,
) -> list[WalkForwardWindow]:
    """Rolling formation/trading splits over a calendar of bars.

    Pairs (or configs) get chosen on the formation segment and traded on the
    trading segment that follows it, so nothing is ever picked with
    knowledge of the period it trades. Counts are in bars, not calendar
    days. A final stub shorter than half a trading segment is dropped.
    """
    windows = []
    i = 0
    n = len(dates)
    while True:
        f_end = i + formation_days - 1
        t_start = f_end + 1
        t_end = min(t_start + trading_days - 1, n - 1)
        if t_start >= n or (t_end - t_start + 1) < trading_days // 2:
            break
        windows.append(WalkForwardWindow(dates[i], dates[f_end], dates[t_start], dates[t_end]))
        i += step_days
    return windows


# --- deflated sharpe -------------------------------------------------------

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """Expected best Sharpe out of n_trials tries on pure noise (per-period units).

    Extreme-value approximation from Bailey & Lopez de Prado (2014).
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
    """Probability that the observed Sharpe beats the best-of-n_trials noise Sharpe.

    The best in-sample Sharpe out of many tries is biased upward by
    construction. The DSR corrects for that, and for skew and fat tails.
    Above 0.95 means the result is unlikely to be pure selection. The
    classic way to cheat is to undercount n_trials, so count every
    configuration evaluated, not just the ones kept.

    If the variance of Sharpes across trials was not recorded, 1/T (the
    variance of a null Sharpe estimate) is used. That is the lenient choice.
    """
    r = returns.dropna()
    T = len(r)
    if T < 30:
        raise ValueError("too few observations for a meaningful DSR")

    sr = float(r.mean() / r.std())  # per period, not annualized
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))

    var_trials = 1.0 / T if var_sharpe_across_trials is None else var_sharpe_across_trials
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


# --- regimes ---------------------------------------------------------------

def regime_breakdown(returns: pd.Series, regimes: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Performance per named window (name -> (start, end)) plus the full sample.

    Pairs strategies break in liquidity crises (spreads blow through stops)
    and in violent factor rotations (pairs that were secretly one factor bet
    decouple together). This is where you look for that.
    """
    rows = {"full_sample": performance_summary(returns)}
    for name, (start, end) in regimes.items():
        sub = returns.loc[pd.Timestamp(start): pd.Timestamp(end)]
        if len(sub) >= 5:
            rows[name] = performance_summary(sub)
    return pd.DataFrame(rows)
