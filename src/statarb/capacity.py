"""Capacity: how fast does net performance decay as AUM grows?

Square-root market impact, the standard empirical form:

    impact = k * sigma_daily * sqrt(Q / ADV)

Q is dollars traded, ADV the name's average daily dollar volume, and k an
empirical constant somewhere in 0.1 to 1. k is the least-known input, so
it is set a priori (default 0.1) and sensitivity in k is reported.

The method is to re-price the same trade stream at each AUM. Signals and
weights don't change, only the cost per unit of turnover grows. In return
space the per-name drag is

    k * sigma * |dW|^1.5 * sqrt(AUM / ADV)

Assumptions: whole-day participation, impact fully paid and never
recovered, ADV from a trailing window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.backtest.metrics import cagr, sharpe_ratio

TRADING_DAYS = 252


def daily_impact_drag(
    weight_changes: pd.DataFrame,  # |dW| per name per bar, as a fraction of AUM
    adv_dollars: pd.DataFrame,     # trailing ADV per name per bar, in dollars
    sigma: pd.DataFrame,           # trailing per-bar return vol per name
    aum: float,
    k: float = 0.1,
) -> pd.Series:
    """Impact cost per bar, as a fraction of AUM."""
    dw = weight_changes.abs()
    adv = adv_dollars.reindex_like(dw)
    sig = sigma.reindex_like(dw)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_name = k * sig * dw.pow(1.5) * np.sqrt(aum / adv)
    return per_name.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1)


def capacity_curve(
    base_returns: pd.Series,
    weight_changes: pd.DataFrame,
    adv_dollars: pd.DataFrame,
    sigma: pd.DataFrame,
    aum_grid: list[float],
    k: float = 0.1,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Net Sharpe and CAGR at each AUM in the grid.

    `base_returns` should already be net of commissions, spread and borrow,
    which barely depend on size; impact is added here. `periods_per_year`
    must match the bar frequency (252 daily, 1095 for 8h bars).

    The AUM where Sharpe halves and where it hits zero are stored in
    `df.attrs`. Only meaningful when the base Sharpe is positive.
    """
    rows = []
    for aum in aum_grid:
        drag = daily_impact_drag(weight_changes, adv_dollars, sigma, aum, k)
        net = base_returns - drag
        rows.append(
            {
                "aum": aum,
                "sharpe": sharpe_ratio(net, periods_per_year=periods_per_year),
                "cagr": cagr((1 + net).cumprod(), periods_per_year=periods_per_year),
                "annual_impact_drag": float(drag.mean() * periods_per_year),
                "avg_participation": float(
                    (weight_changes.abs() * aum / adv_dollars).mean().mean()
                ),
            }
        )
    df = pd.DataFrame(rows).set_index("aum")

    base_sharpe = df["sharpe"].iloc[0]
    df.attrs["aum_sharpe_zero"] = _first_crossing(df["sharpe"], 0.0)
    df.attrs["aum_sharpe_half"] = _first_crossing(df["sharpe"], base_sharpe / 2)
    return df


def _first_crossing(sharpe_by_aum: pd.Series, level: float) -> float | None:
    """AUM where Sharpe first drops to `level`, interpolated in log-AUM."""
    s = sharpe_by_aum.dropna()
    below = s[s <= level]
    if below.empty:
        return None
    hi = below.index[0]
    pos = s.index.get_loc(hi)
    if pos == 0:
        return float(hi)
    lo = s.index[pos - 1]
    f = (s[lo] - level) / (s[lo] - s[hi])
    return float(np.exp(np.log(lo) + f * (np.log(hi) - np.log(lo))))


def trailing_adv_dollars(
    close: pd.DataFrame, volume: pd.DataFrame, window: int = 21
) -> pd.DataFrame:
    """Trailing average daily dollar volume. Uses the unadjusted close on purpose."""
    return (close * volume).rolling(window, min_periods=window).mean()


def trailing_sigma(adj_close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return adj_close.pct_change(fill_method=None).rolling(window, min_periods=window).std()
