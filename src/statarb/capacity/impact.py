"""Capacity analysis: how does net performance degrade with AUM?

Model: square-root market impact (the standard empirical result — see
Almgren et al. 2005, and the "square-root law" literature):

    impact_bps(trade) = k * sigma_daily * sqrt(Q / ADV) * 1e4

where Q is the traded dollar amount, ADV the name's average daily dollar
volume, sigma_daily its daily return vol, and k ~ 0.1-1 an empirical
constant (we default to a conservative-ish 0.1 and report sensitivity
in k — the honest move, since k is the least-known input).

Method: re-price the SAME backtest trade stream at increasing AUM. Signals
and weights don't change (the strategy doesn't know it's bigger); only the
cost per unit of turnover grows with participation. This isolates pure
impact-driven decay. At each AUM:

    per-name daily impact drag = sum_i k * sigma_i * sqrt(|dW_i| * AUM / ADV_i)
                                   * (|dW_i| * AUM) / AUM        [in return space]
                               = k * sigma_i * |dW_i|^{3/2} * (AUM / ADV_i)^{1/2}

The capacity estimate is where net Sharpe (or net CAGR) hits zero — plus
the AUM where Sharpe halves, which is often the more decision-relevant
number for an allocator.

Assumptions to state in the writeup:
- whole-day participation (no intraday scheduling — Almgren-Chriss optimal
  execution would trade this off against timing risk);
- impact is fully paid, none recovered (permanent+temporary lumped —
  conservative);
- ADV computed on a trailing window, point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sqrt_law_impact_bps(
    trade_dollars: float, adv_dollars: float, sigma_daily: float, k: float = 0.1
) -> float:
    """One-shot impact in bps for a single trade in a single name."""
    if adv_dollars <= 0 or trade_dollars <= 0:
        return 0.0
    return k * sigma_daily * np.sqrt(trade_dollars / adv_dollars) * 1e4


def daily_impact_drag(
    weight_changes: pd.DataFrame,   # |dW| per name per day (fraction of AUM)
    adv_dollars: pd.DataFrame,      # trailing ADV per name per day ($)
    sigma_daily: pd.DataFrame,      # trailing daily vol per name per day
    aum: float,
    k: float = 0.1,
) -> pd.Series:
    """Impact cost in RETURN space (fraction of AUM) per day."""
    dw = weight_changes.abs()
    adv = adv_dollars.reindex_like(dw)
    sig = sigma_daily.reindex_like(dw)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_name = k * sig * dw.pow(1.5) * np.sqrt(aum / adv)
    return per_name.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1)


def capacity_curve(
    base_returns: pd.Series,
    weight_changes: pd.DataFrame,
    adv_dollars: pd.DataFrame,
    sigma_daily: pd.DataFrame,
    aum_grid: list[float],
    k: float = 0.1,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Net Sharpe / CAGR as a function of AUM.

    base_returns must already include commissions/spread/borrow (which are
    roughly size-independent in this range); impact is added on top here.

    `periods_per_year` must match the bar frequency of the inputs (252 daily,
    1095 for 8h crypto settlements) — otherwise both the annualized Sharpe and
    the reported drag are rescaled by sqrt of the ratio.

    NOTE: this curve is only interpretable for a strategy with positive base
    Sharpe. Applied to a no-edge strategy it will report "capacity" at the
    smallest AUM on the grid, which means nothing.
    """
    from statarb.backtest.metrics import cagr, sharpe_ratio

    rows = []
    for aum in aum_grid:
        drag = daily_impact_drag(weight_changes, adv_dollars, sigma_daily, aum, k)
        net = base_returns - drag
        equity = (1 + net).cumprod()
        rows.append(
            {
                "aum": aum,
                "sharpe": sharpe_ratio(net, periods_per_year=periods_per_year),
                "cagr": cagr(equity, periods_per_year=periods_per_year),
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
    """Log-linear interpolation of the AUM where Sharpe first drops to `level`."""
    s = sharpe_by_aum.dropna()
    below = s[s <= level]
    if below.empty:
        return None
    hi = below.index[0]
    pos = s.index.get_loc(hi)
    if pos == 0:
        return float(hi)
    lo = s.index[pos - 1]
    # interpolate in log-AUM space
    f = (s[lo] - level) / (s[lo] - s[hi])
    return float(np.exp(np.log(lo) + f * (np.log(hi) - np.log(lo))))


def trailing_adv_dollars(
    close: pd.DataFrame, volume: pd.DataFrame, window: int = 21
) -> pd.DataFrame:
    """Point-in-time trailing average daily dollar volume (uses UNADJUSTED close)."""
    return (close * volume).rolling(window, min_periods=window).mean()


def trailing_sigma(adj_close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return adj_close.pct_change(fill_method=None).rolling(window, min_periods=window).std()
