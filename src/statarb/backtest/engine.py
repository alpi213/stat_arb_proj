"""Backtest engine.

Look-ahead discipline, stated once and enforced everywhere:

    signal computed from close(t)  ->  position held from close(t + lag)

with lag >= 1 enforced by config validation. Concretely, the weight applied
to the return over (t-1, t] is the weight decided at close(t - lag)…(t-1),
i.e. `target_weights.shift(lag)`.

Position sizing: dollar-neutral-ish per pair. For hedge ratio beta (log-log
regression coefficient), $1 of y is hedged by $beta of x. With a gross
budget g per pair, weights are:

    w_y = sign * g / (1 + beta),   w_x = -sign * beta * g / (1 + beta)

so |w_y| + |w_x| = g. When beta != 1 the pair carries a small net dollar
exposure; residual beta-neutralization is listed as an extension in
docs/LIMITATIONS.md.

Weights are held constant while in a position (no daily re-hedging as beta
drifts): re-hedging every bar would churn turnover and drown the edge in
costs. The hedge ratio used at entry is frozen for the life of the trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.backtest.costs import CostModel
from statarb.hedge.base import HedgeEstimator
from statarb.signals.zscore import generate_positions, rolling_zscore

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    pair: str
    side: str            # "long_spread" | "short_spread"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    exit_reason: str
    holding_days: int
    pnl_return: float    # cumulative return contribution on portfolio capital


@dataclass
class PairBacktestResult:
    pair: str
    daily_returns: pd.Series          # net, as fraction of PORTFOLIO capital
    gross_returns: pd.Series          # before costs
    weights: pd.DataFrame             # columns: [y_ticker, x_ticker], lagged (as held)
    zscore: pd.Series
    spread: pd.Series
    beta: pd.Series
    trades: list[Trade] = field(default_factory=list)
    events: pd.Series | None = None


def backtest_pair(
    adj_close: pd.DataFrame,
    y: str,
    x: str,
    hedge: HedgeEstimator,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_days: int = 40,
    gross_per_pair: float = 0.10,
    execution_lag_bars: int = 1,
    costs: CostModel | None = None,
    trade_start: pd.Timestamp | None = None,
    forced_exit_date: pd.Timestamp | None = None,
) -> PairBacktestResult:
    """Backtest one pair.

    trade_start:      first date positions may be held (formation/trading
                      split — signals before this date are zeroed).
    forced_exit_date: liquidate on/after this date (delisting or end of a
                      walk-forward trading window).
    """
    costs = costs or CostModel()
    px = adj_close[[y, x]].dropna()
    log_y, log_x = np.log(px[y]), np.log(px[x])

    hed = hedge.estimate(log_y, log_x)
    z = rolling_zscore(hed["spread"], window=zscore_window)
    sig = generate_positions(
        z, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, max_holding_days=max_holding_days
    )

    pos = sig["position"].copy()
    if trade_start is not None:
        pos.loc[: pd.Timestamp(trade_start)] = 0.0
    if forced_exit_date is not None:
        pos.loc[pd.Timestamp(forced_exit_date):] = 0.0

    # Freeze the hedge ratio at entry for the life of each trade.
    beta_used = hed["beta"].copy()
    in_pos = pos != 0
    entry_flags = in_pos & ~in_pos.shift(1, fill_value=False)
    beta_frozen = beta_used.where(entry_flags).ffill()
    beta_eff = beta_frozen.where(in_pos)

    denom = 1.0 + beta_eff.clip(lower=0.0)
    w_y = pos * gross_per_pair / denom
    w_x = -pos * gross_per_pair * beta_eff / denom
    target = pd.DataFrame({y: w_y, x: w_x}, index=px.index).fillna(0.0)

    # THE look-ahead guard: weights held over (t-1, t] were decided at t-lag.
    held = target.shift(execution_lag_bars).fillna(0.0)

    # fill_method=None: propagate real gaps as NaN rather than silently
    # forward-filling a stale price into a fabricated 0% return (see
    # crypto/funding.py for the version-dependent bug this avoids). The
    # explicit .fillna(0.0) below is the deliberate policy for any gap
    # that survives the upstream quality filters: no data, no return.
    rets = px.pct_change(fill_method=None).fillna(0.0)
    gross = (held * rets).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    short_weight = held.clip(upper=0.0).abs().sum(axis=1)
    net = gross - costs.trading_cost(turnover) - costs.borrow_cost(short_weight)

    trades = _extract_trades(f"{y}/{x}", pos, sig["event"], net, execution_lag_bars)

    return PairBacktestResult(
        pair=f"{y}/{x}",
        daily_returns=net,
        gross_returns=gross,
        weights=held,
        zscore=z,
        spread=hed["spread"],
        beta=hed["beta"],
        trades=trades,
        events=sig["event"],
    )


def _extract_trades(
    pair: str, pos: pd.Series, events: pd.Series, net: pd.Series, lag: int
) -> list[Trade]:
    trades: list[Trade] = []
    entry_date: pd.Timestamp | None = None
    side = ""
    prev = 0.0
    for date, p in pos.items():
        if prev == 0 and p != 0:
            entry_date, side = date, ("long_spread" if p > 0 else "short_spread")
        elif prev != 0 and p == 0 and entry_date is not None:
            # P&L accrues on held weights, which lag decisions by `lag` bars.
            window = net.loc[entry_date:date]
            pnl = float((1 + window).prod() - 1) if len(window) else 0.0
            trades.append(
                Trade(
                    pair=pair,
                    side=side,
                    entry_date=entry_date,
                    exit_date=date,
                    exit_reason=str(events.loc[date]) or "forced",
                    holding_days=int(pos.index.get_loc(date) - pos.index.get_loc(entry_date)),
                    pnl_return=pnl,
                )
            )
            entry_date = None
        prev = p
    if entry_date is not None:  # still open at end of sample
        window = net.loc[entry_date:]
        trades.append(
            Trade(
                pair=pair, side=side, entry_date=entry_date, exit_date=None,
                exit_reason="open_at_end",
                holding_days=len(window),
                pnl_return=float((1 + window).prod() - 1),
            )
        )
    return trades


def backtest_portfolio(
    pair_results: list[PairBacktestResult],
    initial_capital: float = 1_000_000,
    max_gross_leverage: float = 2.0,
) -> dict:
    """Aggregate pair-level streams into a portfolio.

    Pair returns are already expressed as fractions of total capital (each
    pair was budgeted `gross_per_pair`), so aggregation is a sum. If summed
    gross exposure ever exceeds the leverage cap, all pairs are scaled down
    proportionally that day (a simple, honest capacity-of-leverage rule).
    """
    if not pair_results:
        raise ValueError("no pair results to aggregate")

    idx = pair_results[0].daily_returns.index
    for r in pair_results[1:]:
        idx = idx.union(r.daily_returns.index)

    rets = pd.DataFrame(
        {r.pair: r.daily_returns.reindex(idx).fillna(0.0) for r in pair_results}
    )
    gross_exp = pd.DataFrame(
        {r.pair: r.weights.abs().sum(axis=1).reindex(idx).fillna(0.0) for r in pair_results}
    )
    total_gross = gross_exp.sum(axis=1)
    scale = (max_gross_leverage / total_gross).clip(upper=1.0).fillna(1.0)

    port_ret = (rets.mul(scale, axis=0)).sum(axis=1)
    equity = initial_capital * (1 + port_ret).cumprod()

    all_trades = [t for r in pair_results for t in r.trades]
    return {
        "returns": port_ret,
        "equity_curve": equity,
        "per_pair_returns": rets,
        "gross_exposure": total_gross,
        "leverage_scale": scale,
        "trades": all_trades,
    }
