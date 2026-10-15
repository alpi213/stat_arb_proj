"""Pair backtester.

Timing rule, applied in exactly one place:

    signal from close(t)  ->  position held from close(t + lag),  lag >= 1

so the weight earning the return over (t-1, t] is `target.shift(lag)`.

Sizing: for hedge ratio beta, $1 of y is hedged with $beta of x. With a
gross budget g per pair,

    w_y =  sign * g / (1 + beta),    w_x = -sign * beta * g / (1 + beta)

so |w_y| + |w_x| = g. Beta is frozen at entry for the life of the trade, so
rehedging every bar would churn turnover and drown the edge in costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.hedge import Hedge
from statarb.signals import generate_positions, rolling_zscore

TRADING_DAYS = 252


@dataclass
class CostModel:
    """Costs in return space (fractions of capital), matching the weights.

    Commission and half-spread are paid per side on traded notional. Borrow
    accrues daily on the short leg. Half the book is always short in a pairs
    strategy, so leaving it out flatters the Sharpe.
    """

    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    borrow_fee_annual_bps: float = 50.0

    def trading_cost(self, turnover: pd.Series) -> pd.Series:
        """`turnover` is the sum of |weight changes| that bar."""
        per_side = (self.commission_bps + self.half_spread_bps) * 1e-4
        return turnover * per_side

    def borrow_cost(self, short_weight: pd.Series) -> pd.Series:
        """`short_weight` is the sum of |negative weights| held that bar."""
        daily_rate = self.borrow_fee_annual_bps * 1e-4 / TRADING_DAYS
        return short_weight * daily_rate


@dataclass
class Trade:
    pair: str
    side: str  # long_spread | short_spread
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    exit_reason: str
    holding_days: int
    pnl_return: float  # compounded return on portfolio capital


@dataclass
class PairBacktestResult:
    pair: str
    daily_returns: pd.Series   # net of costs, fraction of portfolio capital
    gross_returns: pd.Series
    weights: pd.DataFrame      # as held (already lagged); columns [y, x]
    zscore: pd.Series
    spread: pd.Series
    beta: pd.Series
    trades: list[Trade] = field(default_factory=list)
    events: pd.Series | None = None


def backtest_pair(
    adj_close: pd.DataFrame,
    y: str,
    x: str,
    hedge: Hedge,
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

    `trade_start` zeroes positions before that date (the hedge may be
    estimated on earlier data, but nothing is held). `forced_exit_date`
    flattens from that date on (end of a walk-forward segment).
    """
    costs = costs or CostModel()
    px = adj_close[[y, x]].dropna()
    log_y, log_x = np.log(px[y]), np.log(px[x])

    hed = hedge.estimate(log_y, log_x)
    z = rolling_zscore(hed["spread"], window=zscore_window)
    sig = generate_positions(z, entry_z, exit_z, stop_z, max_holding_days)

    pos = sig["position"].copy()
    if trade_start is not None:
        pos.loc[: pd.Timestamp(trade_start)] = 0.0
    if forced_exit_date is not None:
        pos.loc[pd.Timestamp(forced_exit_date):] = 0.0

    # Beta at entry, held for the life of the trade.
    in_pos = pos != 0
    entered = in_pos & ~in_pos.shift(1, fill_value=False)
    beta = hed["beta"].where(entered).ffill().where(in_pos)

    denom = 1.0 + beta.clip(lower=0.0)
    target = pd.DataFrame(
        {y: pos * gross_per_pair / denom, x: -pos * gross_per_pair * beta / denom},
        index=px.index,
    ).fillna(0.0)

    # The look-ahead guard. Weights held over (t-1, t] were decided at t-lag.
    held = target.shift(execution_lag_bars).fillna(0.0)

    # fill_method=None so a real gap stays NaN instead of becoming a fake 0%
    # return from a stale price. Gaps that survive cleaning earn nothing.
    rets = px.pct_change(fill_method=None).fillna(0.0)
    gross = (held * rets).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    short_weight = held.clip(upper=0.0).abs().sum(axis=1)
    net = gross - costs.trading_cost(turnover) - costs.borrow_cost(short_weight)

    return PairBacktestResult(
        pair=f"{y}/{x}",
        daily_returns=net,
        gross_returns=gross,
        weights=held,
        zscore=z,
        spread=hed["spread"],
        beta=hed["beta"],
        trades=_extract_trades(f"{y}/{x}", pos, sig["event"], net),
        events=sig["event"],
    )


def _extract_trades(pair: str, pos: pd.Series, events: pd.Series, net: pd.Series) -> list[Trade]:
    trades: list[Trade] = []
    entry: pd.Timestamp | None = None
    side = ""
    prev = 0.0
    for date, p in pos.items():
        if prev == 0 and p != 0:
            entry, side = date, ("long_spread" if p > 0 else "short_spread")
        elif prev != 0 and p == 0 and entry is not None:
            window = net.loc[entry:date]
            trades.append(Trade(
                pair=pair, side=side, entry_date=entry, exit_date=date,
                exit_reason=str(events.loc[date]) or "forced",
                holding_days=int(pos.index.get_loc(date) - pos.index.get_loc(entry)),
                pnl_return=float((1 + window).prod() - 1) if len(window) else 0.0,
            ))
            entry = None
        prev = p
    if entry is not None:  # still open at the end of the sample
        window = net.loc[entry:]
        trades.append(Trade(
            pair=pair, side=side, entry_date=entry, exit_date=None,
            exit_reason="open_at_end", holding_days=len(window),
            pnl_return=float((1 + window).prod() - 1),
        ))
    return trades


def backtest_portfolio(
    pair_results: list[PairBacktestResult],
    initial_capital: float = 1_000_000,
    max_gross_leverage: float = 2.0,
) -> dict:
    """Sum pair streams into a portfolio.

    Each pair's returns are already a fraction of total capital, so the
    portfolio return is a sum. On any day total gross exposure exceeds the
    leverage cap, every pair is scaled down proportionally.
    """
    if not pair_results:
        raise ValueError("no pair results to aggregate")

    idx = pair_results[0].daily_returns.index
    for r in pair_results[1:]:
        idx = idx.union(r.daily_returns.index)

    rets = pd.DataFrame({r.pair: r.daily_returns.reindex(idx).fillna(0.0) for r in pair_results})
    gross_exp = pd.DataFrame(
        {r.pair: r.weights.abs().sum(axis=1).reindex(idx).fillna(0.0) for r in pair_results}
    )
    total_gross = gross_exp.sum(axis=1)
    scale = (max_gross_leverage / total_gross).clip(upper=1.0).fillna(1.0)

    port_ret = rets.mul(scale, axis=0).sum(axis=1)
    return {
        "returns": port_ret,
        "equity_curve": initial_capital * (1 + port_ret).cumprod(),
        "per_pair_returns": rets,
        "gross_exposure": total_gross,
        "leverage_scale": scale,
        "trades": [t for r in pair_results for t in r.trades],
        "pair_results": pair_results,
    }
