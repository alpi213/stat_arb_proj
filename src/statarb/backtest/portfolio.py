"""Generic cross-sectional weights backtester.

Complements `backtest.engine` (which is pair-specific). Anything that can be
expressed as "target weights per instrument per period" runs through here:
crypto carry, cross-sectional momentum, factor tilts.

Contract (mirrors the equity engine's discipline):
    weights.loc[t]        = position HELD over (t, t+1]
    asset_returns.loc[t]  = return EARNED over (t, t+1]

Both are indexed at t, so they multiply elementwise with no shifting inside
this function. Callers are responsible for building `asset_returns` with the
forward-looking convention (see crypto.funding.carry_returns) — this keeps
exactly one place in the codebase where a lag can be wrong, instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossSectionalCostModel:
    """Costs for a two-legged delta-neutral position (e.g. spot + perp).

    Each unit of |weight change| requires trading BOTH legs, so per-unit
    turnover cost is the sum across legs. Defaults are Binance retail taker
    fees, which is the conservative assumption; a VIP tier or maker rebates
    would improve this materially and should be sensitivity-tested.
    """

    leg1_fee_bps: float = 4.0        # USD-M futures taker
    leg2_fee_bps: float = 7.5        # spot taker
    leg1_half_spread_bps: float = 1.0
    leg2_half_spread_bps: float = 1.0
    borrow_annual_bps: float = 0.0   # for short-spot legs, if used
    periods_per_year: int = 1095     # 8h settlements

    @property
    def per_turnover_cost(self) -> float:
        return (
            self.leg1_fee_bps + self.leg2_fee_bps
            + self.leg1_half_spread_bps + self.leg2_half_spread_bps
        ) * 1e-4

    def trading_cost(self, turnover: pd.Series) -> pd.Series:
        return turnover * self.per_turnover_cost

    def carry_cost(self, short_weight: pd.Series) -> pd.Series:
        rate = self.borrow_annual_bps * 1e-4 / self.periods_per_year
        return short_weight * rate


def backtest_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    costs: CrossSectionalCostModel | None = None,
    initial_capital: float = 1_000_000,
    max_gross_leverage: float = 1.0,
) -> dict:
    """Run a weights-based cross-sectional backtest.

    Returns dict with: returns (net), gross_returns, equity_curve, turnover,
    gross_exposure, per_asset_pnl, weights (post-leverage-cap).
    """
    costs = costs or CrossSectionalCostModel()

    idx = weights.index.intersection(asset_returns.index)
    cols = weights.columns.intersection(asset_returns.columns)
    w = weights.loc[idx, cols].fillna(0.0)
    r = asset_returns.loc[idx, cols]

    # A weight on an asset with no return observation is unfillable — zero it
    # rather than silently treating a NaN return as 0% (which would credit a
    # position we could not have held).
    w = w.where(r.notna(), 0.0)
    r = r.fillna(0.0)

    gross_exp = w.abs().sum(axis=1)
    scale = (max_gross_leverage / gross_exp).clip(upper=1.0).replace([np.inf], 1.0).fillna(1.0)
    w = w.mul(scale, axis=0)

    per_asset = w * r
    gross = per_asset.sum(axis=1)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()  # initial build-out is a real cost
    short_w = w.clip(upper=0.0).abs().sum(axis=1)

    net = gross - costs.trading_cost(turnover) - costs.carry_cost(short_w)
    equity = initial_capital * (1 + net).cumprod()

    return {
        "returns": net,
        "gross_returns": gross,
        "equity_curve": equity,
        "turnover": turnover,
        "gross_exposure": w.abs().sum(axis=1),
        "per_asset_pnl": per_asset,
        "weights": w,
    }
