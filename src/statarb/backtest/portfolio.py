"""Weights-based backtester for anything cross-sectional (crypto carry here).

Convention, mirroring the pair engine:

    weights.loc[t]        position held over (t, t+1]
    asset_returns.loc[t]  return earned over (t, t+1]

Both indexed at t, so they multiply with no shifting here. The caller
builds `asset_returns` forward-looking (see crypto.funding.carry_returns).
That keeps the one place a lag can go wrong to exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossSectionalCostModel:
    """Costs for a two-legged position (spot + perp). Both legs trade on every rebalance.

    Defaults are Binance retail taker fees, the conservative case. VIP tiers
    and maker fills are much cheaper; the backtest script sweeps that.
    """

    leg1_fee_bps: float = 4.0         # USD-M futures taker
    leg2_fee_bps: float = 7.5         # spot taker
    leg1_half_spread_bps: float = 1.0
    leg2_half_spread_bps: float = 1.0

    def trading_cost(self, turnover: pd.Series) -> pd.Series:
        per_unit = (
            self.leg1_fee_bps + self.leg2_fee_bps
            + self.leg1_half_spread_bps + self.leg2_half_spread_bps
        ) * 1e-4
        return turnover * per_unit


def backtest_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    costs: CrossSectionalCostModel | None = None,
    initial_capital: float = 1_000_000,
    max_gross_leverage: float = 1.0,
) -> dict:
    """Returns a dict with returns (net), gross_returns, equity_curve,
    turnover, gross_exposure, per_asset_pnl and the weights actually used."""
    costs = costs or CrossSectionalCostModel()

    idx = weights.index.intersection(asset_returns.index)
    cols = weights.columns.intersection(asset_returns.columns)
    w = weights.loc[idx, cols].fillna(0.0)
    r = asset_returns.loc[idx, cols]

    # No return observation means the position could not have been held.
    # Zero the weight rather than crediting a NaN as 0%.
    w = w.where(r.notna(), 0.0)
    r = r.fillna(0.0)

    gross_exp = w.abs().sum(axis=1)
    scale = (max_gross_leverage / gross_exp).clip(upper=1.0).replace([np.inf], 1.0).fillna(1.0)
    w = w.mul(scale, axis=0)

    per_asset = w * r
    gross = per_asset.sum(axis=1)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()  # building the first book costs money too

    net = gross - costs.trading_cost(turnover)
    return {
        "returns": net,
        "gross_returns": gross,
        "equity_curve": initial_capital * (1 + net).cumprod(),
        "turnover": turnover,
        "gross_exposure": w.abs().sum(axis=1),
        "per_asset_pnl": per_asset,
        "weights": w,
    }
