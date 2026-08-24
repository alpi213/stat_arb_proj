"""Transaction and holding cost models.

Costs enter the backtest in return space (fractions of portfolio capital),
matching how positions are expressed as capital weights.

Three components:
- commission_bps:      per side, per traded notional
- half_spread_bps:     you cross half the bid-ask every time you trade
- borrow_fee:          annual bps charged on short notional, accrued daily —
                       half the book is always short in a pairs strategy, so
                       omitting this flatters the Sharpe materially.

Extension point: make half_spread_bps per-name (small caps are wider) by
passing a Series instead of a scalar — the interface accepts either.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252


@dataclass
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    borrow_fee_annual_bps: float = 50.0

    @property
    def per_side_cost(self) -> float:
        """Fractional cost per unit of traded notional."""
        return (self.commission_bps + self.half_spread_bps) * 1e-4

    def trading_cost(self, turnover: pd.Series) -> pd.Series:
        """turnover = sum of |weight changes| that day (fraction of capital)."""
        return turnover * self.per_side_cost

    def borrow_cost(self, short_notional_weight: pd.Series) -> pd.Series:
        """short_notional_weight = sum of |negative weights| held that day."""
        daily_rate = self.borrow_fee_annual_bps * 1e-4 / TRADING_DAYS
        return short_notional_weight * daily_rate
