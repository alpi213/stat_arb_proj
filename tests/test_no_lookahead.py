"""The most important tests here. A backtest with leakage is worse than none.

1. Shocking the last bar's price must not change any earlier position.
2. Every weight change must be preceded by a signal event on an earlier bar.
3. On independent random walks the engine must not print money.
"""

import numpy as np
import pandas as pd

from statarb.backtest import CostModel, backtest_pair
from statarb.hedge import KalmanHedge
from tests.conftest import make_cointegrated_pair


def _run(prices: pd.DataFrame):
    return backtest_pair(
        prices.rename(columns={"y": "YY", "x": "XX"}),
        y="YY",
        x="XX",
        hedge=KalmanHedge(delta=1e-5),
        zscore_window=40,
        costs=CostModel(commission_bps=0, half_spread_bps=0, borrow_fee_annual_bps=0),
    )


def test_future_perturbation_does_not_change_past_positions():
    prices = make_cointegrated_pair(n=800)
    base = _run(prices)

    shocked = prices.copy()
    shocked.iloc[-1, shocked.columns.get_loc("y")] *= 1.10
    pert = _run(shocked)

    pd.testing.assert_frame_equal(base.weights.iloc[:-1], pert.weights.iloc[:-1])
    pd.testing.assert_series_equal(base.daily_returns.iloc[:-1], pert.daily_returns.iloc[:-1])


def test_positions_are_lagged_relative_to_signal():
    res = _run(make_cointegrated_pair(n=800))
    changes = res.weights.diff().abs().sum(axis=1)
    change_days = changes[changes > 1e-12].index
    event_days = res.events[res.events != ""].index
    for d in change_days:
        assert (event_days < d).any(), f"weight change on {d} with no prior signal event"


def test_zero_cost_symmetric_noise_has_no_free_lunch():
    """One sample path, so the band is wide; a leaky engine typically shows |Sharpe| > 3."""
    rng = np.random.default_rng(2024)
    n = 1500
    dates = pd.bdate_range("2016-01-01", periods=n)
    prices = pd.DataFrame(
        {
            "y": np.exp(np.cumsum(rng.normal(0, 0.015, n)) + 4),
            "x": np.exp(np.cumsum(rng.normal(0, 0.015, n)) + 4),
        },
        index=dates,
    )
    r = _run(prices).daily_returns
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    assert abs(sharpe) < 3.0
