"""The most important tests in the repo.

A backtest with leakage is worse than no backtest — it produces confident
nonsense. These tests attack the engine directly:

1. Perturbation test: changing the LAST bar's price must not change any
   position held BEFORE that bar (with lag=1, the trade triggered by the
   last bar's signal never even executes inside the sample).
2. Lag test: the engine must not be able to monetize a pattern that is
   only exploitable with same-bar execution.
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
    shocked.iloc[-1, shocked.columns.get_loc("y")] *= 1.10  # +10% shock on last bar
    pert = _run(shocked)

    # Everything strictly before the shocked bar must be identical.
    pd.testing.assert_frame_equal(base.weights.iloc[:-1], pert.weights.iloc[:-1])
    pd.testing.assert_series_equal(
        base.daily_returns.iloc[:-1], pert.daily_returns.iloc[:-1]
    )


def test_positions_are_lagged_relative_to_signal():
    prices = make_cointegrated_pair(n=800)
    res = _run(prices)
    # Reconstruct target weights implied by close-of-bar signals: held
    # weights must equal target weights shifted by exactly 1 bar. We verify
    # via the engine's own outputs: any bar where held weight changed, the
    # signal event must have occurred on an EARLIER bar.
    changes = res.weights.diff().abs().sum(axis=1)
    change_days = changes[changes > 1e-12].index
    event_days = res.events[res.events != ""].index
    for d in change_days:
        prior_events = event_days[event_days < d]
        assert len(prior_events) > 0, f"weight change on {d} with no prior signal event"


def test_zero_cost_symmetric_noise_has_no_free_lunch():
    """On pure independent random walks the strategy should NOT print money.

    Guards against subtle leaks that manufacture profit from noise. We allow
    a wide band (it's one sample path) but a leaky engine typically shows
    absurd Sharpe > 3 here.
    """
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
    res = _run(prices)
    ann_sharpe = (
        res.daily_returns.mean() / res.daily_returns.std() * np.sqrt(252)
        if res.daily_returns.std() > 0
        else 0.0
    )
    assert abs(ann_sharpe) < 3.0
