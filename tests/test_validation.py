import numpy as np
import pandas as pd
import pytest

from statarb.capacity import capacity_curve
from statarb.validation import deflated_sharpe_ratio, expected_max_sharpe, make_windows


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(100) > expected_max_sharpe(10) > 0


def test_dsr_penalizes_many_trials():
    rng = np.random.default_rng(1)
    # genuine edge: mean/std per day ~ 0.1 (annual Sharpe ~ 1.6)
    r = pd.Series(
        rng.normal(0.001, 0.01, 1000),
        index=pd.bdate_range("2018-01-01", periods=1000),
    )
    few = deflated_sharpe_ratio(r, n_trials=5)
    many = deflated_sharpe_ratio(r, n_trials=50_000)
    assert few["dsr"] > many["dsr"]


def test_walkforward_windows_never_overlap_formation_and_trading():
    dates = pd.bdate_range("2015-01-01", periods=1600)
    windows = make_windows(dates, formation_days=504, trading_days=126, step_days=126)
    assert len(windows) >= 3
    for w in windows:
        assert w.formation_end < w.trading_start
        assert w.trading_start <= w.trading_end
    # consecutive trading segments should tile without gaps (step == trading)
    for a, b in zip(windows, windows[1:], strict=False):
        gap_days = dates.get_loc(b.trading_start) - dates.get_loc(a.trading_end)
        assert gap_days == 1


def test_capacity_curve_is_monotone_decreasing_in_aum():
    rng = np.random.default_rng(3)
    n = 750
    idx = pd.bdate_range("2019-01-01", periods=n)
    base = pd.Series(rng.normal(0.0005, 0.004, n), index=idx)
    dw = pd.DataFrame({"AAA": np.full(n, 0.01), "BBB": np.full(n, 0.01)}, index=idx)
    adv = pd.DataFrame({"AAA": np.full(n, 5e7), "BBB": np.full(n, 2e7)}, index=idx)
    sig = pd.DataFrame({"AAA": np.full(n, 0.015), "BBB": np.full(n, 0.02)}, index=idx)

    curve = capacity_curve(base, dw, adv, sig, aum_grid=[1e6, 1e7, 1e8, 1e9], k=0.5)
    sharpes = curve["sharpe"].to_numpy()
    assert all(np.diff(sharpes) <= 1e-9)
    assert curve.attrs["aum_sharpe_zero"] is None or curve.attrs["aum_sharpe_zero"] > 1e6


def test_dsr_requires_enough_observations():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 10))
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(r, n_trials=10)
