"""Tests for the funding-carry path.

The one that matters most is `test_carry_returns_are_forward_looking_by_exactly_one`:
the strategy earns f_{t+1} from a decision made at t. Off by one in either
direction and the backtest is either look-ahead or quietly throws the edge away.
Several of the others pin down data bugs found on real Binance history.
"""

import numpy as np
import pandas as pd

from statarb.backtest import CrossSectionalCostModel, backtest_weights, exit_size
from statarb.crypto.funding import build_carry_panel, carry_returns, carry_signal

FREE = CrossSectionalCostModel(0, 0, 0, 0)


def _panel(n=300, symbols=("AAA", "BBB", "CCC"), seed=0, spread=1e-5):
    """Synthetic panel with persistent (AR(1)) funding, the property the signal uses."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    mid = len(symbols) // 2
    funding, perp, spot, vol = {}, {}, {}, {}
    for i, s in enumerate(symbols):
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.9 * f[t - 1] + rng.normal(0, 2e-5) + (i - mid) * spread
        px = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 10)
        funding[s] = pd.Series(f, index=idx)
        spot[s] = pd.Series(px, index=idx)
        perp[s] = pd.Series(px * (1 + rng.normal(0, 1e-4, n)), index=idx)
        vol[s] = pd.Series(np.full(n, 5e8), index=idx)
    return build_carry_panel(funding, perp, spot, vol, min_history=10)


def test_panel_alignment():
    p = _panel()
    assert set(p) == {"funding", "perp", "spot", "basis", "dollar_volume"}
    assert p["funding"].shape == p["perp"].shape == p["spot"].shape
    assert p["basis"].abs().mean().mean() < 100  # bps


def test_carry_returns_are_forward_looking_by_exactly_one():
    p = _panel()
    r = carry_returns(p)
    spot_ret = p["spot"].pct_change(fill_method=None).shift(-1)
    perp_ret = p["perp"].pct_change(fill_method=None).shift(-1)
    implied_funding = (r - spot_ret + perp_ret).iloc[:-1]
    pd.testing.assert_frame_equal(
        implied_funding, p["funding"].shift(-1).iloc[:-1], check_names=False, atol=1e-12
    )


def test_signal_uses_no_future_funding():
    """A shock to the last funding print must not change any earlier weight."""
    p = _panel()
    w1 = carry_signal(p, lookback=10, top_n=2)
    p2 = {k: v.copy() for k, v in p.items()}
    p2["funding"].iloc[-1] += 0.05
    w2 = carry_signal(p2, lookback=10, top_n=2)
    pd.testing.assert_frame_equal(w1.iloc[:-1], w2.iloc[:-1])


def test_thin_book_runs_under_invested_not_concentrated():
    """With only 3 symbols and a 15% cap, no position may exceed the cap and
    gross must stay below 1. Without the cap a shrinking book once put 33%
    of capital into a single contract that was mid-delisting."""
    cap = 0.15
    w = carry_signal(_panel(), lookback=10, top_n=2, max_weight_per_symbol=cap)
    active = w[w.abs().sum(axis=1) > 0]
    assert len(active) > 0
    assert w.abs().to_numpy().max() <= cap + 1e-9
    gross = active.abs().sum(axis=1)
    assert gross.max() <= 1.0 + 1e-9
    assert gross.max() < 1.0


def test_wide_book_reaches_full_gross():
    syms = [f"S{i:02d}" for i in range(12)]
    p = _panel(symbols=syms, seed=5, spread=2e-5)
    w = carry_signal(p, lookback=10, top_n=5, max_weight_per_symbol=0.15)
    gross = w[w.abs().sum(axis=1) > 0].abs().sum(axis=1)
    assert np.isclose(gross.max(), 1.0, atol=1e-9)


def test_costs_reduce_returns_and_leave_gross_untouched():
    p = _panel()
    w = carry_signal(p, lookback=10, top_n=2)
    r = carry_returns(p)
    free = backtest_weights(w, r, FREE)
    pricey = backtest_weights(w, r, CrossSectionalCostModel(20, 20, 5, 5))
    assert pricey["returns"].sum() < free["returns"].sum()
    pd.testing.assert_series_equal(free["gross_returns"], pricey["gross_returns"])


def test_leverage_cap_binds():
    p = _panel()
    w = carry_signal(p, lookback=10, top_n=3)
    res = backtest_weights(w, carry_returns(p), max_gross_leverage=0.5)
    assert res["gross_exposure"].max() <= 0.5 + 1e-9


def test_unfillable_weight_is_zeroed():
    """A weight on an asset with no return must not quietly earn 0%."""
    idx = pd.date_range("2023-01-01", periods=5, freq="8h")
    w = pd.DataFrame({"AAA": [1.0] * 5}, index=idx)
    r = pd.DataFrame({"AAA": [0.01, np.nan, 0.01, 0.01, 0.01]}, index=idx)
    res = backtest_weights(w, r, FREE)
    assert res["weights"].loc[idx[1], "AAA"] == 0.0
    assert res["per_asset_pnl"].loc[idx[1], "AAA"] == 0.0


def test_exit_size_counts_only_a_held_contract_that_stops_printing():
    """AAA is held and dies at bar 2; BBB is wanted but never held;
    the last bar is the sample end."""
    idx = pd.date_range("2023-01-01", periods=5, freq="8h")
    w = pd.DataFrame({"AAA": [0.5] * 5, "BBB": [0.0, 0.0, 0.0, 0.0, 0.5]}, index=idx)
    r = pd.DataFrame({"AAA": [0.01, 0.01, np.nan, np.nan, np.nan],
                      "BBB": [0.01, 0.01, 0.01, 0.01, np.nan]}, index=idx)
    size = exit_size(w, backtest_weights(w, r, FREE))
    assert size.tolist() == [0.0, 0.0, 0.5, 0.0, 0.0]


def test_jittered_funding_timestamps_do_not_create_holes():
    """Binance stamps settlements with ms jitter (00:00:00.006). A plain ceil()
    pushed those a whole bucket forward and left 16% of the panel empty."""
    n = 90
    clean = pd.date_range("2023-01-01", periods=n, freq="8h")
    rng = np.random.default_rng(3)
    jittered = clean + pd.to_timedelta(rng.integers(0, 40, n), unit="ms")
    f = pd.Series(np.full(n, 1e-4), index=jittered)
    px = pd.Series(np.linspace(100, 110, n), index=clean)
    panel = build_carry_panel(
        {"AAA": f}, {"AAA": px}, {"AAA": px},
        {"AAA": pd.Series(np.full(n, 1e9), index=clean)}, min_history=10,
    )
    got = panel["funding"]["AAA"]
    assert got.notna().sum() >= n - 1, f"lost settlements: {got.isna().sum()} holes"
    np.testing.assert_allclose(got.dropna().unique(), [1e-4])


def test_four_hour_funding_folds_into_eight_hour_buckets():
    n = 120
    idx4 = pd.date_range("2023-01-01", periods=n, freq="4h")
    grid8 = pd.date_range("2023-01-01", periods=n // 2, freq="8h")
    f = pd.Series(np.full(n, 1e-4), index=idx4)
    px = pd.Series(np.linspace(100, 110, len(grid8)), index=grid8)
    panel = build_carry_panel(
        {"AAA": f}, {"AAA": px}, {"AAA": px},
        {"AAA": pd.Series(np.full(len(grid8), 1e9), index=grid8)}, min_history=10,
    )
    got = panel["funding"]["AAA"].dropna()
    assert np.isclose(got.mode().iloc[0], 2e-4), f"got {got.mode().iloc[0]}"


def test_both_legs_cut_at_shared_last_print():
    """Perp stops printing two bars before spot (a real delisting pattern).
    Forward-filling each leg on its own froze the perp while spot crashed,
    fabricating bars like spot -96%, perp 0%. Both legs must be cut at the
    last bar where both had a real price."""
    n = 20
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    perp_idx = idx[:-2]
    perp = pd.Series(np.linspace(100, 90, len(perp_idx)), index=perp_idx)
    spot = pd.Series(np.linspace(100, 10, n), index=idx)
    funding = pd.Series(np.full(n, 1e-4), index=idx)
    vol = pd.Series(np.full(n, 1e9), index=idx)
    panel = build_carry_panel(
        {"AAA": funding}, {"AAA": perp}, {"AAA": spot}, {"AAA": vol}, min_history=5,
    )
    p_out, s_out = panel["perp"]["AAA"], panel["spot"]["AAA"]
    cutoff = perp_idx[-1]

    assert p_out.loc[p_out.index > cutoff].isna().all()
    assert s_out.loc[s_out.index > cutoff].isna().all()
    assert abs(p_out.loc[cutoff] - 90.0) < 1e-9
    assert pd.notna(s_out.loc[cutoff])

    perp_ret = p_out.pct_change(fill_method=None)
    spot_ret = s_out.pct_change(fill_method=None)
    fabricated = (perp_ret.abs() < 1e-9) & (spot_ret.abs() > 0.5)
    assert not fabricated.any(), "stale-ffill artifact reproduced"


def test_zero_volume_bars_are_not_treated_as_prices():
    """Binance keeps publishing bars for a dead contract with volume 0 and a
    frozen price (UNFIUSDT: seven days pinned at 1.730 while spot kept
    trading). Those bars must carry no price, earn no P&L, and never be
    entered into."""
    n, dead_from = 30, 20
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    perp_px = np.concatenate([np.linspace(100, 95, dead_from), np.full(n - dead_from, 95.0)])
    spot_px = np.concatenate([np.linspace(100, 95, dead_from), np.linspace(95, 40, n - dead_from)])
    volume = np.concatenate([np.full(dead_from, 1e9), np.zeros(n - dead_from)])
    panel = build_carry_panel(
        {"AAA": pd.Series(np.full(n, 1e-4), index=idx)},
        {"AAA": pd.Series(perp_px, index=idx)},
        {"AAA": pd.Series(spot_px, index=idx)},
        {"AAA": pd.Series(volume, index=idx)},
        min_history=5,
    )
    assert panel["perp"]["AAA"].iloc[dead_from:].isna().all()
    assert panel["spot"]["AAA"].iloc[dead_from:].isna().all()
    assert carry_returns(panel)["AAA"].iloc[dead_from:].isna().all()

    # Rebalancing every bar isolates the eligibility gate: no new entry into a dead book.
    w = carry_signal(panel, lookback=5, top_n=1, min_dollar_volume=0.0, rebalance_every=1)
    assert (w["AAA"].iloc[dead_from:] == 0).all()

    # With a slower rebalance the position may legitimately still be held for
    # a few bars (you cannot sell into an empty book either), but it earns nothing.
    w_slow = carry_signal(panel, lookback=5, top_n=1, min_dollar_volume=0.0, rebalance_every=3)
    res = backtest_weights(w_slow, carry_returns(panel), FREE)
    assert (res["per_asset_pnl"]["AAA"].iloc[dead_from:] == 0).all()


def test_signal_is_deterministic_across_symbol_order():
    """Ranking, not dict or set order, must decide the book. Iterating a set
    while filling slots once made results depend on Python's per-process
    hash seed (Sharpe 5.7851 vs 5.7895 on back-to-back runs)."""
    syms = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(11)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    funding, perp, spot, vol = {}, {}, {}, {}
    for i, s in enumerate(syms):
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.9 * f[t - 1] + rng.normal(0, 2e-5) + (i - 5) * 2e-5
        px = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 10)
        funding[s] = pd.Series(f, index=idx)
        spot[s] = pd.Series(px, index=idx)
        perp[s] = pd.Series(px * (1 + rng.normal(0, 1e-4, n)), index=idx)
        vol[s] = pd.Series(np.full(n, 5e8), index=idx)

    def build(order):
        p = build_carry_panel(
            {k: funding[k] for k in order}, {k: perp[k] for k in order},
            {k: spot[k] for k in order}, {k: vol[k] for k in order}, min_history=10,
        )
        return carry_signal(p, lookback=10, top_n=3, rebalance_every=1)

    pd.testing.assert_frame_equal(build(syms)[syms], build(list(reversed(syms)))[syms])


def test_rebalance_offset_moves_the_trading_bars():
    p = _panel()
    w = carry_signal(p, lookback=10, top_n=1, rebalance_every=5, rebalance_offset=2)
    changed = np.flatnonzero(w.diff().abs().sum(axis=1).to_numpy() > 0)
    assert len(changed) > 0
    assert all(i % 5 == 2 for i in changed)
    assert (w.iloc[:2] == 0).all().all()  # nothing held before the first rebalance
