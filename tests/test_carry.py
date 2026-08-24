"""Tests for the crypto funding-carry path.

The critical one is `test_carry_returns_are_forward_looking_by_exactly_one`:
the whole strategy hinges on earning f_{t+1} from a decision made at t. Off by
one in either direction and the backtest is either look-ahead (fatal) or
silently dropping the edge.
"""

import numpy as np
import pandas as pd

from statarb.backtest import CrossSectionalCostModel, backtest_weights
from statarb.crypto.funding import build_carry_panel, carry_returns, carry_signal


def _panel(n=300, symbols=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    funding, perp, spot, vol = {}, {}, {}, {}
    for i, s in enumerate(symbols):
        # persistent funding (AR(1)) — the property the signal exploits
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.9 * f[t - 1] + rng.normal(0, 2e-5) + (i - 1) * 1e-5
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
    assert p["basis"].abs().mean().mean() < 100  # bps, sane magnitude


def test_carry_returns_are_forward_looking_by_exactly_one():
    p = _panel()
    r = carry_returns(p)
    f = p["funding"]
    # The funding component of r.loc[t] must be f.loc[t+1], not f.loc[t].
    spot_ret = p["spot"].pct_change(fill_method=None).shift(-1)
    perp_ret = p["perp"].pct_change(fill_method=None).shift(-1)
    implied_funding = (r - spot_ret + perp_ret).iloc[:-1]
    pd.testing.assert_frame_equal(
        implied_funding, f.shift(-1).iloc[:-1], check_names=False, atol=1e-12
    )


def test_signal_uses_no_future_funding():
    """Perturbing the LAST funding observation must not change earlier weights."""
    p = _panel()
    w1 = carry_signal(p, lookback=10, top_n=2)
    p2 = {k: v.copy() for k, v in p.items()}
    p2["funding"].iloc[-1] += 0.05  # enormous shock on the final settlement
    w2 = carry_signal(p2, lookback=10, top_n=2)
    pd.testing.assert_frame_equal(w1.iloc[:-1], w2.iloc[:-1])


def test_weights_respect_concentration_cap_and_never_exceed_full_gross():
    """A THIN book must run under-invested, not concentrated.

    Equal-weighting as sign/len(held) silently turns a shrinking book into a
    concentrated one. Measured on real data, the book collapsed to 3 names and
    put 33% of capital into UNFIUSDT mid-delisting, costing -9.9% in 4 days.
    Gross exposure is now a consequence of how many opportunities exist.
    """
    cap = 0.15
    p = _panel()  # only 3 symbols -> deliberately thinner than 1/cap
    w = carry_signal(p, lookback=10, top_n=2, max_weight_per_symbol=cap)
    active = w[w.abs().sum(axis=1) > 0]
    assert len(active) > 0
    # no single position may exceed the cap
    assert w.abs().to_numpy().max() <= cap + 1e-9
    # gross never exceeds 1.0, and here is BELOW it because the book is thin
    gross = active.abs().sum(axis=1)
    assert gross.max() <= 1.0 + 1e-9
    assert gross.max() < 1.0, "3-symbol book should be under-invested, not full gross"


def test_wide_book_reaches_full_gross():
    """With enough opportunities, gross returns to 1.0 (cap stops binding)."""
    n, syms = 300, [f"S{i:02d}" for i in range(12)]
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    funding, perp, spot, vol = {}, {}, {}, {}
    for i, s in enumerate(syms):
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.9 * f[t - 1] + rng.normal(0, 2e-5) + (i - 6) * 2e-5
        px = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 10)
        funding[s] = pd.Series(f, index=idx)
        spot[s] = pd.Series(px, index=idx)
        perp[s] = pd.Series(px * (1 + rng.normal(0, 1e-4, n)), index=idx)
        vol[s] = pd.Series(np.full(n, 5e8), index=idx)
    p = build_carry_panel(funding, perp, spot, vol, min_history=10)
    w = carry_signal(p, lookback=10, top_n=5, max_weight_per_symbol=0.15)
    gross = w[w.abs().sum(axis=1) > 0].abs().sum(axis=1)
    assert np.isclose(gross.max(), 1.0, atol=1e-9)


def test_costs_reduce_returns_monotonically():
    p = _panel()
    w = carry_signal(p, lookback=10, top_n=2)
    r = carry_returns(p)
    free = backtest_weights(w, r, CrossSectionalCostModel(0, 0, 0, 0))
    pricey = backtest_weights(w, r, CrossSectionalCostModel(20, 20, 5, 5))
    assert pricey["returns"].sum() < free["returns"].sum()
    # gross stream must be identical — costs must not touch pre-cost P&L
    pd.testing.assert_series_equal(free["gross_returns"], pricey["gross_returns"])


def test_leverage_cap_binds():
    p = _panel()
    w = carry_signal(p, lookback=10, top_n=3)
    r = carry_returns(p)
    res = backtest_weights(w, r, max_gross_leverage=0.5)
    assert res["gross_exposure"].max() <= 0.5 + 1e-9


def test_unfillable_weight_is_zeroed():
    """A weight on an asset with a missing return must not silently earn 0%."""
    idx = pd.date_range("2023-01-01", periods=5, freq="8h")
    w = pd.DataFrame({"AAA": [1.0] * 5}, index=idx)
    r = pd.DataFrame({"AAA": [0.01, np.nan, 0.01, 0.01, 0.01]}, index=idx)
    res = backtest_weights(w, r, CrossSectionalCostModel(0, 0, 0, 0))
    assert res["weights"].loc[idx[1], "AAA"] == 0.0
    assert res["per_asset_pnl"].loc[idx[1], "AAA"] == 0.0


def test_jittered_funding_timestamps_do_not_create_holes():
    """Binance settlement stamps carry ms jitter (00:00:00.006).

    Naive ceil() bucketing pushes an on-time payment a full interval forward,
    colliding with the next one and leaving the true slot empty. This cost us
    16% of the panel before it was caught.
    """
    n = 90
    clean = pd.date_range("2023-01-01", periods=n, freq="8h")
    rng = np.random.default_rng(3)
    jitter = pd.to_timedelta(rng.integers(0, 40, n), unit="ms")
    jittered = clean + jitter

    f = pd.Series(np.full(n, 1e-4), index=jittered)
    px = pd.Series(np.linspace(100, 110, n), index=clean)
    panel = build_carry_panel(
        {"AAA": f}, {"AAA": px}, {"AAA": px},
        {"AAA": pd.Series(np.full(n, 1e9), index=clean)},
        min_history=10,
    )
    got = panel["funding"]["AAA"]
    # every settlement must land in its own bucket — no holes, no collisions
    assert got.notna().sum() >= n - 1, f"lost settlements: {got.isna().sum()} holes"
    np.testing.assert_allclose(got.dropna().unique(), [1e-4])


def test_four_hour_funding_folds_into_eight_hour_buckets():
    """A 4h-schedule symbol must sum two payments into each 8h bucket."""
    n = 120
    idx4 = pd.date_range("2023-01-01", periods=n, freq="4h")
    f = pd.Series(np.full(n, 1e-4), index=idx4)
    grid8 = pd.date_range("2023-01-01", periods=n // 2, freq="8h")
    px = pd.Series(np.linspace(100, 110, len(grid8)), index=grid8)
    panel = build_carry_panel(
        {"AAA": f}, {"AAA": px}, {"AAA": px},
        {"AAA": pd.Series(np.full(len(grid8), 1e9), index=grid8)},
        min_history=10,
    )
    got = panel["funding"]["AAA"].dropna()
    # interior buckets receive exactly two 4h payments
    assert np.isclose(got.mode().iloc[0], 2e-4), f"got {got.mode().iloc[0]}"


def test_settles_at_true_price_not_stale_ffill_at_delisting():
    """Reproduces the exact bug found on LUNAUSDT/SRMUSDT/etc: perp stops
    printing 2 bars before spot (a real delisting pattern -- the exchange
    halts the two markets on different schedules). Naive per-field
    ffill(limit=1) froze perp's last price while spot kept crashing,
    fabricating bars like spot_ret=-96%, perp_ret=0%. The panel must instead
    truncate BOTH legs at their shared last-real-print bar.
    """
    n = 20
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    perp_idx, spot_idx = idx[:-2], idx  # perp goes dark 2 bars before spot
    perp = pd.Series(np.linspace(100, 90, len(perp_idx)), index=perp_idx)
    spot = pd.Series(np.linspace(100, 10, len(spot_idx)), index=spot_idx)  # spot craters
    funding = pd.Series(np.full(n, 1e-4), index=idx)
    vol = pd.Series(np.full(n, 1e9), index=idx)

    panel = build_carry_panel(
        {"AAA": funding}, {"AAA": perp}, {"AAA": spot}, {"AAA": vol}, min_history=5,
    )
    p_out, s_out = panel["perp"]["AAA"], panel["spot"]["AAA"]
    shared_cutoff = perp_idx[-1]  # min(last real perp, last real spot)

    # both legs NaN beyond the shared cutoff -- no frozen perp vs live spot
    assert p_out.loc[p_out.index > shared_cutoff].isna().all()
    assert s_out.loc[s_out.index > shared_cutoff].isna().all()
    # the cutoff bar itself keeps BOTH real prices (the true settlement point)
    assert abs(p_out.loc[shared_cutoff] - 90.0) < 1e-9
    assert pd.notna(s_out.loc[shared_cutoff])

    # the exact bug signature must not appear anywhere: a ~0% perp return
    # sitting against a huge spot move in the same bar
    perp_ret = panel["perp"]["AAA"].pct_change(fill_method=None)
    spot_ret = panel["spot"]["AAA"].pct_change(fill_method=None)
    fabricated = (perp_ret.abs() < 1e-9) & (spot_ret.abs() > 0.5)
    assert not fabricated.any(), "stale-ffill artifact reproduced"


def test_zero_volume_bars_are_not_treated_as_prices():
    """A dead contract that still PUBLISHES bars must not fabricate P&L.

    Binance keeps printing klines for delisting contracts with volume=0 and
    the price frozen at the last real trade. Measured on UNFIUSDT: perp volume
    0 and price pinned at 1.730 for seven days while spot kept trading. A
    frozen perp against a live spot is naked directional exposure, not a
    hedge -- it produced a -9.9% artifact we first mistook for a real event.
    """
    n = 30
    idx = pd.date_range("2023-01-01", periods=n, freq="8h")
    dead_from = 20
    perp_px = np.concatenate([np.linspace(100, 95, dead_from),
                              np.full(n - dead_from, 95.0)])       # frozen
    spot_px = np.concatenate([np.linspace(100, 95, dead_from),
                              np.linspace(95, 40, n - dead_from)])  # keeps moving
    volume = np.concatenate([np.full(dead_from, 1e9),
                             np.zeros(n - dead_from)])              # book dies
    panel = build_carry_panel(
        {"AAA": pd.Series(np.full(n, 1e-4), index=idx)},
        {"AAA": pd.Series(perp_px, index=idx)},
        {"AAA": pd.Series(spot_px, index=idx)},
        {"AAA": pd.Series(volume, index=idx)},
        min_history=5,
    )
    # zero-volume bars carry no valid mark on either leg
    assert panel["perp"]["AAA"].iloc[dead_from:].isna().all()
    assert panel["spot"]["AAA"].iloc[dead_from:].isna().all()
    # and therefore cannot generate the fabricated one-sided return
    r = carry_returns(panel)["AAA"]
    assert r.iloc[dead_from:].isna().all()
    # the signal must refuse to open a position into a dead book. Checked at
    # rebalance_every=1 to isolate the gate: with a slower rebalance the book
    # is legitimately still HELD for a few bars (you cannot sell into a
    # zero-volume book either), and the NaN return above already prevents any
    # P&L from being booked on those bars.
    w = carry_signal(panel, lookback=5, top_n=1, min_dollar_volume=0.0,
                     rebalance_every=1)
    assert (w["AAA"].iloc[dead_from:] == 0).all()

    # no P&L may be booked on dead bars regardless of rebalance cadence
    w_slow = carry_signal(panel, lookback=5, top_n=1, min_dollar_volume=0.0)
    res = backtest_weights(w_slow, carry_returns(panel),
                           CrossSectionalCostModel(0, 0, 0, 0))
    assert (res["per_asset_pnl"]["AAA"].iloc[dead_from:] == 0).all()


def test_signal_is_deterministic_across_processes():
    """Same data + same config must give byte-identical weights.

    `carry_signal` fills its remaining slots from a ranked candidate list and
    stops at a cap. Iterating a set there made the surviving book depend on
    Python's per-process string hash salt: back-to-back runs of the identical
    backtest returned Sharpe 5.7851 and 5.7895, and the cap could discard a
    better-scoring name in favour of a worse one. Guarded here by building the
    panel twice with symbol names supplied in DIFFERENT insertion orders --
    ranking, not dict/set ordering, must decide the book.
    """
    n, syms = 200, [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(11)
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
            {k: spot[k] for k in order}, {k: vol[k] for k in order},
            min_history=10,
        )
        return carry_signal(p, lookback=10, top_n=3, rebalance_every=1)

    w1 = build(syms)
    w2 = build(list(reversed(syms)))
    pd.testing.assert_frame_equal(w1[syms], w2[syms])
