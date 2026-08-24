"""Funding-carry panel construction and signal.

Timing convention — the thing that makes or breaks this backtest:

    decision at settlement t  ->  position held over (t, t+1]  ->  earns f_{t+1}

At time t you observe funding rates up to and including f_t (already paid) and
the current basis. You then choose weights for the NEXT interval, which earns
the NEXT funding payment f_{t+1}. So the trading problem is genuinely
*predicting* funding, never observing it. `carry_signal` returns weights
indexed at t meaning "held over (t, t+1]"; the backtester shifts returns
accordingly and never lets f_{t+1} inform w_t.

Why a naive predictor works: funding is strongly autocorrelated (crowded
positioning persists for days-to-weeks), so an EWMA of past funding is a
legitimately good forecast. That is the edge — persistence of a risk-transfer
premium, not a fitted pattern.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Binance USD-M funding settles every 8h -> 3/day -> 1095/yr
PERIODS_PER_YEAR = 365 * 3


def build_carry_panel(
    funding: dict[str, pd.Series],
    perp_close: dict[str, pd.Series],
    spot_close: dict[str, pd.Series],
    quote_volume: dict[str, pd.Series] | None = None,
    min_history: int = 200,
    freq: str = "8h",
) -> dict[str, pd.DataFrame]:
    """Align per-symbol series onto a REGULAR `freq` grid.

    Binance does not use one funding schedule: most symbols settle every 8h,
    but some were moved to 4h. Taking the union of raw funding timestamps
    produces a ragged grid on which 8h symbols are NaN half the time — which
    silently corrupts every downstream rolling statistic. So we normalize:

    - funding is SUMMED into each bucket (it is a cash flow: two 4h payments
      inside one 8h bucket are economically one 8h payment). Timestamps are
      rounded UP to the bucket close, so a payment at 04:00 belongs to the
      08:00 bucket — it happened *during* that interval, not before it.
    - prices are taken as the last observation at//before the grid point.

    Returns dict with keys: funding, perp, spot, basis, dollar_volume.
    """
    keep = [s for s, f in funding.items() if len(f.dropna()) >= min_history]
    dropped = sorted(set(funding) - set(keep))
    if dropped:
        logger.info("dropping %d symbols with short history: %s", len(dropped), dropped[:10])
    if not keep:
        raise ValueError("no symbols passed the min_history filter")

    raw = pd.DataFrame({s: funding[s] for s in keep}).sort_index()
    grid = pd.date_range(raw.index.min().ceil(freq), raw.index.max().ceil(freq), freq=freq)

    f_cols = {}
    for s in keep:
        ser = funding[s].dropna()
        # Binance settlement stamps carry millisecond jitter (e.g.
        # 00:00:00.006). Bucketing those directly with ceil() shoves an
        # on-time payment a FULL interval forward, where it collides with the
        # next one and leaves the true slot empty — we measured 634 such
        # holes (16% of the panel) before snapping. Round the jitter away
        # first, THEN ceil, so that:
        #   00:00:00.006 -> 00:00  (an on-time 8h settlement, stays put)
        #   04:00        -> 08:00  (a 4h settlement, folds into its 8h bucket)
        snapped = ser.index.round("1min").ceil(freq)
        agg = ser.groupby(snapped).sum()
        f_cols[s] = agg.reindex(grid)
    f = pd.DataFrame(f_cols, index=grid)

    n_per_bucket = pd.Series(
        {s: len(funding[s].dropna()) for s in keep}
    )
    logger.info(
        "panel grid %s: %d rows; raw funding obs per symbol %d..%d",
        freq, len(grid), int(n_per_bucket.min()), int(n_per_bucket.max()),
    )

    def _align(d: dict[str, pd.Series]) -> pd.DataFrame:
        cols = {}
        for s in keep:
            ser = d.get(s)
            if ser is None or len(ser.dropna()) == 0:
                cols[s] = pd.Series(np.nan, index=grid)
            else:
                # last observation at or before each grid point; never backward
                cols[s] = ser.dropna().reindex(grid, method="ffill", limit=1)
        return pd.DataFrame(cols, index=grid)

    perp = _align(perp_close)
    spot = _align(spot_close)
    vol = _align(quote_volume) if quote_volume else pd.DataFrame(np.nan, index=grid, columns=keep)

    # SETTLE AT THE TRUE DELISTING PRICE, not a stale ffilled one.
    #
    # A delta-neutral position needs a genuine, simultaneous quote on BOTH
    # legs. In practice the two legs stop trading at slightly different
    # times when a contract is delisted (Binance typically halts the perp
    # and the spot market on different schedules). `_align`'s
    # ffill(limit=1) — needed elsewhere to paper over an isolated missing
    # bar — does this independently per field, so the leg that stopped
    # first gets its last print frozen for one extra bar while the other
    # leg keeps moving. That fabricates a basis divergence that never
    # existed: measured directly on LUNAUSDT/SRMUSDT/etc, this produced
    # spurious bars like spot_ret=-96%, perp_ret=0% right at the real
    # settlement point — a data artifact, not economics.
    #
    # Fix: for each symbol, find the last bar where BOTH legs had a real
    # (non-ffilled) print — the last moment a position could actually be
    # marked and closed on both legs — and NaN everything past it for that
    # symbol, on both legs. The final realized return is then computed
    # real-price-to-real-price on both sides (the true delisting price),
    # and the position goes flat cleanly afterward: `backtest_weights`
    # already zeroes weight/P&L on a NaN return.
    for s in keep:
        last_perp = perp_close.get(s, pd.Series(dtype=float)).dropna()
        last_spot = spot_close.get(s, pd.Series(dtype=float)).dropna()
        if last_perp.empty or last_spot.empty:
            continue
        cutoff = min(last_perp.index.max(), last_spot.index.max())
        beyond = grid > cutoff
        if beyond.any():
            perp.loc[beyond, s] = np.nan
            spot.loc[beyond, s] = np.nan
            vol.loc[beyond, s] = np.nan

    # ZERO-VOLUME BARS ARE NOT PRICES.
    #
    # The truncation above only catches a contract whose data STOPS. Binance
    # often keeps PUBLISHING bars for a dead contract: zero volume, price
    # frozen at the last real trade. Measured on UNFIUSDT, the perp printed
    # volume=0 with price pinned at 1.730 for seven straight days before its
    # delisting, while the spot leg kept trading. A frozen perp against a live
    # spot is not a hedged position — it silently becomes naked directional
    # exposure, and the resulting P&L is pure artifact. (This cost us a -9.9%
    # "loss" we initially mistook for a genuine delisting event.)
    #
    # A bar with no trades has no valid mark, so NaN the price: `carry_returns`
    # then yields NaN and `backtest_weights` flattens the position instead of
    # marking it against a stale quote.
    if quote_volume:
        dead = vol.fillna(0.0) <= 0.0
        perp = perp.mask(dead)
        spot = spot.mask(dead)

    # basis in bps: how far the perp trades above spot
    basis = (perp / spot - 1.0) * 1e4

    return {"funding": f, "perp": perp, "spot": spot, "basis": basis, "dollar_volume": vol}


def carry_signal(
    panel: dict[str, pd.DataFrame],
    lookback: int = 21,           # ~7 days of 8h settlements
    top_n: int = 10,
    min_predicted_bps: float = 1.0,
    max_predicted_bps: float = 50.0,
    allow_short_carry: bool = True,
    min_dollar_volume: float = 5e6,
    rebalance_every: int = 3,     # 3 x 8h = daily
    hysteresis: float = 2.0,      # hold while rank <= top_n * hysteresis
    max_weight_per_symbol: float = 0.15,
) -> pd.DataFrame:
    """Cross-sectional funding carry weights.

    w > 0  : long spot / short perp  (collect positive funding)
    w < 0  : short spot / long perp  (collect negative funding)

    Every input to row t uses data through t only. Weights are equal-weight
    within the selected basket, capped at `max_weight_per_symbol`, and sum to
    AT MOST 1.0 gross — a thin book deliberately runs under-invested rather
    than concentrated (see CONCENTRATION CAP below).

    TURNOVER CONTROL — this is not a tuning knob, it is what makes the
    strategy implementable. Funding accrues every settlement regardless of
    whether you trade, so the optimal book is *sticky*: you want the carry,
    not the churn. Naive "re-rank and equal-weight every 8h" turns over ~50%
    of the book per settlement (~540x/yr), which at realistic fees costs more
    than the entire premium. Two standard fixes:

    - `rebalance_every`: only revisit the book every k settlements.
    - `hysteresis`: an incumbent position is retained while it stays within
      `top_n * hysteresis` in rank, so a name oscillating around the cutoff
      is not traded in and out. Only clear losers are replaced.

    Set rebalance_every=1, hysteresis=1.0 to recover the naive behaviour.

    CONCENTRATION CAP — `max_weight_per_symbol`. Equal-weighting as
    `sign / len(held)` silently turns a SHRINKING book into a concentrated
    one: when few names pass the funding/liquidity gates, each surviving
    position balloons. Measured on the point-in-time universe, the book
    collapsed to 3 names in Nov 2024 and put 33% of capital into a single
    symbol (UNFIUSDT) that was in the middle of its spot delisting — and
    flipped from short-carry to long-carry mid-event. That one position cost
    -9.9% over four days.

    The cap divides by `max(len(held), 1/max_weight)` instead, so a thin book
    runs at REDUCED GROSS rather than concentrated full gross. If there are
    only 3 opportunities, hold 3 small positions — do not lever into them.
    Gross exposure becomes a consequence of opportunity count, which is the
    economically correct behaviour for a carry book.
    """
    f = panel["funding"]
    # EWMA forecast of next funding, computed causally (ewm at t uses only
    # rows <= t, and f_t is already paid at t).
    pred = f.ewm(span=lookback, min_periods=max(5, lookback // 3)).mean()

    # Liquidity gate: don't select a symbol we could not actually trade.
    vol = panel["dollar_volume"].rolling(lookback, min_periods=5).mean()
    tradeable = (vol >= min_dollar_volume) | vol.isna()
    # A rolling MEAN is not enough: one liquidation-driven volume spike keeps
    # the average above the floor for weeks while the book is already dead.
    # UNFIUSDT printed $400M on a single day, then zero volume for the next
    # seven — and stayed "eligible" the whole time. Require the CURRENT bar to
    # have traded at all, which is the observable a real desk would check.
    tradeable &= panel["dollar_volume"].fillna(0.0) > 0.0
    # LISTING gate: the contract must actually be settling funding right now.
    # Without this a delisted symbol stays "eligible" forever — its volume is
    # NaN (which the liquidity gate lets through) and the EWMA carries its last
    # funding value indefinitely, so the book would hold a contract that no
    # longer exists. Harmless on a survivor-only universe where nothing ever
    # dies; essential on a point-in-time one, which is exactly the universe
    # where survivorship bias is being measured.
    tradeable &= panel["funding"].notna() & panel["perp"].notna()

    pred_bps = pred * 1e4
    # Upper gate (`max_predicted_bps`) is a RISK CONTROL, not a return filter.
    # Funding of -250bp per 8h is not a carry opportunity, it is a short
    # squeeze: the spot leg is unborrowable, the perp is being manipulated,
    # and the size you could actually execute is a small fraction of what the
    # backtest assumes. Without this gate a single such episode (TRBUSDT,
    # Sept-Oct 2023) contributed 36% of gross P&L in our sample — a number
    # that would not have survived contact with a real order book.
    eligible = (
        tradeable
        & pred_bps.abs().ge(min_predicted_bps)
        & pred_bps.abs().le(max_predicted_bps)
    )
    scores = pred_bps.where(eligible)
    keep_n = max(int(round(top_n * hysteresis)), top_n)

    weights = pd.DataFrame(0.0, index=f.index, columns=f.columns)
    held: dict[str, float] = {}       # symbol -> sign of current position
    # A thin book runs at reduced gross rather than concentrated full gross.
    min_denom = int(np.ceil(1.0 / max_weight_per_symbol)) if max_weight_per_symbol > 0 else 1

    def _apply(t, held):
        if not held:
            return
        syms = list(held)
        denom = max(len(syms), min_denom)
        weights.loc[t, syms] = [held[s] / denom for s in syms]

    for i, (t, row) in enumerate(scores.iterrows()):
        if i % rebalance_every != 0:
            # Between rebalances the book is unchanged (no trading).
            _apply(t, held)
            continue

        row = row.dropna()
        if row.empty:
            held = {}
            continue

        longs = row[row > 0]
        shorts = row[row < 0] if allow_short_carry else row.iloc[:0]
        # Fill candidates must stay in RANK ORDER (lists), because the slot cap
        # below stops early — iterating a set here made the choice depend on
        # Python's per-process string hash salt, so the same config on the same
        # data produced different books (measured: Sharpe 5.7851 vs 5.7895 on
        # back-to-back runs) and could drop a better-scoring name for a worse
        # one. Sets are still fine for the pure membership tests.
        fresh_long = list(longs.nlargest(top_n).index)
        fresh_short = list(shorts.nsmallest(top_n).index)
        wide_long = set(longs.nlargest(keep_n).index)
        wide_short = set(shorts.nsmallest(keep_n).index)

        new_held: dict[str, float] = {}
        # 1. incumbents survive if still inside the wider band, same direction
        for s, sign in held.items():
            if sign > 0 and s in wide_long:
                new_held[s] = 1.0
            elif sign < 0 and s in wide_short:
                new_held[s] = -1.0
        # 2. fill remaining slots from the fresh top-N
        for s in fresh_long:
            if len(new_held) >= 2 * top_n:
                break
            new_held.setdefault(s, 1.0)
        for s in fresh_short:
            if len(new_held) >= 2 * top_n:
                break
            new_held.setdefault(s, -1.0)

        held = new_held
        _apply(t, held)

    return weights


def carry_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-symbol return of a UNIT carry position over (t, t+1], indexed at t.

    Components, for w=+1 (long spot, short perp):
        + f_{t+1}                       funding received at next settlement
        + spot_ret - perp_ret           basis convergence (delta-neutral residual)

    Indexed at t so it lines up with weights decided at t. The backtester
    therefore never needs to shift again — and cannot double-shift.
    """
    f_next = panel["funding"].shift(-1)
    # fill_method=None is NOT optional here. pandas' pct_change() defaults to
    # fill_method="pad" on <3.0, which forward-fills NaN gaps before
    # differencing -- silently converting a masked dead-contract price into a
    # fabricated 0% return, exactly the artifact `build_carry_panel` NaNs
    # those bars to prevent. This default is also version-dependent: it
    # flipped in pandas 3.0, so identical code produced correct results
    # locally (pandas 3.0.5) and a fabricated 0% return in CI (pandas 2.3.3)
    # -- caught by test_zero_volume_bars_are_not_treated_as_prices.
    spot_ret = panel["spot"].pct_change(fill_method=None).shift(-1)
    perp_ret = panel["perp"].pct_change(fill_method=None).shift(-1)
    return (f_next + spot_ret - perp_ret).replace([np.inf, -np.inf], np.nan)
