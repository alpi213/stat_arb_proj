"""Funding-carry panel, signal and per-asset returns.

Timing, which is the whole game here:

    decide at settlement t  ->  hold over (t, t+1]  ->  earn f_{t+1}

At t you know every funding payment up to f_t. You pick weights for the
next interval, and those weights earn the next payment. `carry_signal`
returns weights indexed at t meaning "held over (t, t+1]"; `carry_returns`
returns the matching forward return, also indexed at t. The backtester
multiplies them and never shifts.

Funding is strongly autocorrelated (crowded positioning persists for
days), so an EWMA of past funding is a decent forecast. That persistence
is the edge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 365 * 3  # 8h settlements


def load_panel(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the parquet files written by scripts/11_download_crypto.py."""
    d = Path(data_dir)
    frames = {name: pd.read_parquet(d / f"{name}.parquet")
              for name in ("funding", "perp_close", "spot_close", "quote_volume")}
    as_dict = {k: {c: v[c] for c in v} for k, v in frames.items()}
    return build_carry_panel(
        as_dict["funding"], as_dict["perp_close"], as_dict["spot_close"], as_dict["quote_volume"]
    )


def build_carry_panel(
    funding: dict[str, pd.Series],
    perp_close: dict[str, pd.Series],
    spot_close: dict[str, pd.Series],
    quote_volume: dict[str, pd.Series] | None = None,
    min_history: int = 200,
    freq: str = "8h",
) -> dict[str, pd.DataFrame]:
    """Align everything onto a regular `freq` grid.

    Returns funding, perp, spot, basis (bps) and dollar_volume frames.

    Binance does not settle every symbol on the same schedule: most are 8h,
    some moved to 4h. A ragged union of raw timestamps would leave 8h
    symbols NaN half the time and corrupt every rolling statistic, so
    funding is summed into each bucket (two 4h payments are one 8h payment)
    and prices take the last print at or before each grid point.

    Three data problems around delistings are handled here, each of which
    fabricated P&L before it was caught:

    1. Settlement timestamps carry millisecond jitter (00:00:00.006). A
       plain ceil() pushes an on-time payment a whole interval forward,
       colliding with the next one. 16% of the panel was holes before the
       timestamps were rounded to the minute first.
    2. Perp and spot stop printing at different times when a contract dies.
       Forward-filling each leg separately freezes one leg while the other
       moves, which looks like a huge basis move that never happened. Both
       legs are cut at their shared last real print instead.
    3. Binance keeps publishing bars for a dead contract: volume 0, price
       frozen. UNFIUSDT's perp sat at 1.730 for seven days while spot kept
       trading. A zero-volume bar has no valid mark, so it becomes NaN and
       the backtester flattens the position.
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
        # Round the jitter away, then ceil: 00:00:00.006 stays at 00:00,
        # a 04:00 payment folds into the 08:00 bucket.
        bucket = ser.index.round("1min").ceil(freq)
        f_cols[s] = ser.groupby(bucket).sum().reindex(grid)
    f = pd.DataFrame(f_cols, index=grid)
    logger.info("panel grid %s: %d rows, %d symbols", freq, len(grid), len(keep))

    def align(d: dict[str, pd.Series]) -> pd.DataFrame:
        cols = {}
        for s in keep:
            ser = d.get(s)
            if ser is None or ser.dropna().empty:
                cols[s] = pd.Series(np.nan, index=grid)
            else:
                cols[s] = ser.dropna().reindex(grid, method="ffill", limit=1)
        return pd.DataFrame(cols, index=grid)

    perp = align(perp_close)
    spot = align(spot_close)
    vol = align(quote_volume) if quote_volume else pd.DataFrame(np.nan, index=grid, columns=keep)

    # Cut both legs at the last bar where both had a real print.
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

    # A bar with no trades has no price.
    if quote_volume:
        dead = vol.fillna(0.0) <= 0.0
        perp = perp.mask(dead)
        spot = spot.mask(dead)

    basis = (perp / spot - 1.0) * 1e4
    return {"funding": f, "perp": perp, "spot": spot, "basis": basis, "dollar_volume": vol}


def carry_signal(
    panel: dict[str, pd.DataFrame],
    lookback: int = 21,              # 7 days of 8h settlements
    top_n: int = 10,
    min_predicted_bps: float = 1.0,
    max_predicted_bps: float = 50.0,
    allow_short_carry: bool = True,
    min_dollar_volume: float = 5e6,
    rebalance_every: int = 21,       # settlements between rebalances (7 days)
    hysteresis: float = 3.0,         # keep a holding while its rank <= top_n * hysteresis
    max_weight_per_symbol: float = 0.15,
    rebalance_offset: int = 0,       # which settlement in the cycle rebalances
) -> pd.DataFrame:
    """Cross-sectional carry weights, indexed at t and held over (t, t+1].

    w > 0 is long spot / short perp (collect positive funding), w < 0 the
    reverse. Equal weight within the basket, capped per symbol, gross at
    most 1.0. Every input to row t uses data through t only.

    Turnover control is what makes this tradeable, not a tuning knob.
    Funding accrues whether or not you trade, so the right book is sticky.
    Re-ranking every 8h turns over ~480x a year and loses money at retail
    fees. Two fixes: only revisit the book every `rebalance_every`
    settlements and keep an incumbent while it is still inside the top
    `top_n * hysteresis`, so a name wobbling around the cutoff is not traded
    in and out. rebalance_every=1, hysteresis=1.0 recovers the naive book.
    `rebalance_offset` shifts which settlements rebalance; results that
    swing with it are calendar luck, not signal.

    The per-symbol cap matters when the book thins out. Without it,
    `sign / len(held)` turned 3 survivors into 33% positions, one of
    them a contract mid-delisting. With it, a thin book remains under-invested.

    `max_predicted_bps` is a risk control. Funding of -250bp per 8h is a
    short squeeze with an unborrowable spot leg, not carry. One such
    episode (TRBUSDT, autumn 2023) was 36% of gross P&L before this gate.
    """
    f = panel["funding"]
    pred = f.ewm(span=lookback, min_periods=max(5, lookback // 3)).mean()
    pred_bps = pred * 1e4

    # Liquidity: a rolling mean can be propped up for weeks by one liquidation
    # spike, so also require that the current bar actually traded.
    vol = panel["dollar_volume"].rolling(lookback, min_periods=5).mean()
    tradeable = (vol >= min_dollar_volume) | vol.isna()
    tradeable &= panel["dollar_volume"].fillna(0.0) > 0.0
    # Listing: the contract must be settling funding right now, or a delisted
    # symbol stays eligible forever on its last EWMA value.
    tradeable &= f.notna() & panel["perp"].notna()

    eligible = (
        tradeable
        & pred_bps.abs().ge(min_predicted_bps)
        & pred_bps.abs().le(max_predicted_bps)
    )
    scores = pred_bps.where(eligible)
    keep_n = max(int(round(top_n * hysteresis)), top_n)
    min_denom = int(np.ceil(1.0 / max_weight_per_symbol)) if max_weight_per_symbol > 0 else 1

    weights = pd.DataFrame(0.0, index=f.index, columns=f.columns)
    held: dict[str, float] = {}  # symbol -> +1 / -1

    def write(t, held):
        if held:
            denom = max(len(held), min_denom)
            weights.loc[t, list(held)] = [sign / denom for sign in held.values()]

    for i, (t, row) in enumerate(scores.iterrows()):
        if (i - rebalance_offset) % rebalance_every != 0:
            write(t, held)
            continue

        row = row.dropna()
        if row.empty:
            held = {}
            continue

        longs = row[row > 0]
        shorts = row[row < 0] if allow_short_carry else row.iloc[:0]
        # Lists, in rank order. Iterating a set here made the book depend on
        # Python's per-process hash seed and changed results run to run.
        fresh_long = list(longs.nlargest(top_n).index)
        fresh_short = list(shorts.nsmallest(top_n).index)
        wide_long = set(longs.nlargest(keep_n).index)
        wide_short = set(shorts.nsmallest(keep_n).index)

        new_held: dict[str, float] = {}
        for s, sign in held.items():  # incumbents survive inside the wide band
            if sign > 0 and s in wide_long:
                new_held[s] = 1.0
            elif sign < 0 and s in wide_short:
                new_held[s] = -1.0
        for s in fresh_long:  # then fill from the fresh top-N
            if len(new_held) >= 2 * top_n:
                break
            new_held.setdefault(s, 1.0)
        for s in fresh_short:
            if len(new_held) >= 2 * top_n:
                break
            new_held.setdefault(s, -1.0)

        held = new_held
        write(t, held)

    return weights


def carry_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return of a unit carry position (w = +1) over (t, t+1], indexed at t.

        f_{t+1} + spot_ret - perp_ret

    fill_method=None is required: older pandas forward-fills before
    differencing, which turns a NaN dead-contract price into a fake 0%
    return. The default changed in pandas 3.0, so the same code gave
    different answers locally and in CI until this was pinned.
    """
    f_next = panel["funding"].shift(-1)
    spot_ret = panel["spot"].pct_change(fill_method=None).shift(-1)
    perp_ret = panel["perp"].pct_change(fill_method=None).shift(-1)
    return (f_next + spot_ret - perp_ret).replace([np.inf, -np.inf], np.nan)
