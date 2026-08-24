"""Spread standardization and the entry/exit/stop state machine.

Position convention (per unit of pair):
    +1 = long the spread  (long y, short beta*x) — entered when z <= -entry_z
    -1 = short the spread (short y, long beta*x) — entered when z >= +entry_z
     0 = flat

Exits:
    - reversion:  |z| <= exit_z
    - hard stop:  |z| >= stop_z  -> flat AND locked out until z re-crosses
                  exit_z (a broken pair shouldn't be immediately re-entered
                  at an "even better" level — that's how blowups compound)
    - time stop:  held longer than max_holding_days
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    spread: pd.Series, window: int = 60, min_periods: int | None = None
) -> pd.Series:
    """Z-score of the spread against its own trailing window.

    Rolling (not full-sample) stats: the full-sample mean/std would use
    future data. min_periods defaults to the full window — no z-scores
    from half-formed windows.
    """
    mp = window if min_periods is None else min_periods
    mean = spread.rolling(window, min_periods=mp).mean()
    std = spread.rolling(window, min_periods=mp).std()
    return (spread - mean) / std.replace(0.0, np.nan)


def generate_positions(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_days: int = 40,
) -> pd.DataFrame:
    """Run the state machine over a z-score series.

    Returns DataFrame with:
        position: -1/0/+1 held AT THE CLOSE of each bar (the backtester
                  applies the execution lag; this module knows nothing
                  about execution)
        event:    "" | "entry_long" | "entry_short" | "exit_revert"
                  | "exit_stop" | "exit_time"
    """
    zv = z.to_numpy()
    n = len(z)
    pos = np.zeros(n, dtype=float)
    events = [""] * n

    holding = 0
    locked_out = False
    prev = 0.0

    for t in range(n):
        cur = prev
        zt = zv[t]
        if np.isnan(zt):
            # No signal — hold state but keep the clock ticking if invested.
            if cur != 0:
                holding += 1
            pos[t] = cur
            prev = cur
            continue

        if locked_out and abs(zt) <= exit_z:
            locked_out = False

        if cur == 0:
            if not locked_out:
                if zt <= -entry_z:
                    cur, holding = +1.0, 0
                    events[t] = "entry_long"
                elif zt >= entry_z:
                    cur, holding = -1.0, 0
                    events[t] = "entry_short"
        else:
            holding += 1
            stopped = abs(zt) >= stop_z
            reverted = (cur > 0 and zt >= -exit_z) or (cur < 0 and zt <= exit_z)
            timed_out = holding >= max_holding_days
            if stopped:
                cur = 0.0
                events[t] = "exit_stop"
                locked_out = True
            elif reverted:
                cur = 0.0
                events[t] = "exit_revert"
            elif timed_out:
                cur = 0.0
                events[t] = "exit_time"

        pos[t] = cur
        prev = cur

    return pd.DataFrame({"position": pos, "event": events}, index=z.index)
