"""Z-score the spread, then run the entry/exit/stop rules.

Position per pair:
    +1  long the spread (long y, short beta*x), entered when z <= -entry_z
    -1  short the spread,                      entered when z >= +entry_z
     0  flat

Exits:
    reversion   |z| <= exit_z
    hard stop   |z| >= stop_z, and then locked out until |z| falls back
                inside exit_z. A pair that just blew through its stop
                should not be re-entered at an "even better" level.
    time stop   held for max_holding_days
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Z-score against the trailing window. Full-sample stats would peek at the future."""
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std()
    return (spread - mean) / std.replace(0.0, np.nan)


def generate_positions(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_days: int = 40,
) -> pd.DataFrame:
    """Walk the z-score series and decide the position held at each close.

    Returns columns:
        position  -1 / 0 / +1 at the close of each bar. Execution lag is the
                  backtester's job, not this function's.
        event     "" or one of entry_long, entry_short, exit_revert,
                  exit_stop, exit_time
    """
    zv = z.to_numpy()
    n = len(z)
    pos = np.zeros(n)
    events = [""] * n

    cur = 0.0
    holding = 0
    locked_out = False

    for t in range(n):
        zt = zv[t]
        if np.isnan(zt):
            # No signal today: keep the position, keep the clock running.
            if cur != 0:
                holding += 1
            pos[t] = cur
            continue

        if locked_out and abs(zt) <= exit_z:
            locked_out = False

        if cur == 0:
            if not locked_out:
                if zt <= -entry_z:
                    cur, holding = 1.0, 0
                    events[t] = "entry_long"
                elif zt >= entry_z:
                    cur, holding = -1.0, 0
                    events[t] = "entry_short"
        else:
            holding += 1
            stopped = abs(zt) >= stop_z
            reverted = (cur > 0 and zt >= -exit_z) or (cur < 0 and zt <= exit_z)
            if stopped:
                cur = 0.0
                events[t] = "exit_stop"
                locked_out = True
            elif reverted:
                cur = 0.0
                events[t] = "exit_revert"
            elif holding >= max_holding_days:
                cur = 0.0
                events[t] = "exit_time"

        pos[t] = cur

    return pd.DataFrame({"position": pos, "event": events}, index=z.index)
