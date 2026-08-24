"""Walk-forward windowing.

The single most important anti-overfitting device in this project: pairs are
re-selected on each formation window, then traded on the immediately
following out-of-sample window. Stitching the out-of-sample segments
together gives an equity curve where NO pair was ever chosen with knowledge
of the period it traded in.

This module only builds the windows; orchestration lives in
scripts/05_walkforward.py so the window logic stays trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


def make_windows(
    dates: pd.DatetimeIndex,
    formation_days: int = 504,
    trading_days: int = 126,
    step_days: int = 126,
) -> list[WalkForwardWindow]:
    """Build rolling formation/trading splits over a trading calendar.

    All arithmetic is in TRADING days (index positions), not calendar days.
    The final window is dropped if its trading segment would be shorter
    than half of `trading_days` (a stub window's metrics are noise).
    """
    windows = []
    i = 0
    n = len(dates)
    while True:
        f_start = i
        f_end = i + formation_days - 1
        t_start = f_end + 1
        t_end = min(t_start + trading_days - 1, n - 1)
        if t_start >= n or (t_end - t_start + 1) < trading_days // 2:
            break
        windows.append(
            WalkForwardWindow(
                formation_start=dates[f_start],
                formation_end=dates[f_end],
                trading_start=dates[t_start],
                trading_end=dates[t_end],
            )
        )
        i += step_days
    return windows
