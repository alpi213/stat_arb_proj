"""Regime analysis: does the strategy survive the periods that break pairs?

Pairs strategies characteristically fail in exactly two regimes:
- liquidity crises (2020-03): spreads blow through stops before reverting,
  and shorts get expensive/recalled;
- violent factor rotations (2022): "cointegrated" pairs that were secretly
  a rates/growth factor bet decouple together.

Reporting per-regime metrics is how you show you know that.
"""

from __future__ import annotations

import pandas as pd

from statarb.backtest.metrics import performance_summary


def regime_breakdown(
    returns: pd.Series,
    regimes: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Metrics per named regime window plus the full sample.

    regimes: name -> (start, end) ISO dates, e.g. from config
             validation.regimes.
    """
    rows = {"full_sample": performance_summary(returns)}
    for name, (start, end) in regimes.items():
        sub = returns.loc[pd.Timestamp(start): pd.Timestamp(end)]
        if len(sub) < 5:
            continue
        rows[name] = performance_summary(sub)
    return pd.DataFrame(rows)
