"""Data quality: explicit, documented handling of missing/bad data.

Policy (documented here, enforced in code, mentioned in the README):

1. Drop tickers whose usable history is shorter than `min_history_days`.
2. Drop tickers with more than `max_missing_pct` missing closes.
3. Forward-fill remaining gaps up to `ffill_limit` bars (holidays, halts) —
   NEVER backfill: backfilling leaks the future into the past.
4. Flag (not silently fix) suspicious returns (>|50%| daily) for manual review;
   at daily frequency on liquid names these are almost always data errors.
5. Everything dropped/flagged is recorded in a QualityReport that gets saved
   next to the data — the audit trail is part of the deliverable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    dropped_short_history: list[str] = field(default_factory=list)
    dropped_too_many_gaps: list[str] = field(default_factory=list)
    filled_gaps: dict[str, int] = field(default_factory=dict)
    suspicious_returns: dict[str, list[str]] = field(default_factory=dict)  # ticker -> dates
    kept: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for t in self.dropped_short_history:
            rows.append({"ticker": t, "issue": "short_history", "detail": ""})
        for t in self.dropped_too_many_gaps:
            rows.append({"ticker": t, "issue": "too_many_gaps", "detail": ""})
        for t, n in self.filled_gaps.items():
            rows.append({"ticker": t, "issue": "gaps_ffilled", "detail": str(n)})
        for t, dates in self.suspicious_returns.items():
            rows.append({"ticker": t, "issue": "suspicious_return", "detail": ",".join(dates)})
        return pd.DataFrame(rows, columns=["ticker", "issue", "detail"])


def clean_prices(
    data: dict[str, pd.DataFrame],
    min_history_days: int = 750,
    max_missing_pct: float = 0.02,
    ffill_limit: int = 5,
    suspicious_abs_return: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], QualityReport]:
    report = QualityReport()
    adj = data["adj_close"]

    keep: list[str] = []
    for t in adj.columns:
        s = adj[t]
        valid = s.dropna()
        if len(valid) < min_history_days:
            report.dropped_short_history.append(t)
            continue
        # missing pct measured inside the ticker's own live range only
        live = s.loc[valid.index.min(): valid.index.max()]
        miss_pct = live.isna().mean()
        if miss_pct > max_missing_pct:
            report.dropped_too_many_gaps.append(t)
            continue
        n_gaps = int(live.isna().sum())
        if n_gaps:
            report.filled_gaps[t] = n_gaps
        keep.append(t)

    cleaned: dict[str, pd.DataFrame] = {}
    for fld, df in data.items():
        sub = df[keep].copy()
        if fld != "volume":
            sub = sub.ffill(limit=ffill_limit)  # forward only — no lookahead
        cleaned[fld] = sub

    rets = cleaned["adj_close"].pct_change(fill_method=None)
    hits = rets.abs() > suspicious_abs_return
    for t in keep:
        dates = rets.index[hits[t].fillna(False)]
        if len(dates):
            report.suspicious_returns[t] = [str(d.date()) for d in dates]
            logger.warning("Suspicious daily returns for %s on %s", t, list(dates))

    report.kept = keep
    logger.info(
        "Quality: kept %d, dropped %d (short history), %d (gaps)",
        len(keep), len(report.dropped_short_history), len(report.dropped_too_many_gaps),
    )
    return cleaned, report


def align_to_trading_calendar(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict all fields to dates where at least 90% of names traded.

    Guards against provider glitches that emit a near-empty bar (e.g. one
    ticker printing on a holiday), which would otherwise create fake gaps
    in everything else.
    """
    close = data["close"]
    good_days = close.notna().mean(axis=1) >= 0.9
    idx = close.index[good_days]
    return {k: v.loc[idx] for k, v in data.items()}
