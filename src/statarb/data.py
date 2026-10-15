"""Prices in, clean wide DataFrames out.

Everything downstream works on one shape:

    dict[field -> DataFrame(index=dates, columns=tickers)]

with fields open, high, low, close, adj_close, volume. Signal math uses
adj_close (split and dividend adjusted; cointegration on raw closes would
"detect" every dividend as mean reversion). Capacity uses close * volume,
because impact depends on the notional that actually printed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]


# --- universe --------------------------------------------------------------

def load_universe(path: str | Path) -> dict[str, list[str]]:
    """Read configs/universe.csv into sector -> sorted tickers.

    The file is a static list of large, long-lived names. That is a
    survivorship tilt and it is stated in docs/LIMITATIONS.md.
    """
    df = pd.read_csv(path)
    missing = {"ticker", "sector"} - set(df.columns)
    if missing:
        raise ValueError(f"universe file {path} missing columns: {missing}")
    sectors: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        sectors.setdefault(str(row.sector).strip(), []).append(str(row.ticker).strip())
    return {k: sorted(set(v)) for k, v in sectors.items()}


def all_tickers(sectors: dict[str, list[str]]) -> list[str]:
    return sorted({t for ts in sectors.values() for t in ts})


# --- download --------------------------------------------------------------

def download_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    logger.info("Downloading %d tickers from yfinance (%s to %s)", len(tickers), start, end)
    # auto_adjust=False keeps both the adjusted and the unadjusted close.
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False,
        progress=True, group_by="column", threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data; check tickers, dates, connectivity")

    names = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    }
    out: dict[str, pd.DataFrame] = {}
    for src, dst in names.items():
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[src]
        else:  # single-ticker download comes back flat
            df = raw[[src]]
            df.columns = tickers
        df = df.sort_index()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        out[dst] = df
    return out


# --- cleaning --------------------------------------------------------------

@dataclass
class QualityReport:
    """What was dropped, filled or flagged. Saved next to the data."""

    dropped_short_history: list[str] = field(default_factory=list)
    dropped_too_many_gaps: list[str] = field(default_factory=list)
    filled_gaps: dict[str, int] = field(default_factory=dict)
    suspicious_returns: dict[str, list[str]] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"ticker": t, "issue": "short_history", "detail": ""}
                for t in self.dropped_short_history]
        rows += [{"ticker": t, "issue": "too_many_gaps", "detail": ""}
                 for t in self.dropped_too_many_gaps]
        rows += [{"ticker": t, "issue": "gaps_ffilled", "detail": str(n)}
                 for t, n in self.filled_gaps.items()]
        rows += [{"ticker": t, "issue": "suspicious_return", "detail": ",".join(d)}
                 for t, d in self.suspicious_returns.items()]
        return pd.DataFrame(rows, columns=["ticker", "issue", "detail"])


def align_to_trading_calendar(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep only dates where at least 90% of names traded.

    A provider glitch that prints one ticker on a holiday would otherwise
    create a fake gap in every other name.
    """
    close = data["close"]
    idx = close.index[close.notna().mean(axis=1) >= 0.9]
    return {k: v.loc[idx] for k, v in data.items()}


def clean_prices(
    data: dict[str, pd.DataFrame],
    min_history_days: int = 750,
    max_missing_pct: float = 0.02,
    ffill_limit: int = 5,
    suspicious_abs_return: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], QualityReport]:
    """Apply the data policy and record what it did.

    1. Drop names with fewer than `min_history_days` usable bars.
    2. Drop names missing more than `max_missing_pct` of bars in their own live range.
    3. Forward-fill small gaps (holidays, halts). Never backfill: that leaks the future.
    4. Flag, but don't touch, daily moves beyond `suspicious_abs_return`.
    """
    report = QualityReport()
    adj = data["adj_close"]

    keep: list[str] = []
    for t in adj.columns:
        valid = adj[t].dropna()
        if len(valid) < min_history_days:
            report.dropped_short_history.append(t)
            continue
        live = adj[t].loc[valid.index.min(): valid.index.max()]
        if live.isna().mean() > max_missing_pct:
            report.dropped_too_many_gaps.append(t)
            continue
        if live.isna().any():
            report.filled_gaps[t] = int(live.isna().sum())
        keep.append(t)

    cleaned: dict[str, pd.DataFrame] = {}
    for fld, df in data.items():
        sub = df[keep].copy()
        if fld != "volume":
            sub = sub.ffill(limit=ffill_limit)
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


# --- storage ---------------------------------------------------------------

class PriceStore:
    """One parquet file per field in `cache_dir`, plus a manifest.json.

    A few MB at 100 names and ten years of daily bars, so no database.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, pd.DataFrame], meta: dict | None = None) -> None:
        for fld, df in data.items():
            df.to_parquet(self.cache_dir / f"{fld}.parquet")
        first = next(iter(data.values()))
        manifest = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "fields": sorted(data),
            "tickers": sorted(first.columns.tolist()),
            "start": str(first.index.min().date()),
            "end": str(first.index.max().date()),
            **(meta or {}),
        }
        (self.cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Saved %s to %s", sorted(data), self.cache_dir)

    def load(self, fields: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for fld in fields:
            path = self.cache_dir / f"{fld}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run scripts/01_download_data.py first")
            out[fld] = pd.read_parquet(path)
        return out
