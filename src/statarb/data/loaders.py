"""Price download adapters.

All providers return the same canonical shape so the rest of the codebase is
provider-agnostic:

    dict[field -> DataFrame(index=DatetimeIndex, columns=tickers)]

with fields: "open", "high", "low", "close", "adj_close", "volume".

Prices are split- AND dividend-adjusted via `adj_close` (used for signal
math), while unadjusted `close` is kept for realistic notional/volume-based
capacity calculations.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    provider: str = "yfinance",
) -> dict[str, pd.DataFrame]:
    if provider == "yfinance":
        return _download_yfinance(tickers, start, end)
    if provider == "stooq":
        return _download_stooq(tickers, start, end)
    raise ValueError(f"Unknown provider: {provider!r}")


def _download_yfinance(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    logger.info("Downloading %d tickers from yfinance (%s to %s)", len(tickers), start, end)
    # auto_adjust=False so we get BOTH adjusted and unadjusted closes.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=True,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data — check tickers/dates/connectivity")

    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out: dict[str, pd.DataFrame] = {}
    for src, dst in mapping.items():
        df = raw[src] if isinstance(raw.columns, pd.MultiIndex) else raw[[src]]
        if not isinstance(raw.columns, pd.MultiIndex):
            df.columns = tickers  # single-ticker download
        out[dst] = df.sort_index()
        out[dst].index = pd.DatetimeIndex(out[dst].index).tz_localize(None)
    return out


def _download_stooq(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Stooq fallback via pandas-datareader.

    NOTE: Stooq US symbols use a `.US` suffix and data is split-adjusted but
    NOT dividend-adjusted — acceptable for a robustness check, not for the
    headline backtest. TODO: implement if yfinance becomes unreliable.
    """
    raise NotImplementedError(
        "Stooq adapter not implemented yet. Use provider='yfinance', or implement "
        "via pandas_datareader.DataReader(f'{ticker}.US', 'stooq', ...)."
    )
