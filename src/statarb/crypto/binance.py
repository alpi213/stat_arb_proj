"""Binance public REST endpoints. No API key, no account access.

Rate limits are generous for this (2400 weight/min on futures); a short
sleep between pages and a retry on 429/418 is enough. fapi.binance.com is
blocked in some countries and returns HTTP 451 there.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"  # USD-M perpetual futures
SAPI = "https://api.binance.com"   # spot

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


@dataclass
class BinanceClient:
    timeout: int = 20
    max_retries: int = 5
    sleep_between: float = 0.25

    def get(self, url: str, params: dict) -> list | dict:
        for attempt in range(self.max_retries):
            try:
                r = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning("request error (%s), retry %d", exc, attempt + 1)
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                time.sleep(self.sleep_between)
                return r.json()
            if r.status_code in (429, 418):
                wait = int(r.headers.get("Retry-After", 2**attempt))
                logger.warning("rate limited, sleeping %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code == 451:
                raise RuntimeError("HTTP 451: Binance blocks this jurisdiction")
            raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
        raise RuntimeError(f"exhausted retries for {url}")


def fetch_funding_history(
    symbol: str, start: str, end: str, client: BinanceClient | None = None
) -> pd.Series:
    """Funding rate actually paid at each settlement. Index is the settlement time (UTC).

    This is a realized cash flow, so using f_t at time t is not look-ahead.
    Predicting f_{t+1} is the trading problem.
    """
    client = client or BinanceClient()
    t0 = int(pd.Timestamp(start).timestamp() * 1000)
    t_end = int(pd.Timestamp(end).timestamp() * 1000)
    out = []
    while t0 < t_end:
        batch = client.get(
            f"{FAPI}/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": t0, "endTime": t_end, "limit": 1000},
        )
        if not batch:
            break
        out.extend(batch)
        last = batch[-1]["fundingTime"]
        if last <= t0 or len(batch) < 1000:
            break
        t0 = last + 1
    if not out:
        return pd.Series(dtype=float, name=symbol)
    df = pd.DataFrame(out)
    s = pd.Series(
        df["fundingRate"].astype(float).to_numpy(),
        index=pd.to_datetime(df["fundingTime"], unit="ms"),
        name=symbol,
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_klines(
    symbol: str,
    start: str,
    end: str,
    interval: str = "8h",
    market: str = "perp",
    client: BinanceClient | None = None,
) -> pd.DataFrame:
    """OHLCV bars from the perp (fapi) or spot (api) market, indexed by bar CLOSE time.

    The close of the bar [T, T+interval) is not known until T+interval.
    Indexing it at T would put a future price on the current timestamp.
    """
    client = client or BinanceClient()
    base, path = (FAPI, "/fapi/v1/klines") if market == "perp" else (SAPI, "/api/v3/klines")
    # Page caps differ: futures 1500, spot 1000. Using the wrong one makes the
    # "short page means done" check fire on a full page and truncate history.
    limit = 1500 if market == "perp" else 1000

    t0 = int(pd.Timestamp(start).timestamp() * 1000)
    t_end = int(pd.Timestamp(end).timestamp() * 1000)
    rows = []
    while t0 < t_end:
        batch = client.get(
            f"{base}{path}",
            {"symbol": symbol, "interval": interval, "startTime": t0, "endTime": t_end,
             "limit": limit},
        )
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= t0 or len(batch) < limit:
            break
        t0 = last_open + 1
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])

    df = pd.DataFrame(rows, columns=_KLINE_COLS)
    # Binance close_time is T+interval minus 1ms; ceil restores the clean boundary.
    close_time = pd.to_datetime(df["close_time"], unit="ms").dt.ceil("s")
    df.index = pd.DatetimeIndex(close_time).ceil(interval)
    df.index.name = "close_time"
    df = df[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()
