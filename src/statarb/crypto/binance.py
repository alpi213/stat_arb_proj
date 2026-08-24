"""Binance public REST adapters — no API key required.

All endpoints used here are public market data. We never touch authenticated
(account/trading) endpoints; this is a research tool, not a trading bot.

Rate limits: fapi allows 2400 request-weight/min. klines weight ~2-10 per
call, fundingRate weight 1. We page with a modest sleep and retry on 429/418.

Geo note: fapi.binance.com is blocked from some jurisdictions (notably US).
If you hit HTTP 451, use a permitted region or swap in Bybit/OKX — the
adapter shape below is deliberately thin so another venue is a small change.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"      # USD-M perpetual futures
SAPI = "https://api.binance.com"       # spot

# Binance kline column layout (same for spot and futures)
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
            if r.status_code in (429, 418):  # rate limited / banned-temporarily
                wait = int(r.headers.get("Retry-After", 2**attempt))
                logger.warning("rate limited, sleeping %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code == 451:
                raise RuntimeError(
                    "HTTP 451: Binance blocks this jurisdiction. Use a permitted "
                    "region, or switch the adapter to Bybit/OKX."
                )
            raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
        raise RuntimeError(f"exhausted retries for {url}")


def list_perp_symbols(
    client: BinanceClient | None = None,
    quote: str = "USDT",
    min_onboard_days: int = 400,
) -> pd.DataFrame:
    """Currently-listed USD-M perpetual contracts.

    IMPORTANT survivorship caveat: this returns symbols listed TODAY. Perps
    that were delisted (many small-caps were) are absent, which biases
    backtests upward exactly as stale equity index membership does. The
    `onboardDate` filter at least avoids trading a symbol before it existed.
    Documented in docs/CRYPTO_LIMITATIONS.md.
    """
    client = client or BinanceClient()
    info = client.get(f"{FAPI}/fapi/v1/exchangeInfo", {})
    rows = []
    for s in info["symbols"]:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == quote
            and s.get("status") == "TRADING"
        ):
            onboard = pd.to_datetime(s["onboardDate"], unit="ms")
            rows.append(
                {
                    "symbol": s["symbol"],
                    "base": s["baseAsset"],
                    "onboard_date": onboard,
                    "age_days": (pd.Timestamp.utcnow().tz_localize(None) - onboard).days,
                }
            )
    df = pd.DataFrame(rows).sort_values("onboard_date")
    return df[df["age_days"] >= min_onboard_days].reset_index(drop=True)


def fetch_funding_history(
    symbol: str,
    start: str,
    end: str,
    client: BinanceClient | None = None,
) -> pd.Series:
    """Realized funding rate per settlement (usually 8h). Index = settlement UTC time.

    This is the rate ACTUALLY PAID at that timestamp — a realized cash flow,
    not a forecast. Using it as of its own timestamp is therefore not
    look-ahead, but predicting the NEXT one is the actual trading problem.
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
        if last <= t0:
            break
        t0 = last + 1
        if len(batch) < 1000:
            break
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
    """OHLCV bars. market: "perp" (fapi) or "spot" (api). Index = bar OPEN time UTC."""
    client = client or BinanceClient()
    base = FAPI if market == "perp" else SAPI
    path = "/fapi/v1/klines" if market == "perp" else "/api/v3/klines"
    # Page size caps differ per venue: futures allows 1500, spot only 1000.
    # Using the wrong cap makes the short-page break fire on a FULL page and
    # silently truncates history (spot stopped at exactly 1000 bars).
    limit = 1500 if market == "perp" else 1000
    t0 = int(pd.Timestamp(start).timestamp() * 1000)
    t_end = int(pd.Timestamp(end).timestamp() * 1000)
    rows = []
    while t0 < t_end:
        batch = client.get(
            f"{base}{path}",
            {
                "symbol": symbol, "interval": interval,
                "startTime": t0, "endTime": t_end, "limit": limit,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= t0:
            break
        t0 = last_open + 1
        if len(batch) < limit:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])
    df = pd.DataFrame(rows, columns=_KLINE_COLS)
    # Index by CLOSE time, not open time. The `close` price of the bar
    # [T, T+interval) is not observable until T+interval, so indexing it at T
    # would place a future price at the current timestamp — a look-ahead bug
    # that silently flatters any strategy using these prices for returns.
    # Binance close_time is T+interval-1ms; ceil() restores the clean boundary.
    close_time = pd.to_datetime(df["close_time"], unit="ms").dt.ceil("s")
    df.index = pd.DatetimeIndex(close_time).ceil(interval)
    df.index.name = "close_time"
    keep = ["open", "high", "low", "close", "volume", "quote_volume"]
    df = df[keep].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()
