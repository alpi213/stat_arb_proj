"""Crypto step 0: build a POINT-IN-TIME perp universe (fixes survivorship).

`list_perp_symbols()` returns contracts listed TODAY, so any backtest built on
it silently excludes every perp that died — and perps die precisely when carry
blows up (LUNA, FTT, SRM). That biases results upward by an unknown amount.
This script measures that amount instead of hand-waving at it.

Source: Binance's own public data archive at data.binance.vision, which keeps
a directory per symbol that EVER traded, delisted ones included. This is
authoritative rather than reconstructed — strictly better than scraping
archived `exchangeInfo` responses out of the Wayback Machine, which is
rate-limited, frequently truncated, and only as complete as the crawl.

Listing the archive gives the full symbol set; the live `fundingRate` endpoint
still serves history for delisted symbols, so first/last settlement timestamps
give each contract's true active window.

Output: configs/universe_perp_pit.csv with
    symbol, first_funding, last_funding, delisted, active_days

Usage:
    python scripts/14_build_pit_universe.py [--quote USDT]
"""

from __future__ import annotations

import argparse
import logging
import re

import pandas as pd
import requests

from statarb.crypto.binance import FAPI, BinanceClient  # noqa: F401
from statarb.utils import setup_logging

logger = logging.getLogger("pit_universe")

ARCHIVE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/fundingRate/"


def list_all_symbols_ever(quote: str = "USDT") -> list[str]:
    """Every UM perp symbol that ever had funding data, from the archive."""
    r = requests.get(ARCHIVE, params={"delimiter": "/", "prefix": PREFIX}, timeout=60)
    r.raise_for_status()
    syms = re.findall(rf"<Prefix>{re.escape(PREFIX)}([^/]+)/</Prefix>", r.text)
    if re.search(r"<IsTruncated>true</IsTruncated>", r.text):
        logger.warning("archive listing truncated — symbol set is incomplete")
    return sorted(s for s in syms if s.endswith(quote))


def live_symbols(client: BinanceClient) -> set[str]:
    info = client.get(f"{FAPI}/fapi/v1/exchangeInfo", {})
    return {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL"}


def archive_active_months() -> dict[str, list[str]]:
    """symbol -> sorted list of YYYY-MM months that have funding data.

    Derived by listing every key under the archive prefix and parsing the
    filenames (…/BTCUSDT/BTCUSDT-fundingRate-2019-09.zip). One paginated sweep
    (~30 requests) replaces 2x833 API probes, and is authoritative about when
    each contract actually traded.

    Do NOT try to get the first settlement from the REST API with
    `startTime=0`: Binance silently IGNORES it and returns the LATEST record,
    so first and last come back identical and every contract looks like it
    lived for zero days. That bug produced the first version of this file.
    """
    months: dict[str, list[str]] = {}
    token = None
    pat = re.compile(
        rf"<Key>{re.escape(PREFIX)}([^/]+)/\1-fundingRate-(\d{{4}}-\d{{2}})\.zip</Key>"
    )
    while True:
        params = {"prefix": PREFIX, "list-type": "2", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(ARCHIVE, params=params, timeout=60)
        r.raise_for_status()
        for sym, ym in pat.findall(r.text):
            months.setdefault(sym, []).append(ym)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return {k: sorted(v) for k, v in months.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quote", default="USDT")
    p.add_argument("--out", default="configs/universe_perp_pit.csv")
    args = p.parse_args()
    setup_logging()

    client = BinanceClient(sleep_between=0.12)
    everything = list_all_symbols_ever(args.quote)
    live = live_symbols(client)
    logger.info("archive: %d %s perps ever; %d currently listed; %d delisted",
                len(everything), args.quote, len(set(everything) & live),
                len(set(everything) - live))

    months = archive_active_months()
    logger.info("archive listing: month coverage for %d symbols", len(months))

    rows = []
    for sym in everything:
        ms = months.get(sym)
        if not ms:
            continue
        first = pd.Timestamp(ms[0] + "-01")
        # last active month -> end of that month
        last = pd.Timestamp(ms[-1] + "-01") + pd.offsets.MonthEnd(1)
        rows.append({
            "symbol": sym,
            "first_funding": first,
            "last_funding": last,
            "delisted": sym not in live,
            "active_days": (last - first).days,
            "n_months": len(ms),
        })

    df = pd.DataFrame(rows).sort_values("first_funding").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    n_del = int(df["delisted"].sum())
    logger.info("wrote %s: %d symbols (%d delisted, %.0f%%)",
                args.out, len(df), n_del, 100 * n_del / max(len(df), 1))
    # How many died mid-sample? Those are the ones a survivor-only universe hides.
    died = df[df["delisted"] & (df["last_funding"] >= "2022-01-01")]
    logger.info("delisted with activity after 2022-01-01 (the hidden ones): %d", len(died))
    logger.info("\n%s", died.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
