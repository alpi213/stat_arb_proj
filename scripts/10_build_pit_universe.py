"""Build a point-in-time perp universe, including contracts that have since died.

Binance's live exchangeInfo only lists contracts trading today. Perps get
delisted precisely when carry blows up (LUNA, FTT, SRM), so a universe
built from it is biased upward by an unknown amount. Binance's public data
archive (data.binance.vision) keeps a folder per symbol that ever traded,
with one funding file per month, which gives each contract's true active
window without any API probing.

Do not try to get a contract's first settlement from the REST API with
startTime=0: Binance ignores it and returns the latest record, so every
contract looks like it lived for zero days.

Writes configs/universe_perp_pit.csv with
    symbol, first_funding, last_funding, delisted, active_days, n_months

    python scripts/10_build_pit_universe.py [--quote USDT]
"""

from __future__ import annotations

import argparse
import logging
import re

import pandas as pd
import requests

from statarb.crypto.binance import FAPI, BinanceClient
from statarb.utils import setup_logging

logger = logging.getLogger("pit_universe")

ARCHIVE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/fundingRate/"


def archive_active_months() -> dict[str, list[str]]:
    """symbol -> sorted YYYY-MM months with funding data, from one paginated listing."""
    months: dict[str, list[str]] = {}
    pat = re.compile(
        rf"<Key>{re.escape(PREFIX)}([^/]+)/\1-fundingRate-(\d{{4}}-\d{{2}})\.zip</Key>"
    )
    token = None
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


def live_symbols(client: BinanceClient) -> set[str]:
    info = client.get(f"{FAPI}/fapi/v1/exchangeInfo", {})
    return {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quote", default="USDT")
    p.add_argument("--out", default="configs/universe_perp_pit.csv")
    args = p.parse_args()
    setup_logging()

    months = archive_active_months()
    live = live_symbols(BinanceClient(sleep_between=0.12))

    rows = []
    for sym, ms in sorted(months.items()):
        if not sym.endswith(args.quote):
            continue
        first = pd.Timestamp(ms[0] + "-01")
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
    logger.info("wrote %s: %d %s perps ever listed, %d delisted",
                args.out, len(df), args.quote, n_del)
    died = df[df["delisted"] & (df["last_funding"] >= "2022-01-01")]
    logger.info("delisted after 2022-01-01 (the ones a survivor universe hides): %d", len(died))


if __name__ == "__main__":
    main()
