"""Crypto step 1: download perp funding + perp/spot klines from Binance public API.

Only symbols that have BOTH a USD-M perpetual and a spot pair are kept —
the carry trade needs both legs.

Usage:
    python scripts/10_download_crypto.py [--n-symbols 40] [--start 2022-01-01]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from statarb.crypto.binance import (
    BinanceClient,
    fetch_funding_history,
    fetch_klines,
    list_perp_symbols,
)
from statarb.utils import setup_logging

logger = logging.getLogger("crypto_dl")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-symbols", type=int, default=40)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/crypto")
    p.add_argument("--min-onboard-days", type=int, default=500)
    p.add_argument("--symbols-file", default=None,
                   help="point-in-time universe CSV from scripts/14_build_pit_universe.py. "
                        "Selects symbols ACTIVE AT --start including since-delisted ones, "
                        "which is what removes survivorship bias.")
    args = p.parse_args()
    setup_logging()

    end = args.end or pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = BinanceClient()

    if args.symbols_file:
        # Point-in-time selection: everything that already existed at --start,
        # oldest first. Symbols that later died are KEPT — excluding them is
        # precisely the survivorship bias this path exists to remove.
        pit = pd.read_csv(args.symbols_file, parse_dates=["first_funding", "last_funding"])
        pit = pit[pit["first_funding"] <= pd.Timestamp(args.start)]
        pit = pit.sort_values("first_funding")
        perps = pit.rename(columns={"first_funding": "onboard_date"})[
            ["symbol", "onboard_date", "delisted", "last_funding"]
        ]
        logger.info("PIT universe: %d symbols active at %s (%d since delisted)",
                    len(perps), args.start, int(perps["delisted"].sum()))
    else:
        perps = list_perp_symbols(client, min_onboard_days=args.min_onboard_days)
        logger.info("%d perps meet the age filter (SURVIVOR-ONLY)", len(perps))

    funding, perp_px, spot_px, qvol = {}, {}, {}, {}
    kept, skipped = [], []
    for row in perps.itertuples(index=False):
        if len(kept) >= args.n_symbols:
            break
        sym = row.symbol
        try:
            spot = fetch_klines(sym, args.start, end, "8h", "spot", client)
            if spot.empty:
                skipped.append((sym, "no spot pair"))
                continue
            perp = fetch_klines(sym, args.start, end, "8h", "perp", client)
            f = fetch_funding_history(sym, args.start, end, client)
            if perp.empty or f.empty:
                skipped.append((sym, "no perp/funding"))
                continue
            funding[sym], perp_px[sym], spot_px[sym] = f, perp["close"], spot["close"]
            qvol[sym] = perp["quote_volume"]
            kept.append(sym)
            logger.info("[%2d/%d] %-14s funding=%d bars perp=%d spot=%d",
                        len(kept), args.n_symbols, sym, len(f), len(perp), len(spot))
        except Exception as exc:
            skipped.append((sym, str(exc)[:80]))
            logger.warning("%s failed: %s", sym, str(exc)[:120])

    if not kept:
        raise SystemExit("no symbols downloaded — check connectivity / geo-restrictions")

    pd.DataFrame(funding).sort_index().to_parquet(out / "funding.parquet")
    pd.DataFrame(perp_px).sort_index().to_parquet(out / "perp_close.parquet")
    pd.DataFrame(spot_px).sort_index().to_parquet(out / "spot_close.parquet")
    pd.DataFrame(qvol).sort_index().to_parquet(out / "quote_volume.parquet")
    n_delisted = 0
    if args.symbols_file:
        dmap = dict(zip(perps["symbol"], perps["delisted"], strict=False))
        n_delisted = sum(1 for s_ in kept if dmap.get(s_, False))
    (out / "manifest.json").write_text(json.dumps({
        "symbols": kept, "skipped": skipped,
        "start": args.start, "end": end, "interval": "8h",
        "source": "binance public API",
        "point_in_time": bool(args.symbols_file),
        "n_delisted_included": n_delisted,
    }, indent=2))
    if args.symbols_file:
        logger.info("included %d since-delisted symbols (survivorship-corrected)", n_delisted)
    logger.info("saved %d symbols to %s (skipped %d)", len(kept), out, len(skipped))


if __name__ == "__main__":
    main()
