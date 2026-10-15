"""Download funding, perp closes and spot closes from Binance's public API.

Symbols come from the point-in-time universe file, so contracts that were
active at --start but have since been delisted are included. Pass
--survivors-only to drop them and measure the survivorship bias.

    python scripts/11_download_crypto.py --n-symbols 141 --start 2022-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from statarb.crypto.binance import BinanceClient, fetch_funding_history, fetch_klines
from statarb.utils import setup_logging

logger = logging.getLogger("crypto_dl")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols-file", default="configs/universe_perp_pit.csv",
                   help="from scripts/10_build_pit_universe.py")
    p.add_argument("--n-symbols", type=int, default=None, help="cap, oldest contracts first")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/crypto")
    p.add_argument("--survivors-only", action="store_true",
                   help="drop since-delisted contracts (reintroduces survivorship bias)")
    args = p.parse_args()
    setup_logging()

    end = args.end or pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = BinanceClient()

    pit = pd.read_csv(args.symbols_file, parse_dates=["first_funding", "last_funding"])
    pit = pit[pit["first_funding"] <= pd.Timestamp(args.start)].sort_values("first_funding")
    if args.survivors_only:
        pit = pit[~pit["delisted"]]
    logger.info("%d symbols active at %s (%d since delisted)",
                len(pit), args.start, int(pit["delisted"].sum()))

    funding, perp_px, spot_px, qvol = {}, {}, {}, {}
    kept, skipped = [], []
    for row in pit.itertuples(index=False):
        if args.n_symbols and len(kept) >= args.n_symbols:
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
            logger.info("[%3d] %-14s funding=%d perp=%d spot=%d", len(kept), sym,
                        len(f), len(perp), len(spot))
        except Exception as exc:
            skipped.append((sym, str(exc)[:80]))
            logger.warning("%s failed: %s", sym, str(exc)[:120])

    if not kept:
        raise SystemExit("no symbols downloaded; check connectivity / geo-restrictions")

    pd.DataFrame(funding).sort_index().to_parquet(out / "funding.parquet")
    pd.DataFrame(perp_px).sort_index().to_parquet(out / "perp_close.parquet")
    pd.DataFrame(spot_px).sort_index().to_parquet(out / "spot_close.parquet")
    pd.DataFrame(qvol).sort_index().to_parquet(out / "quote_volume.parquet")

    delisted = set(pit.loc[pit["delisted"], "symbol"])
    n_delisted = sum(s in delisted for s in kept)
    (out / "manifest.json").write_text(json.dumps({
        "symbols": kept, "skipped": skipped, "start": args.start, "end": end,
        "interval": "8h", "source": "binance public API",
        "survivors_only": args.survivors_only, "n_delisted_included": n_delisted,
    }, indent=2))
    logger.info("saved %d symbols (%d since delisted) to %s; skipped %d",
                len(kept), n_delisted, out, len(skipped))


if __name__ == "__main__":
    main()
