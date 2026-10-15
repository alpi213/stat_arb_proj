"""Download prices, clean them, save them to the parquet store.

    python scripts/01_download_data.py [--config configs/base.yaml]
"""

import argparse
import logging

from statarb.config import load_config
from statarb.data import (
    PriceStore,
    align_to_trading_calendar,
    all_tickers,
    clean_prices,
    download_prices,
    load_universe,
)
from statarb.utils import setup_logging

logger = logging.getLogger("download")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    sectors = load_universe(cfg.data.universe_file)
    tickers = all_tickers(sectors)
    logger.info("Universe: %d tickers, %d sectors", len(tickers), len(sectors))

    data = download_prices(tickers, cfg.data.start, cfg.data.end)
    data = align_to_trading_calendar(data)
    cleaned, report = clean_prices(data, cfg.data.min_history_days, cfg.data.max_missing_pct)

    store = PriceStore(cfg.data.cache_dir)
    store.save(cleaned, meta={"source": "yfinance", "config": args.config})
    report.to_frame().to_csv(cfg.data.cache_dir / "quality_report.csv", index=False)
    logger.info("Done. %d tickers usable.", len(report.kept))


if __name__ == "__main__":
    main()
