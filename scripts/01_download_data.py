"""Step 1-2: download prices, clean them, save to the parquet store.

Usage:
    python scripts/01_download_data.py [--config configs/base.yaml]
"""

import argparse
import logging

from statarb.config import load_config
from statarb.data import PriceStore, clean_prices, download_prices, load_universe
from statarb.data.quality import align_to_trading_calendar
from statarb.utils import setup_logging

logger = logging.getLogger("download")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    universe = load_universe(cfg.data.universe_file)
    logger.info("Universe: %d tickers, %d sectors", len(universe.tickers), len(universe.sectors()))

    data = download_prices(
        universe.tickers, cfg.data.start, cfg.data.end, provider=cfg.data.provider
    )
    data = align_to_trading_calendar(data)
    cleaned, report = clean_prices(
        data,
        min_history_days=cfg.data.min_history_days,
        max_missing_pct=cfg.data.max_missing_pct,
    )

    store = PriceStore(cfg.data.cache_dir)
    store.save(cleaned, meta={"provider": cfg.data.provider, "config": args.config})

    report_path = cfg.data.cache_dir / "quality_report.csv"
    report.to_frame().to_csv(report_path, index=False)
    logger.info("Quality report -> %s", report_path)
    logger.info("Done. %d tickers usable.", len(report.kept))


if __name__ == "__main__":
    main()
