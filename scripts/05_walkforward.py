"""Walk-forward: reselect pairs on each formation window, then trade the next one.

The stitched out-of-sample series is the headline equity result.

    python scripts/05_walkforward.py [--config configs/base.yaml]
"""

import argparse
import logging

from statarb.backtest import performance_summary
from statarb.config import load_config
from statarb.data import PriceStore, load_universe
from statarb.pipeline import hedge_from_config, run_walkforward
from statarb.utils import plot_equity, setup_logging

logger = logging.getLogger("walkforward")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    adj_close = PriceStore(cfg.data.cache_dir).load(["adj_close"])["adj_close"]
    sectors = load_universe(cfg.data.universe_file)
    wf = run_walkforward(cfg, adj_close, sectors, hedge_from_config(cfg))

    equity = cfg.backtest.initial_capital * (1 + wf.returns).cumprod()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    wf.returns.to_csv(cfg.output_dir / "walkforward_returns.csv")
    wf.windows.to_csv(cfg.output_dir / "walkforward_windows.csv", index=False)
    plot_equity(equity, cfg.output_dir / "figures" / "equity_walkforward.png",
                title="Walk-forward OOS equity")

    summary = performance_summary(wf.returns, equity)
    summary["total_pairs_tested"] = wf.n_tested
    summary.to_csv(cfg.output_dir / "walkforward_summary.csv")
    logger.info("\n%s", summary.to_string())


if __name__ == "__main__":
    main()
