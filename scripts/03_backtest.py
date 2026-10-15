"""Backtest the selected pairs out of sample, with a static and a Kalman hedge.

Trading starts the bar after the formation window ends. Writes
results/summary.csv plus returns, trades and figures for each hedge.

    python scripts/03_backtest.py [--config configs/base.yaml]
"""

import argparse
import json
import logging

import pandas as pd

from statarb.backtest import performance_summary
from statarb.config import load_config
from statarb.data import PriceStore
from statarb.pipeline import first_oos_date, hedge_from_config, run_pairs
from statarb.utils import plot_equity, plot_pair_diagnostics, setup_logging

logger = logging.getLogger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    adj_close = PriceStore(cfg.data.cache_dir).load(["adj_close"])["adj_close"]
    pairs_df = pd.read_csv(cfg.output_dir / "pairs_selected.csv")
    pairs = list(zip(pairs_df["y"], pairs_df["x"], strict=True))
    meta = json.loads((cfg.output_dir / "selection_meta.json").read_text())
    trade_start = first_oos_date(adj_close.index, pd.Timestamp(meta["formation_end"]))
    logger.info("Formation ends %s; trading starts %s", meta["formation_end"], trade_start.date())

    fig_dir = cfg.output_dir / "figures"
    summaries = {}
    for method in ("static", "kalman"):
        port = run_pairs(cfg, adj_close, pairs, hedge_from_config(cfg, method), trade_start)
        oos = port["returns"].loc[trade_start:]
        equity = cfg.backtest.initial_capital * (1 + oos).cumprod()
        summaries[method] = performance_summary(oos, equity, port["trades"], port["gross_exposure"])

        plot_equity(equity, fig_dir / f"equity_{method}.png", title=f"OOS equity, {method} hedge")
        for res in port["pair_results"][:3]:
            plot_pair_diagnostics(res, fig_dir / f"pair_{res.pair.replace('/', '_')}_{method}.png")
        oos.to_csv(cfg.output_dir / f"returns_{method}.csv")
        pd.DataFrame([vars(t) for t in port["trades"]]).to_csv(
            cfg.output_dir / f"trades_{method}.csv", index=False
        )

    table = pd.DataFrame(summaries)
    table.to_csv(cfg.output_dir / "summary.csv")
    logger.info("\n%s", table.to_string())


if __name__ == "__main__":
    main()
