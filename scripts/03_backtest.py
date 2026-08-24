"""Step 4-6: backtest the selected pairs out-of-sample.

Trading starts the day AFTER the formation window ends (positions before
that are zeroed inside backtest_pair via trade_start).

Compares static vs kalman hedging when --compare is passed — the
before/after that makes the Kalman upgrade a story instead of a claim.

Usage:
    python scripts/03_backtest.py [--config configs/base.yaml] [--compare]
"""

import argparse
import json
import logging

import pandas as pd

from statarb.backtest import CostModel, backtest_pair, backtest_portfolio, performance_summary
from statarb.config import Config, load_config
from statarb.data import PriceStore
from statarb.hedge import make_hedge
from statarb.utils import setup_logging
from statarb.utils.plotting import plot_equity, plot_pair_diagnostics

logger = logging.getLogger("backtest")


def run(cfg: Config, hedge_method: str, adj_close: pd.DataFrame, pairs: pd.DataFrame,
        trade_start: pd.Timestamp) -> dict:
    costs = CostModel(
        commission_bps=cfg.backtest.costs.commission_bps,
        half_spread_bps=cfg.backtest.costs.half_spread_bps,
        borrow_fee_annual_bps=cfg.backtest.costs.borrow_fee_annual_bps,
    )
    hedge_kwargs = (
        cfg.hedge.kalman.model_dump() if hedge_method == "kalman" else cfg.hedge.static.model_dump()
    )
    results = []
    for row in pairs.itertuples(index=False):
        hedge = make_hedge(hedge_method, **hedge_kwargs)
        res = backtest_pair(
            adj_close,
            y=row.y,
            x=row.x,
            hedge=hedge,
            zscore_window=cfg.signals.zscore_window,
            entry_z=cfg.signals.entry_z,
            exit_z=cfg.signals.exit_z,
            stop_z=cfg.signals.stop_z,
            max_holding_days=cfg.signals.max_holding_days,
            gross_per_pair=cfg.backtest.gross_per_pair,
            execution_lag_bars=cfg.backtest.execution_lag_bars,
            costs=costs,
            trade_start=trade_start,
        )
        results.append(res)

    port = backtest_portfolio(
        results,
        initial_capital=cfg.backtest.initial_capital,
        max_gross_leverage=cfg.backtest.risk.max_gross_leverage,
    )
    port["pair_results"] = results
    # Restrict reporting to the out-of-sample period.
    oos = port["returns"].loc[trade_start:]
    port["oos_returns"] = oos
    port["oos_equity"] = cfg.backtest.initial_capital * (1 + oos).cumprod()
    return port


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--compare", action="store_true", help="run static AND kalman")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    store = PriceStore(cfg.data.cache_dir)
    adj_close = store.load(["adj_close"])["adj_close"]
    pairs = pd.read_csv(cfg.output_dir / "pairs_selected.csv")
    meta = json.loads((cfg.output_dir / "selection_meta.json").read_text())
    formation_end = pd.Timestamp(meta["formation_end"])
    pos = adj_close.index.get_indexer([formation_end], method="nearest")[0]
    trade_start = adj_close.index[min(pos + 1, len(adj_close.index) - 1)]
    logger.info(
        "Formation ends %s; OOS trading starts %s", formation_end.date(), trade_start.date()
    )

    methods = ["static", "kalman"] if args.compare else [cfg.hedge.method]
    summaries = {}
    for method in methods:
        port = run(cfg, method, adj_close, pairs, trade_start)
        summary = performance_summary(
            port["oos_returns"], port["oos_equity"], port["trades"], port["gross_exposure"]
        )
        summaries[method] = summary

        fig_dir = cfg.output_dir / "figures"
        plot_equity(port["oos_equity"], fig_dir / f"equity_{method}.png",
                    title=f"OOS equity — {method} hedge")
        for res in port["pair_results"][:3]:
            safe = res.pair.replace("/", "_")
            plot_pair_diagnostics(res, fig_dir / f"pair_{safe}_{method}.png")

        port["oos_returns"].to_csv(cfg.output_dir / f"returns_{method}.csv")
        trades_df = pd.DataFrame([vars(t) for t in port["trades"]])
        trades_df.to_csv(cfg.output_dir / f"trades_{method}.csv", index=False)

    table = pd.DataFrame(summaries)
    table.to_csv(cfg.output_dir / "summary.csv")
    logger.info("\n%s", table.to_string())
    logger.info("Artifacts written to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
