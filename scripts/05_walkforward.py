"""Step 8a: walk-forward analysis.

For each rolling window: re-select pairs on the formation segment, trade
them on the following out-of-sample segment, force-flat at segment end.
The stitched OOS return stream is the headline result of the project —
no pair in it was ever selected with knowledge of its trading period.

Usage:
    python scripts/05_walkforward.py [--config configs/base.yaml]
"""

import argparse
import logging

import pandas as pd

from statarb.backtest import CostModel, backtest_pair, backtest_portfolio, performance_summary
from statarb.config import load_config
from statarb.data import PriceStore, load_universe
from statarb.hedge import make_hedge
from statarb.pairs import select_pairs
from statarb.utils import setup_logging
from statarb.utils.plotting import plot_equity
from statarb.validation import make_windows

logger = logging.getLogger("walkforward")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    store = PriceStore(cfg.data.cache_dir)
    adj_close = store.load(["adj_close"])["adj_close"]
    universe = load_universe(cfg.data.universe_file)
    sectors = universe.sectors()

    wf = cfg.validation.walkforward
    windows = make_windows(
        adj_close.index,
        formation_days=wf.formation_days,
        trading_days=wf.trading_days,
        step_days=wf.step_days,
    )
    logger.info("Walk-forward: %d windows", len(windows))

    costs = CostModel(
        commission_bps=cfg.backtest.costs.commission_bps,
        half_spread_bps=cfg.backtest.costs.half_spread_bps,
        borrow_fee_annual_bps=cfg.backtest.costs.borrow_fee_annual_bps,
    )
    hedge_kwargs = (
        cfg.hedge.kalman.model_dump()
        if cfg.hedge.method == "kalman"
        else cfg.hedge.static.model_dump()
    )

    oos_segments: list[pd.Series] = []
    total_tested = 0
    window_rows = []

    for i, w in enumerate(windows):
        sel = select_pairs(
            adj_close,
            sectors,
            formation_start=w.formation_start,
            formation_end=w.formation_end,
            adf_pvalue_max=cfg.pairs.adf_pvalue_max,
            min_half_life_days=cfg.pairs.min_half_life_days,
            max_half_life_days=cfg.pairs.max_half_life_days,
            top_n=cfg.pairs.top_n_pairs,
        )
        total_tested += sel.n_tested
        if not sel.pairs:
            logger.warning("Window %d: no pairs selected", i)
            continue

        results = []
        for c in sel.pairs:
            hedge = make_hedge(cfg.hedge.method, **hedge_kwargs)
            # Estimation may see formation data (that's its training set);
            # positions are only allowed inside the trading segment.
            res = backtest_pair(
                adj_close.loc[w.formation_start: w.trading_end],
                y=c.result.y,
                x=c.result.x,
                hedge=hedge,
                zscore_window=cfg.signals.zscore_window,
                entry_z=cfg.signals.entry_z,
                exit_z=cfg.signals.exit_z,
                stop_z=cfg.signals.stop_z,
                max_holding_days=cfg.signals.max_holding_days,
                gross_per_pair=cfg.backtest.gross_per_pair,
                execution_lag_bars=cfg.backtest.execution_lag_bars,
                costs=costs,
                trade_start=w.trading_start,
                forced_exit_date=w.trading_end,
            )
            results.append(res)

        port = backtest_portfolio(
            results,
            initial_capital=cfg.backtest.initial_capital,
            max_gross_leverage=cfg.backtest.risk.max_gross_leverage,
        )
        seg = port["returns"].loc[w.trading_start: w.trading_end]
        oos_segments.append(seg)
        seg_sharpe = performance_summary(seg)["sharpe"]
        window_rows.append(
            {
                "window": i,
                "trading_start": w.trading_start.date(),
                "trading_end": w.trading_end.date(),
                "n_pairs": len(sel.pairs),
                "n_tested": sel.n_tested,
                "sharpe": seg_sharpe,
            }
        )
        logger.info(
            "Window %d [%s..%s]: %d pairs, OOS Sharpe %.2f",
            i, w.trading_start.date(), w.trading_end.date(), len(sel.pairs),
            seg_sharpe if pd.notna(seg_sharpe) else float("nan"),
        )

    if not oos_segments:
        raise SystemExit("No OOS segments produced — loosen selection thresholds?")

    # Windows may overlap if step_days < trading_days; average overlapping days.
    stitched = pd.concat(oos_segments, axis=1).mean(axis=1).sort_index()
    equity = cfg.backtest.initial_capital * (1 + stitched).cumprod()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stitched.to_csv(cfg.output_dir / "walkforward_returns.csv")
    pd.DataFrame(window_rows).to_csv(cfg.output_dir / "walkforward_windows.csv", index=False)
    plot_equity(equity, cfg.output_dir / "figures" / "equity_walkforward.png",
                title="Walk-forward OOS equity")

    summary = performance_summary(stitched, equity)
    summary["total_pairs_tested"] = total_tested  # feed this into the DSR
    summary.to_csv(cfg.output_dir / "walkforward_summary.csv")
    logger.info("\n%s", summary.to_string())


if __name__ == "__main__":
    main()
