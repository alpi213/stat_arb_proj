"""The steps the equity scripts share, so they are written once.

Scripts parse arguments and write files. Everything that runs a backtest
from a Config lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from statarb.backtest import CostModel, backtest_pair, backtest_portfolio, performance_summary
from statarb.config import Config
from statarb.hedge import Hedge, make_hedge
from statarb.pairs import select_pairs
from statarb.validation import make_windows

logger = logging.getLogger(__name__)


def cost_model(cfg: Config) -> CostModel:
    c = cfg.backtest.costs
    return CostModel(c.commission_bps, c.half_spread_bps, c.borrow_fee_annual_bps)


def hedge_from_config(cfg: Config, method: str | None = None) -> Hedge:
    method = method or cfg.hedge.method
    params = cfg.hedge.kalman if method == "kalman" else cfg.hedge.static
    return make_hedge(method, **params.model_dump())


def first_oos_date(dates: pd.DatetimeIndex, formation_end: pd.Timestamp) -> pd.Timestamp:
    """The bar after the formation window ends."""
    pos = dates.get_indexer([formation_end], method="nearest")[0]
    return dates[min(pos + 1, len(dates) - 1)]


def run_pairs(
    cfg: Config,
    adj_close: pd.DataFrame,
    pairs: list[tuple[str, str]],
    hedge: Hedge,
    trade_start: pd.Timestamp,
    forced_exit_date: pd.Timestamp | None = None,
) -> dict:
    """Backtest every (y, x) pair with the config's settings and aggregate."""
    s, b = cfg.signals, cfg.backtest
    results = [
        backtest_pair(
            adj_close, y=y, x=x, hedge=hedge,
            zscore_window=s.zscore_window, entry_z=s.entry_z, exit_z=s.exit_z,
            stop_z=s.stop_z, max_holding_days=s.max_holding_days,
            gross_per_pair=b.gross_per_pair, execution_lag_bars=b.execution_lag_bars,
            costs=cost_model(cfg), trade_start=trade_start, forced_exit_date=forced_exit_date,
        )
        for y, x in pairs
    ]
    return backtest_portfolio(results, b.initial_capital, b.max_gross_leverage)


@dataclass
class WalkForwardResult:
    returns: pd.Series        # stitched out-of-sample returns
    windows: pd.DataFrame     # one row per window
    n_tested: int             # pair hypotheses tested across all windows


def run_walkforward(
    cfg: Config, adj_close: pd.DataFrame, sectors: dict[str, list[str]], hedge: Hedge
) -> WalkForwardResult:
    """Re-select pairs on each formation window, trade them on the next one.

    No pair in the stitched series was chosen with knowledge of the period
    it traded in. This is the headline result of the equity part.
    """
    wf = cfg.validation.walkforward
    windows = make_windows(adj_close.index, wf.formation_days, wf.trading_days, wf.step_days)
    logger.info("Walk-forward: %d windows", len(windows))

    segments: list[pd.Series] = []
    rows = []
    n_tested = 0
    for i, w in enumerate(windows):
        sel = select_pairs(
            adj_close, sectors, w.formation_start, w.formation_end,
            adf_pvalue_max=cfg.pairs.adf_pvalue_max,
            min_half_life_days=cfg.pairs.min_half_life_days,
            max_half_life_days=cfg.pairs.max_half_life_days,
            top_n=cfg.pairs.top_n_pairs,
        )
        n_tested += sel.n_tested
        if not sel.pairs:
            logger.warning("Window %d: no pairs selected", i)
            continue

        # The hedge may train on formation data; positions are only allowed
        # inside the trading segment and are flattened at its end.
        port = run_pairs(
            cfg, adj_close.loc[w.formation_start: w.trading_end],
            [(c.result.y, c.result.x) for c in sel.pairs], hedge,
            trade_start=w.trading_start, forced_exit_date=w.trading_end,
        )
        seg = port["returns"].loc[w.trading_start: w.trading_end]
        segments.append(seg)
        sharpe = performance_summary(seg)["sharpe"]
        rows.append({
            "window": i,
            "trading_start": w.trading_start.date(),
            "trading_end": w.trading_end.date(),
            "n_pairs": len(sel.pairs),
            "n_tested": sel.n_tested,
            "sharpe": sharpe,
        })
        logger.info(
            "Window %d [%s..%s]: %d pairs, OOS Sharpe %.2f",
            i, w.trading_start.date(), w.trading_end.date(), len(sel.pairs),
            sharpe if pd.notna(sharpe) else float("nan"),
        )

    if not segments:
        raise RuntimeError("no out-of-sample segments produced; loosen the selection thresholds?")

    # Windows overlap when step_days < trading_days; average the overlap.
    stitched = pd.concat(segments, axis=1).mean(axis=1).sort_index()
    return WalkForwardResult(stitched, pd.DataFrame(rows), n_tested)
