"""Step 10: the null experiment — how much Sharpe does the SEARCH invent?

Runs the complete pipeline (cointegration screening, pair selection,
walk-forward, backtest) on synthetic price series that are independent
random walks by construction. There is NO cointegration and NO edge in this
data. Any positive Sharpe found is pure selection bias.

This is the control group. Without it, "my backtest shows Sharpe 2" is an
uncalibrated statement: you cannot tell skill from search luck unless you
know what the search produces on noise.

Usage:
    python scripts/08_null_experiment.py [--n-universes 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from statarb.backtest import CostModel, backtest_pair, backtest_portfolio, performance_summary
from statarb.config import load_config
from statarb.hedge import make_hedge
from statarb.pairs import select_pairs
from statarb.utils import setup_logging
from statarb.validation import expected_max_sharpe, make_windows

logger = logging.getLogger("null")


def synthetic_universe(n_names: int, n_days: int, n_sectors: int, seed: int):
    """Independent geometric random walks — zero cointegration by construction.

    A mild common market factor is included because real equities share one;
    it creates correlation but NOT cointegration (the spread of two
    correlated random walks is still a random walk).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    market = rng.normal(0.0002, 0.008, n_days)
    cols, sectors = {}, {}
    for i in range(n_names):
        idio = rng.normal(0.0, 0.012, n_days)
        beta = rng.uniform(0.7, 1.3)
        cols[f"N{i:03d}"] = np.exp(np.cumsum(beta * market + idio) + np.log(50))
        sectors.setdefault(f"S{i % n_sectors}", []).append(f"N{i:03d}")
    return pd.DataFrame(cols, index=dates), sectors


def walkforward_sharpe(cfg, adj_close, sectors, method, hkw) -> tuple[float, int]:
    wf = cfg.validation.walkforward
    windows = make_windows(adj_close.index, wf.formation_days, wf.trading_days, wf.step_days)
    costs = CostModel(
        commission_bps=cfg.backtest.costs.commission_bps,
        half_spread_bps=cfg.backtest.costs.half_spread_bps,
        borrow_fee_annual_bps=cfg.backtest.costs.borrow_fee_annual_bps,
    )
    segs, tested = [], 0
    for w in windows:
        sel = select_pairs(
            adj_close, sectors, w.formation_start, w.formation_end,
            adf_pvalue_max=cfg.pairs.adf_pvalue_max,
            min_half_life_days=cfg.pairs.min_half_life_days,
            max_half_life_days=cfg.pairs.max_half_life_days,
            top_n=cfg.pairs.top_n_pairs,
        )
        tested += sel.n_tested
        if not sel.pairs:
            continue
        res = [
            backtest_pair(
                adj_close.loc[w.formation_start: w.trading_end],
                y=c.result.y, x=c.result.x, hedge=make_hedge(method, **hkw),
                zscore_window=cfg.signals.zscore_window, entry_z=cfg.signals.entry_z,
                exit_z=cfg.signals.exit_z, stop_z=cfg.signals.stop_z,
                max_holding_days=cfg.signals.max_holding_days,
                gross_per_pair=cfg.backtest.gross_per_pair,
                execution_lag_bars=cfg.backtest.execution_lag_bars, costs=costs,
                trade_start=w.trading_start, forced_exit_date=w.trading_end,
            )
            for c in sel.pairs
        ]
        port = backtest_portfolio(res, cfg.backtest.initial_capital,
                                  cfg.backtest.risk.max_gross_leverage)
        segs.append(port["returns"].loc[w.trading_start: w.trading_end])
    if not segs:
        return float("nan"), tested
    stitched = pd.concat(segs, axis=1).mean(axis=1).sort_index()
    return float(performance_summary(stitched)["sharpe"]), tested


CONFIGS = [
    ("static", {"refit_every_days": 63}),
    ("kalman", {"delta": 1e-5, "obs_var": 1e-3, "spread_mode": "innovation"}),
    ("kalman", {"delta": 1e-5, "obs_var": 1e-3, "spread_mode": "residual"}),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-universes", type=int, default=5)
    p.add_argument("--n-names", type=int, default=60)
    p.add_argument("--n-days", type=int, default=2600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    setup_logging()

    cfg = load_config("configs/base.yaml")
    sharpes, total_pairs = [], 0
    for u in range(args.n_universes):
        px, sectors = synthetic_universe(args.n_names, args.n_days, 8, seed=args.seed + u)
        for method, hkw in CONFIGS:
            s, tested = walkforward_sharpe(cfg, px, sectors, method, hkw)
            total_pairs += tested
            if not np.isnan(s):
                sharpes.append({"universe": u, "config": f"{method}/{hkw.get('spread_mode','-')}",
                                "sharpe": s})
                logger.info("null universe %d  %-22s Sharpe %+.2f", u, method, s)

    df = pd.DataFrame(sharpes)
    n_trials = max(total_pairs, 1)
    out = {
        "n_synthetic_universes": args.n_universes,
        "n_configs_per_universe": len(CONFIGS),
        "n_backtests_run": len(df),
        "n_pair_hypotheses_tested": total_pairs,
        "null_sharpe_mean": float(df["sharpe"].mean()),
        "null_sharpe_std": float(df["sharpe"].std()),
        "null_sharpe_max": float(df["sharpe"].max()),
        "null_sharpe_min": float(df["sharpe"].min()),
        "theoretical_expected_max_sharpe_annual":
            float(expected_max_sharpe(n_trials, 1.0 / args.n_days) * np.sqrt(252)),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "null_experiment.json").write_text(json.dumps(out, indent=2))
    logger.info("\n=== NULL EXPERIMENT (no edge exists in this data) ===\n%s",
                json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
