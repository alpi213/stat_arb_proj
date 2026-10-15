"""Null experiment: how much Sharpe does the search itself manufacture?

Runs the whole pipeline (selection, walk-forward, backtest) on independent
random walks. There is no cointegration and no edge in this data, so any
Sharpe it finds is selection bias. This is the control group: without it,
the backtesting Sharpe is a number with no bar to clear.

    python scripts/07_null_experiment.py [--n-universes 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from statarb.backtest import performance_summary
from statarb.config import load_config
from statarb.hedge import make_hedge
from statarb.pipeline import run_walkforward
from statarb.utils import setup_logging
from statarb.validation import expected_max_sharpe

logger = logging.getLogger("null")

CONFIGS = [
    ("static", {"refit_every_days": 63}),
    ("kalman", {"delta": 1e-5, "obs_var": 1e-3, "spread_mode": "innovation"}),
    ("kalman", {"delta": 1e-5, "obs_var": 1e-3, "spread_mode": "residual"}),
]


def synthetic_universe(n_names: int, n_days: int, n_sectors: int, seed: int):
    """Independent geometric random walks with a mild common factor.

    The factor makes them correlated, like real equities, but not
    cointegrated: the spread of two correlated random walks is still a
    random walk.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-universes", type=int, default=5)
    p.add_argument("--n-names", type=int, default=60)
    p.add_argument("--n-days", type=int, default=2600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    setup_logging()
    cfg = load_config("configs/base.yaml")

    rows, total_tested = [], 0
    for u in range(args.n_universes):
        px, sectors = synthetic_universe(args.n_names, args.n_days, 8, seed=args.seed + u)
        for method, kwargs in CONFIGS:
            try:
                wf = run_walkforward(cfg, px, sectors, make_hedge(method, **kwargs))
            except RuntimeError:  # no window selected any pairs
                continue
            total_tested += wf.n_tested
            sharpe = float(performance_summary(wf.returns)["sharpe"])
            label = f"{method}/{kwargs.get('spread_mode', '-')}"
            rows.append({"universe": u, "config": label, "sharpe": sharpe})
            logger.info("null universe %d  %-22s Sharpe %+.2f", u, label, sharpe)

    df = pd.DataFrame(rows)
    out = {
        "n_synthetic_universes": args.n_universes,
        "n_configs_per_universe": len(CONFIGS),
        "n_backtests_run": len(df),
        "n_pair_hypotheses_tested": total_tested,
        "null_sharpe_mean": float(df["sharpe"].mean()),
        "null_sharpe_std": float(df["sharpe"].std()),
        "null_sharpe_max": float(df["sharpe"].max()),
        "null_sharpe_min": float(df["sharpe"].min()),
        "theoretical_expected_max_sharpe_annual":
            float(expected_max_sharpe(max(total_tested, 1), 1.0 / args.n_days) * np.sqrt(252)),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "null_experiment.json").write_text(json.dumps(out, indent=2))
    logger.info("\nNull experiment (no edge exists in this data):\n%s", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
