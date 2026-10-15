"""Deflated Sharpe and regime breakdown on the walk-forward returns.

    python scripts/06_validate.py [--config configs/base.yaml] [--returns path.csv]
"""

import argparse
import json
import logging

import pandas as pd

from statarb.config import load_config
from statarb.utils import setup_logging
from statarb.validation import deflated_sharpe_ratio, regime_breakdown

logger = logging.getLogger("validate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--returns", default=None,
                        help="returns csv (default: results/walkforward_returns.csv)")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    path = args.returns or (cfg.output_dir / "walkforward_returns.csv")
    returns = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]

    # Trials = pairs tested across all windows x configs tried. The pair count
    # comes from the walk-forward run; the config count is yours to keep
    # honest in validation.n_trials.
    windows_path = cfg.output_dir / "walkforward_windows.csv"
    pairs_tested = int(pd.read_csv(windows_path)["n_tested"].sum()) if windows_path.exists() else 1
    n_trials = max(pairs_tested, 1) * max(cfg.validation.n_trials, 1)

    dsr = deflated_sharpe_ratio(returns, n_trials=n_trials)
    (cfg.output_dir / "deflated_sharpe.json").write_text(json.dumps(dsr, indent=2, default=str))
    logger.info("Deflated Sharpe:\n%s", json.dumps(dsr, indent=2, default=str))

    regimes = regime_breakdown(returns, cfg.validation.regimes)
    regimes.to_csv(cfg.output_dir / "regime_breakdown.csv")
    logger.info("Regimes:\n%s", regimes.loc[
        ["sharpe", "max_drawdown", "annual_vol", "worst_period"]
    ].to_string())

    verdict = "passes" if dsr["passes_95pct"] else "does not pass"
    logger.info("DSR = %.3f: %s the 95%% test (n_trials=%d)", dsr["dsr"], verdict, n_trials)


if __name__ == "__main__":
    main()
