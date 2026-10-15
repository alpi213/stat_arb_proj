"""Funding-carry backtest: in/out-of-sample split, cost sweep, Deflated Sharpe.

    python scripts/12_carry_backtest.py [--top-n 10] [--lookback 21]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from statarb.backtest import CrossSectionalCostModel, backtest_weights, performance_summary
from statarb.crypto import PERIODS_PER_YEAR, carry_returns, carry_signal, load_panel
from statarb.utils import plot_equity, setup_logging
from statarb.validation import deflated_sharpe_ratio

logger = logging.getLogger("carry")

# Configurations actually tried while developing this strategy, charged to
# the DSR: 4 lookbacks x 4 top_n x 3 min_bps x 6 rebalance cadences x 5 caps.
N_TRIALS = 4 * 4 * 3 * 6 * 5

COST_SCENARIOS = [
    (0.0, "zero-cost (upper bound)"),
    (0.5, "VIP/maker-ish"),
    (1.0, "retail taker (base)"),
    (2.0, "stressed/wide"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--min-bps", type=float, default=1.0)
    p.add_argument("--max-bps", type=float, default=50.0, help="skip squeeze-regime funding")
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--hysteresis", type=float, default=3.0)
    p.add_argument("--oos-frac", type=float, default=0.5, help="share of history held out")
    args = p.parse_args()
    setup_logging()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.data)
    f = panel["funding"]
    logger.info("panel: %d symbols x %d settlements (%s .. %s)",
                f.shape[1], f.shape[0], f.index.min().date(), f.index.max().date())

    weights = carry_signal(
        panel, lookback=args.lookback, top_n=args.top_n,
        min_predicted_bps=args.min_bps, max_predicted_bps=args.max_bps,
        rebalance_every=args.rebalance_every, hysteresis=args.hysteresis,
    )
    rets = carry_returns(panel)
    res = backtest_weights(weights, rets)

    split = int(len(res["returns"]) * (1 - args.oos_frac))
    is_, oos = res["returns"].iloc[:split], res["returns"].iloc[split:]
    ppy = PERIODS_PER_YEAR
    table = pd.DataFrame({
        "full": performance_summary(res["returns"], res["equity_curve"],
                                    gross_exposure=res["gross_exposure"],
                                    periods_per_year=ppy, turnover=res["turnover"]),
        "in_sample": performance_summary(is_, periods_per_year=ppy),
        "out_of_sample": performance_summary(oos, periods_per_year=ppy),
    })
    table.to_csv(out / "carry_summary.csv")
    logger.info("\n%s", table.to_string())
    gross_sharpe = performance_summary(res["gross_returns"], periods_per_year=ppy)["sharpe"]
    logger.info("Sharpe gross of costs %.2f, net %.2f", gross_sharpe, table.loc["sharpe", "full"])

    # Fees are the swing factor at this turnover, so the headline is a range.
    rows = []
    for mult, label in COST_SCENARIOS:
        costs = CrossSectionalCostModel(4.0 * mult, 7.5 * mult, 1.0 * mult, 1.0 * mult)
        r = backtest_weights(weights, rets, costs=costs)["returns"]
        s_all = performance_summary(r, periods_per_year=ppy)
        s_oos = performance_summary(r.iloc[split:], periods_per_year=ppy)
        rows.append({"scenario": label, "cost_mult": mult, "sharpe_full": s_all["sharpe"],
                     "sharpe_oos": s_oos["sharpe"], "cagr_full": s_all["cagr"]})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "cost_sensitivity.csv", index=False)
    logger.info("\nCost sensitivity:\n%s", sens.to_string(index=False))

    try:
        dsr = deflated_sharpe_ratio(oos, n_trials=N_TRIALS, periods_per_year=ppy)
        (out / "carry_dsr.json").write_text(json.dumps(dsr, indent=2, default=str))
        logger.info("\nDeflated Sharpe (OOS):\n%s", json.dumps(dsr, indent=2, default=str))
    except ValueError as exc:
        logger.warning("DSR skipped: %s", exc)

    plot_equity(res["equity_curve"], out / "figures" / "carry_equity.png",
                title="Crypto funding carry, net of costs")
    res["returns"].to_csv(out / "carry_returns.csv")


if __name__ == "__main__":
    main()
