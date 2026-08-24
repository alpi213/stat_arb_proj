"""Crypto step 2: funding-carry backtest with the full honesty layer.

Runs an out-of-sample split, the same Deflated Sharpe and regime machinery
used for equities, and a cost-sensitivity sweep (fees are THE swing factor
at 8h turnover, so a single fee assumption would be dishonest).

Usage:
    python scripts/11_carry_backtest.py [--top-n 10] [--lookback 21]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from statarb.backtest import CrossSectionalCostModel, backtest_weights, performance_summary
from statarb.crypto.funding import PERIODS_PER_YEAR, build_carry_panel, carry_returns, carry_signal
from statarb.utils import setup_logging
from statarb.utils.plotting import plot_equity
from statarb.validation import deflated_sharpe_ratio

logger = logging.getLogger("carry")


def load_panel(data_dir: Path) -> dict[str, pd.DataFrame]:
    f = pd.read_parquet(data_dir / "funding.parquet")
    perp = pd.read_parquet(data_dir / "perp_close.parquet")
    spot = pd.read_parquet(data_dir / "spot_close.parquet")
    vol = pd.read_parquet(data_dir / "quote_volume.parquet")
    return build_carry_panel(
        {c: f[c] for c in f}, {c: perp[c] for c in perp},
        {c: spot[c] for c in spot}, {c: vol[c] for c in vol},
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--min-bps", type=float, default=1.0)
    p.add_argument("--max-bps", type=float, default=50.0,
                   help="risk control: skip squeeze-regime funding")
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--hysteresis", type=float, default=3.0)
    p.add_argument("--oos-frac", type=float, default=0.5,
                   help="fraction of history reserved for out-of-sample")
    args = p.parse_args()
    setup_logging()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    panel = load_panel(Path(args.data))
    logger.info("panel: %d symbols x %d settlements (%s .. %s)",
                panel["funding"].shape[1], panel["funding"].shape[0],
                panel["funding"].index.min().date(), panel["funding"].index.max().date())

    weights = carry_signal(
        panel, lookback=args.lookback, top_n=args.top_n,
        min_predicted_bps=args.min_bps, max_predicted_bps=args.max_bps,
        rebalance_every=args.rebalance_every, hysteresis=args.hysteresis,
    )
    asset_rets = carry_returns(panel)

    costs = CrossSectionalCostModel(periods_per_year=PERIODS_PER_YEAR)
    res = backtest_weights(weights, asset_rets, costs=costs)

    split = int(len(res["returns"]) * (1 - args.oos_frac))
    oos = res["returns"].iloc[split:]
    is_ = res["returns"].iloc[:split]

    summ_all = performance_summary(res["returns"], res["equity_curve"],
                                   gross_exposure=res["gross_exposure"],
                                   periods_per_year=PERIODS_PER_YEAR,
                                   turnover=res["turnover"])
    summ_is = performance_summary(is_, periods_per_year=PERIODS_PER_YEAR)
    summ_oos = performance_summary(oos, periods_per_year=PERIODS_PER_YEAR)
    table = pd.DataFrame({"full": summ_all, "in_sample": summ_is, "out_of_sample": summ_oos})
    table.to_csv(out / "carry_summary.csv")
    logger.info("\n%s", table.to_string())

    # Gross vs net: how much of the edge do fees eat?
    gross_sharpe = performance_summary(
        res["gross_returns"], periods_per_year=PERIODS_PER_YEAR)["sharpe"]
    logger.info("Sharpe gross of costs: %.2f  |  net: %.2f", gross_sharpe, summ_all["sharpe"])

    # Cost sensitivity — the honest headline is a RANGE, not a point.
    rows = []
    for mult, label in [(0.0, "zero-cost (upper bound)"), (0.5, "VIP/maker-ish"),
                        (1.0, "retail taker (base)"), (2.0, "stressed/wide")]:
        c = CrossSectionalCostModel(
            leg1_fee_bps=4.0 * mult, leg2_fee_bps=7.5 * mult,
            leg1_half_spread_bps=1.0 * mult, leg2_half_spread_bps=1.0 * mult,
            periods_per_year=PERIODS_PER_YEAR,
        )
        r = backtest_weights(weights, asset_rets, costs=c)
        s_all = performance_summary(r["returns"], periods_per_year=PERIODS_PER_YEAR)
        s_oos = performance_summary(r["returns"].iloc[split:], periods_per_year=PERIODS_PER_YEAR)
        rows.append({"scenario": label, "cost_mult": mult,
                     "sharpe_full": s_all["sharpe"], "sharpe_oos": s_oos["sharpe"],
                     "cagr_full": s_all["cagr"]})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "cost_sensitivity.csv", index=False)
    logger.info("\n=== COST SENSITIVITY ===\n%s", sens.to_string(index=False))

    # Deflation: count the configurations tried (grid below mirrors what a
    # researcher would realistically sweep). Undercounting this is cheating.
    # Honest trial count: hedge/lookback x top_n x min_bps x rebalance x cap.
    # This sweep really was run (see conversation log), so it is counted.
    n_trials = 4 * 4 * 3 * 6 * 5
    try:
        dsr = deflated_sharpe_ratio(oos, n_trials=n_trials)
        (out / "carry_dsr.json").write_text(json.dumps(dsr, indent=2, default=str))
        logger.info("\n=== DEFLATED SHARPE (OOS) ===\n%s",
                    json.dumps(dsr, indent=2, default=str))
    except ValueError as exc:
        logger.warning("DSR skipped: %s", exc)

    plot_equity(res["equity_curve"], out / "figures" / "carry_equity.png",
                title="Crypto funding carry — net of costs")
    res["returns"].to_csv(out / "carry_returns.csv")
    logger.info("artifacts -> %s", out)


if __name__ == "__main__":
    main()
