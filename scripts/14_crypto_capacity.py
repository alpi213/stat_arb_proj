"""Capacity curve for the carry strategy.

Same square-root impact model as the equity path, applied to the same
trade stream at increasing AUM. ADV is the perp's trailing dollar volume;
--leg min haircuts it to stand in for the thinner spot book. --exit-penalty
charges that fraction of the position on each exit from a dying contract;
--sample oos keeps only the second half of the bars.

    python scripts/14_crypto_capacity.py [--k 0.1] [--leg perp|min]
        [--exit-penalty 0.1] [--sample full|oos]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from statarb.backtest import backtest_weights, exit_size, performance_summary
from statarb.capacity import capacity_curve
from statarb.crypto import PERIODS_PER_YEAR, carry_returns, carry_signal, load_panel
from statarb.utils import plot_capacity_curve, setup_logging

logger = logging.getLogger("crypto_capacity")

AUM_GRID = [1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 2.5e9]
SPOT_HAIRCUT = 0.35  # alt spot books are typically this fraction of perp depth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--k", type=float, default=0.1, help="impact coefficient")
    p.add_argument("--leg", default="perp", choices=["perp", "min"],
                   help="ADV basis: perp volume, or a haircut spot proxy")
    p.add_argument("--exit-penalty", type=float, default=0.0,
                   help="fraction of the position lost on each exit from a dying contract")
    p.add_argument("--sample", default="full", choices=["full", "oos"],
                   help="all bars, or the second half (same split as 12_carry_backtest.py)")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--hysteresis", type=float, default=3.0)
    args = p.parse_args()
    setup_logging()

    panel = load_panel(args.data)
    weights = carry_signal(panel, lookback=args.lookback, top_n=args.top_n,
                           rebalance_every=args.rebalance_every, hysteresis=args.hysteresis)
    res = backtest_weights(weights, carry_returns(panel))
    net = res["returns"] - args.exit_penalty * exit_size(weights, res)
    start = len(net) // 2 if args.sample == "oos" else 0
    net = net.iloc[start:]
    ppy = PERIODS_PER_YEAR
    base_sharpe = performance_summary(net, periods_per_year=ppy)["sharpe"]
    annual_turnover = res["turnover"].iloc[start:].mean() * ppy
    logger.info("base Sharpe %.2f, turnover %.0fx/yr", base_sharpe, annual_turnover)

    dw = res["weights"].diff().abs().fillna(0.0).iloc[start:]
    adv = panel["dollar_volume"].rolling(21, min_periods=5).mean()
    if args.leg == "min":
        adv = adv * SPOT_HAIRCUT
    sigma = panel["perp"].pct_change().rolling(21, min_periods=5).std()

    cols = dw.columns.intersection(adv.columns)
    curve = capacity_curve(
        net, dw[cols], adv[cols].loc[dw.index], sigma[cols].loc[dw.index],
        aum_grid=AUM_GRID, k=args.k, periods_per_year=ppy,
    )
    out = Path(args.out)
    tag = args.leg
    if args.exit_penalty:
        tag += f"_exit{args.exit_penalty * 100:g}"
    if args.sample == "oos":
        tag += "_oos"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / f"capacity_carry_{tag}.csv")
    plot_capacity_curve(curve, out / "figures" / f"capacity_carry_{tag}.png")
    logger.info("\n%s", curve.to_string())

    half, zero = curve.attrs["aum_sharpe_half"], curve.attrs["aum_sharpe_zero"]
    (out / f"capacity_verdict_{tag}.json").write_text(json.dumps({
        "leg_basis": args.leg,
        "impact_k": args.k,
        "exit_penalty": args.exit_penalty,
        "sample": args.sample,
        "base_sharpe": float(base_sharpe),
        "annual_turnover": float(annual_turnover),
        "aum_sharpe_half": half,
        "aum_sharpe_zero": zero,
        "median_perp_adv_8h_usd": float(np.nanmedian(adv.to_numpy())),
    }, indent=2, default=str))
    logger.info("Sharpe halves at ~%s; zero at ~%s (k=%.2f, leg=%s)",
                f"${half/1e6:,.0f}M" if half else "beyond grid",
                f"${zero/1e6:,.0f}M" if zero else "beyond grid", args.k, args.leg)


if __name__ == "__main__":
    main()
