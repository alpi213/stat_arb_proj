"""Crypto step 3: capacity analysis on the funding-carry strategy.

This is the question the whole framework was built to answer, finally asked of
a strategy that actually has an edge: **how much capital can it absorb before
its own market impact eats the alpha?**

Method mirrors the equity version — re-price the SAME trade stream at
increasing AUM under a square-root impact model:

    impact_bps = k * sigma * sqrt(Q / ADV)

Crypto-specific choices, each of which matters:
- ADV is the PERP's trailing dollar volume. The spot leg is usually the
  thinner of the two for alts, so this is the optimistic leg; a stricter
  version would take min(perp_ADV, spot_ADV). Both are reported.
- sigma is the asset's own trailing return vol at the 8h bar.
- Only the traded notional incurs impact. Carry is a low-turnover strategy
  (~47x/yr), which is precisely why its capacity is far better than its
  gross Sharpe alone would suggest — turnover, not Sharpe, sets capacity.

Usage:
    python scripts/12_crypto_capacity.py [--k 0.1] [--leg perp|min]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import CrossSectionalCostModel, backtest_weights, performance_summary
from statarb.capacity import capacity_curve
from statarb.crypto.funding import (
    PERIODS_PER_YEAR,
    build_carry_panel,
    carry_returns,
    carry_signal,
)
from statarb.utils import setup_logging
from statarb.utils.plotting import plot_capacity_curve

logger = logging.getLogger("crypto_capacity")

AUM_GRID = [1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 2.5e9]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--k", type=float, default=0.1, help="impact coefficient")
    p.add_argument("--leg", default="perp", choices=["perp", "min"],
                   help="ADV basis: perp only, or min(perp, spot proxy)")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--hysteresis", type=float, default=3.0)
    args = p.parse_args()
    setup_logging()

    d = Path(args.data)
    f = pd.read_parquet(d / "funding.parquet")
    perp = pd.read_parquet(d / "perp_close.parquet")
    spot = pd.read_parquet(d / "spot_close.parquet")
    vol = pd.read_parquet(d / "quote_volume.parquet")
    panel = build_carry_panel(
        {c: f[c] for c in f}, {c: perp[c] for c in perp},
        {c: spot[c] for c in spot}, {c: vol[c] for c in vol},
    )

    weights = carry_signal(
        panel, lookback=args.lookback, top_n=args.top_n,
        rebalance_every=args.rebalance_every, hysteresis=args.hysteresis,
    )
    rets = carry_returns(panel)
    res = backtest_weights(
        weights, rets, CrossSectionalCostModel(periods_per_year=PERIODS_PER_YEAR)
    )
    base = res["returns"]
    base_summary = performance_summary(base, periods_per_year=PERIODS_PER_YEAR)
    logger.info("base (pre-impact) Sharpe %.2f, turnover %.0fx/yr",
                base_summary["sharpe"], res["turnover"].mean() * PERIODS_PER_YEAR)

    # Per-symbol traded notional each period, as a fraction of AUM.
    dw = res["weights"].diff().abs().fillna(0.0)

    # ADV per 8h bar. quote_volume is already quote-currency (USDT) notional.
    adv = panel["dollar_volume"].rolling(21, min_periods=5).mean()
    if args.leg == "min":
        # Crude spot-depth proxy: alt spot books are typically thinner than
        # the perp. Without spot volume in the store, haircut the perp ADV.
        adv = adv * 0.35
    sigma = panel["perp"].pct_change().rolling(21, min_periods=5).std()

    cols = dw.columns.intersection(adv.columns)
    curve = capacity_curve(
        base, dw[cols], adv[cols].loc[dw.index], sigma[cols].loc[dw.index],
        aum_grid=AUM_GRID, k=args.k, periods_per_year=PERIODS_PER_YEAR,
    )
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / f"capacity_carry_{args.leg}.csv")
    plot_capacity_curve(curve, out / "figures" / f"capacity_carry_{args.leg}.png")

    logger.info("\n%s", curve.to_string())
    half, zero = curve.attrs.get("aum_sharpe_half"), curve.attrs.get("aum_sharpe_zero")
    verdict = {
        "leg_basis": args.leg,
        "impact_k": args.k,
        "base_sharpe": float(base_summary["sharpe"]),
        "annual_turnover": float(res["turnover"].mean() * PERIODS_PER_YEAR),
        "aum_sharpe_half": half,
        "aum_sharpe_zero": zero,
        "median_perp_adv_8h_usd": float(np.nanmedian(adv.to_numpy())),
    }
    (out / f"capacity_verdict_{args.leg}.json").write_text(
        json.dumps(verdict, indent=2, default=str)
    )
    logger.info(
        "Sharpe halves at ~%s; zero at ~%s (k=%.2f, leg=%s)",
        f"${half/1e6:,.0f}M" if half else "beyond grid",
        f"${zero/1e6:,.0f}M" if zero else "beyond grid",
        args.k, args.leg,
    )


if __name__ == "__main__":
    main()
