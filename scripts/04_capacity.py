"""Capacity curve: net Sharpe against AUM under square-root impact.

Reruns the pair backtest from config (cheap, avoiding stale weight
files), then reprices its turnover at each AUM in the grid.

    python scripts/04_capacity.py [--config configs/base.yaml] [--method kalman] [--k 0.1]
"""

import argparse
import json
import logging

import pandas as pd

from statarb.capacity import capacity_curve, trailing_adv_dollars, trailing_sigma
from statarb.config import load_config
from statarb.data import PriceStore
from statarb.pipeline import first_oos_date, hedge_from_config, run_pairs
from statarb.utils import plot_capacity_curve, setup_logging

logger = logging.getLogger("capacity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method", default=None, help="hedge method (default: from config)")
    parser.add_argument("--k", type=float, default=None, help="impact coefficient override")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    method = args.method or cfg.hedge.method
    k = cfg.capacity.impact_coef if args.k is None else args.k

    data = PriceStore(cfg.data.cache_dir).load(["adj_close", "close", "volume"])
    pairs_df = pd.read_csv(cfg.output_dir / "pairs_selected.csv")
    pairs = list(zip(pairs_df["y"], pairs_df["x"], strict=True))
    meta = json.loads((cfg.output_dir / "selection_meta.json").read_text())
    trade_start = first_oos_date(data["adj_close"].index, pd.Timestamp(meta["formation_end"]))

    port = run_pairs(cfg, data["adj_close"], pairs, hedge_from_config(cfg, method), trade_start)
    base_returns = port["returns"].loc[trade_start:]

    # Traded notional per name per day, summed across pairs that share a name.
    dw: dict[str, pd.Series] = {}
    for res in port["pair_results"]:
        for name, col in res.weights.diff().abs().items():
            dw[name] = dw.get(name, 0) + col.fillna(0.0)
    dw_df = pd.DataFrame(dw).fillna(0.0).loc[trade_start:]

    adv = trailing_adv_dollars(data["close"], data["volume"], cfg.capacity.adv_window_days)
    sigma = trailing_sigma(data["adj_close"], cfg.capacity.adv_window_days)
    curve = capacity_curve(
        base_returns, dw_df,
        adv[dw_df.columns].loc[dw_df.index], sigma[dw_df.columns].loc[dw_df.index],
        aum_grid=cfg.capacity.aum_grid, k=k,
    )
    curve.to_csv(cfg.output_dir / f"capacity_{method}.csv")
    plot_capacity_curve(curve, cfg.output_dir / "figures" / f"capacity_{method}.png")

    logger.info("\n%s", curve.to_string())
    half, zero = curve.attrs["aum_sharpe_half"], curve.attrs["aum_sharpe_zero"]
    logger.info(
        "Sharpe halves at ~%s; reaches zero at ~%s (k=%.2f)",
        f"${half/1e6:,.0f}M" if half else "beyond grid",
        f"${zero/1e6:,.0f}M" if zero else "beyond grid", k,
    )


if __name__ == "__main__":
    main()
