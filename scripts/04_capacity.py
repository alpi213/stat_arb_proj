"""Step 7: capacity analysis — net Sharpe vs AUM.

Re-prices the backtest's turnover stream under a square-root impact model
at each AUM in the grid, and reports where Sharpe halves / dies.

Requires scripts 01-03 to have run (uses returns_<method>.csv and the
per-pair weights recomputed from the same config).

Usage:
    python scripts/04_capacity.py [--config configs/base.yaml] [--k 0.1]
"""

import argparse
import json
import logging

import pandas as pd

from statarb.capacity import capacity_curve
from statarb.capacity.impact import trailing_adv_dollars, trailing_sigma
from statarb.config import load_config
from statarb.data import PriceStore
from statarb.utils import setup_logging
from statarb.utils.plotting import plot_capacity_curve

logger = logging.getLogger("capacity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method", default=None, help="hedge method; defaults to config")
    parser.add_argument("--k", type=float, default=None, help="impact coefficient override")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    method = args.method or cfg.hedge.method
    k = args.k if args.k is not None else cfg.capacity.impact_coef

    store = PriceStore(cfg.data.cache_dir)
    data = store.load(["adj_close", "close", "volume"])

    base = pd.read_csv(cfg.output_dir / f"returns_{method}.csv", index_col=0, parse_dates=True)
    base_returns = base.iloc[:, 0]

    # Rebuild the daily |dW| per name from the backtest run.
    # We re-run the pair backtests deterministically from the same config
    # (cheap at this scale) rather than serializing weights — one source of
    # truth, no stale-file bugs.
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "bt03", Path(__file__).parent / "03_backtest.py"
    )
    bt03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt03)

    pairs = pd.read_csv(cfg.output_dir / "pairs_selected.csv")
    meta = json.loads((cfg.output_dir / "selection_meta.json").read_text())
    formation_end = pd.Timestamp(meta["formation_end"])
    idx = data["adj_close"].index
    pos = idx.get_indexer([formation_end], method="nearest")[0]
    trade_start = idx[min(pos + 1, len(idx) - 1)]
    port = bt03.run(cfg, method, data["adj_close"], pairs, trade_start)

    # Aggregate per-name weight changes across pairs (names can repeat).
    weight_changes: dict[str, pd.Series] = {}
    for res in port["pair_results"]:
        dw = res.weights.diff().abs()
        for name in dw.columns:
            weight_changes[name] = weight_changes.get(name, 0) + dw[name].fillna(0.0)
    dw_df = pd.DataFrame(weight_changes).fillna(0.0).loc[trade_start:]

    adv = trailing_adv_dollars(data["close"], data["volume"], cfg.capacity.adv_window_days)
    sigma = trailing_sigma(data["adj_close"], cfg.capacity.adv_window_days)

    curve = capacity_curve(
        base_returns.loc[trade_start:],
        dw_df,
        adv[dw_df.columns].loc[dw_df.index],
        sigma[dw_df.columns].loc[dw_df.index],
        aum_grid=cfg.capacity.aum_grid,
        k=k,
    )
    curve.to_csv(cfg.output_dir / f"capacity_{method}.csv")
    plot_capacity_curve(curve, cfg.output_dir / "figures" / f"capacity_{method}.png")

    logger.info("\n%s", curve.to_string())
    half, zero = curve.attrs.get("aum_sharpe_half"), curve.attrs.get("aum_sharpe_zero")
    logger.info(
        "Sharpe halves at ~ %s; reaches zero at ~ %s (k=%.2f)",
        f"${half/1e6:,.0f}M" if half else "beyond grid",
        f"${zero/1e6:,.0f}M" if zero else "beyond grid",
        k,
    )


if __name__ == "__main__":
    main()
