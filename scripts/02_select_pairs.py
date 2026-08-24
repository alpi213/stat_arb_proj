"""Step 3: select cointegrated pairs on the formation window ONLY.

Writes results/pairs_selected.csv (with the n_tested count for DSR).

Usage:
    python scripts/02_select_pairs.py [--config configs/base.yaml]
"""

import argparse
import json
import logging

import pandas as pd

from statarb.config import load_config
from statarb.data import PriceStore, load_universe
from statarb.pairs import select_pairs
from statarb.utils import setup_logging

logger = logging.getLogger("select_pairs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    store = PriceStore(cfg.data.cache_dir)
    adj_close = store.load(["adj_close"])["adj_close"]
    universe = load_universe(cfg.data.universe_file)

    # Formation window = first `formation_window_days` trading days.
    # (The walk-forward script re-selects on rolling windows; this script is
    # the simple single-split version for the first working backtest.)
    dates = adj_close.index
    formation_end = dates[min(cfg.pairs.formation_window_days, len(dates)) - 1]

    out = select_pairs(
        adj_close,
        universe.sectors(),
        formation_start=dates[0],
        formation_end=formation_end,
        adf_pvalue_max=cfg.pairs.adf_pvalue_max,
        min_half_life_days=cfg.pairs.min_half_life_days,
        max_half_life_days=cfg.pairs.max_half_life_days,
        top_n=cfg.pairs.top_n_pairs,
    )

    rows = [
        {
            "y": c.result.y,
            "x": c.result.x,
            "sector": c.sector,
            "beta": c.result.beta,
            "pvalue": c.result.pvalue,
            "half_life_days": c.result.half_life_days,
            "rank_score": c.rank_score,
        }
        for c in out.pairs
    ]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg.output_dir / "pairs_selected.csv", index=False)
    (cfg.output_dir / "selection_meta.json").write_text(
        json.dumps(
            {
                "n_tested": out.n_tested,
                "formation_start": str(out.formation_start.date()),
                "formation_end": str(out.formation_end.date()),
                "config": args.config,
            },
            indent=2,
        )
    )
    logger.info(
        "Selected %d pairs (of %d tested) -> %s",
        len(rows), out.n_tested, cfg.output_dir / "pairs_selected.csv",
    )


if __name__ == "__main__":
    main()
