"""Turnover sweep for the funding-carry book: how sticky does it need to be?

    python scripts/15_carry_turnover_sweep.py [--workers 16]

Grid over rebalance cadence x hysteresis x top_n x lookback. For each book:
annual turnover, net Sharpe at the cost scenarios of 12_carry_backtest.py
(full sample and the held-out second half), the break-even cost per unit
of turnover, and the Deflated Sharpe on the held-out half.

Costs are linear in turnover, so one backtest per book prices every
scenario: net = gross - mult * 13.5bp * turnover.

A second pass reruns two books at every rebalance phase (which settlement
in the cycle trades). A cadence whose Sharpe swings with phase is reporting
calendar luck, not a better turnover trade-off.

The DSR is computed twice. Lenient: the 1440 development configs charged by
12_carry_backtest.py, with the null variance 1/T. Strict: those plus every
book here, with the Sharpe variance measured across this grid.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import CrossSectionalCostModel, backtest_weights
from statarb.backtest.metrics import cagr, max_drawdown, sharpe_ratio
from statarb.crypto import PERIODS_PER_YEAR, carry_returns, carry_signal, load_panel
from statarb.utils import setup_logging
from statarb.validation import deflated_sharpe_ratio

logger = logging.getLogger("sweep")

PPY = PERIODS_PER_YEAR
DEV_TRIALS = 4 * 4 * 3 * 6 * 5  # same charge as 12_carry_backtest.py
COST_MULTS = [0.0, 0.5, 1.0, 2.0]  # same scenarios as 12_carry_backtest.py
COST_PER_UNIT = float(CrossSectionalCostModel().trading_cost(pd.Series([1.0])).iloc[0])
GRID = {
    "rebalance_every": [1, 3, 6, 9, 15, 21, 42, 63],  # 8h settlements: 8h .. 21d
    "hysteresis": [1.0, 1.5, 2.0, 3.0, 5.0],
    "top_n": [5, 10, 15, 20],
    "lookback": [9, 21, 42],
}
PHASE_BOOKS = [  # the default book, and the grid's best region
    {"top_n": 10, "lookback": 21, "hysteresis": 3.0},
    {"top_n": 20, "lookback": 42, "hysteresis": 3.0},
]

_panel = _rets = None


def _init(data_dir: str) -> None:
    global _panel, _rets
    _panel = load_panel(data_dir)
    _rets = carry_returns(_panel)


def run_book(params: dict) -> dict:
    res = backtest_weights(carry_signal(_panel, **params), _rets)
    gross, turn = res["gross_returns"], res["turnover"]
    split = len(gross) // 2  # same 50/50 split as 12_carry_backtest.py
    net = gross - COST_PER_UNIT * turn

    out = dict(params)
    out["annual_turnover"] = float(turn.mean() * PPY)
    for seg, sl in [("full", slice(None)), ("oos", slice(split, None))]:
        for m in COST_MULTS:
            net = gross[sl] - m * COST_PER_UNIT * turn[sl]
            out[f"sharpe_{m}x_{seg}"] = sharpe_ratio(net, 0.0, PPY)
    out["sharpe_1.0x_is"] = sharpe_ratio(net[:split], 0.0, PPY)
    # Cost per unit of turnover at which the OOS net mean is zero.
    out["breakeven_bps_oos"] = float(gross[split:].mean() / turn[split:].mean() * 1e4)
    out["cagr_1x_oos"] = cagr((1 + net[split:]).cumprod(), PPY)
    out["maxdd_1x_full"] = max_drawdown((1 + net).cumprod())
    out["oos_net"] = net[split:]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(GRID)
    books = [dict(zip(keys, v, strict=True)) for v in itertools.product(*GRID.values())]
    phases = [{**b, "rebalance_every": k, "rebalance_offset": o}
              for b in PHASE_BOOKS for k in GRID["rebalance_every"] if k > 1 for o in range(k)]
    logger.info("%d grid books + %d phase books on %d workers",
                len(books), len(phases), args.workers)
    with Pool(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        rows = pool.map(run_book, books, chunksize=4)
        phase_rows = pool.map(run_book, phases, chunksize=4)

    n_strict = DEV_TRIALS + len(books)
    var_strict = float(np.var([r["oos_net"].mean() / r["oos_net"].std() for r in rows], ddof=1))
    for r in rows + phase_rows:
        oos = r.pop("oos_net")
        r["dsr_oos_lenient"] = deflated_sharpe_ratio(oos, DEV_TRIALS)["dsr"]
        r["dsr_oos_strict"] = deflated_sharpe_ratio(oos, n_strict, var_strict)["dsr"]

    grid = pd.DataFrame(rows)
    grid.to_csv(out / "turnover_sweep.csv", index=False)
    phase = pd.DataFrame(phase_rows)
    phase.to_csv(out / "turnover_sweep_phase.csv", index=False)

    # Honest pick: best book by in-sample net Sharpe, judged on the held-out half.
    pick = grid.loc[grid["sharpe_1.0x_is"].idxmax()]
    meta = {
        "n_books": len(books),
        "n_trials_strict": n_strict,
        "var_sharpe_per_period_strict": var_strict,
        "is_selected": pick.to_dict(),
        "share_profitable_1x_oos": float((grid["sharpe_1.0x_oos"] > 0).mean()),
        "share_profitable_2x_oos": float((grid["sharpe_2.0x_oos"] > 0).mean()),
        "share_dsr_lenient_pass": float((grid["dsr_oos_lenient"] > 0.95).mean()),
        "share_dsr_strict_pass": float((grid["dsr_oos_strict"] > 0.95).mean()),
    }
    (out / "turnover_sweep_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    logger.info("\n%s", json.dumps(meta, indent=2, default=float))

    cols = ["annual_turnover", "sharpe_0.0x_oos", "sharpe_1.0x_oos", "sharpe_2.0x_oos",
            "dsr_oos_lenient"]
    summary = (phase.groupby(["top_n", "lookback", "rebalance_every"])[cols]
               .agg(["mean", "min", "max"]))
    logger.info("\nAcross rebalance phases:\n%s", summary.round(2).to_string())


if __name__ == "__main__":
    main()
