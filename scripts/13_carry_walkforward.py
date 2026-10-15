"""Walk-forward over carry hyperparameters.

The single-split backtest uses a config picked by hand after seeing the
whole sample. Here each formation window picks the best config from a
small grid by in-window Sharpe, trades it on the next window, and the
out-of-sample segments are stitched. Carry has no pair selection, so the
thing being re-chosen is the config itself.

    python scripts/13_carry_walkforward.py [--formation 1095] [--trading 274]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import backtest_weights, performance_summary
from statarb.crypto import PERIODS_PER_YEAR, carry_returns, carry_signal, load_panel
from statarb.utils import plot_equity, setup_logging
from statarb.validation import deflated_sharpe_ratio, make_windows

logger = logging.getLogger("carry_wf")

# Small on purpose: every entry is a trial the DSR gets charged for.
GRID = list(itertools.product(
    [9, 21, 63],   # lookback: 3, 7, 21 days of 8h settlements
    [5, 10],       # top_n per side
    [9, 21],       # rebalance_every
))


def run_config(panel, rets, lookback, top_n, rebalance_every) -> pd.Series:
    w = carry_signal(panel, lookback=lookback, top_n=top_n, rebalance_every=rebalance_every)
    idx = w.index.intersection(rets.index)
    return backtest_weights(w.loc[idx], rets.loc[idx])["returns"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--formation", type=int, default=1095, help="bars, ~1 year")
    p.add_argument("--trading", type=int, default=274, help="bars, ~3 months")
    args = p.parse_args()
    setup_logging()

    panel = load_panel(args.data)
    rets = carry_returns(panel)
    ppy = PERIODS_PER_YEAR
    windows = make_windows(panel["funding"].index, args.formation, args.trading, args.trading)
    logger.info("walk-forward: %d windows, grid of %d configs", len(windows), len(GRID))

    def sliced(start, end):
        return {k: v.loc[start:end] for k, v in panel.items()}, rets.loc[start:end]

    segments, rows = [], []
    for i, w in enumerate(windows):
        # Score every config on the formation window only.
        form_panel, form_rets = sliced(w.formation_start, w.formation_end)
        best, best_sr = None, -np.inf
        for cfg in GRID:
            r = run_config(form_panel, form_rets, *cfg)
            if len(r.dropna()) < 50:
                continue
            sr = performance_summary(r, periods_per_year=ppy)["sharpe"]
            if pd.notna(sr) and sr > best_sr:
                best, best_sr = cfg, sr
        if best is None:
            logger.warning("window %d: no viable config", i)
            continue

        # Trade it on the next window. The signal warms up on formation data,
        # so the panel spans both, but only the trading segment is kept.
        trade_panel, trade_rets = sliced(w.formation_start, w.trading_end)
        seg = run_config(trade_panel, trade_rets, *best).loc[w.trading_start: w.trading_end]
        segments.append(seg)

        seg_sr = performance_summary(seg, periods_per_year=ppy)["sharpe"]
        rows.append({
            "window": i,
            "trading_start": str(w.trading_start.date()),
            "trading_end": str(w.trading_end.date()),
            "chosen_lookback": best[0], "chosen_top_n": best[1], "chosen_rebalance": best[2],
            "in_sample_sharpe": round(float(best_sr), 2),
            "oos_sharpe": round(float(seg_sr), 2) if pd.notna(seg_sr) else None,
        })
        logger.info("window %d [%s..%s] chose lookback=%d top_n=%d reb=%d | IS %.2f -> OOS %.2f",
                    i, w.trading_start.date(), w.trading_end.date(), *best, best_sr,
                    seg_sr if pd.notna(seg_sr) else float("nan"))

    if not segments:
        raise SystemExit("no out-of-sample segments produced")

    stitched = pd.concat(segments).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    equity = (1 + stitched).cumprod()
    summary = performance_summary(stitched, equity, periods_per_year=ppy)

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    chosen = pd.DataFrame(rows)
    chosen.to_csv(out / "carry_walkforward_windows.csv", index=False)
    stitched.to_csv(out / "carry_walkforward_returns.csv")
    summary.to_csv(out / "carry_walkforward_summary.csv")
    plot_equity(equity * 1e6, out / "figures" / "carry_walkforward_equity.png",
                title="Crypto carry, walk-forward OOS (config re-chosen per window)")
    logger.info("\nWindows:\n%s", chosen.to_string(index=False))
    logger.info("\nStitched OOS:\n%s", summary.to_string())

    n_trials = len(GRID) * max(len(rows), 1)  # every config scored in every window
    try:
        dsr = deflated_sharpe_ratio(stitched, n_trials=n_trials)
        (out / "carry_walkforward_dsr.json").write_text(json.dumps(dsr, indent=2, default=str))
        logger.info("\nDSR (n_trials=%d):\n%s", n_trials, json.dumps(dsr, indent=2, default=str))
    except ValueError as exc:
        logger.warning("DSR skipped: %s", exc)

    # If the winner jumps around every window, the in-window Sharpe is fitting noise.
    if len(chosen) > 1:
        cols = ["chosen_lookback", "chosen_top_n", "chosen_rebalance"]
        modal = chosen[cols].mode().iloc[0]
        share = (chosen[cols] == modal).all(axis=1).mean()
        logger.info("modal config %s chosen in %.0f%% of windows", modal.to_dict(), share * 100)


if __name__ == "__main__":
    main()
