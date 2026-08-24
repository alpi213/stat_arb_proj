"""Crypto step 4: walk-forward hyperparameter selection for the carry strategy.

The single-split result (Sharpe 3.23 OOS) used hyperparameters chosen BY HAND
after looking at the whole sample — lookback=21, top_n=10, rebalance_every=21,
hysteresis=3.0, max_predicted_bps=50. Even with a held-out split, that choice
saw the out-of-sample data through the researcher's eyes. This script removes
that channel:

    on each formation window   -> pick the best config by IN-WINDOW Sharpe
    on the following window    -> trade that config, out of sample
    stitch the OOS segments    -> an equity curve where no hyperparameter was
                                  ever chosen with knowledge of the period it
                                  traded

Carry has no pair-selection step, so unlike the equity path the thing being
re-selected is the configuration itself. If the stitched OOS Sharpe collapses
relative to the hand-tuned single split, the hand-tuning was the edge. If it
holds up, the config choice is incidental and the premium is doing the work.

Usage:
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

from statarb.backtest import CrossSectionalCostModel, backtest_weights, performance_summary
from statarb.crypto.funding import (
    PERIODS_PER_YEAR,
    build_carry_panel,
    carry_returns,
    carry_signal,
)
from statarb.utils import setup_logging
from statarb.utils.plotting import plot_equity
from statarb.validation import deflated_sharpe_ratio, make_windows

logger = logging.getLogger("carry_wf")

# Deliberately small grid. Every entry is a trial, and the DSR at the end is
# charged for all of them (n_windows x len(grid)).
GRID = list(itertools.product(
    [9, 21, 63],      # lookback (3d, 7d, 21d of 8h settlements)
    [5, 10],          # top_n per side
    [9, 21],          # rebalance_every
))


def slice_panel(panel: dict[str, pd.DataFrame], start, end) -> dict[str, pd.DataFrame]:
    return {k: v.loc[start:end] for k, v in panel.items()}


def run_config(panel, rets, lookback, top_n, reb, costs) -> pd.Series:
    w = carry_signal(
        panel, lookback=lookback, top_n=top_n,
        rebalance_every=reb, hysteresis=3.0, max_predicted_bps=50.0,
    )
    idx = w.index.intersection(rets.index)
    res = backtest_weights(w.loc[idx], rets.loc[idx], costs=costs)
    return res["returns"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--formation", type=int, default=1095)  # ~1y of 8h bars
    p.add_argument("--trading", type=int, default=274)     # ~3m
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
    rets = carry_returns(panel)
    costs = CrossSectionalCostModel(periods_per_year=PERIODS_PER_YEAR)

    windows = make_windows(
        panel["funding"].index,
        formation_days=args.formation,
        trading_days=args.trading,
        step_days=args.trading,
    )
    logger.info("walk-forward: %d windows, grid of %d configs", len(windows), len(GRID))

    segments, rows = [], []
    for i, w in enumerate(windows):
        # --- formation: score every config on IN-WINDOW data only ---
        form_panel = slice_panel(panel, w.formation_start, w.formation_end)
        form_rets = rets.loc[w.formation_start: w.formation_end]
        best, best_sr = None, -np.inf
        for lookback, top_n, reb in GRID:
            r = run_config(form_panel, form_rets, lookback, top_n, reb, costs)
            if len(r.dropna()) < 50:
                continue
            sr = performance_summary(r, periods_per_year=PERIODS_PER_YEAR)["sharpe"]
            if pd.notna(sr) and sr > best_sr:
                best, best_sr = (lookback, top_n, reb), sr

        if best is None:
            logger.warning("window %d: no viable config", i)
            continue

        # --- trading: apply the chosen config out of sample ---
        # The signal needs formation history to warm up its EWMA, so the panel
        # spans formation_start..trading_end, but only the trading segment is
        # kept for P&L. Weights inside the trading segment still depend only
        # on data at or before their own timestamp.
        trade_panel = slice_panel(panel, w.formation_start, w.trading_end)
        trade_rets = rets.loc[w.formation_start: w.trading_end]
        r_full = run_config(trade_panel, trade_rets, *best, costs)
        seg = r_full.loc[w.trading_start: w.trading_end]
        segments.append(seg)

        seg_sr = performance_summary(seg, periods_per_year=PERIODS_PER_YEAR)["sharpe"]
        rows.append({
            "window": i,
            "trading_start": str(w.trading_start.date()),
            "trading_end": str(w.trading_end.date()),
            "chosen_lookback": best[0], "chosen_top_n": best[1], "chosen_rebalance": best[2],
            "in_sample_sharpe": round(float(best_sr), 2),
            "oos_sharpe": round(float(seg_sr), 2) if pd.notna(seg_sr) else None,
        })
        logger.info(
            "window %d [%s..%s] chose lookback=%d top_n=%d reb=%d | IS %.2f -> OOS %.2f",
            i, w.trading_start.date(), w.trading_end.date(), *best, best_sr,
            seg_sr if pd.notna(seg_sr) else float("nan"),
        )

    if not segments:
        raise SystemExit("no OOS segments produced")

    stitched = pd.concat(segments).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    equity = (1 + stitched).cumprod()
    summary = performance_summary(
        stitched, equity, periods_per_year=PERIODS_PER_YEAR
    )

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    wf = pd.DataFrame(rows)
    wf.to_csv(out / "carry_walkforward_windows.csv", index=False)
    stitched.to_csv(out / "carry_walkforward_returns.csv")
    summary.to_csv(out / "carry_walkforward_summary.csv")
    plot_equity(equity * 1e6, out / "figures" / "carry_walkforward_equity.png",
                title="Crypto carry — walk-forward OOS (configs re-chosen per window)")

    logger.info("\n=== WALK-FORWARD WINDOWS ===\n%s", wf.to_string(index=False))
    logger.info("\n=== STITCHED OOS ===\n%s", summary.to_string())

    # Honest trial count: every config scored, in every window.
    n_trials = len(GRID) * max(len(rows), 1)
    try:
        dsr = deflated_sharpe_ratio(stitched, n_trials=n_trials)
        (out / "carry_walkforward_dsr.json").write_text(
            json.dumps(dsr, indent=2, default=str)
        )
        logger.info("\n=== DSR (walk-forward, n_trials=%d) ===\n%s",
                    n_trials, json.dumps(dsr, indent=2, default=str))
    except ValueError as exc:
        logger.warning("DSR skipped: %s", exc)

    # Config stability is the real diagnostic: if the winner jumps around every
    # window, the in-window Sharpe is fitting noise, not finding a setting.
    if len(wf) > 1:
        modal = wf[["chosen_lookback", "chosen_top_n", "chosen_rebalance"]].mode().iloc[0]
        stable = (
            wf[["chosen_lookback", "chosen_top_n", "chosen_rebalance"]] == modal
        ).all(axis=1).mean()
        logger.info("modal config %s chosen in %.0f%% of windows",
                    modal.to_dict(), stable * 100)


if __name__ == "__main__":
    main()
