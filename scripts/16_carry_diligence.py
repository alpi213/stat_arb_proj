"""Diligence checks on the base carry book, one output file per README claim.

    python scripts/16_carry_diligence.py

Writes to results_crypto/:
    pnl_attribution.csv            funding vs basis, per bar
    breadth.csv                    per-symbol P&L; effective breadth in diligence.json
    top_k_removal.csv              Sharpe with the best k bars removed
    stress_regimes.csv             named crisis windows
    concentration.csv              top-symbol share with and without the squeeze gate
    naive_book.csv                 the same signal re-ranked every 8h
    survivorship.csv               point-in-time universe vs survivors only
    exit_penalty_sensitivity.csv   charge for exiting a dying contract
    rebalance_phase_worst_bar.csv  base book at each rebalance phase, worst bar
    carry_walkforward_probe.csv    walk-forward window 7, lookback varied alone
    diligence.json                 the scalars quoted in the README
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest import backtest_weights, exit_size, performance_summary
from statarb.backtest.metrics import cagr, max_drawdown, sharpe_ratio
from statarb.crypto import PERIODS_PER_YEAR, carry_returns, carry_signal, load_panel
from statarb.utils import setup_logging
from statarb.validation import make_windows, regime_breakdown

logger = logging.getLogger("diligence")

PPY = PERIODS_PER_YEAR
BASE = dict(lookback=21, top_n=10, min_predicted_bps=1.0, max_predicted_bps=50.0,
            rebalance_every=21, hysteresis=3.0)  # 12_carry_backtest.py defaults
REGIMES = {
    "luna_collapse": ("2022-05-05", "2022-05-20"),
    "three_ac_celsius": ("2022-06-10", "2022-06-25"),
    "ftx_collapse": ("2022-11-06", "2022-11-20"),
    "rally_2023_24": ("2023-10-01", "2024-03-31"),
    "aug_2024_unwind": ("2024-08-01", "2024-08-10"),
    "drawdown_2025": ("2025-02-01", "2025-04-30"),
}
EXIT_PENALTIES = [0.0, 0.1, 0.25, 0.5]


def sharpe(r: pd.Series) -> float:
    return sharpe_ratio(r, 0.0, PPY)


def run(panel: dict, rets: pd.DataFrame, **kw) -> dict:
    return backtest_weights(carry_signal(panel, **{**BASE, **kw}), rets)


def attribution(panel: dict, res: dict) -> tuple[pd.DataFrame, dict]:
    """Split gross P&L into the three terms of carry_returns."""
    w = res["weights"]
    terms = {
        "funding": panel["funding"].shift(-1),
        "spot": panel["spot"].pct_change(fill_method=None).shift(-1),
        "perp": -panel["perp"].pct_change(fill_method=None).shift(-1),
    }
    df = pd.DataFrame({k: (w * v.reindex_like(w).fillna(0.0)).sum(axis=1)
                       for k, v in terms.items()})
    df["basis"] = df["spot"] + df["perp"]
    df["gross"] = res["gross_returns"]
    df["costs"] = res["gross_returns"] - res["returns"]
    df["net"] = res["returns"]
    tot = df.sum()
    return df, {
        "funding_share_of_gross": tot["funding"] / tot["gross"],
        "basis_share_of_gross": tot["basis"] / tot["gross"],
        "max_abs_decomposition_error":
            float((df["funding"] + df["basis"] - df["gross"]).abs().max()),
    }


def breadth(res: dict) -> tuple[pd.DataFrame, dict]:
    """(sum of per-symbol P&L std)^2 / var(total P&L), over symbols ever held."""
    pnl = res["per_asset_pnl"].loc[:, (res["weights"] != 0).any()]
    held = res["weights"][pnl.columns] != 0
    corr = pnl.corr().to_numpy()
    table = pd.DataFrame({
        "pnl_total": pnl.sum(),
        "pnl_variance": pnl.var(),
        "bars_held": held.sum(),
        "max_abs_weight": res["weights"][pnl.columns].abs().max(),
        "share_of_gross_pnl": pnl.sum() / res["gross_returns"].sum(),
    }).sort_values("pnl_total", ascending=False)
    n_pos = (res["weights"] != 0).sum(axis=1)
    return table, {
        "effective_breadth": float(pnl.std().sum() ** 2 / pnl.sum(axis=1).var()),
        "avg_pairwise_corr": float(np.nanmean(corr[~np.eye(len(corr), dtype=bool)])),
        "n_unique_symbols_held": int(pnl.shape[1]),
        "max_simultaneous_positions": int(n_pos.max()),
        "avg_simultaneous_positions_when_invested": float(n_pos[n_pos > 0].mean()),
    }


def top_k_removal(net: pd.Series, split: int) -> pd.DataFrame:
    rows = []
    for k in [0, 1, 5, 10, 20]:
        full, oos = net.drop(net.nlargest(k).index), net.iloc[split:]
        oos = oos.drop(oos.nlargest(k).index)
        rows.append({"k_removed": k, "sharpe_full": sharpe(full), "sharpe_oos": sharpe(oos)})
    return pd.DataFrame(rows)


def stress(net: pd.Series) -> pd.DataFrame:
    rows = ["start", "end", "n_periods", "total_return", "sharpe", "max_drawdown",
            "annual_vol", "worst_period"]
    return regime_breakdown(net, REGIMES, periods_per_year=PPY).loc[rows].T


def concentration(panel: dict, rets: pd.DataFrame, base: dict) -> pd.DataFrame:
    rows = []
    for label, res in [("gated (base)", base),
                       ("no max_predicted_bps gate", run(panel, rets, max_predicted_bps=np.inf))]:
        by_sym = res["per_asset_pnl"].sum().sort_values(ascending=False)
        gross = res["gross_returns"].sum()
        rows.append({"run": label, "gross_pnl_total": gross, "sharpe_full": sharpe(res["returns"]),
                     "top_symbol": by_sym.index[0], "top_symbol_share": by_sym.iloc[0] / gross,
                     "top3_symbols": ",".join(by_sym.index[:3]),
                     "top3_share": by_sym.iloc[:3].sum() / gross})
    return pd.DataFrame(rows)


def survivorship(panel: dict, rets: pd.DataFrame, base: dict, pit_path: Path, split: int):
    pit = pd.read_csv(pit_path).set_index("symbol")["delisted"]
    dead = [s for s in panel["funding"].columns if pit.get(s, False)]
    alive = [s for s in panel["funding"].columns if s not in dead]
    surv = run({k: v[alive] for k, v in panel.items()}, rets[alive])
    rows = []
    for label, res, n, nd in [("point_in_time", base, len(alive) + len(dead), len(dead)),
                              ("survivors_only", surv, len(alive), 0)]:
        net = res["returns"]
        rows.append({"universe": label, "n_symbols": n, "n_delisted": nd,
                     "sharpe_full": sharpe(net), "sharpe_oos": sharpe(net.iloc[split:]),
                     "cagr": cagr((1 + net).cumprod(), PPY),
                     "max_drawdown": max_drawdown((1 + net).cumprod())})
    return pd.DataFrame(rows), dead


def exit_penalty(panel: dict, base: dict, split: int) -> pd.DataFrame:
    """Charge a fraction of the position each time a held contract's return goes
    missing. The engine otherwise exits it at the last good price for free."""
    size = exit_size(carry_signal(panel, **BASE), base)
    rows = []
    for p in EXIT_PENALTIES:
        net = base["returns"] - p * size
        rows.append({"exit_penalty": p, "sharpe_full": sharpe(net),
                     "sharpe_oos": sharpe(net.iloc[split:]),
                     "max_drawdown_full": max_drawdown((1 + net).cumprod()),
                     "total_exit_cost": float(p * size.sum()), "n_exits": int((size > 0).sum())})
    return pd.DataFrame(rows)


def phase_worst_bar(panel: dict, rets: pd.DataFrame, split: int) -> pd.DataFrame:
    """The base book at every rebalance phase, with its worst bar and who caused it."""
    rows = []
    for offset in range(BASE["rebalance_every"]):
        res = run(panel, rets, rebalance_offset=offset)
        t = res["returns"].idxmin()
        pnl = res["per_asset_pnl"].loc[t]
        rows.append({"rebalance_offset": offset, "sharpe_oos": sharpe(res["returns"].iloc[split:]),
                     "worst_bar": t, "worst_return": float(res["returns"].min()),
                     "worst_symbol": pnl.idxmin(), "worst_symbol_pnl": float(pnl.min())})
    return pd.DataFrame(rows)


def walkforward_probe(panel: dict, rets: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Window 7 of 13_carry_walkforward.py, varying only lookback."""
    w = make_windows(panel["funding"].index, 1095, 274, 274)[window]
    sub = {k: v.loc[w.formation_start: w.trading_end] for k, v in panel.items()}
    rows = []
    for lookback in [9, 21, 63]:
        res = backtest_weights(carry_signal(sub, lookback=lookback, top_n=10, rebalance_every=21),
                               rets.loc[w.formation_start: w.trading_end])
        seg = res["returns"].loc[w.trading_start: w.trading_end]
        wts = res["weights"].loc[w.trading_start: w.trading_end]
        pnl = res["per_asset_pnl"].loc[w.trading_start: w.trading_end]
        unfi = pnl.get("UNFIUSDT", pd.Series(0.0))
        rows.append({"window": window, "lookback": lookback, "oos_sharpe": sharpe(seg),
                     "annual_vol": float(seg.std() * np.sqrt(PPY)),
                     "max_drawdown": max_drawdown((1 + seg).cumprod()),
                     "total_return": float((1 + seg).prod() - 1),
                     "max_weight_UNFIUSDT": float(wts.get("UNFIUSDT", pd.Series(0.0)).abs().max()),
                     "pnl_UNFIUSDT": float(unfi.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crypto")
    p.add_argument("--out", default="results_crypto")
    p.add_argument("--pit", default="configs/universe_perp_pit.csv")
    args = p.parse_args()
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.data)
    rets = carry_returns(panel)
    base = run(panel, rets)
    net = base["returns"]
    split = len(net) // 2  # same 50/50 split as 12_carry_backtest.py

    attr, attr_meta = attribution(panel, base)
    attr.to_csv(out / "pnl_attribution.csv")
    br, br_meta = breadth(base)
    br.to_csv(out / "breadth.csv")
    top_k_removal(net, split).to_csv(out / "top_k_removal.csv", index=False)
    stress(net).to_csv(out / "stress_regimes.csv")
    conc = concentration(panel, rets, base)
    conc.to_csv(out / "concentration.csv", index=False)

    naive = run(panel, rets, rebalance_every=1, hysteresis=1.0)
    naive_tab = performance_summary(naive["returns"], periods_per_year=PPY,
                                    turnover=naive["turnover"])
    naive_tab["sharpe_gross"] = sharpe(naive["gross_returns"])
    naive_tab["sharpe_oos"] = sharpe(naive["returns"].iloc[split:])
    naive_tab.to_csv(out / "naive_book.csv")

    surv, dead = survivorship(panel, rets, base, Path(args.pit), split)
    surv.to_csv(out / "survivorship.csv", index=False)
    exit_penalty(panel, base, split).to_csv(out / "exit_penalty_sensitivity.csv", index=False)
    phase_worst_bar(panel, rets, split).to_csv(out / "rebalance_phase_worst_bar.csv", index=False)
    walkforward_probe(panel, rets).to_csv(out / "carry_walkforward_probe.csv", index=False)

    meta = {"signal_kwargs": BASE, "oos_start": str(net.index[split]), "attribution": attr_meta,
            "breadth": br_meta, "delisted_symbols": dead,
            "lunausdt_pnl_total":
                float(base["per_asset_pnl"].get("LUNAUSDT", pd.Series(0.0)).sum())}
    (out / "diligence.json").write_text(json.dumps(meta, indent=2, default=float))
    logger.info("\n%s", json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    main()
