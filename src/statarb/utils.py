"""Logging setup and the three standard plots."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_equity(equity: pd.Series, out: Path, title: str = "Equity curve") -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    ax1.plot(equity.index, equity.values, lw=1.2)
    ax1.set_title(title)
    ax1.set_ylabel("Equity ($)")
    dd = equity / equity.cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="tab:red")
    ax2.set_ylabel("Drawdown")
    return _save(fig, out)


def plot_pair_diagnostics(result, out: Path) -> Path:
    """Spread, z-score with entry/stop bands, and the beta path for one pair."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result.spread.index, result.spread.values, lw=0.8)
    axes[0].set_title(f"{result.pair}: spread")
    axes[1].plot(result.zscore.index, result.zscore.values, lw=0.8)
    for lvl, c in [(2, "tab:orange"), (-2, "tab:orange"), (3.5, "tab:red"), (-3.5, "tab:red")]:
        axes[1].axhline(lvl, ls="--", lw=0.6, color=c)
    axes[1].set_title("z-score")
    axes[2].plot(result.beta.index, result.beta.values, lw=0.8, color="tab:green")
    axes[2].set_title("hedge ratio (beta)")
    return _save(fig, out)


def plot_capacity_curve(curve: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(curve.index, curve["sharpe"], marker="o")
    ax.axhline(0, color="k", lw=0.8)
    half = curve.attrs.get("aum_sharpe_half")
    zero = curve.attrs.get("aum_sharpe_zero")
    if half:
        ax.axvline(half, ls="--", color="tab:orange", label=f"Sharpe halves ~ ${half/1e6:.0f}M")
    if zero:
        ax.axvline(zero, ls="--", color="tab:red", label=f"Sharpe ~ 0 at ~ ${zero/1e6:.0f}M")
    ax.set_xlabel("AUM ($, log scale)")
    ax.set_ylabel("Net Sharpe (after impact)")
    ax.set_title("Capacity curve: net Sharpe vs AUM")
    ax.legend()
    return _save(fig, out)
