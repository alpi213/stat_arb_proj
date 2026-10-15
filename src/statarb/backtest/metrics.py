"""Performance metrics.

Risk-free rate is 0 unless `rf_annual` is passed. `periods_per_year` is
252 for daily bars and 1095 for 8h crypto settlements; getting it wrong
rescales Sharpe by the square root of the ratio, which is the easiest way
to report a fake number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> float:
    ex = returns - rf_annual / periods_per_year
    sd = ex.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0,
                  periods_per_year: int = TRADING_DAYS) -> float:
    ex = returns - rf_annual / periods_per_year
    downside = ex[ex < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(ex.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    years = len(equity) / periods_per_year
    if years <= 0 or equity.iloc[0] <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity, periods_per_year) / mdd) if mdd > 0 else np.nan


def performance_summary(
    returns: pd.Series,
    equity: pd.Series | None = None,
    trades: list | None = None,
    gross_exposure: pd.Series | None = None,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    turnover: pd.Series | None = None,
) -> pd.Series:
    if equity is None:
        equity = (1 + returns).cumprod()

    out = {
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "n_periods": len(returns),
        "cagr": cagr(equity, periods_per_year),
        "annual_vol": annual_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf_annual, periods_per_year),
        "sortino": sortino_ratio(returns, rf_annual, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods_per_year),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "worst_period": float(returns.min()),
        "best_period": float(returns.max()),
        "pct_positive": float((returns > 0).mean()),
    }

    closed = [t for t in (trades or []) if t.exit_date is not None]
    if closed:
        pnls = np.array([t.pnl_return for t in closed])
        out.update({
            "n_trades": len(closed),
            "hit_rate": float((pnls > 0).mean()),
            "avg_win": float(pnls[pnls > 0].mean()) if (pnls > 0).any() else np.nan,
            "avg_loss": float(pnls[pnls < 0].mean()) if (pnls < 0).any() else np.nan,
            "avg_holding_days": float(np.mean([t.holding_days for t in closed])),
            "stop_rate": float(np.mean([t.exit_reason == "exit_stop" for t in closed])),
        })

    if gross_exposure is not None:
        out["avg_gross_exposure"] = float(gross_exposure.mean())
    if turnover is not None:
        out["avg_turnover"] = float(turnover.mean())
        out["annual_turnover"] = float(turnover.mean() * periods_per_year)

    return pd.Series(out)
