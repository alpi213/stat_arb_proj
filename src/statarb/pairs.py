"""Find cointegrated pairs and rank them.

Engle-Granger in two steps: regress log y on log x, then test the residual
for stationarity. If the residual is stationary the pair is cointegrated
and the residual is the thing to trade.

Log prices throughout: the hedge ratio is then an elasticity, the spread is
symmetric in percentage terms, and dollar sizing falls out directly.

Two things worth knowing about the test:
- It is not symmetric. Regressing y on x and x on y can disagree, so both
  directions are run and the stronger one kept.
- The ADF critical values for a fitted residual are not the plain ADF ones.
  statsmodels' `coint` uses the right (MacKinnon) surfaces, so it is used
  instead of a hand-rolled ADF on the residual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CointResult:
    y: str                 # leg held long when the spread is low
    x: str                 # hedge leg
    alpha: float
    beta: float            # hedge ratio from the log-price regression
    pvalue: float          # Engle-Granger p-value
    half_life_days: float  # of the residual


def engle_granger(log_y: pd.Series, log_x: pd.Series) -> CointResult:
    """Test both directions and return the one with the lower p-value."""
    df = pd.concat({"y": log_y, "x": log_x}, axis=1).dropna()
    if len(df) < 100:
        raise ValueError(f"too few overlapping observations: {len(df)}")

    directions = (
        (str(log_y.name), df["y"], str(log_x.name), df["x"]),
        (str(log_x.name), df["x"], str(log_y.name), df["y"]),
    )
    best: CointResult | None = None
    for y_name, a, x_name, b in directions:
        _, pvalue, _ = coint(a, b)
        ols = sm.OLS(a.to_numpy(), sm.add_constant(b.to_numpy())).fit()
        resid = pd.Series(ols.resid, index=df.index)
        result = CointResult(
            y=y_name,
            x=x_name,
            alpha=float(ols.params[0]),
            beta=float(ols.params[1]),
            pvalue=float(pvalue),
            half_life_days=half_life(resid),
        )
        if best is None or result.pvalue < best.pvalue:
            best = result
    assert best is not None
    return best


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life from an AR(1) fit.

    d(spread)_t = theta * spread_{t-1} + noise, so HL = -ln 2 / ln(1 + theta).
    theta >= 0 means no reversion at all; return +inf so filters reject it.
    """
    s = spread.dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    theta = sm.OLS(ds.to_numpy(), sm.add_constant(lag.to_numpy())).fit().params[1]
    if theta >= 0:
        return float("inf")
    return float(-np.log(2.0) / np.log(1.0 + theta))


@dataclass(frozen=True)
class PairCandidate:
    result: CointResult
    sector: str
    rank_score: float  # lower is better


@dataclass
class SelectionOutput:
    pairs: list[PairCandidate]
    n_tested: int  # every hypothesis tested, for the Deflated Sharpe later
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp


def select_pairs(
    adj_close: pd.DataFrame,
    sectors: dict[str, list[str]],
    formation_start: str | pd.Timestamp,
    formation_end: str | pd.Timestamp,
    adf_pvalue_max: float = 0.05,
    min_half_life_days: float = 2.0,
    max_half_life_days: float = 60.0,
    top_n: int = 20,
    max_per_name: int = 2,
) -> SelectionOutput:
    """Rank within-sector pairs by cointegration strength on the formation window.

    The window is sliced here, not by the caller, so nobody can accidentally
    select pairs on the full sample. Within-sector only: it is an economic
    prior, and it cuts the number of hypotheses tested. `max_per_name` stops
    the portfolio from becoming one big bet on a single ticker.
    """
    start, end = pd.Timestamp(formation_start), pd.Timestamp(formation_end)
    log_px = np.log(adj_close.loc[start:end])

    candidates: list[PairCandidate] = []
    n_tested = 0
    for sector, tickers in sectors.items():
        available = [t for t in tickers if t in log_px.columns]
        for a, b in combinations(available, 2):
            n_tested += 1
            try:
                res = engle_granger(log_px[a].rename(a), log_px[b].rename(b))
            except (ValueError, np.linalg.LinAlgError):
                continue
            if res.pvalue > adf_pvalue_max:
                continue
            if not (min_half_life_days <= res.half_life_days <= max_half_life_days):
                continue
            if res.beta <= 0:  # a negative hedge ratio is a momentum bet, not a pair
                continue
            # Rank by p-value, tie-break toward faster reversion.
            score = res.pvalue + 0.001 * res.half_life_days
            candidates.append(PairCandidate(result=res, sector=sector, rank_score=score))

    candidates.sort(key=lambda c: c.rank_score)

    picked: list[PairCandidate] = []
    uses: dict[str, int] = {}
    for c in candidates:
        y, x = c.result.y, c.result.x
        if uses.get(y, 0) >= max_per_name or uses.get(x, 0) >= max_per_name:
            continue
        picked.append(c)
        uses[y] = uses.get(y, 0) + 1
        uses[x] = uses.get(x, 0) + 1
        if len(picked) >= top_n:
            break

    logger.info(
        "Formation %s..%s: tested %d pairs, %d passed filters, kept %d",
        start.date(), end.date(), n_tested, len(candidates), len(picked),
    )
    return SelectionOutput(picked, n_tested, start, end)
