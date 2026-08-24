"""Pair selection — the part of the pipeline where leakage usually happens.

Discipline enforced here:
- Selection uses ONLY data inside [formation_start, formation_end]. The
  function takes an explicit window and slices internally, so a caller
  cannot accidentally pass the full sample.
- Candidates are restricted to within-sector pairs: an economic prior that
  cuts the number of hypotheses tested (see Deflated Sharpe in
  validation/) and avoids spurious cross-sector cointegration.
- The multiple-testing count (`n_tested`) is returned alongside the picks —
  it feeds the Deflated Sharpe Ratio later. Don't throw it away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from statarb.pairs.cointegration import CointResult, engle_granger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairCandidate:
    result: CointResult
    sector: str
    rank_score: float  # lower is better


@dataclass
class SelectionOutput:
    pairs: list[PairCandidate]
    n_tested: int          # total hypotheses tested — needed for DSR
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

    `max_per_name` caps how many selected pairs share one ticker, so the
    portfolio isn't secretly one big bet on a single name.
    """
    start, end = pd.Timestamp(formation_start), pd.Timestamp(formation_end)
    window = adj_close.loc[start:end]
    log_px = np.log(window)

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
            if res.beta <= 0:
                # negative hedge ratio => "pair" is actually a momentum bet
                continue
            # Rank: primarily p-value, tie-broken toward faster reversion.
            score = res.pvalue + 0.001 * res.half_life_days
            candidates.append(PairCandidate(result=res, sector=sector, rank_score=score))

    candidates.sort(key=lambda c: c.rank_score)

    picked: list[PairCandidate] = []
    name_counts: dict[str, int] = {}
    for c in candidates:
        y, x = c.result.y, c.result.x
        if name_counts.get(y, 0) >= max_per_name or name_counts.get(x, 0) >= max_per_name:
            continue
        picked.append(c)
        name_counts[y] = name_counts.get(y, 0) + 1
        name_counts[x] = name_counts.get(x, 0) + 1
        if len(picked) >= top_n:
            break

    logger.info(
        "Formation %s..%s: tested %d pairs, %d passed filters, kept %d",
        start.date(), end.date(), n_tested, len(candidates), len(picked),
    )
    return SelectionOutput(
        pairs=picked, n_tested=n_tested, formation_start=start, formation_end=end
    )
