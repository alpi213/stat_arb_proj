"""Point-in-time universe handling.

Survivorship bias is the #1 silent killer of stat arb backtests: if you take
*today's* index members and backtest them on 2018, you have conditioned on
survival — the delisted names (the ones whose pairs blew up) are exactly the
ones you removed.

The universe file is a point-in-time membership table:

    ticker,sector,entry_date,exit_date

- ``entry_date``: first date the name is eligible for pair formation.
- ``exit_date``: last date it may be held (blank = still active). A name that
  was delisted/acquired gets its real exit date, and the backtest must force
  positions flat at that date.

The shipped ``configs/universe.csv`` is a *static* list of large, long-lived
names — an honest limitation documented in docs/LIMITATIONS.md. To upgrade to
true point-in-time S&P membership, replace the file with one generated from a
PIT source (e.g. Wikipedia edit history of the S&P 500 constituents page, or
a commercial dataset) — the code below already supports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Member:
    ticker: str
    sector: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None  # None = still active


class Universe:
    def __init__(self, members: list[Member]):
        self.members = members

    @property
    def tickers(self) -> list[str]:
        return sorted({m.ticker for m in self.members})

    def active_on(self, date: pd.Timestamp | str) -> list[Member]:
        """Members that are in the universe on `date` (point-in-time view)."""
        d = pd.Timestamp(date)
        return [
            m
            for m in self.members
            if m.entry_date <= d and (m.exit_date is None or d <= m.exit_date)
        ]

    def active_through(
        self, start: pd.Timestamp | str, end: pd.Timestamp | str
    ) -> list[Member]:
        """Members active for the *entire* [start, end] window.

        Used for pair formation: a pair needs a full formation window of
        overlapping history. Names that exit mid-window are excluded from
        *formation* but their exit still forces liquidation in *trading*.
        """
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return [
            m
            for m in self.members
            if m.entry_date <= s and (m.exit_date is None or e <= m.exit_date)
        ]

    def sectors(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for m in self.members:
            out.setdefault(m.sector, []).append(m.ticker)
        return {k: sorted(set(v)) for k, v in out.items()}

    def exit_date_of(self, ticker: str) -> pd.Timestamp | None:
        for m in self.members:
            if m.ticker == ticker:
                return m.exit_date
        return None


def load_universe(path: str | Path) -> Universe:
    df = pd.read_csv(path)
    required = {"ticker", "sector", "entry_date", "exit_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe file {path} missing columns: {missing}")
    members = []
    for row in df.itertuples(index=False):
        exit_date = pd.Timestamp(row.exit_date) if pd.notna(row.exit_date) else None
        members.append(
            Member(
                ticker=str(row.ticker).strip(),
                sector=str(row.sector).strip(),
                entry_date=pd.Timestamp(row.entry_date),
                exit_date=exit_date,
            )
        )
    return Universe(members)
