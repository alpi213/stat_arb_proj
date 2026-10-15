"""Crypto perpetual-futures funding carry.

Perps never expire, so exchanges tie them to spot with a funding payment
between longs and shorts every 8h. When leveraged longs crowd in, funding
is positive and longs pay shorts. Long spot plus short perp collects that
payment with no directional exposure. The edge is a payment for risk
transfer, not a pattern found by searching, which is why it holds up out
of sample where cointegration did not.

Modeled: funding flipping sign (the signal goes flat or short), basis
moves, taker fees and spread on both legs. Not modeled: exchange
counterparty risk, liquidation, margin calls, withdrawal freezes. See
docs/LIMITATIONS.md.
"""

from statarb.crypto.binance import BinanceClient, fetch_funding_history, fetch_klines
from statarb.crypto.funding import (
    PERIODS_PER_YEAR,
    build_carry_panel,
    carry_returns,
    carry_signal,
    load_panel,
)

__all__ = [
    "BinanceClient",
    "fetch_funding_history",
    "fetch_klines",
    "PERIODS_PER_YEAR",
    "build_carry_panel",
    "carry_returns",
    "carry_signal",
    "load_panel",
]
