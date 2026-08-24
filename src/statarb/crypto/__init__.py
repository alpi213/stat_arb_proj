"""Crypto perpetual-futures funding carry.

Reuses the equity framework's honesty layer (walk-forward, Deflated Sharpe,
capacity, regime analysis) on a different — and structurally sounder — edge.

Why the edge exists (and this matters more than any backtest):
perpetual futures have no expiry, so exchanges tether the perp price to spot
via a *funding payment* exchanged between longs and shorts every 8h. When the
perp trades above spot (crowded leveraged longs), funding is positive and
longs pay shorts. A delta-neutral position — long spot, short perp — collects
that payment while carrying no directional exposure.

This is a structural risk-transfer premium (leveraged speculators paying for
leverage), not a statistical anomaly found by searching. That is why it
survives out-of-sample far better than cointegration does.

What can go wrong, and is modeled here:
- funding flips negative (you pay) — handled by the signal going flat/short
- basis moves against you before convergence — captured in the P&L
- costs: taker fees on both legs, spread crossing, perp/spot slippage
Not modeled: exchange counterparty risk, liquidation cascades, margin calls
during basis spikes, withdrawal freezes. See docs/CRYPTO_LIMITATIONS.md.
"""

from statarb.crypto.binance import (
    BinanceClient,
    fetch_funding_history,
    fetch_klines,
    list_perp_symbols,
)
from statarb.crypto.funding import build_carry_panel, carry_signal

__all__ = [
    "BinanceClient",
    "fetch_funding_history",
    "fetch_klines",
    "list_perp_symbols",
    "build_carry_panel",
    "carry_signal",
]
