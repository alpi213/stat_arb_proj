# Crypto funding-carry: limitations and failure modes

The equity strategy's honest conclusion was "the edge doesn't survive costs."
Crypto carry has a real edge — which makes it *more* important, not less, to
be explicit about what would kill it. A Sharpe 2-3 result here is only
credible alongside this list.

## Data

1. **Survivorship — MEASURED, and smaller than expected.** Built a true
   point-in-time universe from Binance's own archive (`data.binance.vision`,
   `scripts/14_build_pit_universe.py`): 833 USDT perps ever listed, 180
   delisted, with verified active windows (LUNAUSDT 2021-01→2022-05, matching
   its collapse). Ran the strategy on 130 symbols active at 2022-01-01, 15 of
   them since delisted, against the same 115 that survived:

   |                     | Sharpe (full) | Sharpe (OOS) | CAGR  | Max DD |
   | ------------------- | ------------- | ------------ | ----- | ------ |
   | Survivor-only (115) | 5.81          | 5.68         | 22.9% | −2.6% |
   | Point-in-time (130) | 5.79          | 5.58         | 22.3% | −2.5% |

   Survivorship inflates Sharpe by **+0.02 full / +0.10 OOS** — far smaller
   than assumed. Delta-neutrality is structurally protective: when a coin
   collapses, BOTH legs collapse together. LUNAUSDT contributed **+0.025% of
   AUM** across its entire May 2022 death.

   **Reaching that number required three fixes, and produced two wrong
   conclusions on the way — both recorded here rather than deleted.**
   Backtesting contracts through their own deaths is exactly where exchange
   data is least trustworthy:

   a. *Stale ffill at delisting.* `ffill(limit=1)` ran independently per
   field, so when perp and spot stopped printing at different times the
   dead leg's price froze for an extra bar while the live one moved —
   fabricating a basis divergence (measured: spot_ret −96%, perp_ret 0%
   on the same LUNAUSDT bar). Fixed by truncating both legs at their
   shared last real print.

   b. *Zero-volume bars treated as prices.* The fix in (a) only catches data
   that STOPS. Binance keeps PUBLISHING bars for a dying contract with
   volume 0 and price frozen at the last trade: UNFIUSDT printed volume 0
   with the perp pinned at 1.730 for **seven days** while spot kept
   trading. A frozen perp against a live spot is naked directional
   exposure, not a hedge. Fixed by treating a no-trade bar as having no
   valid mark. The liquidity gate was also hardened — a rolling MEAN let
   one $400M liquidation spike keep UNFI "eligible" through all seven
   dead days, so the current bar must now have traded.

   c. *No per-symbol concentration cap.* Weights were `sign / len(held)`, so
   a SHRINKING book silently became a CONCENTRATED one: it collapsed to 3
   names and put 33% of capital into UNFIUSDT mid-delisting. Fixed with
   `max_weight_per_symbol` (default 0.15) — a thin book now runs
   under-invested rather than concentrated.

   **The two wrong conclusions.** Earlier versions of this document reported
   survivorship inflation of +0.16/+0.04 on a −11.4% max drawdown, and stated
   that *"UNFIUSDT's real spot delisting caused a genuine −10.6% single-symbol
   drawdown — a real, previously-hidden risk event."* Both were artifacts of
   (b) and (c), not economics. The loss was largely fabricated by marking a
   live spot leg against a frozen perp price, then amplified by the missing
   concentration cap. Corrected figures are in the table above.

   **The assumption that now carries the result.** NaN-ing dead bars means
   the backtest exits a dying position at its last good price for free, which
   is optimistic — a real desk faces slippage or is simply stuck. Charging an
   explicit penalty on every position that dies while held:

   | Exit penalty | OOS Sharpe |
   | ------------ | ---------- |
   | 0% (current) | 5.58       |
   | 10%          | 4.74       |
   | 25%          | 2.87       |
   | 50%          | 1.73       |

   **Defensible range: OOS Sharpe ~1.7–5.6, central estimate ~3–5.** Modelling
   this exit properly is the highest-value open item in the project.

   Two structural risks remain unmeasured and are likely larger still:
   execution feasibility during a de-peg (see #8 — perp liquidity went to
   literally $0 on SRMUSDT and LUNAUSDT) and exchange counterparty risk (#9).
2. **One venue.** Binance only. Cross-venue funding dispersion is part of the
   real opportunity set and a real risk (a Binance-specific dislocation looks
   like alpha here).
3. **8h bars, not tick.** Entry/exit is assumed at the settlement bar close.
   Real execution happens inside the bar at unknown prices.
4. **Realized funding, not predicted-at-decision.** We predict f_{t+1} from an
   EWMA of past funding. Binance publishes a *live predicted* funding rate
   intraday which a production system would use and which is strictly more
   informative. Our version is therefore conservative on signal quality.

## Execution & cost

5. **Taker fees assumed on both legs, every rebalance.** Realistic for retail
   (4bps futures + 7.5bps spot). A real desk posts maker orders and pays far
   less — the cost-sensitivity table spans this. But maker orders don't always
   fill, and non-fills on one leg leave you *directionally exposed*, which is
   not modeled at all.
6. **No slippage vs. size.** Cost is linear in turnover with no impact term.
   The capacity module (sqrt-law) should be applied here using perp order-book
   depth and spot ADV — this is the natural next step and the most
   Pexabit-relevant one, since carry capacity binds hard.
7. **Perfect delta neutrality assumed.** Spot and perp notionals are matched
   exactly and continuously. In reality the hedge drifts intra-interval, and
   rebalancing it costs money.

## Structural / tail risks — the ones that actually blow up carry

8. **Liquidation cascade risk.** The carry trade is short perp. In a violent
   melt-up, perp shorts face margin calls exactly when funding spikes against
   them. Backtest P&L is smooth; the real position can be force-closed at the
   worst moment. **Nothing in this backtest models margin or liquidation.**
9. **Exchange counterparty risk.** Funds sit on a centralized exchange. FTX
   (Nov 2022) is inside this backtest's date range and would have returned
   -100% on balances there, with a Sharpe that looked excellent right up to
   the failure. This is the dominant unmodeled risk and it is not
   diversifiable by trading more symbols on the same venue.
10. **Withdrawal freezes / stablecoin depegs.** USDT depeg (or a freeze)
    breaks both the collateral and the quote leg simultaneously.
11. **Funding regime shift.** The premium exists because leveraged longs
    dominate. In a prolonged bear market funding goes persistently negative;
    the strategy can trade the other side, but the *magnitude* of the premium
    has compressed as the trade got crowded (2019 >> 2024).

## Statistical

12. **~3.5 years, one venue, one regime cycle.** Short sample by equity
    standards. The DSR correction is applied, but it cannot fix "crypto has
    only had a handful of independent macro regimes."
13. **Effective breadth — measured and better than assumed.** We expected
    alt-coin carry positions to share one crowded-long factor, collapsing
    effective breadth. Measured average pairwise correlation of per-symbol
    carry P&L is **0.010**, giving **effective breadth 28.2** against 39
    nominal positions. Funding is driven by symbol-specific positioning, not
    a single common factor, so the sqrt(N) diversification benefit is largely
    real here. This was an assumption we were wrong about — recorded rather
    than quietly deleted.
14. **Single-name concentration was severe before it was controlled.**
    Un-gated, **TRBUSDT contributed 36% of gross P&L**, almost all of it from
    the Sept-Oct 2023 squeeze when 8h funding hit -250 to -300bp. That is not
    tradeable carry; it is a manipulated market with an unborrowable spot leg.
    The `max_predicted_bps=50` gate exists for this. Any funding-carry
    backtest without such a gate should be assumed to be reporting a squeeze,
    not a strategy.
15. **Extreme non-normality.** OOS skew +2.9, kurtosis ~290. Sharpe is a poor
    summary for a distribution like this, which is precisely why the Deflated
    Sharpe (which penalizes skew/kurtosis explicitly) lands far below the raw
    number — see the results table in the README.
