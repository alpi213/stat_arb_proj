**Disclaimer:** This repository is a personal research project conducted independently by the author. It was created using publicly available market data and executed entirely on personal hardware and personal time. It does not contain any proprietary information, data or technology belonging to any employer, past or persent, nor does it reflect the views or strategies of any financial institution. 

# Statistical Arbitrage: Cointegration Pairs with Dynamic Hedging & Capacity Analysis

A mid-frequency equity pairs-trading research framework built around one question most
backtests dodge: **does the edge survive costs, and how much capital can it absorb?**

- **Signal**: Engle-Granger cointegration within sectors, spread z-score entry/exit with
  hard and time stops.
- **Hedging**: static OLS baseline vs. **Kalman-filter dynamic hedge ratio** (before/after
  comparison is a first-class output).
- **Honesty layer**: execution lag enforced by construction, commissions + spread +
  **short borrow fees**, point-in-time universe interface, walk-forward re-selection,
  **Deflated Sharpe Ratio**, regime breakdowns.
- **Capacity analysis**: square-root-law market impact re-pricing of the trade stream
  across an AUM grid → *net Sharpe vs. AUM curve* and an explicit capacity estimate.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on Unix)
pip install -e ".[dev]"
pytest                            # verify the engine before trusting any result

python scripts/01_download_data.py    # download + clean + cache prices (yfinance)
python scripts/02_select_pairs.py     # cointegration scan on the formation window
python scripts/03_backtest.py --compare   # static vs kalman, OOS equity + metrics
python scripts/04_capacity.py         # net Sharpe vs AUM curve
python scripts/05_walkforward.py      # rolling re-selection — the headline result
python scripts/06_validate.py         # Deflated Sharpe + regime breakdown
```

All parameters live in [configs/base.yaml](configs/base.yaml). Every experiment =
config file + git commit hash. Outputs land in `results/` (figures in
`results/figures/`).

## Repository layout

```
configs/            base.yaml (all parameters), universe.csv (point-in-time membership)
src/statarb/
  data/             download adapters, parquet store, PIT universe, quality policy
  pairs/            Engle-Granger, half-life, Johansen, leak-safe selection
  hedge/            HedgeEstimator interface: StaticHedge, KalmanHedge
  signals/          rolling z-score + entry/exit/stop state machine
  backtest/         engine (lagged execution), cost model, metrics
  capacity/         sqrt-law impact, capacity curve
  validation/       walk-forward windows, Deflated Sharpe, regime analysis
  crypto/           Binance adapters, funding-carry panel + signal
scripts/            numbered pipeline: 01-06 equity, 07-08 sweep/null, 10-12 crypto
tests/              incl. adversarial no-lookahead tests on the engine
docs/               ARCHITECTURE, DATA, LIMITATIONS, PRODUCTION, CRYPTO_LIMITATIONS
```

## Methodology in one page

1. **Universe** (~80 liquid US names + sector ETFs, 2015-2025 daily): membership file
   carries `entry_date`/`exit_date` per name so delistings force liquidation — the
   interface is point-in-time even though the shipped list is static
   (see [docs/LIMITATIONS.md](docs/LIMITATIONS.md)).
2. **Pair formation**: within-sector only (economic prior + fewer hypotheses). EG test
   both directions on log prices, keep p < 0.05, half-life in [2, 60] days, positive
   hedge ratio, ≤2 pairs per name. The number of pairs *tested* is recorded — it feeds
   the DSR.
3. **Hedge ratio**: Kalman filter with random-walk state `[alpha, beta]`; the tradeable
   spread is the *prior* innovation (the posterior has already seen today's price).
   `delta` controls drift speed; sensitivity reported, not hidden.
4. **Trading rule**: z-score of spread over rolling 60d. Enter at |z| ≥ 2, exit at
   |z| ≤ 0.5, hard stop at |z| ≥ 3.5 **with lockout** (a broken pair can't be
   immediately re-entered), time stop at 40 days.
5. **Execution & costs**: signal at close *t* → position from close *t+1* (lag ≥ 1
   enforced by config validation). 1 bp commission + 2.5 bp half-spread per side,
   50 bp/yr borrow on the short leg. Hedge frozen at entry (no daily re-hedge churn).
6. **Capacity**: same trade stream re-priced at each AUM with impact
   `k·σ·√(Q/ADV)`; reports AUM where Sharpe halves and where it dies, plus
   sensitivity in `k`.
7. **Validation**: walk-forward (2y formation → 6m trading, rolled), DSR using the
   *true* count of hypotheses tested, regime slices (COVID crash, 2022 rate shock,
   2023 banking stress).

## Results — the edge does not survive costs

79 US names + sector ETFs, 2015-2025. **Headline is the walk-forward number**, where
pairs are re-selected on each trailing formation window and never chosen with knowledge
of the period they trade.

| Metric                                    | Static hedge                              | Kalman hedge     |
| ----------------------------------------- | ----------------------------------------- | ---------------- |
| Sharpe, single-split OOS (2017-2025, net) | −0.03                                    | −0.21           |
| **Sharpe, walk-forward OOS (net)**  | —                                        | **−0.45** |
| CAGR (walk-forward)                       | —                                        | −1.24%          |
| Max drawdown (walk-forward)               | —                                        | −13.1%          |
| Hit rate / avg holding                    | 62% / 19d                                 | 59% / 3.5d       |
| Capacity                                  | not meaningful — base Sharpe is negative |                  |

Gross signal *exists* (62-70% hit rates) but average loss exceeds average win and
round-trip costs of 5-7bps consume the remainder. The Kalman variant churns badly at
3.5-day holds because a well-specified filter drives its own innovations toward white
noise — `spread_mode="residual"` was added to address this and is reported, not
silently swapped in.

### The number that matters most

Per-window walk-forward Sharpes swing from **+2.40 to −2.38** across 6-month segments.
Cherry-picking the single best 2-year window in 8.5 years of OOS data yields **+1.20**;
the worst yields −2.16; the median is −0.43. A 2-year Sharpe of 3 is simply not in this
data, honestly measured.

### Null experiment — calibrating the search itself

`scripts/08_null_experiment.py` runs the **entire pipeline** on synthetic independent
random walks with zero cointegration by construction:

|                                                            | Value                    |
| ---------------------------------------------------------- | ------------------------ |
| Null Sharpe (mean / range)                                 | −0.22 / [−0.51, +0.06] |
| Pair hypotheses tested                                     | ~26,900                  |
| **Theoretical expected-max Sharpe under pure noise** | **1.28**           |

That 1.28 is the bar. Any result below it is indistinguishable from what search-through-
noise manufactures on its own. Reporting a strategy Sharpe without this control is an
uncalibrated claim — the honest conclusion here is a **negative result**, and it is
worth more than a tuned positive one.

Figures: `results/figures/equity_walkforward.png`, `pair_*_kalman.png`.

## What is deliberately NOT modeled

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the full list with rationale:
intraday execution, borrow availability/recall, PIT index membership (interface built,
data source not wired), dividends on short positions, margin financing rates.
Naming what's missing is part of the result.

## Testing

`pytest` runs unit tests for every statistical component against synthetic data with
known ground truth, plus **adversarial engine tests**: perturbing a future price must
not change past positions, and the engine must not manufacture Sharpe from independent
random walks. If you touch the engine, run these first.

---

# Part II: Crypto funding carry

The equity strategy's honest verdict was "no edge survives costs." Rather than
tune it until a number appeared, the framework was pointed at a market where a
*structural* premium exists: perpetual-futures funding.

Perps have no expiry, so exchanges tether them to spot via a funding payment
between longs and shorts every 8h. Crowded leveraged longs pay shorts. A
delta-neutral book — long spot, short perp — collects that payment with no
directional exposure. The edge is risk transfer, not a statistical artifact,
which is why it survives out-of-sample where cointegration did not.

## Results (130 Binance USD-M perps, point-in-time, 2022-01 to 2025-06, 8h)

Universe is point-in-time: every perp active at 2022-01-01, **including the 15
that have since been delisted**. See the survivorship section below.

| Metric                                    | Full sample    | Out-of-sample                  |
| ----------------------------------------- | -------------- | ------------------------------ |
| Sharpe (net of costs)                     | **5.79** | **5.60**                 |
| Sharpe (gross)                            | 7.70           | 7.88                           |
| CAGR                                      | 22.2%          | —                             |
| Max drawdown                              | −2.5%         | −1.9%                         |
| Annual vol                                | 3.5%           | 2.8%                           |
| Annual turnover                           | 45×           | —                             |
| **Deflated Sharpe (n_trials=1440)** | —             | **0.9999 — passes 95%** |

**Do not read 5.60 as the forward estimate.** One modelling assumption still
carries the number: when a contract's book dies while held, the backtest exits
at the last good price for free. Charging an explicit penalty instead:

| Exit penalty on a dying position | OOS Sharpe |
| -------------------------------- | ---------- |
| 0% (current assumption)          | 5.60       |
| 10%                              | 4.74       |
| 25%                              | 2.87       |
| 50%                              | 1.73       |

**The defensible range is OOS Sharpe ~1.7–5.6; a central estimate is ~3–5.**
Even punishing every dying position by 25% it clears 2.8. Quantifying this
assumption matters more than the headline figure.

**Reproducibility.** These figures are produced bit-identically by the commands in
[Run it](#run-it) below. The pipeline was non-deterministic until a set-iteration bug
in the position-slot fill was fixed (Python randomizes string hashing per process, so
back-to-back runs differed by ~0.01 Sharpe and could drop a better-ranked name for a
worse one); `test_signal_is_deterministic_across_processes` guards it.

### Cost sensitivity — viability is set by fee tier

| Scenario                | Sharpe (full) | Sharpe (OOS) |
| ----------------------- | ------------- | ------------ |
| Zero cost (upper bound) | 7.70          | 7.88         |
| VIP / maker-ish (0.5×) | 6.79          | 6.80         |
| Retail taker (base)     | 5.79          | 5.60         |
| Stressed / wide (2×)   | 3.74          | 3.19         |

Viability is set by **execution quality and fee tier**, not signal cleverness.
The spot leg dominates: ~7.5–10bps at Binance VIP0 with no maker discount, so
a round trip on both legs costs ~27bps against ~3bps/day of funding. Turnover
control (sticky book, hysteresis, weekly rebalance) is what makes it clear —
naive 8h re-ranking turns over ~480×/yr and loses outright.

### Stress regimes

Re-run on the point-in-time universe with all data fixes applied (the earlier
version of this table used the pre-fix 40-symbol survivor universe):

| Regime                            | Return  | Sharpe | Max DD  |
| --------------------------------- | ------- | ------ | ------- |
| LUNA collapse (May 2022)          | +2.98%  | 20.6   | −0.16% |
| 3AC / Celsius (Jun 2022)          | +2.21%  | 14.3   | −0.13% |
| FTX collapse (Nov 2022)           | +1.81%  | 10.6   | −0.29% |
| 2023 rally (Oct 2023 – Mar 2024) | +13.95% | 10.7   | −0.72% |
| Aug-2024 unwind                   | +0.09%  | 2.3    | −0.14% |
| 2025 drawdown (Feb–Apr)          | +2.87%  | 2.8    | −1.89% |

Positive in every stress window. The eye-watering in-regime Sharpes are an
artifact of annualizing a two-week window — read the return and drawdown
columns, not the Sharpe column.

Carry *gained* through FTX because panic drove funding deeply negative and the
short-carry side paid. Note the irony carefully: **capital held on FTX would
have gone to zero regardless**, and no backtest metric would have shown it.

## What the diligence found (and why it matters more than the Sharpe)

- **TRBUSDT alone was 36% of gross P&L** un-gated — the Sept-Oct 2023 squeeze,
  where 8h funding hit −300bp. Unborrowable spot, manipulated perp: not
  tradeable carry. The `max_predicted_bps` gate exists for this.
- **Effective breadth 28.2** of 39 nominal (avg pairwise corr 0.010) — we
  *assumed* alt carry shared one crowded-long factor and were wrong; measured
  and corrected in [docs/CRYPTO_LIMITATIONS.md](docs/CRYPTO_LIMITATIONS.md).
- **99.6% of P&L is the funding component**, 0.4% basis — it is a pure carry
  trade, as designed, not a disguised directional bet.
- Removing the top 1/5/10/20 return periods barely moves Sharpe (4.6→4.3), so
  the edge is broad across time rather than a handful of lucky days.

Three look-ahead / data bugs were caught building this, each of which inflated
results before being fixed: klines indexed by open-time while using close
prices; Binance's spot endpoint capping pages at 1000 (vs 1500) and silently
truncating history; and millisecond jitter in funding timestamps
(`00:00:00.006`) colliding under `ceil()` bucketing and blowing 634 holes (16%)
in the panel. All three now have regression tests.

## Walk-forward: was the config hand-tuned?

The single-split result used hyperparameters chosen by hand after seeing the whole
sample. `scripts/13_carry_walkforward.py` removes that channel: on each 1-year
formation window it picks the best config by *in-window* Sharpe from a 12-point grid,
trades it for the next 3 months out of sample, and stitches the segments.

| Metric                    | Walk-forward OOS (2023-01 → 2025-06)   |
| ------------------------- | --------------------------------------- |
| Sharpe                    | **7.22**                          |
| CAGR                      | 22.1%                                   |
| Max drawdown              | −1.0%                                  |
| Annual vol                | 2.8%                                    |
| Modal config chosen in    | 30% of windows                          |
| **Deflated Sharpe** | **1.00 — passes** (n_trials=120) |

### Config stability — a claim this project got wrong, then corrected

An earlier version of this section reported that **"tuning adds only +0.13 Sharpe"**,
concluding hyperparameters were nearly irrelevant and the premium did all the work.
**That measurement was taken before the delisting-price and zero-volume fixes, and it
does not survive them.** Re-tested on window 7 (Oct-Dec 2024), varying only `lookback`:

| lookback | Sharpe | Ann. vol | Max DD | Held UNFIUSDT?       |
| -------- | ------ | -------- | ------ | -------------------- |
| 9        | 3.63   | 3.6%     | −1.0% | no                   |
| 21       | 0.79   | 10.7%    | −4.4% | **yes, 14.3%** |
| 63       | 4.92   | 3.2%     | −1.0% | no                   |

One parameter swings that window's Sharpe from 0.8 to 4.9 — and the entire difference
is *whether the book happened to hold one dying contract*. The walk-forward picked
`lookback=63` from formation-window data, before the event; it could not have known.
**That is luck, not skill**, and the honest reading is that the walk-forward number is
flattered by having dodged a tail event the fixed-config run walked into.

This is why the single-split figure (5.60 OOS) is quoted as the headline rather than
the walk-forward's 7.22: the gap between them is a tail-risk lottery, not added value
from adaptation.

## Survivorship — measured, not assumed

Built a true point-in-time universe — 833 USDT perps ever listed on Binance, 180 since
delisted, verified against known collapses (LUNAUSDT active 2021-01→2022-05, matching
its depeg) — via `scripts/14_build_pit_universe.py`, sourced from Binance's own archive
(`data.binance.vision`), not Wayback-scraped snapshots. Then ran the strategy on 130
symbols active at 2022-01-01, 15 of them since delisted, against the same 115 survivors:

|                                    | Sharpe (full) | Sharpe (OOS) | CAGR  | Max DD |
| ---------------------------------- | ------------- | ------------ | ----- | ------ |
| Survivor-only (115)                | 5.81          | 5.68         | 22.9% | −2.6% |
| Point-in-time (130, incl. 15 dead) | 5.79          | 5.60         | 22.3% | −2.5% |

**Survivorship inflates Sharpe by only +0.02 full / +0.10 OOS** — far smaller than
assumed going in. The reason is structural: in a delta-neutral book, when a coin
collapses *both legs collapse together*, so the hedge protects you. LUNAUSDT
contributed **+0.025% of AUM** across its entire May 2022 death.

### Getting to that number took three bug fixes, and two wrong conclusions on the way

Measuring survivorship honestly meant backtesting contracts through their own deaths —
which turned out to be where the data is least trustworthy:

1. **Stale `ffill` at delisting.** `ffill(limit=1)` froze one leg's price for an extra
   bar when perp and spot stopped printing at different times, fabricating a basis
   divergence. Fixed by truncating both legs at their shared last real print.
2. **Zero-volume bars treated as prices.** Binance keeps *publishing* bars for a dying
   contract — volume 0, price frozen at the last trade. UNFIUSDT printed volume 0 with
   the perp pinned at 1.730 for **seven days** while spot kept trading. A frozen perp
   against a live spot is naked directional exposure, not a hedge. Fixed by treating a
   no-trade bar as having no valid mark.
3. **No per-symbol concentration cap.** Weights were `sign / len(held)`, so a
   *shrinking* book silently became a *concentrated* one — it collapsed to 3 names and
   put 33% of capital into UNFIUSDT mid-delisting. Fixed with `max_weight_per_symbol`;
   a thin book now runs under-invested rather than concentrated.

**The wrong conclusions, recorded rather than deleted:** an earlier version of this
README reported survivorship inflation of +0.16/+0.04 on a −11.4% max drawdown, and
attributed that drawdown to *"UNFIUSDT's real spot delisting causing a genuine −10.6%
single-symbol loss."* Both were artifacts of bugs 2 and 3. The loss was largely
fabricated by marking a live spot leg against a frozen perp price, and amplified by the
missing concentration cap. Corrected numbers are in the table above.

Full detail and mechanism in
[docs/CRYPTO_LIMITATIONS.md](docs/CRYPTO_LIMITATIONS.md#data).

## Capacity — the question the framework was built for

Capacity analysis only says something about a strategy that has an edge, so this is the
first time in the project it produces a meaningful curve. Same square-root impact model
as the equity path (`impact = k·σ·√(Q/ADV)`), re-pricing the *same* trade stream at
increasing AUM.

| AUM   | Net Sharpe | Annual impact drag |
| ----- | ---------- | ------------------ |
| $1M   | 5.44       | 1.0%               |
| $10M  | 4.67       | 3.3%               |
| $50M  | 3.31       | 7.4%               |
| $100M | 2.37       | 10.4%              |
| $250M | 0.78       | 16.5%              |
| $500M | −0.58     | 23.3%              |

**Sharpe halves at ~$77M; reaches zero at ~$373M** (k=0.10, perp ADV basis).

Bounds, because a single point estimate here would be false precision:

| Assumption                               | Sharpe → ½ | Sharpe → 0 |
| ---------------------------------------- | ------------ | ----------- |
| Base (k=0.10, perp ADV)                  | $77M         | $373M       |
| Conservative (k=0.10, thin-spot haircut) | $30M         | $130M       |

Capacity scales exactly as **1/k²** under the sqrt law, and `k` is the least-known input
(set a priori, not calibrated from own fills) — so treat these as order-of-magnitude.
The honest read: this is a **$30-80M strategy**, not a $500M one. Note also that these
figures inherit the headline Sharpe's costless-exit assumption; on the ~3-5 central
estimate the capacity numbers scale down accordingly.

The structural insight worth keeping: **turnover, not Sharpe, sets capacity.** Carry
survives to $77M *because* the sticky book turns over only 45×/yr. The same gross edge
harvested with naive 8h re-ranking (480×/yr) would both lose money outright at retail
fees and exhaust its capacity an order of magnitude sooner.

## Run it

```bash
python scripts/14_build_pit_universe.py
python scripts/10_download_crypto.py --symbols-file configs/universe_perp_pit.csv     --n-symbols 141 --start 2022-01-01 --end 2025-06-30
python scripts/11_carry_backtest.py       # headline table above
python scripts/13_carry_walkforward.py    # walk-forward
python scripts/12_crypto_capacity.py      # capacity curve
```

Step 1 reconstructs the point-in-time universe from Binance's archive; step 2
downloads it *including since-delisted contracts*. Running step 2 without
`--symbols-file` gives a survivor-only universe and will NOT reproduce the
numbers above — that difference is the survivorship measurement.

Limitations — read before quoting any number above:
[docs/CRYPTO_LIMITATIONS.md](docs/CRYPTO_LIMITATIONS.md). The largest
unmodeled risks are **survivorship** (only currently-listed perps; delisted
ones died precisely when carry blew up), **exchange counterparty risk**, and
**no margin/liquidation modeling** — the backtest's smooth equity curve cannot
be force-closed, but a real position can.
