**Disclaimer:** This repository is a personal research project conducted independently by the author.
It was created using publicly available market data and executed entirely on personal hardware
and personal time. It does not contain any proprietary information, data or technology belonging 
to any employer, past or present, nor does it reflect the views or strategies of any financial institution.

# Does the edge survive costs, and how much capital can it take?

Two strategies run through the same set of checks: execution lag, realistic
costs, walk-forward selection, a multiple-testing correction, a null
experiment, and a capacity curve.

- **Equity pairs** (cointegration, static vs Kalman hedge): no edge survives costs.
  The result is negative and kept that way.
- 
- **Crypto funding carry** (long spot, short perp): a real, structural edge.
  Defensible out-of-sample Sharpe is 1.7 to 5.6 depending on one assumption that is
  priced explicitly, and capacity is on the order of $30M to $80M.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest                            # 33 tests, including adversarial no-lookahead tests
```

Equity pipeline (all parameters in [configs/base.yaml](configs/base.yaml), outputs in `results/`):

```bash
python scripts/01_download_data.py    # yfinance -> cleaned parquet
python scripts/02_select_pairs.py     # cointegration scan on the formation window
python scripts/03_backtest.py         # static and Kalman hedge, out of sample
python scripts/04_capacity.py         # net Sharpe vs AUM
python scripts/05_walkforward.py      # rolling re-selection: the headline number
python scripts/06_validate.py         # Deflated Sharpe + regime slices
python scripts/07_null_experiment.py  # same pipeline on random walks
```

Crypto pipeline (outputs in `results_crypto/`):

```bash
python scripts/10_build_pit_universe.py                                   # every perp ever listed
python scripts/11_download_crypto.py --n-symbols 141 --start 2022-01-01 --end 2025-06-30
python scripts/12_carry_backtest.py
python scripts/13_carry_walkforward.py
python scripts/14_crypto_capacity.py
```

Step 11 reads the point-in-time universe from step 10 and includes contracts that
have since been delisted. Pass `--survivors-only` to drop them; the difference between
the two runs is the survivorship measurement below.

## Layout

```
configs/        base.yaml (every equity parameter), universe.csv, universe_perp_pit.csv
src/statarb/
  data.py       yfinance download, cleaning policy, parquet store, universe file
  pairs.py      Engle-Granger, half-life, within-sector pair selection
  hedge.py      StaticHedge (refit OLS) and KalmanHedge
  signals.py    rolling z-score and the entry/exit/stop rules
  backtest/     pair engine with lag and costs; weights engine for carry; metrics
  capacity.py   square-root impact, capacity curve
  validation.py walk-forward windows, Deflated Sharpe, regime breakdown
  pipeline.py   the steps the equity scripts share
  crypto/       Binance adapters, carry panel, carry signal
scripts/        01-07 equity, 10-14 crypto; each parses args and writes files
tests/
docs/           DESIGN.md, LIMITATIONS.md, PRODUCTION.md
```

## Part I: equity pairs

### Method

1. **Universe.** 79 liquid US names plus sector ETFs, daily bars, 2015 to 2025. The
   list is static, so it leans toward survivors. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
2. **Pairs.** Within sector only. Engle-Granger on log prices in both directions,
   keep p < 0.05, half-life between 2 and 60 days, positive hedge ratio, at most two
   pairs per name. The count of pairs tested is saved; it feeds the Deflated Sharpe.
3. **Hedge ratio.** Kalman filter with a random-walk state [alpha, beta]. The traded
   spread is the prior innovation (the posterior has already seen today's price).
   A residual mode is also available and reported.
4. **Rules.** Z-score over a rolling 60 days. Enter at |z| >= 2, exit at |z| <= 0.5,
   stop at |z| >= 3.5 with a lockout, time stop at 40 days.
5. **Execution and costs.** Signal at close t, position from close t+1. The config
   validator refuses a lag below one bar. 1 bp commission and 2.5 bp half-spread per
   side, 50 bp a year borrow on the short leg. Beta is frozen at entry.
6. **Capacity.** The same trade stream re-priced at each AUM with impact
   k * sigma * sqrt(Q / ADV). Reports where Sharpe halves and where it hits zero.
7. **Validation.** Walk-forward (two years formation, six months trading, rolled).
   Deflated Sharpe using the true number of hypotheses. Regime slices for the COVID
   crash, the 2022 rate shock and the 2023 banking stress.

### Result: the edge does not survive costs

The headline is the walk-forward number, where pairs are re-selected on each
trailing window and never chosen with knowledge of the period they trade.

| Metric                                   | Static hedge | Kalman hedge    |
| ---------------------------------------- | ------------ | --------------- |
| Sharpe, single split OOS (net)           | -0.03        | -0.21           |
| **Sharpe, walk-forward OOS (net)** |              | **-0.45** |
| CAGR, walk-forward                       |              | -1.24%          |
| Max drawdown, walk-forward               |              | -13.1%          |
| Hit rate / average holding               | 62% / 19d    | 59% / 3.5d      |

A gross signal exists (62 to 70% hit rates), but the average loss is larger than the
average win and 5 to 7 bp round trips take the rest. The Kalman variant churns: a
well-specified filter drives its own innovations toward white noise, so the z-score
flips sign quickly and trades last 3.5 days.

Per-window walk-forward Sharpes range from +2.40 to -2.38. The best two-year window
in 8.5 years of out-of-sample data gives +1.20, the worst -2.16, the median -0.43.

### The null experiment

`scripts/07_null_experiment.py` runs the entire pipeline on independent random walks
with no cointegration by construction.

|                                       | Value                  |
| ------------------------------------- | ---------------------- |
| Null Sharpe, mean / range             | -0.29 / [-0.62, +0.08] |
| Pair hypotheses tested                | ~50,000                |
| Expected best Sharpe under pure noise | **1.32**         |

Anything below 1.32 is what searching through noise produces on its own. The real
data never clears it, so the honest conclusion is a negative result.

## Part II: crypto funding carry

Perps have no expiry, so exchanges tie them to spot with a funding payment between
longs and shorts every 8 hours. When leveraged longs crowd in, longs pay shorts. Long
spot plus short perp collects that payment with no directional exposure. The edge is
a payment for risk transfer, not a pattern found by searching, which is why it holds
up out of sample where cointegration did not.

### Result (130 Binance USD-M perps, point in time, Jan 2022 to Jun 2025, 8h bars)

| Metric                          | Full sample    | Out of sample  |
| ------------------------------- | -------------- | -------------- |
| Sharpe, net of costs            | **5.79** | **5.60** |
| Sharpe, gross                   | 7.70           | 7.88           |
| CAGR                            | 22.2%          |                |
| Max drawdown                    | -2.5%          | -1.9%          |
| Annual vol                      | 3.5%           | 2.8%           |
| Annual turnover                 | 45x            |                |
| Deflated Sharpe (n_trials=1440) |                | 0.9999, passes |

**Do not read 5.60 as a forward estimate.** One assumption carries the number: when a
contract's book dies while held, the backtest exits at the last good price for free.
Charging a penalty instead:

| Exit penalty on a dying position | OOS Sharpe |
| -------------------------------- | ---------- |
| 0% (current)                     | 5.60       |
| 10%                              | 4.74       |
| 25%                              | 2.87       |
| 50%                              | 1.73       |

The defensible range is **1.7 to 5.6**, central estimate 3 to 5.

### Costs set viability

| Scenario                | Sharpe, full | Sharpe, OOS |
| ----------------------- | ------------ | ----------- |
| Zero cost (upper bound) | 7.70         | 7.88        |
| VIP / maker-ish (0.5x)  | 6.79         | 6.80        |
| Retail taker (base)     | 5.79         | 5.60        |
| Stressed / wide (2x)    | 3.74         | 3.19        |

A round trip on both legs costs about 27 bp against about 3 bp a day of funding.
Turnover control (a sticky book, hysteresis, weekly rebalance) is what makes it work.
Re-ranking every 8 hours turns over about 480x a year and loses money outright.

### Stress regimes

| Regime                            | Return  | Sharpe | Max DD |
| --------------------------------- | ------- | ------ | ------ |
| LUNA collapse (May 2022)          | +2.98%  | 20.6   | -0.16% |
| 3AC / Celsius (Jun 2022)          | +2.21%  | 14.3   | -0.13% |
| FTX collapse (Nov 2022)           | +1.81%  | 10.6   | -0.29% |
| 2023 rally (Oct 2023 to Mar 2024) | +13.95% | 10.7   | -0.72% |
| Aug 2024 unwind                   | +0.09%  | 2.3    | -0.14% |
| 2025 drawdown (Feb to Apr)        | +2.87%  | 2.8    | -1.89% |

Positive in every window. Read the return and drawdown columns, not the Sharpe
column, which annualizes a two-week window. Carry gained through FTX because panic
drove funding negative and the short-carry side paid. Capital held on FTX would still
have gone to zero, and no backtest metric would show it.

### What the diligence found

- **TRBUSDT alone was 36% of gross P&L** before the `max_predicted_bps` gate. Funding
  of -300 bp per 8h is a short squeeze with an unborrowable spot leg, not carry.
- **Effective breadth is 28.2** of 39 nominal positions (average pairwise correlation
  0.010). The expected single crowded-long factor is not there.
- **99.6% of P&L is funding**, 0.4% basis. It is a pure carry trade, as designed.
- Removing the top 1, 5, 10 or 20 return periods barely moves Sharpe (4.6 to 4.3).

Six data bugs were found and fixed, each with a regression test: klines indexed by
open time while using close prices; the spot endpoint silently capping pages at 1000
and truncating history; millisecond jitter in funding timestamps colliding under
ceil() and leaving 634 holes (16% of the panel); forward-fill freezing one leg at a
delisting; zero-volume bars treated as prices; and a set iteration that made results
depend on Python's per-process hash seed.

### Walk-forward over hyperparameters

The single-split config was chosen by hand after seeing the whole sample.
`scripts/13_carry_walkforward.py` picks the best config from a 12-point grid on each
one-year formation window and trades it for the next three months.

| Metric                 | Walk-forward OOS (Jan 2023 to Jun 2025) |
| ---------------------- | --------------------------------------- |
| Sharpe                 | **7.22**                          |
| CAGR                   | 22.1%                                   |
| Max drawdown           | -1.0%                                   |
| Modal config chosen in | 30% of windows                          |
| Deflated Sharpe        | 1.00, passes (n_trials=120)             |

This is not the headline. On window 7 (Oct to Dec 2024), varying only `lookback`:

| lookback | Sharpe | Ann. vol | Max DD | Held UNFIUSDT? |
| -------- | ------ | -------- | ------ | -------------- |
| 9        | 3.63   | 3.6%     | -1.0%  | no             |
| 21       | 0.79   | 10.7%    | -4.4%  | yes, 14.3%     |
| 63       | 4.92   | 3.2%     | -1.0%  | no             |

One parameter swings the window from 0.8 to 4.9, and the whole difference is whether
the book happened to hold one dying contract. The walk-forward picked 63 from formation
data and could not have known. That is luck, so the lower single-split number is quoted.

### Survivorship, measured

`scripts/10_build_pit_universe.py` rebuilds the universe from Binance's own archive:
901 USDT perps ever listed, 244 since delisted, with active windows checked against
known collapses (LUNAUSDT active Jan 2021 to May 2022). The strategy then ran on the 130
symbols active at Jan 2022, 15 of them since delisted, and on the 115 survivors alone.

|                                    | Sharpe, full | Sharpe, OOS | CAGR  | Max DD |
| ---------------------------------- | ------------ | ----------- | ----- | ------ |
| Survivor-only (115)                | 5.81         | 5.68        | 22.9% | -2.6%  |
| Point-in-time (130, incl. 15 dead) | 5.79         | 5.60        | 22.3% | -2.5%  |

Survivorship inflates Sharpe by +0.02 full and +0.10 OOS. In a delta-neutral book both
legs collapse together, so the hedge protects you. LUNAUSDT's entire death contributed
+0.025% of AUM.

An earlier version of this README reported inflation of +0.16 / +0.04 on an -11.4% max
drawdown and attributed the drawdown to a real UNFIUSDT delisting loss. Both numbers
were artifacts of two of the data bugs above (a frozen perp marked against a live spot,
amplified by a missing concentration cap). The corrected numbers are in the table.

### Capacity

Same square-root impact model as the equity path, applied to the same trade stream.

| AUM   | Net Sharpe | Annual impact drag |
| ----- | ---------- | ------------------ |
| $1M   | 5.44       | 1.0%               |
| $10M  | 4.67       | 3.3%               |
| $50M  | 3.31       | 7.4%               |
| $100M | 2.37       | 10.4%              |
| $250M | 0.78       | 16.5%              |
| $500M | -0.58      | 23.3%              |

| Assumption                               | Sharpe halves | Sharpe hits zero |
| ---------------------------------------- | ------------- | ---------------- |
| Base (k=0.10, perp ADV)                  | $77M          | $373M            |
| Conservative (k=0.10, thin-spot haircut) | $30M          | $130M            |

Capacity scales as 1/k^2 and k is set a priori, not calibrated from fills, so these are
order-of-magnitude figures. The honest read is a $30M to $80M strategy. Carry gets that
far because the sticky book turns over only 45x a year. Turnover, not Sharpe, sets
capacity.

## Before quoting any number

Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md). The largest unmodeled risks for carry
are exchange counterparty risk (FTX is inside the sample) and margin or liquidation on
the short perp leg. For equities, the static universe leans toward survivors, which
biases an already negative result upward.
