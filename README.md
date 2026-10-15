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
- **Crypto funding carry** (long spot, short perp): a structural edge with a fragile
  headline. Out-of-sample Sharpe is 5.60 if the nine exits from dying contracts are
  free, 2.92 if each costs 10% of the position, and 0.52 at 25%. Moving the weekly
  rebalance to a different settlement alone moves it between 2.7 and 6.7. Sharpe
  halves at $30M to $77M of AUM.

Every number below is written by a script in `scripts/` to a file in `results/` or
`results_crypto/`; the file is named under each table.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest                            # 39 tests, including adversarial no-lookahead tests
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
python scripts/12_carry_backtest.py         # base book, cost scenarios, Deflated Sharpe
python scripts/13_carry_walkforward.py      # config re-chosen per window
python scripts/14_crypto_capacity.py        # --leg min, --exit-penalty 0.1, --sample oos
python scripts/15_carry_turnover_sweep.py   # 480 books + rebalance-phase robustness
python scripts/16_carry_diligence.py        # attribution, breadth, stress, survivorship, exit penalty
```

Step 11 reads the point-in-time universe from step 10 and includes contracts that
have since been delisted. Step 16 measures survivorship by rerunning the book without
them.

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
scripts/        01-07 equity, 10-16 crypto; each parses args and writes files
tests/
docs/           DESIGN.md, LIMITATIONS.md, PRODUCTION.md
```

## Part I: equity pairs

### Method

1. **Universe.** 72 liquid US names plus 9 sector ETFs ([configs/universe.csv](configs/universe.csv)),
   daily bars, 2015 to 2025. The list is static, so it leans toward survivors. See
   [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
2. **Pairs.** Within sector only. Engle-Granger on log prices in both directions,
   keep p < 0.05, half-life between 2 and 60 days, positive hedge ratio, at most two
   pairs per name. The count of pairs tested is saved; it feeds the Deflated Sharpe.
3. **Hedge ratio.** Kalman filter with a random-walk state [alpha, beta]. The traded
   spread is the prior innovation (the posterior has already seen today's price).
   A residual mode is also available.
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

| Metric                             | Static hedge | Kalman hedge |
| ---------------------------------- | ------------ | ------------ |
| Sharpe, single split OOS (net)     | -0.03        | -0.21        |
| **Sharpe, walk-forward OOS (net)** |              | **-0.45**    |
| CAGR, walk-forward                 |              | -1.24%       |
| Max drawdown, walk-forward         |              | -13.0%       |
| Hit rate / average holding         | 62% / 19d    | 59% / 3.5d   |
| Average win / average loss         | +23 / -37 bp | +10 / -15 bp |

`results/summary.csv` (single split), `results/walkforward_summary.csv`.

Most trades win, but the average loss is larger than the average win, and costs of
3.5 bp per side on each leg take the rest. The Kalman variant churns: a well-specified
filter drives its own innovations toward white noise, so the z-score flips sign
quickly and trades last 3.5 days.

The 17 six-month walk-forward windows range from +2.40 to -2.79, with a median of
-0.78; 6 of 17 are positive (`results/walkforward_windows.csv`). The Deflated Sharpe,
charged for the 5,287 pairs tested across windows, is 0.000001
(`results/deflated_sharpe.json`).

### The null experiment

`scripts/07_null_experiment.py` runs the entire pipeline on independent random walks
with no cointegration by construction.

|                                       | Value                  |
| ------------------------------------- | ---------------------- |
| Null Sharpe, mean / range             | -0.29 / [-0.62, +0.08] |
| Pair hypotheses tested                | 49,980                 |
| Expected best Sharpe under pure noise | **1.32**               |

`results/null_experiment.json`.

Searching this many hypotheses in noise would find a best Sharpe of about 1.32. The
real walk-forward Sharpe of -0.45 is nowhere near it, so the honest conclusion is a
negative result.

## Part II: crypto funding carry

Perps have no expiry, so exchanges tie them to spot with a funding payment between
longs and shorts every 8 hours. When leveraged longs crowd in, longs pay shorts. Long
spot plus short perp collects that payment with no directional exposure. The edge is
a payment for risk transfer, not a pattern found by searching, which is why it holds
up out of sample where cointegration did not.

### Result (127 Binance USD-M perps, point in time, Jan 2022 to Jun 2025, 8h bars)

130 perps are downloaded (15 since delisted); 127 have enough history to enter the
panel (12 since delisted). The base book holds the top 10 per side by 7-day funding,
re-ranks weekly and keeps incumbents inside the top 30.

| Metric                          | Full sample | Out of sample  |
| ------------------------------- | ----------- | -------------- |
| Sharpe, net of costs            | **5.79**    | **5.60**       |
| Sharpe, gross                   | 7.70        | 7.88           |
| CAGR                            | 22.3%       | 16.6%          |
| Max drawdown                    | -2.5%       | -1.9%          |
| Annual vol                      | 3.5%        | 2.8%           |
| Annual turnover                 | 45x         |                |
| Deflated Sharpe (n_trials=1440) |             | 0.9999, passes |

`results_crypto/carry_summary.csv`, `cost_sensitivity.csv` (gross = zero cost),
`carry_dsr.json`. Out of sample is the second half of the bars, from 2023-10-01.

**Do not read 5.60 as a forward estimate.** Two things move it more than any modeling
choice.

**Exits from dying contracts.** When a held contract stops printing, the engine exits
at the last good price for free. That happens 9 times in the base book. Charging a
fraction of the position on each:

| Exit penalty | Sharpe, full | Sharpe, OOS | Max DD |
| ------------ | ------------ | ----------- | ------ |
| 0% (current) | 5.79         | 5.60        | -2.5%  |
| 10%          | 4.59         | 2.92        | -3.1%  |
| 25%          | 2.61         | 0.52        | -11.3% |
| 50%          | 0.90         | -0.73       | -27.0% |

`results_crypto/exit_penalty_sensitivity.csv`.

The real cost of exiting a delisting contract is unknown, and the result swings from
strong to negative across that range. Modeling it is the most valuable open item.

**Which settlement the book rebalances on.** Weekly rebalancing means one of 21 settlements
in the cycle trades. Rerunning the base book at each one, with nothing else changed,
gives an OOS Sharpe from 2.75 to 6.71, mean 5.16 (`turnover_sweep_phase.csv`). The
reported 5.60 is one draw. The four weakest phases share one bad trade: they held
LINAUSDT into its perp delisting in March 2025 and lost about 4% of the book on one bar
(`rebalance_phase_worst_bar.csv`).

### Costs set viability

| Scenario                | Sharpe, full | Sharpe, OOS |
| ----------------------- | ------------ | ----------- |
| Zero cost (upper bound) | 7.70         | 7.88        |
| VIP / maker-ish (0.5x)  | 6.79         | 6.80        |
| Retail taker (base)     | 5.79         | 5.60        |
| Stressed / wide (2x)    | 3.74         | 3.19        |

`results_crypto/cost_sensitivity.csv`.

A round trip on both legs costs 27 bp (4 bp futures and 7.5 bp spot taker fees plus
1 bp half-spread per leg, each way). Funding earns the book 6.8 bp a day on average
(`pnl_attribution.csv`).

### How sticky the book needs to be

`scripts/15_carry_turnover_sweep.py` runs 480 books (rebalance every 8h to 21 days,
hysteresis 1x to 5x, top 5 to 20, lookback 3 to 14 days) and reruns the base book at
every rebalance phase.

| Book (top 10, 7-day lookback)            | Turnover | OOS Sharpe, base costs | OOS Sharpe, 2x costs |
| ---------------------------------------- | -------- | ---------------------- | -------------------- |
| Naive: re-rank every 8h, no hysteresis   | 150x     | 2.81                   | -0.89                |
| Daily, hysteresis 3x (phase mean)        | 89x      | 3.78                   | 1.31                 |
| Weekly, hysteresis 3x (phase mean)       | 45x      | 5.16 (2.75 to 6.71)    | 2.71                 |
| Two-weekly, hysteresis 3x (phase mean)   | 29x      | 5.51 (2.49 to 7.48)    | 3.32                 |
| Three-weekly, hysteresis 3x (phase mean) | 21x      | 5.64 (1.81 to 8.12)    | 3.82                 |

`results_crypto/turnover_sweep.csv` (naive row), `turnover_sweep_phase.csv` (the rest).

The naive book still makes money at base costs (4.08 full sample) but fails the
Deflated Sharpe (0.61) and loses money at 2x costs. Averaged over phases, a sticky book
rebalanced every 5 to 21 days beats it at every cost level. Inside that range the cadences tie at base costs,
and slower ones do better at 2x costs but spread more across phases. Weekly is a
reasonable choice inside a flat region, not an optimum.

Picking the best of the 480 books on the first half and trading it on the second
gives a 5-day, top-5 book: in-sample 9.43, out of sample 3.65, Deflated Sharpe 0.83,
which fails (`turnover_sweep_meta.json`).

The Deflated Sharpe depends on how trials are charged. At the 1440 development trials
of `12_carry_backtest.py` with the textbook null variance, the base book passes (0.9999).
Charging all 1920 trials with the Sharpe variance measured across the 480 books, the
base book fails (0.33) and 33 of the 480 books pass (`turnover_sweep.csv`).
That version overstates noise (much of the spread across books is real, and the books
overlap heavily), so the truth is between the two.

### Stress regimes

| Regime          | Window (inclusive)       | Return | Sharpe | Max DD |
| --------------- | ------------------------ | ------ | ------ | ------ |
| LUNA collapse   | 2022-05-05 to 2022-05-20 | +2.98% | 20.6   | -0.16% |
| 3AC / Celsius   | 2022-06-10 to 2022-06-25 | +1.47% | 14.6   | -0.13% |
| FTX collapse    | 2022-11-06 to 2022-11-20 | +1.44% | 9.9    | -0.29% |
| 2023 rally      | 2023-10-01 to 2024-03-31 | +13.95%| 10.7   | -0.72% |
| Aug 2024 unwind | 2024-08-01 to 2024-08-10 | -0.01% | -0.2   | -0.14% |
| 2025 drawdown   | 2025-02-01 to 2025-04-30 | +2.87% | 2.8    | -1.89% |

`results_crypto/stress_regimes.csv`, from `regime_breakdown(returns, windows,
periods_per_year=1095)`. Windows are whole UTC days, matched on each return's date.

Five of six windows are positive and the sixth is flat. Read the return and drawdown
columns, not the Sharpe column, which annualizes a two-week window. Capital held on
FTX would still have gone to zero, and no backtest metric would show it.

### What the diligence found

All from `scripts/16_carry_diligence.py`.

- **Squeeze gate.** Without `max_predicted_bps`, TRBUSDT is the top source of gross
  P&L at 11.7% and full-sample Sharpe is 6.57. With it, the top symbol is 7.3% and
  Sharpe 5.79 (`concentration.csv`). The gate gives up return on purpose.
- **Breadth.** 120 symbols are held at some point, at most 20 at once. Effective
  breadth is 34.8 and the average pairwise P&L correlation is 0.002 (`diligence.json`).
- **Attribution.** 95.1% of gross P&L is funding and 4.9% basis (`pnl_attribution.csv`).
- **Tail dependence.** Removing the best 1, 5, 10 or 20 bars gives a full-sample Sharpe
  of 5.86, 5.95, 5.91 and 5.66, and OOS 5.64, 5.62, 5.37 and 4.85 (`top_k_removal.csv`).

Six data bugs were found and fixed: klines indexed by open time while using close
prices; the spot endpoint silently capping pages at 1000 and truncating history;
millisecond jitter in funding timestamps colliding under ceil() and leaving holes in
the panel; forward-fill freezing one leg at a delisting; zero-volume bars treated as
prices; and a set iteration that made results depend on Python's per-process hash
seed. The last four have regression tests in `tests/test_carry.py`.

### Walk-forward over hyperparameters

The single-split config was chosen by hand after seeing the whole sample.
`scripts/13_carry_walkforward.py` picks the best config from a 12-point grid on each
one-year formation window and trades it for the next three months.

| Metric                 | Walk-forward OOS (Jan 2023 to Jun 2025) |
| ---------------------- | --------------------------------------- |
| Sharpe                 | **7.22**                                |
| CAGR                   | 22.2%                                   |
| Max drawdown           | -0.9%                                   |
| Modal config chosen in | 30% of windows                          |
| Deflated Sharpe        | 1.00, passes (n_trials=120)             |

`results_crypto/carry_walkforward_summary.csv`, `carry_walkforward_windows.csv`,
`carry_walkforward_dsr.json`.

This is not the headline. Window Sharpes range from -0.14 to 17.76, and on window 7
(Oct to Dec 2024), varying only `lookback` moves it from 8.05 to 11.62:

| lookback | Sharpe | Ann. vol | Max DD | Held UNFIUSDT? |
| -------- | ------ | -------- | ------ | -------------- |
| 9        | 8.18   | 1.6%     | -0.29% | no             |
| 21       | 8.05   | 1.5%     | -0.35% | yes, 14.3%     |
| 63       | 11.62  | 1.2%     | -0.17% | no             |

`results_crypto/carry_walkforward_probe.csv`.

The walk-forward chose 63 from formation data. The other two lookbacks earn about
8 on the same window, so part of 7.22 is the luck of that pick. The lower
single-split number is quoted.

### Survivorship, measured

`scripts/10_build_pit_universe.py` rebuilds the universe from Binance's own archive:
901 USDT perps ever listed, 244 since delisted (`configs/universe_perp_pit.csv`;
LUNAUSDT is active Jan 2021 to May 2022). The book then runs on the 127-symbol
panel and again without its 12 since-delisted symbols.

|                                    | Sharpe, full | Sharpe, OOS | CAGR  | Max DD |
| ---------------------------------- | ------------ | ----------- | ----- | ------ |
| Survivor-only (115)                | 5.81         | 5.70        | 22.9% | -2.6%  |
| Point-in-time (127, incl. 12 dead) | 5.79         | 5.60        | 22.3% | -2.5%  |

`results_crypto/survivorship.csv`.

Survivorship inflates Sharpe by +0.02 full and +0.10 OOS. In a delta-neutral book both
legs collapse together, so the hedge protects you. LUNAUSDT's entire death contributed
+0.09% of AUM (`diligence.json`).

An earlier version of this README reported inflation of +0.16 / +0.04 on an -11.4% max
drawdown and attributed the drawdown to a real UNFIUSDT delisting loss. Both numbers
were artifacts of two of the data bugs above (a frozen perp marked against a live spot,
amplified by a missing concentration cap).

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

`results_crypto/capacity_carry_perp.csv`.

| Assumption                               | Sharpe halves | Sharpe hits zero |
| ---------------------------------------- | ------------- | ---------------- |
| Base (k=0.10, perp ADV)                  | $77M          | $373M            |
| Conservative (k=0.10, thin-spot haircut) | $30M          | $131M            |

`results_crypto/capacity_verdict_perp.json`, `capacity_verdict_min.json`.

Both rows exit dying contracts for free. Charging 10% on each exit, and keeping
only the out-of-sample half (where all nine exits fall), shrinks capacity:

| Perp ADV / thin-spot, Sharpe halves | Full sample  | Out of sample |
| ----------------------------------- | ------------ | ------------- |
| Free exits                          | $77M / $30M  | $38M / $15M   |
| 10% exit penalty                    | $65M / $25M  | $23M / $9M    |

`capacity_verdict_{perp,min}[_exit10][_oos].json`. The out-of-sample book at a 10%
penalty starts from a Sharpe of 2.92, so it is the case that matches that headline.

Capacity scales as 1/k^2 and k is set a priori, not calibrated from fills, so these are
order-of-magnitude figures.

## Before quoting any number

Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md). The largest unmodeled risks for carry
are the cost of exiting a delisting contract, exchange counterparty risk (FTX is inside
the sample), and margin or liquidation on the short perp leg. For equities, the static
universe leans toward survivors, which biases an already negative result upward.
