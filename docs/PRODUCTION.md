# Path to production

What separates this research repo from a deployable system, in the order you'd
actually build it. Useful both as a roadmap and as interview material ("here's
exactly what I'd add before risking capital").

## 1. Data infrastructure

- Replace yfinance with a contractual vendor (Polygon, Databento, Norgate for PIT
  membership). Store **raw vendor payloads immutably** + adjustment factors
  separately; derive adjusted series on read. Never mutate history in place.
- Nightly ingestion job with checks that fail loudly: row counts vs calendar,
  cross-vendor spot checks, adjustment-factor diffs. A silent data gap is a silent
  position error two days later.
- At >500 names or intraday bars, move the store to ClickHouse/TimescaleDB behind
  the existing `PriceStore` interface.

## 2. Research/production parity

- The signal path (`hedge` → `signals`) must be the SAME code in backtest and live.
  The engine differs (simulated fills vs broker API); the signal code must not.
  Guard with a **shadow test**: run yesterday's live inputs through the backtester
  and assert identical target positions.
- Pin the environment (lockfile/container digest). Log (config hash, code hash,
  data manifest hash) with every run — three hashes fully identify any result.

## 3. Execution layer

- Order manager: target-position in, child orders out. Start with broker
  IOC/limit-at-mid via IBKR or Alpaca paper first.
- Model→actual slippage telemetry from day one: realized shortfall per order vs the
  cost model's prediction. This is also how you eventually **calibrate `k`** in the
  impact model from your own fills — closing the loop on the capacity estimate.
- Borrow: pre-trade locate check; hard-to-borrow names excluded at selection time,
  borrow rate fed per-name into `CostModel` (interface already accepts it).

## 4. Risk controls (before first live dollar)

- Pre-trade: max order size vs ADV, max gross/net, max per-name concentration,
  fat-finger price collars.
- Intraday kill conditions: portfolio drawdown limit, per-pair z-score beyond stop
  without a fill, data staleness (no price update in N minutes → flatten nothing,
  alert, block new orders).
- Cointegration health monitor: rolling ADF p-value per live pair; auto-quarantine
  pairs whose relationship degrades instead of waiting for the stop.

## 5. Operations

- Everything is a DAG (even cron + Makefile beats hidden manual steps; Airflow or
  Prefect when the step count grows): ingest → quality gate → signals → orders →
  reconciliation → P&L attribution.
- End-of-day reconciliation: broker positions/cash vs model book, penny-exact or
  page someone.
- P&L attribution per pair per day (gross alpha, costs, borrow, impact) — when the
  live Sharpe misses the backtest, this is how you find out which assumption broke.

## 6. Capital scaling discipline

- Deploy at a fraction of the model capacity estimate (e.g. 10%), grow only as
  realized slippage matches predicted. The capacity curve from `04_capacity.py`
  becomes a living document recalibrated from telemetry, not a one-off chart.
