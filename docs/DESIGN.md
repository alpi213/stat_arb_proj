## Design rules

1. **Every parameter lives in the config file.** `configs/base.yaml` is validated by pydantic
   in [config.py](../src/statarb/config.py) and any unknown key is rejected. An experiment is fully defined by its config
   file and a git commit.
2. **Look-ahead bias is prevented by how the code is built, not by being careful.** Look-ahead bias
   means accidentally using information that wouldn't have been available at the time of the trade.

   - Hedge estimators compute row t using only data up to t. They filter forward and never 
     smooth with later data.
   - The engine applies the execution delay in exactly one place, and the config refuses 
     any delay shorter than one bar.
   - Pair selection cuts out its own formation window, so no caller can pass in the full sample by mistake.
   - The carry backtester never shifts data. Weights set at t are held over (t, t+1] and asset returns
     dated t are earned over (t, t+1]. The only place a lag could go wrong is `carry_returns` 
     and a dedicated test locks it down.
   - `tests/test_no_lookahead.py` changes a future price and checks that no earlier position moves.
3. **Scripts coordinate and the library does the computing.** Scripts read arguments and write files. 
   Anything that runs a backtest from a config lives in `pipeline.py`, so the backtest, capacity, 
   walk-forward and null-experiment scripts all go through the same code.
4. **The capacity analysis re-runs the backtest** from the config instead of reading saved weights. 
   At this size it costs very little, and it rules out a whole category of bugs caused by outdated files.

## Equity data flow

From raw prices to walk-forward results:

```
universe.csv ──> load_universe ──────────────────────┐
yfinance ──> download_prices ──> clean_prices ──> PriceStore (parquet + manifest + quality report)
                                                      │
                   select_pairs (formation window) <──┤──> adj_close / close / volume
                            │                                        │
          StaticHedge | KalmanHedge .estimate                        │
                            │                                        │
          rolling_zscore ──> generate_positions                      │
                            │                                        │
          backtest_pair (lag, beta frozen at entry, costs)           │
                            │                                        v
          backtest_portfolio ──> metrics ──> capacity_curve (ADV, sigma)
                            │
          run_walkforward (re-select per window) ──> DSR + regimes
```

## What each module is responsible for

| Module         | Responsible for                        | Never                            |
| -------------- | -------------------------------------- | -------------------------------- |
| `data`       | downloading, cleaning rules, storage   | contains signal logic            |
| `pairs`      | cointegration tests, ranking pairs     | looks at data outside its window |
| `hedge`      | alpha, beta and spread over time       | knows about positions            |
| `signals`    | z-score, entry, exit and stop rules    | knows about execution            |
| `backtest`   | execution lag, sizing, costs, totals   | re-estimates anything            |
| `capacity`   | market-impact model, AUM sweep         | changes signals                  |
| `validation` | test windows, Deflated Sharpe, regimes | trades                           |
| `crypto`     | Binance connectors, carry data, signal | knows about the pairs engine     |

Each module depends only on the modules listed below it.

## Data policy

**Sources.** Daily bars from Yahoo Finance through yfinance, with `auto_adjust=False` 
so both adjusted and unadjusted closing prices are kept. All signal calculations use `adj_close`, 
because testing raw closes for cointegration would mistake every dividend for mean reversion. 
Dollar volume for the capacity analysis uses `close * volume`, since market impact 
depends on the amount that actually traded.

**Cleaning rules**, enforced in `data.clean_prices`:

| Rule                 | Setting                                     | Reason                                                             |
| -------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| Minimum history      | 750 days                                    | covers the formation window plus a buffer                          |
| Maximum missing bars | 2% of the period the stock was trading      | beyond that, the series can't be trusted                           |
| Filling gaps         | forward only, up to 5 bars                  | filling backward would use future data                             |
| Trading calendar     | keep days when at least 90% of names traded | removes fake bars created by the data provider                     |
| Return check         | flag daily moves above 50%                  | for liquid stocks, a move that large is almost always a data error |

Everything dropped or flagged is logged in `data/processed/quality_report.csv`
 and `manifest.json` records what was downloaded and when.

**yfinance limitations.** Yahoo recalculates adjusted closes every time a dividend is paid, 
so a cached file and a fresh download will not match exactly. That's acceptable for research, 
but a production system needs a price history that never changes after the fact. 
Occasional missing or zero-volume bars on ETFs are handled by the calendar rule.

**Crypto.** Only Binance's public endpoints are used. Price bars (klines) are indexed
by their closing time, because the close of [T, T+8h) isn't known at T. Funding payments
are rounded to remove millisecond timing noise and then summed into 8-hour buckets.
When a contract is delisted, both legs are cut at the last moment both traded
 and bars with zero volume carry no price.

## Extending the code

- **New hedge model:** add a class with an `estimate(log_y, log_x)` method that returns alpha, 
  beta and spread columns, then register it in `hedge.make_hedge`.
- **New cost component:** extend `CostModel`. The engine only ever calls `trading_cost` and `borrow_cost`.
- **Intraday data:** the engine works at any frequency. The only daily-specific parts are 
  the 252-day annualization constants in `metrics.py` and `engine.py`
