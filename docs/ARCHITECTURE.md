# Architecture

## Design principles

1. **Config-driven, code-stable.** Every knob lives in `configs/*.yaml`, validated by
   pydantic at load ([config.py](../src/statarb/config.py)). An experiment is
   reproducible from (config, commit hash). Changing a parameter never means editing
   code.

2. **Narrow interfaces at the seams that will change.**
   - `download_prices()` returns a canonical `dict[field -> wide DataFrame]` — swap
     yfinance for a paid vendor by adding one adapter.
   - `PriceStore` is a two-method parquet cache — swap for ClickHouse/Timescale
     without touching research code.
   - `HedgeEstimator.estimate()` is the single contract for hedge models — add an
     RLS or particle-filter estimator as a third class, config picks it.

3. **Look-ahead prevention is structural, not disciplinary.**
   - Estimators must produce row *t* from data ≤ *t* (contract on the base class,
     enforced by perturbation tests).
   - The engine applies `shift(execution_lag_bars)` in exactly one place;
     `execution_lag_bars >= 1` is enforced by pydantic (`ge=1`), so a config typo
     cannot create same-bar execution.
   - Pair selection takes an explicit formation window and slices internally — a
     caller can't accidentally select on the full sample.
   - `tests/test_no_lookahead.py` attacks the engine adversarially.

4. **One source of truth for state.** The capacity script re-runs the backtest from
   config rather than deserializing weights — no stale-file bugs at this scale.
   (At production scale you'd persist run artifacts with a run-ID; see
   PRODUCTION.md.)

5. **Scripts orchestrate, the library computes.** Everything in `scripts/` is glue;
   everything in `src/statarb/` is importable, typed, and unit-tested. Notebooks (if
   any) may only call library functions — no logic lives in notebooks.

## Data flow

```
universe.csv ──► load_universe ─┐
yfinance ──► download_prices ──► clean_prices ──► PriceStore (parquet + manifest + quality report)
                                                        │
                    ┌───────────────────────────────────┤
                    ▼                                   ▼
            select_pairs (formation window only)   adj_close / close / volume
                    │                                   │
                    ▼                                   │
      HedgeEstimator.estimate (static | kalman)         │
                    ▼                                   │
      rolling_zscore ──► generate_positions             │
                    ▼                                   │
      backtest_pair (lag, freeze-at-entry hedge, costs) │
                    ▼                                   ▼
      backtest_portfolio ──► metrics ──► capacity_curve (ADV, sigma)
                    ▼
      walk-forward loop (re-select per window) ──► DSR + regimes
```

## Module responsibilities

| Module | Owns | Must never do |
|---|---|---|
| `data` | acquisition, cleaning policy, PIT universe | signal logic |
| `pairs` | statistical tests, ranked selection | touch data outside its window |
| `hedge` | alpha/beta/spread paths | know about positions or costs |
| `signals` | z-score, state machine | know about execution or capital |
| `backtest` | lag, sizing, costs, aggregation, metrics | re-estimate anything |
| `capacity` | impact model, AUM sweep | change signals |
| `validation` | windows, DSR, regimes | trade |

The dependency direction is strictly downward in the table — e.g. `signals` imports
nothing from `backtest`. This is what makes each layer testable in isolation.

## Extending

- **New hedge model**: subclass `HedgeEstimator`, register in `hedge.make_hedge`,
  add a config block.
- **New cost component** (e.g. per-name spreads): extend `CostModel`; the engine
  only calls `trading_cost`/`borrow_cost`.
- **Intraday bars**: the engine is frequency-agnostic except for the 252
  annualization constants in `metrics.py`/`costs.py` — lift them into config first.
- **3+ asset baskets**: `johansen_test` already returns the cointegrating vector;
  generalize `backtest_pair`'s two-column weights to a vector of weights.
