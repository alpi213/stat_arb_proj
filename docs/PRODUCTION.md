# What would have to change before running live

This part contains the elements that would need to be built to bring the model in production.

1. **Data.** yfinance is fine for research, but a live system needs a paid data vendor under contract.
   Every raw delivery is stored exactly as received and price adjustments for splits and dividends are
   kept in a separate table, so the full history can always be rebuilt. Data loads every night and is
   checked before anything uses it. Does the row count match the trading calendar? Do a sample of
   prices agree with a second vendor? If a check fails, the pipeline stops and raises an alert.
   A gap that goes unnoticed today turns into a wrong position two days later.
2. **Research and production parity.** The code that turns data into signals
   (`hedge`, `signals`, `crypto.funding`) is identical in the backtest and in production.
   A daily test proves it by running yesterday’s live inputs through the backtester and confirming
   that the target positions match exactly. Every run also records the code version, the settings
   and the data it used, so any result can be reproduced later.
3. **Execution.** An order manager takes the target positions and splits them into smaller
   orders for the market. From the first trade, it compares each fill with what the cost model
   predicted. That comparison calibrates the market-impact model on real trades instead of
   assumptions, which in turn shows how much capital the strategy can really handle. For equities,
   it confirms shares can be borrowed before any short sale and tracks the borrowing cost
   per stock. For the crypto carry trade, it uses passive limit orders and has a defined
   response when one side of the trade fills and the other doesn’t.
4. **Risk controls.** Every order is checked before it goes out: size relative to the stock’s
   average daily volume, total and net exposure limits, concentration in any one name
   and a reasonable price range. During the day, the system watches portfolio drawdown
   and blocks new orders if market data hasn’t updated within a set number of minutes.
   For the carry trade, it simulates margin calls and liquidation risk on the short perpetual-futures
   side. For pairs trades, it keeps checking whether each pair still moves together and removes
   a pair that breaks down before it reaches its stop-loss.
5. **Operations.** The daily cycle runs as a chain of scheduled steps: load data, check quality,
   compute signals, place orders, reconcile and explain P&L. At the end of each day,
   the positions reported by the broker are matched against the positions the model expects.
   P&L is broken down by position and by day into gross return, trading costs,
   borrowing fees and market impact. When live results fall short of the backtest,
   the breakdown shows which assumption failed.
6. **Scaling.** Trading starts at around 10% of the capital the model says the strategy can take.
   It grows only while real trading costs stay in line with the predictions. Capacity then
   becomes a figure that live trading data keeps updating, not a chart made once
   and never revisited.
