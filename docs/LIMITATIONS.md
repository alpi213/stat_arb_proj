# Known limitations

This section lists what the backtests leave out and which way each gap pushes the results. 
Every number in the README should be read with these caveats in mind.

## Equity pairs

### Data

1. **Survivorship bias.** The universe is a fixed list of large companies that have been around
   for a long time. Pairs where one company went bankrupt or was acquired never 
   make it in, so results look **better** than they would have in real time.
2. **Daily prices only.** The backtest can't see how the spread moved during the day, 
   overnight gaps, or the exact price a trade would have filled at. Execution risk is understated.
3. **Yahoo data quality.** Yahoo sometimes restates its price adjustments and occasionally 
   publishes bad prices. The data-quality checks catch most of this, but not all of it.

### Execution and costs

4. Every trade fills in full at the next day's closing price. Partial fills, trading halts
   and auctions are not modelled.
5. Costs are the same for every stock and every period: 1 bp commission, 2.5 bp half-spread
   and 50 bp a year to borrow shares for shorting. In reality, spreads widen exactly when pairs 
   drift apart, and hard-to-borrow stocks can cost several percent a year. In stressed markets 
   this makes results look **better** than they would be.
6. Borrowed shares are never called back. In a crisis like March 2020, lenders recall shares 
   and force you to close exactly the short positions you most want to keep.
7. Dividends owed on short positions and the financing cost of margin are not included.

### Modelling

8. Two stocks that have always moved together can stop doing so for good, for example 
   after a merger or when the businesses go in different directions. Stop losses, time limits 
   and regular re-selection of pairs reduce this risk, but nothing removes it.
9. The market impact coefficient `k` is set to an assumed value (0.1) rather than estimated 
   from real trades, so the capacity figures are rough orders of magnitude.
10. The Kalman filter's `delta` parameter is shared by all pairs instead of tuned for each one. 
    This choice allows fewer tuned parameters, which means fewer hypotheses tested, 
    therefore keeps the Deflated Sharpe Ratio accurate.
11. The portfolio is only roughly market-neutral. Each pair is dollar-neutral, but there is no hedge
     against overall market moves (beta) at the portfolio level.

### Statistical

12. The test covers one decade, and that decade only happened once. The Deflated Sharpe Ratio
     corrects for having picked the best pairs and settings, but not for the fact that 
    2015–2025 is a single draw from history.

## Crypto funding carry

### Data

1. **Survivorship bias: measured, and small.** The test universe includes every perpetual future
    ever listed on Binance (901 contracts, 244 of them since delisted). Compared with testing 
   only the contracts that survived, the Sharpe ratio is inflated by just +0.02 on the full sample 
   and +0.10 out of sample. Being delta-neutral is what protects the strategy: when a coin 
   collapses, the long and the short leg fall together. LUNA's entire collapse (LUNAUSDT) 














   moved the portfolio by only +0.09% of assets (`diligence.json`, `lunausdt_pnl_total`).

   Getting to that number took three bug fixes, and two wrong conclusions along the way:

   a. *Stale prices at delisting.* The futures contract and the spot coin stop trading at different times. Filling each one forward separately froze one leg while the other kept moving. On a single LUNAUSDT bar this showed up as spot down 96% and the perp unchanged. The fix cuts both legs at the last moment both genuinely traded.

   b. *Bars with no trades treated as real prices.* Binance keeps publishing bars for a dead contract with zero volume and a frozen price. UNFIUSDT's perp sat at 1.730 for seven days while spot kept trading. The fix treats a bar with no trades as having no price, and a coin is only eligible if its current bar actually traded.

   c. *No limit per coin.* Weights were set as `sign / len(held)`, so as coins dropped out the book became concentrated: at one point it held 3 names, with 33% in UNFIUSDT while it was being delisted. The fix caps each coin at 15% (`max_weight_per_symbol=0.15`). When few coins qualify, part of the capital simply stays uninvested.

   Earlier versions of this document reported inflation of +0.16 / +0.04 with a −11.4% drawdown, and described the UNFIUSDT loss as a genuine delisting event. Both came from bugs (b) and (c).

   **The assumption the result depends on most:** bars with no trades are treated as missing, so a position in a dying coin exits at its last good price at no cost. There are 9 such exits in the base book. Charging a penalty on each lowers the out-of-sample Sharpe from 5.60 (no penalty) to 2.92 (10%), 0.52 (25%) and -0.73 (50%) (`exit_penalty_sensitivity.csv`, from `scripts/16_carry_diligence.py`). Modeling these exits properly is the most valuable open item.
2. **One exchange.** Anything specific to Binance, such as a temporary mispricing, shows up here as profit.
3. **8-hour bars.** Trades are assumed to happen at the close of the funding settlement bar.
4. The forecast uses the funding rate that was actually paid, not the rate Binance predicts in real time. A live system would use the predicted rate, so if anything the signal here is conservative.

### Execution and cost

5. Both legs pay taker fees at every rebalance (4 bp on futures, 7.5 bp on spot). A trading desk would post passive maker orders and pay far less, but then one leg might not fill and leave an unhedged position, which isn't modeled. The cost-sensitivity table covers the full range of fees.
6. In the backtest, costs grow in a straight line with trading volume. The capacity script adds market impact on top.
7. The hedge is assumed to be perfectly neutral. Real hedges drift between rebalances and cost money to correct.
8. When carry is negative and the strategy shorts spot, no borrowing cost is charged.

### Structural and tail risk

9. **Liquidation.** The trade is short the perpetual future. In a sharp rally, shorts face margin calls at exactly the moment funding moves against them. Margin is not modeled.
10. **Counterparty risk.** The money sits on a centralized exchange. The FTX collapse (November 2022) falls inside the test period, and anyone with funds there lost all of it. This is the largest risk the model leaves out, and adding more coins on the same exchange does nothing to diversify it.
11. A withdrawal freeze or a stablecoin losing its peg would hit the collateral and the quote currency at the same time.
12. The funding premium exists because leveraged buyers outnumber sellers. It has shrunk as more traders joined in (2019 paid far more than 2024).

### Statistical

13. About 3.5 years of data, one exchange and one market cycle. That is short compared with equity backtests.
14. The 120 symbols ever held (at most 20 at once) behave like 34.8 independent bets; the average pairwise correlation of their P&L is 0.002 (`diligence.json`). The starting assumption, that they would all move as one crowded-long factor, did not hold.
15. Without the `max_predicted_bps` filter, TRBUSDT is the largest single source of gross profit at 11.7%, against 7.3% for the top coin with the filter (`concentration.csv`). The filter gives up Sharpe on purpose (6.57 without it, 5.79 with it): the extra profit is squeeze-regime funding with an often unborrowable spot leg.
16. Returns are far from normally distributed (out-of-sample skew +1.8, kurtosis around 70). The Sharpe ratio summarizes them poorly, which is why the Deflated Sharpe Ratio, which adjusts directly for skew and kurtosis, is used instead.
