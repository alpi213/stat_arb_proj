# Limitations — what this project does NOT model, and why it matters

Listing these is not an apology; it is the boundary of validity of the results.

## Data

1. **Universe is survivor-tilted.** PIT interface exists, PIT membership data is not
   wired (see DATA.md). Effect: results biased **up**.
2. **Daily bars only.** Intraday spread dynamics, overnight vs intraday gap risk, and
   the actual fill price within the day are invisible. Effect: understates execution
   risk; a close-to-close backtest can't see that the z-score touched ±4 intraday.
3. **Yahoo data quality.** Restated adjustments, occasional bad prints. Mitigated by
   the quality layer, not eliminated.

## Execution & costs

4. **Fills at next close, always available.** No partial fills, no halts, no
   auction dynamics. One full bar of latency is conservative in timing but
   optimistic in fill certainty.
5. **Flat cost assumptions** (1 bp commission, 2.5 bp half-spread, 50 bp/yr borrow,
   uniform across names and time). Reality: spreads widen exactly when pairs
   dislocate (costs and opportunities are correlated); borrow on specials can be
   100s of bps. Effect: biased **up** in stress periods.
6. **Borrow availability assumed.** Shorts are never recalled. In 2020-03 style
   stress, recalls force-close exactly the positions you most want to keep.
7. **No dividend flows on shorts** (short pays the dividend) and **no financing
   spread** on margin. Partially offset by using total-return-adjusted prices, but
   not exact.

## Modeling

8. **Cointegration can break permanently** (mergers, index changes, business model
   divergence). The stop + time-stop + walk-forward re-selection mitigate; nothing
   eliminates regime risk. The 2022 slice in the regime report is the stress test.
9. **Impact model is a calibration guess.** The √-law form is well-supported
   empirically, but `k` is set a priori (default 0.1, sensitivity reported), not
   estimated from own executions — estimating it requires trading. Capacity numbers
   are order-of-magnitude, and should be presented that way.
10. **Pair-level Kalman `delta` is global**, not tuned per pair — deliberate, to keep
    the hypothesis count down for the DSR. A per-pair tune would need nested
    walk-forward.
11. **Net exposure is only approximately zero** (dollar-neutral per pair with
    beta-weighted legs; no portfolio-level beta neutralization against SPY). Residual
    market beta is small but nonzero — report it, or add a beta hedge overlay as an
    extension.

## Statistical

12. **One history.** Every number is one sample path of one decade. The DSR corrects
    for selection across pairs/configs, not for the fact that 2015-2025 happened only
    once. Bootstrap confidence intervals on Sharpe are a cheap addition (extension).
