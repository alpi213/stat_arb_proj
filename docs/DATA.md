# Data policy

Every choice below is a modeling decision with P&L consequences; documenting them is
part of the deliverable.

## Sources

- **Primary**: Yahoo Finance via `yfinance`, daily bars, `auto_adjust=False` so both
  adjusted and unadjusted closes are retained.
- **`adj_close`** (split + dividend adjusted): all signal math — cointegration on
  raw closes would "detect" every dividend as mean reversion.
- **`close` × `volume`** (unadjusted): dollar ADV for capacity — impact depends on
  the notional actually printed, not the adjusted series.

## Survivorship bias

- The universe file `configs/universe.csv` is a **point-in-time membership table**
  (`ticker, sector, entry_date, exit_date`). All selection/backtest code respects
  entry/exit dates; a delisted name forces liquidation at its exit date.
- **Honest caveat**: the shipped file lists large, long-lived names with blank exit
  dates — i.e. it is still a survivor-tilted universe. The *interface* removes the
  bias; the *data* to populate it (historical index membership) must come from a PIT
  source. Upgrade path:
  1. Scrape the revision history of the S&P 500 constituents Wikipedia page (free,
     imperfect), or
  2. CRSP/Compustat via WRDS, or a commercial PIT membership file.
- Directional effect of the remaining bias: overstates results (bad pairs that died
  are missing). Say this in any writeup.

## Cleaning rules (enforced in `data/quality.py`)

| Rule | Setting | Rationale |
|---|---|---|
| Min usable history | 750 days | formation window + buffer |
| Max missing bars | 2% of live range | more = untrustworthy series |
| Gap fill | forward-fill, limit 5 | never backfill (backfill = look-ahead) |
| Calendar | keep days where ≥90% of names traded | kills phantom provider bars |
| Return sanity | flag \|daily move\| > 50% | data errors on liquid daily names |

Everything dropped/flagged goes to `data/processed/quality_report.csv`, and
`manifest.json` records what was downloaded when — the audit trail ships with the
data.

## Known yfinance caveats

- Adjusted closes are retroactively restated on every dividend — cached data and a
  fresh download won't match exactly. Fine for research; a production system needs a
  vendor with immutable history (or store raw + adjustment factors separately).
- Occasional missing/zero volume bars on ETFs; the ≥90%-traded calendar rule absorbs
  most of these.
- Rate limits: the downloader batches in one call; if it fails, re-run (idempotent).
