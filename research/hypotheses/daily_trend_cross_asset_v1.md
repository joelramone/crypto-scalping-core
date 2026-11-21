# HYP-DAILY-TREND-XASSET-001: Daily Trend Rule Across ETFs and Gold

## Registration

- **Hypothesis ID:** `HYP-DAILY-TREND-XASSET-001`
- **Status:** `XASSET_DOES_NOT_SUPPORT` (run 2026-11-21, EXP-000028, result: `research/backtests/daily_trend_cross_asset_v1.json`). 2 of 4 assets passed (SPY, EFA); TLT and GLD failed. Closed under the anti-tuning rule.
- **Purpose:** cross-asset evidence for the daily trend rule of `HYP-DAILY-TREND-BTC-001` (EXP-000027). This is descriptive in-sample evidence. It does not change or rescue EXP-000027, and it authorizes no trading.
- **Registered:** 2026-11-21, before the datasets were downloaded.

Question: does the **same frozen rule**, unchanged, improve risk-adjusted returns over buy-and-hold in major traditional asset classes?

## Asset basket (fixed before data)

| Ticker | Asset class |
|---|---|
| `SPY` | US large-cap equities (S&P 500) |
| `EFA` | Developed-market equities ex-US |
| `TLT` | Long-term US Treasuries (20+ years) |
| `GLD` | Gold |

One ETF per major asset class, chosen for long history and liquidity. No asset may be added or removed after seeing results.

## Contamination disclosure

These histories are public. Trend rules on equity indices, notably the 10-month / 200-day moving-average rule popularized in 2007, are known to have reduced drawdowns, especially in 2008. In-sample results here are weak evidence, as for BTC. The window 2008–2026 postdates that popularization only partially, and that is noted, not relied on.

## Data

- **Source:** Yahoo Finance chart API (`query1.finance.yahoo.com/v8/finance/chart`), daily, **dividend- and split-adjusted** via `adjclose`. This is an unofficial endpoint: data can be revised, so each file's SHA-256 is recorded at download and verified before the run.
- **Adjusted open:** `open × adjclose / close` for the same day.
- **Window:** `2005-01-01` to `2026-11-01` (exclusive), downloaded under owner authorization **TASK-036B** with `python -m app.research.download_yahoo`.
- **SHA-256 recorded after download, before any strategy run:** each file has 5,492 rows (2005-01-03 to 2026-10-30).
  - SPY `856a0ddb4d690e6b39a164b01a7bdd5c7db067ce0b09aef3d1d5ebabb4123f27`
  - EFA `88f2efbb87fd8059d680b173cec94bf2d09a4d3115b9f1030d75ef877b806976`
  - TLT `4353c055643124b15b12e86b9fc9311a63ab2d99d7bf22a81b718213a53aba4e`
  - GLD `c5be9eff7f25d4b46bd259bc04924eb744a64a05cb9aa414557523b1b98e12b5`

## Frozen rule (identical to HYP-DAILY-TREND-BTC-001)

- `position[t] = mean(close[t] > SMA_n[t] for n in (50, 100, 200))` on adjusted closes, with SMA lengths counted in **trading days**.
- The position is decided after the close of day `t` and held from `open[t+1]` to `open[t+2]` (adjusted opens).
- Long or flat only. Cash earns 0%, which understates a real strategy, since cash would earn T-bill interest.
- Fee: 0.1% per unit of notional traded, charged on position changes. Buy-and-hold pays one entry fee.
- The evaluation starts on the first day all SMAs exist, and buy-and-hold starts the same day.
- Annualization uses **252** trading days.

## Metrics

For each asset, both strategy and buy-and-hold, over the full window and each half: CAGR, annualized volatility, Sharpe (rf = 0), max drawdown, Calmar, time in market, position changes and calendar-year returns.

## Pre-registered verdict

An asset **passes** if, in the full window:

1. strategy Sharpe > buy-and-hold Sharpe, **and**
2. |strategy max drawdown| ≤ 0.6 × |buy-and-hold max drawdown|.

- **`XASSET_SUPPORTS_TREND`** if at least 3 of the 4 assets pass.
- **`XASSET_DOES_NOT_SUPPORT`** otherwise.

Half-window results are reported but not part of the verdict.

## Anti-tuning rule

After results are seen, it is prohibited to change the basket, the rule, the fees, the window, the cash assumption or the pass criteria.
