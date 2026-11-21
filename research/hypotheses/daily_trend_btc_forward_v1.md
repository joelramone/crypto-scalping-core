# HYP-DAILY-TREND-BTC-FWD-001: Daily Trend Following on BTC, Forward Paper Test

## Registration

- **Hypothesis ID:** `HYP-DAILY-TREND-BTC-FWD-001`
- **Status:** `FORWARD_IN_PROGRESS`
- **Registered:** 2026-11-21. The forward window starts on **2026-11-21**: the first holding day runs from open 2026-11-21 to open 2026-11-22, whose outcome was unknown at registration.
- **Correction before any forward data existed:** a first draft started the window on 2026-11-01. That was wrong, because 2026-11-01 to 2026-11-20 had already happened (BTC fell about 20%, which is public) and was seen while drafting. The start moved to the registration day, and the verdict date moved by the same 20 days. No rule or criterion changed.
- **Origin:** the rule of `HYP-DAILY-TREND-BTC-001` (EXP-000027, `TREND_REJECT` in-sample, failed only its drawdown check). This is a **new** hypothesis, judged **only on data that did not exist** when the rule was frozen. It does not reopen or rescue EXP-000027, whose verdict stands.
- **Paper only:** no real money is involved, and this protocol does not authorize live trading.

## Why a forward test

The in-sample history of BTC is public knowledge and cannot give clean evidence. Prices from 2026-11-21 onward were unknown when this rule and these criteria were fixed. A forward test is therefore the only clean test available.

## Frozen rule (identical to HYP-DAILY-TREND-BTC-001)

- **Markets:** Binance spot `BTCUSDT` (primary) and `ETHUSDT` (control), daily candles, long/flat only.
- **Signal:** `position[t] = mean(close[t] > SMA_n[t] for n in (50, 100, 200))`, decided after the close of day `t` and held from `open[t+1]` to `open[t+2]`.
- **Costs and cash:** fee 0.1% per unit of notional traded; cash earns 0%.
- **Portfolio start:** both portfolios start flat before the first forward holding day, so the strategy pays an entry fee on its first position. Buy-and-hold pays one entry fee on 2026-11-21.
- **Warm-up history:** only the days before 2026-11-21 are used, solely to compute the SMAs. They are never evaluated.

## Data and tracking

- `python -m app.research.daily_trend_forward` downloads daily klines from `2026-03-01` to the last completed UTC day. It uses `data-api.binance.vision`, Binance's public market-data mirror, which matched `api.binance.com` and the frozen TASK-035B dataset exactly when checked on 2026-11-21. It falls back to `api.binance.com` if the mirror fails.
- Each run validates the data (contiguous days, sane OHLC) and rewrites `research/forward/daily_trend_btc_fwd_v1/status.json` and `report.md`.
- **Revision guard:** if a forward day already recorded in `status.json` comes back with a different return, the run fails instead of silently rewriting history.
- Tracking runs automatically once a month and is committed to the repository.

## Evaluation (fixed now, before any forward data)

**Verdict date:** the first run on or after **2028-11-21** (two years of forward data). Earlier runs report descriptive progress only, with verdict `FORWARD_IN_PROGRESS`.

At the verdict date:

1. **`FORWARD_INCONCLUSIVE`** if BTC buy-and-hold never had a drawdown of at least 20% in the forward window. Trend following is expected to help mainly in large declines, and without one the test has no power.
2. **`FORWARD_CONFIRMED`** if all of the following hold:
   - BTC strategy Sharpe > BTC buy-and-hold Sharpe;
   - |BTC strategy max drawdown| ≤ |BTC buy-and-hold max drawdown|;
   - ETH strategy Sharpe > ETH buy-and-hold Sharpe.
3. **`FORWARD_REJECTED`** otherwise.

## Anti-tuning rule

No change to the rule, the costs, the start date, the verdict date or the criteria is allowed at any point. Interim results must not trigger early stopping, early verdicts or modifications.

A `FORWARD_CONFIRMED` verdict would justify discussing a small live allocation under a separate decision. It would not authorize one.
