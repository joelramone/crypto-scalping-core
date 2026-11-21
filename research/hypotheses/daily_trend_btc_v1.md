# HYP-DAILY-TREND-BTC-001: Daily Trend Following on BTC (long/flat spot)

## Registration

- **Hypothesis ID:** `HYP-DAILY-TREND-BTC-001`
- **Protocol file:** `research/hypotheses/daily_trend_btc_v1.md` (machine-readable twin: `daily_trend_btc_v1.yaml`)
- **Status:** `TREND_REJECT` (run 2026-11-21, EXP-000027, result: `research/backtests/daily_trend_btc_v1.json`). It failed only the drawdown check: max drawdown -49.4% vs the -46.0% limit. Hypothesis ended per the anti-tuning rule.
- **Purpose:** `discovery_baseline`
- **Market:** Binance **spot** `BTCUSDT`, daily candles. Long or flat only: no leverage, no shorting, no perpetuals.
- **Control market:** Binance spot `ETHUSDT`, same rule, unchanged.
- **Registered:** 2026-11-21, before the daily datasets were downloaded.

This document freezes one deterministic test: does holding BTC only while it trends up give **better risk-adjusted returns than simply holding BTC**? The benchmark is buy-and-hold, not zero. It does not claim profitability and does not authorize optimization.

## Mechanism

Crypto prices show time-series momentum: sustained moves tend to persist over weeks to months, driven by slow information diffusion, herding and reflexive flows. The documented benefit of trend-following is mainly **avoiding large drawdowns** (bear markets) while keeping much of the upside, not beating buy-and-hold on raw return.

## Contamination disclosure

There is no untouched daily BTC history. The broad path of BTC since 2017 is public knowledge, and trend-following on BTC is known to have worked in this period. That knowledge biases any in-sample test toward success. This protocol therefore:

- uses only literature-standard moving-average lengths, chosen without testing alternatives;
- averages three lengths instead of picking one;
- requires stability across both halves of the sample;
- requires the same frozen rule to beat buy-and-hold on ETH;
- treats a `TREND_CANDIDATE` verdict as weak evidence that still needs forward (paper) confirmation on data that does not exist yet.

The intraday candle periods used by earlier hypotheses (EXP-000001 to 000026) were 1m–15m studies of a different mechanism. Overlapping daily data is disclosed here, not treated as a reuse violation.

## Data boundaries

| Role | Path | Period (UTC) |
|---|---|---|
| Primary | `data/binance_spot_BTCUSDT_1d_20170817_20261101.csv` | `2017-08-17` to `2026-11-01`, end exclusive |
| Control | `data/binance_spot_ETHUSDT_1d_20170817_20261101.csv` | same |

The data is downloaded under owner authorization **TASK-035B** with `python -m app.research.download_daily`. SHA-256 hashes are recorded in the YAML twin right after download and validation, before any strategy code touches the data. Data from `2026-11-01` onward is reserved for forward confirmation.

## Frozen signal rule

Fixed constants: `SMA_LENGTHS = (50, 100, 200)` days.

```text
sma_n[t]      = mean(close[t-n+1 .. t])          # includes the completed day t
vote_n[t]     = 1 if close[t] > sma_n[t] else 0
position[t]   = (vote_50[t] + vote_100[t] + vote_200[t]) / 3     # 0, 1/3, 2/3 or 1
```

- The position is decided after day `t` closes and is held from `open[t+1]` to `open[t+2]`.
- The fraction of capital not in BTC earns 0% (conservative; no T-bill or Earn yield is credited).
- Evaluation starts on the first day all three SMAs exist (warm-up of 199 days). Buy-and-hold starts on the same day.
- No volatility targeting, stop-loss, filter, regime classifier or other component may be added.

## Frozen costs

- `fee_rate = 0.001` per unit of notional traded (Binance spot VIP0 taker, assumed), charged on `|position[t] - position[t-1]|` at each change.
- Buy-and-hold pays one entry fee.
- Slippage is not modeled. Daily rebalancing in steps of 1/3 on BTCUSDT spot is small relative to typical depth.

## Metrics (both markets, strategy and buy-and-hold)

The full evaluation window and each half (split at the midpoint day) report:

- CAGR, annualized volatility (365 days), Sharpe ratio (rf = 0), maximum drawdown, Calmar ratio (CAGR / |max drawdown|);
- time in market (average position), number of position changes and total fees paid;
- calendar-year returns.

## Pre-registered verdict

**`TREND_CANDIDATE`** only if **all** of the following hold:

1. **BTC, full window:** strategy net CAGR > 0.
2. **BTC, full window:** strategy Sharpe > buy-and-hold Sharpe.
3. **BTC, full window:** strategy max drawdown ≤ 0.6 × buy-and-hold max drawdown (in absolute value).
4. **BTC, each half:** strategy Sharpe > buy-and-hold Sharpe.
5. **ETH control, full window:** strategy Sharpe > buy-and-hold Sharpe.

Otherwise the verdict is **`TREND_REJECT`**.

A `TREND_CANDIDATE` does not authorize live trading. The permitted next step is forward paper tracking from `2026-11-01` under the same frozen rule.

## Anti-tuning termination rule

After any outcome has been seen, it is prohibited to:

- change the SMA lengths, the voting scheme, the execution timing or the fees;
- add filters, leverage, shorting or volatility scaling;
- change the evaluation window or the split point;
- swap the control asset.

If the verdict is `TREND_REJECT`, this hypothesis ends.
