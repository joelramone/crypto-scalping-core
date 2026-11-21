# Binance BTCUSDT Funding Carry Economics V1

## Summary

A delta-neutral carry (long spot BTC + short USD-M BTCUSDT perpetual, equal notional) **would have earned positive funding in every 365-day window since September 2019**, net of a full round trip of fees. Unlike the candle strategies (EXP-000021 to 000026), the source of return is real and has persisted.

**The yield is regime-dependent and is currently low.** It peaked in the 2021 bull market and has fallen to its historical minimum in the last 12 months. On capital, with the perpetual unlevered (1x), the last 365 days returned about **1.6% per year** after fees, before basis, slippage and conversion costs. That is likely below what plain USD or stablecoin yield pays, so **carry is not attractive at today's rates**, even though it has never had a losing year.

This is a descriptive measurement, not a strategy backtest. There are no signals or tuned parameters.

## Context

- **Gate 1 (eligibility), partly updated:** the owner states, as of 2026-11-21, that they have a KYC-approved Binance account with access to USDT perpetuals from Argentina. That is owner attestation, not a reviewed legal determination. The previous gate report covered Bybit only and stays unchanged ([funding_carry_decision_gate_v1.md](funding_carry_decision_gate_v1.md)).
- **Gate 3 (data), partly resolved for Binance:** the settled funding history is public. It was downloaded under owner authorization **TASK-034B** with `python -m app.research.download_funding`. The download covers 7,826 settlements from 2019-09-10 08:00 to 2026-10-31 16:00 UTC, with no gaps, every settlement on the 8h grid, and a maximum |rate| of 0.30%. SHA-256 `1b1779e108fe23ab5c56e74aa6c24855c8d4400fa56fb4efbe20aeeecb167e8b`. Historical pre-settlement estimates and executable two-leg prices are still not available.
- **Gate 2 (economics):** this report quantifies the funding side under stated fee assumptions. Basis, margin and liquidation mechanics remain unmodeled.

Funding data is a different data type from the candle datasets. Using it here consumes no reserved candle period.

## Method

- Source: `research/backtests/binance_funding_carry_economics_v1.json`, produced by `python -m app.research.funding_carry_economics`, which verifies the dataset hash.
- **Funding sign:** a positive rate means longs pay shorts, so the short perpetual leg receives `rate × notional` at each settlement.
- **Round trip:** four fills (buy spot, sell perp, then sell spot, buy perp), charged once per holding period.
- **Capital per unit of notional:** `1 + 1/leverage` (spot fully paid plus perp initial margin).

| Fee scenario (VIP0 assumptions, not verified against the account) | Spot per fill | Perp per fill | Round trip |
|---|---:|---:|---:|
| `taker_all_fills` | 0.100% | 0.050% | 0.300% |
| `maker_all_fills_bnb_discount` | 0.075% | 0.018% | 0.186% |

## Results

### By calendar year (holding the whole year, taker fees)

| Year | Funding (annualized, on notional) | Settlements positive | Longest negative streak | Worst 30 days | Net on notional | Net on capital 1x | Net on capital 3x |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019 (from Sep 10) | 7.48% | 82% | 9 (3 days) | +0.03% | 6.51% | 3.26% | 4.88% |
| 2020 | 17.19% | 86% | 24 (8 days) | −1.31% | 16.89% | 8.45% | 12.67% |
| 2021 | 30.61% | 93% | 9 | −0.36% | 30.31% | 15.15% | 22.73% |
| 2022 | 4.16% | 78% | 10 | −0.23% | 3.86% | 1.93% | 2.90% |
| 2023 | 7.87% | 90% | 7 | +0.06% | 7.57% | 3.78% | 5.67% |
| 2024 | 11.92% | 92% | 18 (6 days) | −0.07% | 11.63% | 5.81% | 8.72% |
| 2025 | 5.13% | 87% | 10 | +0.11% | 4.83% | 2.41% | 3.62% |
| 2026 (to Oct 31) | 3.12% | 75% | 14 | −0.30% | 2.76% | 1.38% | 2.07% |

Whole period: 11.46% per year of funding on notional, and 11.41% net of one taker round trip. On capital that is 5.7% at 1x and 8.6% at 3x. 85.8% of settlements were positive.

### Rolling 365-day funding sum (6,732 windows)

| Min | P10 | Median | P90 | Max | Windows below zero | Windows below taker round trip |
|---:|---:|---:|---:|---:|---:|---:|
| 3.21% | 4.12% | 8.60% | 30.12% | 33.56% | 0% | 0% |

The worst window ends on **2026-10-11**: today's yield is the lowest in the series. The best window ends on 2021-06-01.

### Recent

| Window | Funding (annualized) | Settlements positive | Net on capital 1x (taker) | Net on capital 3x (taker) | Days to recover the round trip (taker / maker) |
|---|---:|---:|---:|---:|---:|
| Last 365 days | 3.48% | 77% | 1.59% | 2.39% | 31.5 / 19.5 |
| Last 90 days | 5.86% | 93% | 2.32% | 3.48% | 18.7 / 11.6 |

## Interpretation

1. **The return source is real:** BTC perpetual longs have paid shorts on balance in every year, and no 365-day window lost money even after a full taker round trip.
2. **The size depends on the regime:** yields above 15% on notional came only in strong bull phases (2020–2021). Since 2022 the typical level is 4–12%, and the last year about 3.5%.
3. **On capital, the return is roughly halved:** the spot leg must be fully paid. Leverage on the perpetual leg raises the return on capital, but it adds liquidation risk on the short when BTC rallies, unless margin is topped up. At 3x, a rise of about 30% without a top-up liquidates the short and leaves the spot leg unhedged.
4. **Comparison with alternatives:** at 1.6–2.4% on capital, today's carry likely pays less than holding USD or stablecoins in a yield product, with more operational risk. This report did not verify any risk-free rate, so the comparison needs the rate actually available to the owner.

## What is not modeled

Basis at entry and exit, slippage and spread, margin top-ups and liquidation, interest on idle collateral, taxes, ARS↔USDT conversion costs, and fee tiers, which are VIP0 assumptions and not verified.

## Decision implications

- Carry on BTC alone is **not worth deploying at current funding levels**, given the capital it ties up.
- Possible next steps, each needing a separate owner decision:
  1. **Conditional carry:** only hold while trailing funding is above a threshold that clears an alternative yield. This needs a preregistered rule frozen before looking at forward data. The data used here is descriptive and would have to be split.
  2. **Altcoin perpetuals:** funding is often higher, but with greater basis, liquidity and delisting risk. A separate dataset authorization would be needed.
  3. **Park carry** until BTC funding rises again. The downloader and analysis can be re-run on new data at no cost.
