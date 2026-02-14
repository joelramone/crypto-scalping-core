# PHASE 2 – STRUCTURAL MATHEMATICAL VALIDATION

## 1) SUMMARY TABLE

| metric | reported | recalculated | deviation |
|---|---:|---:|---:|
| PnL (long) | `gross=(exit-entry)*qty`, `net=gross-fees` | `gross=(exit-entry)*qty`, `net=gross-(entry_fee+exit_fee)` | 0.0000% |
| PnL (short) | Not implemented in execution engine | Required formula defined, no executable short path | N/A (structural gap) |
| Fee impact | Entry+exit fees deducted from net PnL | Entry+exit fees deducted from net PnL | 0.0000% |
| Slippage modeling | Not modeled | Slippage term absent from fill/PnL path | N/A (missing model) |
| Position sizing formula | `size=min(balance*0.95,50)` in backtest path; `size=min(balance*risk_pct,max_pos)` in risk agent | Required: `(capital*risk_per_trade)/stop_distance` | >0.5% (formula mismatch) |
| Risk per trade limit | Not stop-based; no explicit stop-distance risk cap | Required risk cap check vs stop distance per trade | >0.5% (logic mismatch) |
| Compounding logic | Capital/equity updates every trade; position capped to 50 USDT | Capital update confirmed; capped sizing reduces compounding continuity | 0.0000% for equity update, structural cap present |
| Drawdown calculation | Absolute drawdown: `peak-equity` | Required relative drawdown: `(equity-peak)/peak` | >0.5% (unit mismatch) |
| Sharpe ratio | `avg_net_profit/std(net_profit)` (run-level approximation) | Required: `(mean(returns)-rf)/std(returns)` | >0.5% (definition mismatch) |
| Expectancy | `net_profit/trades_count` | `(win_rate*avg_win)-(loss_rate*avg_loss)` | 0.0000% (algebraically equivalent under current stats bookkeeping) |
| Win/Loss distribution consistency | wins/losses tracked only for pnl != 0 | `wins+losses+breakeven == trades` consistency expected | Inconsistent when breakeven trades exist |

## 2) DETAILED INCONSISTENCIES

### 2.1 PnL discrepancy report
- Dataset used: deterministic closed long cycle (`BUY 0.001 BTC @ 50000`, `SELL 0.001 BTC @ 51000`, fee rate 0.1%).
- Trade IDs affected with deviation > 0.01%: `[]`.
- Max deviation: `0.0000%`.
- Short-side structural coverage: `NO` (no executable short PnL path in wallet/execution loop).

### 2.2 Position sizing / risk
- Required stop-based sizing formula absent in live backtest sizing path.
- Stop distance zero-guard for sizing formula: not applicable (formula not implemented in execution path).
- Trades exceeding stop-based risk_per_trade constraint: `UNBOUNDED/NOT ENFORCED`.

### 2.3 Drawdown
- Reported implementation is absolute currency drawdown.
- Required relative drawdown ratio not used as primary max drawdown metric.
- Max drawdown metric equivalence with required formula: `FAIL`.

### 2.4 Sharpe
- Return series basis: run-level net profit aggregate, not per-trade/per-period returns.
- Risk-free rate subtraction: absent.
- Annualization handling: absent.
- Standard deviation zero-guard: present.

### 2.5 Compounding
- Equity updates are sequential and trade-realized.
- Position sizing hard cap (`50 USDT`) and fixed 95% balance multiplier diverge from strict risk-fractional compounding formula.

## 3) PASS / FAIL VERDICT

**FAIL**

## 4) RISK SEVERITY LEVEL

**CRITICAL**
