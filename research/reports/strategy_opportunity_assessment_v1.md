# Strategy Research Inventory & Opportunity Assessment v1

## Scope, provenance and decision boundary

Documentation-only assessment, dated 2026-10-17, against `main` at `4b6926626c9440876f9c4531ffe9e7d8fc70e328`. **No market experiments, backtests, optimization, parameter tuning, new experiment IDs, dataset access, Research Memory writes, strategy changes or production authorization.** Proposed candidates below are conceptual hypotheses for human review, not preregistrations or executable specifications.

**Repository facts** are drawn from source code, persisted journals, summary leaderboards and reports. **Speculation** is explicitly confined to the opportunity candidates and their qualitative comparison. No external market study or new empirical measurement supports a claim that any proposal is profitable. “Unexplored” means no corresponding strategy implementation or completed experiment was found in the inspected repository inventory; it does not mean novel in financial markets.

The principal historical source is the unchanged [EXP-000021–024 failure report](failure_patterns_exp_000021_000024.md), SHA-256 `42411a9a93199ee4f401ce7cb4227a48cfea7b28d0ebabfa55cb434ee7cf2b76`. Its source inventory links the four journals and leaderboards and fingerprints their bytes. No linked market dataset was opened. [Research Memory index](../memory/index.csv) was read only to reconcile history, not updated.

## Repository facts: implemented inventory and evidence status

The [research registry](../../app/research/backtester.py) exposes **13 strategy entries: 12 signal implementations and one empty template**. Counting code entries is not counting independent, validated hypotheses. In particular, early indicator baselines and later preregistered families overlap economically.

| Research entry | Implemented mechanism | Persisted evidence and status; no inferred approval |
|---|---|---|
| [baseline_trend](../../app/research/strategies/baseline_trend.py) | Long trend/EMA slope, RSI range, volume and ATR conditions | Implemented; no completed experiment explicitly linked to this strategy in the inspected journals/index. Outcome unverified. |
| [mean_reversion](../../app/research/strategies/mean_reversion.py) | Long RSI/EMA extension in elevated volatility | EXP-000001–002 describe the early sweep; [stored smoke leaderboard](../leaderboards/mean_reversion_smoke_v1.csv) first-row net PF 0.513603. Negative research result, not a newly assigned authoritative baseline verdict. |
| [bollinger_reversion](../../app/research/strategies/bollinger_reversion.py) | Long below lower Bollinger band with RSI, volume and ATR gates | Implemented, configuration exists; no completed journal/index result found. Do not call it successful or rejected by an unrun gate. |
| [ema_pullback](../../app/research/strategies/ema_pullback.py) | Long pullback between aligned EMAs with RSI/ATR/volume gates | Implemented; no completed journal/index result found. |
| [pullback](../../app/research/strategies/pullback.py) | Empty template returning no entry or exit signals | Scaffold only, not a tested trading mechanism. |
| [donchian_breakout](../../app/research/strategies/donchian_breakout.py) | Long prior-channel break with trend/volume/ATR conditions; quality and close-location variants | EXP-000003–015 record imported or completed research, not independent confirmations. [External 2026 validation](donchian_breakout_15m_external_oos_2026_v1.md) explicitly says **REJECT**; candidate PF 0.785140, 97 trades. [HIGH_VOL replication](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md) remains a separate frozen, untested registration, not a new approved trading filter. |
| [volatility_exhaustion](../../app/research/strategies/volatility_exhaustion.py) | Long re-entry into lower Bollinger band following an excursion | Implemented frozen family; [closure/status report](volatility_exhaustion_baseline_v1.md) says **NOT EXECUTED / NOT EVALUATED**, discovery dataset unavailable at that attempt. This is not a performance rejection. |
| [momentum_pullback_continuation](../../app/research/strategies/momentum_pullback_continuation.py) | Long EMA-aligned bearish pullback followed by bullish prior-high confirmation | Implemented and [preregistered](../hypotheses/HYP-MOMENTUM-PULLBACK-BASELINE-001.md). No matching completed journal/index entry or standalone result report found. Outcome unverified. |
| [regime_transition](../../app/research/strategies/regime_transition.py) | Long only on a transition into the official TREND_UP state | Implemented/frozen; [persisted report](regime_transition_baseline_v1.md) says not executed because the discovery CSV was missing. EXP-000021 later references a rejected long EXP-000020, but its journal/result is absent here. Record the discrepancy; do not reconstruct its metrics or override the available status report. |
| [short_regime_transition](../../app/research/strategies/short_regime_transition.py) | Short transition into TREND_DOWN | [EXP-000021](../journal/EXP-000021.md): BASELINE_REJECT / CLOSED_REJECTED; FEE_DOMINATED. No rescue authorized. |
| [short_momentum_pullback_continuation](../../app/research/strategies/short_momentum_pullback_continuation.py) | Short bearish EMA trend, bullish pullback and bearish continuation | [EXP-000022](../journal/EXP-000022.md): BASELINE_REJECT / CLOSED_REJECTED; FEE_DOMINATED. No rescue authorized. |
| [short_compression_breakdown](../../app/research/strategies/short_compression_breakdown.py) | Strict inside bar followed by close below that bar's low | [EXP-000023](../journal/EXP-000023.md): BASELINE_REJECT / CLOSED_REJECTED; FEE_DOMINATED. No rescue authorized. |
| [short_failed_breakout](../../app/research/strategies/short_failed_breakout.py) | Bullish breakout above frozen prior resistance followed by close back below | [EXP-000024](../journal/EXP-000024.md): BASELINE_REJECT / CLOSED_REJECTED. Journal failure field is blank; the historical report classifies the stored economics as NO_GROSS_EDGE. No rescue authorized. |

The eight later research-family mechanisms are Donchian, volatility exhaustion, long momentum pullback, long regime transition, short regime transition, short momentum pullback, short compression breakdown and short failed breakout. This grouping does not promote the early RSI/EMA variants into independent confirmations.

Runtime/paper implementations also exist outside the research registry:

| Implementation | Mechanism and status |
|---|---|
| [BreakoutTrendStrategy](../../app/strategies/breakout_trend.py) | Runtime EMA/ATR/price/volume signal logic, including short handling. It is not the same implementation as research Donchian; do not transfer Donchian research results to it. No production approval established. |
| [Live bot signal path](../../app/live_bot.py) | Runtime long breakout / optional short breakdown with trend, volume, candle-strength and volatility gates. A separate executable implementation, not another independently validated family or production authorization. |
| [RSIMeanReversionStrategy](../../app/strategies/rsi_mean_reversion.py) | Runtime RSI/EMA/ATR scoring/reversion variant. Implemented; no separately attributable completed research result found. |
| [MeanReversionAgent](../../app/agents/mean_reversion_agent.py) | Runtime EMA extension, volatility and volume-spike reversion logic. An existing overlapping mechanism, not an unexplored idea. No separately attributable performance approval found. |
| [StrategyAgent](../../app/agents/strategy_agent.py) | Legacy paper trend breakout using prior highs, SMA alignment, volatility and slope. Implementation exists; no independently established result in this inventory. |
| [MultiStrategyEngine](../../app/strategies/multi_strategy_engine.py), [HighVolEngine](../../app/strategies/high_vol_engine.py) | Regime routing/gating wrappers around existing signals, not new sources of economic edge or independently validated strategies. Regime detectors, risk/supervisor agents and the executor are infrastructure, not additional strategy families. |

The [optimizer-compatible Bollinger subclass](../../app/research/optimizer/grid_search.py) exposes tunable Bollinger/RSI/volume and exit parameters but retains the same mechanism; it is an implementation variant, not another independent family. Its source was inspected as text only; the optimizer was never imported or executed.

### History and persistence limitations

- Journals EXP-000001–015 and EXP-000021–024 are tracked. Journals EXP-000016–020 are absent; absence is not evidence that those experiments never happened.
- The index contains EXP-000001–015 and EXP-000021–023; EXP-000024 has a persisted journal and leaderboard but no index row. No row is added here. EXP-000001's older journal lacks the standard explicit ID header; the index supplies its association.
- Several earlier Donchian entries reuse the same leaderboard path; the current file is not an immutable per-run snapshot. Repeated rows/experiments must not be counted as independent replication. “Passing configurations” means optimizer eligibility, not a governance approval.
- EXP-000024's preregistration retains a future/reserved execution label while its journal records closure. Existing artifacts establish recorded results, not proof of the original authorization history. Documentation conflicts are not resolved by retesting.

## Repository facts: recurring failures in EXP-000021–024

Values below are quoted from the historical failure report; PnL is USDT and expectancy is USDT per trade. Rounding is retained, with no recomputation from market data.

| Experiment | Trades | Gross PF | Net PF | Gross expectancy | Fee expectancy | Net expectancy | Gross PnL | Fees | Net PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-000021 | 331 | 1.209526 | 0.971714 | 0.069345 | 0.079972 | -0.010627 | 22.953160 | 26.470819 | -3.517658 |
| EXP-000022 | 206 | 1.061365 | 0.845917 | 0.020718 | 0.079992 | -0.059273 | 4.267975 | 16.478293 | -12.210317 |
| EXP-000023 | 925 | 1.050758 | 0.827943 | 0.016459 | 0.079993 | -0.063534 | 15.224734 | 73.993910 | -58.769176 |
| EXP-000024 | 985 | 0.983832 | 0.771355 | -0.005301 | 0.080002 | -0.085304 | -5.221940 | 78.802089 | -84.024029 |

1. **No gross edge:** EXP-000024 loses before fees. Cheaper execution alone does not explain away that failure.
2. **Fees consume positive gross edge:** EXP-000021–023 have positive discovery-period gross expectancy below fee expectancy. None has demonstrated positive net edge.
3. **Insufficient observed win rate for payoff:** actual-minus-break-even win-rate gaps are -0.007054, -0.040694, -0.046499 and -0.064169. This is not the same as requiring a win rate above 50%; the corresponding payoff ratios are 1.277498, 1.279192, 1.156117 and 1.072783.
4. **Temporal instability:** positive/negative months are 6/6, 4/8, 4/8 and 2/10. Top-two positive-month PnL concentration is 0.659866, 0.676778, 0.883464 and 1.000000. The last two exceed the current 0.80 gate; do not select their favorable months after seeing these results.
5. **Overlap/suppression:** rates are 0.286638, 0.377644, 0.403995 and 0.434882. These are suppressed entries while occupied, not measured missed profits. More signals or positions do not establish more independent edge.

The evidence motivates evaluating a different revenue source or information channel. It does **not** demonstrate which new mechanism works, justify lowering fees in the old simulator, or authorize tuning a rejected family.

## Repository facts: costs, fills and acceptance

### Official research execution model

Sources: [research defaults](../../app/research/config.py), [official simulator](../../app/research/simulation.py), [short-family frozen protocol](../hypotheses/HYP-SHORT-FAILED-BREAKOUT-001.md).

- Default fee is `0.0004` of traded notional **per side**, or 4 basis points. With entry notional `N`, entry price `Pe` and exit price `Px`, quantity is `q = N / Pe`; fees are `N × 0.0004 + q × Px × 0.0004`. Approximately 8 basis points, or 0.08 USDT at 100 USDT notional when prices are close, is an illustration of this equation, not an independently measured tariff or exact constant per trade.
- Fixed entry notional is 100 USDT, without compounding. One position at a time; the simulator advances to `exit_index + 1`, suppressing intervening entry opportunities. Single-direction strategies support long or short, not a simultaneous multi-leg hedge.
- Entry is at the completed signal candle's close. Exit scanning starts on the next candle. Within a candle, stop loss is checked before take profit, then strategy exit; maximum holding defaults to the final permitted candle close. Same-candle entry exits are not modeled. Candle-level barrier fills assume execution at the stop/target price, including when real liquidity/gaps might prevent that fill.
- No modeled bid/ask spread, slippage, latency, queue position, partial fill, market impact, funding, borrow, collateral financing or liquidation. There is no maker/taker execution distinction. Accounting includes directional price PnL and explicit entry/exit fees only. This simplification is not an optimistic execution privilege for a new mechanism.
- Short EXP-000021–024 use the frozen 15m exits TP 0.012, SL 0.008, maximum 24 candles and no strategy exits. They are not parameters to alter here.
- [Runtime configuration](../../app/config.py) separately declares `SLIPPAGE_ESTIMATE=0.0002`, `FEE_RATE=0.0004`, `FEE_MULTIPLE_THRESHOLD=3.0` and `MIN_EXPECTED_MOVE_PCT=0.006`. These do not add slippage or those filters to the official research simulator. Do not mix the accounting paths or treat configuration values as observed costs.

For a hypothetical new mechanism, the necessary economic condition is expected gross revenue exceeding **all** entry/exit, spread, slippage, financing and operational costs at executable prices. This is accounting logic, not evidence of an edge. An execution/financing model extension would require separate authorization; none is made or implicitly approved by this assessment.

### Acceptance rules versus statistical evidence

The [authoritative baseline verdict function](../../app/research/governance/verdicts.py) checks in order:

1. Fewer than 100 completed trades: `INSUFFICIENT_SAMPLE`.
2. At least 100 trades, but gross expectancy <= 0, net PF <= 1, net expectancy <= 0, net PnL <= 0, or top-two positive-month concentration > 0.80: `BASELINE_REJECT`.
3. Otherwise: `BASELINE_CANDIDATE`. This is not production approval or proof of statistical significance.

Existing registrations are not perfectly uniform: [long momentum](../hypotheses/HYP-MOMENTUM-PULLBACK-BASELINE-001.md) and [volatility exhaustion status report](volatility_exhaustion_baseline_v1.md) describe a **single-month** concentration rule, while the shared function and later short-family protocols use **top two months**. Preserve each original protocol; do not silently harmonize it or reassign past verdicts.

The separate [Donchian HIGH_VOL replication](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md) requires at least 50 new candidate trades before evaluation and at least 6/8 registered directional associations plus positive economics and no material drawdown deterioration for success. Four or five directions with no economic failure is inconclusive; fewer than four or economic failure is not success. This is a protocol-specific gate, not a general sample-size shortcut.

[Statistical tooling](../../src/analysis/statistical_validation.py) includes a percentile 95% bootstrap expectancy interval, Shapiro and one-sample t-tests, permutation outputs and Monte Carlo distributions. Those tools' existence is not a verified significance result or a universally enforced p-value threshold. No statistical routine was executed here. The 100-trade gate does not address serial dependence, selection among many historical trials or regime concentration. Before future authorization, a protocol must specify independent sampling units, uncertainty estimation appropriate to clustered events, multiple-comparison treatment, a stopping rule and an untouched confirmation split. These are proposed design prerequisites, not new retroactive acceptance gates.

### Data governance

[Existing boundaries](../hypotheses/HYP-REGIME-TRANSITION-001.md) classify 2025 as discovery-used; 2026-01-01 through 2026-08-05 has already been used for validation/research elsewhere and is not a fresh discovery or confirmation holdout. Post-2026-08-05 remains reserved under the Donchian replication protocol. Calendar time passing does not release it. Switching to funding, another venue or order-book data does not automatically release a restricted date range. Any candidate needs a human-approved, nonconflicting data plan before collection, access or outcome inspection. This report does not establish that such data is available.

## Duplication and prohibited rescue map

| Tempting proposal | Why it is not a new mechanism in this repository |
|---|---|
| New RSI, EMA, Bollinger or ATR cutoffs; longer TP/holding to “beat fees” | Existing reversion, trend or pullback mechanisms with retuned thresholds/exits. No rescue of rejected baselines is authorized. Unexecuted families also remain frozen. |
| Filter short transitions by HIGH_VOL, ADX, volume or candle quality | Explicitly prohibited alterations to the frozen regime-transition hypothesis; does not supply an independent economic source of edge. |
| Different inside-bar nesting, range lookbacks or compression ratios | Alternatives were already discussed in the [compression preregistration](../hypotheses/HYP-SHORT-COMPRESSION-BREAKDOWN-001.md); repackaging after rejection risks selecting on the failed result. |
| Different resistance window or candle confirmation for failed breakouts | Same rejected short failed-breakout mechanism with altered definitions. Renaming it liquidity capture is not sufficient novelty. |
| Pick profitable months, change regime labels, increase concurrent trades | Selects historical winners or changes risk/overlap assumptions; does not establish a new causal mechanism. |
| Maker fee discounts, rebates or zero slippage added to old results | Changes costs/fills and requires separate data and execution authorization; is not evidence that an old hypothesis passed. |
| Invert long/short, replace fixed rules with a learned selector or ask an LLM to choose trades | A new wrapper/direction/model alone does not establish independent economics. Reusing known failed-family outcomes to train/select a rescue risks leakage. The research copilot is an evidence interface, not a strategy. |

## Speculative opportunities: conceptual candidates, not registrations

All frequency and regime descriptions below are **qualitative expectations**, not observed signal counts or performance forecasts. No thresholds, parameter grids, experiment IDs or target returns are assigned. Each candidate would require a new, human-reviewed protocol before implementation or data access. Independent information/compensation must remain the entry mechanism, rather than a filter attached to an old rejected rule.

### Candidate A — Inventory-neutral funding/basis carry

- **Economic mechanism:** compensation paid by leveraged perpetual demand may accrue to a hedged position (for example, owned spot against a short perpetual when the funding sign favors shorts). Basis change is a separate source of PnL/risk, not free yield. The direction/sign must follow the venue's actual settlement convention.
- **Cost-adjusted rationale:** financing receipts accumulated over a holding interval could exceed turnover costs without requiring a directional candle forecast. Necessary condition: expected funding receipts plus basis PnL exceed all four leg entry/exit fees, spreads, slippage, financing, rebalance and operational costs. Under the purely illustrative assumption that all four fills cost the existing 4 bps rate, fees alone are approximately 16 bps of one leg's notional. This is not a verified spot/perpetual fee schedule; returns must also account for capital and margin locked in both legs.
- **Novelty:** no inspected research strategy models funding cashflows, simultaneous spot/perpetual legs or basis convergence. It does not change RSI/EMA/regime/pattern rules. It changes the economic objective to carry and would expand beyond a scalping mandate; that strategic choice requires human approval.
- **Required data/indicators:** point-in-time published funding estimates and realized settlements with timestamps, spot/perpetual executable bid/ask and depth, mark/index prices, basis, hedge notionals, actual venue fee tiers, financing/borrow and margin rules. Settlement forecasts cannot be replaced with rates published later. Existing OHLCV summaries and the single-leg simulator are insufficient.
- **Expected frequency:** low turnover relative to intraday directional entries; holding may span multiple venue-defined settlements. No numerical frequency or sample sufficiency is known.
- **Potential regimes:** persistent leveraged positioning and liquid hedging markets; funding sign/persistence is a condition to investigate, not an established repository finding.
- **Main failure risks:** funding reversals, adverse basis moves, margin liquidation despite a delta hedge, spot/perpetual quantity drift, venue/custody risk, capital financing, expensive rebalancing and inability to execute both legs together.
- **Falsifiable predictions:** a prospective, frozen selection based only on funding information available before entry should yield funding-plus-basis revenue above the fully specified cost hurdle. The result should survive charging all legs and financing; if the margin disappears or depends on future settlements known only ex post, reject. A price-direction-only effect without financing compensation would not corroborate this mechanism.
- **Minimum evidence before authorization:** human agreement that carry fits scope; a documented sign/settlement convention; independently verified published-data timestamps and coverage; a venue-specific four-fill/financing accounting example; feasible margin and hedge constraints; a nonreserved data plan and a frozen sampling/uncertainty protocol. This is a feasibility dossier, not permission to measure market outcomes. A lower-frequency sample cannot be called sufficient simply by importing the old 100-trade rule without review.
- **Leakage/overfitting risks:** selecting pairs, venues or settlement windows after seeing carry profitability; survivorship of venues; using revised or realized funding as a pre-entry predictor; ignoring delistings and basis shocks; treating serial settlements in one position as independent trades. New fees/funding accounting requires separate authorized implementation, not a silent simulator change.

### Candidate B — Executable cross-venue price-dislocation convergence

- **Economic mechanism:** asynchronous price discovery across two independently observable venues may briefly leave a hedgeable executable price discrepancy. A synchronized cheap-venue long and rich-venue short tests convergence, rather than a single-market trend or reversal pattern.
- **Cost-adjusted rationale:** only a discrepancy at executable depth exceeding both legs' entry/exit fees, spreads, latency loss, financing and rebalancing costs could qualify. Four fills imply the same illustrative 16 bps fee burden if all cost 4 bps; actual tiers are unknown. Mid-price differences are not executable profit, and no such discrepancy has been established here.
- **Novelty:** relative cross-venue valuation and synchronization are absent from the single-venue OHLC entry rules. It must not become Donchian momentum on a second symbol, nor a post hoc choice of a historically leading venue.
- **Required data/indicators:** sequenced L2 quotes/depth and trade events from both venues; exchange and receipt timestamps, clock offsets, outages, symbol/contract mapping, executable cross-venue spread, inventory and latency estimates, financing and exact fee schedules. No new data is accessed here.
- **Expected frequency:** opportunities could be frequent but very short-lived; eligible opportunities after realistic latency and costs may be rare or zero. Counts and actionable capacity are unknown.
- **Potential regimes:** temporary venue-specific shocks while both venues remain operational and sufficiently liquid. Persistent price separation during capital controls or counterparty distress may be a risk premium rather than convergent mispricing.
- **Main failure risks:** stale/asynchronous books, one-legged fills, queue races, venue withdrawal restrictions, correlated outages, leverage and funding exposure, latency arbitrage competition. This is the highest execution-complexity candidate.
- **Falsifiable predictions:** after receipt-time alignment, conservative latency and executable-depth pricing, a preregistered dislocation signal should retain positive net convergence PnL. If it vanishes at executable quotes, reverses after synchronization, or requires zero latency/instant transfers, reject rather than tune timing retrospectively.
- **Minimum evidence before authorization:** documented lawful/nonreserved dual-venue data access and timestamp quality, contract comparability, pre-funded inventory/hedge feasibility, actual all-leg cost budget and a non-optimistic causal execution specification. Separate engineering authorization would be needed for any multi-leg simulator; no production capability is proposed here.
- **Leakage/overfitting risks:** aligning by future leader prices, optimizing venue pairs/lags after observing convergence, cherry-picking outages, ignoring failed/partial fills, or using exchange timestamps that were not observable at decision time. Overlapping dislocations must not be counted as independent evidence.

### Candidate C — Forced-liquidation flow exhaustion with observable liquidity replenishment

- **Economic mechanism:** forced deleveraging is an externally identified, potentially noninformational flow event. Subsequent observed replenishment of executable liquidity could mark exhaustion of that forced demand/supply and a temporary reversal. Continuing informed flow is an explicit competing explanation.
- **Cost-adjusted rationale:** event-driven forced price impact could, hypothetically, create a recovery larger than taker round-trip fees plus spread/slippage. No recovery magnitude or edge is observed here. An adequate gross displacement must survive executable entry after the event and replenishment become observable; neither the event extreme nor hindsight peak is a permitted fill.
- **Novelty:** the complete trigger is a timestamped liquidation event followed by observed book/trade-flow recovery, with trade direction derived from the forced side. It uses neither frozen resistance failure nor Bollinger re-entry/EMA pullback. A candle-wick proxy, or adding liquidation as a filter to EXP-000024, would collapse back into the rejected mechanism and does not satisfy this proposal.
- **Required data/indicators:** liquidation-event feed with delivery timestamps and documented aggregation/censoring; L2 book updates, aggressive trade side, spread/depth recovery and causal signed flow. Venue event semantics and dropped-message coverage are prerequisites; OHLCV alone cannot identify forced versus informed flow.
- **Expected frequency:** clustered, episodic opportunities rather than persistent bar-by-bar entries. Many feed messages may belong to one liquidation cascade; independent episode count is unknown and could be insufficient.
- **Potential regimes:** isolated leverage unwinds followed by recovering liquidity. Persistent fundamental repricing or impaired venue liquidity is a plausible failure regime, not a filter already calibrated to avoid losses.
- **Main failure risks:** catch-a-falling-knife losses, cascade continuation, stale/partial liquidation feeds, replenishment spoofing/cancellation, wide spreads, event dependence and adverse selection.
- **Falsifiable predictions:** a frozen, causally observable replenishment definition after forced-flow events should yield recovery sufficient to clear unchanged, explicit taker costs at delayed executable prices. If the apparent effect exists only at the liquidation low/high, disappears after feed latency, or is indistinguishable from generic candle reversal, reject the proposed mechanism.
- **Minimum evidence before authorization:** human-reviewed evidence that the feed distinguishes actual forced flow, a timestamp/coverage specification, a safe nonreserved data plan, an event-clustering definition chosen without outcomes, independent signal logic, conservative spread/slippage accounting and a preregistered confirmation/stopping plan. No feed collection or market analysis is authorized by this document.
- **Leakage/overfitting risks:** classifying cascades using their eventual endpoint, waiting for ex-post book recovery minima, selecting event sizes/holding windows to match past rebounds, choosing only surviving venues or successful recoveries, and counting correlated messages as independent observations.

## Speculative comparison and recommendation for human review

| Criterion | Candidate A: funding/basis carry | Candidate B: cross-venue convergence | Candidate C: forced-flow replenishment |
|---|---|---|---|
| Independent source of potential edge | Financing compensation and relative basis | Relative executable valuation across venues | Forced versus informed flow, followed by liquidity recovery |
| Distinct from existing families? | Strong conceptual separation; different cashflow/accounting objective | Strong separation only with synchronized executable cross-venue evidence | Conditional separation; high duplication risk if reduced to candles or added to failed-breakout rules |
| Cost hurdle | Four-fill fees plus financing/rehedging; amortization over holding is only a hypothesis | Four-fill fees plus latency/leg risk; likely severe implementation burden | Two-fill taker fees plus event spread/impact; recovery must remain after observation delay |
| Existing data and simulator sufficient? | No: missing financing/hedged accounting and verified point-in-time inputs | No: missing dual-venue event/latency and multi-leg execution model | No: missing verified liquidation/L2 event semantics and fills |
| Scope fit | Low turnover; explicit human decision on departing from scalping | Short horizon but substantial infrastructure/risk complexity | Closest to intraday research, but difficult causal event identification |
| Primary reason to stop before a market test | Carry cannot cover credible all-leg costs or scope is rejected | Dislocation is a timestamp/mid-price artifact or cannot be hedged | Feed cannot identify forced events causally, or novelty collapses to a rejected pattern |
| Repository empirical support | None for profitability; motivation comes only from historical cost failures | None for profitability; no observed executable dislocation | None for profitability; historical candle reversals do not validate this event mechanism |

**Recommend Candidate A for human review of scope and documentation feasibility first**, not for execution. Its explicit compensation channel and naturally lower turnover offer a clearer question about cost-adjusted revenue than another directional candle pattern. This is a research-design judgment, not a forecast that funding is persistently favorable or sufficient. Its two-leg capital and financing risks are material, and the scope change may reasonably be rejected. Do not substitute Candidate C automatically if A is declined; human review should select whether any new hypothesis is warranted.

The next authorized step should be a separately approved documentation/preregistration decision: confirm the economic question, available nonconflicting data, realistic cost accounting and falsification protocol. No candidate is ready for market testing from current evidence. Do not create an experiment ID, implement a strategy, change the simulator, collect data, inspect reserved periods or deploy capital based on this assessment.

## Documentation verification

Verification for this change checks local source links, registry coverage, transcribed historical metrics against persisted summaries, agreement with the baseline gate source, and a one-file diff. It reads metadata/source/report files only and imports no market-experiment modules. No application test, backtest, optimizer, statistical experiment or report regeneration is needed for this documentation-only change. The PR records actual verification outcomes; these checks are not evidence supporting any speculative opportunity.

Checks completed for this version: 45 local source links resolved; all 13 registry entries covered; 36 historical numeric fields matched the four stored leaderboard rows; the primary report SHA-256 matched; six baseline comparisons matched the gate source by static AST inspection; Markdown table widths and all ten requested dimensions for each of three candidates passed; only this assessment file was added. `git diff --check` passed. No market-dependent test or statistical analysis was run.
