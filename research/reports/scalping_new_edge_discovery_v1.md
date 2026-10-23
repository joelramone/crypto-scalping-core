# Scalping Research: New Edge Discovery V1

## 1. Executive summary

**Recommendation: NO_CANDIDATE_READY. Execution status: NOT AUTHORIZED.** Two economically distinct short-horizon mechanisms are assessed: **A, information in aggressive order flow relative to available liquidity**, and **B, compensation for supplying passive immediacy while bearing inventory/adverse-selection risk**. Both are speculative economic explanations, not implemented signals, preregistrations or demonstrated edges.

The repository supports candle research, not the point-in-time trade/book evidence or execution semantics needed to assess either mechanism realistically. There is no justified cost-adjusted basis for selecting one for feasibility review over the other today. This is an evidence-readiness decision, not proof that either mechanism cannot work. No expected return, win rate, Sharpe ratio, PF or trade frequency is assigned to a candidate.

## 2. Research baseline and constraints

Assessment date: **2026-10-23**. Baseline is fetched `main` at `5f51a10a5fd53eacd04954b90bd90216cc58e542`, the merge of **PR #184**, which includes TASK-009 (`e36a892`). The latest pasted TASK-010 request was read before work. This assessment uses repository source, documentation and filename metadata only; no exchange query or external market-data investigation was performed.

- [Strategy opportunity assessment](strategy_opportunity_assessment_v1.md) supplies the prior inventory and explicitly distinguishes economic ideas from evidence of profitability. Previously discussed ideas are not relabeled as discoveries here.
- [EXP-000021–024 failure report](failure_patterns_exp_000021_000024.md) remains authoritative for the recorded comparison: the four baselines remain BASELINE_REJECT / CLOSED_REJECTED. No rescue, cost reduction, favorable-month selection, threshold change or outcome-based reinterpretation is proposed.
- [Funding-carry decision gate](funding_carry_decision_gate_v1.md) remains **HOLD_PENDING_EXTERNAL_EVIDENCE**. This report does not reopen carry or release any of its gates.
- [Existing data boundaries](../hypotheses/HYP-REGIME-TRANSITION-001.md) classify 2025 as discovery-used and 2026-01-01 through 2026-08-05 as already used elsewhere, not a fresh confirmation holdout. Later data remains reserved under the [Donchian replication protocol](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md). No period is selected or accessed; a different data type does not automatically release reserved dates.
- [Governance verdict rules](../../app/research/governance/verdicts.py) and all preregistrations remain unchanged. The legacy sample/economic gates do not establish significance for a new clustered microstructure process. No experiment/hypothesis ID, trading rule, new indicator, account access, Research Memory write or capital deployment is authorized.

## 3. Existing strategy duplication map

The [registry](../../app/research/backtester.py) contains 13 entries: 12 signal implementations and one empty template. The following groups cover them all; group labels are descriptive, not new hypothesis IDs.

| Existing entries | Mechanism represented in source | Duplication boundary |
|---|---|---|
| [baseline_trend](../../app/research/strategies/baseline_trend.py), [donchian_breakout](../../app/research/strategies/donchian_breakout.py) | Persistent price trend/channel breakout with EMA, RSI, volume and/or volatility conditions; Donchian includes optional quality/close-location gates | A new price/volume cutoff, quality score or higher-resolution breakout is not an independent order-flow mechanism. |
| [mean_reversion](../../app/research/strategies/mean_reversion.py), [bollinger_reversion](../../app/research/strategies/bollinger_reversion.py), [volatility_exhaustion](../../app/research/strategies/volatility_exhaustion.py) | Long extension/reversion or Bollinger excursion/re-entry | Calling a wick or band re-entry “liquidity replenishment” does not make it observed book behavior or compensation for supplying immediacy. |
| [ema_pullback](../../app/research/strategies/ema_pullback.py), [momentum_pullback_continuation](../../app/research/strategies/momentum_pullback_continuation.py), [short_momentum_pullback_continuation](../../app/research/strategies/short_momentum_pullback_continuation.py) | EMA-aligned pullback/continuation; the frozen continuation variants use prior-candle confirmation | Trade flow added as a filter to rejected EXP-000022 would be a rescue. Candidate A must be an independent information/liquidity process, not that filter. |
| [regime_transition](../../app/research/strategies/regime_transition.py), [short_regime_transition](../../app/research/strategies/short_regime_transition.py) | Entry when the official regime changes into TREND_UP/TREND_DOWN | Reweighting regime thresholds or conditioning rejected EXP-000021 on volume/book proxies is not new economics. |
| [short_compression_breakdown](../../app/research/strategies/short_compression_breakdown.py), [short_failed_breakout](../../app/research/strategies/short_failed_breakout.py) | Strict inside-bar downside break; prior-resistance breakout then rejection | Replacing bar windows or calling rejected EXP-000023/024 depletion/exhaustion does not establish a causal event mechanism. |
| [pullback](../../app/research/strategies/pullback.py) | Empty entry/exit template | Absence of an implemented signal is not evidence of opportunity. |

Runtime breakout, RSI/EMA reversion and regime-routing variants are already catalogued in TASK-006; they are not separately validated sources of microstructure edge. Neither speculative mechanism below is implemented in the inspected research registry. That absence is only a capability finding, not evidence of exploitable returns. Two candidates are sufficient to compare distinct revenue sources; adding another reversal/pattern variant would inflate apparent novelty.

## 4. Current data and simulator capabilities

### Data: capability, presence and historical use are different

| Evidence class | Current repository finding | Limitation |
|---|---|---|
| Readers and schemas | [Research CSV reader](../../app/research/backtester.py) expects timestamp/OHLC/volume; [resampling helpers](../../app/research/data_utils.py) support 1m/5m/15m and optional aggregate quote volume, trade counts and taker-buy base/quote volume. [Runtime reader](../../app/data/real_data_loader.py) uses open_time and corresponding candle aggregates. | A schema or parser is not a verified dataset. Aggregated taker volume/trade counts are not sequenced prints, aggressor event timestamps, cancellations, queue depth or executable quotes. |
| Existing retrieval code, not executed | [Spot candle downloader](../../app/data/binance_downloader.py) and [research USD-M candle downloader](../../app/research/download_data.py) describe OHLCV retrieval. [Validator](../../app/research/data_validation/validate_ohlcv.py) checks candle structure/timestamps. | No downloader, reader or validator was invoked on market data. Candle validation cannot certify event-time book/print integrity. |
| Files actually present: filenames only | `data/` contains only `sample_trades.csv`; `research/datasets/`, including raw/processed, contains `.gitkeep` placeholders. `runtime/` is absent. Candidate-signal CSVs, leaderboards, regime/validation/Monte Carlo outputs and the memory index exist elsewhere under `research/`. | No CSV or dataset contents were opened for this assessment. The sample's schema/coverage is unverified; filenames do not establish raw market evidence. No L2/print archive was identified by filename inventory; this is not a claim about external storage. |
| Referenced raw files absent here | `data/BTCUSDT_1m.csv`, `data/binance_klines.csv` and the documented separate raw 2026 candle CSV are not present in this checkout. | Do not download replacements, infer their contents, or treat their absence as a market result. |
| Previously used history: documentation only | The protocols and [stored external-validation report](donchian_breakout_15m_external_oos_2026_v1.md#frozen-setup-and-data-boundary) identify 2025 discovery and partial-2026 candle research. | Historical use is established from documents, not by opening or reanalyzing those datasets. These periods cannot become untouched confirmation data. |
| Synthetic/runtime abstractions | [MarketStream](../../app/data/market_stream.py) cycles synthetic prices; [FeatureBuilder](../../app/data/features.py) derives price momentum; [TradesRepository](../../app/storage/trades_repo.py) stores strategy trade records. | None constitutes a historical public trade tape, order-event archive or model of executable liquidity. |
| Unsupported required evidence | Inspected data/research readers do not provide a historical L2/queue-event reader, sequenced aggressor prints or a latency/partial-fill reconstruction pipeline. Funding/liquidation event histories are also not supported by these readers. | Any such external dependency needs documentary proof of licensing, timestamp lineage, coverage, rule versions and authorized nonconflicting access before collection. Current live API availability does not prove historical reproducibility. |

The file inventory used directory entries only, including ignored paths under the checkout, without opening market datasets. No dataset is considered usable merely because its filename exists.

### Official execution semantics

Source: [official simulator](../../app/research/simulation.py) and [research fee default](../../app/research/config.py), inspected as text only.

| Dimension | What the simulator represents | Consequence for the proposed mechanisms |
|---|---|---|
| Signal availability | A strategy supplies a candle-indexed signal and entry is at that candle's close. | A full-candle aggregate is available only after completion. An intrabar observation cannot be silently assigned an earlier fill, and an event response may already be over by candle close. |
| Intrabar information/exits | Later candle high/low tests barriers; stop is checked before target, then strategy exit, with a holding-time close fallback. | High/low bounds do not reveal event order, quote availability or the path after a specific trade. Barrier-price fills are not passive queue fills. |
| Bid/ask, spread and slippage | No explicit executable bid/ask or implementation-shortfall model. | Mid/close prices cannot establish net entry/exit costs. Spread capture cannot be manufactured from a candle touching a quote. |
| Latency | No observation, transport, decision, order-entry or cancellation latency. | Contemporaneous price impact must not be treated as harvestable after observing the flow. |
| Partial fills / impact | Fixed notional, full fills; no queue, inventory-order lifecycle or size-dependent market impact. | Cannot validate aggressive execution through depth, passive fill probability, cancellations or forced inventory unwind. |
| Position overlap | One position at a time; advances to exit index + 1. | Overlapping signals/episodes are suppressed. Multiple orders/inventory and clustered events are not represented by independent candle trades. |
| Fees / turnover | Fixed entry notional of 100 USDT; fees charged on entry notional and actual exit-value notional using default 0.0004 per side. No maker/taker distinction. | New maker fee tiers/rebates cannot be assumed, and faster turnover is not additional independent edge. Existing costs stay unchanged. |

The separate [runtime configuration](../../app/config.py) declares a slippage estimate, but that does not add slippage to the official research simulator. **Both candidates require separate engineering authorization for an appropriate execution model. Neither can be evaluated faithfully by keeping these semantics unchanged.** No extension is designed or implemented here.

## 5. Candidate mechanisms

Everything in this section is a **speculative hypothesis**, not a repository result. Horizon and signal frequency are unknown. There are no thresholds, windows, indicators, entry/exit parameters or new registrations.

### Candidate A — Aggressive-flow information relative to available liquidity

1. **Mechanism:** sustained directional demand, including the possibility of split orders, may consume opposing liquidity before prices fully adjust. The proposed information source is observed aggressor trades and contemporaneously available depth/replenishment, not the price pattern that follows.
2. **Why an edge might exist:** price adjustment could persist beyond the time the flow is observable, leaving a hypothetical post-observation executable advantage. No such persistence or cost margin is established here.
3. **Why it could disappear:** quotes can adjust before a delayed observer acts; competing liquidity suppliers/fast traders can price the signal immediately; apparent demand can reverse, be hidden or be uninformative. High gross contemporaneous correlation is insufficient.
4. **Independence:** it would test a causal flow/liquidity process without requiring EMA alignment, regime transitions or a breakout. Adding a taker-volume gate to a rejected strategy, or reconstructing imbalance only from a favorable candle, would fail this novelty condition.
5. **Required information:** timestamped trade prints/aggressor classification, sequenced L2 snapshots/deltas, cancellations/replenishment information to the extent observable, matching-engine and receipt timestamps, venue rule versions and executable prices at size.
6. **Repository support:** candle aggregates can preserve total taker-buy volume but not the necessary event sequence. No usable print/L2 archive or corresponding reader is established by this checkout. This is an explicit missing external dependency.
7. **Execution/cost assumptions:** prices available after observation and order latency; both-side taker fees if crossing, spread, depth consumption, impact and partial fills. No zero-latency or pre-impact entry is allowed. Requires separate data and execution-engineering authorization.
8. **Leakage/overfitting risks:** using future prints, revised aggressor labels or later depth to classify a decision; timestamp misalignment; selecting aggregation windows, venues or labels after seeing PnL; counting overlapping demand episodes as independent observations.
9. **Falsifiable prediction:** under a future frozen causal observation protocol, subsequent executable signed price response should persist beyond observation/execution delay and exceed the full cost burden on untouched confirmation evidence. A response restricted to the contemporaneous move, or one erased by executable costs, would reject the mechanism as a tradable edge.
10. **Invalidate further investigation:** no lawful point-in-time print/book provenance; no identifiable post-observation information; inability to bound delays/costs; or reliance on an old rejected signal as the actual trigger. These are stop conditions to assess with authorized evidence, not findings from a new test.

### Candidate B — Passive immediacy provision with inventory/adverse-selection compensation

1. **Mechanism:** liquidity demanders may pay a quoted spread to a supplier willing to carry inventory and execution uncertainty. Hypothetical gross revenue comes from providing immediacy, not forecasting a reversal or receiving an assumed rebate.
2. **Why an edge might exist:** compensation for this service/risk could exceed adverse price moves and inventory disposal costs. This is a distinct compensation channel; no positive net surplus is demonstrated.
3. **Why it could disappear:** informed flow can select stale quotes, faster competitors can dominate queues, spreads can compress, and unfavorable fills or emergency inventory liquidation can consume all quoted spread.
4. **Independence:** earning liquidity-service compensation is different from buying an RSI/Bollinger extension or selling a failed breakout. Passive entry into a rejected directional pattern would still be the old hypothesis, not this mechanism.
5. **Required information:** sequenced L2 and trade events, order arrival/cancellation timestamps, queue-priority rules and sufficiently conservative queue inference, actual maker/taker fee rules and inventory/hedging constraints. L2 alone may not identify individual queue position; level-3/order information or defensible uncertainty bounds may be needed.
6. **Repository support:** no historical queue/order-event archive, passive execution model or validated fill lineage was found. OHLCV and strategy trade logs cannot reconstruct those inputs. The feasibility of any external source remains unknown.
7. **Execution/cost assumptions:** fills only after arrival and actual available priority/quantity; partial fills, adverse selection, cancellation delays, inventory carrying/financing and taker exits/hedges. No “price touched, therefore filled,” automatic two-sided fill, fee discount or rebate assumption. Requires separate engineering authorization.
8. **Leakage/overfitting risks:** knowing the eventual queue outcome, conditioning quotes on future fills/markouts, discarding unfilled orders or orphaned inventory, assuming cancellation precedes an adverse trade, selecting quote/holding parameters from realized winners.
9. **Falsifiable prediction:** for a future frozen inventory and fill protocol, earned spread on eligible actual/conservatively inferred fills should exceed fees, adverse-selection markouts and inventory unwind costs on untouched confirmation evidence. A result depending on touch fills, invented rebates or exclusion of losing inventory would invalidate that claim.
10. **Invalidate further investigation:** no lawful data capable of bounding queue uncertainty; no credible inventory completion/unwind treatment; compensation inseparable from optimistic fills; or a design whose entry is merely a rejected reversion rule. Risk compensation is not an entitlement to profit.

## 6. Cost-first feasibility comparison

**Historical context, not new candidate performance:** the stored failure report records positive gross expectancies below fee expectancy for EXP-000021–023 and negative gross expectancy even before fees for EXP-000024. All remain rejected. Positive gross economics can be too small, and reducing assumed fees cannot repair absent gross edge. Those observations do not estimate either candidate's payoff or attainable costs.

| Cost distinction | Candidate A | Candidate B |
|---|---|---|
| Economic source | Potential information in aggressive demand relative to liquidity | Potential compensation for supplying immediacy and inventory capacity |
| Hypothetical gross advantage | Post-observation directional repricing | Earned spread before adverse selection and inventory costs |
| Required transaction costs | Entry/exit fees, crossed spread, depth/impact, partial execution and unwind | Applicable maker fees, taker inventory exits/hedges, carrying/financing and rebalancing |
| Unknown execution costs | Latency decay, available size, slippage and response to own trading | Queue uncertainty, unfilled/cancelled orders, adverse selection and terminal inventory |
| Cost-first plausibility | Economically interpretable, but no evidence response survives full aggressive costs | Economically interpretable, but a quoted spread does not establish realizable compensation |
| Net profitability | Unverified; no magnitude assigned | Unverified; no rebate, positive fill rate or net margin assumed |

Qualitative ratings below describe evidence/requirements, not invented scores or performance rankings.

| Comparison criterion | Candidate A | Candidate B |
|---|---|---|
| Mechanism independence | Distinct conditional information source; duplication risk if reduced to a momentum/volume filter | Distinct liquidity-service compensation; duplication risk if reduced to passive reversion entry |
| Data availability | Missing verified print/L2 dataset and reader; aggregate schema is insufficient | Missing verified queue/order-event/fill evidence and reader; L2 may itself be insufficient |
| Research reproducibility | Unestablished event lineage, aggressor semantics and receipt times | Unestablished priority, cancellation and inventory outcome lineage |
| Current infrastructure compatibility | Low: candle infrastructure cannot recover event-time execution | Low: no passive order lifecycle/queue/inventory execution model |
| Execution realism | Unresolved post-observation executable advantage | Unresolved fill probability/adverse selection and inventory costs |
| Expected engineering burden | High, qualitatively; separate authorization needed | High with additional queue/inventory uncertainty; separate authorization needed |
| Leakage exposure | High: contemporaneous response, clock alignment and overlapping events | High: hindsight fills, cancellation timing and discarded inventory |
| Risk of duplicating rejected strategies | Material unless direct causal flow is the independent mechanism | Material unless compensation is for liquidity supply rather than an old price trigger |
| Falsifiable preregistration | A causal prediction can be articulated, but data/latency/cost specification is missing | A compensation prediction can be articulated, but defensible fills and inventory accounting are missing |
| Untouched confirmation plan | Not established; no authorized independent event dataset or period | Not established; no authorized independent queue/event dataset or period |

## 7. Candidate recommendation or no-candidate decision

**NO_CANDIDATE_READY.** The repository does not establish an independent gross advantage surviving its required costs for either mechanism, nor the reproducible inputs and realistic execution necessary to investigate it. Selecting a preferred candidate solely from a plausible story or ease of aggregating existing candles would not be evidence-based prioritization. Absence of experiments is not proof of opportunity, and absence of current inputs is not proof that the mechanism can never work.

No candidate is recommended for implementation, testing or preregistration. This decision does not reopen funding carry or any rejected strategy. It also does not demand a profitable backtest before documentary feasibility can be reconsidered; material evidence about lawful data, causal observability and conservative execution would be the appropriate first prerequisite.

## 8. Missing evidence and critical blockers

- **Data provenance/access:** documented rights, schema, historical coverage, snapshot/delta completeness, trade/aggressor semantics, event and receipt timestamps, corrections and delistings. No external provider or current live API is assumed to supply the missing history.
- **Execution/cost specification:** effective fee schedules, executable prices at size, delays and impact/partial-fill assumptions; for B, queue-priority uncertainty, cancellation and complete inventory accounting. No new numerical threshold, favorable fill or maker discount is invented.
- **Governance/confirmation:** human-approved independent sampling units and a genuinely untouched, nonconflicting confirmation plan before any collection/outcome access. Reusing 2025/partial-2026 candle results or selecting on reserved dates is not confirmation. Overlapping events and dependent fills cannot be treated automatically as independent trades.
- **Separate engineering permission:** both mechanisms require a suitable event/execution model and readers. Current simulator limitations are documented constraints, not permission to alter that simulator or deploy a collector.

## 9. Proposed next human decision

Decide whether to authorize **a bounded documentation-only feasibility check of lawful historical print/book/queue data and conservative execution observability, without downloads, queries, period selection or code**, or to leave this discovery track parked. That decision should establish whether either mechanism can be specified causally before choosing a candidate. It does not authorize building infrastructure, initiating a collection, assigning an experiment ID, preregistering or measuring market outcomes.

Keep funding carry on HOLD_PENDING_EXTERNAL_EVIDENCE and EXP-000021–024 closed/rejected. Stop after this report; subsequent research action requires separate human authorization.

## 10. Documentation verification

Checks cover local source references, all registry entries, source-level data schemas, filename-only presence/absence, simulator timing/fill/overlap/fee semantics, distinct candidate revenue sources and all requested comparison dimensions. The three principal prior reports are compared byte-for-byte with the baseline; only this Markdown file may be added. No strategy module is imported or executed, no market dataset/CSV is opened, and no application test, simulation, optimizer, external market API or report-regeneration routine is run. `git diff --check` must pass.

Completed verification: 33 local file/heading references resolved; all 13 registry entries were covered by static source inspection; file-presence claims were checked from metadata without opening datasets; candle schema, fee and simulator timing/overlap statements matched source. Both candidates cover all ten requested dimensions and the comparison covers all ten criteria. Prior principal reports remained byte-identical, Markdown tables were consistent, only this report was added, and `git diff --check` passed. No application tests or research simulations were executed.
