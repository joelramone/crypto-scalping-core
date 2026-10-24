# Microstructure Data & Execution Feasibility Gate V1

## 1. Executive summary

**Decision: HOLD_PENDING_EXTERNAL_EVIDENCE. Gates A, B and C: UNRESOLVED. Execution status: NOT AUTHORIZED.** Candidate A, aggressive order flow relative to available liquidity, is the documentary priority; passive liquidity provision is a secondary comparison, not a selected strategy.

Three source options were examined: **Binance public archives plus official stream specifications; Bybit official market-data specifications; Tardis.dev historical replay documentation/source**. Binance documents archived individual trades but does not establish the required historical L2/receipt-time package. Bybit documents live trades/books and recent trades, not an inspected full historical archive specification. Tardis documents historical trade/L2 replay, but product-specific coverage, capture provenance, completeness, entitlement and licensing remain insufficiently established. No option currently satisfies the entire paired trade/book requirement through the inspected evidence.

This is a research-feasibility decision, not a profitability assessment or proof of impossibility. No dataset, sample or market-data endpoint was accessed. No data acquisition, account creation, subscription, terms acceptance, engineering, experiment or preregistration is authorized.

## 2. Research baseline and boundaries

Baseline: fetched `main` at `a5084b18415ecb086334c195724bfaaf1527fa8d`, confirming TASK-010 merged through **PR #185**. Review sources: [TASK-010](scalping_new_edge_discovery_v1.md), [funding-carry decision](funding_carry_decision_gate_v1.md) and [recorded failure analysis](failure_patterns_exp_000021_000024.md), all preserved unchanged. Candidate A's causal information hypothesis is distinct from a new indicator threshold; this task does not investigate its returns or specify trading rules.

**BTCUSDT is a scope reference only**, not a venue, product or test-period selection. Spot and linear-perpetual representations must not be conflated. EXP-000021–024 stay closed/rejected; no rescue or retuning. Funding carry remains HOLD_PENDING_EXTERNAL_EVIDENCE. [Existing data restrictions](../hypotheses/HYP-REGIME-TRANSITION-001.md) and the [reserved Donchian protocol](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md) remain binding: discovery-used 2025 and already-used partial 2026 are not fresh confirmation; post-2026-08-05 remains reserved. Another data type/provider does not release reserved dates. No new dates, hypothesis/experiment IDs or Research Memory entries are created.

**Method:** source/document text only, with no SDK import, data example execution, dataset-file opening, historical sample, exchange request, WebSocket connection or provider data-API request. Four documentation/pricing pages were proxy-blocked; separately accessible publisher-owned GitHub sources were inspected at pinned commits. No restriction bypass or network configuration change was attempted. Missing descriptions are not feature unavailability.

## 3. Candidate A data requirements

| Required evidence | Minimum documentary requirement | Finding from inspected sources |
|---|---|---|
| Individual executions and aggressor | Nonaggregated trade IDs, price/quantity, taker side or an unambiguous maker-side mapping, with unknown classifications retained | Binance archive documents individual trades separately from aggTrades [B1]; Bybit explicitly defines taker side [Y1, Y2]; Tardis Trade type permits buy/sell/unknown aggressor [T2]. Product-specific completeness not demonstrated. |
| Event and receipt times | Timestamp meaning/unit, matching/exchange event time and independently identified capture-receipt time, clock calibration/uncertainty and batching | Native feeds document event fields; Tardis types/replay expose timestamp/localTimestamp [T2, T3]. A field name does not prove capture location, clock accuracy, publication lag or the future researcher's observation time. |
| Historical L2 snapshots and deltas | An aligned initial state and all relevant updates, bid/ask prices and absolute quantities, stream/schema versions | Binance/Bybit live reconstruction is documented [B3, Y3, Y4]. Tardis advertises historical snapshots/incrementals [T1]. Historical coverage of the entire intended trade/book combination remains unestablished. |
| Sequence and recovery | Applicable sequence rules, resets, duplicates/out-of-order treatment, gap indicators, resnapshot semantics and explicit unusable intervals | Native specifications help define handling, but do not establish a historical gap-free capture. Tardis normalized BookChange type does not require native sequence IDs; exchange-native lineage must be verified separately [T2]. |
| Instrument rules | Base/quote/settlement identity, tick/lot constraints and effective historical changes | Current Binance/Bybit filter metadata is documented [B4, Y5]. A time-versioned rule archive is not established; current filters cannot be retroactively assigned to historical events. |
| Corrections, retention and rights | Exact symbol/product/channel coverage, timestamped correction/version policy, immutable identifiers/checksums, retention and permitted research use | Binance documents archive checksums and replacement history, with explicit warranty limits [B1, B2]. No option provides a verified complete entitlement/provenance package for this investigation. |

Historical archived events, capturable live feeds, derived candles and provider feature claims are separate evidence classes. A recent-trades response is not a deep archive; a sampled book snapshot is not a complete delta stream; taker-volume candles are not individual executions. Documentation of a possible collection route is not permission to start prospective capture.

## 4. Historical data-provider comparison

Statuses apply to the **stated scope**: VERIFIED_DOCUMENTARY means the specific documentary statement is supported, not that a dataset was validated; PARTIALLY_VERIFIED means a material requirement remains; UNKNOWN is unestablished. DOCUMENTED_UNAVAILABLE is reserved for an explicit source statement, never silence, an inaccessible page or a failed file lookup. No required archive feature is classified unavailable merely because it was not verified.

### Binance public archives and native specifications

| Requirement | Evidence | Source | Status |
|---|---|---|---|
| Historical trades | Public-data README explicitly separates individual trade archives from aggregated trades and candles, including spot and USD-M formats | B1, Data Information | VERIFIED_DOCUMENTARY |
| Aggressor classification | Individual trade schema includes isBuyerMaker. For that matched trade, buyer-as-maker implies seller-as-taker; this is not trader intent or a cross-product identity | B1; B3, Trade Streams | VERIFIED_DOCUMENTARY |
| Historical L2 snapshots | Terms mention order-book metrics generically; inspected archive README does not establish complete price-level snapshot coverage | B1, B2 | UNKNOWN |
| Historical L2 deltas | Live diff-depth updates are specified, but no complete historical delta archive was established from these documents | B3 | UNKNOWN |
| Event timestamps | Archive schema includes trade time and warns of microsecond spot timestamps from 2025 onward; live timestamp units are separately documented; scope/limit: event fields known; historical receipt/clock lineage unknown | B1, B3 | PARTIALLY_VERIFIED |
| Sequence reconstruction | Live spot procedure aligns snapshot lastUpdateId with buffered U/u, discards stale updates and restarts after a detected gap; it warns about finite snapshot depth; scope/limit: live rule, not historical capture proof or USD-M rule | B3, local-book procedure | PARTIALLY_VERIFIED |
| Retention and coverage | README states daily/monthly archives and broad symbol coverage; corrections can replace files. Terms disclaim completeness/timeliness; scope/limit: exact applicable product/channel history and missingness unknown | B1, B2 §§2, 10 | PARTIALLY_VERIFIED |
| Research-use permissions | Dataset Terms grant conditional CC BY-NC-SA 4.0 noncommercial use, including described personal nonproduction research; commercial use needs separate written licensing; scope/limit: applicable full terms and project-use classification require human review | B2 §§3–4 | PARTIALLY_VERIFIED |
| Access and pricing | Terms explicitly describe dataset access free of charge subject to conditions; public archive route is documented. No data access/authentication test was performed; scope/limit: for conditional free-access statement only | B1; B2 §2.3 | VERIFIED_DOCUMENTARY |
| Reproducibility | Checksums/replacement history provide a versioning route; unknown historical L2/receipt times and incomplete capture prevent end-to-end reproduction | B1, CHECKSUM/Updates | PARTIALLY_VERIFIED |

The README's “MIT” label is not a substitute for its separately linked **Dataset Terms**. Those terms state that commercial/live proprietary use is not granted under the noncommercial arrangement and disclaim data completeness. This task only read documentation; it did not access/utilize a dataset or accept terms. These conditional rights do not establish a fatal incompatibility for documentary/noncommercial research, nor grant future production rights. Generic “order book metrics” wording does not prove L2 snapshots/deltas.

### Bybit official specifications

| Requirement | Evidence | Source | Status |
|---|---|---|---|
| Historical trades | Recent-public-trades interface and real-time trade stream documented; no inspected complete historical archive specification; scope/limit: recent/live, not established historical retention | Y1, Y2 | PARTIALLY_VERIFIED |
| Aggressor classification | REST side and WS S are explicitly “Side of taker”; WS trades include match time and may share a seq across messages; scope/limit: for field semantics | Y1, Y2 | VERIFIED_DOCUMENTARY |
| Historical L2 snapshots | REST book is a current snapshot; WS begins with/reset-to snapshots. No historical snapshot product coverage verified; scope/limit: for historical requirement | Y3, Y4 | UNKNOWN |
| Historical L2 deltas | WS delta semantics exist; no historical incremental archive documented by the inspected sources; scope/limit: for historical requirement | Y4 | UNKNOWN |
| Event timestamps | Trade T/time, system ts and book matching-engine cts are defined; scope/limit: no archived local receipt/clock guarantees | Y1–Y4 | PARTIALLY_VERIFIED |
| Sequence reconstruction | Book u/seq and reset rules described; zero quantity deletes a level. Multiple trade messages may use the same seq, so contiguity cannot be invented; scope/limit: live rules do not certify archived gap detection/recovery | Y2, Y4 | PARTIALLY_VERIFIED |
| Retention and coverage | Recent REST limits and live product/channel specifications do not define complete historical retention; scope/limit: for required archive | Y1–Y4 | UNKNOWN |
| Research-use permissions | No inspected data-use agreement establishes the required archive/research/redistribution rights | Y1–Y5 | UNKNOWN |
| Access and pricing | Public live/recent market-data documentation is accessible as source; archive entitlement, paid/free availability and pricing not established; scope/limit: for historical access/cost | Y1–Y4 | UNKNOWN |
| Reproducibility | Live reset/update procedures are described; RPI orders are excluded from visible book. A complete licensed archive with receipt times/corrections is unverified | Y3, Y4 | PARTIALLY_VERIFIED |

The trade specifications identify RPI trades while the documented visible book excludes RPI orders [Y1–Y4]. A future flow-versus-depth study cannot assume every printed trade consumed displayed liquidity; scope and reconciliation must be specified without hindsight.

Prospective capture is only a possible future option. It is neither authorized nor a solution to historical reconstruction by itself. Public feed visibility does not establish lawful archival use, complete liquidity, reliable future capture or trading eligibility.

### Tardis.dev historical replay option

| Requirement | Evidence | Source | Status |
|---|---|---|---|
| Historical trades | Publisher README advertises tick-level historical trade replay in native and normalized formats; scope/limit: for advertised route, not exact product coverage | T1 | VERIFIED_DOCUMENTARY |
| Aggressor classification | Trade schema defines liquidity-taker side buy/sell/unknown, distinct from maker side; scope/limit: mapping/unknown rates and historical channel correctness unvalidated | T2, Trade | PARTIALLY_VERIFIED |
| Historical L2 snapshots | Publisher explicitly advertises full-depth snapshots with historical replay; scope/limit: specific archive coverage/initial state and guarantees unconfirmed | T1 | PARTIALLY_VERIFIED |
| Historical L2 deltas | Publisher advertises incremental updates; BookChange has isSnapshot, bids and asks; scope/limit: exact native gap lineage and channel completeness unconfirmed | T1, T2 | PARTIALLY_VERIFIED |
| Event timestamps | Normalized types expose timestamp and localTimestamp; replay parser reads capture timestamps and preserves microsecond information in its implementation; scope/limit: capture location, clock discipline, delay and applicability not established | T2, T3 | PARTIALLY_VERIFIED |
| Sequence reconstruction | SDK advertises OrderBook reconstruction. Normalized BookChange has no required native-sequence field; raw/native replay may retain it. Client warns about initial normalization snapshots at UTC day start; scope/limit: source warning is not a universal dataset SLA or proof of recovery | T1–T3 | PARTIALLY_VERIFIED |
| Retention and coverage | General historical replay capability advertised; current catalogue/retention and exact BTCUSDT product/channel overlap not verified; scope/limit: for intended research slice | T1; H3 inaccessible | UNKNOWN |
| Research-use permissions | Client/server code availability does not establish provider/exchange dataset-use and redistribution rights | T1–T3; H3 inaccessible | UNKNOWN |
| Access and pricing | Client exposes API-key configuration; pricing page and detailed access terms inaccessible. No key, account or subscription used; scope/limit: configurable authentication; full/free/paid entitlement and price UNKNOWN | T1, T3; H4 | PARTIALLY_VERIFIED |
| Reproducibility | Native/normalized historical replay and local caching are documented capabilities. Immutable dataset revisions, corrected/original views and reproducible catalogue versions unestablished | T1–T3 | PARTIALLY_VERIFIED |

This is the most explicit **inspected historical trade-plus-book route**, not a selected vendor, recommended purchase or proven complete dataset. No free sample policy, subscription tariff, licensed data right or completeness guarantee is assumed. No server, package, container or SDK was installed/run; only its documentation and type/parser source were read. A client parser exposing localTimestamp does not prove that it equals the future researcher's receipt time.

## 5. Point-in-time reconstruction feasibility

The target is the information available **before a decision**, not a retrospectively perfect book. Event time orders exchange activity; capture receipt time represents a particular observer/location. Neither alone establishes the other, and provider capture is not automatically the researcher's future latency path. Clock skew, precision, batching and delayed publication need stated uncertainty rather than an assumed universal timestamp ordering.

Minimum provenance for a valid future specification:

- Provider/source identity, instrument/product/channel identity, schema and effective rule versions, timezone/units, exchange event time, system-publication time where defined, and capture receipt time with location/clock discipline and uncertainty.
- Raw message/trade identifiers and native sequence bounds, exact starting snapshot, snapshot alignment, reset/disconnect markers, duplicate/out-of-order handling, gaps and resynchronizations. Never interpolate missing depth and call it observed executable liquidity. A later snapshot cannot fill an earlier decision's missing state.
- Explicit trade/book synchronization rule accounting for different clocks/channels and batch order; a shared or nearby timestamp does not establish causal order. Bybit trade seq reuse must not be mistaken for missing or duplicate messages automatically.
- Original versus corrected archive view, capture/publication/version dates, immutable checksums/manifests, and complete coverage/missingness declarations. Binance explicitly permits later archive replacements [B1]; correction information available only later must not silently become information observable at the historical decision.
- A reproducible, authorized access/license specification and a decision-availability rule that excludes future prints, delayed deltas, revised instrument constraints or hindsight aggressor labels. Clock/latency choices, sampling units and confirmation governance must be fixed before outcomes, not selected to improve results.

Live reconstruction specifications can inform a future protocol, but no inspected option yet establishes that complete paired historical lineage. Tardis source warns that its book-normalized replay starting mid-day may lack initial snapshots [T3]; successful parsing alone does not prove a valid starting book. Exact receipt-time semantics, gaps, source corrections and coverage need provider confirmation before any proposed data acquisition. No historical period is selected here.

## 6. Execution-model gap assessment

The [official simulator](../../app/research/simulation.py) enters at signal-candle close, checks later candle barriers with stop-before-target priority, assumes complete fixed-notional fills, and advances to exit index + 1. Its [fee default](../../app/research/config.py) is 0.0004 per side without maker/taker distinction. It has no explicit event-time book, bid/ask, latency, partial fill or impact model. [Candle readers/resampling](../../app/research/data_utils.py) can retain aggregate taker volume/counts, not the event sequence required here. Existing reported fee failures motivate full costs, not reduced historical fees or a new profitability assertion.

For Candidate A, minimum **specification requirements**, not a simulator architecture or implementation plan, are:

| Requirement | Conservative execution condition | Current blocker |
|---|---|---|
| Causal observation | Signal sees only messages received/available before decision, with incomplete-state refusal explicitly specified | Historical receipt/clock provenance and paired synchronization unverified |
| Submission latency | Separately identify observation, computation, transmission and exchange arrival delays; no pre-impact fill | No calibrated or defensibly bounded arrival assumptions |
| Executable bid/ask/depth | Use opposing book at order arrival, not mid/close; respect tick/lot/effective instrument rules | No confirmed aligned executable historical depth/rule versions |
| Marketable/partial fills | Account for finite available quantity, price levels, residual order size, rejection and time-in-force; no unlimited fill | Candle full-fill semantics cannot represent this |
| Slippage/market impact | Distinguish spread, implementation shortfall and own trading effects without double charging actual fill costs | Displayed L2 is cancellable and historical replay lacks the counterfactual effect of one's own orders |
| Maker/taker fees | Complete entry, exit, residual/unwind costs under applicable effective fee schedules; no assumed discount/rebate | Venue/account/use-specific schedules not established by this review |
| Exit and inventory completion | All residual exposure, failed/unfilled/rejected orders and terminal unwind included; no censoring orphaned inventory | No inventory/order lifecycle or defensible emergency fill evidence |
| Deterministic replay | Fixed source versions, ordering/tie rules, reset/gap treatment and declared assumptions must reproduce the same state | Provider replay/software availability alone is insufficient proof of historical source determinism |
| Incomplete evidence | Exclude/abstain or bound an unusable episode explicitly; do not invent depth, zero latency, certain fills or favorable prices | No authorized uncertainty policy or causally valid data package yet |

Observed depth does not guarantee all of it is available at arrival. A replay without the strategy's own orders is not a validated market-impact counterfactual. These unknowns prevent declaring execution observability sufficient. They do not prove that an appropriately bounded future model is impossible. **Separate engineering authorization would be required**; current candle semantics cannot validate Candidate A unchanged. No implementation, delay threshold or order rule is proposed.

## 7. Candidate B secondary comparison

Passive liquidity provision additionally needs queue-priority rules, order arrival/acknowledgment, partial fills, cancellation arrival, adverse selection, inventory risk and forced taker exits. Price-level L2 does **not** reveal exact queue position, hidden orders, individual cancellations or the counterfactual priority of a hypothetical order. Order-level data may help but is not a guarantee of perfect observability; defensible uncertainty bounds would still be needed.

No inspected source establishes the complete order-level/cancellation/priority lineage needed for exact passive fills. A candle touch or L2 price match is not a fill, and simultaneous spread capture cannot be assumed. Candidate B therefore has additional unresolved observability requirements relative to A. It is not the selected research priority or an automatic fallback. No quoting, inventory or trading parameters are assigned.

## 8. Three-gate decision matrix

PASS means sufficiently established for the stated documentary planning gate, not empirically validated or authorized. BLOCKED requires a demonstrated critical incompatibility. Missing information, an inaccessible page or absent archive description is UNRESOLVED.

| Gate | Evidence supporting a possible route | Critical unresolved condition | Classification |
|---|---|---|---|
| A — Data provenance | Binance documents archived individual trades and correction/checksum practices; Tardis documents historical trade/L2 replay; native Binance/Bybit feeds document reconstruction mechanics | No inspected provider establishes complete applicable paired trade/book coverage, original/corrected versions, receipt/clock lineage and gap recovery sufficient for a point-in-time dataset | UNRESOLVED |
| B — Lawful and reproducible access | Binance supplies conditional noncommercial dataset terms and a free-access statement; provider/API/source routes are traceable | Project-use applicability/full terms, any required paid/key entitlement, exact channel coverage, archival retention and immutable reproducibility are not sufficiently established for a complete route; other providers' data rights remain unknown | UNRESOLVED |
| C — Execution observability | Native book/trade fields could inform a conservative marketable-order specification | Arrival-time depth, delays, partial/rejected fills, impact and complete inventory-cost treatment cannot yet be specified defensibly from verified evidence; current simulator is inadequate | UNRESOLVED |

**0 PASS, 0 BLOCKED, 3 UNRESOLVED.** Binance's noncommercial limitations do not establish a fatal incompatibility for this explicitly nonexecuting documentary research task; no commercial/live entitlement is granted or presumed. No source silence or network restriction is promoted into NOT_FEASIBLE.

## 9. Consolidated feasibility decision

**HOLD_PENDING_EXTERNAL_EVIDENCE.** Candidate A remains the documentary priority requested by the task, not a selected strategy or demonstrated edge. There is insufficient evidence for FEASIBLE_FOR_HUMAN_REVIEW because the paired historical data route, lawful reproducibility and execution conditions do not all pass. No authoritative evidence establishes a critical incompatibility requiring NOT_FEASIBLE.

Data availability alone would not establish net expectancy; no performance measurement is made. This conclusion preserves TASK-010's NO_CANDIDATE_READY, funding carry's hold and every rejected/frozen hypothesis. No acquisition, engineering or experiment is initiated.

## 10. Missing evidence and next human decision

The smallest package able to change the decision is **one coherent source-specific documentary dossier**, not another general catalogue or a data sample:

1. **Provenance/coverage statement:** available product/channel mapping for the BTCUSDT scope reference, individual trades/aggressor semantics, aligned historical L2 snapshots/deltas, original sequence/reset/gap records, event/capture timestamps and clock limits, historical instrument changes, corrections and immutable versions. No historical period is chosen; any future period must be separately authorized and nonreserved.
2. **Lawful access/retention statement:** applicable research/archival/redistribution rights, commercial versus noncommercial limits, actual free/paid/authenticated entitlement and pricing, retention/completeness limitations and reproducible access/versioning. A code license is not a dataset license. No account, purchase, key or terms acceptance is requested by this task.
3. **Conservative observability statement:** what can actually be known at order arrival and what must be bounded or excluded for a marketable order, including latency, partial/rejected fills, depth/impact uncertainty and inventory completion. A provider may not be able to answer every execution question; any additional modeling assumptions need human review and separate engineering authority.

**Next human decision:** authorize only a bounded documentary clarification of that dossier for an appropriate potential source, or keep the microstructure track parked. Tardis's explicit historical trade/book claim identifies a concrete clarification route, not a vendor selection, subscription recommendation or automatic source preference. Prospective capture is a conditional later option if historical provenance cannot be established; it requires separate authorization, legal review and a nonconflicting period decision before capture, and must not be used to fill past observations retrospectively.

Stop after this report. No engineering change, collector, dataset access, preregistration, experiment ID, market outcome test or capital deployment follows from this decision.

## 11. Sources and verification

### Repository evidence

Local sources above were inspected without running their code. Prior reports and their references preserve the historical results; no leaderboard/dataset was opened to recompute outcomes. Verification date for this assessment: **2026-10-23**. Provider statements below are publisher documents/source at inspected commits, not independently tested operational coverage or an accepted contract.

### Publisher source registry

| ID | Publisher document/source at pinned commit | Relevant section, supported claim and limit |
|---|---|---|
| B1 | Binance Public Data README: [binance-archive](https://github.com/binance/binance-public-data/blob/d571690eb66760763b0bd6e27e8cd35603c7e244/README.md) | Data Information: spot/USD-M individual trades versus aggTrades/klines; CHECKSUM and Updates; Dataset Terms link. Archive route/field definitions and version replacement described, not complete L2 or receipt-time coverage. Verified 2026-10-23. |
| B2 | Binance Vision Dataset Terms v1.0, dated 2026-08-26: [binance-terms](https://github.com/binance/binance-public-data/blob/d571690eb66760763b0bd6e27e8cd35603c7e244/TERMS_AND_CONDITIONS.md) | §§2–4: conditional free access/noncommercial CC BY-NC-SA rights, additional terms and commercial restrictions; §10: no completeness/timeliness warranty. Read as a document only; applicability/legal review and full incorporated terms are not resolved. Verified 2026-10-23. |
| B3 | Binance Spot WebSocket Streams: [binance-streams](https://github.com/binance/binance-spot-api-docs/blob/f4a9c0be68235f758b46f571f7313c63d1306b25/web-socket-streams.md) | Trade Streams, Diff. Depth Stream and How to manage a local order book correctly: maker-side indicator, event/sequence fields, snapshot alignment, gap restart and finite-depth warning. Live spot specification, not an archived L2/receipt-time guarantee or USD-M instruction. Verified 2026-10-23. |
| B4 | Binance Spot Filters: [binance-filters](https://github.com/binance/binance-spot-api-docs/blob/f4a9c0be68235f758b46f571f7313c63d1306b25/filters.md) | Symbol price/lot/notional filters. Current specification only; no historical rule-value archive or selected instrument constraints. Verified 2026-10-23. |
| Y1 | Bybit Get Recent Public Trades: [bybit-trades](https://github.com/bybit-exchange/docs/blob/19518b90fb36e1527eaf99ff692076487a977921/docs/v5/market/recent-trade.mdx) | Request/response tables: recent-history scope, taker side, time, execution IDs/seq. Not proof of deep historical retention. Verified 2026-10-23. |
| Y2 | Bybit public Trade stream: [bybit-ws-trades](https://github.com/bybit-exchange/docs/blob/19518b90fb36e1527eaf99ff692076487a977921/docs/v5/websocket/public/trade.mdx) | Real-time description and response table: fill time T, system ts, taker S, shared seq across messages. No historical archive or observer-clock guarantee. Verified 2026-10-23. |
| Y3 | Bybit Get Orderbook: [bybit-book](https://github.com/bybit-exchange/docs/blob/19518b90fb36e1527eaf99ff692076487a977921/docs/v5/market/orderbook.mdx) | Snapshot, bids/asks, u/seq/ts/cts and RPI exclusion. Current snapshot interface, not a historical capture product. Verified 2026-10-23. |
| Y4 | Bybit WebSocket Orderbook: [bybit-ws-book](https://github.com/bybit-exchange/docs/blob/19518b90fb36e1527eaf99ff692076487a977921/docs/v5/websocket/public/orderbook.mdx) | Process snapshot/delta and response fields: reset/zero-delete semantics, sequence and matching-engine timestamp. Not a complete archived gap/receipt-time guarantee. Verified 2026-10-23. |
| Y5 | Bybit Get Instruments Info: [bybit-rules](https://github.com/bybit-exchange/docs/blob/19518b90fb36e1527eaf99ff692076487a977921/docs/v5/market/instrument.mdx) | Price/quantity/funding/contract metadata and product sections. No time-effective historical rule archive established. Verified 2026-10-23. |
| T1 | Tardis.dev client and Tardis Machine READMEs: [tardis-node](https://github.com/tardis-dev/tardis-node/blob/b6dfda49edecd6555f3abff524c7611ee3fbadce/README.md); [tardis-machine](https://github.com/tardis-dev/tardis-machine/blob/0a2ae615c7328df9f2c91d17f32ceac6abde5ba7/README.md) | Features and replay examples: explicit historical snapshots/incrementals/trades, native/normalized replay, reconstruction and caching. A documented provider capability claim, not verified exact coverage, retention, price or dataset licensing. Verified 2026-10-23. |
| T2 | Tardis.dev normalized type declarations: [tardis-types](https://github.com/tardis-dev/tardis-node/blob/b6dfda49edecd6555f3abff524c7611ee3fbadce/src/types.ts) | Trade, BookChange and NormalizedData: aggressor buy/sell/unknown, timestamp/localTimestamp, isSnapshot and price/amount arrays. BookChange has no mandatory native sequence field; type availability is not data completeness. Verified 2026-10-23. |
| T3 | Tardis.dev replay source: [tardis-replay](https://github.com/tardis-dev/tardis-node/blob/b6dfda49edecd6555f3abff524c7611ee3fbadce/src/replay.ts) | Replay options/parser and validateReplayNormalizedOptions: optional API-key configuration, capture timestamp parsing, disconnect handling and UTC day-start snapshot warning. Source was not imported/executed; no actual replay or assurance of capture clock accuracy. Verified 2026-10-23. |

All thirteen pinned documents above returned HTTP 200. Publication repositories were identified by native read-only Git HEAD checks; no archive index, zipped dataset, sample feed or SDK example was fetched/executed. Binance source README labels its repository “MIT” but separately links Dataset Terms B2. The absent standalone LICENSE lookup (HTTP 404) does not override the accessible dataset terms or establish a permission.

### Documentation access limitations

The four following **documentation/pricing** requests returned exit 56 and HTTP result `000`, with exact error `curl: (56) CONNECT tunnel failed, response 403`. No page content or price was verified and no restriction was bypassed:

- **H1:** [Binance rendered stream documentation](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams); independent official source B3 was accessible.
- **H2:** [Bybit rendered orderbook documentation](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook); independent official source Y4 was accessible.
- **H3:** [Tardis.dev documentation](https://docs.tardis.dev/); pinned publisher source/readmes provide limited evidence, not complete service terms/catalogue.
- **H4:** [Tardis.dev pricing page](https://tardis.dev/#pricing); paid/free tiers, exact entitlement and monetary pricing remain UNKNOWN.

No secondary-provider assertion is promoted into a verified completeness or license guarantee. A blocked page is not a documented missing feature. Reading terms/documentation is not dataset acquisition, a purchase, account onboarding or acceptance of service terms.

### Verification scope

Checks resolve local references, tie publisher links to successfully retrieved commit-pinned documents and supporting excerpts, require all ten statused requirements for each of the three options, distinguish historical/live/aggregate/provider-claim evidence, and check the three UNRESOLVED gate classifications. Prior TASK-010, funding decision and failure reports remain byte-identical. Only this Markdown file may be added and `git diff --check` must pass. No application tests, research simulation or performance measurement is appropriate or run for this documentary task.

Completed documentary verification: eight local source references resolved; thirteen pinned publisher documents and four explicitly blocked documentation/pricing targets were traced; supporting source excerpts were cross-checked; all thirty provider requirement rows use the allowed statuses; all eleven sections and three UNRESOLVED gates were verified. Prior principal reports stayed byte-identical, table widths were consistent, only this Markdown file was added, and `git diff --check` passed. No data or execution test ran.
