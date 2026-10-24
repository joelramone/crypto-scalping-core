# Tardis Historical Microstructure Evidence Dossier V1

## 1. Executive summary

**HOLD_PENDING_PROVIDER_EVIDENCE. Gates A/B/C: UNRESOLVED. Execution: NOT AUTHORIZED.** Tardis publishes a historical trade/L2 replay route and source-level reconstruction details, but the inspected material does not establish a complete, licensed, repeatably versioned, causally timed archive for an eventual common venue/product/symbol scope. No incompatibility sufficient for BLOCKED was verified.

New source-specific evidence beyond TASK-011: separate spot/futures mapper behavior; native trade-ID/aggressor mapping; snapshot buffering and timestamp fallback; disconnect handling; minute-offset archive retrieval and caching; and the SDK's MPL-2.0 software license. These are documented client behaviors, not archive completeness, clock, retention or data-license guarantees [T1–T8, sections below; verified 2026-10-24]. Terms, pricing and detailed documentation were inaccessible through the configured proxy. No price, research entitlement or operational guarantee is inferred.

Candidate A remains a priority for documentary clarification only. No trading edge or profitability is assessed. Even a complete archive would not reveal the counterfactual fills, routing latency or impact of an order never submitted.

## 2. Scope and previous research constraints

Baseline: `origin/main` at `7e3dca0` includes **PR #186**, merging TASK-011 commit `c27557bb8ac5d4de48e07bc80e45681e3aa4f0cd`. The [TASK-011 assessment](microstructure_data_execution_feasibility_v1.md) remains HOLD_PENDING_EXTERNAL_EVIDENCE; [TASK-010](scalping_new_edge_discovery_v1.md) remains NO_CANDIDATE_READY. This dossier does not replace either decision. [Funding carry](funding_carry_decision_gate_v1.md) stays HOLD_PENDING_EXTERNAL_EVIDENCE; [EXP-000021–024](failure_patterns_exp_000021_000024.md) stay closed/rejected.

Only Tardis publisher material was investigated. BTCUSDT is conceptual only: no venue, product, research period, thresholds or trading rules are selected. Spot and perpetual archives are distinct requirements. The product examples below illustrate Tardis implementation differences, not a venue recommendation. Dates in publisher examples are not selected research dates.

[Regime-transition data restrictions](../hypotheses/HYP-REGIME-TRANSITION-001.md#data-boundaries) and [Donchian confirmation restrictions](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md#confirmation-dataset-and-evaluation-gate) remain binding: 2025 is discovery-used, 2026-01-01 through 2026-08-05 already used, and later data reserved. Changing provider/data type does not release those boundaries. No dataset, sample, outcome, data API or private account was accessed; no SDK replay, acquisition, engineering, experiment, preregistration or Research Memory write occurred.

## 3. Historical trade and L2 coverage

Statuses describe the exact documentary scope stated: **VERIFIED_DOCUMENTARY** supports a statement in inspected publisher material, not validated data; **PARTIALLY_VERIFIED** leaves material requirements unresolved; **UNKNOWN** means not established; **DOCUMENTED_UNAVAILABLE** requires an explicit negative statement. No inaccessible page is classified unavailable. All source checks below are dated **2026-10-24**; source IDs resolve in §12.

| Requirement | Documentary finding and limit | Source/section | Status |
|---|---|---|---|
| Historical individual trades | README advertises tick-by-tick historical replay, distinct from derived bars; source mapper handles native `@trade`. Exact archive completeness/coverage remains unverified | T1 Features; T4 BinanceTradesMapper | PARTIALLY_VERIFIED |
| Aggressor side | Trade type explicitly means liquidity taker, with buy/sell/unknown. Illustrative mapper maps native maker flag `m` to sell/buy; filters INSURANCE_FUND/ADL/NA off-book trades. Normalized and raw populations need explicit reconciliation | T2 Trade; T4 BinanceTradesMapper.map | PARTIALLY_VERIFIED |
| Native trade identifiers | Example mapper uses native `t` as string ID; generic Trade allows undefined. Uniqueness scope and all product/history mappings are not guaranteed by the type | T2 Trade; T4 BinanceTradesMapper.map | PARTIALLY_VERIFIED |
| Historical L2 snapshots | README explicitly advertises full-depth snapshots. Mapper consumes depthSnapshot/lastUpdateId and buffers preceding deltas; exact archive depth limits and initial-state availability unresolved | T1 Features; T4 BinanceBookChangeMapper | PARTIALLY_VERIFIED |
| Incremental L2 | README advertises incrementals; inspected mapper reads bids/asks, interprets zero amount as deletion when applying buffered updates. No complete-capture guarantee | T1 Features; T4 mapBookDepthUpdate/applyLevelUpdates | PARTIALLY_VERIFIED |
| Snapshot/replay alignment | Initial overlap checks differ by product; normalization warns about starting away from UTC day start. This is client behavior, not proof every requested interval can reconstruct | T3 validateReplayNormalizedOptions; T4 book mappers | PARTIALLY_VERIFIED |
| Channel-specific coverage | Client lists trade/depth/depthSnapshot and distinct exchange IDs. Accepted filters are not historical channel availability dates or missingness manifests | T5 BINANCE_CHANNELS/BINANCE_FUTURES_CHANNELS; T4 getFilters | PARTIALLY_VERIFIED |
| Native sequence identifiers | Raw mapper types/logic use U/u, snapshot lastUpdateId, and futures pu for initial alignment; normalized BookChange does not require these fields. Ongoing continuity is not certified by successful normalization | T2 BookChange; T4 mapBookDepthUpdate | PARTIALLY_VERIFIED |
| Reset/resynchronization | Replay blank-line disconnect markers reset active normalizers; later messages recreate mappers. Inspected snapshot mapper ignores additional manual snapshots once initialized; not proof all silent gaps/resets are recorded | T3 normalizeReplayLineBatches; T4 BinanceBookChangeMapper.map | PARTIALLY_VERIFIED |
| Instrument/symbol mapping | Distinct exchange identifier plus symbol; mapper lowercases requested filters and retains native trade symbols/uppercases book stream symbols. Historical contract identity, renames, effective tick/lot/rules need a versioned catalogue | T2; T4 getFilters/map; T5 | PARTIALLY_VERIFIED |
| Same-scope trade/book package | T1 demonstrates a single exchange/symbol request with both trade/book normalizers; T4 provides both mappings. This documents an interface route, not paired historical coverage or completeness for a selected product | T1 opening example; T4 exchangeMappers | PARTIALLY_VERIFIED |
| Exact product/channel historical availability | No inspected authoritative coverage manifest establishes continuous same-symbol trades, snapshots and deltas or retention guarantees | T1/T4/T5 limitations; H1/H2 inaccessible | UNKNOWN |

**Spot versus perpetual distinction:** T4 registers separate `binance` and `binance-futures` mappings, with a separate `binance-delivery` identity as well. Spot initial overlap uses U ≤ lastUpdateId + 1 ≤ u; the futures mapper accepts boundary overlap or pu == lastUpdateId. They cannot be treated as one interchangeable feed. This is evidence about supported client paths only: it neither establishes BTCUSDT availability for every path nor identifies the full contract/settlement history. No dates or product are selected. No claim about all other Tardis exchanges is extrapolated from this example.

## 4. Timestamp/provenance assessment

| Requirement | Evidence and causal limitation | Source/section | Status |
|---|---|---|---|
| Exchange timestamps | Example trade timestamp comes from T; book deltas from E; snapshot uses T when present, otherwise localTimestamp. A normalized timestamp can therefore have different origins; never assume all are matching-engine event time | T4 trade/book map methods | PARTIALLY_VERIFIED |
| Provider local timestamps | Replay parser reads the timestamp prefixed to archived lines and supplies it to normalizers. Recorder-format comment documents fractional digits, not capture location or event-to-receipt semantics | T3 parseReplayTimestamp/normalizeReplayLineBatches | PARTIALLY_VERIFIED |
| Capture receipt origin/clock | No inspected specification establishes capture boundary, host/location, clock discipline, drift, precision versus accuracy, cross-channel offset or uncertainty | T2/T3 limitations; H1/H2 inaccessible | UNKNOWN |
| Replay timestamps | Replay yields archived localTimestamp, not an observed historical trader's receipt time or the present replay wall clock. Date parser retains milliseconds; optional raw withMicroseconds/normalized path adds μs. Representation precision does not establish accuracy | T3 replay/parseReplayMicroseconds/normalizeReplayLineBatches | VERIFIED_DOCUMENTARY |
| Message ordering | Parser traverses stored line batches in order and invokes normalizers lazily; source has exchange-dependent segmentation hooks. No global exchange-order or simultaneous trade/book causality guarantee follows | T3 replay/normalizeReplayLineBatches/getReplayNormalizedSegments | PARTIALLY_VERIFIED |
| Missing messages | Blank lines can produce disconnect markers and mapper resets; no statement establishes detection of every silent packet/message loss or complete missingness | T3 replay/normalizeReplayLineBatches | PARTIALLY_VERIFIED |
| Duplicates/out-of-order | Example mapper drops some updates at/below snapshot boundary; this is not a universal deduplication or out-of-order recovery guarantee for all trades/deltas | T4 spot/futures mapBookDepthUpdate | PARTIALLY_VERIFIED |
| Publication delay and revisions | Capture-to-availability delay, corrections policy, original versus repaired views and when corrections became known are not established | T1/T3/T8 limitations; H1/H2 | UNKNOWN |
| Versioned provenance | Source commits pin client interpretation; no inspected dataset release identifier pins captured bytes/clock/rule history | T3/T8; §6 | PARTIALLY_VERIFIED |

**Point-in-time assessment:** a potentially implementable replay interface is documented, but a causally valid specification is not yet established. Provider capture time must not be equated with hypothetical trader availability. An eventual specification would need event/publication/receipt meanings and units, capture uncertainty, stable ordering/tie rules, native lineage, initial book availability, explicit gaps and original/corrected views. Buffered snapshot assembly must not be treated as information known before its emission; a later snapshot cannot repair earlier decisions. Unknown clock relationships cannot be resolved by sorting only on exchange timestamp. These are required evidence/design boundaries, not an implemented replay or proposed trading rule.

## 5. Licensing, access and pricing

| Requirement | Finding as of 2026-10-24 | Source/section | Status |
|---|---|---|---|
| SDK software license | Inspected tardis-node LICENSE is Mozilla Public License 2.0; scope is Covered Software, with limited grants. This does not grant market-data rights | T7 §§1, 2, 6 | VERIFIED_DOCUMENTARY |
| Dataset license / exchange restrictions | Applicable provider agreement and underlying exchange restrictions not verified | H3 inaccessible; T7 is not a dataset agreement | UNKNOWN |
| Personal research use | No inspected data license establishes applicable permissions/conditions | H3; T1/T7 scope limits | UNKNOWN |
| Commercial research use | Applicability, user class and internal commercial use rights not established | H3 | UNKNOWN |
| Internal archival / post-subscription retention | Software caches data, but technical caching is not permission to retain raw archives or an entitlement after cancellation | T1 Features; T8 cache paths; H3 | UNKNOWN |
| Derived-data rights | Feature/result storage, publication, model outputs and derived-data definitions unresolved | H3 | UNKNOWN |
| Redistribution restrictions | Raw/normalized data sharing and collaborator access rights unresolved; no permissive right assumed | H3 | UNKNOWN |
| Subscription/API key | Client forwards optional configured API key; worker checks access at range boundaries. This establishes authentication support, not anonymous/full/free access | T3 replay; T8 replay range retrieval | PARTIALLY_VERIFIED |
| Historical entitlements | Products/channels/date range, users and download/retrieval restrictions not established for a lawful research account | T8 access-check comments; H1–H3 | UNKNOWN |
| Retention guarantee | Provider retention/service continuity/retrieval after correction not verified | H1–H3 | UNKNOWN |
| Pricing model | No verified tariff, currency, billing unit, tax, plan limits or quote validity. No numeric cost estimate supplied | H4 inaccessible | UNKNOWN |
| Dataset versions offered | No inspected statement establishes immutable releases or original-versus-corrected archive entitlements | T3/T8 limitations; H1/H2 | UNKNOWN |

No trial policy or free-day access is inferred from uncredentialed README examples. No account was created, terms accepted, purchase made or credentials requested. The repository names and MPL license establish software provenance only; commercial data-use applicability remains UNKNOWN.

## 6. Historical reproducibility

Two researchers could express equivalent source requests and pin identical software, but **byte-identical historical retrieval is not verified**. Same request identifiers do not establish an unchanged archive. An authenticated, licensed route also remains unresolved.

| Requirement | Evidence / missing condition | Source/section | Status |
|---|---|---|---|
| Request identity | Exchange, from/to, filters (channels/symbols), raw/normalized mode and normalizer commit can be specified; no request executed | T3 ReplayOptions/replayNormalized; T5 | VERIFIED_DOCUMENTARY |
| Archive organization / pagination | Worker addresses date/from plus minute offsets, optional sliceSize and filters; caches compressed slices. This is inspected client retrieval logic, not a guaranteed permanent server contract | T8 getDataFeedSlice/replay range retrieval | PARTIALLY_VERIFIED |
| Dataset releases / immutable archives | No inspected release IDs or immutable historical-byte guarantee | T1/T3/T8 limitations | UNKNOWN |
| Historical correction lineage | Replacement notices, correction times, access to superseded bytes and “as originally observed” view not established | H1/H2; T3/T8 limitations | UNKNOWN |
| Symbol/channel mapping | Client constants and product mappers exist; historical catalogue/schema changes need a versioned manifest | T4/T5 | PARTIALLY_VERIFIED |
| Snapshot/delta alignment | Boundary rules and reset behavior documented in client, with start-time caveat; coverage/continuity still unverified | T3/T4 | PARTIALLY_VERIFIED |
| Sequence gaps | Normalized BookChange omits native IDs; raw lineage and complete gap/resynchronization reporting require confirmation | T2/T4 | PARTIALLY_VERIFIED |
| Retention | No verified duration, withdrawal or reproducible re-access commitment | H1–H3 | UNKNOWN |
| Checksums / provenance metadata | No inspected publisher content manifest guarantees immutable checksums of archived messages. Client cache keys/request hashes are not a content-integrity certificate | T8 cache organization; T3 | UNKNOWN |

Minimum future reproducibility identity would include provider dataset/revision, venue/product/native instrument, channel and schema versions, original/corrected view, snapshot boundary, gaps/resets, timestamp provenance, retrieval options and licensed content hashes. These are evidence requirements only; no manifest generator, data request or archive is created here. Publisher software commits and documentary SHA-256 below pin what was inspected, not the underlying market history.

## 7. Candidate A execution observability

Categories: **Directly observable from documented historical data** concerns published fields, conditional on actual coverage; **Potentially reconstructable with explicit assumptions** is not an observed hypothetical execution; **Not established by available documentation** is unresolved. No row validates an archive or designs a simulator.

| Element | Classification | Conservative interpretation / unresolved condition |
|---|---|---|
| Information known at decision time | Potentially reconstructable with explicit assumptions | Trades/visible depth only after demonstrable observation; no future snapshot, trade or correction. Needs §4 provenance [T3/T4] |
| Exchange event versus capture fields | Directly observable from documented historical data | Mapped timestamp and archived localTimestamp values are exposed; their operational relationship/clock accuracy is not established [T2–T4] |
| Order submission after observation | Potentially reconstructable with explicit assumptions | Must follow observation/computation; the archive contains no hypothetical submission record [T1–T4 scope] |
| Routing/arrival latency | Not established by available documentation | Provider clock precision is not trader-path latency; requires future human-approved bounds, not zero delay |
| Executable spread | Potentially reconstructable with explicit assumptions | Best visible opposing prices require a valid causally aligned book; displayed quote is not a guaranteed fill [T2/T4] |
| Depth at hypothetical arrival | Potentially reconstructable with explicit assumptions | Arrival time and valid book state required; hidden liquidity, cancellations and book coverage remain uncertain [T4] |
| Partial fills | Potentially reconstructable with explicit assumptions | Displayed depth may bound scenarios, not prove realized allocation or full quantity; no actual hypothetical fill record |
| Rejected/unfilled orders | Not established by available documentation | Future account/risk/rule checks, routing failures and time-in-force outcomes not observed in public archives |
| Market impact | Not established by available documentation | Historical replay excludes the counterfactual response to one's own order |
| Transaction fees | Not established by available documentation | Effective historical product/account fee tier and entry costs not supplied by trade/book schemas; existing fees cannot be silently reduced |
| Position closing costs | Potentially reconstructable with explicit assumptions | Closing depth/spread plus separately evidenced fees/latency/impact required; a favorable terminal mark is not liquidation |
| Inventory completion | Potentially reconstructable with explicit assumptions | Residual quantities, failed entry/exit and terminal inventory require explicit conservative future treatment; no automatic complete round trip |

The negative execution findings are limits of the inspected public trade/book sources [T1–T6], not claims that Tardis sells a simulator or records private orders. Even complete archives cannot identify own counterfactual impact, account-specific rejection, trader routing latency or certain future fills. A provider can clarify capture/depth/rules; humans must separately decide defensible latency/impact bounds, residual inventory treatment and whether uncertainty precludes research. The [current candle simulator](../../app/research/simulation.py) does not represent these event-level semantics; separate engineering authorization would still be required. No changes are made.

## 8. Three-gate decision matrix

| Gate | Classification | Evidence and unmet critical condition |
|---|---|---|
| A — Historical provenance | UNRESOLVED | Historical trade/L2 interfaces, raw mapping and partial reconstruction are documented; same-scope coverage, clock/receipt lineage, gaps and original/corrected views are not established (§§3–4) |
| B — Lawful access and reproducibility | UNRESOLVED | SDK license is verified, dataset rights/entitlements/pricing/retention and immutable retrieval are not (§§5–6). Inaccessible terms are not an incompatibility |
| C — Conservative execution feasibility | UNRESOLVED | Visible-book reconstruction is plausible conditionally; causal arrival, costs, fills/impact bounds and inventory completion are not sufficiently specified (§7). No impossibility is demonstrated |

PASS requires the relevant evidence, not an attractive SDK interface. BLOCKED requires verified incompatibility; none was established. No gate is promoted based on unavailable pages or presumed future engineering.

## 9. Consolidated decision

**HOLD_PENDING_PROVIDER_EVIDENCE.** Source-specific progress does not close the critical requirements. Neither readiness for protocol feasibility review nor demonstrated infeasibility follows from the inspected material. Existing research decisions remain unchanged. No profitability claim or permission to collect, implement, preregister or experiment follows.

## 10. Minimal missing evidence package

Request authoritative documents or written clarification, **not data samples or another generic API inventory**:

1. **Paired coverage/provenance statement:** product-separated catalogue showing how a future same-symbol trade/snapshot/delta scope is identified; native IDs/channel coverage, depth limits, snapshot timing, sequence/reset/gap/duplicate handling, native-to-normalized exclusions and historical rule mapping. No research period needs selection now.
2. **Timestamp/causality statement:** capture boundary/location, clock synchronization and uncertainty, event/receipt/publication meanings, cross-channel order, disconnect/missingness reporting, and which snapshot/correction information became available when.
3. **Applicable rights and entitlement offer:** data agreement covering this user's personal/commercial research classification, exchange restrictions, internal archives/post-subscription retention, derived outputs/collaborators/redistribution, product/history access, and current priced terms with currency/billing unit/date/limits. MPL software rights are insufficient.
4. **Repeatability statement:** immutable release or retained revision IDs, original/corrected access, changes/retention policy, content checksums/manifests and schema/client version linkage. If unavailable, provider should state the limitation explicitly rather than promise identical results from identical requests.

These are provider-answerable questions. Separately, a future **human research-design decision** must address observer-to-trader latency, counterfactual impact/partial execution, fee evidence and complete inventory accounting. Provider documentation cannot turn hypothetical fills into observations. Nothing here authorizes contacting the provider on the user's behalf.

## 11. Recommended next human action

Decide whether to obtain the four bounded documentary items from Tardis through an authorized human, or park Candidate A pending access to authoritative material. Reassess A/B only on the returned evidence; review C separately for conservative modeling feasibility. Do not purchase, obtain data, choose a venue/product/period, design an execution simulator, preregister or run experiments on this dossier's authority. Stop after documentation delivery.

## 12. Authoritative sources and verification

All documents verified **2026-10-24**. Accessible publisher-owned repository sources are pinned to the HEAD commits resolved during this task: tardis-node `c949a30567d6c3e3c237852c9960401cf5b66bdf`; tardis-machine `aa261a8571bc1a2b2543086d8428744043e3f925`. Code was inspected as text only. README capability claims, interface definitions, implementation behavior and operational guarantees are deliberately distinguished.

| ID | Pinned publisher source | Relevant sections | SHA-256 of inspected document |
|---|---|---|---|
| T1 | [node-readme](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/README.md) | Opening example; Features; native replay | `69f5064ff8c3aa0f3a27d486ee2379c0d87f3edcbcb7629fb1b7c1b4142d2499` |
| T2 | [node-types](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/src/types.ts) | Trade; BookChange; NormalizedData | `4ad1f108d3b4c085c50a78daf0189e394c61c7b53c0f9f714b9ed37d88d166ce` |
| T3 | [node-replay](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/src/replay.ts) | replay; timestamp parsers; normalizeReplayLineBatches; getReplayNormalizedSegments; validation | `c57ba1703b8c367c014dd3497c4b5bff2dd166d0530100dccb5a618636c9d194` |
| T4 | [node-binance](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/src/mappers/binance.ts) | exchangeMappers; BinanceTradesMapper; spot/futures book mappers | `b20944b36483f97167db4688829ff638fffbc226bc4dd5e300eb52ddc00df8bd` |
| T5 | [node-consts](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/src/consts.ts) | EXCHANGES; Binance channel lists/mapping | `5219d251a1e4dde83946f186c80509585ea656b99a1c8c5aff329ee4a7e52a35` |
| T6 | [machine-readme](https://raw.githubusercontent.com/tardis-dev/tardis-machine/aa261a8571bc1a2b2543086d8428744043e3f925/README.md) | Introduction; Features | `eb7d9677b89da668b6b2bb7ab8b9bf1f8458b915047ba640e920a83c3996b541` |
| T7 | [node-license](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/LICENSE) | MPL-2.0 §§1–2, 6 | `1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5` |
| T8 | [node-worker](https://raw.githubusercontent.com/tardis-dev/tardis-node/c949a30567d6c3e3c237852c9960401cf5b66bdf/src/worker.ts) | Range retrieval; getDataFeedSlice; cache organization | `5425320d252f753b83460d17dd4b5602ba79c89f149fb3e463dfed7ec4feef5a` |

Inaccessible pages (H1–H4 respectively): [documentation](https://docs.tardis.dev/), [FAQ](https://docs.tardis.dev/faq), [terms URL](https://tardis.dev/terms-of-service), [pricing](https://tardis.dev/#pricing). Each returned curl exit 56, HTTP 000, exact error `curl: (56) CONNECT tunnel failed, response 403`. URL paths do not establish their contents or existence beyond the denied request. No proxy/TLS bypass, alternate mirror or credentials were used. A speculative `tardis-dev/docs` Git repository lookup returned `Repository not found`; that does not prove documentation or data is unavailable. Separately accessible publisher SDK repositories were read; no data endpoint, archive index, sample or example payload was retrieved.

Verification: PR #186 ancestry confirmed; all local references resolve; eight publisher source documents retrieved successfully and pinned; four blocked pages disclosed; source excerpts support the critical field/mapping/license statements; all coverage, provenance, licensing and reproducibility rows have explicit allowed statuses; execution rows use the three requested observability categories; A/B/C are explicitly UNRESOLVED. Only this Markdown file is added, existing tracked files remain unchanged, and `git diff --check` passes. No application tests, SDK replay, market-data access, research simulation or optimizer was run.
