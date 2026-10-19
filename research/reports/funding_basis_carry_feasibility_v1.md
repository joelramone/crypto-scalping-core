# Funding/Basis Carry Feasibility Assessment V1

## 1. Executive summary

**Preliminary classification: NEEDS_MORE_EVIDENCE for Binance, Bybit and OKX. Recommendation B: investigate specified documentary gaps. Execution status: NOT AUTHORIZED.** None is established as profitable, accessible to this user, or ready for preregistration. This is a human-approved parallel research track; the scalping track and its frozen/rejected hypotheses remain unchanged.

A long-spot/short-linear-perpetual structure has an independent potential revenue source: funding transfers and changes in the spot/perpetual basis. It is not a rescue of a rejected candle strategy. Positive funding alone does not establish positive net PnL, capital efficiency or survival of the hedge. Financing, four-leg transaction costs, basis divergence, margin location and failed execution can dominate funding receipts.

Documentation verification date: **2026-10-19**. Direct official documentation pages for all three venues were blocked by the network proxy (`curl: (56) CONNECT tunnel failed, response 403`, HTTP result `000`). Official GitHub documentation sources for Bybit and a spot specification/deprecated futures connector for Binance were accessible and pinned to commits. OKX rules could not be independently verified. Accessible source text verifies only the specific statements cited below, not current API operation, complete venue rules or account eligibility.

No market-data API, account API, exchange account or historical dataset was accessed. Only documentation/source text and existing repository reports/protocols were read. API paths mentioned in this report are documentation references, not endpoints that were called. No source SDK was imported or executed.

## 2. Scope and research boundaries

Repository baseline: `main` commit `1a9bee39637a3582d1d3204d9467c2340d3bdadd` (merge of PR #181, TASK-006). This assessment extends Candidate A in the existing [strategy opportunity assessment](strategy_opportunity_assessment_v1.md) [R1]; it does not recreate or modify that report. The unchanged [historical failure report](failure_patterns_exp_000021_000024.md) [R2] establishes cost-dominated gross-positive outcomes in EXP-000021–023 and gross-negative economics in EXP-000024. Those outcomes motivate a cost-first feasibility question; they do not constitute evidence for funding carry.

**What the repository establishes:** the [official simulator](../../app/research/simulation.py) [R3] models one directional position, candle-close entry, subsequent candle exits, fixed entry notional, and price PnL less entry/exit fees. Its [default fee rate](../../app/research/config.py) [R4] is 0.0004 per side. It has no simultaneous spot/perpetual legs, funding cashflow, borrow/financing, collateral accounts, partial fills, spreads, slippage, market impact or liquidation model. It therefore cannot currently validate this mechanism. A different cost assumption in a future carry protocol must not rewrite old research results.

**What remains unknown:** whether a particular lawful account/product is available; exact current settlement and funding sign rules for each contract; point-in-time estimate history; complete all-leg costs; feasible capital/margin arrangements; executable hedge quality; and any empirical profitability or frequency. The report assigns no new experiment or hypothesis ID, numerical acceptance threshold, symbol, venue choice, discovery period or confirmation period.

The [existing data boundary](../hypotheses/HYP-REGIME-TRANSITION-001.md) and [Donchian confirmation reservation](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md) [R5] remain binding: 2025 is discovery-used; 2026-01-01 through 2026-08-05 has already been used elsewhere; post-2026-08-05 is reserved. Neither a new source type nor passage of time releases a reserved interval. No period is designated or inspected here. [Research Memory storage](../../app/research/memory/experiment_store.py) [R6] uses journals and an index under `research/memory`; its writers and ID allocation are out of scope and were not invoked.

## 3. Economic mechanism

### 3.1 Spot-perpetual funding carry

For explanation only, consider owned spot and a short **linear, quote-currency-settled** perpetual on the same base asset. Equal base quantities approximately offset first-order price delta. Matching displayed quote notionals alone is not sufficient when contract multipliers, rounding, inverse settlement or collateral exposures differ. Inverse/coin-margined products require separate payoff equations and are not silently treated as linear contracts.

A perpetual has no assumed fixed maturity at which its price must equal spot. Funding is a periodic cash transfer that can be a receipt or payment for the short. A position may need to be open at the venue's applicable settlement observation; exact inclusion timing, rounding, caps, interval changes and debiting rules must be verified per contract. Do not assume an eight-hour interval, constant funding, positive funding or guaranteed collection by entering just before a displayed time.

For the algebra below only, define a normalized rate `r_k` as **positive when a short receives funding**. Under that chosen convention, a short's funding cashflow is `N_k × r_k`, with `N_k` the applicable funding valuation notional. This is a definition for an illustrative ledger, not a verified common sign convention for all exchange APIs. Raw rates, reported fee fields and cash credits must be normalized separately using documented exchange rules.

| Venue | Funding sign and settlement interpretation supported by inspected evidence | What cannot be assumed |
|---|---|---|
| Binance | Deprecated official connector describes funding history and adjustable funding-interval metadata [B1, B3]. It does not establish a current contract's rate-to-short-cashflow sign convention. **Market-rate sign mapping and actual credit/debit rules unverified.** | Do not populate the normalized `r_k` from a raw Binance field until its sign, valuation and settlement rules are documented. |
| Bybit | Official transaction-log documentation explicitly says `funding` positive = received, negative = paid; `fee` positive = expense, negative = rebate. `change = cashFlow + funding - fee`. Its `feeRate` description uses the market rate for Buy and its negation for Sell [Y6]. This verifies ledger-field meanings, not a standalone universal mapping of a quoted public rate to the short's actual cash amount. | Do not equate a positive transaction-log cash field with a positive public `fundingRate`, or assume every short receives funding. The complete contract cash calculation and settlement inclusion rule remain to be verified. |
| OKX | Official documentation site was inaccessible [O1]. **Rate sign, receipt/payment field signs, valuation and settlement inclusion rules unverified.** | No OKX funding convention is asserted from memory or borrowed from another exchange. |

Maintaining the hedge requires monitoring actual filled base quantities, fees deducted in base currency, contract specifications and rebalances. A short-perpetual margin account may lose collateral during a spot rally even while owned spot appreciates elsewhere. “Hedge mode” normally concerns simultaneous long and short positions within a derivatives product; it is not the definition of a spot/perpetual hedge and does not prove spot gains are recognized as margin.

### 3.2 Funding, basis and convergence are separate

Define linear-contract basis in quote currency per base unit as `B_t = F_t - S_t`, using comparable perpetual and spot valuations. For fixed equal base quantity `q`, ignoring costs and funding:

```text
spot price PnL       = q × (S_T - S_0)
short perpetual PnL = q × (F_0 - F_T)
paired price PnL     = q × (B_0 - B_T)
```

Basis narrowing benefits this pair; widening hurts it. Before liquidation/closure, a mark-price basis change is mark-to-market PnL, not automatically spendable spot cash. At actual closure, use the two executable fill prices for realized basis PnL. Do not add basis PnL a second time after already summing both legs' price PnLs. Financing and funding remain separate ledger components. With changing hedge quantity, keep a per-fill and per-settlement ledger rather than applying the fixed-`q` shortcut to the entire history.

**Dated-futures cash-and-carry is different:** an expiry contract may settle against a specified index or delivery process, potentially realizing an entry premium at maturity if the hedge and settlement assets align. The settlement rule, delivery fees and residual spot/index mismatch must be verified. Periodic perpetual funding is not automatically a revenue component of a dated future. A perpetual has no assumed expiry convergence guarantee. No dated-futures product availability or settlement promise is established for these venues by this assessment.

Delta neutrality is not capital neutrality, liquidity neutrality or absence of counterparty/default risk. A hedge can have small aggregate price delta and still face basis loss, separate margin calls, collateral haircuts, financing costs and forced unwind.

## 4. Exchange comparison

Evidence labels: **verified source statement** means the specified pinned official document contains that statement, not that a live account/product was checked. **Limited** means a deprecated source or incomplete rule. **Unknown** means neither satisfaction nor failure is established. No current numerical trading tariff, actual minimum order value, margin percentage or geographic eligibility is asserted.

| Criterion | Binance | Bybit | OKX |
|---|---|---|---|
| Funding settlement and frequency | **Limited:** deprecated official USD-M connector documents history and funding information, including symbols with adjusted `fundingIntervalHours` [B1]. Actual contract interval, UTC schedule, inclusion timing and rate formula unknown. | **Verified source:** funding-history document says intervals differ by symbol; instrument metadata exposes `fundingInterval` in minutes; tickers expose `nextFundingTime` and `fundingIntervalHour` [Y1, Y2, Y7]. No universal interval is inferred. Precise settlement/debit and inclusion rules still missing. | **Unknown:** official docs blocked [O1]; do not assume frequency, caps or sign convention. |
| Historical funding availability | **Limited:** source describes `GET /fapi/v1/fundingRate`, start/end parameters and default/max page sizes 500/1000 [B1]. Retention, missing records and point-in-time estimate history unverified; no endpoint called. | **Verified source:** `/v5/market/funding/history`; 1–200 results/page, default 200; passing only startTime errors. It documents USDT/USDC and inverse perpetual coverage [Y1]. Earliest retention/completeness and historical pre-settlement estimates unknown. | **Unknown:** no verified history endpoint, retention or coverage. Documentation access is the first prerequisite [O1]. |
| Spot/perpetual fees; maker/taker | **Unknown applicable rates:** futures connector documents a signed user commission-rate route [B2], which was not called. It is not a public spot fee schedule, guarantee of current tiers or proof of a maker rebate. No fee tier is borrowed from the repository. | **Verified distinction:** account fee-rate documentation exposes `makerFeeRate` and `takerFeeRate` for spot/linear/inverse categories [Y3]. Current public schedules, user VIP tier, product eligibility, discounts and actual rates remain unknown; private endpoint not called. | **Unknown:** current spot/perpetual maker/taker schedules and applicable account tier [O1]. |
| Initial/maintenance margin; collateral | **Limited:** deprecated connector exposes margin-type/leverage and leverage-bracket operations [B2]. Current tiers, haircuts, collateral eligibility and offset treatment unverified. | **Verified fields:** position documentation distinguishes initial/maintenance margin and mark-price variants, with portfolio-mode fields possibly empty [Y5]. Values, risk tiers, collateral haircuts and account-specific constraints unknown. | **Unknown:** account mode, collateral and risk-tier requirements [O1]. |
| Liquidation and loss allocation | **Unknown current rules:** mark-trigger logic, margin ratios, liquidation fees and loss-allocation/ADL processes need current product rules. Endpoint presence is insufficient. | **Verified limits:** position docs describe `liqPrice` as estimated for cross margin, possibly empty for bounds and absent for portfolio mode; status includes `Liq`/`Adl` [Y5]. Exact liquidation waterfall, fees and practical offset protection unverified. | **Unknown:** trigger, maintenance rules, liquidation fees and ADL/loss-allocation details [O1]. |
| Hedge/position mode | **Limited:** deprecated source describes Hedge/One-way mode via `dualSidePosition` [B2]. Availability for a selected current product/account needs confirmation. | **Verified source:** mode page describes both sides versus one-way, default one-way, and support for USDT perpetual [Y4]. Other product/account combinations must be checked; page intro/table breadth is not a universal account entitlement. | **Unknown:** supported product/account combinations [O1]. |
| Minimum quantities; precision | **Verified spot schema only:** `LOT_SIZE`, `MARKET_LOT_SIZE`, `PRICE_FILTER`, `MIN_NOTIONAL`/`NOTIONAL` and quantity/price steps are documented [B4]. Actual symbol values and current perpetual filters unknown. | **Verified schema:** instrument page exposes derivative `minOrderQty`, `qtyStep`, `minNotionalValue`, price `tickSize`, and spot minimum-amount rules [Y2]. Actual symbol constraints and rounding residuals unknown. | **Unknown:** symbol constraints, multipliers and precision [O1]. |
| Geography/account eligibility | **Unknown:** legal entity, residence, KYC, derivatives/spot permissions and API eligibility were not checked. | **Unknown:** same; public docs do not establish this user's eligibility. | **Unknown:** same; product/entity restrictions need authoritative current terms. |
| API/market-data limitations | **Limited:** connector documents depth, exchange information and mark/funding routes, but is deprecated [B1, B3]. Funding-info docstring and implementation route disagree; do not use it as verified current behavior. Current rate limits, historical depth access and operational guarantees unknown. | **Verified source:** linear instruments may require pagination; documented default 500/page and maximum 1000; spot pagination differs [Y2]. History page limits and millisecond timestamps are documented [Y1]. Full rate limits, historical book availability, receipt-time provenance and estimates archive unknown. | **Unknown:** endpoint coverage, rate limits, depth history and timestamp guarantees [O1]. |
| Qualitative engineering complexity | **High:** conceptual two-leg/collateral ledger plus migration from incomplete/deprecated source evidence. Estimate is architectural judgment, not a measured implementation plan. | **High:** somewhat better documentary field coverage, but still two-leg/event-time accounting, venue rules and operational controls. No claim of simpler real trading. | **High and least bounded:** documentation access and specifications must precede any engineering estimate beyond this qualitative assessment. |
| Preliminary feasibility status | **NEEDS_MORE_EVIDENCE** | **NEEDS_MORE_EVIDENCE** | **NEEDS_MORE_EVIDENCE** |

Bybit has the strongest **accessible documentary coverage in this session**, not an established advantage in profitability, safety, eligibility or execution. A blocked documentation site is not evidence that a venue lacks a feature. No venue is selected.

## 5. Cost and capital accounting

### 5.1 Conceptual ledger and break-even condition

Use signed funding receipts minus payments, actual fills and a common quote-currency numeraire. For fixed equal linear base quantities:

```text
funding cashflow C_fund = Σ_k N_k × r_k       (normalized short-receives sign)
price/basis PnL         = q × (B_0 - B_T)
cash transaction costs = spot entry + spot exit + perp entry + perp exit fees
                         + execution friction + cash financing/borrow
                         + rebalance fees/friction + actual operational charges
Π_cash                  = C_fund + q × (B_0 - B_T) - C_cash
Π_economic              = Π_cash - opportunity cost of committed capital
```

Fees use each leg's traded notional and maker/taker status at each fill; not a single generic rate. Full four-fill fees are `Σ_i |notional_i| × fee_rate_i`, allowing only documented rebates. If spot is owned outright, short-spot borrowing is not inherently required; borrowing arises if cash/spot is financed or a different structure is chosen. Funding itself is already `C_fund` and must not also be counted as a financing expense. Margin is capital pledged, not an additional PnL expense; actual collateral interest and lost opportunity are separate costs.

**Spread/slippage convention:** either compute paired price PnL using a reference price and charge measured bid/ask crossing and implementation shortfall separately, or use actual executable fills and do not deduct those same frictions again. All scenarios below use reference prices plus an explicit all-fill friction allowance. Counterparty default or an outage loss is a stress scenario, not a made-up expected-cost probability; contingent reserves are not automatically expenses.

With `C_total = C_cash + opportunity cost`, economic break-even requires:

```text
Σ_k N_k × r_k >= C_total + q × (B_T - B_0)
```

Only under explicitly assumed constant `N`, equal rate `r` and `m` settlements:

```text
r_break_even = [C_total + q × (B_T - B_0)] / (m × N)
```

Amortizing fixed entry/exit costs over more settlements reduces their per-period share. Financing, rebalancing, funding sign and basis risk need not remain constant as holding time increases. Solve with time-dependent costs/rates; do not extrapolate a fixed daily yield or assume enough favorable settlements will occur.

### 5.2 Committed capital and efficiency

For non-netted accounts, committed capital includes purchased spot, dedicated derivative collateral, cash for fees and an operational/stress liquidity reserve. Use capital tied up across both legs and locations, with an explicit treatment of peak additional margin. A margin offset may reduce this denominator only when the specific legal entity/account documents and recognizes it. Do not count the same asset twice if it is valid shared collateral; do not omit spot just because the derivative uses leverage.

A transparent interval return is `Π_cash / K_committed` (and separately `Π_economic / K_committed` if opportunity cost is charged). Denominator convention, capital additions and valuation date must be stated. Capital efficiency cannot be assessed from funding divided by futures initial margin alone. No annualized return is asserted without a specified exposure path, capital schedule and observable funding history.

### 5.3 Illustrative scenarios only — no historical rates or observed outcomes

**All values in this subsection are invented accounting assumptions, not exchange quotes, observed funding, recommended parameters or acceptance thresholds.** Assume one base unit of spot at 10000 USDT and a short of one base unit in a linear perpetual at the same initial reference price; funding valuation notional held at 10000 USDT for the arithmetic. Assume ten abstract settlement periods, with no asserted exchange clock schedule. For the flat-basis illustrations, also hold fill notionals constant when calculating fees.

Assume spot fee 10 bps per fill, perpetual fee 4 bps per fill: two spot fills cost 20 USDT and two perpetual fills cost 8 USDT. These are **not verified venue rates** and do not alter the existing simulator. Add a 12 USDT all-fill spread/slippage allowance, 5 USDT cash financing and 3 USDT rebalance costs: total assumed cash costs **48 USDT**. Opportunity cost is excluded from the numbers and must be added for an economic-return comparison. Assume committed capital of 10000 USDT spot + 2000 USDT derivative collateral + 1000 USDT separate liquidity reserve = **13000 USDT**; any later cash injection must increase the capital calculation. The reserve is not an expense or automatic margin credit.

| Case | Assumed funding and basis path | Illustrative arithmetic and interpretation |
|---|---|---|
| Positive funding, still a loss | Ten settlements of +2 bps each; basis unchanged | Funding = 10000 × 0.0002 × 10 = 20. Net cash PnL = 20 - 48 = **-28 USDT**. Positive funding is insufficient. |
| Break-even/amortization | Basis unchanged; ten settlements; assumed costs frozen | Required average rate = 48 / (10 × 10000) = **4.8 bps per settlement**. At a hypothetical constant +2 bps, 24 settlements would cover 48 under a fixed-cost algebraic simplification. Actual costs/rates change with duration; this is not a forecast that holding longer makes the trade profitable. |
| Favorable hypothetical carry | Ten settlements of +6 bps each; basis unchanged | Funding = 60. Net cash PnL = **12 USDT**. Interval return = 12 / 13000 = **0.0923%** on assumed committed capital, not 12/2000 and not annualized. |
| Funding reversal | First five settlements +6 bps; next five -8 bps; basis unchanged | Funding = 30 - 40 = **-10**. Net cash PnL = -10 - 48 = **-58 USDT**. A short can pay funding. |
| Basis widening | Same assumed +60 funding; basis moves from 0 to +300 USDT per base unit | Paired reference-price PnL = **-300**; net = 60 - 300 - 48 = **-288 USDT** under the same simplified cost allowance. Future convergence is not assured. Real fill-dependent costs must be recalculated. |
| Margin exhaustion despite hedge | Stress spot rises from 10000 to 12000; perpetual mark rises to 12300; gains/losses are in separate accounts | Spot mark gain = +2000; short mark loss = -2300; aggregate basis loss = -300. The assumed 2000 isolated derivative collateral cannot fund that short loss, even with hypothetical +60 funding. Maintenance requirements would constrain survival before zero equity. The spot gain and external reserve are not assumed automatically available. This is not a venue liquidation-price calculation. |
| Partial/failed hedge leg | Spot fills one base unit; short fills only 0.7; common price then falls 10% with unchanged basis | Residual long delta = 0.3 base units; price exposure alone loses **300 USDT**. If the short never fills, spot price exposure alone loses **1000 USDT**. Do not apply the fully hedged funding/cost totals unchanged to this case. Emergency completion/unwind costs and timing need separate accounting. |

These examples compare algebraic cashflows, not historical trades or trading performance. No actual capital amount, leverage level, instrument or entry rule is proposed.

## 6. Data feasibility

### 6.1 Minimum point-in-time evidence for any future protocol

| Required input | Why it is needed | Current evidence/limitation |
|---|---|---|
| Published funding estimates, revisions and delivery timestamps | Reconstruct what was knowable before entry, not just the final rate | Bybit ticker schema contains funding and next-funding fields [Y7]; that does not establish an archived estimate/revision series. Unverified for all three venues. |
| Realized settlement rates and actual credit/debit conventions | Reconcile funding units, sign, valuation and settlement eligibility | History interfaces documented for Bybit and in deprecated Binance source [Y1, B1]; no records collected, no retention/completeness assurance. Actual user cash ledgers not accessed. |
| Publication, exchange-event, local receipt and settlement timestamps | Causality, latency, cutoff inclusion and revisions | Timestamp field documentation is incomplete evidence for point-in-time observability; receipt/revision lineage remains missing. |
| Executable spot and perpetual bid/ask, depth and contract multipliers | Match quantities and charge executable four-leg costs | No order book or trade data collected; current schema metadata is not historical executable depth. |
| Mark/index and comparable basis observations | Funding valuation, risk marks, divergence and settlement references | Bybit schemas expose mark/index fields [Y5, Y7]; no mark/index or basis history inspected. Basis must be computed from synchronized like-for-like prices in a future authorized study. |
| Time-effective maker/taker fee schedules and discounts | Fee budget for every fill, instrument and tier | Account-specific API schemas are not verified fee schedules for this user [B2, Y3]. |
| Margin tiers, collateral haircuts, financing/borrow rates and eligibility | Capital feasibility, financing cashflows and liquidation stress | Product/account rules and lawful access incomplete. No private account information retrieved. |
| Data completeness and integrity specification | Detect gaps, duplicates, outages, timestamp drift, stale books, delistings and symbol migrations | No completeness claim can be made from documentation alone. Snapshot versions, timestamp units, licenses, corrections and missingness policy must be specified before access. |

A final-rate history cannot substitute for a contemporaneously published estimate history in entry decisions. Documentation examples are not market observations and have not been used as funding evidence. Public access to an endpoint does not prove lawful use, archival rights, sufficient retention or permission to inspect a reserved period.

### 6.2 Existing infrastructure suitability and conceptual gaps

The [OHLCV helpers](../../app/research/data_utils.py) and the single-position simulator [R3, R4] are not sufficient. Candle aggregation may support a reference-price context but cannot reconstruct funding information availability, two books, intraperiod margin calls or partial hedge fills. The current 0.0004 per-side default is not a complete four-leg/account-specific carry budget.

Conceptual extensions, **not implemented or authorized**, would include an immutable point-in-time event ledger; separate funding-estimate and final-settlement streams; spot/perpetual inventory, contract multiplier and financing ledgers; separate collateral/margin location and stress accounting; and conservative partial-fill/latency/executable-depth semantics. Each requires a separately approved specification and data plan. No new dependency, infrastructure, connector, schema or code is added here.

No carry result is written into Research Memory. Existing ID, experiment and report writers are unsuitable as implicit permission to start this track. A future protocol must define its independent sampling unit (for example, a hedged holding episode rather than each correlated settlement), uncertainty treatment and stopping rules. The [existing baseline gate](../../app/research/governance/verdicts.py) [R7] is a historical strategy verdict policy, not an automatic carry feasibility or statistical-significance rule. No numerical sample requirement is invented here.

## 7. Risk register

“Measurable” means potentially measurable under a future approved data/account specification, not measured in this task. Mitigation cannot be assumed implemented, costless or successful.

| Risk | Evidence needed to measure it | Possible mitigation requiring separate approval | Structural remainder |
|---|---|---|---|
| Funding sign reversal | Point-in-time estimates, revisions, settlements and sign reconciliation | Conservative exit/holding policy and budget for payments | Future funding is uncertain; closing itself costs money. |
| Basis divergence | Synchronized executable basis and mark/index stress path | Position limits and collateral/liquidity buffers | No fixed perpetual expiry guarantees convergence. |
| Liquidation despite hedge | Maintenance tiers, marks, account location, haircuts and cross-offset rules | Adequate collateral and feasible transfer/reduction policy | Spot gains elsewhere may be unavailable when needed. |
| Collateral/margin exhaustion | Available-versus-pledged balances, financing and gap stress | Precommitted liquidity reserve with location constraints | Cash calls can exceed reserves; leverage amplifies failure. |
| Exchange default/custody failure | Legal claims, custody segregation, withdrawal constraints and recovery terms | Exposure limits and diversification assessed with all extra costs | Counterparty and legal recovery risks cannot be eliminated by delta neutrality. |
| Trading halts or delisting | Halt/settlement rules and historical coverage specification | Approved emergency hedge/unwind plan | No guaranteed ability to execute both legs or withdraw. |
| Failed/partial hedge execution | Per-fill quantity, receipt timestamps, cancellations, rejection and rounding | Bounded residual exposure and reconciliation controls | Leg execution is not atomic; emergency liquidity may disappear. |
| Liquidity deterioration | Depth at executable size, spreads and stressed implementation shortfall | Conservative size/cost model, conditional refusal to trade | Displayed depth is cancellable; market impact is uncertain. |
| Borrowing/financing exposure | Loan tenor, availability, variable interest, recall and collateral terms | Prefer demonstrably financeable structure; maintain liquidity budget | Financing costs/availability can change; opportunity cost persists even without a loan. |
| API/operational outage | Sequencing, clock drift, stale-data detection, rate limits and outage procedures | Readiness checks, reconciliation and authorized fail-safe design | Software/network/exchange outages and delayed observations remain possible. |

No reserve size or risk limit in this register is prescribed. Unknown probabilities are not converted into invented expected-return adjustments. Risk mitigations themselves require economic and operational feasibility review.

## 8. Feasibility gates

These are **preliminary documentary feasibility classifications**, not authoritative research verdicts, statistical acceptance rules, strategy approvals or permission to run anything.

- **FEASIBLE_FOR_PREREGISTRATION:** every applicable dimension below has supporting, internally consistent and current documentary evidence, with a human-approved nonconflicting lawful data-access plan and feasible all-leg cost/capital/operational specification. There are no unresolved critical eligibility, point-in-time or execution gaps. This only supports a human decision to draft a protocol; it does not establish a positive edge or authorize execution.
- **NEEDS_MORE_EVIDENCE:** one or more critical requirements is unknown, unverifiable or conflicted, but no documented fatal incompatibility has been established. Unknown is never scored as passed. This is the current outcome for all venues.
- **NOT_FEASIBLE:** reliable evidence establishes an unresolvable incompatibility with the approved scope, law/account access, required data observability, execution/capital constraints or governance. Examples are documented ineligibility, no lawful causally valid data route, or unavailable required hedging/collateral arrangements. A blocked page alone does not establish a fatal feature absence. No quantitative hurdle or fatal finding is invented for this assessment.

| Dimension | Supporting evidence | Critical missing evidence and relevant risk | Current dimension assessment |
|---|---|---|---|
| 1. Independent economic mechanism | Funding cashflow plus paired basis accounting differs from price-pattern strategies [R1] | No observed cost-adjusted carry edge; cash-sign definitions incomplete | Conceptually distinct only; economic viability unproven for all venues. |
| 2. Complete cost accounting | Four-fill, financing, basis and capital framework in section 5 | Actual fee tiers, execution shortfall, financing and operational costs | Not satisfied for any venue. |
| 3. Accessible and lawful data | Partial public documentation and source access [B1–B4, Y1–Y7] | User/product eligibility, licensing, archival access, retention and OKX documentation | Unknown, not satisfied. Documentation access is not data authorization. |
| 4. Point-in-time reproducibility | History/ticker timestamp schemas for Bybit, limited Binance history metadata | Pre-entry estimate/revision archives, receipt times and completeness | Not established for any venue. |
| 5. Realistic execution assumptions | Quantity/filter schemas and partial mode support | Executable two-leg fills, latency, partial quantities, conservative costs and emergency policy | Not established; current single-leg simulator is insufficient. |
| 6. Capital and margin feasibility | Some margin/mode fields and illustrative stress accounting | Current maintenance tiers, offsets, collateral location, reserves and available capital | Unknown for all venues; no private account access. |
| 7. Research governance compatibility | Parallel documentation track is authorized; existing boundaries unchanged [R5–R7] | A human-approved carry data plan and any future protocol are absent | Documentation scope compatible; future access/execution not approved. |
| 8. Operational complexity | Two-leg ledger, collateral location and outage risks identified | Feasible operational specification and legal/product constraints | High qualitative complexity for all; OKX uncertainty greatest due lack of inspected rules. |

**Decision matrix:** Binance = NEEDS_MORE_EVIDENCE (deprecated/incomplete derivatives source; fees, sign, margin and eligibility unknown). Bybit = NEEDS_MORE_EVIDENCE (better schema coverage but no confirmed all-leg economics, point-in-time archive or lawful account access). OKX = NEEDS_MORE_EVIDENCE (primary rule documentation inaccessible; no detailed rules verified). No venue meets all gates; no venue is declared NOT_FEASIBLE from missing evidence alone.

## 9. Evidence gaps

Prioritized documentary gaps, not authorization to retrieve account or market data:

1. Obtain accessible current official product rules for funding sign, valuation, settlement timing and schedule changes on the exact legal entity/contract. For Binance, replace the deprecated connector as authoritative product-rule evidence; for OKX, obtain inspectable primary documentation first.
2. Establish lawful user/entity/product eligibility and permitted API/data use without assuming residence or access. No private-account check is made here.
3. Document effective spot and derivative maker/taker schedules, discounts, all-fill costs, financing terms, minimum quantities and rounding; reconcile funding cash sign separately from public rate sign. No fixed public tariff is substituted for account-specific terms.
4. Establish whether contemporaneous funding estimates/revisions and executable depth can be archived and reproduced lawfully, with documented retention and timestamp quality. Final funding history alone is insufficient.
5. Establish current margin tiers, collateral haircuts, netting/offset and liquidation/loss-allocation rules; verify that liquidity can remain where the short needs it under a basis/price shock.
6. Present a conservative conceptual execution and capital budget with explicit failed-leg handling and independent sampling units. Any capital amounts or limits require human approval; the toy scenarios are not defaults.
7. Obtain a separate human decision on an eligible, nonreserved data plan before any future preregistration. Do not inspect market outcomes to fill these documentary gaps or choose a favorable period.

## 10. Recommendation and next human decision

**B — Investigate specific missing documentary evidence.** Do not yet proceed to a substantive preregistration, implementation or market experiment. The independent economic mechanism warrants keeping the documentary track open; the current evidence does not establish viability or require stopping it.

The next human decision is whether to authorize a further **documentation-only** clarification of exact product/legal-entity rules and lawful point-in-time data availability. Bybit's accessible schemas can serve as a documentation checklist, not a preferred trading venue. Binance needs current derivative rules beyond a deprecated SDK; OKX needs accessible primary documentation. No account connection, historical data collection, data period, experiment ID or test is authorized by this recommendation.

If critical requirements become evidenced and compatible, a human may separately decide whether to draft a preregistration. If reliable documentation instead establishes ineligibility, unavailable valid data or infeasible hedge/capital arrangements, classify that proposed setup NOT_FEASIBLE rather than relax constraints to rescue it. Stop this task after delivering the report. Existing scalping research and rejected hypotheses remain unchanged.

## 11. Sources and verification date

All source retrieval attempts below occurred **2026-10-19**. Static documentation source files only were retrieved, with TLS verification retained. No live funding, ticker, order-book, fee/account, instrument-state or trading endpoint was called. Numeric response examples embedded in documentation were not used as observations or inputs to the assessment.

### Repository sources

- **R1:** [TASK-006 assessment](strategy_opportunity_assessment_v1.md), at repository baseline `1a9bee3`. Provides conceptual motivation and explicit missing-data/cost constraints, not carry profitability evidence.
- **R2:** [Historical failure report](failure_patterns_exp_000021_000024.md), unchanged SHA-256 `42411a9a93199ee4f401ce7cb4227a48cfea7b28d0ebabfa55cb434ee7cf2b76`.
- **R3:** [Simulator](../../app/research/simulation.py): fixed-notional, one-position price-PnL accounting and explicit two-side fees.
- **R4:** [Research defaults](../../app/research/config.py) and [OHLCV helpers](../../app/research/data_utils.py): existing fee constant and candle-oriented infrastructure.
- **R5:** [Regime-transition data restrictions](../hypotheses/HYP-REGIME-TRANSITION-001.md) and [Donchian reserved confirmation protocol](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md).
- **R6:** [Research Memory store](../../app/research/memory/experiment_store.py): journal/index structure, not invoked.
- **R7:** [Authoritative baseline policy](../../app/research/governance/verdicts.py): existing policy remains unchanged, not a carry feasibility gate.

### Accessible official source snapshots

These URLs were retrieved successfully (HTTP 200). Bybit official documentation repository HEAD was `2bad003ac0000c36fe3b7172cd1756d88f5a29ce`; Binance spot specification HEAD was `f0ee79c8ce24dbeda6645211b9504e237fddef19`; Binance deprecated connector HEAD was `a6bfbbf10fe2c1b4eb76fc24ffb82eb94bf9df89`. Retrieval proves document contents at those commits, not operational deployment or current account terms.

- **B1:** [Binance USD-M market metadata source](https://github.com/binance/binance-futures-connector-python/blob/a6bfbbf10fe2c1b4eb76fc24ffb82eb94bf9df89/binance/um_futures/market.py), especially `funding_rate`, `funding_info`, mark-price and depth descriptions. `funding_info` docstring names `/fapi/v1/fundingInfo`, but its implementation calls `/fapi/v1/fundingRate`; this inconsistency limits reliance on the SDK, rather than establishing the correct live behavior.
- **B2:** [Binance USD-M account source](https://github.com/binance/binance-futures-connector-python/blob/a6bfbbf10fe2c1b4eb76fc24ffb82eb94bf9df89/binance/um_futures/account.py), mode, margin/leverage/brackets and signed commission-route descriptions only. No account request was made.
- **B3:** [Binance connector status](https://github.com/binance/binance-futures-connector-python/blob/a6bfbbf10fe2c1b4eb76fc24ffb82eb94bf9df89/README.md): explicitly deprecated, refers readers to a new modular connector. No connector was installed.
- **B4:** [Binance spot filters](https://github.com/binance/binance-spot-api-docs/blob/f0ee79c8ce24dbeda6645211b9504e237fddef19/filters.md): price/quantity/minimum-notional schema; sample values are not current instrument limits.
- **Y1:** [Bybit funding history specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/market/history-fund-rate.mdx): per-symbol intervals, history request semantics and pagination. Its eight-hour example is an example, not a universal settlement rule.
- **Y2:** [Bybit instruments specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/market/instrument.mdx): funding interval, leverage/quantity/price constraints, spot minimum amount and pagination. No live symbol metadata queried.
- **Y3:** [Bybit fee-rate specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/account/fee-rate.mdx): maker/taker field definitions, not verified fee values.
- **Y4:** [Bybit position-mode specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/position/position-mode.mdx): one-way/both-sides descriptions and product support text; account/product confirmation still required.
- **Y5:** [Bybit position specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/position/position.mdx): margin/mark/liquidation fields, account-mode caveats and ADL status. Not a complete liquidation waterfall.
- **Y6:** [Bybit transaction-log specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/account/transaction-log.mdx): signed funding receipts/payments versus signed trading fees; warns that USDC session PnL and funding can share a record but remain distinct fields.
- **Y7:** [Bybit ticker specification](https://github.com/bybit-exchange/docs/blob/2bad003ac0000c36fe3b7172cd1756d88f5a29ce/docs/v5/market/tickers.mdx): funding/next-funding/mark/index metadata; not an archive of point-in-time observations.

### Inaccessible primary documentation targets

- **B0:** [Binance current funding-history documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History): CONNECT HTTP 403; no page content verified.
- **Y0:** [Bybit public documentation page](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate): CONNECT HTTP 403; the pinned official source Y1 was accessible instead.
- **O1:** [OKX API V5 official documentation](https://www.okx.com/docs-v5/en/): CONNECT HTTP 403; no OKX rule content verified. No account eligibility, fee rate, sign convention, funding schedule, margin rule, minimum order size or API retention/rate limit is supplied without evidence.

Verification of this change is documentary: local path resolution; source traceability and scope labels; independent arithmetic checks of the invented examples; unchanged source-report hashes and protected repository paths; Markdown consistency; and a one-file diff with `git diff --check`. None establishes empirical carry profitability or permission to proceed beyond documentation.

Completed checks: 18 local path citations resolved; 11 commit-pinned official source snapshots were cross-checked against retrieved documentation; three blocked primary-page attempts were explicitly labeled; illustrative arithmetic and all 11 section/table structures passed; prior report bytes and historical source hash remained unchanged. Only this Markdown report was added, and `git diff --check` passed. No application test or market/statistical experiment was run.
