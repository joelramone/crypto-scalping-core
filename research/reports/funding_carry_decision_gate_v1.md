# Funding Carry Decision Gate V1

## 1. Executive decision summary

**Recommendation: HOLD_PENDING_EXTERNAL_EVIDENCE.** All three critical gates are **UNRESOLVED**. Existing repository evidence does not justify preregistration review; it also does not establish a verified incompatibility warranting STOP_FUNDING_CARRY. Missing information is neither PASS nor BLOCKED.

Decision date: **2026-10-22**. The inspected `main` includes **PR #183**, merge commit `7db96b2012fe553feb6173283b81dad20d2e18c7`, integrating TASK-008 (`658451f`). This assessment uses only the three existing reports below, without refreshing external sources or repeating their documentary investigations. Their source verification dates remain their own; this report does not imply that exchange rules were reverified today.

Scope remains a conceptual long-spot/short-linear-USDT-perpetual hedge. **Execution is NOT AUTHORIZED.** No strategy, symbol, leverage, capital amount, data period, hypothesis or experiment ID is selected. No market data, private account, reserved dataset, backtest, optimizer or Research Memory writer was accessed. Existing reports and governance remain unchanged.

## 2. Evidence already established

- [TASK-006: opportunity comparison and recommendation](strategy_opportunity_assessment_v1.md#speculative-comparison-and-recommendation-for-human-review) identifies funding/basis compensation as conceptually distinct from rejected candle-pattern strategies. It explicitly provides no empirical evidence of carry profitability and notes the scope change beyond scalping.
- [TASK-007: accounting framework](funding_basis_carry_feasibility_v1.md#5-cost-and-capital-accounting) separates funding cashflows, basis PnL, four-fill costs, financing and committed capital. Its numeric examples are assumptions, not Bybit tariffs, returns or sufficient capital requirements. [Its feasibility gates](funding_basis_carry_feasibility_v1.md#8-feasibility-gates) were not satisfied.
- [TASK-008: funding and fee evidence](bybit_funding_carry_evidence_gap_v1.md#3-funding-mechanics) documents narrow cash-ledger sign conventions and interface definitions. Its [fee discussion](bybit_funding_carry_evidence_gap_v1.md#4-trading-fees) clarifies conditional fee currency without establishing applicable rates. [Margin evidence](bybit_funding_carry_evidence_gap_v1.md#5-margin-and-liquidation) does not establish a complete liquidation or capital model.
- [TASK-008: Argentina](bybit_funding_carry_evidence_gap_v1.md#6-argentina-eligibility) records an official conditional route for an already qualified “Argentina international account.” All seven legal/product/account eligibility questions remain UNKNOWN. Regional routing is not legal permission, account authorization or a product entitlement.
- [TASK-008: point-in-time data](bybit_funding_carry_evidence_gap_v1.md#7-point-in-time-data-feasibility) distinguishes documented settled funding history and live interfaces from unestablished historical estimate/revision, executable depth and rule-version archives. [Its gap matrix](bybit_funding_carry_evidence_gap_v1.md#8-evidence-gap-matrix) closes no entire critical prerequisite.

These are documented findings and limitations, not independently revalidated exchange facts. Previously blocked documentation pages are an evidence-access limitation, not proof of feature absence or user ineligibility.

## 3. Three-gate decision matrix

**PASS** requires affirmative evidence sufficient for the stated gate. **BLOCKED** requires a verified restriction or structural incompatibility. **UNRESOLVED** applies when critical evidence is missing or incomplete. Passing a documentary gate would not authorize implementation or execution.

| Gate | Evidence sufficient in the repository? | Classification | Reason and condition needed to reconsider |
|---|---|---|---|
| 1 — Legal and product eligibility | No. Conditional regional API guidance is documented; Argentina residence/KYC, governing entity, spot, USDT-perpetual and API permissions are not established (TASK-008 §6). | UNRESOLVED | Current authoritative terms must establish lawful access for the relevant resident/entity/product scope, with any remaining account-specific authorization requirement explicitly distinguished. Website/API visibility cannot satisfy this gate. No applicable restriction is verified, so BLOCKED is not justified. |
| 2 — Economic and operational feasibility | No. Conceptual cash/basis accounting and field definitions exist, but full funding valuation/inclusion, applicable spot/perpetual fees, financing, margin/collateral, liquidation and capital requirements remain incomplete (TASK-007 §5; TASK-008 §§3–5, 8). | UNRESOLVED | Current effective rules must support a cost-complete, realistic two-leg ledger and a feasible collateral/operational specification. A symbolic framework or a toy capital amount is not enough. No invented rate, automatic spot-to-margin offset or leverage assumption may fill the gaps. |
| 3 — Research data feasibility | No. Public settled-rate history and live book/ticker interfaces are documented, but lawful reproducible historical estimates/revisions, synchronized executable prices/depth, receipt timestamps and contract/rule versions are not established (TASK-008 §7). | UNRESOLVED | An authoritative source specification must establish permitted use, point-in-time lineage, coverage, retention and completeness for all required inputs, compatible with reserved periods. Final settlements, current metadata and candles cannot replace pre-entry evidence. |

**Consolidation:** 0 PASS, 0 BLOCKED, 3 UNRESOLVED. Therefore PROCEED_TO_HUMAN_PREREGISTRATION_REVIEW is unsupported. STOP_FUNDING_CARRY is also unsupported because no fatal restriction or incompatibility is verified. The resulting decision is **HOLD_PENDING_EXTERNAL_EVIDENCE**.

## 4. Critical evidence still missing

1. **Legal/product scope:** authoritative current jurisdiction/KYC and governing-entity terms establishing whether an Argentina resident can lawfully use the required spot and USDT perpetual products. Account-specific permission, if necessary, remains distinct from public documentation; no account inspection is authorized here.
2. **Complete economics and survival rules:** the public-rate-to-cashflow funding formula, payer/receiver convention, settlement inclusion/valuation and deductions; applicable time-effective four-fill fees and financing; collateral eligibility/haircuts/offsets, maintenance and liquidation rules; and capital/transfer constraints. Without them, the repository cannot specify a complete conservative budget or survival model.
3. **Lawful point-in-time provenance:** access and use rights, historical estimate/revision availability, settlement coverage, executable two-leg price/depth records and timestamp lineage, plus time-effective contract/fee/margin rule versions. A documented API field is not proof of an archive or its completeness.

A prospective collection design is only a **conditional future option** if suitable historical evidence cannot be established. The repository does not prove that prospective collection is viable or the only option. Any such option requires separate human authorization, a lawful source/retention plan and a nonconflicting data-period decision **before** collection. It does not release reserved dates, permit backfilling market data now or convert live interfaces into historical evidence.

## 5. Smallest external evidence package required

Request evidence supporting the three decisions, rather than another broad inventory or a market performance test. Each item needs an authoritative issuer, applicable legal entity/product/account scope, verification/effective date, source reference and explicit limitations. No API key, identity document, private account access or market-data sample is requested by this task.

| Package component | Minimum decision-relevant material | Gate affected |
|---|---|---|
| A — Eligibility determination | Current official terms or authoritative clarification for Argentina residency, governing entity, KYC and spot/USDT-perpetual/API product permissions, stating where an individual authorization would still be required. Public general access must not be presented as personal permission. | 1 |
| B — Economic/operational rule set | Effective funding settlement/calculation and fee/financing rules, together with collateral, margin, liquidation and transfer constraints sufficient to specify a consistent four-fill/cashflow/capital ledger. Applicable tiers and unresolved account-specific dependencies must be explicit; no invented numbers are needed. | 2 |
| C — Data-access/provenance specification | Authoritative historical data-product documentation or provider confirmation covering estimates/revisions, settlements, executable prices/depth/timestamps, contract/rule versions, retention/completeness and lawful research/archival use. If unavailable historically, document that limitation and a proposed prospective option for a **separate** human decision, without starting capture or selecting dates. | 3 |

A packet that leaves a critical item unknown does not pass the corresponding gate. No historical outcomes or demonstrated profits are necessary to supply this documentary package, and a profitability claim cannot replace it. Repository evidence alone cannot resolve these remaining external facts.

## 6. Recommendation for research prioritization

**Park funding-carry preregistration and implementation work pending the external package.** Do not allocate further effort to repeating the same API-schema inventory, simulator extensions or market testing while these gates remain unresolved. Retain the conceptual opportunity without treating it as validated or permanently rejected.

Eligibility and product/rule evidence should be obtained before committing engineering effort; data provenance is equally necessary and must not be deferred until after a test. TASK-006's earlier preference for reviewing carry was a documentary prioritization, not an obligation to proceed. This hold does not authorize switching to another speculative candidate, revisiting rejected scalping hypotheses or changing the existing research track. No new strategy-selection recommendation is made.

## 7. Next human decision

Choose whether to **seek components A–C through an authorized documentary channel or leave the carry track parked**. This report does not request or grant account access, authorize data acquisition, designate a period or initiate preregistration.

When material external evidence arrives, reassess only the affected gates against the source versions and their applicability. Proceed to human preregistration review only if all critical gates PASS; recommend stopping the proposed setup only if authoritative evidence establishes an actual BLOCKED condition. Otherwise retain HOLD_PENDING_EXTERNAL_EVIDENCE. Even all gates passing would still require a separate human decision before any implementation, data collection, experiment or capital deployment.

Documentary verification for this change checks local file/section references, consistency of the three UNRESOLVED classifications with TASK-008, the consolidation rule, preservation of the three source reports and the one-file diff. No new web investigation or empirical research was performed. `git diff --check` must pass. Stop after delivering this assessment.

Verification completed: nine local file/section references resolved; three UNRESOLVED gates and the HOLD consolidation matched the prior evidence; all three source reports remained byte-identical; the seven requested sections and table structure passed. Only this Markdown file was added, and `git diff --check` passed.
