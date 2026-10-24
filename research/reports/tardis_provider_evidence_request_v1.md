# Tardis Provider Evidence Request Package V1

## 1. Purpose and current research status

Prepare documentary questions and an **unsent** English email for human review concerning Candidate A — aggressive order flow relative to available liquidity. No provider investigation or communication is performed in this task. Preparation is not receipt or verification of evidence.

Baseline verified 2026-10-24: latest fetched `main`, `8f900df5b2d6f1be63101541b38a2e68f2366cdf`, integrates TASK-012 through **PR #187**, including commit `96f9e9e92d6cfd46ccf2a42cab94f36e9e9cb27a`. Authoritative gap source: [TASK-012 dossier](tardis_microstructure_evidence_dossier_v1.md), especially §§3–7 and 10. Prior context: [TASK-011](microstructure_data_execution_feasibility_v1.md) and [TASK-010](scalping_new_edge_discovery_v1.md).

**Current, unchanged: TASK-012 HOLD_PENDING_PROVIDER_EVIDENCE; Gates A/B/C UNRESOLVED.** TASK-011 remains HOLD_PENDING_EXTERNAL_EVIDENCE; TASK-010 remains NO_CANDIDATE_READY; [funding carry](funding_carry_decision_gate_v1.md) remains HOLD_PENDING_EXTERNAL_EVIDENCE; [EXP-000021–024](failure_patterns_exp_000021_000024.md) remain closed/rejected. No edge or profitability is established.

BTCUSDT is a conceptual scope reference only. Neither this document nor its questions selects a venue, spot/perpetual product, historical research period or trading parameters. Requests for coverage metadata are not requests to inspect outcomes or receive market-data samples. The [existing data boundaries](../hypotheses/HYP-REGIME-TRANSITION-001.md#data-boundaries) and [reserved confirmation protocol](../hypotheses/HYP-DONCHIAN-HIGHVOL-001.md#confirmation-dataset-and-evaluation-gate) remain binding: 2025 discovery-used, partial 2026 already used, later data reserved; another provider or schema does not release those restrictions.

## 2. Provider questionnaire

Eight grouped questions, ready to accompany the email. Please answer by ID with current documentation or an attributable written clarification, including effective date, product applicability and limitations. If something is unsupported, unavailable or not guaranteed, state that explicitly. **Documentation only: please do not send datasets, market-data samples or credentials.** Catalogue coverage dates may be described without choosing a research interval. “Acceptance” below means sufficient evidence for that question, not automatic gate PASS.

### Package A — Historical coverage and integrity

| ID | Precise English question | Documented gap | Expected documentary response | Acceptance criterion | Gate affected |
|---|---|---|---|---|---|
| A1 | Can you identify, separately for spot and perpetual products, how a common venue/product/native-symbol scope can be documented as having historical individual trades, native trade IDs, aggressor-side information, L2 snapshots and incremental updates? Please describe channel-specific coverage, depth limits, native-to-normalized exclusions/unknown side values, and versioned symbol/contract specifications, including effective tick/lot rules. BTCUSDT is illustrative only; please do not select a research market or period for us. | Dossier §3: paired coverage, trade identity/side, product separation, instrument rules | Product/channel coverage statement or catalogue documentation and schema/mapping references | Same-scope trade/book coexistence and historical identity/rule mapping are explicitly supported, with coverage boundaries, exclusions and limitations; SDK types or generic marketing alone are insufficient | A; C inputs |
| A2 | How are initial snapshots aligned with native sequences and subsequent updates, and how are resets, resynchronizations, duplicates, delayed/out-of-order messages, disconnects, known capture gaps and missing-data records exposed? Please distinguish captured messages from repaired/reconstructed content and describe documented integrity limitations. | Dossier §§3–4, 6: alignment, silent gaps, reset lineage | Reconstruction/integrity specification plus documentary description of gap/reset/correction records, without sample payloads | Native sequence/snapshot rules, recovery boundaries and detected versus undetectable missingness are explicit, including how unusable intervals are identifiable; no inferred gap-free guarantee | A; C inputs |

### Package B — Timestamp provenance

| ID | Precise English question | Documented gap | Expected documentary response | Acceptance criterion | Gate affected |
|---|---|---|---|---|---|
| B1 | For trades, snapshots and deltas, what exactly do exchange timestamp and localTimestamp represent, including units, fallback origins, capture location and timestamping boundary? How are capture clocks synchronized, and what accuracy/uncertainty or known limitations apply? Which timestamp, if any, is suitable for reconstructing information available to an external observer, and under what observer/location assumptions? | Dossier §4: event versus capture semantics, clock lineage and observer availability | Timestamp/capture specification with clock-method and uncertainty statement | Meanings and fallback cases are distinguished by channel/product; clock precision is separated from accuracy, and any observer-availability claim has stated assumptions/limits rather than equating capture with trader receipt | A; C inputs |
| B2 | What ordering or synchronization guarantees exist across trade/book channels, including equal timestamps, batching, publication delay and delayed/out-of-order arrivals? When are historical corrections published, and how can a researcher distinguish information originally available from later repairs? Please identify where no causal ordering guarantee exists. | Dossier §§4, 6: cross-channel causality, publication and revision timing | Ordering/publication policy and correction-time/availability documentation | Supported ordering, ties, delay limits or explicit lack of guarantees and original-versus-later knowledge are identifiable; sorting by exchange time alone is not accepted as causal proof | A; B repeatability; C inputs |

### Package C — Licensing, access and commercial terms

| ID | Precise English question | Documented gap | Expected documentary response | Acceptance criterion | Gate affected |
|---|---|---|---|---|---|
| C1 | Please provide the applicable market-data license and clarify personal versus commercial research permissions, internal archival and post-subscription retention, derived analytics/publication rights, collaborator access and redistribution restrictions, including exchange-specific conditions. Which user/use classifications require separate permission? We are not asserting a particular classification; the SDK's MPL-2.0 license is distinct from data rights. | Dossier §5: dataset license and use/retention/derived-data rights | Current data agreement, relevant clauses and written applicability clarification | Rights and restrictions for each intended use are attributable and effective-dated; unresolved user classification is flagged for human/legal review; software licensing is not substituted for a data license | B |
| C2 | Which current plans could include the documented trade/L2 scope, and what exchanges, products, channels and historical coverage, user/access limits and authentication requirements do they include? Please provide pricing with billing currency/unit, applicable taxes or exclusions, quote/effective date, validity and any trial/history restrictions, without opening an account or provisioning access. | Dossier §5: entitlement, authentication and price unknown | Dated plan/entitlement documentation or nonbinding written quote | Scope and limits are explicit and prices have currency, billing unit, date and qualifications; no trial or uncredentialed example is treated as unrestricted entitlement; no subscription accepted | B |

### Package D — Reproducibility

| ID | Precise English question | Documented gap | Expected documentary response | Acceptance criterion | Gate affected |
|---|---|---|---|---|---|
| D1 | Are immutable archive IDs or dataset revision IDs available, with access to original versus corrected history, published correction records and content checksums/manifests? Is repeating an identical historical request later guaranteed to return identical data; if not, how are changes identified and superseded versions retained or recoverable? | Dossier §6: immutable bytes, corrections and manifests | Version/correction policy describing identifiers, content-integrity metadata and superseded access | A documented method pins the same historical message content and its correction lineage, or explicitly states the absence/limits of that capability; request hashes alone do not establish content identity | B; A lineage |
| D2 | How are schema versions, symbol/channel mappings and API/replay-client compatibility tied to historical archive revisions? What retention and re-access commitments allow two separately authorized researchers to retrieve the same pinned messages, and what subscription, withdrawal or compatibility limits can prevent this? | Dossier §§5–6: mapping versions, retention and repeatability | Version/compatibility and retention/re-access policy with entitlement dependencies | Interpretation and retrieval identifiers are version-linked; retention/re-access scope and limits are explicit for independent licensed users. A local cache or pinned SDK alone is insufficient | B; A interpretation |

## 3. English email draft

**Subject:** Documentary inquiry: historical trade/L2 provenance, research licensing and reproducibility

Hello Tardis team,

I am exploring the feasibility of future research on aggressive order flow relative to available liquidity. Before considering data access, I would like to establish whether historical individual trades and L2 snapshots/updates can be paired, interpreted causally and reproduced under appropriate research rights.

Could you please review the eight grouped questions included below or attached (A1–D2) and provide current documentation or written clarification? They cover historical coverage/integrity, timestamp provenance, market-data licensing/access/pricing, and archive versioning/repeatability. Please distinguish spot and perpetual products where relevant. BTCUSDT is illustrative; no venue, product or research period has been selected.

In particular, please clarify personal versus commercial research and archival/derived-analytics rights, applicable plans and entitlements, and current pricing with currency, billing unit, effective date and limitations. Explicit statements of unsupported features or absent guarantees would also be helpful.

This is an exploratory documentary inquiry only. Please do not send market-data samples, credentials or provision access; no account creation, subscription or acceptance of terms is requested.

Thank you for your guidance.

Before sending, the human sender should add their own signature, select the provider's appropriate contact route and include the questionnaire from §2. No sender identity, affiliation, address, existing subscription or commercial relationship is asserted here. **Draft only; not sent.**

## 4. Evidence acceptance matrix

This matrix governs a **future human review**, not today's classifications. Every response must have an attributable publisher/authorized representative, document title/URL or written-message reference, version and effective/verification dates, applicable scope, conditions and limitations. Undated, generic, contradictory or unauthenticated assertions remain UNRESOLVED pending clarification. A dated price does not establish usage rights; a sales statement does not override restrictive applicable terms.

| Gate | Documentary evidence sufficient for PASS at future review | Missing information → UNRESOLVED | Verified incompatibility → BLOCKED | Human dependency |
|---|---|---|---|---|
| A — Historical provenance | A1–A2 and B1–B2, with relevant D1–D2 lineage: coherent paired trade/L2 coverage, native identity/side/rules, snapshot alignment, sequence/reset/missingness and timestamp semantics sufficient to specify causal replay, with bounded or explicitly identifiable unusable information | Unconfirmed same-scope coverage, ambiguous clocks/order, silent-gap behavior or correction provenance; inaccessible documents are missing evidence | Authoritative confirmation that no permissible relevant scope supplies required paired historical trades/L2, or that indispensable causal lineage is irretrievably absent and no conservative valid reconstruction is possible | Human must judge whether stated uncertainty permits a causal specification without inventing missing events; metadata availability does not select a period |
| B — Lawful access and reproducibility | C1–C2 and D1–D2, with B2 revision timing: applicable research/archive/derived-use rights, licensed entitlement/access and dated commercial terms, plus repeatable content and interpretation identifiers, corrections and retention/re-access commitments | Unknown use classification, conditional unsigned permissions, incomplete plan scope, price/rights conflict, missing retained revisions or repeatability commitment | Applicable authoritative terms prohibit required intended research/retention, or source confirms necessary content cannot be pinned/recovered for repeatable research and no permitted alternative satisfies the requirement | Human/legal review of actual use and terms, and budget decision; evidence review is not terms acceptance or purchase |
| C — Conservative execution observability | A/B documentary inputs support causal observable state, and a separate recorded human feasibility judgment establishes defensible treatment/bounds for observation-to-arrival latency, executable spread/depth, partial/rejected/unfilled orders, impact, evidenced fees, exit costs and residual inventory, without optimistic fills | L2 alone, unknown observation/arrival mapping, unbounded impact/fills/costs, missing effective fee/rule evidence, or absent human modeling judgment | Verified essential observability incompatibility with no defensible conservative treatment, established by evidence and human review; current candle-simulator limitations alone are not proof of impossibility | Counterfactual trader latency, fills and impact are not provider-observed outcomes; no simulator design or implementation is authorized |

**Gate C cannot PASS merely because Tardis supplies historical L2 or answers every question.** Provider evidence may close input gaps while this gate remains UNRESOLVED. Explicitly documented limitations may be acceptable only where a recorded conservative human judgment addresses them; absence of guarantees is not automatically BLOCKED.

Future consolidated rule, applied without upgrading missing evidence:

- **READY_FOR_HUMAN_PROTOCOL_FEASIBILITY_REVIEW:** all three gates PASS on verified applicable evidence and the separate C feasibility judgment. This permits consideration only, not preregistration, data access, engineering or experiments.
- **HOLD_PENDING_PROVIDER_EVIDENCE:** no verified fatal incompatibility, but any critical gate remains UNRESOLVED, including an outstanding human C judgment. Record provider evidence gaps separately from human-design dependencies; do not ask Tardis to guarantee hypothetical fills to resolve the latter.
- **NOT_FEASIBLE_WITH_TARDIS:** at least one indispensable gate is BLOCKED by a verified incompatibility applicable to the required research, with no evidenced lawful/conservative alternative within scope. Lack of reply, blocked pages or a missing quote cannot trigger this decision.

No rule is applied to hypothetical answers now. TASK-012's current HOLD and all three UNRESOLVED gates remain unchanged.

## 5. Provider-answerable versus research-design questions

| Provider can document | Requires a separate future human decision |
|---|---|
| Paired coverage, exclusions, native IDs/sequences, capture methods, clocks/ordering limits and gap/revision policies (A/B packages) | Whether those limits support causal external-observer availability; refusal of incomplete states and acceptable uncertainty, without choosing on outcomes |
| License applicability, plans, prices, retention and archive/version guarantees (C/D packages) | Actual personal/commercial use classification, legal/budget review and whether to authorize any later acquisition; no account or purchase now |
| Recorded visible book and effective instrument metadata coverage | Trader routing/processing delay, counterfactual impact and fills, partial/unfilled/rejected order and residual-inventory treatment; independent fee evidence and full closing costs |

Provider answers cannot establish net profitability, authorize a strategy or reveal orders never submitted. No question asks for expected returns, profitable periods, parameters or a simulator. Existing simulator semantics are preserved; any later engineering requires separate authorization.

## 6. Response intake and provenance checklist

For a future response, without acquiring market data:

1. Record sender/role or publisher identity, receipt and verification dates, original message/document reference, version/effective date, URL and precise clauses. Preserve an authorized documentary copy and SHA-256 where permitted; keep personal/contact details out of public artifacts unless necessary and authorized.
2. Map each answer to A1–D2, its dossier gap and affected gate. Record supported scope, exclusions, conditional terms, contradictions and unanswered subparts; do not infer assent from silence or substitute an SDK license for data rights.
3. Distinguish contractual commitments, technical specifications, marketing and informal statements. Check applicable exchange restrictions and plan scope. Escalate conflicting or uncertain applicability for human clarification, without accepting terms or credentials.
4. Separate original/corrected history, effective versus verification dates and retained version identifiers. Record what an identical request can actually reproduce. Do not validate by calling APIs, requesting samples, downloading archives or running replay.
5. Apply §4 only in a separately authorized human evidence review, recording reasons and sources per gate; leave unsupported requirements UNRESOLVED and preserve prior reports/Research Memory. New evidence intake does not release reserved data or authorize a research period.

## 7. Next human decision

**Review/edit the draft and eight questions, then decide whether to send them through a human-selected Tardis contact route or leave Candidate A parked.** Add the real sender's signature and attach/include §2 if sending. This task does not authorize the assistant to contact the provider. If a response arrives, the next separate decision is whether to authorize documentary intake/review under §§4–6, without market-data access or experimentation.

No provider contact, email, account, subscription, purchase, terms acceptance, dataset access, period selection, code/simulator change, experiment/hypothesis ID, Research Memory modification, trading or capital deployment is authorized or performed.

## 8. Verification results

Verified 2026-10-24: TASK-012 commit is an ancestor of latest fetched `main`, whose merge message confirms PR #187. All local file/heading references resolve. Eight unique question IDs cover four packages, each with English wording, documented gap, expected response, acceptance criterion and affected gate. The questions trace to TASK-012 §§3–7 and 10; no new provider facts, product entitlement, price or legal applicability are asserted. The draft is unsigned, concise and unsent; data licensing remains distinct from the SDK's MPL-2.0 license. The matrix includes PASS/UNRESOLVED/BLOCKED conditions, provenance/effective-date requirements and human dependencies for each gate.

Only this requested Markdown file is added; all previously tracked artifacts remain unchanged against `origin/main`. `git diff --check` passes. Verification is documentary only: no application tests, market data, SDK replay, backtests, optimizer, external provider investigation or communication. Current research classifications remain unchanged. Stop after delivery.
