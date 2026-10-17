# Optional historical research copilot

The default remains the deterministic local CLI from PR #179. The optional adapter in `app.research.copilot` uses `app.research.query_report` to read only the committed `research/reports/failure_patterns_exp_000021_000024.md`. It never follows links to other artifacts, regenerates the report, loads datasets, reruns experiments, allocates IDs, or writes Research Memory. No new dependency is required; the existing `requests` dependency is imported only on explicit enablement.

## Model availability

As of 2026-10-13, no OpenAI API model ID could be verified in this environment. The proxy returned CONNECT HTTP 403 for `api.openai.com/v1/models`, the official model documentation at `developers.openai.com/api/docs/models`, and `openai.com/api/pricing/`. No OpenAI API credential was present. The Codex assistant model name does not establish API availability, so **there is no assumed `gpt-6` identifier or automatic substitute**.

Configure an exact model ID available to your OpenAI project. Before each enabled request, the adapter calls `GET https://api.openai.com/v1/models/{model_id}` and requires the returned ID to match. It then uses the Responses API with strict JSON schema. Model listing availability alone does not prove Responses/structured-output compatibility; unsupported models, unavailable IDs and API failures return `provider_error`. Real integration remains unvalidated until a compatible model and access are supplied. Mocked tests do not establish real model availability or quality.

## Configuration and usage

Set `RESEARCH_COPILOT_API_KEY` securely through environment settings, not in tracked files, command-line arguments, reports, or chat. Set `RESEARCH_COPILOT_MODEL` to the verified API model ID. Network access to `api.openai.com` is required only for enabled use. Requests use the fixed HTTPS endpoint with TLS verification, a connection/read timeout, no redirects, no automatic retries and `store: false`.

From the repository root, this makes **no API call**, even when a key/model is configured:

```bash
venv/bin/python -m app.research.copilot "¿Por qué fue rechazado EXP-000024?"
```

Explicit opt-in, after configuring the key and verified model:

```bash
venv/bin/python -m app.research.copilot "¿Cuál es el veredicto de EXP-000021?" --experiment EXP-000021 --enable
venv/bin/python -m app.research.copilot "¿Cómo se comparan las causas de rechazo de los cuatro experimentos?" --enable
```

Alternatively, `RESEARCH_COPILOT_ENABLED=true` explicitly enables the adapter for subsequent invocations; unset it to return to the disabled default. `--model` can override the configured identifier. There is no fallback model. Questions are limited to 1000 characters; evidence JSON to 24000 characters; generated selection to 256 output tokens and 12 selected lines. Oversized evidence fails closed instead of silently truncating context.

Output is JSON. `answered` contains exact report quotations and citations (`path`, one-based `line`, `sha256`, `quote`), including table headers. `insufficient_evidence` contains no answer or citations; it covers unsupported questions, missing sources, unavailable experiment scope, empty/oversized evidence and invalid selections. `disabled` performs neither retrieval nor API calls. `configuration_error` and `provider_error` are operational failures and exit with code 2; they are distinct from missing research evidence. Provider bodies and credential-bearing exceptions are never printed.

## Grounding, injection resistance and limitations

The model is an evidence selector, not an autonomous agent. It receives a JSON envelope with the question, evidence and provenance. It has no tools, filesystem access, credentials in its prompt, optimizer interface, trading execution or memory writer. Question and artifact contents are marked as untrusted data in the system instructions. Recognizable instruction-override/secret-exfiltration patterns are rejected or excluded. Returned selections are validated against retrieved lines; unknown citations, extra prose/keys, malformed or incomplete output cannot become an answer. Citations come from retrieval, never from model-authored paths or hashes.

This deliberately extractive interface supports natural-language selection, not generated narrative or calculations. It preserves missing metrics and the distinction between triage and journal governance verdicts. It cannot validate original market runs, authorize live capital, forecast prices or rescue rejected hypotheses. `store: false` does not mean the provider receives no data: explicit opt-in sends the question and retrieved report excerpts to OpenAI, subject to the user's account/data policies.

Injection filtering is not a universal semantic detector. The main controls are absence of tools and returned free prose, fixed destinations, and validated exact quotations. Relevance and completeness of selected lines remain model judgments and cannot be guaranteed by structural validation. Unsupported questions instruct the model to abstain, but human review is still required to determine whether a quotation answers a nuanced question. There is no live-model evaluation yet. Provider selection is not deterministic; the underlying local retrieval and exact citation construction remain deterministic.

## Estimated cost

No live model calls were made and no actual model tariff could be verified. The full-report evidence currently serializes to approximately 10540 characters. A rough planning estimate for a full query is **3000–6000 input tokens plus at most 256 output tokens**; experiment scoping usually sends less. Character estimates are not tokenizer measurements. Reasoning-token accounting and prices depend on the configured model; the output cap can produce an incomplete response, which is rejected.

If the verified model costs `I` USD per million input tokens and `O` USD per million output tokens, estimated generation cost per full query is:

```text
(0.003 × I) + (0.000256 × O)  to  (0.006 × I) + (0.000256 × O) USD
```

This is a planning formula, not a confirmed GPT-6 price or a measured bill. Use the selected model's official current pricing and account usage to establish actual costs. No retry loop or hidden model fallback adds requests.

## Validation

```bash
venv/bin/python -m pytest -q -p no:cacheprovider app/research/test_copilot.py app/research/test_query_report.py app/research/test_analyze_existing.py
```

Tests use mocked responses and synthetic credentials only. They cover EXP-000021–024, unsupported questions, missing evidence, injection, disabled behavior, model verification before generation, exact quotations/provenance, transport controls and read-only execution boundaries. No API credentials or network are required. Historical artifacts and deterministic query implementation remain unchanged.
