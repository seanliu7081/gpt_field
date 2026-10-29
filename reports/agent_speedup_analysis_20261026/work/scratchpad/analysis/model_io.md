# Model-call channel: uc7 (agent-speedup-v2-user-check-7) and v3 (r1-r11)

Read-only analysis of the request/response logs. Nothing under /workspace was modified; no model, HTTP, simulator or supervisor process was started. All paths below are relative to `/workspace/gpt_field` unless absolute. "Estimate" marks a number derived from a fitted model plus a stated assumption; "inference" marks an interpretation that the artifacts do not observe directly.

## 0. Data, definitions, method

| Set | Runs | Calls used | Source |
|---|---|---|---|
| uc7 execution | 13 runs whose `summary.json` has `user_check.version = agent-speedup-v2-user-check-7` (t0_001, t1_001, t2_002, t3_003, t3_005, t4_001, t5_001, t5_002, t6_009, t6_011, t7_001, t8_001, t9_001) | 218 `agent` calls, 218 valid | `runs/agent_speedup_v2_user_check_libero_10_*/requests/*.{request,response,result}.json`, `trace.jsonl` |
| v3 execution | 15 runs r1-r11 (t1_001..t1_012, t7_001, g5_001, g5_002) | 348 valid `agent` calls (+2 failed: t1_003 #43 `unknown_error`, t1_004 #2 `subscription_sharing_user_unavailable`) | `runs/agent_speedup_v3_*/requests`, `trace.jsonl` |
| uc6 contrast | t6_007, t3_002 (`agent-speedup-v2-user-check-6`) | 171 | same layout |
| Preparation (plan) attempts | 26 uc review dirs, 20 v3 review attempts, 4 evidence-campaign trials, 2 v3-baseline attempts | 52 logged attempts + 4 replays | `runs/*_review/requests`, `results/speedup_evidence_20261026T055123Z/evaluations/trial_00*/review/requests`, `runs/v3_completion_baseline_20261026T133633Z/attempts/*/review/requests`, `runs/speedup_next_20261026T101438Z/request_diagnostics.jsonl` |

Extraction used python3 stdlib JSON (numpy only for least squares). Scripts: `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/scripts/{extract.py,attrib.py,analyze.py,runinfo.py}`; derived rows: `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/{uc_rows,v3_rows,review_rows,evidence_rows,baseline_rows,attrib}.jsonl`.

**Latency fields (where per-call latency is recorded).**
1. Primary: `trace.jsonl`, rows with `event == "action"`: `.diagnostics.api_calls[i].latency_s`, `.diagnostics.api_latency_s` (sum for that act) and `.diagnostics.api_request_log` (links to the request file); `.decision_latency_s` is the whole `act()`. The timer wraps `client.request_agent` in `ExecutiveCall` (`src/gpt_field/agent_speedup_v2_user_check_agent.py:258-283`, emitted at `:551-555`); the evaluator writes the row at `scripts/evaluate_agent_speedup_v2.py:1473-1475` and sums `api_latency_s` into `summary.timing_components.model_pipeline_s` at `:1444`.
2. Request logs: `request.json.budget_elapsed_s_before` and `result.json.budget.elapsed_s` use the same client monotonic clock (time since client construction, `src/gpt_field/chatgpt_plan.py:457,520`); their difference ("client interval") is within 0.129 s (uc7 mean) / 0.302 s (v3 mean) of the trace latency.
3. Server: `response.json.created_at` / `completed_at` (integer seconds).
4. v3 only: `result.json.v3_transport_diagnostics.elapsed_s` (`src/gpt_field/agent_speedup_v3_transport.py:259,271`) and `summary.terminal_policy_act_accounting` for a final act that raised (`scripts/evaluate_agent_speedup_v3.py:106-125`).

The uc7 frozen copy matches the current source for `chatgpt_plan.py`, `ordered_transport.py`, `direct_history.py`, `agent_speedup_v2_client.py`, `mark_prompting.py` and `agent_speedup_v2_user_check_agent.py`. `agent_speedup_v2_user_check_agent_client.py` differs only in one import on line 30 (`cmp`/`diff` against `runs/agent_speedup_v2_user_check_libero_10_t1_001/frozen_source`). The current `agent_speedup_v3_client.py` and `agent_speedup_v3_transport.py` are byte-identical to `runs/agent_speedup_v3_t1_012/frozen_runtime/src/gpt_field/` (r11).

## 1. Per-call latency: input size versus reasoning versus output

### 1.1 Distributions and simple correlations

| | uc7 (n=218) | v3 r1-r11 (n=348) | uc6 (n=171) |
|---|---|---|---|
| latency mean / median / p90 / max (s) | 15.47 / 13.65 / 26.00 / 39.77 | 25.50 / 20.64 / 42.41 / 119.56 | 14.36 / 14.19 / 20.59 / 36.04 |
| input tokens mean (min-max) | 57,927 (6,404-153,901) | 174,097 (8,212-495,731) | 238,222 (4,651-492,438) |
| images per request mean (max) | 10.2 (12) | 10.8 (12) | 11.6 (12) |
| wire bytes mean (max) | 3.21 MB (4.69) | 4.22 MB (6.32) | 4.31 MB (5.77) |
| reasoning tokens mean (max) | 139 (766) | 176 (1,552) | 42.5 (440) |
| visible output tokens mean (max) | 210 (342) | 443 (2,594) | 129 (176) |
| r(latency, input tokens) | -0.132 | -0.095 | +0.647 |
| r(latency, wire bytes) | -0.196 | -0.308 | +0.387 |
| r(latency, n images) | -0.261 | -0.389 | +0.072 |
| r(latency, reasoning tokens) | +0.878 | +0.931 | +0.511 |
| r(latency, output tokens) | +0.885 | +0.927 | +0.525 |
| r(latency, visible tokens) | +0.290 | +0.763 | +0.431 |

Spearman rho gives the same ordering (uc7: reasoning +0.868, output +0.860, input -0.123).

### 1.2 Least-squares fits (latency in s; input in thousands of tokens; standard errors in parentheses)

| Model | uc7 | v3 |
|---|---|---|
| L ~ input | R² 0.017; slope -0.024 (0.012) s/ktok | R² 0.009 |
| L ~ reasoning | 10.14 + 0.0383 (0.0014)·reas; R² 0.771 | 14.17 + 0.0642·reas; R² 0.866 |
| L ~ output tokens | 3.15 + 0.0353·out; R² 0.783 | 6.31 + 0.0310·out; R² 0.859 |
| **L ~ input + reasoning + visible** | **3.98 (0.98) + 0.0274 (0.0057)·in_k + 0.0392 (0.0014)·reas + 0.0212 (0.0042)·vis; R² 0.813** | **5.31 (0.54) + 0.0184 (0.0017)·in_k + 0.0522 (0.0012)·reas + 0.0175 (0.0009)·vis; R² 0.940** |
| server time (created→completed) ~ same | 3.13 + 0.0205·in_k + 0.0393·reas + 0.0181·vis; R² 0.852 | 2.98 + 0.0177·in_k + 0.0528·reas + 0.0174·vis; R² 0.950 |
| uc6: L ~ input + reasoning + visible | 0.68 + 0.0265·in_k + 0.0381·reas + 0.0445·vis; R² 0.792 | |

Adding image count and text characters instead of input tokens (uc7: images -0.16 (0.08) s/image, text 0.0104 s per 1k chars) does not change the conclusion.

**Decomposition of the fitted latency** (sums over all calls; the fitted sums equal the measured sums, 3,371.5 vs 3,371.6 s and 8,872.4 vs 8,872.5 s):

| Component | uc7 | v3 |
|---|---|---|
| fixed per call (intercept) | 868 s (25.7%) | 1,848 s (20.8%) |
| input / prefill | 345 s (10.2%) | 1,117 s (12.6%) |
| reasoning tokens | 1,189 s (35.3%) | 3,206 s (36.1%) |
| visible output tokens | 969 s (28.7%) | 2,702 s (30.5%) |

**Answer to (1).** At the history lengths of uc7 and v3, per-call latency is driven by generated tokens: reasoning plus visible output account for 64% (uc7) and 67% (v3) of fitted latency, and a fixed 4-5 s per call for another 21-26%. Input size has a real but second-order partial effect of 0.018-0.027 s per 1k input tokens, about 1.6 s per mean uc7 call and 3.2 s per mean v3 call. The raw correlation with input is slightly negative because later, longer-context calls reason less (r(reasoning, input) = -0.31 uc7, -0.21 v3). Input becomes dominant only in long runs: in uc6, 0.0265 s/ktok × 238k mean tokens = 6.3 s, 44% of its 14.4 s mean call.

### 1.3 Other per-call facts
- **The input effect is server-side prefill, not upload.** The client-side residual (latency − server time) averages 1.89 s (median 1.55) for uc7 and 2.39 s (median 2.14) for v3. It is only weakly related to upload size: +0.23 (0.08) s/MB for uc7 and +0.12 (0.05) s/MB for v3, with R² ≤ 0.034. In the 4 replayed prep requests, authentication took 1.444 s on the first request and about 0.001 s afterwards, and `request_sent → response_headers` took 1.36-1.83 s for 1.48 MB bodies (`runs/speedup_next_20261026T101438Z/request_diagnostics.jsonl`).
- **The first execution call is the slowest.** uc7 first calls average 27.4 s (438 reasoning tokens) against 14.7 s for the rest. v3 first calls average 64.5 s (676 reasoning, 1,396 visible tokens) against 23.7 s. v3 t1_008 call 1 took 119.6 s.
- **The latency tail is reasoning.** Calls with more than 300 reasoning tokens are 31/218 uc7 calls but take 24.8% of uc7 model time, and 58/348 v3 calls taking 34.6% of v3 model time.
- **Service speed drifted between days.** Today's runs were slower than the historical fits predict for the same tokens. Evidence trial_002/003 (uc7 code, t1 ep1, 2026-10-26) measured 251.0 / 277.9 s of execution-call time against 166.6 / 186.8 s predicted (×1.51 / ×1.49). The v3-baseline attempt_003 measured 575.5 s against 425.1 s predicted (×1.35). The excess is server-side (server sums 235 / 259 / 545 s). The cause is unknown: service load, or the protocol's overlapped A/B execution (`final_report_zh.md` §4).

## 2. Non-completing preparation streams and structured output

### 2.1 How the request asks for structured output (code)
- Execution (uc7): `text.format = {type: "json_schema", name: "user_check_agent_first", strict: True, schema: agent_schema()}`, `reasoning.effort = "high"`, `include = ["reasoning.encrypted_content"]`, `stream = True`, `store = False` (`src/gpt_field/agent_speedup_v2_user_check_agent_client.py:355-372`). The schema requires all 7 keys in the order belief, plan, why, type, motion, macro, execution, with `additionalProperties: false` and `maxLength` 1800/1800/600 on belief/plan/why (`:195-215`). The prompt ends "Keep task reasoning concise and output the JSON only." (`:191`).
- Plan (prep): same strict `json_schema` with name `agent_speedup_v2_<kind>` and one stateless user message (`src/gpt_field/agent_speedup_v2_client.py:811-819`). In the logs: 4 images at detail "high", 7,832-char instructions, about 3,911 input tokens.
- v3: inherits both and renames the format to `agent_speedup_v3` with the v3 schema (`src/gpt_field/agent_speedup_v3_client.py:746`). Since prompt 5.1, v3 appends `COMPACT_OUTPUT_INSTRUCTIONS`: "Emit exactly one compact JSON object ... Do not indent or align fields, insert blank lines, or pad between tokens ... do not append whitespace". This is added to the plan instructions (`:40-46, :130`) and to the execution instructions (`:399-400`).
- Key order: insertion order is preserved on the wire by `ordered_dumps` (compact separators, no `sort_keys`) (`src/gpt_field/ordered_transport.py:1-8, 22-25`). `sort_keys=True` survives only in `astra._canonical` (`src/gpt_field/astra.py:117-118`), which now hashes request keys.
- **`max_output_tokens` is never sent.** It is in `UNSUPPORTED_FIELDS` together with temperature, top_p, previous_response_id, conversation and prompt_cache_retention (`src/gpt_field/chatgpt_plan.py:37-41`). It is stripped in `build_payload` (`:429-436`), as the docstring says (`:405-408`). The configured `max_output_tokens = 8192` appears only in logs; every result shows `output_token_limit_sent: null` and every response `max_output_tokens: null`. The visible-token "budgets" are "diagnosis only ... never a cap" (`src/gpt_field/agent_speedup_v2_client.py:116`, `LABELS["VISIBLE_TOKEN_BUDGETS"]`). Responses show the server defaults temperature 1.0 and top_p 0.98.
- **The uc7 transport has no total deadline and no output bound.** `ordered_sse_transport` calls `build_opener(...).open(request, timeout=timeout_s)` with `timeout_s = 180` (`src/gpt_field/chatgpt_plan.py:141`). That is urllib's per-socket-operation timeout, so a stream that keeps sending deltas never expires. The evidence report says the same: "底层180秒读取超时不是整个 SSE 流的总期限" (`final_report_zh.md` §4). The parser accepts only `response.completed` (`chatgpt_plan.py:315-342`).
- **v3 bounded transport (r4.1 onward).** It adds a 180-s total deadline via SIGALRM, `MAX_VISIBLE_CHARACTERS = 131072`, and `MAX_JSON_WHITESPACE_RUN = 2048` counted outside JSON strings. Guard failures are not retried (`src/gpt_field/agent_speedup_v3_transport.py:16-19, 92-124, 224-271`; installed by default at `agent_speedup_v3_client.py:79-82`).

### 2.2 Inventory of preparation tails

| Attempt | Prompt | Outcome | Duration |
|---|---|---|---|
| runs/agent_speedup_v2_user_check_libero_10_t3_004_review | v2-user-check-prompt-5 | no response, `KeyboardInterrupt` | 298.1 s |
| ..._t6_008_review | same | `KeyboardInterrupt` | 634.0 s |
| ..._t6_010_review | same | `KeyboardInterrupt` | 297.2 s |
| runs/agent_speedup_v3_t1_review_001 #1 | same | `stream_missing_completion` | 900.7 s (retry #2 ok, 14.6 s) |
| runs/agent_speedup_v3_t1_review_005 #1 | same | `stream_missing_completion` | 900.5 s |
| runs/agent_speedup_v3_t1_review_005 #2 | same | **completed**, 76,222 output tokens | 771.7 s |
| runs/agent_speedup_v3_t1_review_006 | same, bounded transport | `subscription_output_whitespace_limit` (2,944 visible chars, run 2,049) | 24.9 s |
| evidence trial_004 / trial_005 | same | `EvaluationBudgetExceeded` (900-s campaign cutoff), no response | 893.4 / 891.9 s client interval (901.0 / 901.0 s service) |
| speedup_next prep_001 / prep_003 (replays of trial_002 input) | same | 120-s cutoff, no `response.completed` | 120.0 / 120.0 s |

The three interrupted uc reviews are from 2026-10-21 (file mtimes) and use the same plan prompt as uc7. Their review dirs do not record the uc version label.

### 2.3 What the non-completing streams contained (`request_diagnostics.jsonl`)
The probe logged a cumulative `public_output_text_chars` for every `response.output_text.delta` event. It did not keep the delta text: "No raw stream content or hidden reasoning collected" (`request_results.json` `causal_limits`). From the per-delta increments:

| | prep_002 (completed) | prep_004 (completed) | prep_001 (cut at 120 s) | prep_003 (cut at 120 s) |
|---|---|---|---|---|
| deltas / chars | 294 / 922 | 293 / 924 | 6,493 / 141,110 | 7,115 / 27,853 |
| max increment | 12 | 12 | 37 | 12 |
| normal phase | whole stream | whole stream | 263 deltas, 830 chars, max increment 12, until 19.16 s | 276 deltas, 873 chars, max increment 12, until 10.88 s |
| after switch | n/a | n/a | 6,230 deltas, 140,280 chars; 6,210 of all deltas are 16-37 chars (sum 140,224 = 99.4% of chars); 62.0 deltas/s, 1,397 chars/s | 6,839 deltas, 26,980 chars; 6,613 deltas exactly 4 chars; increments only 1-4; 62.8 deltas/s, 248 chars/s |
| events | `response.completed` at 17.24 / 11.14 s | same | last event `output_text.delta` at 119.59 s; 0 heartbeats | last delta at 119.85 s |

Both censored streams produced normal-looking output first. They then switched abruptly to a constant-rate, uniform-increment regime that ran for about 100 s. A full plan for this input is 920-947 chars (prep_002/004 saved `plan.json` 920/922 chars, pretty-printed; trial_002/003 outputs 947/925 chars), so the switch came at about 90% of a complete plan. The text content is unknown (whitespace and a repeated non-whitespace token cannot be told apart). Distinct-content growth cannot be computed because nothing was retained.

### 2.4 The one completed tail is whitespace
`runs/agent_speedup_v3_t1_005_launch/transport_diagnosis.json` covers prep attempt #2 of `runs/agent_speedup_v3_t1_review_005` (771.7 s; provider timestamps 771 s). Its output_text is 152,567 chars, of which 151,982 are whitespace and 151,950 are outside JSON strings. The compact JSON is 617 chars. One run of 151,642 chars starts at character index 886 at container depth 1 and consists of 75,818 newline+space pairs. Usage: 76,222 output tokens, of which 96 are reasoning. The run starts at 886 chars, close to the 830/873-char switch points in 2.3. The first attempt's 900.5-s stream content was not saved.

### 2.5 Natural experiment: compact-output instruction
Classifying every logged plan request by whether its payload contains "OUTPUT SERIALIZATION":

| Class | Attempts | Tails (≥120 s or failed) | Mean / median duration | Max completed (<120 s) |
|---|---|---|---|---|
| v2-user-check-prompt-5, no compact instruction (34 logged + 4 replays) | 38 | **11** (10 ≥120 s, plus the 24.9-s guard trip) | 167.4 s (censored) / 20.7 s | 29.4 s |
| v3 prompt ≥5.1 with compact instruction | 12 | **0** | 16.5 s / 15.2 s | 26.0 s |
| v2 prompt-4 (medium effort, uc1-uc3) | 6 | 0 | 10.3 s / 9.4 s | 19.0 s |

- One-sided Fisher exact test of 11/38 against 0/12 gives p ≈ 0.032; for 10/38 against 0/12, p ≈ 0.046. Confounders: the compact class also used the v3 plan client and bounded transport, and it is mostly task t1.
- Among the 26 logged completed non-compact outputs, 15 were pretty-printed (starting `"{\n  \"scene\": ..."`, 316-651 whitespace chars outside strings). All 12 compact-class outputs have 0 outside-string whitespace.
- The same input can complete or degenerate. trial_002's input completed in 19.8 s, then degenerated on 2 of 2 replays. trial_004's input hit the 900-s cutoff, then completed on 2 of 2 replays (17.3 s, 11.2 s) (`request_results.json`). The tail is a sampling event, not a fixed property of the input (inference).
- Execution calls never showed it. 0/218 uc7 execution outputs are pretty-printed and their maximum outside-string whitespace run is 0. In v3, `longest_json_whitespace_run` is ≤1 in all 160 calls where the bounded transport recorded it.
- Today's two uc7-based campaigns used the non-compact plan prompt: the trial_004 request contains 0 occurrences of "OUTPUT SERIALIZATION", and the replays reused those saved payloads.

### 2.6 Verdict on the hypothesis
"Structured-output whitespace degeneration causes the long prep tails" is **strongly supported, but not proven for every tail**. Supporting evidence:
1. The one completed tail is 99.6% syntax whitespace.
2. The v3 whitespace guard tripped at 24.9 s on the same prompt.
3. Both censored replays show an abrupt switch, near the same structural position, into a uniform-increment stream that never terminated.
4. Adding a compact-serialization instruction moved the tail rate from 11/38 to 0/12.

The content of 7 tails (3 operator interrupts, 2 × 900-s `stream_missing_completion`, 2 × evidence 900-s) and 2 replays was never stored, so whitespace versus repeated content is unknown for them. The 900-s ends point to a server-side stream limit (inference). They are long because the uc7 path sends no `max_output_tokens`, has only a per-socket timeout, and has no output guard.

## 3. History protocol

### 3.1 Code path
- Every call resends the whole transcript. There is no server state: `store=False`, and `previous_response_id`/`conversation` are unsupported (`chatgpt_plan.py:37-41`). After success, `self._history = payload["input"] + response["output"]` (`chatgpt_plan.py:544`) is copied into `_agent_history` (`agent_speedup_v2_user_check_agent_client.py:442`). The next payload is `_trim_images(_agent_history + [new user message], IMAGE_HISTORY=6)` (`:43, :364`). So the history keeps all earlier observation texts, every accepted assistant output, and every opaque encrypted reasoning item (`history_stats.reasoning_replay = "opaque_encrypted_output_items"`).
- `_trim_images` replaces images older than the 6 newest pairs with the text "[Older camera image omitted by fixed image-retention rule.]" and keeps all text (`src/gpt_field/direct_history.py:45-57`). **The 6 newest pairs (up to 12 images) are re-uploaded in every request.** Each is a lossless PNG (`compress_level=1`, `src/gpt_field/mark_prompting.py:84`) at 512×512 (PNG header), detail "high" (`agent_speedup_v2_client.py:800`). Each image costs 308.5 input tokens; the placeholder costs 12 (response `usage.attribution`).
- Each new observation's text is bounded by compacting large evidence above 45,000 chars and rejecting it above 56,000 chars (`agent_speedup_v2_user_check_agent_client.py:484-495`). In practice it is about 18.2-19.2k chars (≈4.9k tokens) per uc7 call.
- v3 r10/r11 adds a wire-only projection of older messages (`project_agent_history`, `agent_speedup_v3_client.py:587-640, :749`). In r11 t1_012 the projection cut wire input text to 38.3% of the unprojected text by call 9 (100,009 vs 260,802 chars; `request.json.prompt_sizes`).

### 3.2 Growth within runs

| Run | Calls | Wire MB: call 1 → call 6 → last (max) | Input tokens: call 1 → last | Total input tokens | Cached % |
|---|---|---|---|---|---|
| uc7 t1_001 | 13 | 0.73 → 4.12 → 4.08 (4.17) | 6,544 → 75,354 | 541,005 | 2.98 |
| uc7 t0_001 | 14 | 0.74 → 4.27 → 4.57 | 6,606 → 80,900 | 625,817 | 4.05 |
| uc7 t6_009 | 28 | 0.71 → 3.96 → 4.55 | 6,836 → 153,901 | 2,302,039 | 1.20 |
| uc7 t5_001 | 7 | 0.57 → 3.42 → 3.41 | 6,477 → 42,853 | 173,064 | 3.99 |
| uc6 t6_007 | 84 | 0.70 → 4.14 → 5.77 | 4,696 → 492,438 | 20,642,919 | 0.02 |
| v3 t1_001 (r1) | 79 | 0.74 → 4.45 → 6.32 | 8,212 → 495,731 | 20,872,319 | 0.75 |
| v3 t1_008 (r6) | 40 | 0.76 → 4.20 → 5.86 | 11,517 → 397,117 | 8,523,018 | 1.69 |
| v3 t1_012 (r11) | 9 | 0.75 → 3.84 → 3.82 | 8,318 → 43,836 | 249,720 | 3.43 |

Across all 13 uc7 runs: call-1 wire size is 0.50-0.74 MB, call 6 is 3.04-4.27 MB, and from there size stays flat at ≤4.57 MB because the 12-image cap is reached. Token count keeps growing linearly with the number of observation texts, so total input per run grows roughly quadratically in the call count: about 5k × N²/2. uc6 t6_007 predicts 84²/2 × 5.6k ≈ 19.8M against 20.6M measured (estimate).

### 3.3 Composition of input tokens (response `usage.attribution`)

| Share of input tokens | uc7 (12.63M, 218 calls) | v3 (60.59M, 348 calls) | uc6 (40.74M, 171 calls) |
|---|---|---|---|
| re-sent earlier observation text | 75.7% | 84.2% | 91.8% |
| latest observation text | 8.5% (4,923 tok/call) | 4.0% (6,917 tok/call) | 2.2% |
| images (308.5 tok each) | 5.4% (3,150 tok/call) | 1.9% | 1.5% |
| instructions + schema | 4.2% | 2.8% | 0.6% |
| re-sent assistant outputs | 3.2% | 4.9% | 2.2% |
| re-sent reasoning items | 2.5% | 1.8% | 0.9% |

Images dominate bytes (3.2 MB of base64 per 6-pair request, about 78% of the wire) but not tokens.

### 3.4 Prompt caching
- Cached input share is 2.95% for uc7 (372,480 / 12,628,107), 1.13% for v3 (683,008 / 60,585,849) and 0.04% for uc6.
- In uc7 the cached value is exactly 2,304 tokens in 137/218 calls (the instructions+schema prefix: 1,942 + 362 attributed) and 0 in 77.
- `cache_write_tokens = 0` in 897/897 responses.
- Every response carries a different, server-assigned `prompt_cache_key` (e.g., 13 distinct keys in 13 calls of t1_001). The client sends none: it is not in `UNSUPPORTED_FIELDS`, but `prompt_cache_retention` is.
- All 7 hits above 10k tokens happened at call ordinal ≤6, i.e., before `_trim_images` starts rewriting message n−6 on every call. v3 r10/r11 also rewrites the previous latest message through the wire projection. The history is therefore not append-only, which structurally limits prefix reuse.

## 4. The loop is strictly serial (robot idle during inference)
- Evaluator: `state("deciding")` → `decision = policy.act(obs)` → video `recorder.decision` → `obs, success = sim.step(decision.action)` → `policy.observe_transition` → video `recorder.executed`, one after another in a single thread (`scripts/evaluate_agent_speedup_v2.py:1438-1512`; `act` at 1440, `sim.step` at 1483, `observe_transition` at 1506). The uc7 and v3 runners reuse this evaluator.
- Policy: `act()` calls `_ask_agent` → `client.request_agent` synchronously inside the act (`agent_speedup_v2_user_check_agent.py:484-520, 258-283`). v3 wraps the same `_ask_agent`/`act` (`scripts/evaluate_agent_speedup_v3.py:98-125`).
- The model is told so: "Simulation pauses during inference." (`agent_speedup_v2_user_check_agent_client.py:150`).
- The v3 bounded transport *requires* the POSIX main thread and an idle ITIMER_REAL (`agent_speedup_v3_transport.py:55-64`), so v3 inference cannot run in a background thread as built.
- No thread, future or asyncio use exists in the policy, client or transport modules; the only threads are the review UI server and pause events.
- The robot moves only in the gaps between calls. uc7: 2,829.6 s of gaps against 3,371.6 s of model time in 6,202.3 s of control (13 runs; model = 54.4% of control). v3: 2,671.7 s of gaps against ≈8,858-8,887 s of model time in 11,481.9 s (model ≈77%). Examples: uc7 t1_001 182.5 / 407.2 s = 44.8%; v3 t1_008 1,336.5 / 1,593.6 s = 83.9%.

## 5. Levers: how much model time each could remove (estimates)
Basis: the multivariate fits in §1.2 applied call by call to the observed calls (uc7: 218 calls, 3,371.6 s of execution model time; v3: 348 calls, 8,872.5 s). Percentages are of execution model time unless stated. They assume the fitted coefficients are causal and that decision content and call count stay fixed except where stated.

| Lever | Assumption | uc7 | v3 | Example uc7 t1_001 / v3 t1_008 |
|---|---|---|---|---|
| Compact output + bounded transport for **prep** | v3-style compact instruction, 180-s total deadline, 2,048 whitespace guard | prep attempt mean 167.4 s → 16.5 s (observed class means; non-compact mean censored low) | already in v3 (0/12 tails) | removes the 300-900 s tails |
| Shorter **visible** execution output | −50% visible tokens (belief/plan/why prose) | 484 s (14.4%) | 1,351 s (15.2%) | 35.6 s / 177.8 s |
| Non-strict schema | no non-strict calls exist to compare | unknown | unknown | – |
| Lower **reasoning** effort (extra; not in the brief) | −50% reasoning tokens; quality impact unknown | 595 s (17.6%) | 1,603 s (18.1%) | 27.8 s / 230.4 s |
| Shorter **text history** | −80% of re-sent history text tokens | 209 s (6.2%); −100%: 261 s (7.8%) | 752 s (8.5%); −100%: 940 s (10.6%) | 8.2 s / 105.0 s |
| Fewer retained **image pairs** | 1 pair instead of 6, token effect | 15 s (0.45%) | 17 s (0.20%) | – |
| … plus upload bytes | 0.23 / 0.12 s per MB × ~2.5 / ~3.5 MB per call (R² ≤ 0.034, low confidence) | ≈126 s (≈3.7%) | ≈150 s (≈1.7%) | – |
| Lower **image resolution** | upper bound: zero image tokens, plus the bytes effect above | ≤19 s (0.56%) + bytes | ≤21 s (0.24%) + bytes | – |
| **Fewer calls** (bigger committed fragments) | halve N. Merge bound: save N/2 intercepts + 75% of input tokens (quadratic growth). Drop bound: half the calls were unnecessary | 693 s (20.6%) to 1,686 s (50%) | 1,761 s (19.9%) to 4,436 s (50%) | 37 to 91 s / 224 to 668 s |
| **Asynchronous / pipelined** inference | upper bound: next call fully overlapped with the preceding execution gap, Σ min(gap_k, L_{k+1}), with perfect speculation | 2,151 s = 34.7% of control time (63.8% of model time) | 2,233 s = 19.4% of control (25.1% of model time) | 149.5 s (407.2 → 257.7 s control) / 282.0 s |
| **Prompt caching** | 85% of input tokens served from cache with zero prefill time; needs an append-only history and an honoured stable cache key (both unverified) | 294 s (8.7%); upper bound 345 s (10.2%) | 949 s (10.7%); upper bound 1,117 s (12.6%) | 12.6 s / 133.5 s |

Notes on the estimates:
- The levers overlap and do not add. Caching and history shortening both act on the same 10-13% input component.
- Images are a minor latency lever by tokens. Bytes may matter, but the fit is weak.
- Pipelining is bounded by how much execution time exists. v3 is model-bound (gaps are 23% of control), so overlap helps it little.
- The speculation hit rate is unknown.
- The 300-s S300 budget: at historical speed, uc7 t1_001 spent 13.7 s on prep and 182.5 s on execution calls (196 s, 65% of the budget). At today's service speed the comparable trials spent 252.3 and 279.5 s on execution calls alone (`evaluations/trial_00{2,3}/trial/summary.json`, `model_pipeline_s`).

## 6. Unknowns and caveats
- Content of the two censored replay streams and the seven unsaved tails (whitespace or repeated tokens), and why some streams end at about 900 s without `response.completed` (server limit: inference).
- Whether the subscription route would accept `max_output_tokens`, temperature or a client `prompt_cache_key`; whether cached tokens shorten latency on this route (no cache writes are ever reported).
- The OLS coefficients are observational. History length, decision type and reasoning co-vary within runs, and service speed varied by 35-51% across days (§1.3), so lever estimates may be off by that much.
- The quality and success effects of less reasoning, shorter prose, fewer or smaller images, or bigger fragments were not measured.
- No per-phase upload/TTFB timing exists for execution calls; it exists only for the 4 speedup_next prep replays.
- The cause of today's slowdown (service load or overlapped A/B concurrency) is unknown.
