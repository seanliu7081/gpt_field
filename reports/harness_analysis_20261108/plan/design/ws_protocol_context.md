# ws_protocol_context — executive protocol, model calls and context

Design for one workstream of the v3 implementation plan (build on H1's loop; perception as an in-controller
service; H2 robot geometry for path checks; metric estimation taken away from the model).

- Date: 2026-11-08. Read-only design: nothing under `/workspace/gpt_field` was edited; no simulator, model call
  or GPU job was run. Offline measurements below come from small scripts over saved run artifacts
  (`/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_protocol_context_scripts/`).
- `H1/` = `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/` (git `3cc84ac`, clean). `NEW/` = the new
  build root that ws_runner_measurement creates as a copy of H1 (H1 itself is never modified).
- Short file names: `agent_client.py` = `H1/src/gpt_field/agent_speedup_v2_user_check_agent_client.py`,
  `agent.py` = `..._user_check_agent.py`, `fragments.py` = `..._user_check_fragments.py`,
  `v2_client.py` = `H1/src/gpt_field/agent_speedup_v2_client.py`, `uc_client.py` = `..._user_check_client.py`,
  `uc.py` = `H1/src/gpt_field/agent_speedup_v2_user_check.py`, `uc_transport.py` = `..._user_check_transport.py`,
  `runner.py` = `H1/scripts/evaluate_agent_speedup_v2_user_check.py`, `base_loop.py` = `H1/scripts/evaluate_agent_speedup_v2.py`.

---

## 0. Summary

Seven components (PC1–PC7). In landing order: transport + A/B harness first, then honest termination, then the
relaxed plan gate, then the compact context. The checkpoint protocol comes after ws_executor_control's guarded
motion and grasp predicates, and the asynchronous verification last.

| ID | Component | Fixes | Effort (d) |
|---|---|---|---|
| PC1 | Honest termination: terminal `finish` / `give_up`, both oracle cues deleted, completion contract, repeat counts | A1 (prompt side), R6, H1 #17, F1/F9 | 1.5 |
| PC2 | Model-call resilience and telemetry: forced token refresh on 401, transient 5xx/520/deadline retry, thread-safe deadline, per-call tok/s log, stable `prompt_cache_key` option | A3, H1 #16, G2, D3 (cache key), prerequisite for PC6 | 2.0 |
| PC3 | Plan gate: macro-only rules removed, one re-ask via `reask_error`, sanitise-not-abort, **scene inventory** (object nouns + cells) for perception; fold option = user decision | A4, H1 #14, IC7, V1/V2 inputs | 2.5 |
| PC4 | Context v3: static cacheable prefix, append-only transcript with ephemeral blocks stripped at insertion, macro layer removed, one outcome record per decision with verbatim reasons, `why` dropped, quaternion hemisphere canonicalisation, perception-facts slot | D3, H1 #6, F7, F12, IC9, IC15, V6 (H3 F4) | 4.0 |
| PC5 | Checkpoint protocol: `then` / `stop_on` / `grasp` segment fields, resumable remainder, blocking checks only for placement and engagement | D1 (protocol side), H1 #2, IC5 | 3.0 |
| PC6 | Non-blocking verification: call scheduler with virtual-time arrival, real-time pacing, staleness rules, barriers; removes "Simulation pauses during inference" | R4, R9, D1 (async), IC14 | 5.0 |
| PC7 | Prompt A/B harness: model-only replay of saved payloads, interleaved arms, capped | IC9 ("A/B reasoning tokens"), G2, D3 validation | 2.0 |

Three measured corrections to the brief, found while grounding the design:
1. **`w >= 0` does not remove the quaternion flips.** At the tool-down pose `|w| < 0.011`, so w ≥ 0 leaves all 33 flips
   in 5 runs (3/13 T04, 3/21 T08, 5/16 stove-off, 10/42 and 12/36 drawer). Choosing the hemisphere of the
   episode's first measured orientation (`sign(q·q_ref)`) leaves 0 flips in all 5 runs (min `|q·q_ref|` = 0.49).
2. **Prose caps do not bind.** Over 128 responses, belief max 285 chars (cap 1800), plan max 202 (cap 1800),
   why max 154 (cap 600). Capping saves nothing. Dropping `why` saves about 28 visible tokens per call.
3. **The per-message observation text is the quadratic cost, not images.** At T04's last call, 75.6k of 87.4k
   input tokens are replayed observation text (13 × ~5.8k). Images are 3.9k, instructions 2.8k, schema 0.5k.
   A prototype v3 projection of all 128 saved requests cuts the per-message text from 20.8k to 3.8k
   characters (18 %), with 0 lost reason/detail/status strings.

---

## 1. Code checked and offline measurements

### 1.1 Code read (H1, line numbers verified)

| Location | What it does | Used by |
|---|---|---|
| `agent_client.py:41-50` | prompt-6 constants; `IMAGE_HISTORY = 6`; `CONTEXT_KEYS` include `macro`, `native_evidence`, `task_success_source` | PC4 |
| `agent_client.py:72-83` | "YOU OWN EXECUTION" paragraph: macro/observer leftovers; "Choose short informative moves… shorter monitored steps" | PC4, PC5 |
| `agent_client.py:85-105` | response fields and model-written `obstacle_map` boxes | PC4 |
| `agent_client.py:119-155` | segment fields, profile envelopes (0.50 m / 1.2 rad), caps text `sum(max_steps)<=240` (133-134), routing text | PC5 (rendered from capabilities) |
| `agent_client.py:157-167` | `observation_after` semantics; **grasp mandate 161-163** ("set_gripper=close followed by an appropriately small monitored test lift/checkpoint, then inspect new images") and **placement mandate 163-164** | PC5 |
| `agent_client.py:182-183` | "Simulation pauses during inference." (R9 privilege) | PC6 |
| `agent_client.py:186-200` | macro paragraph (never used: 0 grants in 7/7 runs) | PC4 |
| `agent_client.py:205-209` | finish paragraph: "never ends the episode", **"Only the independent evaluator terminates a successful task"**, **"If observation continues, recheck…"**, "There is no give_up action." | PC1 |
| `agent_client.py:221-233` | camera conventions; "Pixels specify rays… infer depth" (224-225); gripper pad facts, no palm | PC4 |
| `agent_client.py:237-260` | schema: belief 1800, plan 1800, why 600; `type` enum incl. `macro`; `obstacle_map` | PC1, PC4 |
| `agent_client.py:310-343`, `545-556` | `_execution_evidence_view` truncation at >45k chars (121/128 requests compacted); 56k hard error | PC4 (IC15) |
| `agent_client.py:408-425` | `build_payload`: fixed instructions; `_trim_images(self._agent_history + [current], 6)` at **417**; no `prompt_cache_key` | PC2, PC4 |
| `agent_client.py:446-502` | `predict` holds `_history_lock` for the whole call; commit at 495 | PC6 |
| `agent_client.py:536-542` | `robot_pose_wxyz: measured_pose(view.state)`; reviewed contract on the first message only | PC4 |
| `direct_history.py:45-57` | `_trim_images`: replaces images of all but the newest 6 image messages, every call → rewrites message n−6 | PC4 |
| `agent.py:118-131` | `_task_contract`: **completion_rule "Only the independent task evaluator can terminate with success"** (128-130, in the first message) | PC1 |
| `agent.py:133-155` | `_capabilities` (4.6k chars per message) incl. `native_macros`, finish "does not stop the evaluator" (151) | PC1, PC4 |
| `agent.py:211-232` | `_context`: everything per call; `recent_feedback[-8:]` (227); `task_success_source` (231) | PC4 |
| `agent.py:234-260` | `_sync_framework` deep copies every step (605) | PC4 |
| `agent.py:366-372` | `finish` = 1-step hold + logged claim | PC1 |
| `agent.py:382-411` | every fragment end returns control; "pending segments are not automatically resumed" (397) | PC5 |
| `agent.py:494-541`, `574` | one executive call per `act()`; hold fallback; `history_stats()` read every step | PC6 |
| `fragments.py:24-26`, `48-74`, `113-176`, `422-434` | caps 6 / 120 / 240; schema with `observation_after`, `route`; total-step check 174-175; `_complete_segment` ends on `observation_after` or `wait` | PC5 (shared with ws_executor_control) |
| `uc.py:98-135` | `plan_validation`: **`ignored_selection` rules 120-134** (sense 121-122, ref, release grip, p/txt) | PC3 |
| `uc.py:84-95`, `138-145`, `287`, `367-371` | `_parse_lossless` abort; `validate_review_plan` raises; called in `_load` (287) **and** `approve` (371) | PC3 |
| `uc_plan.py:54-58` (`agent_speedup_v2_user_check_plan.py`) | extension conformer knows only object/array/string/number/null (no `integer`) | PC3 |
| `runner.py:305-309`, `940-944`, `965-966` | one `request_plan`; auto-approve; `ReviewError` → `parser.error` (non-zero exit) | PC3 |
| `v2_client.py:149-150`, `903-914` | transient codes (4 stream codes) and `_retry_cause`: 401, 5xx other than overload, 520, deadline not retried | PC2 |
| `v2_client.py:1028-1057` | `request_plan`; re-ask carries `previous_plan` at **1040**; retention issues set at **1047** | PC3 |
| `v2_client.py:193-196` | plan prompt "The robot measures every distance…" (true again in v3) | PC3 |
| `chatgpt_plan.py:37-41` | `UNSUPPORTED_FIELDS` contains `prompt_cache_retention`, `max_output_tokens`; **not** `prompt_cache_key` | PC2 |
| `ordered_transport.py` (whole), `uc_transport.py:104-128`, `413-477`, `480-488` | ordered serializer; **SIGALRM deadline only on the main thread** (108-110); stream path read from `client.last_log_path` | PC2, PC6 |
| `chatgpt_auth.py:450-480` | refresh only when local expiry < 60 s | PC2 |
| `astranomy_agentic_motion.py:75-86`, `89-99` | `rotation_error_rad` is sign-invariant (81); `measured_pose` emits the rotation-vector quaternion with no sign rule | PC4 |
| `base_loop.py:1033-1048`, `1437-1566` | per-step `budget()` / `history()` in `state()`; single-threaded loop; `policy.act` at 1479; `sim.step` at 1522; stop on success 1440-1443 / 1553-1566 | PC1, PC6 |
| `agent_speedup_v2_user_check_framework.py:101-112` | `TaskContract.to_dict` `completion_rule` "The task evaluator remains authoritative" | PC1 |
| `/workspace/gpt_field/src/gpt_field/foundation_vision/live.py:34-55`, `108-170`, `274-299` | H3 instruction block (+1.4k chars, the 2× reasoning confound); summary re-sent in every message | PC4, PC7 |

### 1.2 Offline measurements (new; scripts in the scratch folder above)

**Input-token attribution, H1 T04, last call (`token_breakdown.py`, from `usage.attribution`).**

| Item | Tokens | Note |
|---|---|---|
| instructions | 2,761 | constant; cached when routing hits (call 3) |
| response schema (`text.format`) | 511 | constant |
| observation text, all 13 messages | **75,575** | ≈ 5.8k per message, replayed forever |
| images (6 pairs + stubs) | 3,877 | ≈ 309 per 512² image |
| assistant messages | 3,130 | |
| encrypted reasoning replay | 1,571 | |
| **total** | **87,425** | Σ over 13 calls 641,669; observation text = 82 % of Σ |

Call 3 got 16,000 cached tokens: exactly instructions + schema + the two previous user messages + previous
outputs. The server's prefix cache works on this route when routing hits; the key is a routing hint (§PC2).

**Per-message composition, current message (T04 call 13, characters):** capabilities 4,617; macro 2,270;
last_feedback 3,577; execution 3,611 (duplicates last_result); recent_feedback 3,010 (duplicates the transcript);
geometry 1,888 (gripper, image and external camera are static); measured_state 579; native_evidence 220;
region_directions 206 (static); belief echo 390.

**v3 projection prototype (`v3_projection.py`, all 128 H1 executive requests):** static fields removed, macro
removed, duplicate feedback removed, per-step sample arrays removed, floats rounded to 4 decimals → mean 3,824
chars vs 20,835 (ratio 0.18), max 5,790. Requests losing any `reason`/`detail`/`status` string: **0**.

**Visible output (`visible_parts.py`, 128 responses):** mean 254 visible tokens, 910 chars (3.58 chars/token);
prose 44 %, execution 37 %, obstacle_map 6 %. Mean 2.0 segments. Route overrides used in **0 of 255** segments.

**Quaternion flips (`quat_flips.py`):** see §0 item 1.

**Stream timing:** a zero-reasoning call (T04 #2): headers 2.1 s, first visible 4.0 s, 226 visible tokens
streamed in 3.4 s (≈ 66 tok/s). The fitted marginal cost of a visible token is 0.007 s; the streamed rate is
0.015–0.020 s per token. Visible-token savings below are given as that range.

---

## 2. Component designs

Conventions for all components:
- Model-facing JSON uses `ordered_dumps` (insertion order) and the strict structured-output schema; every
  property is required, nullable through `anyOf`.
- All new code is stdlib + numpy (works under H1's `.venv`, numpy 2.5.3, and `.venv-foundation-vision`, numpy
  1.26.4; no numpy-2-only APIs).
- No per-task code, text, threshold or input. Every number is either an existing robot or executor constant,
  or derived as stated.

### PC1 — Honest termination (prompt and policy side)

**Fixes.** A1 (prompt side), R6, H1 #17, h1_zero_shot F1 and verify_h1_zero_shot F1 correction 5 (both cues must
go), h1_speed F9. IC6: do this first.

**Files (NEW).**
- create `src/gpt_field/v3/protocol_schema.py` (shared with PC4/PC5): `type` enum, `termination` object.
- create `src/gpt_field/v3/policy.py`: `AgentFirstV3Policy(AgentFirstV2Policy)`.
- modify `src/gpt_field/agent_speedup_v2_user_check_framework.py:112`: delete `completion_rule`.
- the prompt text lives in `src/gpt_field/v3/prompt.py` (PC4).

**Wire schema (part of SI-1 `ExecutiveDecisionV3`).**
```json
"type": {"type": "string", "enum": ["execute", "resume", "motion", "observe", "finish", "give_up"]},
"termination": {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": false,
   "required": ["code", "evidence"],
   "properties": {"code": {"type": "string", "enum": ["goal_satisfied", "unrecoverable", "unsafe", "other"]},
                  "evidence": {"type": "string", "minLength": 1, "maxLength": 300}}}]}
```
Validation: `finish` ⇔ `termination.code == "goal_satisfied"`; `give_up` ⇔ code ∈ {unrecoverable, unsafe, other};
every other type ⇔ `termination == null`.

**Prompt text.** Delete `agent_client.py:205-209` ("This only requests a final fresh observation… There is no give_up
action."), `agent.py:128-130` (completion_rule), `agent.py:151` (finish capability text), `agent.py:231`
(`task_success_source`, also removed from `CONTEXT_KEYS` at `agent_client.py:49`), `framework.py:112`. Replace
with (≈ 75 tokens, A/B'd by PC7):
```
finish ends the episode now; nothing runs afterwards. Choose it only when every goal relation holds in the
current images, every task object is released and at rest, and the hand has withdrawn so it touches no task
object. give_up ends the episode now as a failure; state the reason in termination.
```
Release and retreat are therefore part of the completion contract. No sentence anywhere says that someone
else decides success or that "the episode continues". A text-snapshot test enforces this (T1.2).

**Algorithm (policy).**
1. `_apply_agent` (replaces `agent.py:366-372`): for `finish` / `give_up`, abort any committed remainder, set
   `self._terminal = {"kind", "code", "evidence", "step", "decision_index", "virtual_time_s"}` and return a
   1-step hold primitive, so the act/observe contract holds.
2. `act()` copies `self._terminal` into `Decision.diagnostics["terminal"]`. Any later `act()` raises
   `EpisodeTerminated`.
3. The runner (ws_runner_measurement, SI-6) executes that hold step, stops stepping, records the termination,
   and scores strict success in that state. No release-and-settle diagnostic by the evaluator.
4. **Repeat counts instead of a give-up threshold.** Each outcome record (PC4) carries
   `failure_signature = skill|reason|sub_reason|gripper_command|held_object` and
   `repeat_count` / `consecutive_repeats` (earlier outcomes with the same signature). The model decides
   whether to give up; no automatic give-up and no numeric limit. The hard bound is the runner's evaluation
   cap (ws_runner_measurement), recorded as failure and never shown to the model.

**Veto removed / parameter added.** Removes the negative oracle ("not done" every turn) and the "no give_up"
restriction. Adds no threshold.

**Interactions.** SI-6 `TerminalRequest` to ws_runner_measurement (scoring at finish, `stop_on_success` off,
first raw success logged as a diagnostic only). PC4 supplies `repeat_count`.

**Offline tests (`tests/test_v3_termination.py`).**
- T1.1 Schema and validator: finish/give_up need the matching code; other types need null. Pass: 100 % of a
  table of 12 valid/invalid cases.
- T1.2 Text snapshot: the rendered instructions and a rendered first/second payload contain none of
  `BANNED_PHRASES = ("independent evaluator", "evaluator terminates", "evaluator remains authoritative",
  "observation continues", "no give_up", "Simulation pauses", "does not stop the evaluator")`. Pass: 0 hits.
- T1.3 Scripted transport answers `finish` at call 3: `act()` returns a hold with `diagnostics.terminal.kind ==
  "finish"`; the next `act()` raises. Pass: exact.
- T1.4 With ws_runner_measurement's stub runner and a stub simulator that reports raw success at step 10 while
  the agent finishes at step 40: the episode continues to step 41 and records `terminal_reason=agent_finish`.
- T1.5 Three identical failed outcomes give `consecutive_repeats` 0, 1, 2; a different reason resets the
  consecutive count to 0 while `repeat_count` keeps the total.

**Live validation.**
- Capped smoke: none useful (a finish needs a complete task).
- Sweep metrics: finish / give_up / cap counts per task; strict success at finish; time to finish; first
  raw-success step (diagnostic); number of post-finish calls (must be 0).

**Expected effect.** An honest baseline. The report estimates +1–2 calls (+20–40 s) per success (§3.2 A1). It
removes the T04-20-type loop: 9 calls, 159 s and 482 actions after `finish` (h1_zero_shot F1).

**Risks.**
- The honest success rate drops (expected).
- The model may finish early or give up too readily. Mitigation: the completion contract text; the sweep
  reports both LIBERO first-success and finish metrics (IC6).
- A 180 s failure cap would fail every current run. Report time-to-finish distributions (A1 caution).

**Effort.** 1.5 engineer-days.

### PC2 — Model-call resilience, telemetry and the cache key

**Fixes.** A3, H1 #16 (2 launches lost: drawer 002 401 `token_expired`; t8 relayout 520), G2 (per-call
throughput), D3 (stable `prompt_cache_key`). It is also the prerequisite for PC6, because the SIGALRM deadline
works only on the main thread (`uc_transport.py:108-110`).

**Files (NEW).**
- create `src/gpt_field/v3/transport.py`: `ThreadSafeSubscriptionTransport`, a vendored copy of
  `uc_transport.py` with these changes:
  - the deadline is socket timeout = min(`timeout_s`, remaining deadline) plus the existing per-line monotonic
    `check()` in `_ObservedLines.__next__`; no signals;
  - an explicit `request_path` per call;
  - after completion, an echo check of `prompt_cache_key`.
- create `src/gpt_field/v3/telemetry.py`.
- modify `src/gpt_field/chatgpt_auth.py:450`: `access_token(self, *, force_refresh: bool = False)`. With
  `force_refresh` it skips the 60 s early return at line 455.
- create `src/gpt_field/v3/client.py` (shared with PC4/PC6): `ExecutiveClientV3` and the retry policy. The plan
  client (`UserCheckV2Client`, main thread) gets the same `_retry_cause` override through a subclass in
  `src/gpt_field/v3/plan_gate.py`.

**Interfaces.**
```python
class ThreadSafeSubscriptionTransport:
    def __init__(self, *, request_deadline_s: float | None, stream_progress: bool, clock=time.monotonic): ...
    def __call__(self, endpoint, payload, headers, timeout_s, *, request_path: Path | None = None) -> Mapping: ...

RETRY_POLICY = "v3-retry-1"
def retry_cause(exc, *, response_received: bool, committed: bool, retries: dict) -> str | None
    # "auth_refresh" : http_status == 401 and retries["auth_refresh"] == 0
    # "overload"     : existing rule (503 server_is_overloaded), OVERLOAD_MAX_ATTEMPTS = 3
    # "transient"    : http_status in 500..599 (incl. 520-524) or code in TRANSIENT_ERROR_CODES
    #                  or code == "request_deadline_exceeded"; at most TRANSIENT_MAX_RETRIES = 2
    # "cache_key_rejected": http_status == 400 and the payload carried prompt_cache_key (once; key then disabled)
    # None otherwise (400 schema errors, 403, local protocol errors)
```
Retries require that no response envelope arrived and nothing was committed; this is the existing invariant of
`v2_client.py:907-908`. On `auth_refresh` the client calls `token_refresher()`
(`lambda: credentials.access_token(force_refresh=True)`) once, then resends the identical payload. Never an API
key and never a model fallback (rule 5).

**Cache key.** `prompt_cache_key = "gf3-" + sha256(build_digest + "/" + episode_id).hexdigest()[:40]`. It is
identical for every executive request of an episode, distinct across episodes, and sent as a top-level
payload field. `prompt_cache_retention` and `max_output_tokens` stay unsent (`chatgpt_plan.py:37-41`). The
mode is `--prompt-cache-key {off,episode}` and part of the frozen configuration hash. The default is decided by
the capped probe (UD4).

**Telemetry row (SI-9 `CallTelemetry`, `trial/call_telemetry.jsonl`, one per HTTP attempt):**
```json
{"ordinal": 7, "kind": "decide|verify|plan", "attempt": 1, "based_on_step": 196, "applied_step": 196,
 "mode": "blocking|async", "utc_start": "...", "dispatch_wall_s": 0.0, "headers_s": 2.1,
 "first_visible_s": 4.0, "completion_s": 7.6, "latency_s": 7.6,
 "input_tokens": 16124, "cached_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
 "visible_tokens": 226, "visible_chars": 789, "request_bytes": 830000, "images": 2,
 "visible_tok_per_s": 66.2, "pre_visible_s": 1.9,
 "latency_model_s": 6.89, "latency_residual_s": 0.7,
 "retries": {"auth_refresh": 0, "overload": 0, "transient": 0, "cache_key_rejected": 0},
 "prompt_cache_key_sent": true, "prompt_cache_key_echo_equal": true,
 "status": "applied|stale_rejected|schema_error|error", "error_code": null}
```
`latency_model_s = 6.89 + 0.0510·reasoning + 0.0071·visible + 0.0192·input_k` is the cross-critic C1 fit on the
11-02/03 window. The residual is the service-speed normaliser that G2 asks for.

**Veto removed / parameter added.** Removes "episode ends on a recoverable infrastructure error". No new numbers:
it reuses `TRANSIENT_MAX_RETRIES = 2`, `OVERLOAD_MAX_WAIT_S = 60`, the existing 180 s deadline and the
`timeout_s = 180` socket timeout.

**Interactions.** The runner constructs `SubscriptionCredentials` and passes `token_provider` and
`token_refresher` (SI-15). Telemetry goes to ws_runner_measurement (SI-9). PC6 requires the thread-safe
deadline.

**Offline tests (`tests/test_v3_transport.py`).**
- T2.1 A fake transport raises HTTP 401 `token_expired` once: `token_refresher` is called once, the same payload
  is resent, the answer is accepted, `retries.auth_refresh == 1`, and the failed attempt is not committed.
- T2.2 401 twice: raises after exactly one refresh.
- T2.3 503 `unknown_error`, 502, 520 and `request_deadline_exceeded`: at most 2 retries each with backoff; the third
  failure raises.
- T2.4 400 `invalid_request_error` without a cache key is not retried. With a cache key: dropped once, resent,
  `cache_key_disabled = True` for the episode.
- T2.5 Deadline from a worker thread: a fake stream yields a heartbeat line every 0.1 s, the deadline is 0.5 s,
  run inside `ThreadPoolExecutor`. It raises `request_deadline_exceeded` by 0.6 s, and
  `signal.getitimer(ITIMER_REAL)` stays `(0, 0)`.
- T2.6 Telemetry fields and the residual are computed from a recorded `stream.json` (T04 #2): 66 ± 1 tok/s.
- T2.7 The cache key is identical across one episode's payloads, differs across two episodes, and does not
  intersect `UNSUPPORTED_FIELDS`.

**Live validation.**
- Capped probe (≤ 14 calls, PC7 arm C): replay T04 calls 1–7 append-only with the key, interleaved with the same
  sequence without the key. Pass: the key is echoed, and calls 2–7 report `cached_tokens` ≥ 80 % of the
  common prefix.
- Sweep: infrastructure-terminated episodes = 0; retry counts reported.

**Expected effect.** Removes the infrastructure losses (2 seen). If the key is honoured, uncached input per call
falls to about the new message (estimate T04 −3 s, drawer 001 −27 s on top of PC4; hypothesis).

**Risks.**
- The server may reject or ignore the key. The probe settles it; the fallback is the size reduction alone.
- A resent deadline request may double server work. Bounded at 2.
- A revoked session cannot refresh. Bounded at once, then an infrastructure failure.

**Effort.** 2.0 engineer-days.

### PC3 — Plan gate, re-ask and scene inventory

**Fixes.** A4, H1 #14 (t9 draft `(turn D close)` aborts auto-approval), IC7 (object nouns and cells for
perception), V1 (H3 used part phrases as SAM prompts), V2 (the obstacle is named only by the operator).

**Files (NEW).**
- create `src/gpt_field/v3/plan_gate.py`.
- modify `src/gpt_field/agent_speedup_v2_user_check.py`:
  - `plan_validation(raw, *, require_native=False, advisory=True)` skips the `ignored_selection` block (lines
    120-134) when `require_native is False`;
  - `validate_review_plan` raises only on parse/structure errors (it is used at 287 and 371, so both must
    change);
  - `semantic_plan` (75-81) also pops `objects[*].x/w`.
- modify `src/gpt_field/agent_speedup_v2_user_check_plan.py`:
  - `EXTENSIONS = ("regions", "placement_goals", "objects")`;
  - `_conform` (54-58) gains `"integer"` with `minimum`/`maximum` and `"boolean"`. Without this the cells
    raise `KeyError`.
- modify `src/gpt_field/agent_speedup_v2_user_check_spatial.py:39-59`: add `INVENTORY_PROPERTIES["objects"]` next to
  `SPATIAL_PLAN_PROPERTIES`.
- modify `src/gpt_field/agent_speedup_v2_user_check_client.py`: add the inventory paragraph to the plan
  instructions (`UserCheckV2Client.__init__`, after `SPATIAL_INSTRUCTIONS`).
- modify NEW runner `prepare_review` (H1 `runner.py:305-330`): call `request_plan_with_reask`.

**Inventory schema (SI-5 `SceneInventory`, plan extension `objects`).**
```json
"objects": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "object",
  "additionalProperties": false, "required": ["id", "noun", "attributes", "x", "w", "parts", "role"],
  "properties": {
    "id": {"type": "string", "enum": ["O1","O2","O3","O4","O5","O6","O7","O8","O9","O10","O11","O12"]},
    "noun": {"type": "string", "minLength": 2, "maxLength": 32},
    "attributes": {"type": "string", "maxLength": 32},
    "x": {"type": "integer", "minimum": 0, "maximum": 64},
    "w": {"type": "integer", "minimum": 0, "maximum": 64},
    "parts": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": ["A","B","C","D","E","F","G","H"]}},
    "role": {"type": "string", "enum": ["manipulated", "destination", "support", "fixture", "other"]}}}}
```
`maxItems` 12 equals H3's `MAX_TARGETS` (`live.py:27`), the perception batch bound. Plan-instruction addition
(≈ 70 tokens, A/B'd):
```
objects: every distinct physical object visible on or above the work surface, including objects the task does
not mention; task objects first. noun is the plain category name without colour or position words; attributes
holds colour or pattern words that tell same-category objects apart; x and w are its cells as for parts; parts
lists the part ids that belong to it.
```
Advisory checks (each raises one issue code):
- `inventory_part_unbound`: a part id is in no object, or in two.
- `inventory_noun_has_direction_word`: a noun contains any of `DIRECTION_RELATIONS` words or "left, right,
  front, behind, near, far, top, bottom". This is the plan's own frozen vocabulary, not a task list.
- `inventory_duplicate_id`.

**Algorithm (`request_plan_with_reask`).**
```
draft = client.request_plan(...)                     # schema error -> SchemaError
issues = advisory_issues(sanitize(draft))            # relaxed plan_validation + inventory checks
if schema error or issues:
    text = issues_text(issues)[:ISSUES_TEXT_MAX_CHARS]   # 300 chars (v2_client _short_text bound)
    client._reask_issues_override = issues            # used instead of native validate_plan at v2_client.py:1047
    try: draft2 = client.request_plan(..., reask_error=text)   # carries previous_plan (v2_client.py:1040)
    except SchemaError: draft2 = None
    choose draft2 if it parses, else draft; final_issues = advisory_issues(chosen)
if no parseable draft after the re-ask: raise InitialPlanFailed  -> runner status "initial_plan_failed"
plan = sanitize(chosen)                               # parse_plan(..).to_json(); diff recorded, never an abort
store = ReviewStore.create(..., plan=plan, preparation={..., "plan_issues": final_issues, "sanitized": diff})
auto-approve: approve() validates structure only; advisory issues travel in goal_contract.plan_issues
```
One re-ask at most, the existing `PLAN_REASKS = 1` semantics. In manual-review mode, issues are displayed and
never block (UD-defaults).

**Fold option (UD1, the user's decision).** The first executive response carries `goal_contract` (the plan
object above, including `objects`) instead of a separate plan call. Effects:
- the plan call goes away: −14–52 s, mean 29 s, partly offset by a longer first call;
- the human-review option is gone;
- `reviewed_goal_contract` no longer exists as an input;
- the first decision has no inventory-seeded perception, only class-agnostic discovery.
Frozen view-frame and region semantics survive, because `build_view_frame` needs only calibration.

**Veto removed / parameter added.** Removes the macro-only `ignored_selection` veto, the abort on any validation
issue, and the abort on non-ASCII round-trip (`uc.py:89-94`). Adds one bounded re-ask and no threshold.

**Interactions.** SI-5 to ws_perception_service, which binds instances using cells as priors, never as truth.
Inventory ids are the shared object vocabulary for SI-8 (targets), SI-11 (labels) and SI-4 (facts).

**Offline tests (`tests/test_v3_plan_gate.py`).**
- T3.1 Fixtures: the 13 V1 drafts (`/workspace/gpt_field/reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/`)
  and the 7 approved uc7.1 plans (`/workspace/gpt_field/runs/agent_speedup_v2_uc71_*/trial/approved_plan.json`),
  copied into `NEW/tests/fixtures/plans/`. Pass: 20/20 approvable with no `ignored_selection`; t9 passes
  (verify_h1_zero_shot `t9_check.py` reproduced the failure under H1).
- T3.2 Scripted client: a draft with a `spatial_spec` issue triggers exactly 1 re-ask; `reask_error` ≤ 300 chars;
  the payload has `previous_plan`; the corrected draft is accepted. If issues remain, it is approved with
  `plan_issues`. `parser.error` is never reached.
- T3.3 A schema error on the first draft triggers a re-ask with `previous_plan: null`, then accepted.
- T3.4 Two schema errors give `InitialPlanFailed`, recorded as `initial_plan_failed` (no traceback exit).
- T3.5 Inventory: schema conformance through `_conform` (integer cells); part binding checks; `semantic_plan`
  hash unchanged when only `objects[*].x/w` change.
- T3.6 A draft containing "café" and a curly quote is accepted after sanitising, with the diff recorded.

**Live validation.**
- Capped smoke (≤ 10 calls; `prepare` only, no robot action): libero_10 t9, T04+can, T08+bottle, LIBERO-10 t6,
  LIBERO-Goal g5.
- Offline scoring of the drafts against layout truth:
  - inventory recall of task objects (target ≥ 95 %);
  - recall of the added can/bottle and the T04 red-mug distractor (report it; the T08 H1 plan never named the
    bottle);
  - re-ask rate;
  - plan-call latency and visible tokens against H1 (17.4–50.4 s).
- Sweep: `initial_plan_failed` count (target 0).

**Expected effect.**
- No t9-type aborts (1 of 13 recorded drafts).
- Enables V1/V2-free perception: object nouns plus cells instead of part phrases; 6 part-phrase prompts in H3
  returned 0 candidates in every frame.
- Plan call +2–6 s from about 300 extra visible tokens (estimate).

**Risks.**
- The inventory raises plan reasoning tokens. A/B it (S-PC3).
- Wrong cells could mis-bind instances. Perception treats cells as priors.
- 12 objects may be too few for cluttered scenes. Unseen objects fall to class-agnostic discovery
  (ws_perception_service).

**Effort.** 2.5 engineer-days.

### PC4 — Context v3 (static prefix, append-only transcript, compact outcomes)

**Fixes.** D3, H1 #6, h1_speed F7, h1_zero_shot F12, IC9, IC15, H3 F4/F11 (stale vision summaries were 24–30 % of
input), the quaternion double cover (cross_critic C9), and R1 on the prompt side ("Pixels specify rays… infer
depth" is replaced by measured facts).

**Files (NEW).**
- create `src/gpt_field/v3/prompt.py`.
- create `src/gpt_field/v3/context.py`.
- create `src/gpt_field/v3/client.py` (the transcript part).
- create fixture `tests/fixtures/h1_requests_text.json`: the 128 H1 requests' text fields with images and
  encrypted reasoning removed, built by a script from the run logs.

**Interfaces.**
```python
# prompt.py
def render_instructions(*, task_sentence: str, capabilities: dict, robot_facts: dict, camera_facts: dict,
                        region_directions: dict, perception_paragraph: str | None,
                        tool_paragraph: str | None) -> str      # rendered once per episode, then frozen
BANNED_PHRASES: tuple[str, ...]                                # PC1 list + "native macro", "macro="

# context.py
@dataclass(frozen=True)
class ModelViewConfig:
    decimals: int = 4                    # see "parameter" below
    image_retention: Literal["current_only", "sliding6_h1"] = "current_only"   # UD2
class QuaternionHemisphere:
    def __init__(self, reference_wxyz): ...          # first measured orientation of the episode
    def canonical(self, wxyz) -> list[float]: ...    # q if q·q_ref >= 0 else -q
def build_observation(*, step: int, decision_index: int, kind: Literal["decide", "verify"], measured: dict,
                      wrist_camera: dict | None, outcome: dict | None, execution: dict, perception: dict | None,
                      verify: dict | None, notes: list[str], goal_contract: dict | None,
                      cfg: ModelViewConfig, hemisphere: QuaternionHemisphere) -> dict
def project_outcome(evidence: dict, *, decision_index: int, based_on_step: int, ended_at_step: int,
                    history: "SignatureCounter") -> dict          # SI-3 OutcomeRecord
EPHEMERAL_KEYS = ("perception", "wrist_camera", "verify")
def strip_ephemeral(message: dict, *, step: int) -> dict          # applied at insertion into the transcript

# client.py (transcript part)
class ExecutiveClientV3:
    def freeze_instructions(self, text: str) -> None              # first act; byte-identical afterwards
    def prepare(self, observation: dict, images: list[tuple[str, str]], *, kind: str, based_on_step: int,
                fingerprint: dict) -> "PreparedRequest"           # payload = instructions + transcript + current
    def commit(self, result: "CallResult") -> None                # transcript += strip_ephemeral(current) + output
```

**Static prefix (instructions).** Rendered from data once per episode:
1. Protocol text: PC1, PC5 and PC6 paragraphs, plus H1 skill semantics `agent_client.py:122-131` (kept). Deleted:
   the macro paragraph 186-200, the macro/observer sentences in 72-83, and "Pixels specify rays… infer depth"
   (224-225). The replacement is the perception paragraph.
2. The task sentence and `region_directions` (static after review; today repeated in every message,
   `agent_client.py:528-535`).
3. `capabilities`: profiles, per-action caps, `control_hz`, `max_segments`, `max_steps`, from
   `fragment_capabilities()` (ws_executor_control). This is the single source of truth, so a B1/D1 change
   updates the text automatically.
4. Robot facts (SI-14, ws_geometry_tools): grip site, pad centres/size, open gap, palm underside 31.04 mm behind
   the grip (`robot_facts_v2.json:18`; C2), fingertip extent, joint limits and base pose (C5).
5. Camera facts: image size, external camera intrinsics and pose when `pose_source` is fixed, wrist intrinsics
   and mount, and the pixel conventions of `agent_client.py:221-223`.
6. `perception_paragraph` (ws_perception_service) and `tool_paragraph` (ws_geometry_tools: target references,
   `locate`). Caveats are stated once here, never per message (H3 F4).

Token budget: protocol text ≤ H1's non-macro protocol text. The whole prefix ≈ H1's 2,761 tokens plus about 1k
for the moved facts (estimate). This is checked by T4.6 and A/B'd by PC7.

**Per-message observation (current request).**
```json
{"request": "decide", "step": 412, "decision": 9,
 "measured": {"pose_wxyz": [-0.0527, -0.1123, 0.6869, 0.0003, -0.9996, 0.0004, 0.0287], "closure": 0.0319,
              "finger_gap_mm": 77.5, "joints_rad": [7 values], "joint_limit_margin_rad": [7 values],
              "gripper_command": "open"},
 "wrist_camera": {"position_world_m": [3], "rotation_world_from_camera": [[3],[3],[3]]},
 "outcome": {"...": "SI-3 OutcomeRecord"},
 "execution": {"active": false, "remainder": [], "obstacle_map_revision": 3},
 "perception": {"...": "SI-4, current only"},
 "verify": null,
 "notes": ["answer based on step 380 not applied: contact_guard at step 391"]}
```
The first message also carries `goal_contract`: inventory, parts (id, text), goals (do, obj, ref, keep),
regions, placement_goals, hold_true and `plan_issues`. It replaces `reviewed_goal_contract`'s 3,461 chars of
mostly rule text.

**Insertion rule (append-only).** After an applied answer:
`transcript += [strip_ephemeral(current_message), *response.output]`.
- Images become one text stub ("[images of step 412 omitted after use]").
- `perception` becomes `{"snapshot": 17, "omitted_after_use": true}`.
- `wrist_camera` and `verify` are removed.
- Stale-rejected answers (PC6) are not committed; the next message's `notes` says why.
- Request k+1 therefore starts with request k's committed prefix, byte for byte.
- `image_retention = "sliding6_h1"` reproduces H1's `_trim_images` for A/B (UD2).

**Outcome record (SI-3).** Built once per decision from the executor evidence (`progress()` +
`segment_trace`):
```json
{"decision": 8, "based_on_step": 347, "ended_at_step": 369, "executed_steps": 22,
 "status": "succeeded|yielded|failed|aborted|stale_answer", "reason": "fragment_complete", "detail": null,
 "segments": [{"i": 0, "skill": "move_to_pose", "status": "succeeded", "reason": "pose_arrived", "steps": 22,
               "residual_mm": 2.1, "residual_rad": 0.01, "route": {"status": "direct", "reason": null},
               "evidence": {"stall_evidence|gap_change_evidence|contact_evidence": "verbatim, executor-sized"}}],
 "remainder": "none|kept|discarded", "pending_segments": [],
 "measured_end": {"pose_wxyz": [7], "finger_gap_mm": 77.5},
 "events": [{"step": 360, "kind": "contact_guard", "detail": "verbatim"}],
 "failure_signature": null, "repeat_count": 0, "consecutive_repeats": 0}
```
- `reason`, `detail` and routing sub-reasons are never truncated (IC15; executor strings are ≤ about 200 chars).
- Per-step `measurements` arrays stay in the audit file only.
- `recent_feedback` is deleted: the transcript holds every earlier outcome once.
- `execution.last_result` is deleted: it duplicated the evidence.
- The 45k compaction path (`agent_client.py:545-553`) is deleted.
- The 56k hard error stays as a guard that should never fire: no silent truncation.

**Removed dead fields.** `macro`, `native_evidence`, `capabilities.native_macros`,
`capabilities.direct_motion.adaptations`, the `macro` schema property and `type`, the validation branch
`agent_client.py:287-298`, and the policy macro branches (`agent.py:316-348`). `_sync_framework` runs only at
decisions. `obstacle_map` (model-written boxes) leaves the schema in the perception stage (v3-c). It is replaced
by `scene_labels` (SI-11) and automatic boxes (ws_perception_service, ws_swept_planner). v3-a and v3-b keep it,
so the honest baseline stays comparable with H1.

**Prose.** Keep `belief` (≤ 1800; the cross-call memory; observed max 285) and `plan` (≤ 600; observed max 202).
Drop `why` (mean 100 chars ≈ 28 visible tokens). UD5 is defaulted to drop; PC7 checks reasoning tokens.

**Quaternion canonicalisation.**
- Only in the model view, applied to every quaternion sent: measured pose, outcome `measured_end`, remainder
  targets. Rule: `q ← q if q·q_ref ≥ 0 else −q`, with `q_ref` = the first measured orientation of the episode
  (episode data, not code).
- Control is untouched: `rotation_error_rad` is sign-invariant (`astranomy_agentic_motion.py:81`), and the model
  may write either sign.
- Evidence: w ≥ 0 removes 0 of 33 flips (|w| < 0.011 at tool-down); the hemisphere rule leaves 0 in 5 runs.

**Parameter added (derived).** `decimals = 4` for metres and radians in the model view (0.1 mm, 1e-4 rad) and 1
for millimetres. Derivation: the smallest executor arrival tolerances are 2 mm and 0.04 rad
(`fragments.py:31-33`); 1/20 of them is 0.1 mm and 0.002 rad. Four decimals is below both, so rounding cannot
change an arrival-level decision. Control inputs stay unrounded (T4.7).

**Interactions.** SI-3 evidence fields from ws_executor_control. SI-4 facts from ws_perception_service (budget
≤ 2,000 chars, current only, numbers rounded as above, no point samples, no disclaimers). SI-14 robot facts from
ws_geometry_tools. Capabilities from ws_executor_control.

**Offline tests (`tests/test_v3_context.py`).**
- T4.1 Re-render all 128 fixture requests:
  - mean per-message chars ≤ 25 % of H1 (prototype 18 %);
  - every `reason` / `detail` / `status` / routing `reason` string present in the full fragment evidence is
    present in the outcome record;
  - consecutive quaternion sign flips = 0.
- T4.2 Append-only: in a 10-call scripted episode, `ordered_dumps(payload[k+1]["input"][:-1]) ==
  ordered_dumps(transcript after commit k)`, and the instructions are byte-identical for all k.
- T4.3 The committed transcript contains no `input_image`, `perception` facts or `wrist_camera`; the current
  message has exactly one image pair.
- T4.4 No key named `macro`, `native_evidence`, `native_macros`, `adaptations`, `recent_feedback` or
  `task_success_source` appears in any payload; the schema has no `macro`.
- T4.5 A 40-segment synthetic evidence projects below the 56k guard without cutting any reason string. An
  oversize input raises instead of truncating.
- T4.6 The prefix is rendered from capabilities: changing a profile limit changes the text. The text contains no
  benchmark object words (mug, plate, bowl, moka, stove, drawer, basket, ketchup, can, bottle, cabinet,
  microwave, wine, cream, cheese, butter, pudding, book, caddy). Prefix tokens are within budget (counted by a
  chars/3.77 proxy).
- T4.7 Rounding: model-view floats have ≤ 4 decimals; the executor receives unrounded state (asserted on the
  primitive inputs).

**Live validation.**
- PC7 A/B, arm A = H1 payloads, arm B = v3 context on the same states; 20 pairs, ≤ 40 calls. Pass (dev aid, not
  a gate): median paired reasoning-token difference ≤ 0; response validity 100 %; input tokens per call
  −60 % or more at later calls.
- Then a capped live smoke: T04+can, ≤ 6 calls, v3-b against v3-a.
- Sweep: input tokens per call; cached share; calls per task; reasoning tokens per call.

**Expected effect (estimates from the prototype; latency model 0.019 s/1k).**
- T04 Σ input 642k → about 205k: −8 s. Drawer 001 5.7M → about 1.5M: −80 s. This matches the report's D3/F7
  range (T04 −10…−30 s including other levers).
- Plus −3 s / −27 s if caching is honoured (hypothesis).
- Dropping `why`: −0.2…−0.6 s per call (T04 −2.6…−7 s).
- The quaternion change is for correctness only.

**Risks.**
- Fewer images may hurt parallax reasoning. UD2, with a sliding6 switch and live smokes.
- The model may underuse older outcomes in the transcript. `repeat_count` and the belief field mitigate this.
- The text change may move reasoning tokens either way (H3 confound). PC7 gates adoption.

**Effort.** 4.0 engineer-days.

### PC5 — Checkpoint protocol and committed sequences (protocol side of D1)

**Fixes.** D1, H1 #2 (69/128 calls were end-of-fragment checkpoints; 6.5–10.5 calls per object), the grasp and
placement mandates `agent_client.py:161-164`, verify_h1_speed F2 correction 2 (the caps force ≥ 2 calls per
pick-and-place), IC5.

**Files (NEW).** `src/gpt_field/v3/protocol_schema.py` (segment v3), `src/gpt_field/v3/policy.py` (protocol state
machine), `src/gpt_field/v3/prompt.py` (text). Executor semantics live in ws_executor_control's fragment module
(SI-12).

**Segment schema v3 (part of SI-1).**
```json
{"skill": {"enum": ["move_to_pose", "move_relative", "approach", "set_gripper", "wait"]},
 "target": "SI-8 TargetRef | null",
 "quaternion_wxyz": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}, {"type": "null"}]},
 "gripper": {"enum": ["keep", "open", "close"]},
 "profile": {"enum": ["transit", "contact"]},
 "stop_on": {"enum": ["arrival", "contact"]},
 "grasp": {"type": "boolean"},
 "then": {"enum": ["continue", "decide", "verify_grasp", "verify_view", "verify_placement", "verify_engagement"]},
 "max_steps": {"type": "integer", "minimum": 1, "maximum": 120}}
```
- `segments` hold 1..`MAX_SEGMENTS` = 16.
- `route` is removed: used in 0 of 255 segments.
- `observation_after` is replaced by `then`.
- Until ws_geometry_tools lands, `target` = `{"frame": "world"|"relative", "xyz_m": [3]}` with H1 semantics
  (world for `move_to_pose`; displacement from stage entry for `move_relative` / `approach`).

Validation:
- `grasp == true` only on `set_gripper close`;
- `stop_on == "contact"` only on motion skills;
- `then` is ignored on the last segment except for its verify kind;
- no total-step cap (ws_executor_control removes `MAX_EXECUTION_STEPS = 240`, `fragments.py:26,174-175`).

**Parameter added (derived).** `MAX_SEGMENTS = 16` = 2 × 8. One measured pick-and-place has 8 phases (T04 mug 1:
rotate/approach, descend, close, test lift, carry, lower, open, retreat; h1_speed §3). Two objects, or one with
intermediate views, fit. It is a schema bound, not a behaviour threshold.

**Protocol state machine (policy).** States: DECIDING → RUNNING(sequence, i) → {CHECKPOINT(kind) | YIELDED(reason)
| COMPLETE} → DECIDING.
- `then = continue`: the executor enters segment i+1 when segment i completes and all armed predicates hold
  (SI-12).
- Yield: any predicate failure, blocked motion (C1), route problem or joint limit stops the sequence. The
  remainder is kept as `pending` and an outcome record is written (PC4). The model may `resume` once the
  situation is fixed, which fixes F4's discarded close+lift.
- `then = decide`: stop; remainder kept; blocking decision.
- `then = verify_placement | verify_engagement`: blocking. The robot holds after this segment until the answer.
  This is the user rule: blocking checks only for placement relations (on/in) and articulated engagement.
- `then = verify_grasp | verify_view`: asynchronous when `--verify-mode async` (PC6); blocking otherwise.
- Answers: `resume` continues the kept remainder; `execute` replaces it; `motion` replaces it with a short move;
  `observe` aborts it and holds for a fresh decision; `finish` / `give_up` are terminal.

**Prompt text.** Replaces `agent_client.py:107-117` partly and `157-167` fully, plus the short-move sentence at
81-83; ≈ 170 tokens against ≈ 150 removed:
```
then decides what follows a segment. continue: the next segment starts when this one completes and the robot's
checks hold; a failed check, blocked motion, route problem or joint limit stops the sequence and outcome reports
the reason; the remaining segments are kept and resume continues them. decide: stop after this segment and return
control, keeping the remaining segments. verify_grasp or verify_view: you receive fresh images after this segment
while the robot continues; it waits before the next contact, grasp or release until you answer.
verify_placement or verify_engagement: the robot holds after this segment until you answer. Use verify_placement
after releasing onto or into a destination and verify_engagement after engaging a mechanism. Commit a whole
sub-task (for example approach, descend, close with grasp=true, lift, carry, lower, open, retreat) when its targets
are known. grasp=true makes the hand check the measured finger gap after closing and while moving; an empty or
slipping grasp stops the sequence. stop_on=contact ends a motion successfully when it is blocked near its end.
```
There is no mandatory test lift or visual check after a grasp. The grasp predicate is measured (G7: all 19
visually confirmed object grasps settled at 9.3–21.5 mm; both H1 failures settled at 1.4 mm or collapsed to about
1 mm. The deliberate closed-finger pushes at 1.4 mm in the drawer runs are `grasp=false`).

**Veto removed / parameter added.**
- Removes the forced post-grasp and post-placement checkpoints.
- Removes discarding of pending segments on any yield (`agent.py:397`).
- Removes the 240-step total cap and the 6-segment cap (ws_executor_control implements both).
- Adds `MAX_SEGMENTS = 16` (derived above). Grasp thresholds come from robot geometry in the executor: empty
  closure gap plus measured gap noise; not a task number.

**Interactions.** SI-12 `CommittedSequence` API from ws_executor_control: segment entry, predicates, `stop_on`,
yields with verbatim codes, `remainder_summary()`, `resume()`, `replace_remainder()`, `abort()`,
`next_is_barrier()`. SI-8 targets from ws_geometry_tools.

**Offline tests (`tests/test_v3_checkpoints.py`, with a fake executor implementing SI-12).**
- T5.1 Validator: 16-segment sequences pass; `grasp=true` on `move_to_pose` fails; `stop_on=contact` on
  `set_gripper` fails; no total cap.
- T5.2 State machine:
  - continue chains through 8 segments with 0 calls;
  - `decide` keeps the remainder, and `resume` continues it at the next segment index;
  - a new `execute` replaces it;
  - a yield keeps it pending, with a verbatim reason in the outcome.
- T5.3 `verify_placement` holds (0 actions until the answer). `verify_grasp` in blocking mode holds; in async
  mode, see PC6.
- T5.4 Text snapshot: no "test lift" or "inspect new images" mandate; the protocol text is within the token
  budget of the removed text plus 30 tokens.

**Live validation.**
- Capped smoke: T04+can, ≤ 6 calls (runner call cap). Pass: mug 1's grasp-to-release runs inside ≤ 2 calls.
  Report `then` usage by kind.
- Sweep metrics:
  - calls per object;
  - checkpoint calls by kind;
  - pending-discard count (target 0);
  - late-detected grasp failures: a failure found at placement after the predicate passed (IC5 item 4);
  - forbidden contact episodes per grasp (ws_runner_measurement metric).

**Expected effect.**
- Report estimates: T04 −4 calls (−60 s); T08 −8 calls (−125 s); drawer −10 calls (−190 s) (§3.2 D1; h1_speed F2;
  hypotheses).
- Caveat (estimate): committed calls are longer. An 8-segment answer is about 510 visible tokens (+2.4–7 s) and
  likely more reasoning (+300 reasoning tokens ≈ +15 s; the first T04 call had 495). So the net saving per avoided
  call is about 10 s, not 15 s.

**Risks.**
- A committed carry pushes through a contact that only a stall would catch (IC5). It lands only after
  ws_executor_control's C1 guarded motion and grasp/lift predicates.
- The model may keep checkpointing out of habit. Measure `then` usage; do not force it.

**Effort.** 3.0 engineer-days.

### PC6 — Non-blocking verification with virtual-time arrival

**Fixes.** R4, R9 ("Simulation pauses during inference", `agent_client.py:182-183`), the async half of D1, IC14,
and IC10 (fresh images at checkpoint steps).

**What "async" means in simulation.**
- A *virtual clock* advances `dt = 0.05 s` (the 20 Hz control period, a robot property) per executed step, plus
  the idle time the robot spends holding while it waits for an answer:
  `t_v = steps·dt + Σ idle_s`.
- A request dispatched at virtual time `t_v0`, whose wall latency is `L` (observation capture to response
  completion), **arrives at `t_v0 + L`**. Its answer is applied at the first step with `t_v ≥ t_v0 + L`.
- *Pacing*: while a request is in flight, the simulator never runs ahead of the wall clock. Before each step
  it waits until `wall − wall_dispatch ≥ t_v(step) − t_v0`.
- If the harness is slower than real time (0.28–0.40 s/step today; about 0.04–0.07 s after A2), an early
  answer is buffered until its virtual arrival step. Results therefore do not depend on harness speed.
- If the committed motion runs out (or reaches a barrier) before arrival, the robot holds. The simulator is
  paused, as in H1, and `idle_s` absorbs `t_v0 + L − t_v`, so the cost is honest without simulating hundreds of
  hold steps.
- Blocking-only mode gives `t_v = motion + Σ L`, which is H1's real-robot-equivalent time.

**Files (NEW).**
- create `src/gpt_field/v3/scheduler.py`.
- `src/gpt_field/v3/client.py`: split `prepare` (main thread) / `send` (worker) / `commit` (main thread). Lock-free
  `history_stats()` / `budget_snapshot()`. Today `v2_client.py:731` takes `_history_lock`, which the runner reads
  every step (`base_loop.py:1047-1048`), so it would stall the loop for the whole call.
- `src/gpt_field/v3/policy.py` integrates the scheduler.
- uses `src/gpt_field/v3/transport.py` (PC2 thread-safe deadline).

**Interfaces.**
```python
CONTROL_PERIOD_S = 1 / fragment_capabilities()["control_hz"]      # 0.05

@dataclass
class VirtualClock:                         # SI-7, shared with ws_perception_service for async perception
    control_period_s: float
    steps: int = 0
    idle_s: float = 0.0
    def now(self) -> float: ...             # steps*dt + idle_s
    def on_step(self) -> None: ...
    def idle_until(self, t_virtual: float) -> float: ...   # adds max(0, t - now); returns it

@dataclass
class Ticket:
    prepared: "PreparedRequest"; future: "concurrent.futures.Future"
    t_v_dispatch: float; wall_dispatch: float; fingerprint: dict
    invalidated_by: dict | None = None

@dataclass(frozen=True)
class Arrival:
    ticket: Ticket; result: "CallResult"; latency_s: float; t_v_arrival: float

class CallScheduler:
    def __init__(self, client, clock: VirtualClock, *, wall=time.monotonic, max_in_flight: int = 1): ...
    def submit(self, prepared) -> Ticket: ...           # worker = ThreadPoolExecutor(max_workers=1)
    def in_flight(self) -> Ticket | None: ...
    def invalidate(self, event: dict) -> None: ...      # yield-class event during flight
    def pace(self) -> None: ...                         # before each sim step while in flight
    def poll(self) -> Arrival | None: ...               # done and clock.now() >= t_v_arrival
    def wait(self) -> Arrival: ...                      # blocks; clock.idle_until(t_v_arrival)

def staleness(ticket: Ticket, decision: dict, *, state_now: dict) -> str | None   # None = applicable
```
Fingerprint at dispatch: `{gripper_command, grasp_armed, held_object, sequence_id, segment_index,
yield_event_count, pose_wxyz, referenced_objects: {id: snapshot_rev}}`.

**Barrier rule (frozen, general).** During an async flight the committed remainder runs until the first barrier
segment, then holds. A barrier is a segment that:
- has `profile == contact` or `stop_on == contact`; or
- changes the gripper (open or close); or
- carries a `then = decide | verify_*` checkpoint of its own; or
- is the end of the sequence.
So no contact, grasp or release can happen while an answer is pending. Gripper state cannot change during a
flight except through a yield, which invalidates the ticket.

**Algorithm (inside `act`, per step).**
```
1. arrival = scheduler.poll(); if arrival: apply(arrival)
2. if executor yielded at the last transition:
       if in flight: scheduler.invalidate(event); hold (no second request while one is in flight)
       else: submit DECIDE with a fresh frame (FrameProvider.capture_now(), SI-13); block via wait()
3. if executor at checkpoint:
       capture a fresh frame; build observation(kind = verify if verify_grasp/view and async else decide)
       submit; async: continue the remainder; blocking: wait()
4. if in flight and (executor at barrier or nothing to run): arrival = scheduler.wait(); apply(arrival)
5. action = executor.next_action(view) (or hold); if in flight: scheduler.pace(); return Decision
apply(arrival):
   if arrival.ticket.invalidated_by: reject (not committed; notes += reason); submit DECIDE with a fresh frame
   elif decision.type == "resume": commit; continue
   elif decision.type in ("finish", "give_up"): commit; abort remainder; terminal (PC1)
   elif decision.type == "observe": commit; abort remainder; submit DECIDE with a fresh frame
   elif decision.type in ("execute", "motion"):
        if staleness(...) is not None: reject (notes); submit DECIDE with a fresh frame
        else: commit; executor.replace_remainder(decision)
staleness: relative-first replacement and pose moved more than the profile arrival tolerance since dispatch
           -> "relative_target_after_motion"; referenced object missing in the current snapshot ->
           "object_not_in_snapshot"; gripper or held object differ -> "gripper_state_changed"
```
Re-localisation at use: object targets resolve at stage entry from the latest snapshot (SI-8, IC14); world
targets are used as written; relative displacements are taken from the measured entry pose (H1 semantics).

**Verify request content.** The same observation format with `request: "verify"` and
`verify = {"kind": "grasp|view", "after_segment": 3, "remainder": [compact pending segments],
"barrier_segment": 6}`. The answer uses the single decision schema (no separate schema). The prompt (PC5
paragraph plus ≈ 40 tokens):
```
While you decide, the robot holds still unless it is running committed segments. An answer to a verify request
applies when it arrives; one that replaces the remaining segments must start with an absolute or object target.
An answer whose observation a later event invalidated is not applied, and you are told why.
```

**Veto removed / parameter added.** Removes the simulator-only pause privilege. Adds no threshold:
- arrival = measured latency;
- pacing = control period (robot property);
- staleness tolerance = existing profile arrival tolerance;
- `max_in_flight = 1` (structural).

**Interactions.**
- SI-13 `FrameProvider.capture_now()` from ws_runner_measurement: a render without a physics step (A2 renders
  only on demand).
- SI-7 shared with ws_perception_service: async perception results become usable at
  `t_v_capture + compute_wall_s`.
- SI-12 `next_is_barrier()` / `replace_remainder()` / `abort()` from ws_executor_control.
- The runner reads `policy.virtual_time_s` and `policy.protocol_stats()`.

**Offline tests (`tests/test_v3_scheduler.py`; fake transport with scripted latency, scripted answers, stub
executor and simulator).**
- T6.1 `L = 0.6 s`, `dt = 0.05`: the answer applies at step `s0 + 12` for both a fast simulator stub (0.01 s/step,
  paced) and a slow one (0.2 s/step, buffered). Pass: exact step.
- T6.2 Pacing: with a ticket in flight, no step k starts before `wall ≥ (k − s0)·dt`. Pass: all recorded
  timestamps.
- T6.3 Barrier: remainder `[lift(transit), carry(transit), lower(contact), open]` after `verify_grasp`: it holds
  before `lower`, and `idle_s == t_v0 + L − t_v(barrier)` to within 1e-9.
- T6.4 A contact-guard yield injected mid-flight: the answer is not committed (transcript unchanged), `notes`
  holds the reason, and a DECIDE with a fresh frame follows.
- T6.5 A relative-first `execute` after motion during flight is rejected (`relative_target_after_motion`); a
  world-first one is applied.
- T6.6 `history_stats()` and `budget_snapshot()` return in < 5 ms while a 1 s call is in flight (no lock held by
  the worker).
- T6.7 Blocking-only mode: `virtual_time_s == steps·dt + Σ L` exactly. With ws_runner_measurement's saved-decision
  replay (simulator, no model calls), the 13 T04 decisions converted to v3 produce H1's 549 actions until
  ws_executor_control changes the executor (development aid).

**Live validation.**
- Capped smoke: T04+can, ≤ 6 calls, `--verify-mode async`. Pass: at least one verify answer applied
  asynchronously; 0 contact or gripper changes during a flight (from the trace).
- Sweep metrics:
  - overlap seconds;
  - idle seconds;
  - stale-rejection rate and its extra calls;
  - late-detected failures;
  - virtual time to finish against blocking mode (same window, interleaved; ws_runner_measurement).

**Expected effect (estimate).** Bounded by motion between checkpoint and barrier.
- T04 at real-time pacing: the h1_speed F2 overlap bound is 25 s for the current protocol. After PC5 removes most
  checkpoint calls, about 2 grasps × 3.5 s (lift 1.3 s + carry 2.2 s) ≈ −7 s.
- Larger on long carries.
- Its main value is honest real-time semantics (R9) and causal timing for async perception.

**Risks.**
- Concurrency bugs. Mitigation: one in flight, deterministic virtual-time tests, a switch to blocking.
- Stale rejections add calls. Measured.
- Simulator holds are paused, so held-object slip during a wait is not simulated (UD6).

**Effort.** 5.0 engineer-days.

### PC7 — Prompt A/B harness (model-only replay)

**Fixes.** IC9 ("A/B test reasoning tokens"), G2 (same-window interleaving), C1 (the H3 instruction confound),
PC2 cache probe, PC3/PC4 adoption checks.

**Files (NEW).** create `validation/prompt_ab.py`, `validation/prompt_ab_report.py`.

**Interface.**
```
.venv/bin/python validation/prompt_ab.py \
  --source-run /workspace/gpt_field/runs/agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001 \
  --source-run /workspace/gpt_field/runs/agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001 \
  --arms A,B --pairs 20 --max-calls 40 --order interleaved --store-dir .private/chatgpt_plan \
  --output validation/results/prompt_ab_<UTC>.json [--sequence-calls 1-7 --cache-key episode]
```
- Arms: A = the saved H1 payload verbatim; B = the same saved state re-rendered by PC4 (and optionally the
  PC1/PC5 text); C = B plus `prompt_cache_key`.
- The re-render applies `strip_ephemeral` to saved history messages, keeps the saved assistant outputs and
  encrypted reasoning, and drops reasoning items in both arms if the server rejects them.
- Each arm's response is validated against its own schema. No simulator and no action.
- Report:
  - paired differences of reasoning tokens, visible tokens, latency and latency residual;
  - cached tokens per sequential call (arm C);
  - validity rate;
  - agreement of `type`.

**Tests.** `--dry-run` builds payloads without network: arm A payload SHA equals the saved one; arm B payloads
pass the PC4 invariants; interleaving order is deterministic; caps are enforced.

**Live validation.** It is itself a capped smoke (≤ 40 calls, about 10 min).

**Power note.** With 20 pairs, a 2× reasoning change (H3-size) is detectable; a 20 % change is not. Treat it as
a development aid, not a gate (rule 4).

**Effort.** 2.0 engineer-days.

---

## 3. Shared interfaces

| ID | Name | Provider → consumer | Shape (summary) |
|---|---|---|---|
| SI-1 | `ExecutiveDecisionV3` | ws_protocol_context → all | `{belief ≤1800, plan ≤600, type ∈ {execute, resume, motion, observe, finish, give_up}, execution: {segments: [SegmentV3 ×1..16]} \| null, motion: H1 local motion \| null, termination: {code, evidence ≤300} \| null, scene_labels: [{object: O1..O12, role: obstacle\|target\|held\|ignore}] \| null, locate: SI-8 \| null}` |
| SI-2 | `SegmentV3` | ws_protocol_context (wire), ws_executor_control (semantics) | §PC5 schema |
| SI-3 | `OutcomeRecord` | ws_executor_control evidence → ws_protocol_context projection | §PC4 |
| SI-4 | `PerceptionFacts` | ws_perception_service → ws_protocol_context | `{snapshot, captured_step, ready_step, status: ok\|degraded\|unavailable, support_z_m, objects: [{id, noun, attributes, state: located\|single_view\|not_found\|ambiguous\|held, center_m, extent_m, top_z_m, uncertainty_m, seen_in, age_steps}], unlisted: [{id: U#, center_m, extent_m, uncertainty_m}], obstacles_in_map: [ids]}`; ≤ 2,000 chars; current only; no point samples or disclaimers; fail-open as `status: unavailable` |
| SI-5 | `SceneInventory` | ws_protocol_context (plan extension) → ws_perception_service | §PC3 `objects` schema plus parts `[id, text]` and `view_frame` |
| SI-6 | `TerminalRequest` + scoring | ws_protocol_context → ws_runner_measurement | `Decision.diagnostics["terminal"] = {kind: finish\|give_up, code, evidence, step, decision_index, virtual_time_s}`; the runner executes that step, stops, scores strict success at that state, logs first raw success as a diagnostic; cap → `budget_exhausted` |
| SI-7 | `VirtualClock` | ws_protocol_context → ws_runner_measurement, ws_perception_service | §PC6; `policy.virtual_time_s`; async results usable at `t_capture + compute_wall` |
| SI-8 | `TargetRef` / `LocateRequest` | ws_geometry_tools → ws_protocol_context (embed), ws_executor_control (resolve at stage entry) | proposal: `{frame: world\|relative\|object, object: O#\|U#\|null, point: center\|top\|grasp\|null, xyz_m: [3]}`; `locate: [{label, points: [{view, step, u, v}], support: plane\|null}]` |
| SI-9 | `CallTelemetry` | ws_protocol_context → ws_runner_measurement | §PC2 row |
| SI-10 | `Capabilities` | ws_executor_control → ws_protocol_context | `fragment_capabilities()` dict (profiles, caps, `control_hz`, `max_segments`, `max_steps`, predicate list), rendered into the prefix |
| SI-11 | `SceneLabels` | ws_protocol_context → ws_perception_service, ws_swept_planner | semantic overrides only, by inventory id; no metres |
| SI-12 | `CommittedSequence` API | ws_executor_control → ws_protocol_context | `enter(view, execution, keep_gripper)`, `next_action(view)`, `observe(before, action, after) -> events`, `state() -> running\|checkpoint(kind)\|yielded(reason)\|complete`, `next_is_barrier()`, `remainder_summary()`, `resume(view)`, `replace_remainder(view, execution)`, `abort()`, `evidence()` (progress + segment_trace, verbatim reasons) |
| SI-13 | `FrameProvider` | ws_runner_measurement → ws_protocol_context | `capture_now() -> {external_rgb, wrist_rgb, geometry, step, wall_s}` (render without a physics step) |
| SI-14 | `RobotFacts` | ws_geometry_tools → ws_protocol_context | `{grip_site_offset_m, pad_centers_m, pad_size_m, open_gap_m, palm_underside_behind_grip_m, fingertip_extent_m, joint_limits_rad[7], base_pose_world, empty_closure_gap_mm}` from MJCF |
| SI-15 | Credentials hooks | ws_runner_measurement → ws_protocol_context | `token_provider()`, `token_refresher()` (force refresh), `build_digest`, `episode_id` |

---

## 4. Sequencing (respects cross_critic IC5, IC6, IC7, IC9, IC10, IC14, IC15)

1. **v3-a "honest baseline".**
   - PC2 (transport, telemetry; cache key off) and PC7 (harness).
   - PC1 together with ws_runner_measurement A1: finish-scoring, cap, `stop_on_success` off (IC6 first).
   - PC3 relaxed gate and re-ask (veto removal, cross_critic order step 2). No other prompt change, so v3-a
     against H1 isolates the termination contract.
2. **v3-b "context".** PC4 except the perception slot and `obstacle_map` removal. Adopted only after the PC7 A/B
   on reasoning tokens (IC9) and a capped T04 smoke. UD2 fixes the image retention. Compaction keeps reasons
   verbatim (IC15).
3. **v3-c "perception slots".**
   - PC3 inventory, once ws_perception_service consumes it (IC7).
   - PC4 perception facts slot, `scene_labels`, and removal of model-written boxes, once automatic boxes plus
     escape semantics exist (IC13 order belongs to ws_perception_service and ws_swept_planner).
   - The tool paragraph and `TargetRef`, once ws_geometry_tools lands.
4. **v3-d "protocol".** PC5, only after ws_executor_control's guarded motion (C1), grasp/lift predicates and cap
   removal (IC5 items 1–3).
5. **v3-e "async".** PC6, after PC5 and ws_runner_measurement's `FrameProvider` (IC10). Re-localise at use
   (IC14). Included in the acceptance build per UD3.
6. Freeze one build (digest), then the user's all-task sweep with a coverage map. All prompt changes land before
   the freeze (h1_zero_shot F8).

---

## 5. User decisions

| ID | Question | Options | Recommendation |
|---|---|---|---|
| UD1 | Fold the plan into the first executive call? | (a) keep the separate plan call: human review stays possible; reviewed goal contract stays; perception gets the inventory before decision 1; +14–52 s serial. (b) fold: −14–52 s minus a longer first call; no human review; no `reviewed_goal_contract` input; first decision without inventory-seeded perception | (a) for the acceptance sweep; A/B (b) afterwards on 3–4 tasks |
| UD2 | Image retention in the transcript | (a) current pair only (append-only, cacheable) (b) H1 sliding 6 pairs (rewrites the prefix each call) | (a) if PC7 plus a capped T04/T08 smoke show no call-count or behaviour regression; else (b) |
| UD3 | Include asynchronous verification in the acceptance build? | (a) yes, `--verify-mode async` (b) no, blocking verify (most checkpoint calls are already removed by PC5) | (a) only if a capped A/B shows no rise in late-detected failures or stale rejections above 1 in 10 verify calls; else (b) |
| UD4 | Send a stable per-episode `prompt_cache_key`? | (a) per episode (b) off | (a) if the capped probe shows the key echoed and cached tokens rising; else (b) |
| UD5 | Drop the `why` prose field? | (a) drop (≈ 28 visible tokens per call) (b) keep | (a), unless PC7 shows reasoning tokens rising |
| UD6 | Simulate the robot's hold steps during blocking waits? | (a) pause simulator, account idle virtual time (b) step hold actions for the full latency (fidelity for slip; +L/dt steps of wall time per call) | (a) by default; one capped (b) smoke on T04 to check held-object gap drift during waits |

---

## 6. Commands

The new runner's flag names are owned by ws_runner_measurement; these are the flags my components add.
- Capped prompt A/B (engineer may run; ≤ 40 calls): see PC7.
- Cache-key probe (≤ 14 calls): `validation/prompt_ab.py --source-run <T04> --arms B,C --sequence-calls 1-7
  --cache-key episode --max-calls 14 --output validation/results/cache_probe_<UTC>.json`.
- Plan smoke (≤ 10 calls, no robot action): `./run.sh --task libero_10_t9 --gpu 0 --prepare-only --plan-reask on`,
  then repeat for `t04_can_obstacle`, `t08_bottle_obstacle`, `libero_10_t6`, `libero_goal_g5`.
- Capped live smoke (≤ 6 calls): `./run.sh --task t04_can_obstacle --gpu 0 --protocol v3 --context v3
  --verify-mode {blocking,async} --image-retention current_only --prompt-cache-key {off,episode}
  --stop-after-calls 6`.
- Full sweep (the **user** starts it; frozen config, no per-task flags): ws_runner_measurement's sweep command with
  `--protocol v3 --context v3 --verify-mode <UD3> --image-retention <UD2> --prompt-cache-key <UD4>
  --plan-mode <UD1>`.

**Sweep metrics this workstream adds** (per call / per task):
- calls per object and per task;
- checkpoint calls by `then` kind;
- resumes;
- stale rejections;
- overlap and idle seconds;
- late-detected grasp failures;
- finish / give_up / cap counts;
- input, cached, reasoning and visible tokens per call;
- latency residual per service window;
- virtual time to finish;
- `initial_plan_failed` and re-ask counts.

---

## 7. Scripts and data (scratch, offline)

`/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_protocol_context_scripts/`:
- `token_breakdown.py` (+ `t04_breakdown.json`): per-call input attribution;
- `v3_projection.py`: per-message projection prototype over 128 requests;
- `visible_parts.py`: visible-output composition;
- `quat_flips.py`: quaternion sign rules.
