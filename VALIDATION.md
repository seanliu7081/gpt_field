# Implementation validation

Validated on 2026-10-09. This file separates executed checks from experiments that require live credentials.

| Check | Result | Evidence |
|---|---|---|
| Full offline test suite | 98 passed, 44 subtests passed | `.venv/bin/python -m pytest -q` before the frozen live batch |
| Analytic reaching fixtures | 12 scene/method runs completed | `runs/offline_demo/summary.json` and per-run traces/images |
| Current simulator CLI smoke | Passed on LIBERO-Spatial, task 0, initial state 0 | `runs/sim_cli_smoke/result.json` |
| Full policy and simulator plumbing | Six executed actions, field refreshes at steps 0 and 5, 216 candidates logged per step | `runs/sim_mock_policy/probe_checks.json` |
| Camera protocol | 256 render, 180-degree rotation, 224 policy images | Saved allowlisted `runs/sim_cli_smoke/observation.json` |
| Exact Astra request generation | Passed without a key or API call | `runs/request_preview/request.json` |
| Report command | Completed on analytic fixture results | `runs/offline_demo/report.json` |
| Separate metric diagnostic | Completed on illustrative reference values | `configs/distance_fidelity.example.json` |
| Live Astra predictions | Retry completed: 3 valid fields from 3 attempts, no schema retry, no simulator actions | `runs/astra_live_pilot_002/summary.json` and `RESULTS.md` |
| Closed-loop Astra bowl task | **0/10 success**, all ten official initial states reached 220 actions; 440 accepted live fields and 2,200 executed actions | `runs/astra_bowl10_001/RESULTS.md`, `summary.json`, `audit.json` |
| Per-episode annotated videos | Ten H.264 MP4s, 76.2 seconds each; all 1,524-frame streams decoded and reconciled with event indexes | `runs/astra_bowl10_001/offline_gallery.html`, `video_package_manifest.json` |
| pi0.5 reference inference | Not run; optional client/server not installed/launched | External checkpoint/server configuration remains required |
| Full LIBERO-10 comparison | Not run | No benchmark performance claim |

The fixture source is `analytic_oracle`, not a language model. Full-field, one-step, and affine methods all succeeded on the two fixtures, with identical 43-step diagonal and 35-step detour results. The waypoint method succeeded diagonally but timed out on the detour. These fixtures do not show an advantage for full field structure over an affine approximation or for three-step lookahead over one-step lookahead.

The simulator uses benchmark code from `/workspace/heading_flow_pub/third_party/LIBERO` and dependencies from `/workspace/venvs/libero-plus/bin/python`. The latter is only a Python runtime here; the worker explicitly rejects LIBERO-plus benchmark roots and verifies the imported benchmark module comes from the selected source. Source hashes and controller details are retained in the artifact. No global environments were changed.

The full pipeline probe uses a deterministic synthetic field predictor, explicitly marked `not_astra` in its run identity. It made zero API calls. Six positive-x normalized commands of 0.2 produced measured displacement approximately `(0.010749, -0.000050, -0.002429)` meters. This verifies controller direction and exposes tracking lag: six nominal 1 cm delta targets did not produce 6 cm of realized motion. The planner's geometric rollout is an approximation, and every executed step replans from measured state. Its deliberate six-step timeout is not a task or model performance result.

The earlier `runs/sim_smoke` artifact used a preliminary direct-224 rendering path. Use `runs/sim_cli_smoke` for evidence of the corrected 256-to-224 protocol. The first attempted runtime lacked `gym`; that failure is retained in `runs/sim_smoke/arvla_missing_gym.log`.

Test coverage includes the exact diagonal no-op regression, masked segment crossings, fixed field centers, rescaling, auxiliary preservation, all-invalid ablations, state isolation, physical action conversion, subprocess termination, model cadence, schema retries, refusal/incomplete responses, usage and long-context cost reservations, interrupted-budget reconciliation, success after reset, run identity checks, and recovery of an interrupted final result append.

## First live Astra attempt

The saved LIBERO-Spatial observation was submitted to the official Responses API with model `gpt-6-astra`. The account returned HTTP 429, error type `insufficient_quota`, code `credit_balance_exhausted`. The runner stopped after one attempt out of the three-attempt cap, with no retry, no validated field, and no simulator actions. No token usage was supplied by the API; zero usage counters are not evidence of a zero charge. This response does not establish whether the account can access Astra after funding is restored.

The full credential was supplied through hidden terminal input and retained only in process memory. A check found no full credential in the saved run artifacts. The raw request, redacted error, configuration, input hash, and summary are in `runs/astra_live_pilot_001`. At that point, API credits were needed before continuing; the credential pasted in chat should be rotated.

## Successful live retry

The user-authorized retry in `runs/astra_live_pilot_002` produced three validated Astra outputs on the identical saved observation, with no schema retries. Full-field first moves were −y, −y, and +x. Each matched its own affine and one-step baselines. Two fields were numerically affine; the first was mostly affine. All three marked every node traversable. These results establish working field generation, but not correct geometry, a dense-field benefit, or task success. No simulator actions were executed.

The requests consumed 2,217 input and 11,799 output tokens; reasoning tokens are included in the output total. Per-request latency was approximately 76–79 seconds. See the run's `RESULTS.md`, raw responses, candidate logs, and `field_slices.png` for the audited evidence.

## Closed-loop bowl task: ten evaluations

The user-requested single-task batch completed on LIBERO-Spatial task 0, initial-state indices 0–9, with a frozen 220-action horizon. All ten episodes timed out without success. There were no API, schema, simulator, or video errors. Each episode used 44 newly generated Astra fields and executed 220 actions; no cached pilot field was used in this evaluation.

The live loop generated a field every five steps, planned a three-step local path at every measured state, executed the first 1 cm OSC delta target, observed the actual result, and replanned. Ten annotated MP4s show these events separately, including both cameras, field slices, candidate and selected paths, measured movement, and terminal outcome. Playback omits API waiting and labels the actual measured latency.

The policy produced 346 open and 94 hold gripper predictions, with zero close commands. All 2,200 executed gripper commands remained open. Actual end-effector travel averaged 2.13 mm per control step; a 1 cm controller target is not a guaranteed 1 cm displacement. This evaluation failed to progress to a commanded grasp within its horizon. The observations do not isolate a single cause or establish metric distance accuracy.

An independent audit reconciled all 440 raw completed `gpt-6-astra` responses with their observations, fields, local plans, actions, and outcomes, finding zero issues. The video packager independently decoded all ten MP4s and checked event/frame counts and ZIP integrity. Usage was 325,368 input and 1,749,677 output tokens; total batch wall time was 59.37 minutes with ten concurrent episodes. See [full results](runs/astra_bowl10_001/RESULTS.md) and [all videos](runs/astra_bowl10_001/offline_gallery.html).

## No-action-limit follow-up

`LiberoConfig(max_steps=None)` uses an unbounded evaluator iterator and infinite
simulator horizon; finite-horizon runs retain their existing behavior. A real
LIBERO simulator smoke check executed 225 neutral actions on the requested task
without horizon termination (`runs/unbounded_sim_smoke/result.json`). It used no
Astra calls and provides no task-performance result. Offline tests additionally
exercise success beyond the old horizon and preservation of interrupted outcomes.
The full offline suite, including recovery and continuation safeguards, passed
183 tests and 44 subtests.

The new single-episode runner retains the previous $50 per-episode API budget.
An uncapped-spending launch was rejected by automatic approval review because
the user's removal of the action limit did not clearly authorize unlimited paid
API usage. No live API request was made by that rejected launch.

The subsequent launch with the finite $50 API budget successfully initialized
the simulator, then received HTTP 401 (`invalid_api_key`) on its first Astra
request. It generated zero fields and executed zero policy actions. The attempt
is incomplete; no success rate can be inferred. Its 3.6-second initial-observation
and error video fully decoded. See [attempt details](runs/astra_bowl_unbounded_001/RESULTS.md)
and `runner_summary.json`. The user's subsequent explicit retry in
`runs/astra_bowl_unbounded_002` was accepted by the API and is executing the live
closed loop. It executed 565 actions using 113 completed Astra fields before the
next conservative API reservation exceeded the retained $50 budget. It did not
succeed and remains an incomplete evaluation, not a task timeout or 0% success
rate. The gripper never closed. The final independent audit passed 17,539 checks
with zero issues; the complete 190.05-second video decoded successfully. See
[full results](runs/astra_bowl_unbounded_002/RESULTS.md). The previous ten-episode
results remain unchanged.

## Approved $100 continuation and transient-error recovery

The user approved a $100 total API cap including this episode's prior usage.
`runs/astra_bowl_continuation_001` replayed all 565 historical actions with zero
state error and matching camera images, then executed ten new actions from two
new Astra fields. Request 116 returned HTTP 503 (`server_is_overloaded`) without
token usage. The episode stopped incomplete at 575 cumulative actions; its
independent terminal audit passed 459 checks with zero issues. The 138-frame,
6.9-second continuation video decoded successfully.

The ledger retains $23.689025 of confirmed-usage conservative estimates plus
the unresolved request's full $26.8644 reservation. These are accounting
estimates/reservations, not an invoice. No reservation was silently released.

The standalone recovery helper passed 12 focused offline tests covering the
carried ledger, unknown usage, provenance mutation, replay mismatch before
dispatch, live coordinators, and previously spent ancestor ledgers. It leaves
the frozen policy/package/executor sources unchanged. A read-only independent
chain preflight passed 1,427 checks before dispatch.

`runs/astra_bowl_continuation_002` reconstructed all 575 actions with 576 exact
state and image-pair checks. It resumed with request 117 and actions 575–579;
the first live-prefix audit passed 1,675 checks with zero issues. That audit is
partial and does not establish a completed outcome. The same $100 total cap,
unknown reservation, policy, initial state, and absent action horizon remain in
effect. Final outcome and combined video will be reported after this segment
terminates.


## Approved $150 total-budget continuation

The second continuation ended at 1,115 cumulative actions and 223 valid fields,
without LIBERO success. Independent terminal audit: **21,505 checks, zero issues**.
Its 540 new actions and 108 fields are preserved in 3,636 video frames (181.8 s);
full decoding and the complete three-segment video/event chain passed.

The user approved $150 total, including prior usage and unresolved reservations.
The new standalone `continue_bowl_chain_budget.py` passed **18 focused offline
tests** and the real read-only terminal preflight. Independent handoff verification
passed **2,750 checks, zero issues**. It carries 224 attempts and $73.291800 of
conservative accounting, including the earlier HTTP 503's full $26.864400
reservation, then starts request 225 after exact 1,115-action reconstruction.
Only the total cap changes from 100 to 150. Policy, planner, simulator, and earlier
runner sources remain frozen. The launched third continuation is still running;
its eventual outcome will be reported separately from these completed checks.


### Read-only review of the later descent plateau

During the $150 continuation, the observed end-effector height remained near
0.909 m despite repeated normalized z actions of -0.2. A local source/configuration
review found no explicit height clamp at this value, finite episode horizon, or
downward-action scaling error. The adapter's -0.01 m desired delta maps to -0.2
and the installed OSC controller maps it back to -0.01 m. Cartesian
`position_limits` is null; planner workspace z limits are [0, 1.5] m. The unbounded
configuration passes an infinite horizon to LIBERO. Recorded controller hashes
match the installed controller files.

Each policy step targets a delta from the current measured position; downward
targets do not accumulate across stalled steps. The finite-stiffness controller
acts for 50 ms per action and clips joint torques. These physical/controller
constraints may affect motion, but the recorded pose/finger observations omit
arm joint state, contacts and torques. This review does not establish the active
constraint or the cause of the plateau. It used no simulator execution or API
calls and changed no frozen source.

## Direct-action history baseline

Implemented separately in `src/gpt_field/direct_history.py`,
`src/gpt_field/direct_history_video.py`, and
`scripts/evaluate_bowl_direct_history.py`. The original sixteen source files
remain byte-identical to the frozen field continuation's manifest.

- Full project suite: **266 tests and 58 subtests passed**.
- New tests cover complete ordered Responses-output retention, assistant phase
  and opaque reasoning, two-image-pair retention with complete textual history,
  five-action execution, measured state-only feedback, reset, invalid output,
  request interruption accounting, task isolation and video event provenance.
- Read-only preflight confirmed official task 0, initial state 0, seed 7,
  unbounded action horizon, configured runtime/source paths, and the request
  contract without credentials, network requests or simulator startup.
- Real-simulator integration: [`runs/direct_history_offline_smoke/summary.json`](runs/direct_history_offline_smoke/summary.json).
  Two scripted mock responses executed ten actions, produced ten measured
  feedback records, and made **zero real API calls**. Five open commands were
  followed by five close commands. Finger joint positions changed from
  `[0.0390486, -0.0390473]` to `[0.0253189, -0.0253184]` during closure.
  This validates actuator plumbing, not grasping or Astra capability.
- Video: [`episode.mp4`](runs/direct_history_offline_smoke/episode.mp4),
  136 decoded H.264 frames, 1600 by 960, 20 fps, 6.8 seconds. The camera panels,
  command chunk, measured trace and explicit scripted-smoke label were visually
  inspected. The event index contains observation, action-chunk generation,
  issued actions, execution and terminal status, with no fabricated field or
  local-planning events.
- Independent completed-artifact audit: **304 checks, zero issues**. The second
  request contains the five exact first-chunk execution-feedback records and the
  complete first mock response. Initial camera images, eight-value robot state,
  seed and task/initial-state identifiers match the original field episode
  exactly. All eighteen current source hashes and the runner hash match this
  run's manifest. The mock response has one assistant item; encrypted reasoning
  and phase retention are exercised by unit tests, not the simulator smoke.

The smoke trial is unscored: `evaluation_completed=false`, `success_rate=null`.
Comparison with the old field policy changes history, action freedom and
within-chunk control as well as the model's output representation; it is not a
single-variable field ablation.

### First real Astra history episode

The separately authorized $150 trial in
[`runs/astra_direct_history_001`](runs/astra_direct_history_001/summary.json)
used the same task 0, initial state 0, and seed 7, with no action horizon. It
executed **450 actions**, received **90 valid real Astra action chunks**, and
recorded 450 measured feedback records. The model issued 82 close commands and
368 open commands. Its first close command was action 31; its last nonzero arm
command was action 190. Actions 191–450 request zero arm motion with the gripper
open. No LIBERO success was observed.

The run terminated normally through the budget guard, with no pending action
or unresolved API usage. Reported usage totals 9,837,048 input and 11,140 output
tokens (including 2,782 reasoning tokens), with zero reported cached input.
The configured conservative cost estimate is **$123.5201**, not an invoice.
The next $26.8644 request reservation would bring the possible total to
$150.3845 and was refused before an API request was made. This new ledger is
separate from the old field episode.

This is one requested, incomplete episode with zero scored episodes and
`success_rate=null`. It establishes working live history/action/feedback
integration and failure to complete within the authorized budget; it does not
establish a general success rate. The final
[video](runs/astra_direct_history_001/episode.mp4) records the full sequence of
observations, model action chunks, execution, and feedback, with API waiting
periods omitted.

Independent completed-segment audit: **12,626 checks, zero issues**. All 90
request histories and 450 feedback records were reconstructed from actual
outputs and transitions; action/output matching, exact original initial camera
images/state, all eighteen frozen source hashes, and usage totals passed.
See [audit evidence](runs/astra_direct_history_001_audit/audit.json).

An evaluator-only replay reproduced all 450 actions, with zero difference in
451 eight-value robot states and 902 camera images. The target bowl rose
8.964 cm from its initial height. Final bowl–plate contact and relative height
were valid, but horizontal body-origin distance was 3.95018 cm, exceeding the
strict 3 cm goal threshold. These diagnostic object/contact observations were
not given to the policy. See
[reconstruction evidence](runs/astra_direct_history_001_diagnostic/report.json).
The user has since authorized continuation of this same episode with a
cumulative $300 cap; this original terminal segment remains preserved.

Continuation implementation adds new files while leaving the eighteen existing
policy/source files and original evaluator unchanged. **36 focused tests
passed**: 23 history-restoration tests and 13 continuation-runner tests. They
check full ordered Responses output preservation, exact next-request content,
feedback-window restoration, inherited usage/cap changes, stale-parent guards,
state/image/status/provenance mismatch rejection before API access, global
action counters, interruption reservations, and credential redaction.
Read-only reconstruction of the actual parent recovered 665 history records,
450 feedback records, and next global request ordinal 91, with cumulative cost
$123.5201 and the cap changed from $150 to $300. No API or simulator was used by
that history-only preflight.

The paid continuation completed its exact replay gate in the same simulator:
451 robot states and camera pairs, all statuses/provenance, zero state or pixel
error, and zero reconstruction API calls. It then executed 175 new actions from
35 new real model responses. Cumulative totals are **625 actions, 125 calls,
625 measured feedback records, 82 close commands, and 543 open commands**.
Every added action is zero arm motion with the gripper open. No success was
observed; the single episode remains incomplete and unscored.

Cumulative usage totals 18,925,179 input tokens and 14,150 output tokens,
including 2,782 reasoning tokens, with zero reported cached input and zero
unknown-usage calls. The configured cost upper estimate is **$280.3785375**.
The next $26.8644 reservation would exceed the new $300 cap, so the runner
stopped before another request. History, controller, and policy configuration
were unchanged; the additional budget did not produce active correction.

The combined-video helper adds **10 passing offline tests**, including actual
MP4 concatenation with exact decoded-frame preservation, ZIP hashes, and
rejection of altered replay/history provenance. The combined video preserves
the historical budget pause and clearly labels resumption of the same episode.

The independent completed-chain audit passed **17,801 checks with zero issues**,
reconciling every request history, all actual actions and feedback, source and
replay provenance, all cumulative usage, and the twelve long-context-priced
requests. See the
[audit report](runs/astra_direct_history_continuation_001_audit/AUDIT.md).

## Field history and execution feedback implementation (2026-10-10)

The new field-history variant preserves the original field schema, 9³/0.02 m
grid, five-action refresh, three-step/0.01 m planner, and orientation/gripper
adapter. It reuses the direct-history transcript protocol and exact observable
execution-feedback schema. Complete accepted field outputs remain in history,
with original observation origins and only the latest two camera pairs. Reward,
success, object poses, and contact remain evaluator-only information.

The complete test suite passed **349 tests and 72 subtests**. New behavioral
checks cover original-planner/action parity over several refreshes; full field
and Responses-item retention; feedback identity, ordering and state continuity;
invalid responses, schema retries, interrupted usage accounting and budget
denial; runner success/error handling and lack of a live action limit; and video
geometry, event ordering, feedback snapshots and visible smoke labeling.
All **37 pre-existing source and runner files** retain their original hashes.

Real LIBERO smoke at `runs/field_history_smoke_001` completed **10 actions,
2 scripted field refreshes, 10 local plans, and 10 measured feedback reports**,
with five open and five close commands. The second field request contains
feedback for executed steps 0–4. There were **zero live API calls and $0 API
usage**. This is a plumbing diagnostic with scripted fields, not model success.

The smoke video is 1600×960, 138 frames at 20 fps (6.9 seconds), with field,
replan, execution/feedback, initial and terminal events. Its rendered frame was
visually inspected; all event records carry the scripted-smoke label. The
[video](runs/field_history_smoke_001/episode.mp4),
[summary](runs/field_history_smoke_001/summary.json),
[test output](runs/field_history_smoke_001/pytest.txt), and
[source check](runs/field_history_smoke_001/frozen_source_check.json) preserve
the evidence.

Live preflight confirms the separately authorized **$300** cap, no action,
API-call or episode-wall-time limit, and a $26.8644 conservative reservation
before each request. The original pricing was rechecked against the official
[Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra).

The authorized live field-history attempt ended at the budget guard after
**350 actions, 70 real field generations, 350 local plans, and 350 execution
feedback records**. All requests produced valid fields with reported usage;
there were no schema retries, unresolved calls, or action horizon. Forty actions
commanded gripper closing, in three intervals (one-based): 216–230, 261–275,
311–320. The run never reached LIBERO success and remains incomplete/unscored.

Cost upper estimate: **$276.37565**. A further full-context request reservation
would raise the possible total to **$303.24005**, above the authorized $300.
Reported usage was 13,718,776 input tokens and 226,565 output tokens, including
17,935 reasoning tokens and 1,096 cached input tokens. Twenty-two requests used
long-context pricing. The 71st request was never sent.

The independent terminal audit passed **29,284 checks, zero issues**. It
reconstructed all request histories and feedback, reconciled usage, checked
source and initial-condition hashes, and reproduced all 350 complete plans and
actions through the original `FieldPolicy` with identical recorded fields and
observations. The original video has 2,382 frames / 119.1 seconds and decodes
cleanly; all refresh, plan and execution events match the trace. See
[audit evidence](runs/astra_field_history_001_audit/AUDIT.md).

An isolated diagnostic helper adds **11 passing tests** and differs from the
audited direct-history replay helper only in its accepted experiment identity.
Its zero-API replay matched 351 robot states and 702 camera images with zero
state or pixel difference. Gripper–target contact occurred only at actions
261–266. The maximum target body-origin height increase was 7.662 mm at action
267, after contact had ceased. Neither trial-lift interval (226–230 or 271–275)
maintained contact, and the bowl returned to the table. These evaluator-only
measurements were never policy inputs and establish no stable pickup in this
attempt. See the [diagnostic](runs/astra_field_history_001_diagnostic/report.json)
and [full result](runs/astra_field_history_001/RESULTS.md).
# ChatGPT subscription access integration

Added a separate official Sign in with ChatGPT OAuth path and a streaming
subscription-only field client. The first probe uses the recorded initial
observation from `astra_field_history_001` to request one Astra field and one
local plan, with zero simulator actions and no scored episode. Existing field
and direct-history API-key runners remain unchanged.

Validation: 89 selected transport/probe/Astra/field-history regression tests
and 70 subtests passed, followed by 40 authentication tests. These tests use
synthetic credentials and mocked network responses, not subscription inference.
An independent review checked OAuth identity validation, credential isolation,
stream completion, absence of API-key/model fallback, and original-policy
preservation. All 41 pre-existing Python source/runner files matched the source
snapshot. The loopback-only login page returned HTTP 200.

The initial setup artifact recorded **awaiting browser authorization**, before
the user completed login. No inference requests were sent during that setup.
See `runs/chatgpt_plan_setup_001/setup.json`; this historical artifact is retained.
The subsequent live result is recorded below. Subscription
requests omit unsupported `max_output_tokens` and use streaming; API-dollar
accounting is not used to claim subscription or credit cost.

## Successful ChatGPT subscription access

On 2026-10-10, the user completed the official browser authorization. The fourth
subscription probe successfully invoked **`gpt-6-astra`** and generated one valid
**9×9×9 field (729 values)** plus a local plan for the original bowl observation.
The response completed in approximately **58 seconds**, reporting **1,099 input
and 3,256 output tokens**, with 286 reasoning tokens included in the output count.
See [the live summary](runs/chatgpt_plan_field_probe_004/summary.json),
[field](runs/chatgpt_plan_field_probe_004/field.json), and
[plan](runs/chatgpt_plan_field_probe_004/plan.json).

This was a static access and planning check: **zero simulator actions**, no
completed episode and no new success rate. There were **zero API-key calls**.
Subscription allowances or authorized credits may be consumed; their cost is
unknown here. The existing full-episode runners still use their original API-key
clients; a complete subscription-driven episode has not been evaluated.

Four separately recorded one-request probes were needed while fixing response
handling. The first two did not report usable token usage. The third returned a
completed Astra response and usage, establishing model access, but the reader
discarded output items preceding an empty terminal `output` array. Its historical
summary predates the separate access/pipeline flags and is preserved unchanged.
The fourth correctly assembled complete `response.output_item.done` objects after
`response.completed`, then passed the original field schema and local planner.
The transport also accepts the observed `text/plain` MIME label only through
strict SSE parsing. Model, prompt, field schema and local planner were unchanged.

The final selected authentication/transport/probe/Astra/field-history regression
suite passed **141 tests and 104 subtests**. All **41 pre-existing Python files**
still matched their pre-subscription hashes. Aggregate probe usage and validation
are recorded in [the connection result](runs/chatgpt_plan_setup_001/live_result.json).
An [independent artifact audit](runs/chatgpt_plan_field_probe_004/audit.json)
passed **56 checks with zero issues**, including exact offline reproduction of
the full 216-candidate local plan and first action with the original planner.

## Subscription episode stopping after the first close command

The user authorized one fresh subscription-driven execution test on the same
LIBERO bowl task, stopping after the first gripper close action. The additive
`evaluate_bowl_subscription_close.py` uses the existing subscription client,
field/history policy, planner, and video recorder. It stops only after `sim.step`
acknowledges an actual positive gripper command and the post-action observation
and execution feedback have been recorded. It sends no further action after
that milestone. This diagnostic is unscored and does not claim a grasp or full
finger closure. The authorization is saved in
`runs/chatgpt_subscription_close_authorization_001.json`.

The preflight passed without authentication or inference. The real LIBERO smoke
in `runs/subscription_close_smoke_001` executed exactly **five open actions and
one close action**, using two scripted fields, six local plans, and six feedback
records. It made **zero network/model calls** and stopped immediately after the
sixth action. The measured finger-joint aperture proxy narrowed by approximately
0.441 mm on that first 50 ms closing action. No live-model ability is inferred
from this smoke.

The new runner has **27 passing tests**. The selected runner, subscription
transport, existing field-history runner and video regression suite passed
**89 tests and 69 subtests**. An independent review found no blocker in the
stopping semantics, pending-action/error bookkeeping, credential isolation, or
recorded smoke. All **45 pre-existing Python files** matched their source hashes.
The smoke artifact audit passed **252 checks with zero issues**. Its 122-frame,
6.1-second, 1600×960 H.264 video decoded fully; the final executed frame was
visually inspected. See `runs/subscription_close_smoke_001_audit/audit.json`.

The live subscription run in `runs/chatgpt_subscription_close_001` reached the
requested diagnostic stop after **241 executed actions**: 240 open commands,
then one close command at zero-based step 240. It used **49 real completed Astra
field generations**, 241 local plans and 241 measured feedback records. No action
followed the acknowledged close, and there were no errors or unresolved requests.
It took **2,430.26 seconds (40.50 minutes)**, with no action-step horizon.

The finger-joint aperture proxy changed from **79.9986 mm to 79.1214 mm**, a
**0.8772 mm** decrease in one 50 ms close-control cycle. This establishes executed
closing motion, not full closure or a grasp. LIBERO success was not observed;
the user-requested early stop remains unscored (`success_rate=null`).

All 49 inference requests used the ChatGPT subscription; **API-key calls: 0**.
Reported usage was **6,411,963 input and 143,125 output tokens**, including
1,729 reasoning tokens. There were zero unknown-usage requests. No API-dollar
price is applied to these subscription counts; subscription/credit cost is unknown.

The independent artifact audit passed **6,134 checks with zero issues**, including
initial robot state and both camera images matching the earlier field run,
all 45 frozen source files, history/feedback retention, and exact stopping.
The **1,673-frame, 83.65-second, 1600×960 H.264 video** decoded fully, and its final
executed frame at 80.55 seconds was visually inspected. See the
[video](runs/chatgpt_subscription_close_001/episode.mp4),
[viewer](runs/chatgpt_subscription_close_001/index.html),
[summary](runs/chatgpt_subscription_close_001/summary.json), and
[audit](runs/chatgpt_subscription_close_001_audit/audit.json).

A separate [policy reconstruction audit](runs/chatgpt_subscription_close_001_audit/policy_audit.json)
passed **2,447 checks with zero issues**. It reproduced every complete local plan,
action and feedback record through the original policy and verified all 49 full
request histories, including exact opaque output retention and image-history
pruning. The first 48 fields commanded open/move; the final field commanded
close/hold. All requests completed with the exact requested `gpt-6-astra` model,
and reported usage reconciled exactly.

## Hybrid subscription control: implementation validation

The user requested Astra fields for coarse motion and Astra direct actions near
the active object interaction region, using the existing ChatGPT subscription.
The explicit stop choice was to attempt the complete transport task. A fresh
episode uses reversible model-predicted routing every five executed actions,
one shared transcript, the original coarse planner, and unchanged direct-action
semantics. There is no action, request-count, or episode-time limit. This is a
new prompt/routing experimental condition, not the original field-only policy.
See `runs/chatgpt_hybrid_authorization_001.json`.

Selected client/policy/original-field/subscription regression tests passed
**140 tests and 83 subtests**. The new runner and video passed **21 tests**.
These verify original coarse action/plan equality, exact direct actions,
field → direct → field transitions, fresh field centers, shared feedback,
preserved cross-controller gripper intent, and full-task stopping after close.
All **47 preceding Python source files** matched their frozen hashes.

The real LIBERO integration smoke in `runs/chatgpt_hybrid_smoke_001` completed
**15 executed actions**, **3 scripted responses**, **2 controller switches**,
**10 local plans**, and **15 feedback records**. The final coarse field's hold
retained the close intent from direct control. There were no live model calls
and no performance score. Its **204-frame, 10.2-second** video decoded fully;
the direct-control camera/action/feedback panel was visually inspected.

The independent smoke artifact audit passed **739 checks with zero issues**
(`runs/chatgpt_hybrid_smoke_001_audit_002/audit.json`), including complete request
history equality, 10 reconstructed original local plans, exact direct commands,
measured feedback, source/provenance checks, and full video decoding. A later
audit-only refinement distinguished proposed, execution-marked and acknowledged
actions for interruption tails; its repeat smoke passed **740 checks**.
Tampered-copy checks rejected altered direct actions and prior model history.

Live run `runs/chatgpt_hybrid_001` reached **LIBERO success after 448 actions**,
with **1/1 scored episodes successful**. It made **90 real subscription requests**:
51 field responses and 39 direct chunks. There were 255 field-planned actions,
193 exact direct actions, and 448 measured feedback records. Zero-based mode
switches were 225 (direct grasp), 385 (field transport), and 415 (direct placement).
The first close command occurred at step 235. Several observed grasp misses
were followed by model-generated reopen/reposition/close/lift attempts; later
camera frames showed the bowl rising and moving with the fingertips.

The simulator first reported success after step 447, and no later action was
sent. The final action was `[0.15,-0.35,-0.08,0,0,0,1]`, still commanding close.
The last open command occurred at step 354. This is benchmark success; no
release command or post-release stability test followed. A single successful
episode is not a reliable population success-rate estimate.

Elapsed time was **2,380.64 seconds (39.68 minutes)**. Reported usage was
**15,010,578 input and 90,284 output tokens**, including 2,916 reasoning tokens.
There were zero unknown-usage requests, zero API-key calls, and no errors.
Subscription monetary cost is not inferred from API-key prices.

The independent [artifact audit](runs/chatgpt_hybrid_001_audit/audit.json) passed
**18,605 checks with zero issues**, including all 90 complete shared request
histories, opaque output equality, exact direct-action slices, independent
recomputation of all 255 original local plans/actions, measured feedback,
mode switches, token reconciliation, and unchanged source hashes. All 47
preceding frozen files and 27 recorded modules matched; initial state and both
RGB views exactly matched the prior same-task field episode.

The **3,559-frame, 177.95-second, 1600×960 H.264 video** decoded fully. Its final
executed frame at 174.85 seconds was visually inspected. The
[viewer](runs/chatgpt_hybrid_001/index.html) links jump points for each controller
handoff, first closure, lifting and benchmark success; the terminal closed-gripper
limitation is visible beside the video. See also the
[MP4](runs/chatgpt_hybrid_001/episode.mp4) and [summary](runs/chatgpt_hybrid_001/summary.json).
