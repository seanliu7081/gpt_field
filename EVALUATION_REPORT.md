# Astra robot-control experiments: methods, results, and analysis

Evidence cutoff: 2026-10-11. For equations, algorithms, and software/model structure, see [METHOD_AND_ARCHITECTURE.md](METHOD_AND_ARCHITECTURE.md). This report summarizes recorded experiments in this repository; preparing it did not run new inference or robot actions. See [IMPLEMENTATION_PROTOCOL.md](IMPLEMENTATION_PROTOCOL.md) for the implemented contracts, control loop, and reproduction commands. The earlier [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) records the original research proposal; [VALIDATION.md](VALIDATION.md) preserves the chronological validation record.

**The hybrid method achieved the LIBERO success predicate in one episode, after 448 actions.** Its observed result is **1/1**, with the gripper still closed at termination. Release and subsequent placement stability were not tested. Direct control and field planning with history each stopped at their API budget before success; both remain incomplete, with `success_rate=null`. These results demonstrate different behaviors, but do not establish that hybrid control has a higher general success rate.

## 1. Task, conditions, and what counts as success

All robot experiments discussed here use the official **LIBERO-Spatial task 0**:

> pick up the black bowl between the plate and the ramekin and place it on the plate

The original finite-horizon field batch used official initial states 0–9. Subsequent single-episode field, direct, and hybrid experiments used **initial state 0, seed 7**. The audited later runs match the earlier initial robot state and both camera images. These are simulator experiments, not physical-hardware deployments. Despite the original proposal's LIBERO-10 scope, a full LIBERO-10 comparison has not been run.

The task's BDDL goal is `(On akita_black_bowl_1 plate_1)`. In the recorded benchmark implementation this requires:

1. The bowl body origin is at or above the plate body origin in z.
2. The bowl and plate have collision contact.
3. Their horizontal body-origin separation is **strictly less than 0.03 m**.

It does **not** require an open gripper, a release action, prior lifting, upright orientation, or a stability dwell. The evaluator stops immediately when LIBERO reports success. Therefore benchmark success and verified released placement must be reported separately. Sources: [task definition](../heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl), [On predicate](../heading_flow_pub/third_party/LIBERO/libero/libero/envs/predicates/base_predicates.py), and [check_ontop implementation](../heading_flow_pub/third_party/LIBERO/libero/libero/envs/object_states/base_object_states.py).

For these dedicated reports, a scored episode ends in benchmark success or a predeclared finite-horizon timeout. An API budget stop, unresolved request, infrastructure error, or user-requested first-close stop is recorded separately. Continuations, deterministic replays, video packages, and static probes are not additional scored episodes. No pooled success rate across these different conditions is meaningful. Some older generic `summary.json` files emit `0.0` for infrastructure/budget stops; this report uses the dedicated runner outcomes and actual termination evidence to distinguish those from scored task failures.

## 2. The three implemented methods

| Component | Direct + history | Plan-only + history | Hybrid + history |
|---|---|---|---|
| Astra output at each refresh | Five normalized 7D actions | A 9³ scalar field plus orientation/gripper/motion auxiliaries | A routing decision and either the field or five direct actions |
| Translation control | Execute the accepted action chunk | Original three-step local planner; execute its first move and replan | Original field planner during coarse motion; exact direct actions near the active interaction region |
| Astra refresh | Every five executed actions | Every five executed actions | Every five executed actions, including a new mode decision |
| Between Astra calls | Execute the remaining accepted actions | Replan on every new measured robot state using the held field | Behavior of the selected branch |
| Historical context | Complete textual observations, accepted model outputs, and measured feedback | Same history protocol, including previous full fields at their original centers | One shared history across all switches |
| Image history | Latest two observation image pairs | Latest two observation image pairs | Latest two observation image pairs |
| Evaluation reward / success provided to model | No | No | No |
| Recorded full-task transport | API key | API key | ChatGPT subscription; zero API-key calls |

The model sees external-camera RGB, wrist-camera RGB, task text, and eight robot-state values: end-effector position, axis-angle orientation, and two finger-joint positions. Feedback reports commands and measured robot motion, not hidden bowl coordinates or grasp/contact truth. A five-action chunk lasts 0.25 simulated seconds at 20 Hz; simulation pauses while inference runs.

The field is a **unitless lower-is-better spatial potential**, not an experimentally calibrated Euclidean or Riemannian distance. It has 729 samples at 2 cm spacing over a cube extending 8 cm from the observed end-effector position in each direction. The original planner uses 1 cm axis moves and a three-step horizon, yielding 216 nominal action sequences. It selects translation; Astra still supplies orientation and gripper intent. Predicted invalid/unknown masks are not ground-truth collision geometry.

The earlier **stateless field** condition has the same field/planning interface but lacks the complete history and explicit execution-feedback transcript. It must not be merged with the later field-history condition. The standalone subscription direct client is implemented and tested offline; the direct episode reported below used the API-key history runner. No separate full standalone direct-subscription episode is claimed.

Sources: [direct policy](src/gpt_field/direct_history.py), [field-history policy](src/gpt_field/field_history.py), [hybrid policy](src/gpt_field/hybrid_policy.py), [hybrid client/prompt](src/gpt_field/hybrid_subscription.py), and [planner](src/gpt_field/planner.py).

## 3. Recorded evaluation outcomes

“Calls” in this table means validated live model outputs, except where an unresolved dispatch is explicitly noted. Actions exclude reset settling and deterministic reconstruction.

| Condition and run | Independent rollouts / initial states | Executed actions | Live outputs | Recorded outcome | Score interpretation |
|---|---|---:|---:|---|---|
| Original stateless field: [bowl10](runs/astra_bowl10_001/summary.json) | 10 / indices 0–9 | 2,200 total; 220 each | 440 fields | All ten hit the declared 220-action horizon; no infrastructure failures | **0/10 = 0%** for this finite-horizon condition |
| Stateless field without an action horizon: [unbounded + continuation chain](runs/astra_bowl_continuation_003/run_state.json) | 1 / index 0, across four segments | 1,865 cumulative acknowledged actions | 373 fields; 375 request records, with two lacking usage | No success; last segment has no final summary and an unresolved final request | Incomplete, unscored; stale `running` file is not proof of a live process |
| Direct + history: [original](runs/astra_direct_history_001/summary.json) + [continuation](runs/astra_direct_history_continuation_001/summary.json) | 1 / index 0 | 625 cumulative | 125 chunks | No success; cumulative $300 budget guard | Incomplete; `success_rate=null` |
| Plan-only + history: [field_history](runs/astra_field_history_001/summary.json) | 1 / index 0 | 350 | 70 fields | No success; $300 budget guard | Incomplete; `success_rate=null` |
| Plan-only + history, first-close diagnostic: [subscription_close](runs/chatgpt_subscription_close_001/summary.json) | 1 diagnostic / index 0 | 241 | 49 fields | Stopped after the first acknowledged closing action | Intentionally early; `success_rate=null` |
| Hybrid + history: [hybrid](runs/chatgpt_hybrid_001/summary.json) | 1 / index 0 | **448** | **90: 51 fields + 39 direct chunks** | **LIBERO success; gripper remained closed** | **1/1 observed**, with no release/stability test |

The no-horizon stateless chain is `astra_bowl_unbounded_002` → `astra_bowl_continuation_001` → `astra_bowl_continuation_002` → `astra_bowl_continuation_003`. The cumulative endpoints are 565 → 575 → 1,115 → 1,865 actions. The last segment contributes 750 new actions; replaying its 1,115-action prefix does not increase the evaluation count. All 1,865 acknowledged actions kept the gripper open. Request 116 returned HTTP 503 without usage. Only 374 request records have saved outcomes; final request 375 has no saved response or result, so its dispatch/completion is unconfirmed. The final segment has no complete terminal audit. The earlier `astra_bowl_unbounded_001` attempt failed authentication before any policy action and is not an additional behavioral failure. See [chain analysis](scripts/analyze_bowl_chain.py) and the saved segment manifests/traces for provenance.

The 0/10 field batch and the 1/1 hybrid result used different horizons, histories, routing, transport, and numbers of initial states. They are not a controlled 0%-versus-100% comparison.

### 3.1 Direct control: lifting followed by persistent inactivity

The original direct segment executed 450 actions from 90 model responses. Its continuation added 175 actions from 35 responses under a cumulative $300 cap, for **625 actions and 125 responses in the same episode**. All accepted actions were faithfully executed.

The first close command was action 31; there were 82 closing commands overall. An evaluator-only replay of the original 450 actions matched all 451 recorded robot states and 902 camera images exactly. It measured a maximum bowl height increase of **8.964 cm**, with gripper contact at the peak. At action 450, bowl–plate contact and the height condition passed, but horizontal separation was **3.95018 cm**, exceeding the strict 3 cm threshold. This object-state measurement belongs to action 450, not the final action 625.

Actions **191–625** were all `[0, 0, 0, 0, 0, 0, -1]`: **435 consecutive stationary-arm/open-gripper commands**. The continuation's 175 actions produced only about 0.121 mm of measured end-effector path drift. This shows model-generated stagnation rather than missing action dispatches. The hypothesis that Astra considered the task finished is plausible, but its internal reason is not established. The model had no explicit “task not yet successful” signal or evaluator threshold in its inputs.

Sources: [cumulative results](runs/astra_direct_history_continuation_001/RESULTS.md), [original replay diagnostic](runs/astra_direct_history_001_diagnostic/report.json), and [chain audit](runs/astra_direct_history_continuation_001_audit/AUDIT.md).

### 3.2 Plan-only: distinguish absent grasp commands from failed grasp execution

In the original stateless ten-episode batch, the model produced 346 `open` and 94 `hold` predictions, and **zero `close` predictions**. All 2,200 executed gripper actions remained open. The measured end-effector travel averaged about 2.13 mm per control cycle, despite a nominal 10 mm translation target. This condition failed to reach a commanded grasp within the 220-action horizon. It does not show that the gripper command interface was broken. [Batch results](runs/astra_bowl10_001/RESULTS.md)

Adding history and measured execution feedback changed the observed behavior: the field-history episode issued **40 closing actions**, in intervals **216–230, 261–275, and 311–320** using one-based action numbers. It also requested reopening, alignment changes, and trial lifts. Nevertheless, the 350-action attempt did not produce a stable pickup.

The exact evaluator-only replay found gripper–target contact only during actions **261–266**. The bowl's maximum body-origin height increase was **7.662 mm** at action 267, after contact had ceased. Neither trial-lift interval, actions 226–230 or 271–275, maintained contact, and the bowl returned to the table. Final bowl–plate contact was false. Thus closing the fingers and generating an upward command were insufficient to establish a grasp. These hidden object/contact measurements were never supplied to Astra. [Field-history results](runs/astra_field_history_001/RESULTS.md), [replay diagnostic](runs/astra_field_history_001_diagnostic/report.json), [contact intervals](runs/astra_field_history_001_diagnostic/grasp_intervals.json)

The subscription first-close diagnostic is a different stopping condition. It sent 240 open actions followed by one close action, then stopped as requested. The finger-joint aperture proxy narrowed from 79.9986 mm to 79.1214 mm in that single 50 ms cycle. This verifies executed closing motion; it does not test a full grasp, lift, or completed task. [Diagnostic summary](runs/chatgpt_subscription_close_001/summary.json)

### 3.3 Hybrid: coarse planning, contact retries, transport, and placement

The hybrid run made one shared-history routing decision every five actions. Its actual execution was:

| One-based executed actions | Controller | Count | Observed role |
|---|---|---:|---|
| 1–225 | Coarse field | 225 | Approach the target bowl |
| 226–385 | Direct | 160 | Alignment, closing, reopening, and grasp/lift retries |
| 386–415 | Coarse field | 30 | Transport toward the plate while retaining closed-gripper intent |
| 416–448 | Direct | 33 | Placement corrections until LIBERO success |

There were **255 original local plans**, **193 exact direct actions**, and **448 measured feedback records**. The first close command occurred at action 236. Initial closing/lifting attempts did not visibly lift the bowl; later camera frames showed the bowl rising and moving with the fingertips. This is visual evidence consistent with a successful grasp and transport, not an independently replayed hidden-contact measurement for the hybrid run.

The handoff back to coarse control generated a fresh field centered at the current measured end-effector position. The controller retained gripper intent across switches. The final direct chunk was truncated after three of its five proposed actions because LIBERO reported success after action 448. No later action was sent.

The final action was `[0.15, -0.35, -0.08, 0, 0, 0, 1]`. The last opening command was action 355; actions 356–448 continued to command closing. Accordingly, the strongest supported result is **benchmark placement success with the gripper still commanded closed**, with post-release stability unmeasured. [Summary](runs/chatgpt_hybrid_001/summary.json), [complete trace](runs/chatgpt_hybrid_001/trace.jsonl), [final executed frame](runs/chatgpt_hybrid_001/final_execution.png), [audit](runs/chatgpt_hybrid_001_audit/AUDIT.md)

## 4. History, observation, and model settings

History is part of all three later variants. They preserve ordered textual observations, complete accepted model outputs, executed actions, and measured feedback. Only the most recent two RGB observation pairs remain attached as images. The hybrid carries one transcript across field/direct switches; switching does not reset history. There is no separate critic model or extra “history-check” inference call.

For the hybrid episode, the logs contain **449 simulator observations**: one initial observation and 448 post-action observations. Astra received **90 current image/state inputs**, at zero-based action boundaries 0, 5, …, 445. Field control replanned between these boundaries from measured state; direct control executed the remaining chunk without another model decision. Observation every action therefore does not mean model inference every action.

The video has only one standalone `OBSERVE` event because that label is emitted at reset. Later post-action images and feedback appear in execution events. This display choice does not mean the model saw only the first image. Complete histories and transitions were checked by the [hybrid artifact audit](runs/chatgpt_hybrid_001_audit/audit.json).

All 125 completed direct responses, all 70 completed field-history responses, all 90 completed hybrid responses, and all 49 completed subscription first-close responses recorded `model=gpt-6-astra`, `reasoning.effort=medium`, and `service_tier=default`. These runs do not substantiate an Ultra/Fast/Ultrafast configuration. Requests omitted the reasoning-effort field; the observed effort comes from actual response metadata. The current runners do not expose reasoning effort or service tier as user-configurable CLI settings. See [request construction](src/gpt_field/astra.py) and the respective saved response records.

The subscription runs used the authenticated ChatGPT transport with **zero API-key fallback calls**. That changes access/accounting, not the robot's low-level controller. No claim of unlimited subscription usage or a measured subscription dollar cost is made.

## 5. Usage, runtime, and the consequences of retained history

| Episode / diagnostic | Input tokens | Output tokens, including reasoning | Reasoning subset | Recorded cost accounting |
|---|---:|---:|---:|---|
| Direct, cumulative 625 actions | 18,925,179 | 14,150 | 2,782 | $280.3785375 conservative estimate under a cumulative $300 cap |
| Plan-only + history, 350 actions | 13,718,776 | 226,565 | 17,935 | $276.37565 conservative estimate under a $300 cap |
| Subscription first-close, 241 actions | 6,411,963 | 143,125 | 1,729 | Subscription monetary cost unknown; zero API-key calls |
| Hybrid, 448 actions | 15,010,578 | 90,284 | 2,916 | Subscription monetary cost unknown; zero API-key calls |

Sources: [direct ledger](runs/astra_direct_history_continuation_001/summary.json), [field ledger](runs/astra_field_history_001/summary.json), [first-close ledger](runs/chatgpt_subscription_close_001/summary.json), [hybrid ledger](runs/chatgpt_hybrid_001/summary.json). All four have zero unknown-usage requests. Field history reported 1,096 cached input tokens; the other three reported zero.

API dollar values are configured conservative estimates, not invoices or current price quotations. The next request needed a $26.8644 reservation: the direct continuation would have reached a possible $307.2429375, and field history $303.24005, so neither request was sent. The unfinished older stateless chain has different unresolved accounting and must not be treated as a settled final bill.

Preserving every textual observation, model output, and feedback message makes later prompts grow. A dense field also adds 729 scalar values to each accepted output and to later histories. The field-history episode's 226,565 output tokens across 70 calls versus the direct episode's 14,150 across 125 calls illustrates a substantial representation overhead in these runs. This is not an equal-trajectory efficiency benchmark: the behaviors, prompts, episode lengths, and long-context pricing exposure differ.

The hybrid took **2,380.64 seconds (39.68 minutes)** wall time. Its 448 actions span **22.4 seconds of commanded simulation time**, while the annotated video lasts 177.95 seconds. The field-history episode took 53.68 minutes, and the first-close diagnostic 40.50 minutes. Simulation pauses during inference, and videos omit those waits. These experiments do not demonstrate control at a real-time 20 Hz model inference rate.

## 6. Interpretation: what explains the observed differences?

### 6.1 Why field planning did not achieve a stable grasp

Several mechanisms are consistent with the evidence; none has been isolated by a controlled ablation.

- **Task transition and history.** Stateless fields never commanded closure in the finite batch; the history-enabled field did. This is consistent with history/feedback helping the model track manipulation progress. Different horizons and stochastic outputs prevent attributing the change to history alone.
- **The field plans end-effector translation, not a complete contact state.** It has no explicit finger–rim contact geometry, grasp force, joint configuration, or object dynamics. Orientation and gripper intent are auxiliary model outputs. A decreasing scalar potential cannot itself verify that the bowl is securely held.
- **The available translation actions differ.** The original field planner selects fixed 1 cm axis primitives. Direct outputs can jointly vary all translation and rotation components, including magnitude, across a five-action sequence. This offers additional freedom for alignment and contact motion; these runs do not quantify how much of the hybrid result came from that freedom.
- **Commanded motion differs from realized motion.** The original batch averaged roughly 2.13 mm realized travel for a nominal 10 mm target. Replanning corrects the starting state, but the planner's geometric rollout still approximates the physical response. In contact, friction, controller compliance, torque limits, or pose error can matter; the available policy state does not identify the active physical cause.
- **A grasp needs evidence over time.** The field-history run closed and attempted lifts but lost contact. Robot-pose/finger feedback helps detect motion, while secure object attachment remains partly visual. A stage label such as `test_lift_black_bowl_grasp` is an intention, not proof of a grasp.

### 6.2 Why direct control stalled after lifting

The direct policy did output further commands; they were repeatedly stationary-arm/open-gripper commands. The trace and audit rule out dropped model outputs as the explanation. Retrospective measurements show that a visually plausible placement missed the precise benchmark distance threshold. Without a task-success signal, the policy had to infer whether correction was needed from images and robot state. Premature completion belief is a hypothesis, not an observed internal state.

Adding an explicit success/reward signal, a stagnation-triggered re-prompt, or a programmed recovery policy would create a new experimental condition. None should be described as already used in these results.

### 6.3 What hybrid success establishes

The run demonstrates that the implemented controller can switch field → direct → field → direct, preserve history and grip intent, perform visually observed grasp retries, and reach the official placement predicate. Coarse transport reused the original field planner; contact actions came from Astra itself.

It does not isolate the causal benefit of a scalar field, establish superiority over direct control, or verify that `near_target` corresponds to physical contact. Routing is Astra's visual judgment, without an evaluated metric contact threshold. The hybrid prompt, action freedom, stopping opportunity, and transport differ from the earlier field run. The API-key field-history run transmitted an 8,192-output-token limit, whereas the subscription transport omitted that limit; the field-history run stopped after 70 calls and the hybrid continued to 90 calls. It also stopped before release, while the direct episode attempted release but missed the positional tolerance.

The initial analytic fixtures and three static live fields do not establish a dense-field advantage: full-field, affine, and one-step planning agreed on the recorded successful fixture outcomes and first-move pilot comparisons. Distance fidelity remains a separate unmeasured research question. [Offline results](runs/offline_demo/summary.json), [static live pilot](runs/astra_live_pilot_002/RESULTS.md)

## 7. Video and audit index

| Condition | Video / viewer | What it shows | Independent evidence |
|---|---|---|---|
| Stateless field, ten episodes | [Gallery](runs/astra_bowl10_001/offline_gallery.html) | Ten real 220-action rollouts, each 76.2 s | [Batch audit](runs/astra_bowl10_001/audit.json) |
| Direct + history, complete chain | [MP4](runs/astra_direct_history_continuation_001_full_episode/full_episode.mp4) / [viewer](runs/astra_direct_history_continuation_001_full_episode/index.html) | 625 actions; 212.2 s, including a labeled resume card | [17,801 checks; zero issues](runs/astra_direct_history_continuation_001_audit/AUDIT.md) |
| Plan-only + history | [MP4](runs/astra_field_history_001/episode.mp4) / [viewer](runs/astra_field_history_001/index.html) | 350 actions; 119.1 s | [29,284 checks; zero issues](runs/astra_field_history_001_audit/AUDIT.md) |
| Subscription first-close diagnostic | [MP4](runs/chatgpt_subscription_close_001/episode.mp4) / [viewer](runs/chatgpt_subscription_close_001/index.html) | 241 actions; 83.65 s; stops after one closing cycle | [6,134 artifact checks](runs/chatgpt_subscription_close_001_audit/audit.json) and [2,447 policy checks](runs/chatgpt_subscription_close_001_audit/policy_audit.json), zero issues |
| Hybrid + history | [MP4](runs/chatgpt_hybrid_001/episode.mp4) / [viewer](runs/chatgpt_hybrid_001/index.html) | 448 actions; 177.95 s | [18,605 checks; zero issues](runs/chatgpt_hybrid_001_audit/AUDIT.md) |

Hybrid jump points: direct grasp/retries **88.35 s**; first closing **92.55 s**; lift check **141.95 s**; field transport **149.55 s**; direct placement **161.65 s**; benchmark success **174.85 s**. These are video times, not inference wall times. [Saved markers](runs/chatgpt_hybrid_001/video_markers.json)

Audits validate recorded data consistency, request histories, executed-action correspondence, original-planner reproduction, provenance, usage, and video integrity. They do not provide additional independent task trials or certify general robot competence. The hybrid audit recomputed all 255 coarse plans and checked all 193 direct actions; prior source files remained unchanged. Evaluator-only object replay diagnostics are distinguished from policy inputs throughout this report.

The following artifacts are useful plumbing/access evidence but must stay out of task success-rate denominators:

- [Offline analytic demo](runs/offline_demo/summary.json): numerical-oracle fields, no Astra inference.
- [Static subscription field probe](runs/chatgpt_plan_field_probe_004/summary.json): one real field and one local plan, zero executed actions.
- [Direct smoke](runs/direct_history_offline_smoke/summary.json), [field-history smoke](runs/field_history_smoke_001/summary.json), and [hybrid smoke](runs/chatgpt_hybrid_smoke_001/summary.json): real simulator execution with scripted model responses, zero live model calls.
- Earlier authentication/transport failures: access or response-handling outcomes, not task-performance failures.

## 8. Next evaluation protocol to resolve the open questions

These are proposed experiments, not completed work:

1. **Run a matched three-method comparison.** Use the same official initial-state list, repeated trials, model settings, transport, camera/state inputs, image retention, and feedback protocol. Report every requested rollout and its stopping cause. With no action horizon, resource-stopped runs must remain visible as incomplete; also report success by common action/model-call budgets where every method has adequate observed exposure.
2. **Separate benchmark success from released placement.** Preserve the existing predicate result, then add a separately specified release-and-stability metric under a new protocol. Define the dwell duration and object-motion tolerance before evaluation; do not retrospectively reinterpret the current hybrid episode as having passed it.
3. **Isolate history from representation.** Compare stateless and history-enabled field policies under matched limits. Compare direct and hybrid under the same history and execution-feedback settings. If an explicit success/reward signal is introduced, apply and label it as a separate condition.
4. **Measure coarse-field usefulness.** Compare full-field, affine, and one-step planners on identical recorded fields and initial conditions, then evaluate closed-loop behavior. Metric distance fidelity should use independent reference data; success alone cannot validate a distance metric.
5. **Measure grasp transitions and efficiency.** Log approach, closure, sustained object lift, transport, release, and placement milestones using evaluator-only diagnostics. Report actions, model calls, tokens, inference latency, switches, and budget truncation. Test smaller/variable field motion primitives or contact routing changes as explicit ablations.

The present evidence supports pursuing the hybrid design and improving grasp/placement verification, while preserving a clear distinction between observed behavior, proposed causes, and measured comparative performance.
